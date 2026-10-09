from dataclasses import replace
from collections import Counter

import pytest

from pipec2.tensor import expr, ir, routes
from pipec2.tensor.construct import construct
from test_tensor_machine import _binding, _instance, _profile, _rebuild
from test_tensor_synth import _instance as _loaded_instance, _pointer_fixture


def _matrix_instance(pattern):
    from pipec2.tensor import fit

    source = _instance(threads=128, width=128, ready=True, profile=_profile(warp_size=32))
    rows = {}
    slots = {}
    for t in range(128):
        for s in range(128):
            if pattern == "wgmma":
                r = 16 * (t >> 5) + ((t >> 2) & 7) + 8 * ((s >> 1) & 1) + 64 * (s >> 6)
                c = 2 * (t & 3) + (s & 1) + 8 * ((s >> 2) & 15)
            else:
                r = 64 * ((t >> 5) & 1) + ((t >> 2) & 7) + 8 * ((s >> 1) & 1) + 16 * (s >> 5)
                c = 64 * (t >> 6) + 2 * (t & 3) + (s & 1) + 8 * ((s >> 2) & 7)
            rows[r, c], slots[r, c] = t, s

    def coordinate(node):
        if node.op == "coord" and node.args[0] in ("participant", "x"):
            return replace(node, args=("row",))
        return replace(node, args=tuple(coordinate(arg) if isinstance(arg, expr.Expr) else arg for arg in node.args))

    placement = replace(source.export.placements[0], components={
        "participant": coordinate(fit.fit_expression(rows, [("participant", 7), ("slot", 7)])),
        "position": coordinate(fit.fit_expression(slots, [("participant", 7), ("slot", 7)])),
    })
    identity = replace(source.export.identity, coords=tuple(replace(domain, name="row") if domain.name == "x" else domain
                                                           for domain in source.export.identity.coords))
    demand = source.imp.demands[0]
    demand = replace(demand, domain=identity.coords, participant=coordinate(demand.participant),
                     identity_coords={"row": coordinate(demand.identity_coords["x"]), "slot": demand.identity_coords["slot"]})
    return _rebuild(source, export=replace(source.export, identity=identity, placements=(placement,)),
                    imp=replace(source.imp, identity=identity, demands=(demand,)))

def _candidates(instance):
    available = frozenset().union(*(routes.facts(instance.export, instance.imp, site) for site in instance.sites))
    return routes.candidates(routes.source_spaces(instance.export), routes.sink_space(instance.imp), available)


def _route(instance, label):
    permitted, _rejected = _candidates(instance)
    return next(route for route in permitted if route.tier == 0 and route.label == label)


def _delivered(result):
    return result.provenance["candidates"][-1]["check"]["provenance"]["terminal_deliveries"]


def test_constructed_register_round_delivers_whole_fragment():
    instance = _instance(threads=2, width=7, ready=True)
    result = construct(instance, _route(instance, "direct"))
    assert result.status == "MODEL_SAT"
    assert _delivered(result)[0]["outputs"] == [list(row) for row in instance.demands]
    issues = [node for node in result.program.graphs["hook"].nodes if isinstance(node, ir.Issue)]
    assert [(node.primitive, node.operands["width"]) for node in issues] == [("reg_copy_round", 7)]


def test_permuted_lanes_require_communication():
    instance = _instance(threads=2, width=2, transpose=True, ready=True)
    assert construct(instance, _route(instance, "direct")).status == "UNKNOWN"
    result = construct(instance, _route(instance, "warp"))
    assert result.status == "MODEL_SAT"
    assert _delivered(result)[0]["outputs"] == [list(row) for row in instance.demands]
    assert any(isinstance(node, ir.Issue) and node.primitive == "shfl_xor"
               for node in result.program.graphs["hook"].nodes)


def test_shuffle_index_is_evaluated_at_the_source_lane():
    source = _instance(threads=2, width=2, transpose=True, ready=True)
    demand = source.imp.demands[0]
    domains = {domain.name: domain for domain in demand.domain}
    demand = replace(demand, identity_coords={**demand.identity_coords,
                     "slot": expr.parse(["modc", ["add", ["coord", "slot"], ["coord", "x"]], 2], domains)})
    instance = _rebuild(source, imp=replace(source.imp, demands=(demand,)))
    result = construct(instance, _route(instance, "warp"))
    assert result.status == "MODEL_SAT"
    assert _delivered(result)[0]["outputs"] == [list(row) for row in instance.demands]


def test_staging_delivers_with_declared_scratch():
    instance = _instance(threads=2, width=3, transpose=True, ready=True)
    result = construct(instance, _route(instance, "staged"))
    assert result.status == "MODEL_SAT"
    assert _delivered(result)[0]["outputs"] == [list(row) for row in instance.demands]
    lends = tuple(replace(resource, extent=resource.extent - 4) if resource.name == "scratch" else resource
                  for resource in instance.imp.lends)
    small = _rebuild(instance, imp=replace(instance.imp, lends=lends))
    failed = construct(small, _route(small, "staged"))
    assert failed.status == "UNKNOWN"
    assert failed.provenance["reason"] == "scratch_too_small"


@pytest.mark.parametrize("pattern", ["wgmma", "tl-reg"])
def test_matrix_staging_is_conflict_free_without_changing_extent(pattern):
    instance = _matrix_instance(pattern)
    result = construct(instance, _route(instance, "staged"))
    assert result.status == "MODEL_SAT"
    assert _delivered(result)[0]["outputs"] == [list(row) for row in instance.demands]
    assert next(r.extent for r in instance.resources if r.name == "scratch") == 65536
    addresses = {}
    for node in result.program.graphs["hook"].nodes:
        if not isinstance(node, ir.Issue) or node.primitive not in ("smem_store_round", "smem_load_round"):
            continue
        visited = set()
        address_expr = expr.parse(node.operands["address"], {
            "participant": expr.Domain("participant", "int", 0, 128),
            "slot": expr.Domain("slot", "int", 0, node.operands["width"]),
        })
        for slot in range(node.operands["width"]):
            for warp in range(4):
                words = set()
                for lane in range(32):
                    address = expr.evaluate(address_expr, {"participant": warp * 32 + lane, "slot": slot})
                    assert address % 4 == 0 and 0 <= address < 65536
                    visited.add(address)
                    words.add(address // 4)
                assert max(Counter(word % 32 for word in words).values()) == 1
        addresses[node.primitive] = visited
    assert addresses["smem_store_round"] == addresses["smem_load_round"] == set(range(0, 65536, 4))


@pytest.mark.parametrize("scalar_bytes", [2, 8])
def test_staging_permutation_preserves_scalar_alignment_and_broadcasts(scalar_bytes):
    from pipec2.tensor.staging import select

    width = 4096 // (32 * scalar_bytes)
    stores = {(p, s): p * width + s for p in range(32) for s in range(width)}
    # A replicated read is one word per bank, not 32 conflicting lanes.
    loads = {(p, s): s for p in range(32) for s in range(width)}
    tables, _, _ = select([stores, loads], scalar_bytes, 4096, ())
    assert set(tables[0].values()) == set(range(0, 4096, scalar_bytes))
    for point, identity in loads.items():
        assert tables[1][point] == tables[0][0, identity]
    degrees = []
    for table in tables:
        for slot in range(width):
            words = {word for p in range(32)
                     for word in range(table[p, slot] // 4, (table[p, slot] + scalar_bytes - 1) // 4 + 1)}
            degrees.append(max(Counter(word % 32 for word in words).values()))
    assert max(degrees) == max(1, scalar_bytes // 4)


def test_chunk_binding_selects_receipts_in_one_program():
    instance = _instance(threads=2, width=4, tmem=True, ready=True)
    chunk = expr.Domain("chunk", "int", 0, 2)
    demand = instance.imp.demands[0]
    domains = {domain.name: domain for domain in demand.domain}
    domains["slot"] = expr.Domain("slot", "int", 0, 2)
    domains["chunk"] = chunk
    demand = replace(demand, domain=tuple(domain for name, domain in domains.items() if name != "chunk"),
                     identity_coords={**demand.identity_coords,
                                      "slot": expr.parse(["add", ["coord", "slot"], ["mulc", 2, ["bind", "chunk"]]], domains)})
    instance = _rebuild(instance, imp=replace(instance.imp, demands=(demand,)), bindings=(_binding(chunk),),
                        profile={**instance.profile, "packet_columns": 2})
    result = construct(instance, _route(instance, "direct"))
    assert result.status == "MODEL_SAT"
    assert {tuple(item["bindings"].items()) for item in _delivered(result)} == {(("chunk", 0),), (("chunk", 1),)}
    for world, delivery in enumerate(_delivered(result)):
        assert delivery["outputs"] == [list(row) for row in instance.world_demands[world]]


def test_padded_shared_source_fits_and_delivers():
    raw = _pointer_fixture(space="smem", threads=2, columns=3)
    result = construct(_loaded_instance(raw), _route(_loaded_instance(raw), "direct"))
    assert result.status == "MODEL_SAT"
    assert len(_delivered(result)[0]["outputs"]) == 6


def test_eight_warps_receive_their_modulo_tmem_subpartition():
    source = _instance(threads=128, width=16, tmem=True, ready=True)
    t = expr.Domain("t", "int", 0, 256)
    slot = expr.Domain("slot", "int", 0, 16)
    domains = {"t": t, "slot": slot}
    demand = replace(source.imp.demands[0], domain=(t, slot),
                     participant=expr.parse(["coord", "t"], domains),
                     identity_coords={"x": expr.parse(["modc", ["coord", "t"], 128], domains),
                                      "slot": expr.parse(["coord", "slot"], domains)})
    sites = (replace(source.sites[0], participants=replace(source.sites[0].participants, set=tuple(range(256)))),)
    instance = _rebuild(source, imp=replace(source.imp, demands=(demand,)), sites=sites,
                        profile=_profile(warp_size=32, packet_columns=16))
    result = construct(instance, _route(instance, "direct"))
    assert result.status == "MODEL_SAT"
    assert _delivered(result)[0]["outputs"] == [list(row) for row in instance.demands]


def test_static_owner_guard_stages_for_consumers_without_private_sources():
    source = _instance(threads=2, width=2, ready=True)
    t = expr.Domain("t", "int", 0, 4)
    domains = {"t": t}
    demand = replace(source.imp.demands[0], domain=(t,),
                     participant=expr.parse(["coord", "t"], domains),
                     position=expr.parse(["const", 0], domains),
                     identity_coords={"x": expr.parse(["modc", ["coord", "t"], 2], domains),
                                      "slot": expr.parse(["divc", ["coord", "t"], 2], domains)})
    sites = (replace(source.sites[0], participants=replace(source.sites[0].participants, set=tuple(range(4)))),)
    lends = tuple(replace(r, extent=1) if r.domain == "metadata" else r for r in source.imp.lends)
    instance = _rebuild(source, imp=replace(source.imp, demands=(demand,), lends=lends), sites=sites)
    result = construct(instance, _route(instance, "staged"))
    assert result.status == "MODEL_SAT"
    assert _delivered(result)[0]["outputs"] == [list(row) for row in instance.demands]


def test_admission_caps_only_expression_free_variables(monkeypatch):
    from pipec2.profile import get_profile
    from pipec2.tensor import plan
    from test_tensor_synth import _fixture, _loaded

    raw = _fixture(already_ready=True, width=4)
    raw["bindings"] = [{"name": "unused", "sort": "int", "domain": {"name": "unused", "sort": "int", "lo": 0, "hi": 2},
                        "owner": "load", "units": "scalar", "observation_scope": "cta", "lifetime": "tile"}]
    tensor, derived = _loaded(raw)
    monkeypatch.setattr(plan, "EXPRESSION_CAP", 4)
    admitted, instance = plan.admit(tensor.exports[0], tensor.imports[0], derived, get_profile("sm_103a"))
    assert admitted.status == "OK"
    assert len(instance.worlds) == 2
    assert all("unused" not in row["free_variables"] for row in admitted.report["analyzers"])


def test_declared_convergence_is_required_for_the_shuffle_route():
    source = _instance(threads=2, width=2, transpose=True, ready=True)
    site = replace(source.sites[0], guarantees=tuple(guarantee for guarantee in source.sites[0].guarantees if guarantee != "converged"))
    instance = _rebuild(source, sites=(site,))
    permitted, rejected = _candidates(instance)
    assert not any(route.shfl for route in permitted)
    missing = {route.name: facts for route, facts in rejected if route.shfl}
    assert missing["warp_constructed"] == {"g:converged"}
    assert construct(instance, _route(source, "warp")).provenance["reason"] == "family_unavailable"


def test_backing_smem_is_available_to_load_but_never_lent_for_staging():
    source = _instance(threads=2, width=2, transpose=True, ready=True)
    backing = next(resource for resource in source.imp.lends if resource.name == "scratch")
    imp = replace(source.imp, lends=tuple(resource for resource in source.imp.lends if resource != backing))
    instance = _rebuild(source, export=replace(source.export, backing=(backing,)), imp=imp)
    available = routes.facts(instance.export, instance.imp, instance.sites[0])
    assert "avail:smem" in available and "lend:smem" not in available
    permitted, rejected = _candidates(instance)
    assert not any(route.relay for route in permitted)
    assert {facts for route, facts in rejected if route.relay} == {frozenset({"lend:smem"})}


def test_export_with_placements_in_several_spaces_is_routed_from_each():
    raw = _pointer_fixture(space="smem", threads=2, columns=3)
    # An inert guarded register placement in front of the shared one must not hide the load route.
    raw["exports"][0]["placements"].insert(0, {
        "guard": ["eq", ["coord", "r"], ["const", 99]], "space": "reg", "units": "scalar",
        "components": {"participant": ["coord", "r"], "position": ["coord", "c"]}})
    raw["exports"][0]["read_capabilities"] = ["reg_copy", "smem_load_scalar"]
    instance = _loaded_instance(raw)
    assert routes.source_spaces(instance.export) == ("smem", "reg")
    permitted, _rejected = _candidates(instance)
    assert {route.source for route in permitted} == {"smem", "reg", "smem+reg"}
    direct = next(route for route in permitted if route.source == "smem" and route.tier == 0 and route.label == "direct")
    assert direct.bring == ("load",)
    joint = next(route for route in permitted if route.source == "smem+reg" and route.tier == 0 and route.label == "direct")
    assert joint.bring == ("load", "copy")
    result = construct(instance, direct)
    assert result.status == "MODEL_SAT"
    assert len(_delivered(result)[0]["outputs"]) == 6


def test_inert_register_placement_beside_a_shared_view_still_aliases():
    from test_tensor_view import _view_fixture
    raw = _view_fixture()
    raw["exports"][0]["placements"].append({
        "guard": ["eq", ["coord", "row"], ["const", 99]], "space": "reg", "units": "scalar",
        "components": {"participant": ["coord", "row"], "position": ["coord", "slot"]}})
    raw["exports"][0]["read_capabilities"] = ["smem_load_scalar", "reg_copy"]
    raw["imports"][0]["lends"] = [lend for lend in raw["imports"][0]["lends"] if lend["domain"] == "reg"]
    raw["imports"][0]["rendezvous"] = []
    instance = _loaded_instance(raw)
    permitted, _rejected = _candidates(instance)
    assert [route.name for route in permitted] == ["view_alias"]
    assert construct(instance, permitted[0]).status == "MODEL_SAT"
