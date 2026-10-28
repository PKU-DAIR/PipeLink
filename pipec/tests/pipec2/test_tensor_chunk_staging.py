from dataclasses import replace

import pytest

from pipec2.model import TensorConnection
from pipec2.profile import get_profile
from pipec2.tensor import check, check_static, expr, ir, lower, plan
from pipec2.tensor.construct import construct
from pipec2.tensor.contracts import (
    CallableParam, Participants, Rendezvous, ResourceLocation, TensorCallable,
)
from test_link import _solution
from test_tensor_construct import _route
from test_tensor_link import _composition
from test_tensor_machine import _binding, _instance, _rebuild


def _chunk_instance(*, tmem=False, column_split=False, scratch_bytes=16384):
    profile = get_profile("sm_103a" if tmem else "sm_120a" if column_split else "sm_90a")
    source = _instance(threads=128, width=256, tmem=tmem, ready=True, profile=profile)
    m, n = expr.Domain("m", "int", 0, 128), expr.Domain("n", "int", 0, 256)
    chunk = expr.Domain("chunk", "bits[32]", 0, 8)
    t, p = expr.Domain("t", "int", 0, 256), expr.Domain("p", "int", 0, 16)
    env = {d.name: d for d in (m, n, chunk, t, p)}
    parse = lambda value: expr.parse(value, env)
    identity = replace(source.export.identity, coords=(m, n))
    if tmem:
        components = {"dp": parse(["coord", "m"]), "column": parse(["coord", "n"])}
    else:
        # Two 64-row warpgroups, each with a WGMMA-like C register map.
        components = {
            "participant": parse(["add", ["const", 128], ["add",
                ["mulc", 128, ["divc", ["coord", "m"], 64]], ["add",
                ["mulc", 32, ["divc", ["modc", ["coord", "m"], 64], 16]], ["add",
                ["mulc", 4, ["modc", ["coord", "m"], 8]],
                ["divc", ["modc", ["coord", "n"], 8], 2]]]]]),
            "position": parse(["add", ["mulc", 2, ["modc", ["divc", ["coord", "m"], 8], 2]],
                ["add", ["modc", ["coord", "n"], 2], ["mulc", 4, ["divc", ["coord", "n"], 8]]]]),
        }
    if column_split:
        # Four 64-thread column groups; each chunk changes the active source
        # cohort and selects one 64-position half of its complete frame.
        components = {
            "participant": parse(["add", ["mulc", 64, ["divc", ["coord", "n"], 64]],
                                  ["divc", ["coord", "m"], 2]]),
            "position": parse(["add", ["mulc", 64, ["modc", ["divc", ["coord", "n"], 32], 2]],
                ["add", ["mulc", 32, ["modc", ["coord", "m"], 2]], ["modc", ["coord", "n"], 32]]]),
        }
    export = replace(source.export, identity=identity,
                     placements=(replace(source.export.placements[0], components=components),))
    first, threads = (0, 256) if tmem or column_split else (128, 384)
    demand = replace(source.imp.demands[0], domain=(t, p),
        participant=parse(["add", ["const", first], ["coord", "t"]]),
        position=parse(["coord", "p"]), identity_coords={
            "m": parse(["modc" if tmem else "divc", ["coord", "t"], 128 if tmem else 2]),
            "n": parse(["add", ["mulc", 32, ["toint", ["bind", "chunk"]]], ["add",
                ["mulc", 16, ["divc" if tmem else "modc", ["coord", "t"], 128 if tmem else 2]],
                ["coord", "p"]]]),
        })
    site = replace(source.sites[0], participants=Participants(set=tuple(range(first, threads))))
    lends = []
    for resource in source.imp.lends:
        if resource.name == "dest":
            resource = replace(resource, extent=16)
        elif resource.name == "temp":
            resource = replace(resource, name="frame", extent=128, binding="caller_frame")
        elif resource.name == "scratch":
            resource = replace(resource, extent=scratch_bytes)
        elif resource.name == "barrier":
            lends.append(replace(resource, name="outer", location=ResourceLocation(None, 0)))
        lends.append(resource)
    params = (CallableParam("source", "source_base", "pipelink_tensor_source"),
              CallableParam("chunk", "binding", "uint32_t"),
              CallableParam("dest", "dest_fragment", "float*"),
              CallableParam("scratch", "scratch", "void*"))
    callable = TensorCallable("access", 'extern "C" __device__ void access(pipelink_tensor_source, uint32_t, float*, void*);',
                              params, {"ready": "access_ready", "done": "access_done"})
    imp = replace(source.imp, identity=identity, demands=(demand,), lends=tuple(lends), callable=callable,
                  rendezvous=tuple(Rendezvous(event, "outer", Participants(set=tuple(range(threads))))
                                   for event in ("ready", "done")))
    return _rebuild(source, export=export, imp=imp, sites=(site,), bindings=(_binding(chunk),))


def _planner(instance):
    pipeline, stages, _, _ = _composition()
    producer, consumer = stages
    producer = replace(producer, tensor=replace(producer.tensor, exports=(instance.export,), imports=(), bindings=()))
    consumer = replace(consumer, tensor=replace(consumer.tensor, exports=(), imports=(instance.imp,), sites=instance.sites,
                       bindings=tuple(_binding(d) for d in instance.derived.binding_domains.values())))
    connection = TensorConnection("chunk", f"{producer.id}:{instance.export.name}", f"{consumer.id}:{instance.imp.name}")
    pipeline = replace(pipeline, tensor_connections=(connection,))
    return plan.ConnectionPlanner(pipeline, (producer, consumer), connection, instance.profile)


@pytest.fixture(scope="module")
def staged_plan():
    instance = _chunk_instance()
    planned = _planner(instance).next()
    return instance, planned


def test_chunk_staging_preserves_identity_frame_and_bounds_scratch(staged_plan):
    instance, planned = staged_plan
    assert planned.status == "MODEL_SAT"
    assert planned.provenance["constructed_class"] == "staged_constructed"
    assert planned.provenance["admission_oracle"]["ok"]
    assert planned.provenance["admission_oracle"]["checked_positions"] == 128 * 256
    assert len(instance.identities) == 128 * 256
    assert instance.exported_storage()["@export:reg"]["extent"] == 128
    assert next(r.extent for r in instance.imp.lends if r.binding == "caller_frame") == 128
    assert 0 < planned.provenance["resource_demands"]["scratch"] <= 16384
    requests = {r["resource"]: r for r in planned.barrier_requests}
    assert requests["outer"]["participants"] == list(range(384))
    assert requests["barrier"]["participants"] == list(range(128, 384))
    assert requests["outer"]["reserved"] == 0
    prepared = lower._prepare(planned, instance.imp, _solution(threads=384), instance.profile)
    barriers = [step for step in prepared["graphs"]["hook"] if step["primitive"] == "cta_rendezvous"]
    assert [prepared["resources"][step["operands"]["resource"]]["base"] for step in barriers] == [1]
    assert prepared["context"]["participants"]["hook"] == list(range(128, 384))


def test_chunk_staging_stores_only_demanded_frame_values(staged_plan):
    instance, planned = staged_plan
    assert planned.status == "MODEL_SAT"
    nodes = planned.program.graphs["hook"].nodes
    stores = [node for node in nodes if isinstance(node, ir.Issue) and node.primitive == "smem_store_scalar"]
    assert len(stores) == 16
    domains = {**instance.derived.binding_domains, "participant": expr.Domain("participant", "int", 128, 384)}
    for world, demands in enumerate(instance.world_demands):
        expected = {identity for _, _, identity in demands}
        actual = set()
        frame = {(cell.participant, cell.index): identity
                 for cell, identity in zip(instance.cells, instance.initial_cells[world])
                 if cell.resource == "@export:reg"}
        for node in stores:
            source = expr.parse(node.operands["src"]["index"], domains)
            for participant in range(128, 384):
                position = expr.evaluate(source, {"participant": participant, "chunk": world})
                actual.add(frame[participant, position])
        assert actual == expected


def test_chunk_staging_rejects_short_scratch():
    instance = _chunk_instance(scratch_bytes=16380)
    result = construct(instance, _route(instance, "staged"))
    assert result.status == "UNKNOWN"
    assert result.provenance["reason"] == "scratch_too_small"


def test_chunk_demand_does_not_shrink_required_caller_frame(staged_plan):
    instance, _ = staged_plan
    lends = tuple(replace(resource, extent=16) if resource.binding == "caller_frame" else resource
                  for resource in instance.imp.lends)
    instance = _rebuild(instance, imp=replace(instance.imp, lends=lends))
    rejected = _planner(instance).next()
    assert rejected.status == "CONTRACT_INVALID"
    assert any(failure.clause_id == "T-PLAN-FRAME" and failure.path == "tensor.imports.consumer.lends"
               for failure in rejected.admission.issues)


@pytest.mark.parametrize("barrier_kind", ["missing", "wrong_cohort", "zero"])
def test_subset_staging_rejects_ineligible_inner_barrier(barrier_kind):
    instance = _chunk_instance()
    lends = tuple(resource for resource in instance.imp.lends if barrier_kind != "missing" or resource.name != "barrier")
    if barrier_kind == "zero":
        lends = tuple(replace(resource, location=ResourceLocation(None, 0)) if resource.name == "barrier" else resource
                      for resource in lends)
    rendezvous = instance.imp.rendezvous
    if barrier_kind == "wrong_cohort":
        rendezvous = tuple(replace(event, barrier="barrier") for event in rendezvous)
    instance = _rebuild(instance, imp=replace(instance.imp, lends=lends, rendezvous=rendezvous))
    result = construct(instance, _route(instance, "staged"))
    assert result.status == "UNKNOWN"
    assert result.provenance["reason"] == "rendezvous_unavailable"


def test_eight_warp_chunk_demand_has_direct_tmem_receipt():
    instance = _chunk_instance(tmem=True)
    planned = _planner(instance).next()
    assert planned.status == "MODEL_SAT"
    assert planned.provenance["constructed_class"] == "direct_constructed"
    assert planned.provenance["admission_oracle"]["ok"]
    assert planned.provenance["resource_demands"]["scratch"] == 0
    prepared = lower._prepare(planned, instance.imp, _solution(threads=256), instance.profile)
    assert [step["primitive"] for step in prepared["graphs"]["hook"]] == ["tmem_packet_receipt"]


def test_chunk_staging_guards_binding_dependent_source_cohorts():
    instance = _chunk_instance(column_split=True)
    planned = _planner(instance).next()
    assert planned.status == "MODEL_SAT"
    assert planned.provenance["constructed_class"] == "staged_constructed"
    assert planned.provenance["admission_oracle"]["ok"]
    assert planned.provenance["admission_oracle"]["checked_positions"] == 32768
    assert 0 < planned.provenance["resource_demands"]["scratch"] <= 16384
    prepared = lower._prepare(planned, instance.imp, _solution(threads=256), instance.profile)
    stores = [step for step in prepared["graphs"]["hook"] if step["primitive"] == "smem_store_scalar"]
    assert len(stores) == 64
    domains = {**instance.derived.binding_domains, "participant": expr.Domain("participant", "int", 0, 256)}
    for chunk in range(8):
        for step in stores:
            guard = expr.parse(step["guard"], domains)
            active = {t for t in range(256) if expr.evaluate(guard, {"participant": t, "chunk": chunk})}
            assert active == set(range(64 * (chunk // 2), 64 * (chunk // 2 + 1)))
        barriers = [step for step in prepared["graphs"]["hook"] if step["primitive"] == "cta_rendezvous"]
        assert all("guard" not in step for step in barriers)


def test_checker_replays_binding_dependent_guards_in_every_world():
    source = _instance(threads=4, width=2, ready=True)
    chunk = expr.Domain("chunk", "int", 0, 2)
    t = expr.Domain("t", "int", 0, 4)
    domains = {"chunk": chunk, "t": t}
    parse = lambda value: expr.parse(value, domains)
    demand = replace(source.imp.demands[0], domain=(t,),
        participant=parse(["coord", "t"]), position=parse(["const", 0]),
        identity_coords={
            "x": parse(["add", ["mulc", 2, ["bind", "chunk"]], ["divc", ["coord", "t"], 2]]),
            "slot": parse(["modc", ["coord", "t"], 2]),
        })
    instance = _rebuild(source, imp=replace(source.imp, demands=(demand,)), bindings=(_binding(chunk),))
    result = construct(instance, _route(instance, "staged"))
    assert result.status == "MODEL_SAT"
    certificate, deliveries = check_static.certify(instance, result.program)
    assert certificate["verified"]
    assert {tuple(row["bindings"].items()) for row in deliveries} == {(("chunk", 0),), (("chunk", 1),)}
    # A constructor bug that freezes the active cohort at world zero must not
    # obtain a certificate, even though its first-world delivery is correct.
    nodes = list(result.program.graphs["hook"].nodes)
    index = next(i for i, node in enumerate(nodes) if isinstance(node, ir.Meta))
    nodes[index] = replace(nodes[index], expr=["lt", ["coord", "participant"], ["const", 2]])
    broken = replace(result.program, graphs={"hook": ir.ControlGraph(tuple(nodes))})
    certificate, deliveries = check_static.certify(instance, broken)
    assert not certificate["verified"] and deliveries is None
    assert check.check(instance, broken, budget=5000).status == "UNSAFE"


def test_static_scalar_store_certificate_rejects_aliasing():
    source = _instance(threads=2, width=1, transpose=True, ready=True)
    nodes = (
        ir.Issue("smem_store_scalar", {"src": {"resource": "@export:reg", "index": 0},
                                      "dst": {"resource": "scratch", "index": 0}}, 1),
        ir.Issue("cta_rendezvous", {"resource": "barrier"}, 2),
        ir.Issue("smem_load_scalar", {"src": {"resource": "scratch", "index": 0},
                                     "dst": {"resource": "dest", "index": 0}}, 3),
        ir.Return(),
    )
    program = source.program({"hook": ir.ControlGraph(nodes)})
    certificate, deliveries = check_static.certify(source, program)
    assert not certificate["verified"] and deliveries is None
    assert check.check(source, program, budget=1000).status == "UNSAFE"


def test_direct_register_copy_guards_binding_dependent_demand_cohorts():
    instance = _chunk_instance(column_split=True)
    m = expr.Domain("m", "int", 0, 128)
    n = expr.Domain("n", "int", 0, 32)
    domains = {**instance.derived.binding_domains, "m": m, "n": n}
    column = expr.parse(["add", ["coord", "n"], ["mulc", 32, ["toint", ["bind", "chunk"]]]], domains)

    def translated(node):
        if node.op == "coord" and node.args[0] == "n":
            return column
        return replace(node, args=tuple(translated(arg) if isinstance(arg, expr.Expr) else arg for arg in node.args))

    placement = instance.export.placements[0].components
    demand = replace(instance.imp.demands[0], domain=(m, n),
        identity_coords={"m": expr.parse(["coord", "m"], domains), "n": column},
        participant=translated(placement["participant"]),
        position=expr.Expr("modc", (translated(placement["position"]), 64), "int"))
    lends = tuple(replace(resource, extent=64) if resource.name == "dest" else resource
                  for resource in instance.imp.lends)
    instance = _rebuild(instance, imp=replace(instance.imp, demands=(demand,), lends=lends))
    planned = _planner(instance).next()
    assert planned.status == "MODEL_SAT"
    assert planned.provenance["constructed_class"] == "direct_constructed"
    assert planned.provenance["admission_oracle"]["ok"]
    assert planned.provenance["admission_oracle"]["checked_positions"] == 32768
    assert planned.provenance["resource_demands"]["scratch"] == 0
    prepared = lower._prepare(planned, instance.imp, _solution(threads=256), instance.profile)
    assert {step["primitive"] for step in prepared["graphs"]["hook"]} <= {"reg_copy", "reg_copy_round"}
    assert all("guard" in step for step in prepared["graphs"]["hook"])
