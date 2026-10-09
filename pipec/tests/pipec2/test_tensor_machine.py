from __future__ import annotations

from dataclasses import replace

import pytest

from pipec2.profile import get_profile
from pipec2.tensor import expr, ir, plan
from pipec2.tensor.check import check
from pipec2.tensor.contracts import (
    Binding, CallableParam, DemandView, Identity, Lend, Participants, PlacementView, Readiness,
    ResourceLocation, ResourceSync, Site, TensorCallable, TensorExport, TensorImport,
)
from pipec2.tensor.machine import CATALOGUE, EMPTY, IdentityVector, Instance, ModelError, Monitor


def _profile(*, tmem=True, **options):
    """Finite test target; every option the model or derivation reads is stated here."""
    target = {"name": "finite-test", "catalogue": "tcgen05-readonly-v1" if tmem else "warp-readonly-v1",
              "fairness": ("owner_eventually_ready", "cta_eventually_scheduled", "target_eventually_completes")}
    if tmem:
        target.update(tmem_lanes=128, tmem_columns=512)
    return {**target, **options}


def _instance(*, threads=2, width=1, transpose=False, tmem=False, ready=False, early_reclaim=False, profile=None):
    x = expr.Domain("x", "int", 0, threads)
    slot = expr.Domain("slot", "int", 0, width)
    env = {"x": x, "slot": slot}
    parse = lambda value: expr.parse(value, env)
    identity = Identity("C", "tile", "complete", (x, slot), "fp32", "ieee754")
    placement = PlacementView(None, "tmem" if tmem else "reg", "dp/column" if tmem else "scalar",
        {"dp": parse(["coord", "x"]), "column": parse(["coord", "slot"])} if tmem else
        {"participant": parse(["coord", "x"]), "position": parse(["coord", "slot"])})
    export = TensorExport("source", identity, (placement,), ("tmem_packet_receipt",) if tmem else ("reg_copy",), Readiness("ready", "reclaim" if early_reclaim else "done"), {}, (), {"produce": "produce", "release": None})
    demand = DemandView("positions", (x, slot), None, {"x": parse(["coord", "x"]), "slot": parse(["coord", "slot"])},
                       parse(["sub", ["const", threads - 1], ["coord", "x"]]) if transpose else parse(["coord", "x"]), parse(["coord", "slot"]))
    call = TensorCallable("access", 'extern "C" __device__ void access(float*);', (CallableParam("dest", "dest_fragment", "float*"),), {"ready": "access_ready", "done": "access_done"})
    lends = (
        Lend("dest", "reg", width, 1, "thread", "fixed", ResourceLocation("dest", None), "consumer", {"from": "invoke", "to": "done"}, None),
        Lend("temp", "reg", width, 1, "thread", "fixed", ResourceLocation("temp", None), "consumer", {"from": "invoke", "to": "done"}, None),
        Lend("scratch", "smem", threads * width * 4, 4, "cta", "fixed", ResourceLocation(None, 0), "consumer", {"from": "invoke", "to": "done"}, None),
        Lend("barrier", "barrier", 1, 1, "cta", "fixed", ResourceLocation(None, 1), "consumer", {"from": "invoke", "to": "done"}, ResourceSync(2, 1)),
        Lend("metadata", "metadata", 2, 1, "warp", "fixed", ResourceLocation("metadata", None), "consumer", {"from": "invoke", "to": "done"}, None),
    )
    imp = TensorImport("consumer", identity, (demand,), call, ("hook",), "done", lends, ())
    site = Site("hook", "consumer", Participants(set=tuple(range(threads))), "invoke",
                ("enter_once", "converged", "full_warps", "no_early_return"), ("ready",) if ready else ())
    profile = _profile(warp_size=threads, packet_columns=width) if profile is None else profile
    return Instance.build(export, imp, plan.derive(export, imp, (site,), (), profile), profile)


def _binding(domain):
    return Binding(domain.name, domain.sort, domain, "owner", "scalar", "cta", "tile")


def _rebuild(instance, *, export=None, imp=None, sites=None, bindings=None, profile=None):
    export = instance.export if export is None else export
    imp = instance.imp if imp is None else imp
    sites = instance.sites if sites is None else sites
    bindings = tuple(_binding(domain) for domain in instance.derived.binding_domains.values()) if bindings is None else bindings
    profile = instance.profile if profile is None else profile
    return Instance.build(export, imp, plan.derive(export, imp, sites, bindings, profile), profile)


def _cell(resource, index=0):
    return {"resource": resource, "index": index}


def _copy(source, destination, next, *, source_index=0, destination_index=0):
    return ir.Issue("reg_copy", {"src": _cell(source, source_index), "dst": _cell(destination, destination_index)}, next)


def _program(instance, *nodes):
    return instance.program({"hook": ir.ControlGraph(tuple(nodes))})


def _step(instance, state, label):
    return instance.apply(state, next(transition for transition in instance.enabled(state) if transition.label == label))


def test_readiness_query_is_not_atomic_with_delivery():
    instance = _instance(threads=1)
    guarded = _program(instance, ir.Branch("ready", 1, 0), _copy("@export:reg", "dest", 2), ir.Return())
    passed = check(instance, guarded, budget=1000)
    assert passed.status == "PASS"
    assert passed.provenance["terminal_deliveries"] == [{"bindings": {}, "outputs": [[0, 0, 0]]}]
    unguarded = _program(instance, _copy("@export:reg", "dest", 1), ir.Return())
    failed = check(instance, unguarded, budget=1000)
    assert failed.status == "UNSAFE"
    assert "readiness" in failed.violations[0].message


def test_observe_ready_only_updates_local_metadata():
    instance = _instance(threads=1)
    program = _program(instance, ir.Issue("observe_ready", {"dest": "seen"}, 1), ir.Branch("seen", 2, 0),
                       _copy("@export:reg", "dest", 3), ir.Return())
    machine = instance.with_program(program)
    initial = machine.initial()
    observed = _step(machine, initial, "step:hook:0")
    assert observed.cells == initial.cells
    assert observed.owner_phase == initial.owner_phase
    assert dict(observed.metadata[0])["seen"] == 0
    assert check(instance, program, budget=1000).status == "PASS"


def test_packet_issue_and_indivisible_completion_are_distinct():
    instance = _instance(tmem=True, ready=True)
    issue = ir.Issue("tmem_packet_receipt", {"packet": 0, "dest": "dest", "offset": 0, "phase": "issue", "handle": "r"}, 1)
    wait = ir.Issue("tmem_packet_receipt", {"phase": "wait", "handle": "r"}, 2)
    program = _program(instance, issue, wait, ir.Return())
    machine = instance.with_program(program)
    initial = machine.initial()
    issued = _step(machine, initial, "collective:hook:0:0")
    assert machine.deliveries(issued) == ((0, 0, EMPTY), (1, 0, EMPTY))
    completed = _step(machine, issued, "complete:hook:0:r")
    assert machine.deliveries(completed) == instance.demands
    assert check(instance, program, budget=1000).status == "PASS"
    premature = _program(instance, issue, _copy("dest", "temp", 2), replace(wait, next=3), ir.Return())
    failed = check(instance, premature, budget=1000)
    assert failed.status == "UNSAFE"
    assert "before completion" in failed.violations[0].message


def test_deferred_read_does_not_snapshot_source_at_issue():
    instance = _instance(tmem=True, ready=True)
    program = _program(instance,
        ir.Issue("tmem_packet_receipt", {"packet": 0, "dest": "dest", "phase": "issue", "handle": "r"}, 1),
        ir.Issue("tmem_packet_receipt", {"phase": "wait", "handle": "r"}, 2), ir.Return())
    machine = instance.with_program(program)
    issued = _step(machine, machine.initial(), "collective:hook:0:0")
    # Deliberately violate the independently checked source-stability premise to
    # exercise read-at-completion.  This state is NOT an admitted owner execution.
    cells = list(issued.cells)
    cells[issued.outstanding[0].sources[0]] = 1
    overwritten = replace(issued, cells=tuple(cells))
    completed = _step(machine, overwritten, "complete:hook:0:r")
    assert "source-stability" in completed.errors[0]
    with pytest.raises(ModelError) as caught:
        _instance(tmem=True, early_reclaim=True)
    assert caught.value.status == "UNSUPPORTED_MODEL"


def test_premature_reclaim_then_last_copy_overwrite_is_unsafe():
    instance = _instance(threads=1, width=2, early_reclaim=True)
    program = _program(instance, ir.Branch("ready", 1, 0),
        _copy("@export:reg", "temp", 2),
        _copy("@export:reg", "dest", 3, source_index=1, destination_index=1),
        _copy("dest", "temp", 4, source_index=1),
        _copy("temp", "dest", 5), ir.Return())
    machine = instance.with_program(program)
    state = _step(machine, machine.initial(), "owner.ready")
    for _ in range(3):
        state = _step(machine, state, "step:hook:0")
    state = _step(machine, state, "owner.reclaim")
    # q0 now exists only in temp[0]; overwriting it destroys the last copy.
    assert state.cells.count(0) == 1
    state = _step(machine, state, "step:hook:0")
    assert 0 not in state.cells
    state = _step(machine, state, "step:hook:0")
    returned = _step(machine, state, "step:hook:0")
    assert "wrong logical identity" in returned.errors[0]
    assert check(instance, program, budget=10000).status == "UNSAFE"


def _transpose(instance, *, synchronize):
    participant = ["coord", "participant"]
    own = ["mulc", 4, participant]
    other = ["mulc", 4, ["sub", ["const", 1], participant]]
    nodes = [ir.Branch("ready", 1, 0),
             ir.Issue("smem_store_scalar", {"src": _cell("@export:reg"), "dst": _cell("scratch", own)}, 2)]
    if synchronize:
        nodes.append(ir.Issue("cta_rendezvous", {"resource": "barrier"}, 3))
    nodes.append(ir.Issue("smem_load_scalar", {"src": _cell("scratch", other), "dst": _cell("dest")}, len(nodes) + 1))
    nodes.append(ir.Return())
    return _program(instance, *nodes)


def test_two_thread_transpose_through_shared_delivers_every_position():
    instance = _instance(transpose=True)
    result = check(instance, _transpose(instance, synchronize=True), budget=10000)
    assert result.status == "PASS"
    assert result.provenance["terminal_deliveries"] == [{"bindings": {}, "outputs": [[0, 0, 1], [1, 0, 0]]}]
    assert all(reduction["verified"] for reduction in result.reductions)
    assert {reduction["kind"] for reduction in result.reductions} <= {
        "exact-state-merging", "identity-vector-interning", "checked-persistent-set", "checked-rendezvous-frontier",
    }


def test_same_epoch_write_read_conflict_is_unsafe_without_publication():
    instance = _instance(transpose=True, ready=True)
    program = _transpose(instance, synchronize=False)
    machine = instance.with_program(program)
    state = machine.initial()
    # Both disjoint stores complete; neither is published.  The next cross-thread
    # read races with the previous writer, even under this otherwise lucky order.
    for member in (0, 0, 1, 1):
        state = _step(machine, state, f"step:hook:{member}")
    failed = _step(machine, state, "step:hook:0")
    assert "same-epoch conflicting" in failed.errors[0]
    assert check(instance, program, budget=10000).status == "UNSAFE"


def test_missing_rendezvous_participant_is_a_finite_progress_failure():
    instance = _instance(ready=True)
    # Each thread first receives its own result.  Participant 1 then returns;
    # participant 0 reaches a barrier that still requires BOTH physical threads.
    program = _program(instance, _copy("@export:reg", "dest", 1),
        ir.Meta("is_one", ["coord", "participant"], 2), ir.Branch("is_one", 4, 3),
        ir.Issue("cta_rendezvous", {"resource": "barrier"}, 4), ir.Return())
    result = check(instance, program, budget=10000)
    assert result.status == "NONPROGRESS"
    assert "finite maximal" in result.violations[0].message
    machine = instance.with_program(program)
    state = machine.initial()
    for member in (0, 0, 0, 0, 1, 1, 1, 1):
        state = _step(machine, state, f"step:hook:{member}")
    assert machine.enabled(state) == ()
    assert state.arrivals[0].actors == (0,)


def test_owner_ready_justice_does_not_hide_post_ready_livelock():
    instance = _instance(threads=1)
    polling_forever = _program(instance, ir.Branch("ready", 0, 0), ir.Return())
    result = check(instance, polling_forever, budget=1000)
    assert result.status == "NONPROGRESS"
    assert "owner.ready" not in result.violations[0].detail["recurrent_labels"]
    assert "reachable fair infinite" in result.violations[0].message


def test_unfair_prefix_safety_is_not_discarded():
    instance = _instance(threads=1)
    program = _program(instance, _copy("@export:reg", "dest", 0), ir.Return())
    assert check(instance, program, budget=1000).status == "UNSAFE"


def test_inconsistent_compassion_assumption_cannot_prove_vacuous_success():
    instance = _instance(threads=1, ready=True)
    # An impossible infinitely recurring owner.ready consequence rules out runs
    # taking step:hook:0 infinitely often, but never removes finite deadlocks.
    monitor = Monitor("compassion", premise=frozenset({"step:hook:0"}), consequence=frozenset({"owner.ready"}), name="finite-test-compassion")
    machine = replace(instance, monitors=instance.monitors + (monitor,))
    program = _program(machine, ir.Branch("ready", 0, 0), ir.Return())
    assert check(machine, program, budget=1000).status == "INCOMPLETE"
    # Explicitly supplying an inconsistent monitor cannot manufacture a PASS.
    assert check(instance, program, budget=1000).status == "NONPROGRESS"


def test_tiny_budget_is_incomplete_not_a_feasibility_verdict():
    instance = _instance(threads=1)
    program = _program(instance, ir.Branch("ready", 1, 0), _copy("@export:reg", "dest", 2), ir.Return())
    assert check(instance, program, budget=1).status == "INCOMPLETE"
    assert check(instance, program, budget=0).explored_states == 0
    assert check(instance, program, budget=1000).status == "PASS"


def test_ir_roundtrip_preserves_cycles_and_digest():
    instance = _instance(threads=1)
    program = _program(instance, ir.Branch("ready", 1, 0), _copy("@export:reg", "dest", 2), ir.Return(instance.demands))
    parsed = ir.parse(ir.to_json_obj(program))
    assert parsed == program
    assert ir.digest(parsed) == ir.digest(program)
    assert check(instance, parsed, budget=1000).status == "PASS"
    invalid = ir.to_json_obj(program)
    invalid["graphs"]["hook"]["nodes"][0] = {"kind": "redistribute"}
    with pytest.raises(ir.IRError):
        ir.parse(invalid)


def test_runtime_binding_worlds_share_one_program():
    instance = _instance(threads=1)
    binding = _binding(expr.Domain("slot_base", "int", values=(0, 1)))
    instance = _rebuild(instance, bindings=(binding,))
    program = _program(instance, ir.Branch("ready", 1, 0), _copy("@export:reg", "dest", 2), ir.Return())
    result = check(instance, program, budget=1000)
    assert result.status == "PASS"
    assert [row["bindings"] for row in result.provenance["terminal_deliveries"]] == [{"slot_base": 0}, {"slot_base": 1}]


def _direct_128():
    source = _instance(threads=128, width=32, tmem=True, ready=True)
    imp = replace(source.imp, lends=tuple(resource for resource in source.imp.lends if resource.name in {"dest", "metadata"}))
    instance = _rebuild(source, imp=imp, profile=_profile(warp_size=32, packet_columns=16))
    packet_base = ["mulc", 2, ["divc", ["coord", "participant"], 32]]
    program = _program(instance,
        ir.Issue("tmem_packet_receipt", {"packet": packet_base, "dest": "dest", "offset": 0, "phase": "issue", "handle": "r"}, 1),
        ir.Issue("tmem_packet_receipt", {"phase": "wait", "handle": "r"}, 2),
        ir.Issue("tmem_packet_receipt", {"packet": ["add", packet_base, ["const", 1]], "dest": "dest", "offset": 16, "phase": "issue", "handle": "r"}, 3),
        ir.Issue("tmem_packet_receipt", {"phase": "wait", "handle": "r"}, 4),
        ir.Return())
    return instance, program


def test_128_thread_direct_receipt_certifies_every_destination():
    from time import monotonic

    instance, program = _direct_128()
    started = monotonic()
    result = check(instance, program, budget=100000)
    assert result.status == "PASS"
    assert monotonic() - started < 20
    assert result.provenance["terminal_deliveries"][0]["outputs"] == [list(row) for row in instance.demands]


def _bound_packets(*, permuted=False):
    source = _instance(threads=4, width=2, tmem=True, ready=True)
    base = expr.Domain("tmem_base", "bits[16]", values=(0, 2))
    domains = {domain.name: domain for domain in source.export.identity.coords}
    domains["tmem_base"] = base
    dp = ["add", ["mulc", 2, ["modc", ["coord", "x"], 2]], ["divc", ["coord", "x"], 2]] if permuted else ["coord", "x"]
    placement = replace(source.export.placements[0], components={
        "dp": expr.parse(dp, domains),
        "column": expr.parse(["add", ["coord", "slot"], ["toint", ["bind", "tmem_base"]]], domains),
    })
    export = replace(source.export, placements=(placement,))
    return _rebuild(source, export=export, bindings=(_binding(base),), profile=_profile(warp_size=2, packet_columns=1))


def _packet_member(instance, actions, half):
    domains = {**instance.derived.binding_domains, "participant": expr.Domain("participant", "int", 0, 4)}
    for action in actions:
        if not isinstance(action, ir.Issue) or action.primitive != "tmem_packet_receipt":
            continue
        if action.operands.get("phase") != "issue" or action.operands["dest"] != "dest" or action.operands["offset"] != half:
            continue
        raw = action.operands["packet"]
        selector = expr.parse(["const", raw] if isinstance(raw, int) else raw, domains)
        if all(instance.packets[expr.evaluate(selector, {"participant": participant, "tmem_base": base})].dp == participant // 2 * 2
               and instance.packets[expr.evaluate(selector, {"participant": participant, "tmem_base": base})].column == base + half
               for base in (0, 2) for participant in range(4)):
            return action
    raise AssertionError("physical per-warp/per-binding packet selector is absent")


def test_source_receipt_permission_and_binding_aware_domain_membership():
    instance = _bound_packets()
    actions = instance.action_domain(families=frozenset({"tmem_packet_receipt"}))["hook"]
    first, second = _packet_member(instance, actions, 0), _packet_member(instance, actions, 1)
    wait = next(action for action in actions if isinstance(action, ir.Issue) and action.operands.get("phase") == "wait")
    program = _program(instance, replace(first, next=1), replace(wait, next=2), replace(second, next=3), replace(wait, next=4), ir.Return())
    result = check(instance, program, budget=20000)
    assert result.status == "PASS"
    assert [entry["bindings"] for entry in result.provenance["terminal_deliveries"]] == [{"tmem_base": 0}, {"tmem_base": 2}]
    denied = _rebuild(instance, export=replace(instance.export, read_capabilities=()))
    assert check(denied, program, budget=20000).status == "UNSAFE"
    assert not any(isinstance(action, ir.Issue) and action.primitive == "tmem_packet_receipt"
                   for action in denied.action_domain(families=frozenset({"tmem_packet_receipt"}))["hook"])


def _round_exchange(instance, *, synchronize=True, alias=False):
    participant, slot = ["coord", "participant"], ["coord", "slot"]
    own = ["mulc", 4, ["add", ["mulc", 2, participant], slot]]
    other = ["mulc", 4, ["add", ["mulc", 2, ["sub", ["const", 1], participant]], slot]]
    if alias:
        own = ["mulc", 4, slot]
    nodes = [ir.Issue("smem_store_round", {"width": 2, "src": "@export:reg", "dst": "scratch", "address": own}, 1)]
    if synchronize:
        nodes.append(ir.Issue("cta_rendezvous", {"resource": "barrier"}, 2))
    nodes.append(ir.Issue("smem_load_round", {"width": 2, "src": "scratch", "dst": "dest", "address": other}, len(nodes) + 1))
    if synchronize:
        nodes.append(ir.Issue("cta_rendezvous", {"resource": "barrier"}, len(nodes) + 1))
    nodes.append(ir.Return())
    return _program(instance, *nodes)


def test_rounds_interleave_individual_slots_and_publish_only_at_rendezvous():
    instance = _instance(width=2, transpose=True, ready=True)
    program = _round_exchange(instance)
    machine = instance.with_program(program)
    initial = machine.initial()
    first = _step(machine, initial, "step:hook:0")
    assert first.round_positions == (1, 0)
    assert first.positions == (0, 0)
    interleaved = _step(machine, first, "step:hook:1")
    assert interleaved.round_positions == (1, 1)
    assert interleaved.published == initial.published
    result = check(instance, program, budget=20000)
    assert result.status == "PASS"
    assert result.provenance["terminal_deliveries"][0]["outputs"] == [list(row) for row in instance.demands]
    reduction = next(item for item in result.reductions if item["kind"] == "checked-persistent-set")
    assert "smem_store_round" in reduction["transition_classes"]
    assert check(instance, _round_exchange(instance, synchronize=False), budget=20000).status == "UNSAFE"
    assert check(instance, _round_exchange(instance, alias=True), budget=20000).status == "UNSAFE"
    first_node = replace(program.graphs["hook"].nodes[0], operands={**program.graphs["hook"].nodes[0].operands, "width": 3})
    too_wide = _program(instance, first_node, *program.graphs["hook"].nodes[1:])
    assert check(instance, too_wide, budget=20000).status == "UNSAFE"


def test_round_domain_contains_public_bit_permutation_relay():
    instance = _bound_packets(permuted=True)
    families = frozenset({"tmem_packet_receipt", "smem_store_round", "smem_load_round", "cta_rendezvous"})
    actions = instance.action_domain(families=families)["hook"]
    domains = {"participant": expr.Domain("participant", "int", 0, 4), "slot": expr.Domain("slot", "int", 0, 2)}
    def find_round(family):
        for action in actions:
            if not isinstance(action, ir.Issue) or action.primitive != family or action.operands["width"] != 2:
                continue
            if (action.operands["src"], action.operands["dst"]) != (("dest", "scratch") if family == "smem_store_round" else ("scratch", "dest")):
                continue
            address = expr.parse(action.operands["address"], domains)
            expected = lambda p, s: 4 * ((p if family == "smem_store_round" else (p % 2) * 2 + p // 2) * 2 + s)
            if all(expr.evaluate(address, {"participant": p, "slot": s}) == expected(p, s) for p in range(4) for s in range(2)):
                return action
        raise AssertionError("published-contract round address is absent")
    first, second = _packet_member(instance, actions, 0), _packet_member(instance, actions, 1)
    wait = next(action for action in actions if isinstance(action, ir.Issue) and action.operands.get("phase") == "wait")
    barrier = next(action for action in actions if isinstance(action, ir.Issue) and action.primitive == "cta_rendezvous")
    program = _program(instance, replace(first, next=1), replace(wait, next=2), replace(second, next=3), replace(wait, next=4),
                       replace(find_round("smem_store_round"), next=5), replace(barrier, next=6),
                       replace(find_round("smem_load_round"), next=7), replace(barrier, next=8), ir.Return())
    result = check(instance, program, budget=100000)
    assert result.status == "PASS"
    assert all(entry["outputs"] == [list(row) for row in instance.demands] for entry in result.provenance["terminal_deliveries"])
    full = instance.action_domain()["hook"]
    assert actions == tuple(node for node in full if not isinstance(node, ir.Issue) or node.primitive in families)


def test_round_local_slot_does_not_capture_a_published_slot_binding():
    source = _instance(width=2, ready=True)
    x, component = expr.Domain("x", "int", 0, 2), expr.Domain("component", "int", 0, 2)
    domains = {"x": x, "component": component}
    identity = replace(source.export.identity, coords=(x, component))
    placement = replace(source.export.placements[0], components={
        "participant": expr.parse(["coord", "x"], domains), "position": expr.parse(["coord", "component"], domains),
    })
    export = replace(source.export, identity=identity, placements=(placement,))
    demand = DemandView("positions", (x, component), None,
                        {name: expr.parse(["coord", name], domains) for name in domains},
                        expr.parse(["coord", "x"], domains), expr.parse(["coord", "component"], domains))
    imp = replace(source.imp, identity=identity, demands=(demand,))
    lends = tuple(replace(resource, extent=32) if resource.name == "scratch" else resource for resource in source.imp.lends)
    binding = _binding(expr.Domain("slot", "bits[8]", values=(1, 2)))
    instance = _rebuild(source, export=export, imp=replace(imp, lends=lends), bindings=(binding,))
    address = ["mulc", 4, ["add", ["add", ["mulc", 2, ["coord", "participant"]], ["coord", "slot"]],
                                ["mulc", 4, ["sub", ["toint", ["bind", "slot"]], ["const", 1]]]]]
    program = _program(instance,
        ir.Issue("smem_store_round", {"width": 2, "src": "@export:reg", "dst": "scratch", "address": address}, 1),
        ir.Issue("cta_rendezvous", {"resource": "barrier"}, 2),
        ir.Issue("smem_load_round", {"width": 2, "src": "scratch", "dst": "dest", "address": address}, 3),
        ir.Return())
    result = check(instance, program, budget=20000)
    assert result.status == "PASS"
    assert result.provenance["terminal_deliveries"] == [
        {"bindings": {"slot": slot}, "outputs": [list(row) for row in instance.demands]} for slot in (1, 2)
    ]


def test_128_actor_rounds_use_exact_arrival_frontier_quotient():
    instance = _instance(threads=128, width=2, ready=True)
    address = ["mulc", 4, ["add", ["mulc", 2, ["coord", "participant"]], ["coord", "slot"]]]
    program = _program(instance,
        ir.Issue("smem_store_round", {"width": 2, "src": "@export:reg", "dst": "scratch", "address": address}, 1),
        ir.Issue("cta_rendezvous", {"resource": "barrier"}, 2),
        ir.Issue("smem_load_round", {"width": 2, "src": "scratch", "dst": "dest", "address": address}, 3),
        ir.Issue("cta_rendezvous", {"resource": "barrier"}, 4),
        ir.Return())
    result = check(instance, program, budget=10000)
    assert result.status == "PASS"
    assert result.provenance["terminal_deliveries"][0]["outputs"] == [list(row) for row in instance.demands]
    quotient = next(item for item in result.reductions if item["kind"] == "checked-rendezvous-frontier")
    assert quotient["applications"] > 0 and quotient["verified"]
    machine = instance.with_program(program)
    state = machine.initial()
    for actor in range(128):
        for _ in range(2):
            state = _step(machine, state, f"step:hook:{actor}")
    for actor in range(128):
        assert not any(transition.label == "rendezvous:barrier" for transition in machine.enabled(state))
        state = _step(machine, state, f"step:hook:{actor}")
        assert len(state.arrivals[0].actors) == actor + 1
    assert tuple(transition.label for transition in machine.enabled(state)) == ("rendezvous:barrier",)
    published = _step(machine, state, "rendezvous:barrier")
    assert published.arrivals == ()
    assert published.published == published.cells


def test_identity_chunks_preserve_prior_states_and_structural_hashes():
    original = IdentityVector(range(130))
    changed = original.updated(((63, 1000), (64, 1001), (64, 64)))
    assert tuple(original) == tuple(range(130))
    assert changed[63:65] == (1000, 64)
    restored = changed.updated(((63, 63),))
    assert restored == original
    assert hash(restored) == hash(original)
    assert {original: "original"}[restored] == "original"


def test_warp_factorization_preserves_intra_warp_premature_reads():
    source = _bound_packets()
    lends = tuple(replace(resource, extent=4) if resource.name == "metadata" else resource for resource in source.imp.lends)
    instance = _rebuild(source, imp=replace(source.imp, lends=lends))
    actions = instance.action_domain(families=frozenset({"tmem_packet_receipt"}))["hook"]
    first, second = _packet_member(instance, actions, 0), _packet_member(instance, actions, 1)
    wait = next(action for action in actions if isinstance(action, ir.Issue) and action.operands.get("phase") == "wait")
    program = _program(instance, ir.Meta("second_warp", ["divc", ["coord", "participant"], 2], 1),
        replace(first, next=2), ir.Branch("second_warp", 3, 4), _copy("dest", "temp", 4),
        replace(wait, next=5), replace(second, next=6), replace(wait, next=7), ir.Return())
    result = check(instance, program, budget=20000)
    assert result.status == "UNSAFE"
    assert "before completion" in result.violations[0].message
    assert any(actor in {2, 3} for step in result.violations[0].detail["trace"] for actor in step["actors"])
    assert any(item["kind"] == "checked-independent-warp-product" for item in result.reductions)


def test_zero_outstanding_generation_capacity_rejects_starting_a_rendezvous():
    source = _instance(width=2, transpose=True, ready=True)
    lends = tuple(replace(resource, sync=ResourceSync(2, 0)) if resource.name == "barrier" else resource
                  for resource in source.imp.lends)
    instance = _rebuild(source, imp=replace(source.imp, lends=lends))
    result = check(instance, _round_exchange(instance), budget=10000)
    assert result.status == "UNSAFE"
    assert "outstanding synchronization-generation capacity" in result.violations[0].message


def test_receipts_cannot_cross_warp_subpartitions_even_when_identities_match():
    source = _bound_packets()
    domains = {domain.name: domain for domain in source.imp.demands[0].domain}
    demand = replace(source.imp.demands[0], participant=expr.parse(
        ["modc", ["add", ["coord", "x"], ["const", 2]], 4], domains))
    instance = _rebuild(source, imp=replace(source.imp, demands=(demand,)))
    actions = instance.action_domain(families=frozenset({"tmem_packet_receipt"}))["hook"]
    first, second = _packet_member(instance, actions, 0), _packet_member(instance, actions, 1)
    def cross_warp(node, next):
        packet = node.operands["packet"]
        packet = ["const", packet] if type(packet) is int else packet
        return replace(node, operands={**node.operands, "packet": ["modc", ["add", packet, ["const", 4]], 8]}, next=next)
    wait = next(node for node in actions if isinstance(node, ir.Issue) and node.operands.get("phase") == "wait")
    # This rotation would deliver the requested identities if cross-subpartition
    # reads existed.  Identity agreement must not manufacture that capability.
    program = _program(instance, cross_warp(first, 1), replace(wait, next=2),
                       cross_warp(second, 3), replace(wait, next=4), ir.Return())
    result = check(instance, program, budget=20000)
    assert result.status == "UNSAFE"
    assert "own DP subpartition" in result.violations[0].message
    assert result.violations[0].detail["trace"][-1]["primitive"] == "tmem_packet_receipt"


def test_grounded_packet_vocabulary_stays_within_each_issuing_warp():
    instance = _bound_packets()
    region = instance.static_region(families=frozenset({"tmem_packet_receipt"}))
    domains = {**instance.derived.binding_domains, "participant": expr.Domain("participant", "int", 0, 4)}
    for action in region["actions"]:
        if action.primitive != "tmem_packet_receipt" or action.operands.get("phase") != "issue":
            continue
        raw = action.operands["packet"]
        selector = expr.parse(["const", raw] if type(raw) is int else raw, domains)
        for world in region["worlds"]:
            for participant in region["participants"]:
                packet = region["packets"][expr.evaluate(selector, {**world, "participant": participant})]
                assert packet["dp"] == participant // instance.warp_size * instance.warp_size


def test_private_round_slots_preserve_sequential_offsets():
    instance = _instance(threads=1, width=3, ready=True)
    seed = ir.Issue("reg_copy_round", {"src": "@export:reg", "dst": "temp", "width": 3}, 1)
    overlap = ir.Issue("reg_copy_round", {"src": "temp", "dst": "temp", "width": 2,
                                       "src_offset": 0, "offset": 1}, 2)
    model = instance.with_program(_program(instance, seed, overlap, ir.Return()))
    state = model.initial()
    first = _step(model, state, "step:hook:0")
    assert first.positions == (0,) and first.round_positions == (1,)
    assert first.cells[model._cell_index[model._declared_cell("temp", 0, 0)]] == 0
    assert first.cells[model._cell_index[model._declared_cell("temp", 0, 1)]] == EMPTY
    state = first
    for _ in range(4):
        state = _step(model, state, "step:hook:0")
    assert [state.cells[model._cell_index[model._declared_cell("temp", 0, slot)]] for slot in range(3)] == [0, 0, 0]
    assert state.positions == (2,) and state.round_positions == (0,)
    too_wide = replace(seed, operands={**seed.operands, "src_offset": 1})
    assert check(instance, _program(instance, too_wide, ir.Return()), budget=1000).status == "UNSAFE"
    copied = ir.Issue("reg_copy_round", {"src": "temp", "dst": "dest", "width": 3}, 3)
    checked = check(instance, _program(instance, seed, overlap, copied, ir.Return()), budget=1000)
    assert checked.status == "UNSAFE"
    assert "wrong logical identity" in checked.violations[0].message


def test_shared_export_is_relative_and_disjoint_from_fixed_scratch():
    source = _instance(threads=2, ready=True)
    placement = PlacementView(None, "smem", "byte", {
        "address": expr.parse(["mulc", 4, ["coord", "x"]], {"x": source.export.identity.coords[0]})})
    export = replace(source.export, placements=(placement,), read_capabilities=("smem_load_scalar",))
    instance = _rebuild(source, export=export)
    program = _program(instance, ir.Issue("smem_load_round", {
        "src": "@export:smem", "dst": "dest", "width": 1,
        "address": ["mulc", 4, ["coord", "participant"]]}, 1), ir.Return())
    assert instance._declared_cell("@export:smem", 0, 0) != instance._declared_cell("scratch", 0, 0)
    assert check(instance, program, budget=10000).status == "PASS"
    not_ready = _rebuild(instance, sites=(replace(instance.sites[0], observes=()),))
    assert check(not_ready, program, budget=10000).status == "UNSAFE"
    overwrite = _program(instance, program.graphs["hook"].nodes[0], ir.Issue("smem_store_round", {
        "src": "dest", "dst": "@export:smem", "width": 1,
        "address": ["mulc", 4, ["coord", "participant"]]}, 2), ir.Return())
    failed = check(instance, overwrite, budget=10000)
    assert failed.status == "UNSAFE"
    assert "read-only export" in failed.violations[0].message
    cyclic = replace(export, readiness=Readiness("done", "done"))
    with pytest.raises(ModelError, match="readiness"):
        _rebuild(source, export=cyclic)


def test_participant_guards_are_recomputable_but_binding_metadata_needs_storage():
    source = _instance(ready=True)
    imp = replace(source.imp, lends=tuple(resource for resource in source.imp.lends if resource.name != "metadata"))
    binding = _binding(expr.Domain("choice", "int", values=(0, 1)))
    instance = _rebuild(source, imp=imp, bindings=(binding,))
    suffix = (ir.Branch("which", 2, 2), _copy("@export:reg", "dest", 3), ir.Return())
    static = _program(instance, ir.Meta("which", ["coord", "participant"], 1), *suffix)
    assert check(instance, static, budget=1000).status == "PASS"
    dynamic = _program(instance, ir.Meta("which", ["bind", "choice"], 1), *suffix)
    rejected = check(instance, dynamic, budget=1000)
    assert rejected.status == "UNSAFE"
    assert "metadata" in rejected.violations[0].message


@pytest.mark.parametrize("missing", ["converged", "full_warps"])
def test_shuffle_requires_both_declared_warp_guarantees(missing):
    source = _instance(transpose=True, ready=True)
    site = replace(source.sites[0], guarantees=tuple(guarantee for guarantee in source.sites[0].guarantees if guarantee != missing))
    instance = _rebuild(source, sites=(site,))
    program = _program(instance, ir.Issue("shfl_xor", {
        "src": _cell("@export:reg"), "dst": _cell("dest"), "mask": 1}, 1), ir.Return())
    assert check(instance, program, budget=1000).status == "UNSAFE"
    assert not any(isinstance(action, ir.Issue) and action.primitive == "shfl_xor"
                   for action in instance.action_domain(families=frozenset({"shfl_xor"}))["hook"])


@pytest.mark.parametrize("missing", ["converged", "no_early_return"])
def test_lent_barrier_does_not_replace_site_guarantees(missing):
    source = _instance(ready=True)
    site = replace(source.sites[0], guarantees=tuple(guarantee for guarantee in source.sites[0].guarantees if guarantee != missing))
    instance = _rebuild(source, sites=(site,))
    program = _program(instance, _copy("@export:reg", "dest", 1),
                       ir.Issue("cta_rendezvous", {"resource": "barrier"}, 2), ir.Return())
    assert check(instance, program, budget=1000).status == "UNSAFE"
    assert not any(isinstance(action, ir.Issue) and action.primitive == "cta_rendezvous"
                   for action in instance.action_domain(families=frozenset({"cta_rendezvous"}))["hook"])


def test_backing_does_not_grant_shared_store_permission():
    source = _instance(threads=1, ready=True)
    backing = next(resource for resource in source.imp.lends if resource.name == "scratch")
    imp = replace(source.imp, lends=tuple(resource for resource in source.imp.lends if resource != backing))
    instance = _rebuild(source, export=replace(source.export, backing=(backing,)), imp=imp)
    program = _program(instance, ir.Issue("smem_store_scalar", {
        "src": _cell("@export:reg"), "dst": _cell("scratch")}, 1), ir.Return())
    result = check(instance, program, budget=1000)
    assert result.status == "UNSAFE"
    assert "not permitted" in result.violations[0].message
    assert not any(isinstance(action, ir.Issue) and action.primitive.startswith("smem_store")
                   for action in instance.action_domain(families=frozenset({"smem_store_scalar", "smem_store_round"}))["hook"])


@pytest.mark.parametrize("primitive", [
    "smem_store_scalar", "smem_store_round", "smem_load_scalar", "smem_load_round",
    "reg_copy", "reg_copy_round", "shfl_xor", "tmem_packet_receipt",
])
def test_every_payload_write_preserves_export_backing(primitive):
    source = _instance(threads=1, ready=True, tmem=primitive == "tmem_packet_receipt")
    shared_read = primitive.startswith("smem_load")
    if shared_read:
        placement = PlacementView(None, "smem", "byte", {"address": expr.parse(["const", 0], {})})
        source = _rebuild(source, export=replace(source.export, placements=(placement,), read_capabilities=("smem_load_scalar",)))
    source_name = "@export:smem" if shared_read else "@export:reg"
    name = "scratch" if primitive.startswith("smem_store") else "temp"
    storage = next(resource for resource in source.imp.lends if resource.name == name)
    backing = replace(storage, name="backing", owner="producer")
    instance = _rebuild(source, export=replace(source.export, backing=(backing,)))
    destination = "backing"  # owner backing is not model storage: only @export:* reads reach it
    if primitive == "tmem_packet_receipt":
        issue = ir.Issue(primitive, {"packet": 0, "dest": destination, "phase": "issue", "handle": "r"}, 1)
    elif primitive.endswith("_round"):
        operands = {"width": 1, "src": source_name, "dst": destination}
        if primitive in {"smem_store_round", "smem_load_round"}:
            operands["address"] = 0
        issue = ir.Issue(primitive, operands, 1)
    else:
        operands = {"src": _cell(source_name), "dst": _cell(destination)}
        if primitive == "shfl_xor":
            operands["mask"] = 0
        issue = ir.Issue(primitive, operands, 1)
    result = check(instance, _program(instance, issue, ir.Return()), budget=1000)
    assert result.status == "UNSAFE"


def test_lend_lifetime_must_reach_import_completion():
    source = _instance(ready=True)
    lends = tuple(replace(resource, lifetime={"from": "invoke", "to": "reclaim"})
                  if resource.name == "scratch" else resource for resource in source.imp.lends)
    with pytest.raises(ModelError, match="lifetime through import completion"):
        _rebuild(source, imp=replace(source.imp, lends=lends))


def test_export_backing_cannot_supply_polling_metadata_capacity():
    source = _instance(threads=1)
    metadata = next(resource for resource in source.imp.lends if resource.domain == "metadata")
    imp = replace(source.imp, lends=tuple(resource for resource in source.imp.lends if resource != metadata))
    instance = _rebuild(source, export=replace(source.export, backing=(metadata,)), imp=imp)
    program = _program(instance, ir.Issue("observe_ready", {"dest": "seen"}, 1),
                       ir.Branch("seen", 2, 0), _copy("@export:reg", "dest", 3), ir.Return())
    result = check(instance, program, budget=1000)
    assert result.status == "UNSAFE"
    assert "metadata" in result.violations[0].message


def test_lent_barrier_does_not_allow_modifying_an_exported_barrier():
    source = _instance(threads=1, ready=True)
    barrier = next(resource for resource in source.imp.lends if resource.domain == "barrier")
    backing = replace(barrier, name="owner_barrier", owner="producer", location=ResourceLocation(None, 2))
    instance = _rebuild(source, export=replace(source.export, backing=(backing,)))
    program = _program(instance, _copy("@export:reg", "dest", 1),
                       ir.Issue("cta_rendezvous", {"resource": "owner_barrier"}, 2), ir.Return())
    result = check(instance, program, budget=1000)
    assert result.status == "UNSAFE"


def test_catalogue_offers_the_packet_receipt_only_on_targets_with_tmem():
    sm103, sm90 = CATALOGUE(get_profile("sm_103a")), CATALOGUE(get_profile("sm_90a"))
    assert "tmem_packet_receipt" in sm103
    assert set(sm103) - set(sm90) == {"tmem_packet_receipt"}
    assert set(CATALOGUE(_profile(tmem=False))) == set(sm90)
    assert CATALOGUE(_profile(tmem_dps=64)).keys() == sm103.keys()


@pytest.mark.parametrize("profile", [get_profile("sm_90a"), get_profile("sm_120a"), _profile(tmem=False, warp_size=32, packet_columns=16)])
def test_tmem_source_on_a_target_without_tmem_is_contract_invalid(profile):
    with pytest.raises(ModelError, match="space tmem is absent from the target profile") as caught:
        _instance(threads=32, width=16, tmem=True, ready=True, profile=profile)
    assert caught.value.status == "CONTRACT_INVALID"
    assert [failure.clause_id for failure in caught.value.failures] == ["T-CONTRACT"]
    source = _instance(threads=32, width=16, tmem=True, ready=True)
    admission, admitted = plan.admit(source.export, source.imp, replace(source.derived, catalogue="warp-readonly-v1"), profile)
    assert (admission.status, admitted) == ("CONTRACT_INVALID", None)


def test_register_source_models_on_both_targets_with_target_specific_digests():
    sm103 = _instance(threads=32, width=16, ready=True, profile=get_profile("sm_103a"))
    sm90 = _instance(threads=32, width=16, ready=True, profile=get_profile("sm_90a"))
    assert (sm103.tmem_lanes, sm90.tmem_lanes) == (128, None)
    assert sm90.cells == sm103.cells and sm90.demands == sm103.demands
    assert sm90.model_digest != sm103.model_digest
    assert sm90.static_region()["ok"] and sm90.static_region()["packet_subpartition"] is None


def test_finite_profile_must_state_every_derivation_option():
    source = _instance(threads=1)
    with pytest.raises(ModelError, match="does not state 'fairness'"):
        plan.derive(source.export, source.imp, source.sites, (), {"warp_size": 1, "packet_columns": 1})
    with pytest.raises(ModelError, match="unknown primitive catalogue"):
        _rebuild(source, profile=_profile(catalogue="tcgen06-v2", warp_size=1, packet_columns=1))
