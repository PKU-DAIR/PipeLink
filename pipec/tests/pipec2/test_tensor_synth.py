from __future__ import annotations

from dataclasses import replace

import pytest

from pipec2.canonjson import canonical_dumps
from pipec2.model import TensorConnection
from pipec2.profile import get_profile
from pipec2.schema import load_pipeline, load_stage_manifest
from pipec2.tensor import check, ir, machine, plan, synth
from pipec2.tensor.contracts import TensorContractError, load_stage_tensor
from test_schema import _valid_pipeline, _valid_stage
from test_tensor_machine import _profile


def _domain(name, size):
    return {"name": name, "sort": "int", "lo": 0, "hi": size}


def _resource(name, domain, extent):
    return {
        "name": name, "domain": domain, "extent": extent,
        "align": 4 if domain == "smem" else 1,
        "scope": "thread" if domain == "reg" else "cta",
        "binding": "relocatable" if domain in {"smem", "barrier"} else "fixed",
        "location": {"handle": f"pipelink_tensor_test_{name}" if domain in {"smem", "barrier"} else None,
                     "base": None if domain in {"smem", "barrier"} else 0},
        "owner": "load", "lifetime": {"from": "ready", "to": "done"},
        "sync": {"generations": 1, "capacity": 1} if domain == "barrier" else None,
    }


def _fixture(*, readiness=True, shared=False, already_ready=False, width=1):
    participants = 2 if shared else 1
    size = 2 if shared else width
    identity = {"result": "acc", "work": "tile", "version": "final",
                "coords": [_domain("x", size)], "dtype": "f32", "interpretation": "ieee754"}
    lends = [_resource("dest", "reg", width)]
    if shared:
        lends += [_resource("scratch", "smem", 8),
                  _resource("barrier", "barrier", 1)]
    frame = _resource("frame", "reg", width)
    frame["binding"] = "caller_frame"
    lends.append(frame)
    if not shared:
        outer = _resource("outer", "barrier", 1)
        outer.update(binding="fixed", location={"handle": None, "base": 0})
        lends.append(outer)
    return {
        "version": "tensor-v2",
        "exports": [{"name": "source", "identity": identity,
                     "placements": [{"guard": None, "space": "reg", "units": "scalar",
                                     "components": {"participant": ["coord", "x"] if shared else ["const", 0],
                                                    "position": ["const", 0] if shared else ["coord", "x"]}}],
                     "read_capabilities": ["reg_copy"],
                     "readiness": {"after": "ready", "reuse_after": "done"},
                     "fixed_bindings": {}, "backing": [], "calls": {"produce": "produce", "release": None}}],
        "imports": [{"name": "sink", "identity": identity,
                     "demands": [{"kind": "positions", "domain": [_domain("i", size)],
                                  "guard": None, "identity_coords": {"x": ["coord", "i"]},
                                  "participant": ["sub", ["const", 1], ["coord", "i"]] if shared else ["const", 0],
                                  "position": ["const", 0] if shared else ["coord", "i"]}],
                     "callable": {"symbol": "pipelink_tensor_test_import",
                                  "declaration": 'extern "C" __device__ void pipelink_tensor_test_import(float*);',
                                  "events": {"ready": "pipelink_tensor_test_import_ready", "done": "pipelink_tensor_test_import_done"},
                                  "params": [{"name": "dest", "meaning": "dest_fragment", "ctype": "float*"}]},
                     "sites": ["load"], "completion": "done", "lends": lends,
                     "rendezvous": [{"event": event, "barrier": "barrier" if shared else "outer", "cohort": {"set": list(range(participants))}}
                                    for event in ("ready", "done")]}],
        "sites": [{"name": "load", "role": "load", "participants": {"set": list(range(participants))},
                   "invocation": "tile",
                   "guarantees": ["enter_once", "converged", "no_early_return"] if shared else ["enter_once"],
                   "observes": ["ready"] if readiness or already_ready else []}],
        "bindings": [],
    }


def _loaded(raw, profile=None):
    tensor = load_stage_tensor(raw, "$.tensor")
    profile = get_profile("sm_103a") if profile is None else profile
    derived = plan.derive(tensor.exports[0], tensor.imports[0], tensor.sites, tensor.bindings, profile)
    return tensor, derived


def _instance(raw, profile=None):
    profile = get_profile("sm_103a") if profile is None else profile
    tensor, derived = _loaded(raw, profile)
    return machine.Instance.build(tensor.exports[0], tensor.imports[0], derived, profile)


def _admission(raw):
    tensor, derived = _loaded(raw)
    admission, _ = plan.admit(tensor.exports[0], tensor.imports[0], derived, get_profile("sm_103a"))
    return admission


def test_loader_invalid_remains_distinct_from_model_unsupported():
    invalid = _fixture()
    invalid["imports"][0]["sites"] = ["nonexistent"]
    with pytest.raises(TensorContractError) as raised:
        _loaded(invalid)
    assert {failure.clause_id for failure in raised.value.failures} == {"T-CONTRACT"}
    unsupported = _fixture()
    demand = unsupported["imports"][0]["demands"][0]
    del demand["participant"], demand["position"]
    demand.update(kind="memory_view", space="gmem", units="byte", components={"address": ["const", 0]})
    admission = _admission(unsupported)
    assert admission.status == "UNSUPPORTED_MODEL"
    assert "memory_view" in admission.issues[0].message


def test_connection_identity_inconsistency_is_contract_invalid():
    raw = _fixture()
    tensor, _ = _loaded(raw)
    changed = replace(tensor.imports[0], identity=replace(tensor.imports[0].identity, version="different"))
    profile = get_profile("sm_103a")
    derived = plan.derive(tensor.exports[0], changed, tensor.sites, tensor.bindings, profile)
    admission, _ = plan.admit(tensor.exports[0], changed, derived, profile)
    assert admission.status == "CONTRACT_INVALID"
    assert admission.issues[0].clause_id == "T-CONTRACT"


def test_expression_caps_reject_referenced_and_unused_domains(monkeypatch):
    monkeypatch.setattr(plan, "EXPRESSION_CAP", 0)
    admission = _admission(_fixture())
    assert admission.status == "UNSUPPORTED_MODEL"
    raw = _fixture(width=2)
    raw["exports"][0]["placements"][0]["components"]["position"] = ["const", 0]
    monkeypatch.setattr(plan, "EXPRESSION_CAP", 1)
    admission = _admission(raw)
    assert admission.status == "UNSUPPORTED_MODEL"


def test_owner_adequacy_is_not_a_vacuous_synthesis_answer(monkeypatch):
    monkeypatch.setattr(plan, "MODEL_STATE_BUDGET", 0)
    admission = _admission(_fixture())
    assert admission.status == "UNSUPPORTED_MODEL"
    assert not admission.report["owner_adequacy"]["ok"]


def test_missing_identity_records_necessary_impossibility_premise():
    raw = _fixture()
    raw["exports"][0]["placements"][0]["guard"] = ["eq", ["const", 0], ["const", 1]]
    admission = _admission(raw)
    coverage = admission.report["logical_coverage"]
    assert not coverage["ok"]
    assert coverage["missing"][0]["identity"] == [0]
    assert coverage["impossibility_premise"]
    assert coverage["not_sufficient_for_realizability"]


def test_published_readiness_controller_delivers_exact_values():
    instance = _instance(_fixture())
    found = synth.synthesize(instance, synth.Bounds(2, rlimit=10_000_000))
    assert found.status == "MODEL_SAT"
    assert check.check(instance, found.program, budget=100_000).status == "PASS"
    evidence = found.provenance["candidates"][-1]["check"]
    assert evidence["status"] == "PASS"
    assert evidence["provenance"]["terminal_deliveries"][0]["outputs"] == [[0, 0, 0]]


def test_runtime_readiness_requires_room_for_a_polling_step():
    instance = _instance(_fixture(readiness=False))
    too_short = synth.synthesize(instance, synth.Bounds(1, rlimit=20_000_000))
    assert too_short.status == "BOUNDED_UNSAT"
    assert too_short.program is None
    assert too_short.scope_record["bounds"]["K"] == 1
    assert all(candidate["check"]["status"] in {"UNSAFE", "NONPROGRESS"}
               for candidate in too_short.provenance["candidates"])
    found = synth.synthesize(instance, synth.Bounds(2, rlimit=20_000_000))
    assert found.status == "MODEL_SAT"
    checked = check.check(instance, found.program, budget=100_000)
    assert checked.status == "PASS"
    assert checked.provenance["terminal_deliveries"] == [
        {"bindings": {}, "outputs": [[0, 0, 0]]},
    ]


def test_exact_blocking_advances_to_a_different_candidate():
    found = synth.synthesize(_instance(_fixture(readiness=False)), synth.Bounds(1))
    assignments = [candidate["assignment_digest"] for candidate in found.provenance["candidates"]]
    assert found.status == "BOUNDED_UNSAT"
    assert len(assignments) >= 2
    assert len(assignments) == len(set(assignments))


def test_tiny_rlimit_is_unknown_not_unsat():
    found = synth.synthesize(_instance(_fixture()), synth.Bounds(2, rlimit=1))
    assert found.status == "UNKNOWN"
    assert found.program is None


def test_two_thread_shared_exchange_delivers_typed_identities():
    instance = _instance(_fixture(shared=True, already_ready=True, readiness=False))
    found = synth.synthesize(instance, synth.Bounds(3, rlimit=100_000_000))
    assert found.status == "MODEL_SAT"
    checked = check.check(instance, found.program, budget=100_000)
    assert checked.status == "PASS"
    assert all(execution["outputs"] == [[0, 0, 1], [1, 0, 0]]
               for execution in checked.provenance["terminal_deliveries"])


def test_same_exchange_without_shared_storage_is_bounded_unsat():
    raw = _fixture(shared=True, already_ready=True, readiness=False)
    raw["imports"][0]["lends"] = [lend for lend in raw["imports"][0]["lends"] if lend["domain"] != "smem"]
    found = synth.synthesize(_instance(raw), synth.Bounds(3, rlimit=20_000_000))
    assert found.status == "BOUNDED_UNSAT"


def test_static_projection_preserves_unwritten_outputs_and_checks_premise():
    instance = _instance(_fixture(width=2, already_ready=True, readiness=False))
    found = synth.synthesize_static(instance, synth.Bounds(2, action_domain=("reg_copy",)))
    assert found.status == "MODEL_SAT"
    assert found.provenance["synchronous_premise"]["ok"]
    assert check.check(instance, found.program, budget=100_000).status == "PASS"
    too_short = synth.synthesize_static(instance, synth.Bounds(1, action_domain=("reg_copy",)))
    assert too_short.status == "BOUNDED_UNSAT"
    concurrent = synth.synthesize_static(_instance(_fixture(readiness=False)), synth.Bounds(2))
    assert concurrent.status == "UNKNOWN"
    assert not concurrent.provenance["synchronous_premise"]["ok"]


def test_plan_connection_emits_owned_claims_handles_and_deterministic_digest(monkeypatch):
    tensor, _ = _loaded(_fixture(shared=True, already_ready=True, readiness=False))
    base = load_stage_manifest(_valid_stage())
    base = replace(base, target=replace(base.target, schema="2.2"))
    producer = replace(base, id="producer", tensor=tensor)
    consumer = replace(base, id="consumer", tensor=tensor)
    pipeline = load_pipeline(_valid_pipeline())
    connection = TensorConnection("exchange", "producer:source", "consumer:sink")
    monkeypatch.setattr(plan, "SEARCH_K", 3)
    monkeypatch.setattr(plan, "SEARCH_RLIMIT", 100_000_000)
    first = plan.ConnectionPlanner(pipeline, [producer, consumer], connection, get_profile("sm_103a")).next()
    second = plan.ConnectionPlanner(pipeline, [consumer, producer], connection, get_profile("sm_103a")).next()
    assert first.status == second.status == "MODEL_SAT"
    assert first.ir_digest == second.ir_digest == ir.digest(first.program)
    assert canonical_dumps(first.program.to_json_obj()) == canonical_dumps(second.program.to_json_obj())
    assert first.provenance["admission_oracle"]["ok"]
    assert first.provenance["admission_oracle"]["checked_positions"] >= 2
    assert {(claim.domain, claim.extent.const) for claim in first.claims} == {("smem", 8), ("named_barrier", 1)}
    assert all(handle.owner.startswith("claim:consumer.tensor.exchange:") for handle in first.handles)
    assert first.barrier_requests[0]["owner"] == "consumer.tensor.exchange"


def test_incomplete_class_is_not_erased_by_a_later_exhausted_class(monkeypatch):
    raw = _fixture(shared=True, readiness=False)
    raw["imports"][0]["lends"] = [lend for lend in raw["imports"][0]["lends"] if lend["domain"] != "smem"]
    raw["sites"][0]["guarantees"].append("full_warps")
    tensor, _ = _loaded(raw)
    base = load_stage_manifest(_valid_stage())
    base = replace(base, target=replace(base.target, schema="2.2"))
    stages = [replace(base, id=name, tensor=tensor) for name in ("producer", "consumer")]
    pipeline = load_pipeline(_valid_pipeline())
    connection = TensorConnection("incomplete", "producer:source", "consumer:sink")
    monkeypatch.setattr(plan, "SEARCH_K", 1)
    engine = synth.synthesize
    work_limits = iter((0, 10_000_000))

    def budgeted(instance, bounds, mode):
        if mode == "controller":
            bounds = replace(bounds, rlimit=next(work_limits))
        return engine(instance, bounds, mode)

    monkeypatch.setattr(synth, "synthesize", budgeted)
    result = plan.ConnectionPlanner(pipeline, stages, connection, get_profile("sm_103a")).next()
    statuses = {entry["status"] for entry in result.provenance["classes"]
                if entry["provenance"].get("generator") != "constructive-v1"}
    assert statuses == {"UNKNOWN", "BOUNDED_UNSAT"}
    assert result.status == "UNKNOWN"
    assert result.program is None


def test_native_static_shuffle_reads_the_whole_pre_action_warp():
    raw = _fixture(shared=True, already_ready=True, readiness=False)
    raw["sites"][0]["guarantees"].append("full_warps")
    instance = _instance(raw, _profile(warp_size=2))
    result = synth.synthesize_static(instance, synth.Bounds(2, action_domain=("reg_copy", "shfl_xor")))
    assert result.status == "MODEL_SAT"
    verified = check.check(instance, result.program, budget=100_000)
    assert verified.status == "PASS"
    assert verified.provenance["terminal_deliveries"] == [
        {"bindings": {}, "outputs": [[0, 0, 1], [1, 0, 0]]},
    ]


def test_native_static_exchange_requires_publication():
    raw = _fixture(shared=True, already_ready=True, readiness=False)
    raw["imports"][0]["lends"] = [lend for lend in raw["imports"][0]["lends"] if lend["domain"] != "barrier"]
    raw["imports"][0]["rendezvous"] = []
    result = synth.synthesize_static(
        _instance(raw), synth.Bounds(3, action_domain=("smem_store_round", "smem_load_round")),
    )
    assert result.status == "BOUNDED_UNSAT"
    assert result.program is None


def _native_bank_instance():
    raw = _fixture(shared=True, already_ready=True, readiness=False, width=2)
    raw["exports"][0]["identity"]["coords"] = [_domain("row", 4), _domain("col", 2)]
    raw["exports"][0]["placements"][0].update(
        space="tmem", units="dp/column",
        components={"dp": ["coord", "row"], "column": ["add", ["bind", "base"], ["coord", "col"]]},
    )
    raw["exports"][0]["read_capabilities"] = ["tmem_packet_receipt"]
    domain = {"name": "base", "sort": "int", "values": [0, 2]}
    raw["bindings"] = [{"name": "base", "sort": "int", "domain": domain, "owner": "load",
                        "units": "column", "observation_scope": "cta", "lifetime": "tile"}]
    raw["sites"][0]["participants"] = {"set": [0, 1, 2, 3]}
    raw["imports"][0]["demands"][0].update(
        domain=[_domain("p", 4), _domain("s", 2)],
        identity_coords={"row": ["modc", ["add", ["coord", "p"], ["const", 1]], 4], "col": ["coord", "s"]},
        participant=["coord", "p"], position=["coord", "s"],
    )
    lends = raw["imports"][0]["lends"]
    next(lend for lend in lends if lend["name"] == "scratch")["extent"] = 32
    next(lend for lend in lends if lend["name"] == "barrier")["sync"]["generations"] = 2
    metadata = _resource("metadata", "metadata", 1)
    metadata["scope"] = "warp"
    lends.append(metadata)
    raw["sites"][0]["guarantees"].append("full_warps")
    return _instance(raw, _profile(warp_size=2, packet_columns=1))


def test_static_bank_origins_preserve_versions_and_runtime_bindings():
    instance = _native_bank_instance()
    families = ("tmem_packet_receipt", "smem_store_round", "cta_rendezvous", "smem_load_round")
    found = synth.synthesize_static(instance, synth.Bounds(8, action_domain=families))
    assert found.status == "MODEL_SAT"
    verified = check.check(instance, found.program, budget=100_000)
    assert verified.status == "PASS"
    expected = [[0, 0, 1], [0, 1, 5], [1, 0, 2], [1, 1, 6],
                [2, 0, 3], [2, 1, 7], [3, 0, 0], [3, 1, 4]]
    assert verified.provenance["terminal_deliveries"] == [
        {"bindings": {"base": 0}, "outputs": expected},
        {"bindings": {"base": 2}, "outputs": expected},
    ]
    too_short = synth.synthesize_static(instance, synth.Bounds(6, action_domain=families))
    assert too_short.status == "BOUNDED_UNSAT"


def _pointer_fixture(space, *, threads=2, columns=2, participant_start=0, rows_per_thread=1):
    raw = _fixture(already_ready=True, readiness=False)
    identity = raw["exports"][0]["identity"]
    identity["coords"] = [_domain("r", threads * rows_per_thread), _domain("c", columns)]
    raw["imports"][0]["identity"] = identity
    placement = ({"participant": ["coord", "r"], "position": ["coord", "c"]} if space == "reg" else
                 {"address": ["mulc", 4, ["add", ["mulc", columns + 4, ["coord", "r"]], ["coord", "c"]]]})
    raw["exports"][0]["placements"] = [{"guard": None, "space": space,
                                       "units": "scalar" if space == "reg" else "byte", "components": placement}]
    raw["exports"][0]["read_capabilities"] = ["reg_copy"] if space == "reg" else ["smem_load_scalar"]
    if space == "smem":
        backing = _resource("source_storage", "smem", 4 * (columns + 4) * threads * rows_per_thread)
        backing.update(binding="fixed", location={"handle": None, "base": 0})
        raw["exports"][0]["backing"] = [backing]
    raw["sites"][0]["participants"] = {"set": list(range(participant_start, participant_start + threads))}
    imp = raw["imports"][0]
    imp["callable"] = {"symbol": "pointer_import",
                       "declaration": 'extern "C" __device__ void pointer_import(pipelink_tensor_source, float*);',
                       "events": {"ready": "pointer_import_ready", "done": "pointer_import_done"},
                       "params": [{"name": "source_base", "meaning": "source_base", "ctype": "pipelink_tensor_source"},
                                  {"name": "dest", "meaning": "dest_fragment", "ctype": "float*"}]}
    width = columns * rows_per_thread
    imp["lends"][0]["extent"] = width
    next(lend for lend in imp["lends"] if lend["binding"] == "caller_frame")["extent"] = width
    for rendezvous in imp["rendezvous"]:
        rendezvous["cohort"] = {"set": list(range(participant_start + threads))}
    imp["demands"] = [{"kind": "positions",
                      "domain": [{"name": "t", "sort": "int", "lo": participant_start, "hi": participant_start + threads},
                                 _domain("p", width)],
                      "guard": None, "participant": ["coord", "t"], "position": ["coord", "p"],
                      "identity_coords": {
                          "r": ["add", ["sub", ["coord", "t"], ["const", participant_start]],
                                ["mulc", threads, ["divc", ["coord", "p"], columns]]],
                          "c": ["modc", ["coord", "p"], columns]}}]
    return raw


def _pointer_plan(raw, profile=None):
    profile = get_profile("sm_103a") if profile is None else profile
    tensor, _ = _loaded(raw, profile)
    base = load_stage_manifest(_valid_stage())
    base = replace(base, target=replace(base.target, schema="2.2"))
    producer_bindings = set(replace(tensor.exports[0], fixed_bindings={}).residual_bindings)
    stages = [
        replace(base, id="producer", tensor=replace(
            tensor, imports=(), sites=(),
            bindings=tuple(binding for binding in tensor.bindings if binding.name in producer_bindings))),
        replace(base, id="consumer", tensor=replace(
            tensor, exports=(),
            bindings=tuple(binding for binding in tensor.bindings if binding.name not in producer_bindings))),
    ]
    candidate = plan.ConnectionPlanner(load_pipeline(_valid_pipeline()), stages,
                                     TensorConnection("pointer", "producer:source", "consumer:sink"), profile).next()
    return candidate, tensor.imports[0]


def test_immutable_shared_source_plans_without_store_or_rendezvous():
    raw = _pointer_fixture("smem")
    planned, _ = _pointer_plan(raw)
    assert planned.status == "MODEL_SAT"
    issues = [node for graph in planned.program.graphs.values() for node in graph.nodes if isinstance(node, ir.Issue)]
    assert {node.primitive for node in issues} == {"smem_load_round"}
    assert check.check(_instance(raw), planned.program, budget=100_000).status == "PASS"


def test_full_private_fragment_is_one_checked_round():
    raw = _pointer_fixture("reg", threads=1, columns=32)
    result = synth.synthesize_static(_instance(raw), synth.Bounds(1, action_domain=("reg_copy_round",)))
    assert result.status == "MODEL_SAT"
    issues = [node for graph in result.program.graphs.values() for node in graph.nodes if isinstance(node, ir.Issue)]
    assert len(issues) == 1
    assert issues[0].primitive == "reg_copy_round"
    assert issues[0].operands["width"] == 32
    assert check.check(_instance(raw), result.program, budget=100_000).status == "PASS"


def test_private_round_cannot_exchange_other_lanes():
    raw = _fixture(shared=True, already_ready=True, readiness=False)
    raw["sites"][0]["guarantees"].append("full_warps")
    instance = _instance(raw, _profile(warp_size=2))
    direct = synth.synthesize_static(instance, synth.Bounds(2, action_domain=("reg_copy_round",)))
    assert direct.status == "BOUNDED_UNSAT"
    shuffled = synth.synthesize_static(instance, synth.Bounds(2, action_domain=("reg_copy_round", "shfl_xor")))
    assert shuffled.status == "MODEL_SAT"
    assert any(isinstance(node, ir.Issue) and node.primitive == "shfl_xor"
               for graph in shuffled.program.graphs.values() for node in graph.nodes)
    assert check.check(instance, shuffled.program, budget=100_000).status == "PASS"


def test_permuted_register_bank_stages_from_immutable_source():
    raw = _pointer_fixture("reg", threads=4, columns=2)
    raw["exports"][0]["placements"][0]["components"]["participant"] = [
        "add", ["mulc", 2, ["modc", ["coord", "r"], 2]], ["divc", ["coord", "r"], 2]]
    lends = raw["imports"][0]["lends"]
    lends.extend([_resource("scratch", "smem", 32), _resource("barrier", "barrier", 1)])
    lends[-1]["sync"]["generations"] = 2
    raw["sites"][0]["guarantees"] = ["enter_once", "converged", "no_early_return"]
    instance = _instance(raw)
    families = ("smem_store_round", "smem_load_round", "cta_rendezvous")
    result = synth.synthesize_static(instance, synth.Bounds(4, action_domain=families))
    assert result.status == "MODEL_SAT"
    assert check.check(instance, result.program, budget=100_000).status == "PASS"
    unpublished = synth.synthesize_static(instance, synth.Bounds(4, action_domain=families[:2]))
    assert unpublished.status == "BOUNDED_UNSAT"


def test_a_site_without_full_warps_does_not_remove_the_shuffle_route_from_another_site():
    """Two disjoint sites: A (lanes 0,1, full warps) swaps lanes, B (lane 2) copies locally."""
    raw = _fixture(shared=True, already_ready=True, readiness=False)
    raw["sites"][0]["guarantees"].append("full_warps")
    raw["sites"].append({**raw["sites"][0], "name": "solo", "participants": {"set": [2]},
                         "guarantees": ["enter_once", "converged", "no_early_return"]})
    identity = raw["exports"][0]["identity"]
    identity["coords"] = [_domain("x", 3)]
    raw["imports"][0]["identity"] = identity
    raw["exports"][0]["placements"] = [{"guard": None, "space": "reg", "units": "scalar",
                                       "components": {"participant": ["coord", "x"], "position": ["const", 0]}}]
    raw["exports"][0]["read_capabilities"] = ["reg_copy"]
    imp = raw["imports"][0]
    imp["sites"] = ["load", "solo"]
    imp["lends"] = [lend for lend in imp["lends"] if lend["domain"] != "smem"]
    # lanes 0,1 swap (needs a shuffle at site "load"); lane 2 copies its own value at "solo".
    half = ["divc", ["coord", "t"], 2]
    imp["demands"] = [{"kind": "positions", "domain": [_domain("t", 3)], "guard": None,
                       "participant": ["coord", "t"], "position": ["const", 0],
                       "identity_coords": {"x": ["add", ["modc", ["add", ["coord", "t"], ["sub", ["const", 1], half]], 2],
                                                 ["mulc", 2, half]]}}]
    tensor, _ = _loaded(raw, _profile(warp_size=2))
    base = load_stage_manifest(_valid_stage())
    base = replace(base, target=replace(base.target, schema="2.2"))
    stages = [replace(base, id=name, tensor=tensor) for name in ("producer", "consumer")]
    planner = plan.ConnectionPlanner(load_pipeline(_valid_pipeline()), stages,
                                     TensorConnection("swap", "producer:source", "consumer:sink"), _profile(warp_size=2))
    assert planner.terminal is None, planner.terminal and planner.terminal.admission.to_json_obj()
    assert "warp" in {route["label"] for route in planner.provenance["candidates"]}
    planned = planner.next()
    assert planned.status == "MODEL_SAT"
    issued = {node.primitive for graph in planned.program.graphs.values() for node in graph.nodes if isinstance(node, ir.Issue)}
    assert "shfl_xor" in issued


def test_identities_split_across_tmem_and_shared_placements_keep_a_joint_bring_domain():
    """slot 0 lives in TMEM, slot 1 in shared memory: one bounded route must hold both brings."""
    from test_tensor_machine import _instance as _machine_instance, _rebuild
    from pipec2.tensor import expr, routes
    source = _machine_instance(threads=2, width=2, tmem=True, ready=True, profile=_profile(warp_size=2, packet_columns=1))
    placement = source.export.placements[0]
    domains = {domain.name: domain for domain in source.export.identity.coords}
    tmem = replace(placement, guard=expr.parse(["eq", ["coord", "slot"], ["const", 0]], domains))
    shared = replace(placement, guard=expr.parse(["eq", ["coord", "slot"], ["const", 1]], domains), space="smem",
                     units="byte", components={"address": expr.parse(["mulc", 4, ["coord", "x"]], domains)})
    available = frozenset({"cap:tmem_packet_receipt", "avail:smem", "lend:reg"})
    permitted, _rejected = routes.candidates(routes.source_spaces(replace(source.export, placements=(tmem, shared))),
                                             "reg", available)
    joint = [route for route in permitted if route.label == "direct" and route.source == "tmem+smem"]
    assert [route.tier for route in joint] == [0, 1]
    assert joint[0].bring == ("receipt", "load")
    assert {"tmem_packet_receipt", "smem_load_round"} <= joint[1].searched(available)
    # End to end: the joint bounded domain finds the old four-operation program
    # (receipt issue/wait into dest[0], shared load into temp, copy into dest[1]).
    instance = _rebuild(source, export=replace(source.export, placements=(tmem, shared),
                                               read_capabilities=("tmem_packet_receipt", "smem_load_scalar")))
    bounds = synth.Bounds(joint[1].bound(instance, plan.SEARCH_K), None, tuple(joint[1].searched(available)), 100_000_000)
    found = synth.synthesize(instance, bounds, "static")
    if found.status == "UNKNOWN" and found.provenance.get("static_premise_failed"):
        found = synth.synthesize(instance, bounds, "controller")
    assert found.status == "MODEL_SAT", found.provenance
    issued = {node.primitive for graph in found.program.graphs.values() for node in graph.nodes if isinstance(node, ir.Issue)}
    assert {"tmem_packet_receipt", "smem_load_round"} <= issued or {"tmem_packet_receipt", "smem_load_scalar"} <= issued


@pytest.mark.parametrize("extent", [None, 1])
def test_register_source_requires_sufficient_caller_frame(extent):
    raw = _pointer_fixture("reg", threads=32, columns=2)
    lends = raw["imports"][0]["lends"]
    frame = next(lend for lend in lends if lend["binding"] == "caller_frame")
    if extent is None:
        lends.remove(frame)
    else:
        frame["extent"] = extent
    candidate, _ = _pointer_plan(raw)
    assert candidate.status == "CONTRACT_INVALID"
    assert [(issue.clause_id, issue.path) for issue in candidate.admission.issues] == [
        ("T-PLAN-FRAME", "tensor.imports.sink.lends")]
    assert candidate.admission.issues[0].detail["required_extent"] == 2


def test_outer_barrier_is_claimed_for_sync_free_transport():
    raw = _pointer_fixture("smem", threads=32, columns=2)
    imp = raw["imports"][0]
    imp["lends"] = [lend for lend in imp["lends"] if lend["binding"] != "caller_frame"]
    barrier = next(lend for lend in imp["lends"] if lend["name"] == "outer")
    barrier.update(binding="relocatable", location={"handle": "pipelink_outer", "base": None})
    for event in imp["rendezvous"]:
        event["cohort"] = {"set": list(range(64))}
    candidate, _ = _pointer_plan(raw)
    assert candidate.status == "MODEL_SAT"
    assert not any(isinstance(node, ir.Issue) and node.primitive == "cta_rendezvous"
                   for graph in candidate.program.graphs.values() for node in graph.nodes)
    assert any(claim.handle == "pipelink_outer" and claim.domain == "named_barrier"
               for claim in candidate.claims)
    assert next(request for request in candidate.barrier_requests
                if request["handle"] == "pipelink_outer")["participants"] == list(range(64))
