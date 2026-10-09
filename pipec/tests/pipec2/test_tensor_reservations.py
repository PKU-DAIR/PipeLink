from dataclasses import replace

import pytest

import pipec2.link as link
from pipec2.model import Extent, Handle, Role, Variant
from pipec2.profile import get_profile
from pipec2.tensor import ir, lower, plan
from pipec2.tensor.contracts import Lend, Participants, Rendezvous, ResourceLocation
from test_tensor_link import _composition, _install, _native_tools, _Planner
from test_tensor_lower import _native_plan
from test_tensor_machine import _instance, _rebuild


def _lend(stages, name, handle):
    return Lend(name, "smem", 128, 16, "cta", "relocatable",
                ResourceLocation(handle, None), stages[1].id,
                {"from": "ready", "to": "done"}, None)


@pytest.mark.parametrize("sequence", [((0, 16), (32, 16), (0, 0)),
                                     ((64, 16), (0, 0), (0, 32))])
def test_shared_handle_demands_rebuild_on_candidate_retry(tmp_path, monkeypatch, sequence):
    pipeline, stages, template, adapter = _composition()
    stages = [replace(stage, variants=[Variant({}, 0, None)]) for stage in stages]
    handle = template.handles[0]
    warp = Handle("pipelink_load_warp", "base", "bind", "warp", None,
                  "claim:org.consumer:pipelink_load_warp")
    launch = Handle("pipelink_launch_regs", "regs", "bind", "regs", [96], "launch")
    pipeline = replace(pipeline, handles={warp.name: warp, launch.name: launch})
    stages[1] = replace(stages[1], claims=[
        replace(template.claims[0], domain="warp", handle=warp.name,
                extent=Extent(1, {}, {}), align=1, roles=["load"])])
    lends = tuple(_lend(stages, name, handle.name) for name in ("left", "right"))
    stages[1] = replace(stages[1], roles={"load": Role("org_consumer_load", 1)},
                        tensor=replace(stages[1].tensor, imports=(
        replace(stages[1].tensor.imports[0], lends=lends),)))
    candidates = [replace(template, ir_digest=str(index) * 64,
                          claims=(replace(template.claims[0], extent=Extent(max(demands), {}, {})),)
                          if max(demands) else (),
                          handles=(handle,) if max(demands) else (), barrier_requests=(),
                          provenance={**template.provenance,
                                      "resource_demands": dict(zip(("left", "right"), demands))})
                  for index, demands in enumerate(sequence)]
    _native_tools(monkeypatch)
    _install(monkeypatch, pipeline, stages, candidates[0], adapter)
    monkeypatch.setattr(plan, "ConnectionPlanner", lambda *_args: _Planner(candidates))
    allocations = []
    # _install substitutes the solver; exercise the actual allocator here.
    from pipec2.solve import solve

    def allocate(composition, owners, canon, expansions, profile, ordered_pairs):
        solution = solve(composition, list(owners), dict(canon), list(expansions), profile, ordered_pairs)
        allocations.append((solution.launch.dyn_smem_bytes,
                            composition.forced_bindings.get(handle.name),
                            tuple(c.extent.const for s in owners for c in s.claims if c.domain == "smem")))
        return solution

    monkeypatch.setattr(link, "_solve_pipeline", allocate)

    def emit(candidate, *_args, **_kwargs):
        if candidate.ir_digest != candidates[-1].ir_digest:
            raise lower.LoweringUnsupported("route", "no_recipe", "try next certified route")
        return adapter

    monkeypatch.setattr(lower, "emit_adapter", emit)
    link.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert allocations == [(max(demands), None if max(demands) else 0,
                            (max(demands),) if max(demands) else ()) for demands in sequence]


def test_outer_rendezvous_is_a_live_barrier_without_body_reference():
    instance = _instance(threads=32, width=1, tmem=False, ready=True)
    barrier = next(r for r in instance.imp.lends if r.domain == "barrier")
    barrier = replace(barrier, binding="relocatable", location=ResourceLocation("pipelink_outer", None))
    imp = replace(instance.imp, lends=tuple(barrier if r.domain == "barrier" else r for r in instance.imp.lends),
                  rendezvous=(Rendezvous("ready", barrier.name, Participants(set=tuple(range(32)))),))
    instance = _rebuild(instance, imp=imp)
    pipeline, stages, _, _ = _composition()
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=(replace(imp, name="received"),)))
    program = instance.program({"hook": ir.ControlGraph((ir.Return(),))})
    claims, _, requests, _, demands = plan._resource_demands(
        program, instance.derived, stages[1], pipeline.tensor_connections[0], pipeline, {})
    assert demands[barrier.name] == 1
    assert [(claim.handle, claim.extent.const) for claim in claims] == [("pipelink_outer", 1)]
    assert requests[0]["participants"] == list(range(32))


def test_lowering_rejects_claim_smaller_than_certified_demand():
    candidate, imp, solved, _ = _native_plan(staged=True)
    candidate = replace(candidate, claims=(replace(candidate.claims[0], extent=Extent(124, {}, {})),))
    with pytest.raises(lower.LoweringUnsupported, match="does not cover demand"):
        lower._resource_bindings(candidate, imp, solved, {"scratch"})


def test_materialized_view_reserves_sparse_high_water_not_touched_byte_count():
    from test_tensor_synth import _pointer_plan
    from test_tensor_view import _view_fixture

    raw = _view_fixture(transpose=True)
    scratch = next(r for r in raw["imports"][0]["lends"] if r["name"] == "scratch")
    scratch["extent"] = 128
    raw["imports"][0]["demands"][0]["components"]["address"] = [
        "add", ["const", 16], ["add", ["mulc", 4, ["coord", "row"]],
                                 ["mulc", 32, ["coord", "slot"]]]]
    candidate, _ = _pointer_plan(raw, get_profile("sm_103a"))
    assert candidate.status == "MODEL_SAT"
    assert candidate.provenance["resource_demands"]["scratch"] == 56
    assert next(c.extent.const for c in candidate.claims if c.domain == "smem") == 56


def test_alias_view_and_unused_barrier_have_zero_demand():
    from test_tensor_synth import _pointer_plan, _resource
    from test_tensor_view import _view_fixture

    raw = _view_fixture()
    raw["imports"][0]["lends"].append(_resource("unused_barrier", "barrier", 1))
    candidate, _ = _pointer_plan(raw, get_profile("sm_103a"))
    assert candidate.status == "MODEL_SAT"
    assert candidate.provenance["resource_demands"]["scratch"] == 0
    assert candidate.provenance["resource_demands"]["unused_barrier"] == 0
    assert candidate.claims == ()


def test_fanout_ignores_unused_lend_at_sibling_storage_base():
    from test_tensor_link import _fanout_composition
    from test_link import _solution

    pipeline, stages, candidate, _ = _fanout_composition()
    resource = replace(_lend(stages, "scratch", "pipelink_shared"),
                       binding="fixed", location=ResourceLocation(None, 0))
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=tuple(
        replace(imp, lends=(resource,)) for imp in stages[1].tensor.imports)))
    plans = {connection.name: replace(candidate, provenance={
        "adapter_owner": connection.name, "resource_demands": {"scratch": demand}})
        for connection, demand in zip(pipeline.tensor_connections, (0, 32))}
    assert plan.check_fanout_storage(stages, pipeline.tensor_connections, plans, _solution()) == ()
    plans["acc"] = replace(candidate, provenance={
        "adapter_owner": "acc", "resource_demands": {"scratch": 16}})
    failures = plan.check_fanout_storage(stages, pipeline.tensor_connections, plans, _solution())
    assert failures[0].detail["intervals"] == [[0, 16], [0, 32]]
