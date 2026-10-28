from dataclasses import replace

import pytest

from pipec2.model import SchemaError, SolveError, StageRef
from pipec2.profile import get_profile
from pipec2.schema import load_stage_manifest, validate_assembly
from pipec2.solve import _candidate_values, solve
from test_solve import _regs_free_fixture, _stage


def _owner(values, *, roles=False, name="opaque.owner", claim_roles=None, group="launch"):
    raw = _stage(name, "compute", 4, [{
        "domain": "regs", "handle": "pipelink_launch_regs", "reg_group": group,
        "roles": [] if claim_roles is None else claim_roles, "values": values,
    }], [{"key": {}, "index": 0}])
    if not roles:
        raw["stage"]["roles"] = {}
    return raw


def _composition(*owners, pipeline_values=(128, 168, 192)):
    _, _, stages, pipeline = _regs_free_fixture(launch_values=list(pipeline_values))
    stages.extend(owners)
    pipeline = replace(pipeline, stages=[*pipeline.stages, *(StageRef(f"{owner.id}.plink", owner.id) for owner in owners)])
    return pipeline, stages


@pytest.mark.parametrize("roles", [False, True])
def test_any_stage_can_publish_role_free_launch_cap_domain(roles):
    owner = load_stage_manifest(_owner([95, 168, 255], roles=roles))
    assert owner.claims[0].roles == []
    assert owner.claims[0].values == [95, 168, 255]


def test_roleless_launch_constraint_sets_uniform_cap_without_register_groups():
    owner = load_stage_manifest(_owner([168]))
    pipeline, stages = _composition(owner)
    assert validate_assembly(pipeline, stages) == []
    solution = solve(pipeline, stages, {}, [], get_profile("sm_103a"))
    assert solution.launch.regs_cap == 168
    assert solution.reg_groups == []


def test_launch_domains_intersect_across_stages_and_pipeline():
    first = load_stage_manifest(_owner([128, 168]))
    second = load_stage_manifest(_owner([168, 192, 255], name="another.owner"))
    pipeline, stages = _composition(first, second, pipeline_values=(128, 168, 192))
    assert validate_assembly(pipeline, stages) == []
    _, values = _candidate_values(pipeline, stages, get_profile("sm_103a"))
    assert values["pipelink_launch_regs"] == (168,)


def test_disjoint_launch_domains_reject_with_register_clause():
    owner = load_stage_manifest(_owner([168]))
    pipeline, stages = _composition(owner, pipeline_values=(128, 192))
    failures = validate_assembly(pipeline, stages)
    assert {(failure.clause_id, failure.path) for failure in failures} == {
        ("L-REGS", "pipeline.handles.pipelink_launch_regs.values")}
    with pytest.raises(SolveError) as caught:
        _candidate_values(pipeline, stages, get_profile("sm_103a"))
    assert {(failure.clause_id, failure.path) for failure in caught.value.failures} == {
        ("L-REGS", "pipeline.handles.pipelink_launch_regs.values")}


@pytest.mark.parametrize("values,claim_roles,group", [
    ([256], [], "launch"), ([168], ["compute"], "launch"), ([168], [], "compute"),
])
def test_launch_constraint_rejects_invalid_domain_or_role_allocation(values, claim_roles, group):
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(_owner(values, roles=True, claim_roles=claim_roles, group=group))
    assert any(failure.clause_id == "L-REGS" and failure.path.startswith("$.claims[0]")
               for failure in caught.value.failures)


def test_nonlaunch_register_claim_still_requires_role_ownership():
    raw = _owner([168])
    raw["claims"][0].update(handle="pipelink_compute_regs", reg_group="compute")
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert any(failure.clause_id == "L-REGS" and failure.path == "$.claims[0].roles"
               for failure in caught.value.failures)
