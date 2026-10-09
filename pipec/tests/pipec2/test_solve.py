from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from typing import Any

import pytest

from pipec2.canonjson import canonical_dumps
from pipec2.model import SolveError
from pipec2.profile import get_profile
from pipec2.schema import load_pipeline, load_stage_manifest, validate_assembly
from pipec2.solve import solve


ZERO_HASH = "0" * 64


def _extent(const: int = 0, **per: int) -> dict[str, Any]:
    return {"const": const, "per": per}


def _claim(
    domain: str,
    handle: str,
    extent: dict[str, Any],
    *,
    align: int = 1,
    offset: int | None = None,
    lifetime: str = "kernel",
    roles: list[str] | None = None,
    reg_group: str | None = None,
    values: list[int] | None = None,
) -> dict[str, Any]:
    if domain == "regs":
        result: dict[str, Any] = {
            "domain": domain,
            "handle": handle,
            "roles": roles,
            "values": values,
        }
        if reg_group is not None:
            result["reg_group"] = reg_group
        return result
    result = {
        "domain": domain,
        "handle": handle,
        "extent": extent,
        "align": align,
    }
    if offset is not None:
        result["offset"] = offset
    if lifetime != "kernel":
        result["lifetime"] = lifetime
    if roles is not None:
        result["roles"] = roles
    return result


def _base_endpoint(handle: str, *, space: str = "smem") -> dict[str, Any]:
    decl = 'extern "C" void solver_fixture_call();'
    return {
        "name": "data",
        "direction": "produce",
        "owner_role": "load",
        "space": space,
        "scope": "cta",
        "proxy": "generic",
        "tile": {"dtype": "u32", "shape": [1]},
        "layout": {
            "encoding": "pl-layout-v2", "elem_bits": 32,
            **({"lanes": [0, 1], "cols": 1} if space == "tmem" else {"shape": [1], "stride": [1]}),
        },
        "protocol": {
            "kind": "call",
            "export": "noop",
            "signature": "void()",
            "decl": decl,
            "decl_sha256": hashlib.sha256(decl.encode("utf-8")).hexdigest(),
        },
        "binds": {"base": handle},
    }


def _stage(
    stage_id: str,
    role: str,
    warps: int,
    claims: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    *,
    compiled: list[int] | str = [1, 1, 1],
) -> dict[str, Any]:
    prefix = stage_id.replace(".", "_").replace("-", "_") + "_"
    return {
        "pipelink_abi": {
            "schema": "2.0-draft",
            "target": {"profile": "sm_103a", "lto_arch": "lto_103", "toolkit": "cuda-13.3.0"},
        },
        "stage": {
            "id": stage_id,
            "version": "1.0.0",
            "symbol_prefix": prefix,
            "entry_abi": "v2",
            "roles": {role: {"entry": f"{prefix}role_{role}", "warps": warps}},
            "exports": {},
            "entry_signature": "void(const void* params, unsigned variant)",
            "variants": variants,
        },
        "endpoints": [],
        "claims": claims,
        "topology": {
            "cta_group": {"fixed": 1},
            "cluster": {"compiled": compiled, "launch": {"x_multiple_of": 1, "max": [4, 4, 1]}},
            "multicast": [],
            "persistent": False,
            "scheduler": "static",
        },
        "params": {
            "bytes": 32,
            "align": 16,
            "type": f"{prefix}Params",
            "sha256_of_layout_probe": ZERO_HASH,
        },
    }


def _handle(cls: str, owner: str, *, domain: str | None = None, values: list[int] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"class": cls, "binding": "bind", "owner": owner}
    if domain is not None:
        result["domain"] = domain
    if values is not None:
        result["values"] = values
    return result


def _pipeline(
    stage_ids: list[str],
    handles: dict[str, dict[str, Any]],
    *,
    depth_priority: list[str],
    regs_priority: list[str],
    forced: dict[str, int] | None = None,
    phases: list[str] | None = None,
    exclusive: list[list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "pipelink_abi": {
            "schema": "2.0-draft",
            "target": {"profile": "sm_103a", "lto_arch": "lto_103", "toolkit": "cuda-13.3.0"},
        },
        "pipeline": {
            "name": "solver.fixture",
            "stages": [
                {"artifact_path": f"{stage_id}.pla", "stage_id": stage_id}
                for stage_id in stage_ids
            ],
            "handles": handles,
            "phases": phases or ["steady"],
            "exclusive": exclusive or [],
            "domains": {
                "gmem_counter": {"capacity": 16},
                "smem_page": {"capacity": 2, "page_bytes": 4096},
            },
            "depth_priority": depth_priority,
            "regs_priority": regs_priority,
            "forced_bindings": forced or {},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }


def _fixture(
    *,
    forced: dict[str, int] | None = None,
    phases: list[str] | None = None,
    exclusive: list[list[str]] | None = None,
    data_lifetimes: tuple[str, str] = ("kernel", "kernel"),
    data_domains: tuple[str, str] = ("smem", "smem"),
    data_extents: tuple[dict[str, Any], dict[str, Any]] | None = None,
    data_aligns: tuple[int, int] = (1, 1),
    role_warps: tuple[int, int] = (4, 4),
    warp_offsets: tuple[int, int] = (0, 4),
    reg_values: tuple[list[int], list[int]] = ([64], [128]),
    launch_values: list[int] = [96],
):
    data_extents = data_extents or (
        _extent(0, pipelink_depth_a=16),
        _extent(0, pipelink_depth_b=16),
    )
    variants = [
        {"key": {"pipelink_depth_a": a, "pipelink_depth_b": b}, "index": index}
        for index, (a, b) in enumerate(((2, 2), (2, 4), (4, 2), (4, 4)))
    ]
    alpha_claims = [
        _claim(
            data_domains[0],
            "pipelink_data_a",
            data_extents[0],
            align=data_aligns[0],
            lifetime=data_lifetimes[0],
        ),
        _claim("warp", "pipelink_warp_load", _extent(role_warps[0]), offset=warp_offsets[0], roles=["load"]),
        _claim(
            "regs", "pipelink_regs_load", {}, roles=["load"], reg_group="load_wg", values=reg_values[0]
        ),
    ]
    beta_claims = [
        _claim(
            data_domains[1],
            "pipelink_data_b",
            data_extents[1],
            align=data_aligns[1],
            lifetime=data_lifetimes[1],
        ),
        _claim("warp", "pipelink_warp_mma", _extent(role_warps[1]), offset=warp_offsets[1], roles=["mma"]),
        _claim(
            "regs", "pipelink_regs_mma", {}, roles=["mma"], reg_group="mma_wg", values=reg_values[1]
        ),
    ]
    raw_stages = [
        _stage("alpha.stage", "load", role_warps[0], alpha_claims, variants),
        _stage("beta.stage", "mma", role_warps[1], beta_claims, [{"key": {}, "index": 0}]),
    ]
    handles = {
        "pipelink_data_a": _handle("base", "claim:data_a", domain=data_domains[0]),
        "pipelink_data_b": _handle("base", "claim:data_b", domain=data_domains[1]),
        "pipelink_depth_a": _handle("depth", "channel:ring_a", values=[2, 4]),
        "pipelink_depth_b": _handle("depth", "channel:ring_b", values=[2, 4]),
        "pipelink_launch_regs": _handle("regs", "reg_group:launch", domain="regs", values=launch_values),
        "pipelink_regs_load": _handle("regs", "reg_group:load_wg", domain="regs", values=reg_values[0]),
        "pipelink_regs_mma": _handle("regs", "reg_group:mma_wg", domain="regs", values=reg_values[1]),
        "pipelink_warp_load": _handle("base", "claim:warp_load", domain="warp"),
        "pipelink_warp_mma": _handle("base", "claim:warp_mma", domain="warp"),
    }
    raw_pipeline = _pipeline(
        ["alpha.stage", "beta.stage"],
        handles,
        depth_priority=["pipelink_depth_a", "pipelink_depth_b"],
        regs_priority=["pipelink_regs_mma", "pipelink_regs_load", "pipelink_launch_regs"],
        forced=forced,
        phases=phases,
        exclusive=exclusive,
    )
    stages = [load_stage_manifest(raw) for raw in raw_stages]
    pipeline = load_pipeline(raw_pipeline)
    assert validate_assembly(pipeline, stages) == []
    return raw_stages, raw_pipeline, stages, pipeline


def _regs_free_fixture(*, launch_values: list[int] | None):
    raw_stages, raw_pipeline, _, _ = _fixture(
        role_warps=(9, 4),
        warp_offsets=(0, 9),
        launch_values=launch_values or [96],
    )
    for raw_stage in raw_stages:
        raw_stage["claims"] = [
            claim for claim in raw_stage["claims"] if claim["domain"] != "regs"
        ]
    handles = raw_pipeline["pipeline"]["handles"]
    raw_pipeline["pipeline"]["handles"] = {
        name: handle
        for name, handle in handles.items()
        if handle["class"] != "regs" or name == "pipelink_launch_regs"
    }
    if launch_values is None:
        raw_pipeline["pipeline"]["handles"].pop("pipelink_launch_regs")
        raw_pipeline["pipeline"]["regs_priority"] = []
    else:
        raw_pipeline["pipeline"]["regs_priority"] = ["pipelink_launch_regs"]
    stages = [load_stage_manifest(raw) for raw in raw_stages]
    pipeline = load_pipeline(raw_pipeline)
    assert validate_assembly(pipeline, stages) == []
    return raw_stages, raw_pipeline, stages, pipeline


def _small_smem_profile(capacity: int):
    profile = get_profile("sm_103a")
    domains = dict(profile.domains)
    domains["smem"] = replace(domains["smem"], capacity=capacity)
    return replace(profile, domains=domains)


def _clauses(exc: pytest.ExceptionInfo[SolveError]) -> set[str]:
    return {failure.clause_id for failure in exc.value.failures}


def test_joint_depth_solve_uses_lexicographic_priority_under_capacity_tradeoff() -> None:
    _, _, stages, pipeline = _fixture()
    solution = solve(pipeline, stages, {}, [], _small_smem_profile(96))

    assert solution.depths == {"pipelink_depth_a": 4, "pipelink_depth_b": 2}
    assert solution.variants["alpha.stage"] == {"pipelink_depth_a": 4, "pipelink_depth_b": 2}
    tiers = {tier.name: tier for tier in solution.objective}
    assert [name for name in tiers if name.startswith("depth:")] == ["depth:pipelink_depth_a", "depth:pipelink_depth_b"]
    assert tiers["high_water:smem"].value == solution.launch.dyn_smem_bytes <= 96
    assert tiers["high_water:smem"].status in {"proved", "bound_met"}


def _smem_disjoint(solution) -> bool:
    spans = sorted((item.lo, item.hi) for item in solution.intervals["smem"])
    return all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def test_high_water_tier_packs_below_greedy_name_order() -> None:
    # Lexical base minimization would place the small handle first and push the
    # large one past it; the high-water tier must find the tight packing.
    _, _, stages, pipeline = _fixture(data_extents=(_extent(8), _extent(64)))
    solution = solve(pipeline, stages, {}, [], _small_smem_profile(256))
    tiers = {tier.name: tier for tier in solution.objective}
    assert _smem_disjoint(solution)
    assert tiers["high_water:smem"].value == solution.launch.dyn_smem_bytes
    assert tiers["high_water:smem"].value >= tiers["high_water:smem"].bound


def test_mbarrier_scaled_into_smem_rejects_physical_collision() -> None:
    forced = {"pipelink_data_a": 0, "pipelink_data_b": 0}
    _, _, stages, pipeline = _fixture(
        forced=forced,
        data_domains=("smem", "mbarrier"),
        data_extents=(_extent(32), _extent(2)),
    )
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(64))
    assert _clauses(exc) == {"L-OVERLAP"}


@pytest.mark.parametrize(
    ("specimen", "domain", "extent", "capacity"),
    [
        ("F1-tmem-capacity", "tmem", 1024, None),
        ("F2-named-barrier-count", "named_barrier", 16, None),
        ("F2-smem-capacity", "smem", 262144, None),
        ("F3-smem-capacity", "smem", 262144, None),
        ("F4-opcode-capacity", "opcode", 1, 1),
    ],
)
def test_capacity_mutations_map_unsat_cores_to_overlap(
    specimen: str, domain: str, extent: int, capacity: int | None
) -> None:
    raw_stages, raw_pipeline, _, _ = _fixture(
        data_domains=("mbarrier", "smem"),
        data_extents=(_extent(2), _extent(16)),
    )
    # A late impossible resource used to force exhaustive backtracking over the
    # earlier 14,527-position barrier. Opcode needs two individually fitting claims.
    handle = "pipelink_z_resource"
    raw_pipeline["pipeline"]["handles"][handle] = _handle("base", "claim:late", domain=domain)
    raw_stages[0]["claims"].append(_claim(domain, handle, _extent(extent)))
    if specimen == "F3-smem-capacity":
        raw_stages[0]["claims"][-1]["offset"] = 1024
    if capacity is not None:
        raw_pipeline["pipeline"]["domains"][domain] = {"capacity": capacity}
        other = "pipelink_zz_resource"
        raw_pipeline["pipeline"]["handles"][other] = _handle("base", "claim:other", domain=domain)
        raw_stages[1]["claims"].append(_claim(domain, other, _extent(extent)))
    stages = [load_stage_manifest(raw) for raw in raw_stages]
    pipeline = load_pipeline(raw_pipeline)
    assert validate_assembly(pipeline, stages) == []
    started = time.monotonic()
    diagnostics = []
    for ordered in (stages, list(reversed(stages))):
        with pytest.raises(SolveError) as exc:
            solve(pipeline, ordered, {}, [], get_profile("sm_103a"))
        assert _clauses(exc) == {"L-OVERLAP"}
        assert exc.value.failures[0].path == f"solve.capacity.{domain}"
        core = exc.value.failures[0].detail["conflicting_set"]
        assert {"clause_id": "L-OVERLAP", "path": f"solve.capacity.{domain}"} in core
        diagnostics.append(canonical_dumps([failure.to_json_obj() for failure in exc.value.failures]))
    assert diagnostics[0] == diagnostics[1]
    assert time.monotonic() - started < 5


def test_sm120_smem_capacity_is_enforced_by_the_solver() -> None:
    raw_stages, raw_pipeline, _, _ = _fixture(
        data_domains=("mbarrier", "smem"),
        data_extents=(_extent(2), _extent(16)),
    )
    handle = "pipelink_z_resource"
    raw_pipeline["pipeline"]["handles"][handle] = _handle("base", "claim:late", domain="smem")
    raw_stages[0]["claims"].append(_claim("smem", handle, _extent(101376)))
    stages = [load_stage_manifest(raw) for raw in raw_stages]
    pipeline = load_pipeline(raw_pipeline)
    solve(pipeline, stages, {}, [], get_profile("sm_90a"))
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], get_profile("sm_120a"))
    assert _clauses(exc) == {"L-OVERLAP"}
    assert exc.value.failures[0].path == "solve.capacity.smem"


def test_scaled_views_compete_for_capacity_but_shared_aliases_do_not() -> None:
    _, _, stages, pipeline = _fixture(
        data_domains=("smem", "mbarrier"),
        data_extents=(_extent(32), _extent(2)),
    )
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(48))
    assert _clauses(exc) == {"L-OVERLAP"}
    # Both claims denote the same 32-byte physical interval via different units.
    aliased = [
        replace(stage, claims=[replace(stage.claims[0], alias="shared"), *stage.claims[1:]])
        for stage in stages
    ]
    solution = solve(pipeline, aliased, {}, [], _small_smem_profile(32))
    assert solution.native_bases["pipelink_data_a"] == solution.native_bases["pipelink_data_b"] == 0


def test_capacity_lower_bound_keeps_affine_and_table_choices_feasible() -> None:
    _, _, stages, pipeline = _fixture(
        data_extents=(_extent(0, pipelink_depth_a=16), _extent(16)),
    )
    table_claim = replace(
        stages[1].claims[0],
        extent=replace(stages[1].claims[0].extent, const=0, table={"pipelink_depth_b=2": 16, "pipelink_depth_b=4": 80}),
    )
    stages[1] = replace(stages[1], claims=[table_claim, *stages[1].claims[1:]])
    solution = solve(pipeline, stages, {}, [], _small_smem_profile(48))
    assert solution.depths == {"pipelink_depth_a": 2, "pipelink_depth_b": 2}
    assert _smem_disjoint(solution) and solution.launch.dyn_smem_bytes <= 48




def test_endpoint_base_claim_accepts_scaled_view_of_endpoint_space() -> None:
    raw_stages, raw_pipeline, _, _ = _fixture(
        data_domains=("smem_page", "smem"),
        data_extents=(_extent(1), _extent(16)),
    )
    raw_stages[0]["endpoints"] = [_base_endpoint("pipelink_data_a")]
    stages = [load_stage_manifest(raw) for raw in raw_stages]
    pipeline = load_pipeline(raw_pipeline)

    assert validate_assembly(pipeline, stages) == []
    solution = solve(pipeline, stages, {}, [], _small_smem_profile(8192))

    assert any(
        interval.handle == "pipelink_data_a"
        for interval in solution.intervals["smem"]
    )


def test_endpoint_base_claim_rejects_different_physical_domain() -> None:
    raw_stages, raw_pipeline, _, _ = _fixture(
        data_domains=("tmem", "smem"),
        data_extents=(_extent(32), _extent(16)),
    )
    raw_stages[0]["endpoints"] = [_base_endpoint("pipelink_data_a")]
    stages = [load_stage_manifest(raw) for raw in raw_stages]
    pipeline = load_pipeline(raw_pipeline)

    assert validate_assembly(pipeline, stages) == []
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(8192))

    assert _clauses(exc) == {"L-OWNER"}
    assert exc.value.failures[0].path == "stages.alpha.stage.endpoints.data.binds.base"
    assert exc.value.failures[0].detail == {
        "handle": "pipelink_data_a",
        "claim_domain": "tmem",
        "physical_domain": "tmem",
        "endpoint_space": "smem",
    }


def test_phase_exclusive_overlap_is_accepted_but_nonexclusive_is_rejected() -> None:
    forced = {"pipelink_data_a": 0, "pipelink_data_b": 0}
    args = dict(
        forced=forced,
        phases=["a", "b"],
        data_lifetimes=("phase:a", "phase:b"),
        data_extents=(_extent(32), _extent(32)),
    )
    _, _, stages, pipeline = _fixture(exclusive=[["a", "b"]], **args)
    solution = solve(pipeline, stages, {}, [], _small_smem_profile(32))
    assert solution.native_bases["pipelink_data_a"] == solution.native_bases["pipelink_data_b"] == 0

    _, _, stages, pipeline = _fixture(exclusive=[], **args)
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(32))
    assert _clauses(exc) == {"L-OVERLAP"}


def test_register_pool_and_warpgroup_rules_and_none_operation() -> None:
    _, _, stages, pipeline = _fixture(reg_values=([96], [96]), launch_values=[96])
    solution = solve(pipeline, stages, {}, [], _small_smem_profile(128))
    assert solution.regs == {"load_wg": 96, "mma_wg": 96}
    assert {group.op for group in solution.reg_groups} == {"none"}

    _, _, stages, pipeline = _fixture(reg_values=([128], [128]), launch_values=[64])
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(128))
    assert _clauses(exc) == {"L-HANDLE"}
    assert {item["clause_id"] for item in exc.value.failures[0].detail["conflicting_set"]} == {"L-HANDLE", "L-REGS"}

    _, _, stages, pipeline = _fixture(
        role_warps=(2, 2),
        warp_offsets=(0, 2),
        reg_values=([96], [96]),
        launch_values=[96],
    )
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(128))
    assert _clauses(exc) == {"L-OVERLAP"}
    assert "L-REGS" in {
        item["clause_id"] for item in exc.value.failures[0].detail["conflicting_set"]
    }


@pytest.mark.parametrize(
    ("launch_values", "expected_cap"),
    [([96], 96), (None, 24)],
)
def test_regs_free_pipeline_solves_with_empty_partition_and_correct_cap(
    launch_values: list[int] | None,
    expected_cap: int,
) -> None:
    _, _, stages, pipeline = _regs_free_fixture(launch_values=launch_values)

    solution = solve(pipeline, stages, {}, [], _small_smem_profile(128))

    assert solution.launch.threads == 13 * 32
    assert solution.launch.regs_cap == expected_cap
    assert solution.regs == {}
    assert solution.reg_groups == []


def test_regs_free_pipeline_rejects_unclaimed_non_launch_regs_handle() -> None:
    raw_stages, raw_pipeline, _, _ = _regs_free_fixture(launch_values=None)
    raw_pipeline["pipeline"]["handles"]["pipelink_regs_orphan"] = _handle(
        "regs",
        "reg_group:orphan",
        domain="regs",
        values=[64],
    )
    stages = [load_stage_manifest(raw) for raw in raw_stages]
    pipeline = load_pipeline(raw_pipeline)
    assert validate_assembly(pipeline, stages) == []

    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(128))

    assert _clauses(exc) == {"L-REGS"}


def test_register_claims_still_require_full_role_coverage() -> None:
    raw_stages, raw_pipeline, _, _ = _fixture(
        reg_values=([96], [96]),
        launch_values=[96],
    )
    raw_stages[1]["claims"] = [
        claim for claim in raw_stages[1]["claims"] if claim["domain"] != "regs"
    ]
    raw_pipeline["pipeline"]["handles"].pop("pipelink_regs_mma")
    raw_pipeline["pipeline"]["regs_priority"] = [
        "pipelink_regs_load",
        "pipelink_launch_regs",
    ]
    stages = [load_stage_manifest(raw) for raw in raw_stages]
    pipeline = load_pipeline(raw_pipeline)
    assert validate_assembly(pipeline, stages) == []

    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(128))

    assert _clauses(exc) == {"L-REGS"}


def test_forced_bindings_are_preserved_and_infeasible_forcing_rejects() -> None:
    _, _, stages, pipeline = _fixture(forced={"pipelink_data_b": 64})
    solution = solve(pipeline, stages, {}, [], _small_smem_profile(96))
    assert solution.native_bases["pipelink_data_b"] == 64

    _, _, stages, pipeline = _fixture(
        forced={"pipelink_data_a": 1},
        data_aligns=(16, 1),
    )
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(128))
    assert _clauses(exc) == {"L-ALIGN"}


def test_crt_merges_attached_alignment_congruences_and_rejects_contradictions() -> None:
    _, _, stages, pipeline = _fixture(
        data_extents=(_extent(4), _extent(4)),
        data_aligns=(4, 1),
    )
    beta_data = stages[1].claims[0]
    attached = replace(
        beta_data,
        handle="pipelink_data_a",
        extent=replace(beta_data.extent, const=4, per={}),
        align=8,
        offset=4,
    )
    stages_with_alignment = [
        stages[0],
        replace(stages[1], claims=[*stages[1].claims, attached]),
    ]
    solution = solve(pipeline, stages_with_alignment, {}, [], _small_smem_profile(64))
    assert solution.native_bases["pipelink_data_a"] == 4

    contradictory = replace(attached, align=4, offset=6)
    stages_with_contradiction = [
        stages[0],
        replace(stages[1], claims=[*stages[1].claims, contradictory]),
    ]
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages_with_contradiction, {}, [], _small_smem_profile(64))
    assert _clauses(exc) == {"L-ALIGN"}


def test_topology_rejects_compiled_shape_disagreement_and_preserves_dynamic_launch() -> None:
    _, _, stages, pipeline = _fixture()
    mismatched = [
        stages[0],
        replace(stages[1], topology=replace(stages[1].topology, cluster_compiled=(2, 1, 1))),
    ]
    with pytest.raises(SolveError) as exc:
        solve(pipeline, mismatched, {}, [], _small_smem_profile(128))
    assert _clauses(exc) == {"L-TOPO"}

    dynamic = [
        replace(stage, topology=replace(stage.topology, cluster_compiled="dynamic"))
        for stage in stages
    ]
    solution = solve(pipeline, dynamic, {}, [], _small_smem_profile(128))
    assert solution.launch.cluster == "dynamic"


def test_stage_claim_and_mapping_permutations_produce_identical_solution_json() -> None:
    raw_stages, raw_pipeline, stages, pipeline = _fixture()
    expected = solve(pipeline, stages, {}, [], _small_smem_profile(96))

    shuffled_stages_raw = []
    for raw in reversed(copy.deepcopy(raw_stages)):
        raw["claims"] = list(reversed(raw["claims"]))
        raw["stage"]["variants"] = list(reversed(raw["stage"]["variants"]))
        shuffled_stages_raw.append(raw)
    shuffled_pipeline_raw = copy.deepcopy(raw_pipeline)
    shuffled_pipeline_raw["pipeline"]["stages"] = list(reversed(shuffled_pipeline_raw["pipeline"]["stages"]))
    shuffled_pipeline_raw["pipeline"]["handles"] = dict(
        reversed(list(shuffled_pipeline_raw["pipeline"]["handles"].items()))
    )
    shuffled_stages = [load_stage_manifest(raw) for raw in shuffled_stages_raw]
    shuffled_pipeline = load_pipeline(shuffled_pipeline_raw)
    actual = solve(shuffled_pipeline, shuffled_stages, {}, [], _small_smem_profile(96))

    assert canonical_dumps(actual.to_json_obj()) == canonical_dumps(expected.to_json_obj())


def test_budget_exhaustion_after_feasibility_publishes_incumbent_with_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("pipec2.solve")
    _, _, stages, pipeline = _fixture()
    # Enough for the feasibility check, not for every tier's search.
    monkeypatch.setattr(module, "_SOLVE_RLIMIT", 5000)
    solution = solve(pipeline, stages, {}, [], _small_smem_profile(96))
    gaps = [tier for tier in solution.objective if tier.status == "budget_gap"]
    assert gaps
    for tier in gaps:
        maximize = tier.name.split(":", 1)[0] in {"depth", "regs"}
        assert (tier.value < tier.bound) if maximize else (tier.value > tier.bound)
    assert solution.rlimit_used <= 5000
    assert _smem_disjoint(solution) and solution.launch.dyn_smem_bytes <= 96


def test_budget_exhaustion_before_feasibility_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("pipec2.solve")
    _, _, stages, pipeline = _fixture()
    monkeypatch.setattr(module, "_SOLVE_RLIMIT", 1)
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(96))
    assert _clauses(exc) == {"L-BUDGET"}
    assert exc.value.failures[0].detail["phase"] == "feasibility"


def test_diagnostic_budget_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("pipec2.solve")
    _, _, stages, pipeline = _fixture(forced={"pipelink_data_a": 0, "pipelink_data_b": 0})
    monkeypatch.setattr(module, "_DIAGNOSTIC_RLIMIT", 1)
    with pytest.raises(SolveError) as exc:
        solve(pipeline, stages, {}, [], _small_smem_profile(96))
    assert _clauses(exc) == {"L-BUDGET"}
    assert exc.value.failures[0].detail["phase"] == "diagnostic"


@pytest.mark.parametrize("outcome", ["sat", "untied_launch", "unsat", "budget", "diagnostic_budget"])
def test_fresh_process_and_reordered_input_determinism(outcome: str) -> None:
    raw_stages, raw_pipeline, _, _ = _fixture(
        forced={"pipelink_data_a": 0, "pipelink_data_b": 0}
        if outcome in {"unsat", "diagnostic_budget"} else None,
        reg_values=([96], [96]) if outcome == "untied_launch" else ([64], [128]),
        launch_values=[96, 104, 112] if outcome == "untied_launch" else [96],
    )
    if outcome == "untied_launch":
        raw_pipeline["pipeline"]["handles"]["pipelink_launch_regs"]["binding"] = "launch"
        raw_pipeline["pipeline"]["regs_priority"].remove("pipelink_launch_regs")
        assert validate_assembly(
            load_pipeline(raw_pipeline), [load_stage_manifest(stage) for stage in raw_stages]
        ) == []
    runner = """
import importlib, json, sys
from dataclasses import replace
from pipec2.canonjson import canonical_dumps
from pipec2.model import SolveError
from pipec2.profile import get_profile
from pipec2.schema import load_pipeline, load_stage_manifest
m = importlib.import_module("pipec2.solve")
data = json.load(sys.stdin)
if data["outcome"] == "budget":
    m._SOLVE_RLIMIT = 5000
if data["outcome"] == "diagnostic_budget":
    m._DIAGNOSTIC_RLIMIT = 1
p = get_profile("sm_103a")
p = replace(p, domains={**p.domains, "smem": replace(p.domains["smem"], capacity=128 if data["outcome"] == "untied_launch" else 96)})
try:
    result = m.solve(load_pipeline(data["pipeline"]), [load_stage_manifest(s) for s in data["stages"]], {}, [], p)
    print(canonical_dumps({"solution": result.to_json_obj()}), end="")
except SolveError as error:
    print(canonical_dumps({"failures": [f.to_json_obj() for f in error.failures]}), end="")
"""
    outputs = []
    for seed, reorder in ((1, False), (7, False), (1, True), (7, True)):
        stages = copy.deepcopy(raw_stages)
        pipeline = copy.deepcopy(raw_pipeline)
        if reorder:
            stages.reverse()
            for stage in stages:
                stage["claims"].reverse()
                stage["stage"]["variants"].reverse()
                for variant in stage["stage"]["variants"]:
                    variant["key"] = dict(reversed(list(variant["key"].items())))
            pipeline["pipeline"]["stages"].reverse()
            pipeline["pipeline"]["handles"] = dict(reversed(list(pipeline["pipeline"]["handles"].items())))
            for handle in pipeline["pipeline"]["handles"].values():
                if "values" in handle:
                    handle["values"].reverse()
        completed = subprocess.run(
            [sys.executable, "-c", runner],
            input=json.dumps({"outcome": outcome, "stages": stages, "pipeline": pipeline}),
            text=True, capture_output=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        outputs.append(completed.stdout)
    assert outputs == [outputs[0]] * len(outputs)
    decoded = json.loads(outputs[0])
    if outcome == "sat":
        assert decoded["solution"]["depths"] == {"pipelink_depth_a": 4, "pipelink_depth_b": 2}
    elif outcome == "untied_launch":
        assert decoded["solution"]["launch"]["regs_cap"] == 96
        assert {group["op"] for group in decoded["solution"]["reg_groups"]} == {"none"}
    elif outcome == "budget":
        assert any(tier["status"] == "budget_gap" for tier in decoded["solution"]["objective"])
    else:
        assert {f["clause_id"] for f in decoded["failures"]} == {
            "L-OVERLAP" if outcome == "unsat" else "L-BUDGET"
        }
