from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pipec2.canonjson import canonical_bytes
from pipec2.dispatchgen import generate_dispatch, params_layout
from pipec2.link import _link_result
from pipec2.model import SchemaError, SolveError
from pipec2.profile import get_profile
from pipec2.schema import load_pipeline, load_stage_manifest, validate_assembly
from pipec2.solve import solve


ZERO_HASH = "0" * 64
TARGET = {
    "schema": "2.0-draft",
    "target": {
        "profile": "sm_103a",
        "lto_arch": "lto_103",
        "toolkit": "cuda-13.3.0",
    },
}


def _claim(
    domain: str,
    handle: str,
    extent: int,
    *,
    lifetime: str = "kernel",
) -> dict[str, Any]:
    return {
        "domain": domain,
        "handle": handle,
        "extent": {"const": extent, "per": {}},
        "align": 1,
        "lifetime": lifetime,
    }


def _stage(
    stage_id: str,
    claims: list[dict[str, Any]],
    *,
    launched: bool = False,
    params_bytes: int | None = None,
) -> dict[str, Any]:
    prefix = stage_id.replace(".", "_").replace("-", "_") + "_"
    roles = (
        {"worker": {"entry": f"{prefix}role_worker", "warps": 4}}
        if launched
        else {}
    )
    stage_claims = list(claims)
    if launched:
        stage_claims.extend(
            [
                {
                    "domain": "warp",
                    "handle": "pipelink_warp_worker",
                    "extent": {"const": 4, "per": {}},
                    "align": 1,
                    "absolute_base": 0,
                    "roles": ["worker"],
                },
                {
                    "domain": "regs",
                    "handle": "pipelink_regs_worker",
                    "roles": ["worker"],
                    "reg_group": "worker_wg",
                    "values": [64],
                },
            ]
        )
    byte_count = (16 if launched else 0) if params_bytes is None else params_bytes
    params: dict[str, Any] = {
        "bytes": byte_count,
        "align": 16,
        "type": "fixture_params" if byte_count else "none",
        "sha256_of_layout_probe": ZERO_HASH,
    }
    return {
        "pipelink_abi": copy.deepcopy(TARGET),
        "stage": {
            "id": stage_id,
            "version": "1.0.0",
            "symbol_prefix": prefix,
            "entry_abi": "v2",
            "roles": roles,
            "exports": {},
            "entry_signature": "void(const void* params, unsigned variant)",
            "variants": [{"key": {}, "index": 0}],
        },
        "endpoints": [],
        "claims": stage_claims,
        "topology": {
            "cta_group": {"fixed": 1},
            "cluster": {
                "compiled": [1, 1, 1],
                "launch": {"x_multiple_of": 1, "max": [1, 1, 1]},
            },
            "multicast": [],
            "persistent": False,
            "scheduler": "static",
        },
        "params": params,
    }


def _pipeline(
    stage_ids: list[str],
    resource_domain: str,
    resource_handles: list[str],
    domains: dict[str, dict[str, int]],
    *,
    phases: list[str] | None = None,
    exclusive: list[list[str]] | None = None,
    forced: dict[str, int] | None = None,
) -> dict[str, Any]:
    handles: dict[str, dict[str, Any]] = {
        "pipelink_warp_worker": {
            "class": "base",
            "binding": "bind",
            "domain": "warp",
            "owner": "claim:warp_worker",
        },
        "pipelink_regs_worker": {
            "class": "regs",
            "binding": "bind",
            "domain": "regs",
            "values": [64],
            "owner": "reg_group:worker_wg",
        },
        "pipelink_launch_regs": {
            "class": "regs",
            "binding": "bind",
            "domain": "regs",
            "values": [64],
            "owner": "reg_group:launch",
        },
    }
    for handle in resource_handles:
        handles[handle] = {
            "class": "base",
            "binding": "bind",
            "domain": resource_domain,
            "owner": f"claim:{handle.removeprefix('pipelink_')}",
        }
    return {
        "pipelink_abi": copy.deepcopy(TARGET),
        "pipeline": {
            "name": "org.fixture.mk",
            "stages": [
                {"artifact_path": f"{stage_id}.plink", "stage_id": stage_id}
                for stage_id in stage_ids
            ],
            "handles": handles,
            "phases": phases or ["steady"],
            "exclusive": exclusive or [],
            "domains": domains,
            "depth_priority": [],
            "regs_priority": ["pipelink_regs_worker", "pipelink_launch_regs"],
            "forced_bindings": forced or {},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }


def _assembly(
    domain: str,
    extents: list[int],
    lifetimes: list[str],
    *,
    capacity: int,
    phases: list[str] | None = None,
    exclusive: list[list[str]] | None = None,
    forced: dict[str, int] | None = None,
    declare_domain: bool = True,
):
    interpreter_id = "org.fixture.interpreter"
    op_ids = [f"org.fixture.op.{chr(ord('a') + index)}" for index in range(len(extents))]
    resource_handles = [f"pipelink_resource_{index}" for index in range(len(extents))]
    raw_stages = [_stage(interpreter_id, [], launched=True)]
    raw_stages.extend(
        _stage(stage_id, [_claim(domain, handle, extent, lifetime=lifetime)])
        for stage_id, handle, extent, lifetime in zip(
            op_ids, resource_handles, extents, lifetimes, strict=True
        )
    )
    raw_pipeline = _pipeline(
        [interpreter_id, *op_ids],
        domain,
        resource_handles,
        {domain: {"capacity": capacity}} if declare_domain else {},
        phases=phases,
        exclusive=exclusive,
        forced=forced,
    )
    return load_pipeline(raw_pipeline), [load_stage_manifest(raw) for raw in raw_stages]


def _solve(pipeline, stages):
    failures = validate_assembly(pipeline, stages)
    assert failures == []
    return solve(pipeline, stages, {}, [], get_profile("sm_103a"))


def test_phase_exclusive_instruction_semaphores_reuse_pipeline_capacity() -> None:
    pipeline, stages = _assembly(
        "instr_semaphore",
        [15, 14],
        ["phase:instr_a", "phase:instr_b"],
        capacity=32,
        phases=["instr_a", "instr_b"],
        exclusive=[["instr_*", "instr_*"]],
    )

    solution = _solve(pipeline, stages)

    assert solution.native_bases["pipelink_resource_0"] == 0
    assert solution.native_bases["pipelink_resource_1"] == 0
    assert [(item.lo, item.hi, item.phase) for item in solution.intervals["instr_semaphore"]] == [
        (0, 15, "instr_a"),
        (0, 14, "instr_b"),
    ]


def test_concurrent_instruction_semaphores_over_capacity_reject() -> None:
    pipeline, stages = _assembly(
        "instr_semaphore",
        [15, 18],
        ["phase:instr_a", "phase:instr_a"],
        capacity=32,
        phases=["instr_a"],
    )
    assert validate_assembly(pipeline, stages) == []

    with pytest.raises(SolveError) as caught:
        solve(pipeline, stages, {}, [], get_profile("sm_103a"))

    assert [failure.clause_id for failure in caught.value.failures] == ["L-OVERLAP"]


def test_kernel_lifetime_opcodes_bind_distinctly_and_forced_duplicate_rejects() -> None:
    pipeline, stages = _assembly("opcode", [1, 1], ["kernel", "kernel"], capacity=16)
    solution = _solve(pipeline, stages)
    assert len(
        {
            solution.native_bases["pipelink_resource_0"],
            solution.native_bases["pipelink_resource_1"],
        }
    ) == 2

    forced_pipeline, forced_stages = _assembly(
        "opcode",
        [1, 1],
        ["kernel", "kernel"],
        capacity=16,
        forced={"pipelink_resource_0": 3, "pipelink_resource_1": 3},
    )
    assert validate_assembly(forced_pipeline, forced_stages) == []
    with pytest.raises(SolveError) as caught:
        solve(forced_pipeline, forced_stages, {}, [], get_profile("sm_103a"))
    assert [failure.clause_id for failure in caught.value.failures] == ["L-OVERLAP"]


def test_pipeline_domain_declarations_are_optional_profile_subset() -> None:
    raw = _pipeline(
        ["org.fixture.interpreter"],
        "opcode",
        [],
        {"opcode": {"capacity": 16}},
    )
    pipeline = load_pipeline(raw)
    assert set(pipeline.domains) == {"opcode"}

    raw["pipeline"]["domains"]["unknown"] = {"capacity": 1}
    with pytest.raises(SchemaError) as caught:
        load_pipeline(raw)
    assert [failure.clause_id for failure in caught.value.failures] == ["L-HANDLE"]
    assert "subset" in caught.value.failures[0].message


def test_undeclared_pipeline_capacity_claim_rejects_before_solve() -> None:
    pipeline, stages = _assembly(
        "instr_page",
        [1],
        ["phase:instr_a"],
        capacity=13,
        phases=["instr_a"],
        declare_domain=False,
    )

    failures = validate_assembly(pipeline, stages)

    assert [failure.clause_id for failure in failures] == ["L-HANDLE"]
    assert failures[0].path.endswith(".claims.instr_page.pipelink_resource_0.domain")
    assert "omitted from pipeline.domains" in failures[0].message


def test_zero_byte_params_require_callable_only_stage() -> None:
    loaded = load_stage_manifest(_stage("org.fixture.op.attn-partial", [], params_bytes=0))
    assert loaded.roles == {}
    assert loaded.params.bytes == 0

    with pytest.raises(SchemaError) as role_error:
        load_stage_manifest(
            _stage("org.fixture.interpreter", [], launched=True, params_bytes=0)
        )
    assert [failure.clause_id for failure in role_error.value.failures] == ["L-PARAMS"]
    assert "no roles" in role_error.value.failures[0].message



def test_dispatch_and_link_publication_keep_zero_byte_stage_total() -> None:
    pipeline, stages = _assembly("opcode", [1], ["kernel"], capacity=16)
    solution = _solve(pipeline, stages)
    layout = params_layout(stages, get_profile("sm_103a"))

    assert layout.entries == [
        ("org.fixture.interpreter", 0, 16, 16),
        ("org.fixture.op.a", 64, 0, 16),
    ]
    source = generate_dispatch(solution, pipeline, stages, [], layout)
    assert "stage_0000_org_fixture_interpreter[16]" in source
    assert "stage_0001_org_fixture_op_a" not in source

    result = _link_result(
        pipeline,
        stages,
        [],
        solution,
        layout,
        [],
        "pipelink_kernel",
        (64, 0, 0),
        {"dispatch.cu": source},
        None,
    )
    link_json = json.loads(canonical_bytes(result.to_json_obj()))
    assert ["org.fixture.op.a", 64, 0, 16] in link_json["params"]["entries"]




def test_existing_example_pipeline_manifests_still_load() -> None:
    repo = Path(__file__).parents[3]
    expected = {
        "examples/profile-consumer/manifests/pipeline.json": {"gmem_counter", "smem_page"},
    }
    for relative, domains in expected.items():
        pipeline = load_pipeline(json.loads((repo / relative).read_text(encoding="utf-8")))
        assert set(pipeline.domains) == domains, relative
