from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from pipec2.decls import declaration_digest
from pipec2.model import (
    CLAUSE_IDS,
    CanonEndpoint,
    ClauseFailure,
    PipeLinkError,
    SymbolInfo,
    SymbolTable,
)
from pipec2.profile import get_profile
from pipec2.protocol import expand
from pipec2.schema import load_pipeline, load_stage_manifest, validate_assembly
from pipec2.solve import solve
from pipec2.link import _Artifact, _check_symbols

from test_tensor_contracts import _valid_tensor

FIXTURE_DIR = Path(__file__).with_name("fixtures") / "negative"
ZERO_HASH = "0" * 64




def _abi() -> dict[str, Any]:
    return {
        "schema": "2.0-draft",
        "target": {
            "profile": "sm_103a",
            "lto_arch": "lto_103",
            "toolkit": "cuda-13.3.0",
        },
    }


def _stage(
    stage_id: str,
    roles: dict[str, int],
    *,
    endpoints: list[dict[str, Any]] | None = None,
    claims: list[dict[str, Any]] | None = None,
    exports: dict[str, dict[str, str]] | None = None,
    variants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prefix = stage_id.replace(".", "_").replace("-", "_") + "_"
    return {
        "pipelink_abi": _abi(),
        "stage": {
            "id": stage_id,
            "version": "1.0.0",
            "symbol_prefix": prefix,
            "entry_abi": "v2",
            "roles": {
                role: {"entry": f"{prefix}role_{role}", "warps": warps}
                for role, warps in roles.items()
            },
            "exports": exports or {},
            "entry_signature": "void(const void* params, unsigned variant)",
            "variants": variants or [{"key": {}, "index": 0}],
        },
        "endpoints": endpoints or [],
        "claims": claims or [],
        "topology": {
            "cta_group": {"fixed": 1},
            "cluster": {
                "compiled": [1, 1, 1],
                "launch": {"x_multiple_of": 1, "max": [4, 4, 1]},
            },
            "multicast": [],
            "persistent": False,
            "scheduler": "static",
        },
        "params": {
            "bytes": 16,
            "align": 16,
            "type": f"{prefix}Params",
            "sha256_of_layout_probe": ZERO_HASH,
        },
    }


def _pipeline(
    stage_ids: list[str],
    handles: dict[str, dict[str, Any]],
    *,
    phases: list[str] | None = None,
    regs_priority: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "pipelink_abi": _abi(),
        "pipeline": {
            "name": "negative.fixture",
            "stages": [
                {"artifact_path": f"{stage_id}.pla", "stage_id": stage_id}
                for stage_id in stage_ids
            ],
            "handles": handles,
            "phases": phases or ["steady"],
            "exclusive": [],
            "domains": {
                "gmem_counter": {"capacity": 16},
                "smem_page": {"capacity": 2, "page_bytes": 4096},
            },
            "depth_priority": [
                name for name, handle in handles.items() if handle["class"] == "depth"
            ],
            "regs_priority": regs_priority or [],
            "forced_bindings": {},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }


def _mbarrier_protocol(full: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "mbarrier",
        "full": copy.deepcopy(full),
        "empty": {
            "slots": "pipelink_empty",
            "arrive": {"count": "threads(mma)", "by": "mma"},
            "tx_bytes": 0,
        },
        "init": {"owner": "load", "materializer": "dispatch", "phase": 0},
        "release": "once-per-acquire",
    }


def _barrier_endpoint(
    name: str,
    direction: str,
    role: str,
    full: dict[str, Any],
    *,
    shape: list[int] | None = None,
) -> dict[str, Any]:
    tile_shape = shape or [4, 4]
    return {
        "name": name,
        "direction": direction,
        "owner_role": role,
        "space": "smem",
        "scope": "cta",
        "proxy": "async",
        "tile": {"dtype": "bf16", "shape": tile_shape},
        "layout": {
            "encoding": "pl-layout-v2",
            "elem_bits": 16,
            "shape": [[tile_shape[0]], [tile_shape[1]]],
            "stride": [[tile_shape[1]], [1]],
        },
        "ring": {
            "depth": {"handle": "pipelink_depth"},
            "stage_stride_bytes": 32,
        },
        "protocol": _mbarrier_protocol(full),
        "binds": {"full": "pipelink_full", "empty": "pipelink_empty"},
    }


def _barrier_handles() -> dict[str, dict[str, Any]]:
    return {
        "pipelink_depth": {
            "class": "depth",
            "binding": "bind",
            "values": [2, 4],
            "owner": "channel:ring",
        },
        "pipelink_empty": {
            "class": "base",
            "binding": "bind",
            "domain": "mbarrier",
            "owner": "claim:empty",
        },
        "pipelink_full": {
            "class": "base",
            "binding": "bind",
            "domain": "mbarrier",
            "owner": "claim:full",
        },
    }


def _barrier_claims() -> list[dict[str, Any]]:
    return [
        {
            "domain": "mbarrier",
            "handle": "pipelink_full",
            "extent": {"const": 0, "per": {"pipelink_depth": 1}},
            "align": 1,
        },
        {
            "domain": "mbarrier",
            "handle": "pipelink_empty",
            "extent": {"const": 0, "per": {"pipelink_depth": 1}},
            "align": 1,
        },
    ]


def _barrier_base() -> dict[str, Any]:
    full = {
        "slots": "pipelink_full",
        "arrive": {"count": 1, "by": "load"},
        "tx_bytes": "tile",
    }
    producer = _barrier_endpoint("ring", "produce", "load", full)
    consumer = _barrier_endpoint("ring", "consume", "mma", full)
    return {
        "stages": [
            _stage("alpha.stage", {"load": 4}, endpoints=[producer], claims=_barrier_claims()),
            _stage("beta.stage", {"mma": 4}, endpoints=[consumer]),
        ],
        "pipeline": _pipeline(["alpha.stage", "beta.stage"], _barrier_handles()),
    }


def _double_barrier_base() -> dict[str, Any]:
    root = _barrier_base()
    full = {
        "slots": "pipelink_full",
        "arrive": {"count": 1, "by": "load"},
        "tx_bytes": "tile",
    }
    root["stages"][0]["endpoints"].append(
        _barrier_endpoint("other", "produce", "load", full)
    )
    root["stages"][1]["endpoints"].append(
        _barrier_endpoint("other", "consume", "mma", full)
    )
    return root


def _shared_completion_base() -> dict[str, Any]:
    channels = ("root", "member", "leaf")
    root_full = {
        "slots": "pipelink_full",
        "arrive": {"count": 1, "by": "load"},
        "tx_bytes": {"sum": list(channels)},
    }
    producer_endpoints = [
        _barrier_endpoint("root", "produce", "load", root_full),
        _barrier_endpoint(
            "member",
            "produce",
            "load",
            {"member_of": "root", "tx_bytes": "tile"},
            shape=[2, 2],
        ),
        _barrier_endpoint(
            "leaf",
            "produce",
            "load",
            {"member_of": "root", "tx_bytes": "tile"},
            shape=[2, 2],
        ),
    ]
    consumer_endpoints = [
        _barrier_endpoint("root", "consume", "mma", root_full),
        _barrier_endpoint(
            "member",
            "consume",
            "mma",
            {"member_of": "root", "tx_bytes": "tile"},
            shape=[2, 2],
        ),
        _barrier_endpoint(
            "leaf",
            "consume",
            "mma",
            {"member_of": "root", "tx_bytes": "tile"},
            shape=[2, 2],
        ),
    ]
    return {
        "stages": [
            _stage(
                "alpha.stage",
                {"load": 4},
                endpoints=producer_endpoints,
                claims=_barrier_claims(),
            ),
            _stage("beta.stage", {"mma": 4}, endpoints=consumer_endpoints),
        ],
        "pipeline": _pipeline(["alpha.stage", "beta.stage"], _barrier_handles()),
    }


def _call_base() -> dict[str, Any]:
    declaration = 'extern "C" int beta_stage_compute(int value);'
    # The caller declares the channel trampoline, never the callee's symbol.
    caller_declaration = 'extern "C" int pipelink_call_invoke(int value);'

    def endpoint(direction: str) -> dict[str, Any]:
        decl = caller_declaration if direction == "produce" else declaration
        return {
            "name": "invoke",
            "direction": direction,
            "owner_role": "load" if direction == "produce" else None,
            "space": "reg",
            "scope": "cta",
            "proxy": "generic",
            "tile": {"dtype": "s32", "shape": [128, 1]},
            "layout": {"encoding": "pl-layout-v2", "elem_bits": 32, "shape": [128, 1], "stride": [1, 1]},
            "protocol": {
                "kind": "call",
                "export": "compute",
                "signature": "int(int)",
                "decl": decl,
                "decl_sha256": declaration_digest(decl),
            },
            "binds": {},
        }

    return {
        "stages": [
            _stage("alpha.stage", {"load": 4}, endpoints=[endpoint("produce")]),
            _stage(
                "beta.stage",
                {"mma": 4},
                endpoints=[endpoint("consume")],
                exports={
                    "compute": {
                        "symbol": "beta_stage_compute",
                        "decl": declaration,
                        "decl_sha256": declaration_digest(declaration),
                    }
                },
            ),
        ],
        "pipeline": _pipeline(["alpha.stage", "beta.stage"], {}),
    }


def _warp_claim(roles: list[str], warps: int) -> dict[str, Any]:
    return {
        "domain": "warp",
        "handle": "pipelink_warps",
        "extent": {"const": warps, "per": {}},
        "align": 1,
        "offset": 0,
        "roles": roles,
    }


def _regs_claim(
    handle: str,
    roles: list[str],
    *,
    reg_group: str | None,
) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "domain": "regs",
        "handle": handle,
        "roles": roles,
        "values": [64],
    }
    if reg_group is not None:
        claim["reg_group"] = reg_group
    return claim


def _base_handle(domain: str, owner: str) -> dict[str, Any]:
    return {
        "class": "base",
        "binding": "bind",
        "domain": domain,
        "owner": owner,
    }


def _regs_handle(owner: str) -> dict[str, Any]:
    return {
        "class": "regs",
        "binding": "bind",
        "domain": "regs",
        "values": [64],
        "owner": f"reg_group:{owner}",
    }


def _register_base() -> dict[str, Any]:
    roles = {f"r{index}": 2 for index in range(4)}
    claims = [
        _warp_claim(list(roles), 8),
        _regs_claim("pipelink_regs_group_a", ["r0", "r1"], reg_group="group_a"),
        _regs_claim("pipelink_regs_group_b", ["r2", "r3"], reg_group="group_b"),
    ]
    handles = {
        "pipelink_warps": _base_handle("warp", "claim:warps"),
        "pipelink_regs_group_a": _regs_handle("group_a"),
        "pipelink_regs_group_b": _regs_handle("group_b"),
    }
    return {
        "stages": [_stage("gamma.stage", roles, claims=claims)],
        "pipeline": _pipeline(
            ["gamma.stage"],
            handles,
            regs_priority=["pipelink_regs_group_a", "pipelink_regs_group_b"],
        ),
    }


def _implicit_register_base() -> dict[str, Any]:
    roles = {"r0": 2, "r1": 2, "r2": 4}
    claims = [
        _warp_claim(list(roles), 8),
        _regs_claim("pipelink_regs_r0", ["r0", "r1"], reg_group="r0"),
        _regs_claim("pipelink_regs_r2", ["r2"], reg_group=None),
    ]
    handles = {
        "pipelink_warps": _base_handle("warp", "claim:warps"),
        "pipelink_regs_r0": _regs_handle("r0"),
        "pipelink_regs_r2": _regs_handle("r2"),
    }
    return {
        "stages": [_stage("gamma.stage", roles, claims=claims)],
        "pipeline": _pipeline(
            ["gamma.stage"],
            handles,
            regs_priority=["pipelink_regs_r0", "pipelink_regs_r2"],
        ),
    }


def _topology_base() -> dict[str, Any]:
    stages = [
        _stage(
            "alpha.stage",
            {"load": 4},
            claims=[
                {
                    "domain": "warp",
                    "handle": "pipelink_warp_load",
                    "extent": {"const": 4, "per": {}},
                    "align": 1,
                    "offset": 0,
                    "roles": ["load"],
                },
                _regs_claim("pipelink_regs_load", ["load"], reg_group="load_wg"),
            ],
        ),
        _stage(
            "beta.stage",
            {"mma": 4},
            claims=[
                {
                    "domain": "warp",
                    "handle": "pipelink_warp_mma",
                    "extent": {"const": 4, "per": {}},
                    "align": 1,
                    "offset": 4,
                    "roles": ["mma"],
                },
                _regs_claim("pipelink_regs_mma", ["mma"], reg_group="mma_wg"),
            ],
        ),
    ]
    handles = {
        "pipelink_warp_load": _base_handle("warp", "claim:warp_load"),
        "pipelink_warp_mma": _base_handle("warp", "claim:warp_mma"),
        "pipelink_regs_load": _regs_handle("load_wg"),
        "pipelink_regs_mma": _regs_handle("mma_wg"),
    }
    return {
        "stages": stages,
        "pipeline": _pipeline(
            ["alpha.stage", "beta.stage"],
            handles,
            regs_priority=["pipelink_regs_load", "pipelink_regs_mma"],
        ),
    }


def _schema_base() -> dict[str, Any]:
    variants = [
        {
            "key": {"pipelink_depth": 2, "pipelink_variant": 1},
            "index": 0,
        },
        {
            "key": {"pipelink_depth": 4, "pipelink_variant": 1},
            "index": 1,
        },
    ]
    return {"stages": [_stage("schema.stage", {"load": 4}, variants=variants)]}


def _tensor_base() -> dict[str, Any]:
    stage = _stage("tensor.stage", {"epilogue": 4})
    stage["pipelink_abi"]["schema"] = "2.2"
    stage["tensor"] = _valid_tensor()
    return {"stages": [stage]}


def _symbol_base() -> dict[str, Any]:
    return {
        "tables": {
            "alpha.stage": {"strong": ["alpha_stage_entry"]},
            "beta.stage": {"strong": ["beta_stage_entry"]},
        },
        "prefixes": {
            "alpha.stage": "alpha_stage_",
            "beta.stage": "beta_stage_",
        },
    }


BASES: dict[str, Callable[[], dict[str, Any]]] = {
    "barrier": _barrier_base,
    "double_barrier": _double_barrier_base,
    "shared_completion": _shared_completion_base,
    "call": _call_base,
    "register": _register_base,
    "implicit_register": _implicit_register_base,
    "topology": _topology_base,
    "schema": _schema_base,
    "symbols": _symbol_base,
    "tensor": _tensor_base,
}


def _set_path(root: Any, path: list[str | int], value: Any) -> None:
    target = root
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = copy.deepcopy(value)


def _load_case(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as stream:
        case = json.load(stream)
    raw = BASES[case["base"]]()
    mutation = case["mutation"]
    paths = mutation["paths"] if "paths" in mutation else [mutation["path"]]
    for path in paths:
        _set_path(raw, path, mutation["value"])
    return case, raw


def _load_assembly(raw: dict[str, Any]):
    stages = [load_stage_manifest(stage) for stage in raw["stages"]]
    pipeline = load_pipeline(raw["pipeline"])
    assert validate_assembly(pipeline, stages) == []
    return stages, pipeline


def _canon(stages: list[Any]) -> dict[str, CanonEndpoint]:
    dtype_bits = {"bf16": 16, "s32": 32}
    result: dict[str, CanonEndpoint] = {}
    for stage in stages:
        for endpoint in stage.endpoints:
            if endpoint.protocol.kind != "mbarrier":
                continue
            elem_bits = dtype_bits[endpoint.tile.dtype]
            footprint_bits = elem_bits * math.prod(endpoint.tile.shape)
            ring_geom = None
            if endpoint.ring is not None:
                depth = endpoint.ring.depth
                depth_key = depth.handle if depth.handle is not None else str(depth.fixed)
                ring_geom = (depth_key, endpoint.ring.stage_stride_bytes)
            result[f"{stage.id}:{endpoint.name}:{endpoint.direction}"] = CanonEndpoint(
                ("fixture", endpoint.name),
                elem_bits,
                None,
                8,
                footprint_bits,
                ring_geom,
                endpoint.space,
                endpoint.scope,
                endpoint.proxy,
                endpoint.tile.dtype,
                endpoint.tile.shape,
                None,
            )
    return result


def _symbol_table(raw: dict[str, Any]) -> SymbolTable:
    strong = tuple(raw.get("strong", []))
    weak = tuple(raw.get("weak", []))
    undefined = tuple(raw.get("undefined", []))
    info = {
        **{name: SymbolInfo("func", "strong") for name in strong},
        **{name: SymbolInfo("func", "weak") for name in weak},
        **{name: SymbolInfo("func", "undefined") for name in undefined},
    }
    return SymbolTable(strong, weak, undefined, info)


def _run_case(case: dict[str, Any], raw: dict[str, Any]) -> list[ClauseFailure]:
    phase = case["phase"]
    if phase == "symbols.check":
        artifacts = []
        for stage_id, table in raw["tables"].items():
            manifest = _stage(stage_id, {})
            manifest["symbols"] = _symbol_table(table).to_manifest()
            stage = replace(
                load_stage_manifest(manifest),
                symbol_prefix=raw["prefixes"][stage_id],
            )
            artifacts.append(_Artifact(Path(f"{stage_id}.plink"), stage, b"", ZERO_HASH))
        try:
            _check_symbols(artifacts)
        except PipeLinkError as error:
            return error.failures
        raise AssertionError(f"symbol fixture {case['name']!r} was not rejected")

    try:
        if phase == "schema.stage":
            load_stage_manifest(raw["stages"][case.get("stage_index", 0)])
        elif phase == "schema.pipeline":
            load_pipeline(raw["pipeline"])
        elif phase == "protocol.expand":
            stages, pipeline = _load_assembly(raw)
            channel = case["channel"]
            endpoints = [
                endpoint
                for stage in stages
                for endpoint in stage.endpoints
                if endpoint.name == channel
            ]
            producer = next(endpoint for endpoint in endpoints if endpoint.direction == "produce")
            consumer = next(endpoint for endpoint in endpoints if endpoint.direction == "consume")
            expand(
                channel,
                producer,
                consumer,
                stages,
                pipeline,
                _canon(stages),
                get_profile("sm_103a"),
            )
        elif phase == "solve":
            stages, pipeline = _load_assembly(raw)
            solve(pipeline, stages, {}, [], get_profile("sm_103a"))
        else:
            raise AssertionError(f"fixture names unsupported pre-link phase {phase!r}")
    except PipeLinkError as error:
        return error.failures
    raise AssertionError(f"fixture {case['name']!r} was not rejected")


def _assert_negative_fixture(name: str) -> None:
    case, raw = _load_case(name)
    expected_clause = case["expected_clause"]
    assert expected_clause in CLAUSE_IDS

    failures = _run_case(case, raw)

    assert {failure.clause_id for failure in failures} == {expected_clause}
    assert any(
        case["expected_path"] in failure.path
        for failure in failures
    ), [(failure.path, failure.message) for failure in failures]


def test_arrive_count_mismatch_fixture() -> None:
    """Paired arrival counts reject under L-PROTO."""
    _assert_negative_fixture("01_arrive_count_mismatch.json")


def test_wrong_expect_tx_sum_fixture() -> None:
    """A root expect-tx sum must name the exact shared group."""
    _assert_negative_fixture("02_wrong_expect_tx_sum.json")


def test_duplicate_init_materializer_fixture() -> None:
    """One physical barrier ring has one init materializer."""
    _assert_negative_fixture("03_duplicate_init_materializer.json")


def test_duplicate_shared_completion_root_fixture() -> None:
    """Duplicate shared-completion root names reject."""
    _assert_negative_fixture("04_duplicate_shared_completion_root.json")


def test_chained_shared_completion_root_fixture() -> None:
    """member_of may not target another member."""
    _assert_negative_fixture("05_chained_shared_completion_root.json")


def test_mismatched_call_declaration_hash_fixture() -> None:
    """Call declaration hashes bind normalized declaration text."""
    _assert_negative_fixture("06_mismatched_call_declaration_hash.json")


def test_mixed_cta_group_fixture() -> None:
    """Stages must agree on their compiled CTA group."""
    _assert_negative_fixture("07_mixed_cta_group.json")


def test_incompatible_compiled_cluster_shapes_fixture() -> None:
    """Fixed compiled cluster shapes must intersect."""
    _assert_negative_fixture("08_incompatible_compiled_cluster_shapes.json")


def test_duplicate_strong_entry_symbol_fixture() -> None:
    """Strong entry/export symbols are unique across artifacts."""
    _assert_negative_fixture("09_duplicate_strong_entry_symbol.json")


def test_incomplete_reg_group_fixture() -> None:
    """An explicit register group covers its whole warpgroup."""
    _assert_negative_fixture("10_incomplete_reg_group.json")


def test_warpgroup_straddling_reg_group_fixture() -> None:
    """An explicit register group cannot straddle boundaries."""
    _assert_negative_fixture("11_warpgroup_straddling_reg_group.json")


def test_non_warpgroup_aligned_regs_claim_roles_fixture() -> None:
    """D-finding: an implicit one-role regs claim is warpgroup-aligned."""
    _assert_negative_fixture("12_non_warpgroup_aligned_regs_claim_roles.json")


def test_unknown_phase_pair_fixture() -> None:
    """D-finding: every exclusive pair names declared finite phases."""
    _assert_negative_fixture("13_unknown_phase_pair.json")


def test_incomplete_variant_key_tuple_fixture() -> None:
    """D-finding: all variant rows carry one complete handle-key tuple."""
    _assert_negative_fixture("14_incomplete_variant_key_tuple.json")


def test_params_overflow_fixture() -> None:
    """D-finding: a stage parameter blob cannot exceed the profile cap."""
    _assert_negative_fixture("15_params_overflow.json")


@pytest.mark.parametrize("name", [
    "16_unknown_tensor_guarantee.json",
    "17_duplicate_lend_name.json",
    "18_relocatable_lend_without_handle.json",
    "19_tensor_under_draft_schema.json",
])
def test_tensor_v2_negative_fixture(name: str) -> None:
    _assert_negative_fixture(name)
