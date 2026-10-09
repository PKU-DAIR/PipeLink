from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from collections.abc import Callable
from typing import Any

import pytest

from pipec2.canonjson import canonical_dumps
from pipec2.model import Pipeline, SchemaError, StageManifest, TensorLifetime
from pipec2.profile import get_profile
from pipec2.schema import load_pipeline, load_stage_manifest, validate_assembly
from test_tensor_contracts import _valid_tensor


ZERO_HASH = "0" * 64


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _affine(const: int, **per: int) -> dict[str, Any]:
    return {"const": const, "per": per}


def _valid_stage() -> dict[str, Any]:
    decl = 'extern "C" int producer_stage_register_compute(int x);'
    layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 16,
        "shape": [[8], [8]],
        "stride": [[8], [1]],
    }
    return {
        "pipelink_abi": {
            "schema": "2.0-draft",
            "target": {
                "profile": "sm_103a",
                "lto_arch": "lto_103",
                "toolkit": "cuda-13.3.0",
            },
        },
        "stage": {
            "id": "producer.stage",
            "version": "1.0.0",
            "symbol_prefix": "producer_stage_",
            "entry_abi": "v2",
            "roles": {"load": {"entry": "producer_stage_role_load", "warps": 4}},
            "exports": {
                "register_compute": {
                    "symbol": "producer_stage_register_compute",
                    "decl": decl,
                    "decl_sha256": _digest(decl),
                }
            },
            "entry_signature": "void(const void* params, unsigned variant)",
            "variants": [
                {
                    "key": {"pipelink_variant": 1},
                    "index": 0,
                    "entry": "producer_stage_variant_0",
                }
            ],
            "descriptor": {"bytes": 128, "align": 16},
        },
        "endpoints": [
            {
                "name": "ab_ring",
                "direction": "produce",
                "owner_role": "load",
                "space": "smem",
                "scope": "cta",
                "proxy": "async",
                "tile": {"dtype": "bf16", "shape": [8, 8]},
                "layout": layout,
                "ring": {
                    "depth": {"handle": "pipelink_ab_depth"},
                    "stage_stride_bytes": 128,
                },
                "protocol": {
                    "kind": "mbarrier",
                    "full": {
                        "slots": "pipelink_ab_full_slots",
                        "arrive": {"count": 1, "by": "load"},
                        "tx_bytes": {"sum": ["ab_ring"]},
                    },
                    "empty": {
                        "slots": "pipelink_ab_empty_slots",
                        "arrive": {"count": "threads(load)", "by": "load"},
                        "tx_bytes": 0,
                    },
                    "init": {"owner": "load", "materializer": "dispatch", "phase": 0},
                    "release": "once-per-acquire",
                },
                "binds": {
                    "base": "pipelink_ab_smem_base",
                    "full": "pipelink_ab_full_slots",
                    "empty": "pipelink_ab_empty_slots",
                },
            },
            {
                "name": "register_call",
                "direction": "produce",
                "owner_role": "load",
                "space": "reg",
                "scope": "cta",
                "proxy": "generic",
                "tile": {"dtype": "s32", "shape": [128, 1]},
                "layout": {
                    "encoding": "pl-layout-v2",
                    "elem_bits": 32,
                    "shape": [[128], [1]],
                    "stride": [[1], [1]],
                },
                "protocol": {
                    "kind": "call",
                    "export": "register_compute",
                    "signature": "int(int)",
                    "decl": decl,
                    "decl_sha256": _digest(decl),
                },
                "binds": {},
            },
            {
                "name": "named_sync",
                "direction": "produce",
                "owner_role": "load",
                "space": "smem",
                "scope": "cta",
                "proxy": "generic",
                "tile": {"dtype": "u32", "shape": [1]},
                "layout": {"encoding": "pl-layout-v2", "elem_bits": 32, "shape": [1], "stride": [1]},
                "protocol": {"kind": "named_barrier", "id": "pipelink_named_id", "count": 128},
                "binds": {},
            },
            {
                "name": "grid_counter",
                "direction": "produce",
                "owner_role": "load",
                "space": "gmem",
                "scope": "grid",
                "proxy": "generic",
                "tile": {"dtype": "u32", "shape": [1]},
                "layout": {"encoding": "pl-layout-v2", "elem_bits": 32, "shape": [1], "stride": [1]},
                "protocol": {
                    "kind": "counter",
                    "slot": "pipelink_counter_slot",
                    "release": "red.release.gpu",
                    "acquire": "ld.acquire.gpu",
                },
                "binds": {},
            },
        ],
        "claims": [
            {
                "domain": "smem",
                "handle": "pipelink_ab_smem_base",
                "lifetime": "phase:steady",
                "extent": _affine(0, pipelink_ab_depth=128),
                "align": 128,
                "offset": 0,
            },
            {
                "domain": "mbarrier",
                "handle": "pipelink_ab_full_slots",
                "extent": _affine(0, pipelink_ab_depth=1),
                "align": 1,
            },
            {
                "domain": "mbarrier",
                "handle": "pipelink_ab_empty_slots",
                "extent": _affine(0, pipelink_ab_depth=1),
                "align": 1,
            },
            {
                "domain": "regs",
                "handle": "pipelink_regs_load",
                "roles": ["load"],
                "reg_group": "load_wg",
                "values": [64, 72],
            },
            {
                "domain": "warp",
                "handle": "pipelink_warp_load",
                "extent": _affine(4),
                "align": 1,
            },
            {
                "domain": "named_barrier",
                "handle": "pipelink_named_id",
                "extent": _affine(1),
                "align": 1,
            },
            {
                "domain": "gmem_counter",
                "handle": "pipelink_counter_slot",
                "extent": _affine(1),
                "align": 1,
            },
            {
                "domain": "smem_page",
                "handle": "pipelink_page_slot",
                "extent": {
                    "table": {
                        "pipelink_ab_depth=2,pipelink_variant=1": 1,
                        "pipelink_ab_depth=4,pipelink_variant=1": 2,
                    }
                },
                "align": 1,
            },
        ],
        "topology": {
            "cta_group": {"fixed": 2},
            "cluster": {
                "compiled": [2, 1, 1],
                "launch": {"x_multiple_of": 2, "max": [4, 4, 1]},
            },
            "multicast": ["ab_ring"],
            "persistent": True,
            "scheduler": "static",
        },
        "params": {
            "bytes": 64,
            "align": 64,
            "type": "ExampleParams",
            "sha256_of_layout_probe": ZERO_HASH,
        },
        "hints": {"bytes_per_tile": 128},
        "payload": {"container_owned": True},
    }

@pytest.mark.parametrize(
    "section",
    [
        {"mechanism": "probe"},
        {"mechanism": "probe", "defined_strong": ["duplicate", "duplicate"], "defined_weak": [], "undefined": {}},
        {"mechanism": "probe", "defined_strong": ["overlap"], "defined_weak": [], "undefined": {"overlap": "function"}},
        {"mechanism": "probe", "defined_strong": [], "defined_weak": [], "undefined": {"bad_kind": "unknown"}},
    ],
)
def test_manifest_rejects_malformed_recorded_symbol_sets(section: dict[str, Any]) -> None:
    manifest = _valid_stage()
    manifest["symbols"] = section
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(manifest)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L-SYM", "$.symbols"),
    ]



def _valid_pipeline() -> dict[str, Any]:
    return {
        "pipelink_abi": {
            "schema": "2.0-draft",
            "target": {
                "profile": "sm_103a",
                "lto_arch": "lto_103",
                "toolkit": "cuda-13.3.0",
            },
        },
        "pipeline": {
            "name": "miniature",
            "stages": [{"artifact_path": "producer.pla", "stage_id": "producer.stage"}],
            "handles": {
                "pipelink_ab_depth": {
                    "class": "depth",
                    "binding": "bind",
                    "values": [2, 4],
                    "owner": "channel:ab_ring",
                },
                "pipelink_ab_empty_slots": {
                    "class": "base",
                    "binding": "bind",
                    "domain": "mbarrier",
                    "owner": "claim:ab_empty",
                },
                "pipelink_ab_full_slots": {
                    "class": "base",
                    "binding": "bind",
                    "domain": "mbarrier",
                    "owner": "claim:ab_full",
                },
                "pipelink_ab_smem_base": {
                    "class": "base",
                    "binding": "bind",
                    "domain": "smem",
                    "owner": "claim:ab_ring",
                },
                "pipelink_regs_load": {
                    "class": "regs",
                    "binding": "bind",
                    "domain": "regs",
                    "values": [64, 72],
                    "owner": "reg_group:load_wg",
                },
                "pipelink_variant": {
                    "class": "variant",
                    "binding": "fixed",
                    "values": [1],
                    "owner": "stage:producer.stage",
                },
                "pipelink_warp_load": {
                    "class": "base",
                    "binding": "bind",
                    "domain": "warp",
                    "owner": "claim:warp_load",
                },
                "pipelink_named_id": {
                    "class": "base",
                    "binding": "bind",
                    "domain": "named_barrier",
                    "owner": "claim:named_sync",
                },
                "pipelink_counter_slot": {
                    "class": "base",
                    "binding": "bind",
                    "domain": "gmem_counter",
                    "owner": "claim:grid_counter",
                },
                "pipelink_page_slot": {
                    "class": "base",
                    "binding": "bind",
                    "domain": "smem_page",
                    "owner": "claim:page_slot",
                },
            },
            "phases": ["setup", "steady", "instr_gemm0", "instr_gemm1"],
            "exclusive": [["steady", "setup"], ["instr_*", "instr_*"]],
            "domains": {
                "gmem_counter": {"capacity": 16},
                "smem_page": {"capacity": 2, "page_bytes": 4096},
            },
            "depth_priority": ["pipelink_ab_depth"],
            "regs_priority": ["pipelink_regs_load"],
            "forced_bindings": {"pipelink_variant": 1},
            "topology": {"persistent": True, "scheduler": "static"},
        },
    }


def _assert_clause(exc: pytest.ExceptionInfo[SchemaError], clause_id: str) -> None:
    assert exc.value.failures
    assert {failure.clause_id for failure in exc.value.failures} == {clause_id}
    assert [(failure.clause_id, failure.path) for failure in exc.value.failures] == sorted(
        (failure.clause_id, failure.path) for failure in exc.value.failures
    )


@pytest.mark.parametrize("version", ["2.0-draft", "2.2", "3.0"])
def test_valid_stage_and_pipeline_load_model_round_trip_and_assemble(version: str) -> None:
    raw_stage = _valid_stage()
    raw_stage["pipelink_abi"]["schema"] = version
    stage = load_stage_manifest(raw_stage)
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipelink_abi"]["schema"] = version
    pipeline = load_pipeline(raw_pipeline)

    assert isinstance(stage, StageManifest)
    assert isinstance(pipeline, Pipeline)
    assert stage.target.schema == pipeline.target.schema == version
    assert stage.endpoints[0].layout.to_json_obj() == raw_stage["endpoints"][0]["layout"]
    assert stage.variants[0].entry == "producer_stage_variant_0"
    assert pipeline.exclusive == [("instr_gemm0", "instr_gemm1"), ("setup", "steady")]
    assert validate_assembly(pipeline, [stage]) == []
    stage_obj = stage.to_json_obj()
    pipeline_obj = pipeline.to_json_obj()
    assert canonical_dumps(StageManifest.from_json_obj(stage_obj).to_json_obj()) == canonical_dumps(stage_obj)
    assert canonical_dumps(Pipeline.from_json_obj(pipeline_obj).to_json_obj()) == canonical_dumps(pipeline_obj)


def test_provided_call_rejects_non_null_owner_role() -> None:
    raw = _valid_stage()
    raw["endpoints"][1].update(direction="consume", owner_role="load")
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L-PAIR", "$.endpoints[1].owner_role")
    ]


def test_roleless_provided_call_round_trips_without_provider_roles() -> None:
    raw = _valid_stage()
    endpoint = raw["endpoints"][1]
    endpoint.update(direction="consume", owner_role=None)
    raw["endpoints"] = [endpoint]
    raw["stage"]["roles"] = {}
    raw["claims"] = []
    stage = load_stage_manifest(raw)
    assert stage.endpoints[0].owner_role is None
    assert stage.to_json_obj()["endpoints"][0]["owner_role"] is None
    assert StageManifest.from_json_obj(stage.to_json_obj()).endpoints == stage.endpoints


def test_malformed_layout_tree_fails_during_stage_loading() -> None:
    raw = _valid_stage()
    raw["endpoints"][0]["layout"]["shape"] = [[8], []]
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L8", "$.endpoints[0].layout.shape[1]")
    ]


def test_provenance_survives_model_round_trip_without_affecting_assembly() -> None:
    raw = _valid_stage()
    raw["provenance"] = {
        "mechanism": "pack-cu-v1", "inputs": {"owner.cu": "abcdef01" * 8},
        "sections": {"$.stage.roles": "declared", "$.params": "compiled",
                     "$.symbols": "observed", "$.tensor.exports[0].placements": "fitted"},
    }
    stage = load_stage_manifest(raw)
    restored = StageManifest.from_json_obj(stage.to_json_obj())
    assert restored.to_json_obj()["provenance"] == raw["provenance"]
    assert validate_assembly(load_pipeline(_valid_pipeline()), [restored]) == []
    without_sections = copy.deepcopy(raw)
    del without_sections["provenance"]["sections"]
    assert canonical_dumps(stage.to_json_obj()) != canonical_dumps(load_stage_manifest(without_sections).to_json_obj())


def test_absent_provenance_is_omitted_from_model_encoding() -> None:
    encoded = load_stage_manifest(_valid_stage()).to_json_obj()
    assert "provenance" not in encoded
    assert StageManifest.from_json_obj(encoded).to_json_obj() == encoded


@pytest.mark.parametrize(("provenance", "path"), [
    (None, "$.provenance"),
    ({"mechanism": "", "inputs": {}}, "$.provenance.mechanism"),
    ({"mechanism": "capture", "inputs": {"owner.hpp": "A" * 64}}, "$.provenance.inputs.owner.hpp"),
    ({"mechanism": "capture", "inputs": {"owner.hpp": "a" * 63}}, "$.provenance.inputs.owner.hpp"),
    ({"mechanism": "capture", "inputs": {"": "a" * 64}}, "$.provenance.inputs"),
    ({"mechanism": "capture", "inputs": {}, "extra": True}, "$.provenance.extra"),
    ({"mechanism": "capture", "inputs": {}, "sections": None}, "$.provenance.sections"),
    ({"mechanism": "capture", "inputs": {}, "sections": {"": "declared"}}, "$.provenance.sections"),
    ({"mechanism": "capture", "inputs": {}, "sections": {"$.params": "derived"}}, "$.provenance.sections.$.params"),
    ({"mechanism": "capture", "inputs": {}, "sections": {"$.params": ["compiled"]}}, "$.provenance.sections.$.params"),
])
def test_provenance_rejects_invalid_records(provenance: Any, path: str) -> None:
    raw = _valid_stage()
    raw["provenance"] = provenance
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [("L-SYM", path)]


def test_host_schema_preserves_explicit_host_without_stage_params() -> None:
    from pipec2.host.profile import HOST_ABI, dtypes_digest

    raw = _valid_stage()
    raw["pipelink_abi"]["schema"] = "3.0"
    raw["pipelink_abi"]["target"]["host_abi"] = HOST_ABI
    raw["stage"]["roles"] = {}
    raw["endpoints"] = []
    raw["claims"] = []
    raw["params"] = {"bytes": 0, "align": 1, "type": "NoParams", "sha256_of_layout_probe": ZERO_HASH}
    raw["host"] = {"target": {"dtypes": dtypes_digest(raw["pipelink_abi"]["target"]["profile"])}}
    stage = load_stage_manifest(raw)
    assert StageManifest.from_json_obj(stage.to_json_obj()).host == raw["host"]
    raw["params"]["bytes"] = 16
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [("L-HOSTCHAN", "$.params")]


def test_schema_22_without_host_rejects_a_host_abi_stamp() -> None:
    raw = _valid_stage()
    raw["pipelink_abi"]["schema"] = "2.2"
    raw["pipelink_abi"]["target"]["host_abi"] = "host-v1"
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L-HOSTCHAN", "$.pipelink_abi.target")
    ]


def test_schema_22_without_host_requires_params_for_device_roles() -> None:
    raw = _valid_stage()
    raw["pipelink_abi"]["schema"] = "2.2"
    raw["params"]["bytes"] = 0
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L-PARAMS", "$.params.bytes")
    ]




def test_smem_endpoint_accepts_smem_page_base_claim() -> None:
    raw_stage = _valid_stage()
    raw_stage["claims"][0]["domain"] = "smem_page"
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["handles"]["pipelink_ab_smem_base"]["domain"] = "smem_page"

    stage = load_stage_manifest(raw_stage)
    pipeline = load_pipeline(raw_pipeline)

    assert stage.endpoints[0].space == "smem"
    assert stage.claims[0].domain == "smem_page"
    assert validate_assembly(pipeline, [stage]) == []


def _bad_prefix(obj: dict[str, Any]) -> None:
    obj["stage"]["symbol_prefix"] = "chosen_"


def _roles_exports_overlap(obj: dict[str, Any]) -> None:
    obj["stage"]["exports"]["load"] = obj["stage"]["exports"].pop("register_compute")


def _bad_entry_abi(obj: dict[str, Any]) -> None:
    obj["stage"]["entry_abi"] = "v3"


def _variant_incomplete_key(obj: dict[str, Any]) -> None:
    obj["stage"]["variants"] = [
        {"key": {"pipelink_a": 1}, "index": 0},
        {"key": {"pipelink_a": 2, "pipelink_b": 1}, "index": 1},
    ]


def _variant_duplicate_index(obj: dict[str, Any]) -> None:
    obj["stage"]["variants"] = [
        {"key": {"pipelink_variant": 1}, "index": 0},
        {"key": {"pipelink_variant": 2}, "index": 0},
    ]


def _variant_duplicate_entry(obj: dict[str, Any]) -> None:
    obj["stage"]["variants"] = [
        {"key": {"pipelink_variant": 1}, "index": 0, "entry": "producer_stage_v"},
        {"key": {"pipelink_variant": 2}, "index": 1, "entry": "producer_stage_v"},
    ]


def _variant_not_total(obj: dict[str, Any]) -> None:
    obj["stage"]["variants"] = [
        {"key": {"pipelink_a": 1, "pipelink_b": 1}, "index": 0},
        {"key": {"pipelink_a": 1, "pipelink_b": 2}, "index": 1},
        {"key": {"pipelink_a": 2, "pipelink_b": 1}, "index": 2},
    ]


def _unknown_claim_domain(obj: dict[str, Any]) -> None:
    obj["claims"][0]["domain"] = "future_memory"


def _interval_missing_extent(obj: dict[str, Any]) -> None:
    del obj["claims"][0]["extent"]


def _regs_has_interval_field(obj: dict[str, Any]) -> None:
    obj["claims"][3]["align"] = 8


def _regs_missing_roles(obj: dict[str, Any]) -> None:
    del obj["claims"][3]["roles"]


def _bad_regs_granularity(obj: dict[str, Any]) -> None:
    obj["claims"][3]["values"] = [65]


def _noncanonical_extent_key(obj: dict[str, Any]) -> None:
    table = obj["claims"][7]["extent"]["table"]
    value = table.pop("pipelink_ab_depth=2,pipelink_variant=1")
    table["pipelink_variant=1,pipelink_ab_depth=2"] = value


def _incompatible_handle_use(obj: dict[str, Any]) -> None:
    obj["endpoints"][0]["binds"]["full"] = "pipelink_ab_smem_base"


def _missing_mbarrier_init(obj: dict[str, Any]) -> None:
    del obj["endpoints"][0]["protocol"]["init"]


def _member_and_full_side(obj: dict[str, Any]) -> None:
    obj["endpoints"][0]["protocol"]["full"]["member_of"] = "other_ring"


def _unknown_protocol_kind(obj: dict[str, Any]) -> None:
    obj["endpoints"][0]["protocol"] = {"kind": "semaphore"}


def _bad_descriptor(obj: dict[str, Any]) -> None:
    obj["stage"]["descriptor"]["align"] = 24


def _bad_params(obj: dict[str, Any]) -> None:
    obj["params"]["align"] = 48


def _unknown_nested_key(obj: dict[str, Any]) -> None:
    obj["params"]["extra"] = 1


@pytest.mark.parametrize(
    ("case", "mutate", "clause_id"),
    [
        ("prefix normalization", _bad_prefix, "L-SYM"),
        ("role export disjointness", _roles_exports_overlap, "L-SYM"),
        ("entry ABI", _bad_entry_abi, "L-ENTRYABI"),
        ("complete variant keys", _variant_incomplete_key, "L-HANDLE"),
        ("regs value granularity", _bad_regs_granularity, "L-REGS"),
        ("injective variant index", _variant_duplicate_index, "L-HANDLE"),
        ("injective variant entry", _variant_duplicate_entry, "L-HANDLE"),
        ("total variant product", _variant_not_total, "L-HANDLE"),
        ("profile domain existence", _unknown_claim_domain, "L-HANDLE"),
        ("interval presence", _interval_missing_extent, "L-ALIGN"),
        ("regs omitted interval fields", _regs_has_interval_field, "L-REGS"),
        ("regs required roles", _regs_missing_roles, "L-REGS"),
        ("canonical extent table key", _noncanonical_extent_key, "L-HANDLE"),
        ("handle class domain use", _incompatible_handle_use, "L-HANDLE"),
        ("mbarrier init required", _missing_mbarrier_init, "L-PROTO"),
        ("shared member xor full side", _member_and_full_side, "L-PROTO"),
        ("protocol kind", _unknown_protocol_kind, "L-PROTO"),
        ("descriptor alignment", _bad_descriptor, "L-PARAMS"),
        ("params alignment", _bad_params, "L-PARAMS"),
        ("strict unknown nested key", _unknown_nested_key, "L-PARAMS"),
    ],
)
def test_stage_schema_rejections(
    case: str,
    mutate: Callable[[dict[str, Any]], None],
    clause_id: str,
) -> None:
    del case
    obj = _valid_stage()
    mutate(obj)
    with pytest.raises(SchemaError) as exc:
        load_stage_manifest(obj)
    _assert_clause(exc, clause_id)


def _bad_handle_domain(obj: dict[str, Any]) -> None:
    obj["pipeline"]["handles"]["pipelink_regs_load"]["domain"] = "smem"


def _duplicate_handle_values(obj: dict[str, Any]) -> None:
    obj["pipeline"]["handles"]["pipelink_ab_depth"]["values"] = [2, 2]


def _undeclared_forced_handle(obj: dict[str, Any]) -> None:
    obj["pipeline"]["forced_bindings"] = {"pipelink_missing": 1}


def _wrong_priority_class(obj: dict[str, Any]) -> None:
    obj["pipeline"]["depth_priority"] = ["pipelink_regs_load"]


def _bad_forced_value(obj: dict[str, Any]) -> None:
    obj["pipeline"]["forced_bindings"]["pipelink_variant"] = 2


def _unknown_phase(obj: dict[str, Any]) -> None:
    obj["pipeline"]["exclusive"] = [["setup", "missing"]]


def _self_phase(obj: dict[str, Any]) -> None:
    obj["pipeline"]["exclusive"] = [["steady", "steady"]]


def _duplicate_phase_pair(obj: dict[str, Any]) -> None:
    obj["pipeline"]["exclusive"] = [["setup", "steady"], ["steady", "setup"]]


def _unmatched_instruction_pattern(obj: dict[str, Any]) -> None:
    obj["pipeline"]["phases"] = ["setup", "steady"]
    obj["pipeline"]["exclusive"] = [["instr_*", "instr_*"]]


def _unknown_pipeline_domain(obj: dict[str, Any]) -> None:
    obj["pipeline"]["domains"]["unknown"] = {"capacity": 1}


def _missing_page_bytes(obj: dict[str, Any]) -> None:
    del obj["pipeline"]["domains"]["smem_page"]["page_bytes"]


def _bad_pipeline_topology(obj: dict[str, Any]) -> None:
    obj["pipeline"]["topology"]["scheduler"] = "unknown"


def _bad_stamp(obj: dict[str, Any]) -> None:
    obj["pipelink_abi"]["schema"] = "2.0"


def _unknown_pipeline_key(obj: dict[str, Any]) -> None:
    obj["pipeline"]["surprise"] = True


@pytest.mark.parametrize(
    ("case", "mutate", "clause_id"),
    [
        ("handle class-domain consistency", _bad_handle_domain, "L-HANDLE"),
        ("duplicate handle values", _duplicate_handle_values, "L-HANDLE"),
        ("every referenced handle declared", _undeclared_forced_handle, "L-HANDLE"),
        ("priority handle class", _wrong_priority_class, "L-HANDLE"),
        ("forced value allowed", _bad_forced_value, "L-HANDLE"),
        ("unknown exclusive phase", _unknown_phase, "L-PHASE"),
        ("exclusive self pair", _self_phase, "L-PHASE"),
        ("duplicate reversed exclusive pair", _duplicate_phase_pair, "L-PHASE"),
        ("unmatched instruction wildcard", _unmatched_instruction_pattern, "L-PHASE"),
        ("unknown pipeline domain", _unknown_pipeline_domain, "L-HANDLE"),
        ("smem page bytes", _missing_page_bytes, "L-HANDLE"),
        ("pipeline topology", _bad_pipeline_topology, "L-TOPO"),
        ("ABI draft stamp", _bad_stamp, "L-TOPO"),
        ("strict unknown pipeline key", _unknown_pipeline_key, "L-HANDLE"),
    ],
)
def test_pipeline_schema_rejections(
    case: str,
    mutate: Callable[[dict[str, Any]], None],
    clause_id: str,
) -> None:
    del case
    obj = _valid_pipeline()
    mutate(obj)
    with pytest.raises(SchemaError) as exc:
        load_pipeline(obj)
    _assert_clause(exc, clause_id)


def test_single_instruction_phase_pattern_expands_to_no_pairs() -> None:
    obj = _valid_pipeline()
    obj["pipeline"]["phases"] = ["instr_gemm0"]
    obj["pipeline"]["exclusive"] = [["instr_*", "instr_*"]]
    assert load_pipeline(obj).exclusive == []


def test_cross_object_handle_validation_rejects_undeclared_reference() -> None:
    raw_pipeline = _valid_pipeline()
    del raw_pipeline["pipeline"]["handles"]["pipelink_page_slot"]
    failures = validate_assembly(load_pipeline(raw_pipeline), [load_stage_manifest(_valid_stage())])
    assert failures
    assert {failure.clause_id for failure in failures} == {"L-HANDLE"}
    assert any("not declared" in failure.message for failure in failures)


def test_cross_object_handle_validation_rejects_class_domain_disagreement() -> None:
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["handles"]["pipelink_ab_full_slots"]["domain"] = "smem"
    failures = validate_assembly(load_pipeline(raw_pipeline), [load_stage_manifest(_valid_stage())])
    assert failures
    assert {failure.clause_id for failure in failures} == {"L-HANDLE"}
    assert any("domain disagrees" in failure.message for failure in failures)


def test_cross_object_handle_validation_rejects_empty_value_intersection() -> None:
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["handles"]["pipelink_regs_load"]["values"] = [80]
    failures = validate_assembly(load_pipeline(raw_pipeline), [load_stage_manifest(_valid_stage())])
    assert failures
    assert {failure.clause_id for failure in failures} == {"L-HANDLE"}
    assert any("empty intersection" in failure.message for failure in failures)


def test_cross_object_values_intersect_across_referencing_stages() -> None:
    producer = load_stage_manifest(_valid_stage())
    consumer_claims = [
        replace(claim, values=[80]) if claim.domain == "regs" else claim
        for claim in producer.claims
    ]
    consumer = replace(producer, id="consumer.stage", claims=consumer_claims)
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["stages"].append(
        {"artifact_path": "consumer.pla", "stage_id": "consumer.stage"}
    )
    raw_pipeline["pipeline"]["handles"]["pipelink_regs_load"]["values"] = [64, 72, 80]
    failures = validate_assembly(load_pipeline(raw_pipeline), [consumer, producer])
    assert failures
    assert {failure.clause_id for failure in failures} == {"L-HANDLE"}
    assert any("empty intersection" in failure.message for failure in failures)


def test_forced_interval_binding_is_not_treated_as_a_finite_value_constraint() -> None:
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["forced_bindings"]["pipelink_ab_smem_base"] = 0
    assert validate_assembly(
        load_pipeline(raw_pipeline),
        [load_stage_manifest(_valid_stage())],
    ) == []


def test_cross_object_claim_lifetime_must_name_a_pipeline_phase() -> None:
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["phases"] = ["setup", "other", "instr_gemm0", "instr_gemm1"]
    raw_pipeline["pipeline"]["exclusive"] = [["setup", "other"], ["instr_*", "instr_*"]]
    failures = validate_assembly(
        load_pipeline(raw_pipeline),
        [load_stage_manifest(_valid_stage())],
    )
    assert {failure.clause_id for failure in failures} == {"L-PHASE"}
    assert any("undeclared phase" in failure.message for failure in failures)


def test_mbarrier_shared_member_form_loads() -> None:
    obj = _valid_stage()
    obj["endpoints"][0]["protocol"]["full"] = {"member_of": "root_ring", "tx_bytes": "tile"}
    stage = load_stage_manifest(copy.deepcopy(obj))
    assert stage.endpoints[0].protocol.full is not None
    assert stage.endpoints[0].protocol.full.to_json_obj() == {
        "member_of": "root_ring",
        "tx_bytes": "tile",
    }


def test_schema_22_round_trips_optional_physical_initializing_warp() -> None:
    raw = _valid_stage()
    raw["pipelink_abi"]["schema"] = "2.2"
    init = raw["endpoints"][0]["protocol"]["init"]
    init["warp"] = 3
    stage = load_stage_manifest(raw)
    assert stage.endpoints[0].protocol.init.to_json_obj() == init
    restored = StageManifest.from_json_obj(stage.to_json_obj())
    assert restored.endpoints[0].protocol.init.to_json_obj() == init
    assert canonical_dumps(restored.to_json_obj()) == canonical_dumps(stage.to_json_obj())
    del init["warp"]
    assert load_stage_manifest(raw).endpoints[0].protocol.init.to_json_obj() == init


@pytest.mark.parametrize("materializer", ["stage", "dispatch"])
def test_schema_22_derives_init_owner_from_physical_warp(materializer) -> None:
    raw = _valid_stage()
    raw["pipelink_abi"]["schema"] = "2.2"
    init = {"materializer": materializer, "phase": 0, "warp": 3}
    raw["endpoints"][0]["protocol"]["init"] = init
    stage = load_stage_manifest(raw)
    assert stage.endpoints[0].protocol.init.owner is None
    assert stage.endpoints[0].protocol.init.to_json_obj() == init
    restored = StageManifest.from_json_obj(stage.to_json_obj())
    assert canonical_dumps(restored.to_json_obj()) == canonical_dumps(stage.to_json_obj())


def test_schema_rejects_init_without_owner_or_warp() -> None:
    raw = _valid_stage()
    del raw["endpoints"][0]["protocol"]["init"]["owner"]
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L-PROTO", "$.endpoints[0].protocol.init"),
    ]


@pytest.mark.parametrize("warp", [-1, True, "3", None])
def test_schema_rejects_invalid_physical_initializing_warp(warp) -> None:
    raw = _valid_stage()
    raw["pipelink_abi"]["schema"] = "2.2"
    raw["endpoints"][0]["protocol"]["init"]["warp"] = warp
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L-PROTO", "$.endpoints[0].protocol.init.warp"),
    ]


@pytest.mark.parametrize("count", [
    "threads(load)+2*threads(mma)", " 3 * warps( load ) + threads(mma) ", "warps(load)",
])
def test_arrive_sum_grammar(count) -> None:
    raw = _valid_stage()
    raw["endpoints"][0]["protocol"]["empty"]["arrive"]["count"] = count
    assert load_stage_manifest(raw).endpoints[0].protocol.empty.arrive.count == count


@pytest.mark.parametrize("count", ["0*threads(load)", "-2*warps(load)", "threads(load)+", "2+threads(load)", "threads(load)*2"])
def test_arrive_sum_rejects_nonpositive_coefficients_and_other_operators(count) -> None:
    raw = _valid_stage()
    raw["endpoints"][0]["protocol"]["empty"]["arrive"]["count"] = count
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert caught.value.failures[0].clause_id == "L-PROTO"


def test_scheduler_strategy_pins_decode_inputs() -> None:
    raw = _valid_stage()
    strategy = {"id": "clc-sm100-v1", "inputs": [
        "swizzle_size", "raster_order", "cluster_shape_m", "cluster_shape_n",
    ]}
    raw["topology"]["scheduler_strategy"] = strategy
    assert load_stage_manifest(raw).topology.scheduler_strategy.to_json_obj() == strategy
    for invalid in [
        {**strategy, "id": "unknown"},
        {**strategy, "inputs": ["log_swizzle_size", *strategy["inputs"][1:]]},
        {**strategy, "inputs": strategy["inputs"][::-1]},
    ]:
        raw["topology"]["scheduler_strategy"] = invalid
        with pytest.raises(SchemaError) as caught:
            load_stage_manifest(raw)
        assert caught.value.failures[0].clause_id == "L-TOPO"


def _native_stage():
    raw = _valid_stage()
    endpoint = raw["endpoints"][0]
    endpoint["ring"]["depth"] = {"fixed": 2}
    endpoint["protocol"]["init"]["materializer"] = "stage"
    endpoint["protocol"]["full"]["tx_bytes"] = 0
    endpoint["protocol"]["full"]["native"] = {
        "base": "pipelink_ab_smem_base", "offset_bytes": 8, "stride_bytes": 8,
    }
    raw["claims"][0]["extent"] = _affine(24)
    return raw


def test_native_storage_checks_byte_extent_and_model_roundtrip() -> None:
    raw = _native_stage()
    stage = load_stage_manifest(raw)
    assert stage.endpoints[0].protocol.full.native.offset_bytes == 8
    assert StageManifest.from_json_obj(stage.to_json_obj()).to_json_obj() == stage.to_json_obj()
    raw["claims"][0]["extent"] = _affine(23)
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert caught.value.failures[0].clause_id == "L-PROTO"


def test_native_storage_respects_claim_relative_offset() -> None:
    raw = _native_stage()
    raw["claims"][0]["offset"] = 8
    assert load_stage_manifest(raw).endpoints[0].protocol.full.native.offset_bytes == 8
    raw["endpoints"][0]["protocol"]["full"]["native"]["offset_bytes"] = 0
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert caught.value.failures[0].clause_id == "L-PROTO"


@pytest.mark.parametrize("mutation", [
    "missing_claim", "wrong_domain", "negative_offset", "zero_stride", "dispatch_init",
])
def test_native_storage_rejects_invalid_claim_or_unsupported_dispatch_initialization(mutation) -> None:
    raw = _native_stage()
    native = raw["endpoints"][0]["protocol"]["full"]["native"]
    if mutation == "missing_claim":
        native["base"] = "pipelink_missing"
    elif mutation == "wrong_domain":
        native["base"] = "pipelink_warp_load"
    elif mutation == "negative_offset":
        native["offset_bytes"] = -1
    elif mutation == "zero_stride":
        native["stride_bytes"] = 0
    else:
        raw["endpoints"][0]["protocol"]["init"]["materializer"] = "dispatch"
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert caught.value.failures[0].clause_id == "L-PROTO"


def test_one_shot_barrier_requires_no_empty_side_and_single_stage_initialized_slot() -> None:
    raw = _native_stage()
    endpoint = raw["endpoints"][0]
    endpoint["ring"]["depth"] = {"fixed": 1}
    protocol = endpoint["protocol"]
    protocol["release"] = "never"
    empty = protocol.pop("empty")
    assert load_stage_manifest(raw).endpoints[0].protocol.empty is None
    for mutation in ("empty", "depth", "dispatch"):
        invalid = copy.deepcopy(raw)
        if mutation == "empty":
            invalid["endpoints"][0]["protocol"]["empty"] = empty
        elif mutation == "depth":
            invalid["endpoints"][0]["ring"]["depth"] = {"fixed": 2}
        else:
            invalid["endpoints"][0]["protocol"]["init"]["materializer"] = "dispatch"
        with pytest.raises(SchemaError) as caught:
            load_stage_manifest(invalid)
        assert caught.value.failures[0].clause_id == "L-PROTO"


def _signal_stage() -> dict[str, Any]:
    raw = _valid_stage()
    endpoint = raw["endpoints"][0]
    endpoint.update(space="none", proxy="generic", binds={"full": "pipelink_ab_full_slots"})
    for key in ("tile", "layout", "ring"):
        del endpoint[key]
    protocol = endpoint["protocol"]
    del protocol["empty"]
    protocol["release"] = "external"
    protocol["full"]["tx_bytes"] = 0
    return raw


@pytest.mark.parametrize("release,materializer,depth", [
    ("external", "dispatch", 1), ("external", "stage", 3),
    ("never", "dispatch", 1), ("never", "stage", 1),
])
def test_completion_only_endpoint_loads_without_a_tile_or_layout(release, materializer, depth) -> None:
    raw = _signal_stage()
    endpoint = raw["endpoints"][0]
    endpoint["protocol"]["release"] = release
    endpoint["protocol"]["init"]["materializer"] = materializer
    if depth != 1:
        endpoint["ring"] = {"depth": {"fixed": depth}, "stage_stride_bytes": 0}
    stage = load_stage_manifest(raw)
    signal = stage.endpoints[0]
    assert signal.space == "none"
    assert signal.tile.dtype == "none" and signal.tile.shape == ()
    assert signal.protocol.empty is None
    assert signal.ring.depth.fixed == depth
    encoded = stage.to_json_obj()
    assert "tile" not in encoded["endpoints"][0] and "layout" not in encoded["endpoints"][0]
    assert StageManifest.from_json_obj(encoded).endpoints[0] == signal


@pytest.mark.parametrize("mutation,clause", [
    ("tile", "L-PAIR"), ("layout", "L-PAIR"), ("base", "L-PAIR"),
    ("grid", "L-PAIR"), ("async", "L-PAIR"),
    ("tx", "L-PROTO"), ("tx_tile", "L-PROTO"), ("tx_sum", "L-PROTO"),
    ("empty", "L-PROTO"), ("once-per-acquire", "L-PROTO"), ("shared", "L-PROTO"),
    ("never_reused", "L-PROTO"),
])
def test_completion_only_endpoint_rejects_payload_and_release_contract_violations(mutation, clause) -> None:
    raw = _signal_stage()
    endpoint = raw["endpoints"][0]
    protocol = endpoint["protocol"]
    if mutation in {"tile", "layout"}:
        endpoint[mutation] = _valid_stage()["endpoints"][0][mutation]
    elif mutation == "base":
        endpoint["binds"]["base"] = "pipelink_ab_base"
    elif mutation == "grid":
        endpoint["scope"] = "grid"
    elif mutation == "async":
        endpoint["proxy"] = "async"
    elif mutation in {"tx", "tx_tile", "tx_sum"}:
        protocol["full"]["tx_bytes"] = {"tx": 8, "tx_tile": "tile", "tx_sum": {"sum": ["ab_ring"]}}[mutation]
    elif mutation == "empty":
        protocol["empty"] = _valid_stage()["endpoints"][0]["protocol"]["empty"]
    elif mutation == "once-per-acquire":
        protocol["release"] = mutation
    elif mutation == "shared":
        protocol["full"] = {"member_of": "ab_ring", "tx_bytes": 0}
    else:
        protocol["release"] = "never"
        endpoint["ring"] = {"depth": {"fixed": 2}, "stage_stride_bytes": 0}
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert caught.value.failures[0].clause_id == clause


@pytest.mark.parametrize("missing", ["tile", "layout"])
def test_tiled_endpoint_still_requires_payload_description(missing) -> None:
    raw = _valid_stage()
    del raw["endpoints"][0][missing]
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert caught.value.failures[0].clause_id == "L-PAIR"


def test_tiled_endpoint_rejects_external_release() -> None:
    raw = _valid_stage()
    protocol = raw["endpoints"][0]["protocol"]
    protocol["release"] = "external"
    del protocol["empty"]
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert caught.value.failures[0].clause_id == "L-PROTO"


_CLC_STRATEGY = {"id": "clc-sm100-v1", "inputs": [
    "swizzle_size", "raster_order", "cluster_shape_m", "cluster_shape_n",
]}


WARP_PROFILES = ("sm_90a", "sm_120a")


def _warp_target(profile: str, lto_arch: str | None = None) -> dict[str, str]:
    return {"profile": profile, "lto_arch": lto_arch or get_profile(profile).arch.lto, "toolkit": "cuda-13.3.0"}


def _warp_stage(profile: str, lto_arch: str | None = None) -> dict[str, Any]:
    raw = _valid_stage()
    raw["pipelink_abi"]["target"] = _warp_target(profile, lto_arch)
    raw["topology"]["cta_group"] = {"fixed": 1}
    return raw


def _warp_pipeline(profile: str) -> dict[str, Any]:
    raw = _valid_pipeline()
    raw["pipelink_abi"]["target"] = _warp_target(profile)
    return raw


def _failures(exc: pytest.ExceptionInfo[SchemaError]) -> list[tuple[str, str]]:
    return [(failure.clause_id, failure.path) for failure in exc.value.failures]


@pytest.mark.parametrize("profile", WARP_PROFILES)
def test_warp_profile_stage_and_pipeline_load_and_assemble(profile: str) -> None:
    stage = load_stage_manifest(_warp_stage(profile))
    pipeline = load_pipeline(_warp_pipeline(profile))
    assert stage.target.profile == pipeline.target.profile == profile
    assert stage.topology.cta_group == 1
    assert stage.topology.scheduler == pipeline.topology.scheduler == "static"
    assert validate_assembly(pipeline, [stage]) == []


@pytest.mark.parametrize(("profile", "lto_arch"), [
    ("sm_90a", "lto_90"), ("sm_90a", "lto_90a"), ("sm_120a", "lto_120"), ("sm_120a", "lto_120a"),
])
def test_warp_profile_admits_both_lto_spellings(profile: str, lto_arch: str) -> None:
    assert load_stage_manifest(_warp_stage(profile, lto_arch)).target.lto_arch == lto_arch


@pytest.mark.parametrize(("profile", "lto_arch"), [
    ("sm_90a", "lto_103"), ("sm_90a", "lto_103a"), ("sm_90a", "lto_120a"),
    ("sm_120a", "lto_90a"), ("sm_120a", "lto_103a"),
])
def test_warp_profile_rejects_other_lto_arch(profile: str, lto_arch: str) -> None:
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(_warp_stage(profile, lto_arch))
    assert _failures(caught) == [("L-TOPO", "$.pipelink_abi.target.lto_arch")]


def _tmem_claim(obj: dict[str, Any]) -> None:
    obj["claims"].append({
        "domain": "tmem", "handle": "pipelink_acc_tmem", "extent": _affine(32), "align": 32,
    })


def _clc_scheduler(obj: dict[str, Any]) -> None:
    obj["topology"]["scheduler"] = "clc"


def _clc_strategy(obj: dict[str, Any]) -> None:
    obj["topology"]["scheduler_strategy"] = dict(_CLC_STRATEGY)


def _cta_group_pair(obj: dict[str, Any]) -> None:
    obj["topology"]["cta_group"] = {"fixed": 2}


@pytest.mark.parametrize("profile", WARP_PROFILES)
@pytest.mark.parametrize(
    ("mutate", "clause_id", "path"),
    [
        (_tmem_claim, "L-HANDLE", "$.claims[8].domain"),
        (_clc_scheduler, "L-TOPO", "$.topology.scheduler"),
        (_clc_strategy, "L-TOPO", "$.topology.scheduler_strategy.id"),
        (_cta_group_pair, "L-TOPO", "$.topology.cta_group.fixed"),
    ],
    ids=["tmem claim", "clc scheduler", "clc-sm100-v1 strategy", "cta_group 2"],
)
def test_warp_profile_stage_rejects_sm100_vocabulary(
    mutate: Callable[[dict[str, Any]], None], clause_id: str, path: str, profile: str
) -> None:
    obj = _warp_stage(profile)
    mutate(obj)
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(obj)
    assert _failures(caught) == [(clause_id, path)]


@pytest.mark.parametrize("profile", WARP_PROFILES)
def test_warp_profile_pipeline_rejects_clc_scheduler(profile: str) -> None:
    obj = _warp_pipeline(profile)
    obj["pipeline"]["topology"]["scheduler"] = "clc"
    with pytest.raises(SchemaError) as caught:
        load_pipeline(obj)
    assert _failures(caught) == [("L-TOPO", "$.pipeline.topology.scheduler")]


def test_sm103_admits_clc_scheduler_and_strategy() -> None:
    raw_stage = _valid_stage()
    raw_stage["topology"]["scheduler"] = "clc"
    raw_stage["topology"]["scheduler_strategy"] = dict(_CLC_STRATEGY)
    stage = load_stage_manifest(raw_stage)
    assert stage.topology.scheduler == "clc"
    assert stage.topology.scheduler_strategy.id == "clc-sm100-v1"
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["topology"]["scheduler"] = "clc"
    assert load_pipeline(raw_pipeline).topology.scheduler == "clc"


def _tensor_lifetime_stage(lifetime: dict[str, str]) -> dict[str, Any]:
    raw = _valid_stage()
    raw["pipelink_abi"]["schema"] = "2.2"
    raw["tensor"] = _valid_tensor()
    raw["tensor"]["sites"][0]["role"] = "load"
    raw["tensor"]["sites"][0]["guarantees"] = ["enter_once", "single_completion"]
    raw["claims"][0]["lifetime"] = lifetime
    return raw


@pytest.mark.parametrize("lifetime", [
    {"tensor": "accumulator", "from": "tile(m,n,k)", "to": "acc_ready"},
    {"tensor": "accumulator", "from": "acc_ready", "to": "import_complete"},
    {"tensor": "epilogue", "from": "tile_ready", "to": "acc_ready"},
    {"tensor": "epilogue", "from": "acc_ready", "to": "import_complete"},
    {"tensor": "epilogue", "from": "import_complete"},
])
def test_tensor_claim_lifetime_roundtrip_and_own_endpoint_events(lifetime) -> None:
    stage = load_stage_manifest(_tensor_lifetime_stage(lifetime))
    claim = stage.claims[0]
    assert claim.lifetime == TensorLifetime(lifetime["tensor"], lifetime["from"], lifetime.get("to"))
    assert claim.to_json_obj()["lifetime"] == lifetime
    restored = StageManifest.from_json_obj(stage.to_json_obj())
    assert restored.claims == sorted(stage.claims, key=lambda item: (item.domain, item.handle))
    failures = validate_assembly(load_pipeline(_valid_pipeline()), [stage])
    assert not [failure for failure in failures if failure.clause_id == "L-PHASE"]


@pytest.mark.parametrize("lifetime,suffix", [
    ({"tensor": "other.stage:accumulator", "from": "acc_ready"}, ".tensor"),
    ({"tensor": "missing", "from": "acc_ready"}, ".tensor"),
    ({"tensor": "accumulator", "from": "tile_ready"}, ".from"),
    ({"tensor": "epilogue", "from": "tile(m,n,k)"}, ".from"),
    ({"tensor": "accumulator", "from": "acc_ready", "to": "missing"}, ".to"),
    ({"tensor": "accumulator", "from": "acc_ready", "to": "acc_ready"}, ""),
    ({"tensor": "epilogue", "from": "import_complete", "to": "acc_ready"}, ""),
])
def test_tensor_claim_lifetime_rejects_unknown_or_reversed_events(lifetime, suffix) -> None:
    stage = load_stage_manifest(_tensor_lifetime_stage(lifetime))
    failures = validate_assembly(load_pipeline(_valid_pipeline()), [stage])
    assert [(failure.clause_id, failure.path) for failure in failures if failure.clause_id == "L-PHASE"] == [
        ("L-PHASE", "stages.producer.stage.claims.smem.pipelink_ab_smem_base.lifetime" + suffix),
    ]


@pytest.mark.parametrize("lifetime,suffix", [
    ({"tensor": "accumulator"}, ""),
    ({"tensor": "accumulator", "from": "acc_ready", "until": "import_complete"}, ".until"),
    ({"tensor": 1, "from": "acc_ready"}, ".tensor"),
    ({"tensor": "accumulator", "from": ""}, ".from"),
    ({"tensor": "accumulator", "from": "acc_ready", "to": None}, ".to"),
])
def test_tensor_claim_lifetime_rejects_malformed_object(lifetime, suffix) -> None:
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(_tensor_lifetime_stage(lifetime))
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L-PHASE", "$.claims[0].lifetime" + suffix),
    ]


def _counter_stage(scope: str, release: str, acquire: str) -> tuple[dict[str, Any], int]:
    raw = _valid_stage()
    index = next(i for i, endpoint in enumerate(raw["endpoints"]) if endpoint["name"] == "grid_counter")
    raw["endpoints"][index]["scope"] = scope
    raw["endpoints"][index]["protocol"].update(release=release, acquire=acquire)
    return raw, index


@pytest.mark.parametrize("scope,release,acquire", [
    ("grid", "red.release.gpu", "ld.acquire.gpu"),
    ("grid", "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"),
    ("system", "red.release.sys", "ld.acquire.sys"),
    ("system", "red.release.sys", "ld.relaxed.sys+fence.acquire.sys"),
])
def test_counter_endpoint_admits_each_scoped_release_acquire_pair(scope, release, acquire) -> None:
    raw, index = _counter_stage(scope, release, acquire)
    endpoint = load_stage_manifest(raw).endpoints[index]
    assert (endpoint.scope, endpoint.protocol.counter.release_op, endpoint.protocol.counter.acquire_op) == (
        scope, release, acquire)


@pytest.mark.parametrize("scope,release,acquire,clause_id,suffix", [
    ("grid", "red.release.gpu", "ld.acquire.sys", "L-PROTO", ".protocol"),
    ("system", "red.release.sys", "ld.acquire.gpu", "L-PROTO", ".protocol"),
    ("grid", "red.release.gpu", "ld.relaxed.sys+fence.acquire.sys", "L-PROTO", ".protocol"),
    ("grid", "red.release.gpu", "ld.relaxed.gpu", "L-PROTO", ".protocol"),
    ("grid", "red.release.cta", "ld.acquire.cta", "L-PROTO", ".protocol"),
    ("system", "red.release.gpu", "ld.acquire.gpu", "L-PAIR", ".scope"),
    ("grid", "red.release.sys", "ld.acquire.sys", "L-PAIR", ".scope"),
    ("cta", "red.release.gpu", "ld.acquire.gpu", "L-PAIR", ".scope"),
], ids=["mixed-gpu-sys", "mixed-sys-gpu", "mixed-gpu-sys-fence", "relaxed-without-fence", "unknown-pair",
        "system-gpu-ops", "grid-sys-ops", "cta-scope"])
def test_counter_endpoint_rejects_mixed_pairs_and_scope_op_mismatch(scope, release, acquire, clause_id, suffix) -> None:
    raw, index = _counter_stage(scope, release, acquire)
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert _failures(caught) == [(clause_id, f"$.endpoints[{index}]{suffix}")]


@pytest.mark.parametrize("name,space,admitted", [
    ("grid_counter", None, True), ("named_sync", None, True), ("ab_ring", None, True), ("named_sync", "reg", False),
])
def test_only_register_endpoints_require_owner_role(name, space, admitted) -> None:
    raw = _valid_stage()
    index = next(i for i, endpoint in enumerate(raw["endpoints"]) if endpoint["name"] == name)
    raw["endpoints"][index]["owner_role"] = None
    if space is not None:
        raw["endpoints"][index]["space"] = space
    if admitted:
        assert load_stage_manifest(raw).endpoints[index].owner_role is None
        return
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert _failures(caught) == [("L-PAIR", f"$.endpoints[{index}].owner_role")]


def test_system_scope_is_admitted_only_for_counter_endpoints() -> None:
    raw = _valid_stage()
    raw["endpoints"][0]["scope"] = "system"
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert _failures(caught) == [("L-PAIR", "$.endpoints[0].scope")]
