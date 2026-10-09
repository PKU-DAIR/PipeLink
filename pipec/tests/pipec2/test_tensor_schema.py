from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from pipec2 import schema
from pipec2.model import (
    CLAUSES,
    CLAUSE_PHASES,
    LinkResult,
    ParamsLayout,
    PipeLinkError,
    Pipeline,
    SchemaError,
    StageManifest,
    TensorConnection,
)
from pipec2.profile import Profile, get_profile
from pipec2.schema import load_pipeline, load_stage_manifest, validate_assembly
from test_schema import _valid_pipeline, _valid_stage


def _connection(name: str = "accumulator") -> dict[str, str]:
    return {
        "name": name,
        "export": "producer.stage:published",
        "import": "producer.stage:received",
    }


def _tensor_section() -> dict[str, Any]:
    identity = {
        "result": "accumulator",
        "work": "work",
        "version": "final",
        "coords": [{"name": "x", "sort": "int", "lo": 0, "hi": 1}],
        "dtype": "f32",
        "interpretation": "ieee-754",
    }
    return {
        "version": "tensor-v2",
        "exports": [{
            "name": "published",
            "identity": identity,
            "placements": [{
                "guard": None,
                "space": "reg",
                "units": "element",
                "components": {"participant": ["const", 0], "position": ["coord", "x"]},
            }],
            "read_capabilities": ["reg_copy"],
            "readiness": {"after": "ready", "reuse_after": "reuse"},
            "fixed_bindings": {},
            "backing": [],
            "calls": {"produce": "produce", "release": None},
        }],
        "imports": [{
            "name": "received",
            "identity": identity,
            "demands": [{
                "kind": "positions",
                "domain": [{"name": "i", "sort": "int", "lo": 0, "hi": 1}],
                "guard": None,
                "identity_coords": {"x": ["coord", "i"]},
                "participant": ["const", 0],
                "position": ["coord", "i"],
            }],
            "callable": {
                "symbol": "pipelink_import",
                "declaration": 'extern "C" __device__ void pipelink_import(float*);',
                "params": [{"name": "dest", "meaning": "dest_fragment", "ctype": "float*"}],
                "events": {"ready": "pipelink_import_ready", "done": "pipelink_import_done"},
            },
            "sites": ["load"],
            "lends": [],
            "completion": "delivered",
            "rendezvous": [],
        }],
        "sites": [{
            "name": "load",
            "role": "load",
            "participants": {"set": [0]},
            "invocation": "work",
            "guarantees": [],
            "observes": ["ready", "reuse"],
        }],
        "bindings": [],
    }


def _tensor_stage() -> dict[str, Any]:
    raw = _valid_stage()
    raw["pipelink_abi"]["schema"] = "3.0"
    raw["tensor"] = _tensor_section()
    return raw


def test_tensor_connections_parse_and_round_trip_canonically() -> None:
    raw = _valid_pipeline()
    raw["pipeline"]["tensor_connections"] = [_connection("z"), _connection("a")]
    pipeline = load_pipeline(raw)
    expected = [_connection("a"), _connection("z")]
    encoded = pipeline.to_json_obj()
    assert encoded["tensor_connections"] == expected
    assert Pipeline.from_json_obj(encoded).to_json_obj() == encoded
    assert TensorConnection.from_json_obj(expected[0]).import_ == "producer.stage:received"
    assert TensorConnection.from_json_obj(expected[0]).to_json_obj() == expected[0]
    raw["pipeline"]["tensor_connections"].reverse()
    assert load_pipeline(raw).to_json_obj() == encoded


@pytest.mark.parametrize(
    ("connections", "path"),
    [
        (None, "$.pipeline.tensor_connections"),
        (["accumulator"], "$.pipeline.tensor_connections[0]"),
        ([{"name": "accumulator", "export": "producer.stage:published"}],
         "$.pipeline.tensor_connections[0]"),
        ([{**_connection(), "extra": True}], "$.pipeline.tensor_connections[0].extra"),
        ([{**_connection(), "name": 3}], "$.pipeline.tensor_connections[0].name"),
        ([{**_connection(), "export": "published"}], "$.pipeline.tensor_connections[0].export"),
        ([{**_connection(), "export": "producer.stage:published:extra"}],
         "$.pipeline.tensor_connections[0].export"),
        ([{**_connection(), "export": "INVALID:published"}],
         "$.pipeline.tensor_connections[0].export"),
        ([{**_connection(), "import": "producer.stage:"}], "$.pipeline.tensor_connections[0].import"),
        ([{**_connection(), "import": False}], "$.pipeline.tensor_connections[0].import"),
    ],
)
def test_tensor_connection_shape_is_strict(connections: Any, path: str) -> None:
    raw = _valid_pipeline()
    raw["pipeline"]["tensor_connections"] = connections
    with pytest.raises(SchemaError) as caught:
        load_pipeline(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("T-CONTRACT", path)
    ]


def test_tensor_connection_names_are_unique() -> None:
    raw = _valid_pipeline()
    raw["pipeline"]["tensor_connections"] = [_connection(), _connection()]
    with pytest.raises(SchemaError) as caught:
        load_pipeline(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("T-CONTRACT", "$.pipeline.tensor_connections[1].name")
    ]


@pytest.mark.parametrize("version", ["2.2", "3.0"])
def test_stage_tensor_loads_round_trips_and_resolves_connections(version) -> None:
    raw_stage = _tensor_stage()
    raw_stage["pipelink_abi"]["schema"] = version
    stage = load_stage_manifest(raw_stage)
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["tensor_connections"] = [_connection()]
    assert stage.tensor.exports[0].identity.dtype == "f32"
    assert stage.tensor.imports[0].name == "received"
    encoded = stage.to_json_obj()
    restored = StageManifest.from_json_obj(encoded)
    assert restored.to_json_obj() == encoded
    assert restored.tensor == stage.tensor
    assert validate_assembly(load_pipeline(raw_pipeline), [stage]) == []
    ordinary_stage = load_stage_manifest(_valid_stage())
    assert stage.endpoints == ordinary_stage.endpoints
    assert stage.claims == ordinary_stage.claims


@pytest.mark.parametrize(
    ("kind", "reference"),
    [
        ("export", "missing.stage:published"),
        ("export", "producer.stage:received"),
        ("import", "producer.stage:published"),
    ],
)
def test_tensor_connection_references_resolve_in_the_correct_namespace(
    kind: str, reference: str,
) -> None:
    raw_stage = _tensor_stage()
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["tensor_connections"] = [{**_connection(), kind: reference}]
    failures = validate_assembly(load_pipeline(raw_pipeline), [load_stage_manifest(raw_stage)])
    assert [(failure.clause_id, failure.path) for failure in failures] == [
        ("T-CONTRACT", f"pipeline.tensor_connections.accumulator.{kind}")
    ]


def test_tensor_connection_cannot_reference_an_ordinary_stage_surface() -> None:
    raw_pipeline = _valid_pipeline()
    raw_pipeline["pipeline"]["tensor_connections"] = [{
        **_connection(), "export": "producer.stage:register_compute",
    }]
    failures = validate_assembly(load_pipeline(raw_pipeline), [load_stage_manifest(_valid_stage())])
    assert [(failure.clause_id, failure.path) for failure in failures] == [
        ("T-CONTRACT", "pipeline.tensor_connections.accumulator.export"),
        ("T-CONTRACT", "pipeline.tensor_connections.accumulator.import"),
    ]


def test_absent_tensor_does_not_require_tensor_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pipec2.tensor", None)
    monkeypatch.setitem(sys.modules, "pipec2.tensor.contracts", None)
    reloaded = importlib.reload(schema)
    raw_stage = _valid_stage()
    stage = reloaded.load_stage_manifest(raw_stage)
    encoded = stage.to_json_obj()
    assert stage.tensor is None
    assert "tensor" not in encoded
    assert stage.endpoints[0].layout.to_json_obj() == raw_stage["endpoints"][0]["layout"]
    assert StageManifest.from_json_obj(encoded).to_json_obj() == encoded
    pipeline = reloaded.load_pipeline(_valid_pipeline())
    assert pipeline.tensor_connections == ()
    assert reloaded.validate_assembly(pipeline, [stage]) == []


def test_present_null_tensor_is_not_treated_as_absent() -> None:
    raw_stage = _tensor_stage()
    raw_stage["tensor"] = None
    with pytest.raises(PipeLinkError) as caught:
        load_stage_manifest(raw_stage)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("T-CONTRACT", "$.tensor")
    ]


@pytest.mark.parametrize("version", ["2.0-draft"])
def test_draft_schemas_reject_tensor_sections(version: str) -> None:
    raw = _tensor_stage()
    raw["pipelink_abi"]["schema"] = version
    with pytest.raises(PipeLinkError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("T-CONTRACT", "$.tensor")
    ]


def test_tensor_clause_phases_are_registered() -> None:
    expected = {
        "T-CONTRACT": "load",
        "T-MODEL": "layout",
        "T-PLAN": "solve",
        "T-LOWER": "link",
        "T-NATIVE": "link",
    }
    assert expected.keys() <= CLAUSES.keys()
    assert {name: CLAUSE_PHASES[name] for name in expected} == expected


def test_adapter_compile_route_is_distinct_from_stage_lto_arch() -> None:
    profile = get_profile("sm_103a")
    adapter = profile.adapter_compile
    assert adapter.gencode == ("compute_103a", "lto_103a")
    assert adapter.forced_macros == ()
    assert adapter.include_roots_env == (
        "CUTLASS_INCLUDE_DIR",
        "TILELANG_INCLUDE_DIR",
    )
    assert adapter.toolkit == profile.toolkit == "cuda-13.3.0"
    assert profile.lto_arch == "lto_103"
    assert Profile.from_json_obj(profile.to_json_obj()) == profile


def test_link_result_preserves_tensor_outcomes_without_reinterpreting_them() -> None:
    record = {"connections": [{
        "name": "accumulator",
        "status": "UNKNOWN",
        "bounds": {"K": 4, "H": 8},
        "solver_outcome": {"reason": "resource limit"},
        "lowering": None,
        "trusted_premises": [],
        "native": None,
    }]}
    result = LinkResult(
        kernel_symbol="pipelink_kernel",
        threads=128,
        regs_cap=96,
        dyn_smem_bytes=0,
        cluster=(1, 1, 1),
        params=ParamsLayout(0, 64, []),
        bindings={},
        variants={},
        intervals={},
        protocol_expansions=[],
        observed=(72, 0, 0),
        input_hashes={},
        generated_hashes={},
        sass_report=None,
        tensor=record,
    )
    encoded = result.to_json_obj()
    assert encoded["tensor"] == record
    assert LinkResult.from_json_obj(encoded) == result
