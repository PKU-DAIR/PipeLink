from __future__ import annotations

import copy
from typing import Any

import pytest

from pipec2.decls import declaration_digest, normalize_declaration
from pipec2.model import PipeLinkError
from pipec2.tensor import expr
from pipec2.tensor.contracts import StageTensor, TensorContractError, load_stage_tensor


def _domain(name: str, hi: int) -> dict[str, Any]:
    return {"name": name, "sort": "int", "lo": 0, "hi": hi}


def _valid_tensor() -> dict[str, Any]:
    identity = {
        "result": "C", "work": "tile(m,n,k)", "version": "final-accumulator",
        "coords": [_domain("row", 128), _domain("col", 32)],
        "dtype": "fp32", "interpretation": "ieee754-binary32-accumulated",
    }
    return {
        "version": "tensor-v2",
        "exports": [{
            "name": "accumulator", "identity": copy.deepcopy(identity),
            "placements": [{
                "guard": ["lt", ["coord", "row"], ["const", 128]],
                "space": "tmem", "units": "dp/column",
                "components": {
                    "dp": ["coord", "row"],
                    "column": ["add", ["bind", "tmem_base"],
                               ["add", ["coord", "col"], ["mulc", 32, ["bind", "slot"]]]],
                },
            }],
            "read_capabilities": ["tmem_packet_receipt"],
            "readiness": {"after": "acc_ready", "reuse_after": "import_complete"},
            "fixed_bindings": {"tmem_base": 0}, "backing": [],
            "calls": {"produce": "produce", "release": None},
        }],
        "imports": [{
            "name": "epilogue", "identity": copy.deepcopy(identity),
            "demands": [{
                "kind": "positions", "domain": [_domain("r", 128), _domain("c", 32)],
                "guard": None,
                "identity_coords": {"row": ["coord", "r"], "col": ["coord", "c"]},
                "participant": ["coord", "r"], "position": ["coord", "c"],
            }],
            "callable": {
                "symbol": "pipelink_import_accumulator",
                "declaration": 'extern "C" __device__ void pipelink_import_accumulator(pipelink_tensor_source, float*, void*);',
                "params": [
                    {"name": "source", "meaning": "source_base", "ctype": "pipelink_tensor_source"},
                    {"name": "dest", "meaning": "dest_fragment", "ctype": "float*"},
                    {"name": "scratch", "meaning": "scratch", "ctype": "void*"},
                ],
                "events": {"ready": "pipelink_import_accumulator_ready", "done": "pipelink_import_accumulator_done"},
            },
            "sites": ["epilogue_hook"],
            "completion": "import_complete",
            "rendezvous": [{"event": event, "barrier": "rendezvous", "cohort": {"set": list(range(128))}}
                           for event in ("acc_ready", "import_complete")],
            "lends": [
                {"name": "scratch", "domain": "smem", "extent": 4096, "align": 128, "scope": "cta",
                 "binding": "relocatable", "location": {"handle": "pipelink_import_scratch", "base": None},
                 "owner": "epilogue", "lifetime": {"from": "tile_ready", "to": "import_complete"}, "sync": None},
                {"name": "rendezvous", "domain": "barrier", "extent": 1, "align": 1, "scope": "cta",
                 "binding": "fixed", "location": {"handle": None, "base": 1},
                 "owner": "epilogue", "lifetime": {"from": "tile_ready", "to": "import_complete"},
                 "sync": {"generations": 2, "capacity": 128}},
                {"name": "fragment", "domain": "reg", "extent": 32, "align": 1, "scope": "thread",
                 "binding": "fixed", "location": {"handle": "private_fragment", "base": None},
                 "owner": "epilogue", "lifetime": {"from": "tile_ready", "to": "epilogue_return"}, "sync": None},
            ],
        }],
        "sites": [{
            "name": "epilogue_hook", "role": "epilogue",
            "participants": {"set": list(range(128))},
            "invocation": "tile_ready", "guarantees": ["converged"],
            "observes": ["acc_ready", "import_complete"],
        }],
        "bindings": [
            {"name": "tmem_base", "sort": "int", "domain": {"name": "tmem_base", "sort": "int", "values": [0, 64]},
             "owner": "mma", "units": "column", "observation_scope": "cta", "lifetime": "tile"},
            {"name": "slot", "sort": "int", "domain": _domain("slot", 2),
             "owner": "mma", "units": "slot", "observation_scope": "cta", "lifetime": "tile"},
        ],
    }


def _load(obj: dict[str, Any]) -> StageTensor:
    return load_stage_tensor(obj, "$.tensor")


def _failure_paths(obj: dict[str, Any]) -> list[str]:
    with pytest.raises(TensorContractError) as caught:
        _load(obj)
    assert isinstance(caught.value, PipeLinkError)
    assert all(failure.clause_id == "T-CONTRACT" for failure in caught.value.failures)
    return [failure.path for failure in caught.value.failures]


def test_full_contract_roundtrip_preserves_physical_and_logical_maps() -> None:
    obj = _valid_tensor()
    stage = _load(obj)
    encoded = stage.to_json_obj()
    assert encoded == obj
    assert StageTensor.from_json_obj(encoded) == stage
    placement = stage.exports[0].placements[0]
    assert expr.evaluate(placement.components["dp"], {"row": 127}) == 127
    assert expr.evaluate(placement.components["column"], {"tmem_base": 0, "col": 31, "slot": 1}) == 63
    demand = stage.imports[0].demands[0]
    assert expr.evaluate(demand.identity_coords["row"], {"r": 127}) == 127
    assert expr.evaluate(demand.position, {"c": 31}) == 31


def test_callable_normalization_has_derived_digest_not_wire_field() -> None:
    obj = _valid_tensor()
    declaration = obj["imports"][0]["callable"]["declaration"]
    obj["imports"][0]["callable"]["declaration"] = " /* caller ABI */\n  " + declaration + "  // end\n"
    call = _load(obj).imports[0].callable
    assert call.declaration == normalize_declaration(declaration)
    assert call.declaration_digest == declaration_digest(declaration)
    obj["imports"][0]["callable"]["declaration_digest"] = call.declaration_digest
    assert "$.tensor.imports[0].callable.declaration_digest" in _failure_paths(obj)


def test_bad_version_is_contract_failure() -> None:
    obj = _valid_tensor()
    obj["version"] = "tensor-v1"
    assert _failure_paths(obj) == ["$.tensor.version"]


def test_unknown_import_site_is_rejected() -> None:
    obj = _valid_tensor()
    obj["imports"][0]["sites"].append("missing_hook")
    assert _failure_paths(obj) == ["$.tensor.imports[0].sites[1]"]


def test_undeclared_coordinate_is_rejected_at_expression_path() -> None:
    obj = _valid_tensor()
    obj["exports"][0]["placements"][0]["components"]["dp"] = ["coord", "undeclared"]
    assert _failure_paths(obj) == ["$.tensor.exports[0].placements[0].components.dp"]


def test_partner_expression_bindings_cannot_restate_an_undeclared_sort() -> None:
    obj = _valid_tensor()
    obj["exports"][0]["placements"][0]["components"]["dp"] = ["bind", "partner_slot"]
    assert _failure_paths(obj) == ["$.tensor.exports[0].placements[0].components.dp"]


def test_coordinate_and_binding_namespaces_are_not_interchangeable() -> None:
    obj = _valid_tensor()
    obj["exports"][0]["placements"][0]["components"]["dp"] = ["coord", "slot"]
    assert _failure_paths(obj) == ["$.tensor.exports[0].placements[0].components.dp"]
    obj["exports"][0]["placements"][0]["components"]["dp"] = ["bind", "row"]
    assert _failure_paths(obj) == ["$.tensor.exports[0].placements[0].components.dp"]


@pytest.mark.parametrize("collection", ["coords", "binding", "demand"])
def test_empty_domains_are_rejected(collection: str) -> None:
    obj = _valid_tensor()
    if collection == "coords":
        domain = obj["exports"][0]["identity"]["coords"][0]
        path = "$.tensor.exports[0].identity.coords[0]"
    elif collection == "binding":
        domain = obj["bindings"][1]["domain"]
        path = "$.tensor.bindings[1].domain"
    else:
        domain = obj["imports"][0]["demands"][0]["domain"][0]
        path = "$.tensor.imports[0].demands[0].domain[0]"
    domain["hi"] = domain["lo"]
    assert path in _failure_paths(obj)


def test_empty_explicit_binding_domain_is_rejected() -> None:
    obj = _valid_tensor()
    obj["bindings"][0]["domain"]["values"] = []
    assert "$.tensor.bindings[0].domain" in _failure_paths(obj)


@pytest.mark.parametrize(("field", "value"), [("name", "different"), ("sort", "bits[4]")])
def test_binding_domain_matches_the_local_declaration(field: str, value: str) -> None:
    obj = _valid_tensor()
    obj["bindings"][1]["domain"][field] = value
    assert f"$.tensor.bindings[1].domain.{field}" in _failure_paths(obj)


def test_lend_location_rules_distinguish_relocation_from_fixed_base() -> None:
    obj = _valid_tensor()
    lend = obj["imports"][0]["lends"][0]
    lend["location"] = {"handle": None, "base": 0}
    assert _failure_paths(obj) == ["$.tensor.imports[0].lends[0].location.handle"]
    lend["binding"] = "fixed"
    assert _load(obj).imports[0].lends[0].location.base == 0
    lend["location"]["base"] = None
    assert _failure_paths(obj) == ["$.tensor.imports[0].lends[0].location"]


@pytest.mark.parametrize("collection", ["exports", "imports", "sites", "bindings"])
def test_duplicate_record_names_are_rejected(collection: str) -> None:
    obj = _valid_tensor()
    index = len(obj[collection])
    obj[collection].append(copy.deepcopy(obj[collection][0]))
    assert f"$.tensor.{collection}[{index}].name" in _failure_paths(obj)


def test_memory_view_is_valid_even_outside_initial_model_scope() -> None:
    obj = _valid_tensor()
    demand = obj["imports"][0]["demands"][0]
    demand.pop("participant")
    demand.pop("position")
    demand.update(kind="memory_view", space="gmem", units="byte", components={
        "address": ["mulc", 4, ["add", ["mulc", 32, ["coord", "r"]], ["coord", "c"]]],
    })
    call = obj["imports"][0]["callable"]
    call["declaration"] = (
        'extern "C" __device__ void pipelink_import_accumulator(unsigned, const void**, void*);'
    )
    call["params"][1] = {"name": "dest_view", "meaning": "dest_view", "ctype": "const void**"}
    stage = _load(obj)
    assert stage.to_json_obj() == obj
    assert expr.evaluate(stage.imports[0].demands[0].components["address"], {"r": 127, "c": 31}) == 16380


def test_replication_preserves_distinct_demand_indices_for_one_identity() -> None:
    obj = _valid_tensor()
    demand = obj["imports"][0]["demands"][0]
    demand["domain"].append(_domain("copy", 2))
    demand["position"] = ["add", ["coord", "c"], ["mulc", 32, ["coord", "copy"]]]
    stage = _load(obj)
    loaded = stage.imports[0].demands[0]
    first = {"r": 7, "c": 3, "copy": 0}
    second = {"r": 7, "c": 3, "copy": 1}
    assert {name: expr.evaluate(value, first) for name, value in loaded.identity_coords.items()} == {"row": 7, "col": 3}
    assert {name: expr.evaluate(value, second) for name, value in loaded.identity_coords.items()} == {"row": 7, "col": 3}
    assert (expr.evaluate(loaded.position, first), expr.evaluate(loaded.position, second)) == (3, 35)
    assert stage.to_json_obj() == obj


def test_participant_expression_preserves_explicit_domain() -> None:
    obj = _valid_tensor()
    obj["sites"][0]["participants"] = {
        "expr": ["add", ["coord", "lane"], ["mulc", 32, ["coord", "warp"]]],
        "domain": [_domain("lane", 32), _domain("warp", 4)],
    }
    participants = _load(obj).sites[0].participants
    assert participants.to_json_obj() == obj["sites"][0]["participants"]
    assert expr.evaluate(participants.expr, {"lane": 31, "warp": 3}) == 127


def test_references_and_failures_are_complete_sorted_and_order_independent() -> None:
    obj = _valid_tensor()
    obj["version"] = "bad"
    obj["imports"][0]["sites"].append("missing_hook")
    obj["sites"][0]["guarantees"].append("unknown_guarantee")
    obj["imports"][0]["lends"][0]["location"]["handle"] = None
    expected = [
        "$.tensor.imports[0].lends[0].location.handle", "$.tensor.imports[0].sites[1]",
        "$.tensor.sites[0].guarantees[1]", "$.tensor.version",
    ]
    assert _failure_paths(obj) == expected
    reordered = dict(reversed(list(obj.items())))
    assert _failure_paths(reordered) == expected


def test_strict_nested_schema_aggregates_missing_unknown_and_wrong_type_fields() -> None:
    obj = _valid_tensor()
    lends = obj["imports"][0]["lends"]
    lends[0].pop("owner")
    lends[0]["extent"] = True
    lends[1]["location"]["inferred"] = True
    paths = _failure_paths(obj)
    assert "$.tensor.imports[0].lends[0].owner" in paths
    assert "$.tensor.imports[0].lends[0].extent" in paths
    assert "$.tensor.imports[0].lends[1].location.inferred" in paths
    assert paths == sorted(paths)


def test_reference_paths_preserve_indices_after_malformed_array_elements() -> None:
    obj = _valid_tensor()
    obj["imports"][0]["sites"].extend([False, "missing_hook"])
    assert _failure_paths(obj) == [
        "$.tensor.imports[0].sites[1]", "$.tensor.imports[0].sites[2]",
    ]


def test_local_fixed_bindings_require_admitted_values() -> None:
    obj = _valid_tensor()
    obj["exports"][0]["fixed_bindings"]["tmem_base"] = 1
    assert _failure_paths(obj) == ["$.tensor.exports[0].fixed_bindings.tmem_base"]


def test_partner_fixed_binding_is_deferred_without_restating_its_domain() -> None:
    obj = _valid_tensor()
    obj["exports"][0]["fixed_bindings"]["partner_slot"] = 7
    stage = _load(obj)
    assert stage.exports[0].fixed_bindings["partner_slot"] == 7
    assert StageTensor.from_json_obj(stage.to_json_obj()) == stage


def test_residual_bindings_derive_from_placements_including_guards() -> None:
    obj = _valid_tensor()
    placement = obj["exports"][0]["placements"][0]
    placement["guard"] = ["lt", ["bind", "slot"], ["const", 2]]
    stage = _load(obj)
    assert stage.exports[0].residual_bindings == ("slot",)
    obj["exports"][0]["fixed_bindings"]["slot"] = 1
    assert _load(obj).exports[0].residual_bindings == ()


@pytest.mark.parametrize("target", ["same_import", "other_import", "backing"])
def test_lend_names_are_unique_across_the_entire_stage(target: str) -> None:
    obj = _valid_tensor()
    lend = copy.deepcopy(obj["imports"][0]["lends"][0])
    if target == "same_import":
        obj["imports"][0]["lends"].append(lend)
        expected = "$.tensor.imports[0].lends[3].name"
    elif target == "other_import":
        other = copy.deepcopy(obj["imports"][0])
        other.update(name="other", lends=[lend], rendezvous=[])
        obj["imports"].append(other)
        expected = "$.tensor.imports[1].lends[0].name"
    else:
        obj["exports"][0]["backing"].append(lend)
        expected = "$.tensor.imports[0].lends[0].name"
    assert _failure_paths(obj) == [expected]


def test_backing_round_trips_without_becoming_a_consumer_lend() -> None:
    obj = _valid_tensor()
    backing = obj["imports"][0]["lends"].pop(0)
    backing["owner"] = "@export"
    obj["exports"][0]["backing"].append(backing)
    stage = _load(obj)
    assert stage.exports[0].backing[0].owner == "@export"
    assert [lend.name for lend in stage.imports[0].lends] == ["rendezvous", "fragment"]
    assert StageTensor.from_json_obj(stage.to_json_obj()) == stage


def test_guard_and_identity_coordinate_sorts_are_checked_at_load() -> None:
    obj = _valid_tensor()
    demand = obj["imports"][0]["demands"][0]
    demand["guard"] = ["const", 1]
    demand["identity_coords"]["row"] = ["tobits", 7, ["coord", "r"]]
    assert _failure_paths(obj) == [
        "$.tensor.imports[0].demands[0].guard",
        "$.tensor.imports[0].demands[0].identity_coords.row",
    ]
