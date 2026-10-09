"""Family-neutral correctness/safety and resource audit record writers."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .constants import (
    ABSOLUTE_ZERO_RESOURCE_SUBJECTS,
    BARRIER_AUDIT_FIELDS,
    CORRECTNESS_TOLERANCES,
    FAMILY2_COMPARATOR_EVIDENCE_FIELDS,
    FAMILY1_RESOURCE_SUBJECT,
    FAMILY2_RESOURCE_SUBJECT,
    FAMILY2_RESOURCE_VECTOR_FIELDS,
    FAMILY2_VERBOSE_RELINK_EVIDENCE_FIELDS,
    NONFINITE_POLICY,
    REQUIRED_LOCAL_BYTES,
    REQUIRED_RACECHECK_HAZARDS,
    REQUIRED_SPILL_LOAD_BYTES,
    REQUIRED_SPILL_STORE_BYTES,
    REQUIRED_STACK_BYTES,
    REQUIRED_SYNCCHECK_ERRORS,
    SCHEMA_VERSIONS,
)
from .util import require_sha256, utc_now, write_json_atomic


def _subject(record: dict[str, Any]) -> str:
    subject_id = record.get("subject_id")
    if not isinstance(subject_id, str) or subject_id not in CORRECTNESS_TOLERANCES:
        raise ValueError(
            f"subject_id must be one of {sorted(CORRECTNESS_TOLERANCES)}"
        )
    return subject_id


def build_correctness_record(record: dict[str, Any]) -> dict[str, Any]:
    """Close a Gate-A/B/safety record under the plan's subject tolerance."""
    result = deepcopy(record)
    result["schema_version"] = SCHEMA_VERSIONS["correctness"]
    result.setdefault("utc_time", utc_now())
    subject_id = _subject(result)
    for field in ("case_id", "image_id", "stream_map_sha256"):
        if field not in result:
            raise ValueError(f"correctness record requires {field}")
    require_sha256(result["stream_map_sha256"], "stream_map_sha256")
    classes = result.get("stream_classes")
    if not isinstance(classes, list) or not classes or any(
        not isinstance(value, str) or not value for value in classes
    ):
        raise ValueError("correctness record needs non-empty stream_classes")
    gate_a = result.get("gate_a")
    gate_b = result.get("gate_b")
    safety = result.get("safety")
    barrier = result.get("barrier_audit")
    if not isinstance(gate_a, dict) or not isinstance(
        gate_a.get("bitwise_identical"), bool
    ):
        raise ValueError("gate_a.bitwise_identical boolean is required")
    if not isinstance(gate_b, dict) or not isinstance(gate_b.get("passed"), bool):
        raise ValueError("gate_b.passed boolean is required")
    tolerance = gate_b.get("tolerance")
    if not isinstance(tolerance, dict) or set(tolerance) != {"rtol", "atol"}:
        raise ValueError("gate_b.tolerance must contain exactly caller-supplied rtol/atol")
    if any(
        isinstance(tolerance[field], bool)
        or not isinstance(tolerance[field], (int, float))
        for field in ("rtol", "atol")
    ):
        raise ValueError("gate_b rtol/atol must be numeric")
    expected_rtol, expected_atol = CORRECTNESS_TOLERANCES[subject_id]
    if (float(tolerance["rtol"]), float(tolerance["atol"])) != (
        expected_rtol,
        expected_atol,
    ):
        raise ValueError(
            f"{subject_id} Gate-B tolerance must be "
            f"rtol={expected_rtol}, atol={expected_atol}"
        )
    if gate_b.get("nonfinite_policy") != NONFINITE_POLICY:
        raise ValueError("gate_b non-finite policy differs from the required policy")
    if subject_id == "family2" and gate_b.get("reference_outputs_finite") is not True:
        raise ValueError("every Family-2 reference output must be finite")
    if not isinstance(safety, dict):
        raise ValueError("safety record is required")
    if safety.get("racecheck_hazards") != REQUIRED_RACECHECK_HAZARDS:
        raise ValueError("racecheck must report zero hazards")
    if safety.get("synccheck_errors") != REQUIRED_SYNCCHECK_ERRORS:
        raise ValueError("synccheck must report zero errors")
    required_barrier_fields = {"passed", "image_derived", *BARRIER_AUDIT_FIELDS}
    if (
        not isinstance(barrier, dict)
        or set(barrier) != required_barrier_fields
        or barrier.get("passed") is not True
    ):
        raise ValueError("complete passing image-derived barrier audit is required")
    result["tolerance_policy"] = {
        "subject_id": subject_id,
        "rtol": expected_rtol,
        "atol": expected_atol,
        "plan_section": "§4.5" if subject_id == "family2" else "§2.1",
    }
    result["all_true"] = gate_a["bitwise_identical"] and gate_b["passed"]
    return result


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _resource_vector(entry: dict[str, Any], label: str) -> dict[str, int]:
    vector = entry.get("resource_vector")
    expected = {*FAMILY2_RESOURCE_VECTOR_FIELDS, "LOCAL"}
    if not isinstance(vector, dict) or set(vector) != expected:
        raise ValueError(f"{label}.resource_vector must contain exactly {sorted(expected)}")
    return {
        field: _nonnegative_integer(vector[field], f"{label}.resource_vector.{field}")
        for field in (*FAMILY2_RESOURCE_VECTOR_FIELDS, "LOCAL")
    }


def _comparator_entry(entry: object, label: str, *, stock: bool) -> tuple[dict[str, Any], dict[str, int]]:
    if not isinstance(entry, dict):
        raise ValueError(f"family2_comparator.{label} must be an object")
    required = {
        "entry_symbol",
        "specialization_traits",
        "cubin_sha256",
        "sass_sha256",
        "elf_sha256",
        "resource_vector",
        "evidence",
    }
    if stock:
        required |= {"source_sha256", "compile_command", "toolkit", "target"}
    if set(entry) != required:
        raise ValueError(
            f"family2_comparator.{label} fields must be exactly {sorted(required)}"
        )
    if not isinstance(entry["entry_symbol"], str) or not entry["entry_symbol"]:
        raise ValueError(f"family2_comparator.{label}.entry_symbol is required")
    traits = entry["specialization_traits"]
    if not isinstance(traits, dict) or not traits:
        raise ValueError(f"family2_comparator.{label}.specialization_traits is required")
    for hash_field in ("cubin_sha256", "sass_sha256", "elf_sha256"):
        require_sha256(entry[hash_field], f"family2_comparator.{label}.{hash_field}")
    evidence = entry["evidence"]
    if not isinstance(evidence, dict):
        raise ValueError(f"family2_comparator.{label}.evidence must be an object")
    counter_source = evidence.get("counter_source")
    expected_evidence = {*FAMILY2_COMPARATOR_EVIDENCE_FIELDS, "counter_source"}
    if counter_source == "verbose-relink-byte-identical-sass":
        expected_evidence |= set(FAMILY2_VERBOSE_RELINK_EVIDENCE_FIELDS)
    elif counter_source != "direct-final-link":
        raise ValueError(f"family2_comparator.{label} has inadmissible counter source")
    if set(evidence) != expected_evidence:
        raise ValueError(
            f"family2_comparator.{label}.evidence must contain exactly "
            f"{sorted(expected_evidence)} for counter source {counter_source!r}"
        )
    for hash_field in (
        *FAMILY2_COMPARATOR_EVIDENCE_FIELDS,
        *(
            FAMILY2_VERBOSE_RELINK_EVIDENCE_FIELDS
            if counter_source == "verbose-relink-byte-identical-sass"
            else ()
        ),
    ):
        require_sha256(
            evidence[hash_field],
            f"family2_comparator.{label}.evidence.{hash_field}",
        )
    if (
        counter_source == "verbose-relink-byte-identical-sass"
        and evidence["measured_elf_sha256"] != entry["elf_sha256"]
    ):
        raise ValueError(
            f"family2_comparator.{label} verbose-relink measured ELF hash mismatch"
        )
    if (
        counter_source == "verbose-relink-byte-identical-sass"
        and evidence["measured_sass_sha256"] != entry["sass_sha256"]
    ):
        raise ValueError(
            f"family2_comparator.{label} verbose-relink measured SASS hash mismatch"
        )
    if stock:
        require_sha256(entry["source_sha256"], "family2_comparator.stock.source_sha256")
        command = entry["compile_command"]
        if not isinstance(command, list) or not command or any(
            not isinstance(argument, str) or not argument for argument in command
        ):
            raise ValueError("family2_comparator.stock.compile_command must be an argv array")
        if any(
            not isinstance(entry[field], str) or not entry[field]
            for field in ("toolkit", "target")
        ):
            raise ValueError("family2 stock toolkit/target bindings are required")
    return entry, _resource_vector(entry, f"family2_comparator.{label}")


def _family1_wrapped_entry(
    wrapper: object, label: str, *, stock: bool
) -> tuple[dict[str, Any], dict[str, int]]:
    if not isinstance(wrapper, dict) or set(wrapper) != {"register_cap", "entry"}:
        raise ValueError(
            f"family1_exception.{label} must contain exactly register_cap/entry"
        )
    if wrapper["register_cap"] != 255:
        raise ValueError("Family-1 exception linked/control/stock cap must be fixed at 255")
    entry, vector = _comparator_entry(wrapper["entry"], label, stock=stock)
    if entry["evidence"]["counter_source"] != "verbose-relink-byte-identical-sass":
        raise ValueError(
            f"family1_exception.{label} requires direct verbose-relink counters "
            "with byte-identical decoded SASS evidence"
        )
    return entry, vector


def _family1_exception_predicate(
    result: dict[str, Any], resources: dict[str, Any]
) -> dict[str, Any]:
    exception = result.get("family1_exception")
    expected = {
        "mapping_id",
        "row_id",
        "winner_config_sha256",
        "variant",
        "performance_observations_used",
        "discriminator_sha256",
        "waived_components",
        "cap_domain",
        "linked",
        "control",
        "stock",
    }
    if not isinstance(exception, dict) or set(exception) != expected:
        raise ValueError(
            "Family-1 exception requires one complete exact winner/variant/entry mapping"
        )
    if not isinstance(exception["mapping_id"], str) or not exception["mapping_id"]:
        raise ValueError("family1 exception mapping_id is required")
    if exception["row_id"] != result["case_id"]:
        raise ValueError("family1 exception row_id must equal the resource record case_id")
    if exception["variant"] not in {"v0", "v1", "v2", "v3", "v4"}:
        raise ValueError("family1 exception variant must be v0-v4")
    if exception["performance_observations_used"] != 0:
        raise ValueError("family1 exception discriminator must be pre-performance")
    for hash_field in (
        "winner_config_sha256",
        "discriminator_sha256",
    ):
        require_sha256(exception[hash_field], f"family1_exception.{hash_field}")
    waived = exception["waived_components"]
    if (
        not isinstance(waived, list)
        or not waived
        or len(set(waived)) != len(waived)
        or set(waived) - set(FAMILY2_RESOURCE_VECTOR_FIELDS)
    ):
        raise ValueError(
            "family1 waived_components must be a unique non-empty resource-vector subset"
        )
    cap_domain = exception["cap_domain"]
    cap_expected = {
        "requested_cap_min",
        "requested_cap_max",
        "all_requested_caps_tested",
        "effective_cap_count",
        "effective_caps_sha256",
        "direct_counter_table_sha256",
        "per_component",
    }
    if not isinstance(cap_domain, dict) or set(cap_domain) != cap_expected:
        raise ValueError("family1 exception cap_domain is incomplete")
    require_sha256(
        cap_domain["effective_caps_sha256"],
        "family1_exception.cap_domain.effective_caps_sha256",
    )
    require_sha256(
        cap_domain["direct_counter_table_sha256"],
        "family1_exception.cap_domain.direct_counter_table_sha256",
    )
    effective_count = _nonnegative_integer(
        cap_domain["effective_cap_count"],
        "family1_exception.cap_domain.effective_cap_count",
    )
    domain_complete = (
        cap_domain["requested_cap_min"] == 1
        and cap_domain["requested_cap_max"] == 255
        and cap_domain["all_requested_caps_tested"] is True
        and effective_count > 0
    )
    component_proofs = cap_domain["per_component"]
    if (
        not isinstance(component_proofs, dict)
        or set(component_proofs) != set(FAMILY2_RESOURCE_VECTOR_FIELDS)
    ):
        raise ValueError("family1 cap-domain proof must cover every vector component")
    proof_status: dict[str, bool] = {}
    for field in FAMILY2_RESOURCE_VECTOR_FIELDS:
        proof = component_proofs[field]
        if not isinstance(proof, dict) or set(proof) != {
            "tested_effective_cap_count",
            "minimum",
            "zero_unattainable",
        }:
            raise ValueError(f"family1 cap-domain proof for {field} is incomplete")
        tested_count = _nonnegative_integer(
            proof["tested_effective_cap_count"],
            f"family1 cap-domain {field} tested count",
        )
        minimum = _nonnegative_integer(
            proof["minimum"], f"family1 cap-domain {field} minimum"
        )
        proof_status[field] = (
            domain_complete
            and tested_count == effective_count
            and proof["zero_unattainable"] is True
            and minimum > 0
        )
    linked, linked_vector = _family1_wrapped_entry(
        exception["linked"], "linked", stock=False
    )
    control, control_vector = _family1_wrapped_entry(
        exception["control"], "control", stock=False
    )
    stock, stock_vector = _family1_wrapped_entry(
        exception["stock"], "stock", stock=True
    )
    if len({linked["entry_symbol"], control["entry_symbol"], stock["entry_symbol"]}) != 3:
        raise ValueError("family1 comparator entry symbols must be explicit and unambiguous")
    if not (
        linked["specialization_traits"]
        == control["specialization_traits"]
        == stock["specialization_traits"]
    ):
        raise ValueError("family1 linked/control/stock full winner traits must match")
    if linked["cubin_sha256"] != result["image_sha256"]:
        raise ValueError("family1 exception linked cubin hash mismatch")
    if linked["sass_sha256"] != result["sass_sha256"]:
        raise ValueError("family1 exception linked SASS hash mismatch")
    measured_vector = {
        field: resources[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS
    }
    if measured_vector != {
        field: linked_vector[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS
    }:
        raise ValueError("family1 exception linked vector differs from measured resources")
    local_zero = all(
        vector["LOCAL"] == REQUIRED_LOCAL_BYTES
        for vector in (linked_vector, control_vector, stock_vector)
    ) and resources["LOCAL"] == REQUIRED_LOCAL_BYTES
    component_predicates: dict[str, Any] = {}
    for field in FAMILY2_RESOURCE_VECTOR_FIELDS:
        exception_eligible = (
            field in waived
            and proof_status[field]
            and control_vector[field] > 0
            and stock_vector[field] > 0
        )
        linked_below = (
            linked_vector[field] <= control_vector[field]
            and linked_vector[field] <= stock_vector[field]
        )
        passed = linked_below if exception_eligible else linked_vector[field] == 0
        component_predicates[field] = {
            "waiver_requested": field in waived,
            "zero_unattainable_complete_domain": proof_status[field],
            "final_control_nonzero": control_vector[field] > 0,
            "final_stock_nonzero": stock_vector[field] > 0,
            "exception_eligible": exception_eligible,
            "linked_no_greater_than_control_and_stock": linked_below,
            "absolute_zero_fallback": not exception_eligible,
            "passed": passed,
        }
    occupancy_pass = resources["blocks_per_sm"] >= resources["control_blocks_per_sm"]
    all_components_pass = all(
        predicate["passed"] for predicate in component_predicates.values()
    )
    return {
        "mode": "family1-narrow-cap-domain-componentwise-exception",
        "local_zero": local_zero,
        "final_register_cap": 255,
        "pre_performance_discriminator": True,
        "complete_requested_cap_domain": domain_complete,
        "linked_vector": {field: linked_vector[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS},
        "control_vector": {field: control_vector[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS},
        "stock_vector": {field: stock_vector[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS},
        "component_predicates": component_predicates,
        "blocks_per_sm_not_lower_than_exact_control": occupancy_pass,
        "passed": local_zero and all_components_pass and occupancy_pass,
    }


def _family2_predicate(result: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    comparator = result.get("family2_comparator")
    expected = {"mapping_id", "row_id", "linked", "control", "stock"}
    if not isinstance(comparator, dict) or set(comparator) != expected:
        raise ValueError(
            "Family-2 requires one unambiguous hash-bound linked/control/stock comparator mapping"
        )
    if not isinstance(comparator["mapping_id"], str) or not comparator["mapping_id"]:
        raise ValueError("family2 comparator mapping_id is required")
    if comparator["row_id"] != result["case_id"]:
        raise ValueError("family2 comparator row_id must equal the resource record case_id")
    linked, linked_vector = _comparator_entry(comparator["linked"], "linked", stock=False)
    control, control_vector = _comparator_entry(comparator["control"], "control", stock=False)
    stock, stock_vector = _comparator_entry(comparator["stock"], "stock", stock=True)
    if len({linked["entry_symbol"], control["entry_symbol"], stock["entry_symbol"]}) != 3:
        raise ValueError("family2 comparator entry symbols must be explicit and unambiguous")
    if not (
        linked["specialization_traits"]
        == control["specialization_traits"]
        == stock["specialization_traits"]
    ):
        raise ValueError("family2 linked/control/stock specialization traits must match exactly")
    if linked["cubin_sha256"] != result["image_sha256"]:
        raise ValueError("family2 linked comparator cubin hash does not bind the measured image")
    if linked["sass_sha256"] != result["sass_sha256"]:
        raise ValueError("family2 linked comparator SASS hash does not bind the measured image")
    measured_vector = {
        field: resources[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS
    }
    if measured_vector != {
        field: linked_vector[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS
    }:
        raise ValueError("family2 linked comparator vector differs from measured resources")
    local_zero = all(
        vector["LOCAL"] == REQUIRED_LOCAL_BYTES
        for vector in (linked_vector, control_vector, stock_vector)
    ) and resources["LOCAL"] == REQUIRED_LOCAL_BYTES
    linked_equals_control = all(
        linked_vector[field] == control_vector[field]
        for field in FAMILY2_RESOURCE_VECTOR_FIELDS
    )
    linked_below_stock = all(
        linked_vector[field] <= stock_vector[field]
        for field in FAMILY2_RESOURCE_VECTOR_FIELDS
    )
    occupancy_pass = resources["blocks_per_sm"] >= resources["control_blocks_per_sm"]
    return {
        "mode": "family2-componentwise-control-equality-and-stock-ceiling",
        "local_zero": local_zero,
        "linked_vector": {field: linked_vector[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS},
        "control_vector": {field: control_vector[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS},
        "stock_vector": {field: stock_vector[field] for field in FAMILY2_RESOURCE_VECTOR_FIELDS},
        "linked_equals_control_componentwise": linked_equals_control,
        "linked_no_greater_than_stock_componentwise": linked_below_stock,
        "specialization_traits_match": True,
        "mapping_complete": True,
        "blocks_per_sm_not_lower_than_exact_control": occupancy_pass,
        "passed": local_zero and linked_equals_control and linked_below_stock and occupancy_pass,
    }


def build_resource_record(record: dict[str, Any]) -> dict[str, Any]:
    """Close an absolute-zero, narrow Family-1, or Family-2 resource record."""
    result = deepcopy(record)
    result["schema_version"] = SCHEMA_VERSIONS["resource"]
    result.setdefault("utc_time", utc_now())
    subject_id = _subject(result)
    for field in ("case_id", "image_id", "image_sha256", "sass_sha256"):
        if field not in result:
            raise ValueError(f"resource record requires {field}")
    require_sha256(result["image_sha256"], "image_sha256")
    require_sha256(result["sass_sha256"], "sass_sha256")
    resources = result.get("resources")
    required = {
        "REG",
        "STACK",
        "LOCAL",
        "spill_load_bytes",
        "spill_store_bytes",
        "dynamic_smem_bytes",
        "tmem",
        "named_mbarrier_map",
        "warp_map",
        "launch_topology",
        "occupancy",
        "blocks_per_sm",
        "control_blocks_per_sm",
    }
    if not isinstance(resources, dict) or set(resources) != required:
        raise ValueError(f"resources must contain exactly {sorted(required)}")
    integer_fields = (
        "REG",
        "STACK",
        "LOCAL",
        "spill_load_bytes",
        "spill_store_bytes",
        "dynamic_smem_bytes",
        "blocks_per_sm",
        "control_blocks_per_sm",
    )
    for field in integer_fields:
        _nonnegative_integer(resources[field], f"resources.{field}")
    if subject_id == FAMILY2_RESOURCE_SUBJECT:
        if "family1_exception" in result:
            raise ValueError("Family-2 records must not carry a Family-1 exception")
        result["headline_predicate"] = _family2_predicate(result, resources)
        return result
    if "family2_comparator" in result:
        raise ValueError("non-Family-2 records must not carry a Family-2 comparator mapping")
    if subject_id == FAMILY1_RESOURCE_SUBJECT and "family1_exception" in result:
        result["headline_predicate"] = _family1_exception_predicate(result, resources)
        return result
    if "family1_exception" in result:
        raise ValueError("only Family-1 records may carry a Family-1 exception")
    if subject_id not in ABSOLUTE_ZERO_RESOURCE_SUBJECTS:
        raise ValueError(f"no resource predicate is registered for {subject_id}")
    stack_spills_zero = (
        resources["STACK"] == REQUIRED_STACK_BYTES
        and resources["spill_load_bytes"] == REQUIRED_SPILL_LOAD_BYTES
        and resources["spill_store_bytes"] == REQUIRED_SPILL_STORE_BYTES
    )
    local_zero = resources["LOCAL"] == REQUIRED_LOCAL_BYTES
    occupancy_pass = resources["blocks_per_sm"] >= resources["control_blocks_per_sm"]
    result["headline_predicate"] = {
        "mode": "absolute-stack-and-spills-zero",
        "local_zero": local_zero,
        "stack_and_spills_zero": stack_spills_zero,
        "blocks_per_sm_not_lower_than_exact_control": occupancy_pass,
        "passed": local_zero and stack_spills_zero and occupancy_pass,
    }
    return result


def write_correctness_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    result = build_correctness_record(record)
    write_json_atomic(path, result)
    return result


def write_resource_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    result = build_resource_record(record)
    write_json_atomic(path, result)
    return result
