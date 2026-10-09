"""Seal records once; verify persisted records with readers before dispatch."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .constants import (
    ESCALATION_TRIGGER_HIGH,
    ESCALATION_TRIGGER_LOW,
    EXPERIMENT_ONLY_PROTOCOL_VARIANTS,
)

FAST_RESULT_SCHEMA = "m3-fast-result-v1"
ORIGINAL_RESULT_SCHEMA = "m3-original-duration-result-v1"
TRIGGER_EVALUATION_SCHEMA = "m3-fast-trigger-evaluation-v1"
DECISION_SCHEMA = "m3-fast-decision-v1"
TRIGGER_SCHEMA = "m3-fast-escalation-trigger-v1"
PROTOCOL_VARIANTS = {
    "fast-strict-v1",
    "original-escalation-strict-v1",
    "f2-fast-cross-source-v1",
    "f2-original-cross-source-v1",
    "gap-fast-v1",
    "gap-original-v1",
    "negative-launch-scale-v1",
} | set(EXPERIMENT_ONLY_PROTOCOL_VARIANTS)
FAST_TO_ORIGINAL = {
    "fast-strict-v1": "original-escalation-strict-v1",
    "f2-fast-cross-source-v1": "f2-original-cross-source-v1",
    "gap-fast-v1": "gap-original-v1",
}

def _refuse_experiment_only_variant(protocol_variant: object) -> None:
    if protocol_variant in EXPERIMENT_ONLY_PROTOCOL_VARIANTS:
        raise ValueError(
            "experiment-only sweep protocol evidence is refused by family "
            "rehearsal, definitive, freeze, and finalize routes"
        )


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _leg_median(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError("leg median must be finite positive")
    return float(value)


def read_leg(record: object) -> dict[str, Any]:
    """Validate a protocol leg loaded from persisted or external evidence."""
    if not isinstance(record, dict):
        raise ValueError("protocol leg record must be an object")
    schema = record.get("schema_version")
    if schema not in {FAST_RESULT_SCHEMA, ORIGINAL_RESULT_SCHEMA}:
        raise ValueError(f"unknown protocol leg schema: {schema!r}")
    variant = record.get("protocol_variant")
    if variant not in PROTOCOL_VARIANTS:
        raise ValueError(f"unknown protocol variant: {variant!r}")
    fast = variant in FAST_TO_ORIGINAL
    if (schema == FAST_RESULT_SCHEMA) != fast:
        raise ValueError("protocol variant/schema duration mismatch")
    _leg_median(record.get("median_ratio_linked_over_reference"))
    _sha(record.get("eligibility_ledger_sha256"), "eligibility_ledger_sha256")
    _sha(record.get("result_sha256"), "result_sha256")
    unhashed = {key: record[key] for key in record if key != "result_sha256"}
    if _digest(unhashed) != record["result_sha256"]:
        raise ValueError("protocol leg result hash mismatch")
    if record.get("escalation_depth") not in {0, 1}:
        raise ValueError("escalation depth must be zero or one")
    return record


def seal_leg(
    *,
    protocol_variant: str,
    case_id: str,
    gpu: dict[str, Any],
    eligibility_ledger_sha256: str,
    median_ratio_linked_over_reference: float,
    result_path: str,
    escalation_depth: int = 0,
) -> dict[str, Any]:
    """Build and seal a valid leg for direct in-process use."""
    if protocol_variant not in PROTOCOL_VARIANTS:
        raise ValueError(f"unknown protocol variant: {protocol_variant!r}")
    if escalation_depth not in {0, 1}:
        raise ValueError("escalation depth must be zero or one")
    fast = protocol_variant in FAST_TO_ORIGINAL
    record = {
        "schema_version": FAST_RESULT_SCHEMA if fast else ORIGINAL_RESULT_SCHEMA,
        "protocol_variant": protocol_variant,
        "case_id": case_id,
        "gpu": gpu,
        "eligibility_ledger_sha256": _sha(eligibility_ledger_sha256, "eligibility_ledger_sha256"),
        "median_ratio_linked_over_reference": _leg_median(float(median_ratio_linked_over_reference)),
        "result_path": result_path,
        "escalation_depth": escalation_depth,
    }
    record["result_sha256"] = _digest(record)
    return record


def escalation_triggered(median: float) -> bool:
    if isinstance(median, bool) or not isinstance(median, (int, float)) or not math.isfinite(float(median)):
        raise ValueError("fast median must be finite")
    return ESCALATION_TRIGGER_LOW <= float(median) <= ESCALATION_TRIGGER_HIGH


def seal_trigger(fast: dict[str, Any]) -> dict[str, Any]:
    """Seal a trigger for a leg returned by seal_leg/read_leg, without rereading it."""
    _refuse_experiment_only_variant(fast["protocol_variant"])
    if fast["protocol_variant"] not in FAST_TO_ORIGINAL or fast["escalation_depth"] != 0:
        raise ValueError("trigger evaluation requires a depth-zero fast leg")
    record = {
        "schema_version": TRIGGER_EVALUATION_SCHEMA,
        "case_id": fast["case_id"],
        "fast_result_sha256": fast["result_sha256"],
        "fast_median": fast["median_ratio_linked_over_reference"],
        "inclusive_interval": [ESCALATION_TRIGGER_LOW, ESCALATION_TRIGGER_HIGH],
        "triggered": escalation_triggered(
            fast["median_ratio_linked_over_reference"]
        ),
        "evaluated_once": True,
        "gpu": fast["gpu"],
    }
    record["trigger_evaluation_sha256"] = _digest(record)
    return record


def read_trigger_evaluation(value: object) -> dict[str, Any]:
    """Validate a trigger evaluation loaded from persisted or external evidence."""
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != TRIGGER_EVALUATION_SCHEMA
    ):
        raise ValueError("unknown trigger-evaluation schema")
    claimed = _sha(
        value.get("trigger_evaluation_sha256"),
        "trigger_evaluation_sha256",
    )
    unhashed = {
        key: value[key] for key in value
        if key != "trigger_evaluation_sha256"
    }
    if _digest(unhashed) != claimed or value.get("evaluated_once") is not True:
        raise ValueError("trigger-evaluation seal mismatch")
    return value


def decide(
    fast: dict[str, Any],
    original: dict[str, Any] | None = None,
    *,
    trigger_evaluation: dict[str, Any] | None = None,
    direct_f2_audit_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decide using records already returned by the seal/read helpers."""
    _refuse_experiment_only_variant(fast["protocol_variant"])
    if fast["protocol_variant"] not in FAST_TO_ORIGINAL or fast["escalation_depth"] != 0:
        raise ValueError("decision requires one depth-zero fast screening leg")
    evaluation = (
        seal_trigger(fast)
        if trigger_evaluation is None
        else trigger_evaluation
    )
    if (
        evaluation["fast_result_sha256"] != fast["result_sha256"]
        or evaluation["fast_median"] != fast["median_ratio_linked_over_reference"]
        or evaluation["gpu"] != fast["gpu"]
    ):
        raise ValueError("trigger evaluation does not bind the fast leg")
    triggered = evaluation["triggered"]
    if original is not None:
        _refuse_experiment_only_variant(original["protocol_variant"])
    if triggered:
        if original is None:
            raise ValueError("borderline fast median requires an original-duration leg")
        expected = FAST_TO_ORIGINAL[fast["protocol_variant"]]
        if original["protocol_variant"] != expected:
            raise ValueError(f"consumer-specific original variant must be {expected}")
        if original["escalation_depth"] != 1:
            raise ValueError("original-duration leg must be the sole depth-one escalation")
        if original["case_id"] != fast["case_id"]:
            raise ValueError("fast/original case identity mismatch")
        for field in ("physical_index", "uuid", "pci_bus_id"):
            if original["gpu"].get(field) != fast["gpu"].get(field):
                raise ValueError("fast/original legs must stay on the same GPU")
        decision_source = "original-duration-escalation"
        decision_median = original["median_ratio_linked_over_reference"]
    else:
        if original is not None:
            raise ValueError("nonborderline fast result must not run an original-duration leg")
        decision_source = "fast"
        decision_median = fast["median_ratio_linked_over_reference"]
    trigger = {
        "schema_version": TRIGGER_SCHEMA,
        "case_id": fast["case_id"],
        "fast_result_sha256": fast["result_sha256"],
        "trigger_evaluation_sha256": evaluation[
            "trigger_evaluation_sha256"
        ],
        "inclusive_interval": evaluation["inclusive_interval"],
        "fast_median": evaluation["fast_median"],
        "triggered": triggered,
        "evaluated_once": True,
        "gpu": fast["gpu"],
        "original_result_sha256": original["result_sha256"] if original else None,
    }
    trigger["trigger_record_sha256"] = _digest(trigger)
    decision = {
        "schema_version": DECISION_SCHEMA,
        "case_id": fast["case_id"],
        "fast_protocol_variant": fast["protocol_variant"],
        "original_protocol_variant": original["protocol_variant"] if original else None,
        "fast_result_sha256": fast["result_sha256"],
        "original_result_sha256": original["result_sha256"] if original else None,
        "trigger_record_sha256": trigger["trigger_record_sha256"],
        "fast_median_ratio_linked_over_reference": fast["median_ratio_linked_over_reference"],
        "original_median_ratio_linked_over_reference": original["median_ratio_linked_over_reference"] if original else None,
        "decision_source": decision_source,
        "decision_median_ratio_linked_over_reference": decision_median,
        "pairs_pooled": False,
        "escalation_count": 1 if triggered else 0,
        "same_gpu": True,
        "direct_f2_audit_only": bool(direct_f2_audit_only),
    }
    decision["decision_sha256"] = _digest(decision)
    return trigger, decision


def read_decision(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != DECISION_SCHEMA:
        raise ValueError("unknown fast decision schema")
    claimed = _sha(value.get("decision_sha256"), "decision_sha256")
    unhashed = {key: value[key] for key in value if key != "decision_sha256"}
    if _digest(unhashed) != claimed:
        raise ValueError("fast decision hash mismatch")
    if value.get("decision_source") not in {"fast", "original-duration-escalation"}:
        raise ValueError("unknown decision_source")
    if value.get("pairs_pooled") is not False:
        raise ValueError("fast/original pairs must never be pooled")
    return value
