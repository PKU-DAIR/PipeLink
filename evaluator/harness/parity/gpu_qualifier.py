"""Pure telemetry-only load-soak qualification predicates and record reader."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any

QUALIFIER_SCHEMA = "fa4-f2-00-telemetry-qualification-v2"
QUALIFIER_RESULT_SCHEMA = "m3-gpu-telemetry-qualification-result-v2"




def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _reject_scoring(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if "throughput" in lowered or lowered == "ratio" or "_ratio_" in lowered:
                raise ValueError("telemetry qualifier contains forbidden scoring field")
            _reject_scoring(child)
    elif isinstance(value, list):
        for child in value:
            _reject_scoring(child)


def _window(samples: object, label: str) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(samples, list) or len(samples) < 2:
        samples = samples if isinstance(samples, list) else []
        reasons.append(f"{label}-coverage-failure")
    if any(not isinstance(row, dict) or row.get("query_ok") is not True for row in samples):
        reasons.append(f"{label}-query-failure")
    good = [row for row in samples if isinstance(row, dict) and row.get("query_ok") is True]
    clocks = [float(row["clock_mhz"]) for row in good]
    median = float(statistics.median(clocks)) if clocks else None
    if median is None or any(not math.isfinite(clock) or abs(clock - median) > 0.03 * median for clock in clocks):
        reasons.append(f"{label}-clock-window-failure")
    masks = [row.get("reason_mask") for row in good]
    if masks and all(isinstance(mask, int) and (mask & ~0x1) == 0 for mask in masks):
        branch = "original"
    elif masks and all(mask == 0x4 for mask in masks):
        branch = "power-cap"
    else:
        branch = "invalid"
        reasons.append(f"{label}-mask-vocabulary-failure")
    limits = {float(row["power_limit_w"]) for row in good}
    if len(limits) != 1:
        reasons.append(f"{label}-power-limit-failure")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "branch": branch,
        "median_clock_mhz": median,
        "clock_min_mhz": min(clocks) if clocks else None,
        "clock_max_mhz": max(clocks) if clocks else None,
        "mask_set": sorted(set(masks), key=lambda value: -1 if value is None else value),
        "sample_count": len(samples),
        "samples": samples,
    }


def evaluate_qualification(value: object) -> dict[str, Any]:
    _reject_scoring(value)
    if not isinstance(value, dict) or value.get("schema_version") != QUALIFIER_SCHEMA:
        raise ValueError("unknown FA4 load-soak qualifier schema")
    required = {
        "schema_version", "epoch_id", "physical_index", "gpu_uuid", "pci_bus_id",
        "policy_sha256", "baseline_dispatch_sha256", "jit_identity_sha256",
        "input_sha256", "stream_map_sha256", "settled_start_ns", "w1_start_ns",
        "w1_end_ns", "soak_start_ns", "soak_end_ns", "w2_start_ns", "w2_end_ns",
        "soak_continuous", "w1_samples", "w2_samples",
    }
    if set(value) != required:
        raise ValueError("FA4 load-soak qualifier fields are not exact")
    timing = (
        value["settled_start_ns"] <= value["w1_start_ns"] < value["w1_end_ns"]
        == value["soak_start_ns"] < value["soak_end_ns"]
        == value["w2_start_ns"] < value["w2_end_ns"]
        and value["soak_continuous"] is True
    )
    w1 = _window(value["w1_samples"], "w1")
    w2 = _window(value["w2_samples"], "w2")
    reasons = [*w1["reasons"], *w2["reasons"]]
    if not timing:
        reasons.append("load-soak-timing-or-continuity-failure")
    same_branch = w1["branch"] == w2["branch"] and w1["branch"] in {"original", "power-cap"}
    if not same_branch:
        reasons.append("load-soak-branch-drift")
    drift = (
        None
        if w1["median_clock_mhz"] is None or w2["median_clock_mhz"] is None
        else w2["median_clock_mhz"] / w1["median_clock_mhz"] - 1.0
    )
    drift_valid = (
        drift is not None
        and abs(w2["median_clock_mhz"] - w1["median_clock_mhz"])
        <= 0.01 * w1["median_clock_mhz"]
    )
    if not drift_valid:
        reasons.append("load-soak-clock-drift-over-1-percent")
    reasons = list(dict.fromkeys(reasons))
    result = {
        "schema_version": QUALIFIER_RESULT_SCHEMA,
        "qualifier_version": QUALIFIER_SCHEMA,
        "epoch_id": value["epoch_id"],
        "physical_index": value["physical_index"],
        "gpu_uuid": value["gpu_uuid"],
        "pci_bus_id": value["pci_bus_id"],
        "status": "PASS" if not reasons else "INELIGIBLE_TELEMETRY_STATE",
        "reasons": reasons,
        "w1": w1,
        "w2": w2,
        "same_branch": same_branch,
        "median_drift_fraction_w2_over_w1": drift,
        "drift_within_1_percent": drift_valid,
        "soak_continuous": value["soak_continuous"],
        "attempt_count": 1,
        "retry_permitted_same_epoch": False,
        "telemetry_only": True,
        "nominal_sustained_seconds": 80,
        "policy_sha256": value["policy_sha256"],
        "baseline_dispatch_sha256": value["baseline_dispatch_sha256"],
        "jit_identity_sha256": value["jit_identity_sha256"],
        "input_sha256": value["input_sha256"],
        "stream_map_sha256": value["stream_map_sha256"],
    }
    result["qualification_result_sha256"] = _digest(result)
    return read_qualification(result)


def read_qualification(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != QUALIFIER_RESULT_SCHEMA:
        raise ValueError("unknown load-soak qualification-result schema")
    if value.get("qualifier_version") != QUALIFIER_SCHEMA:
        raise ValueError("qualifier v1/v2 dispatch mismatch")
    _reject_scoring(value)
    claimed = value.get("qualification_result_sha256")
    unhashed = {key: value[key] for key in value if key != "qualification_result_sha256"}
    if not isinstance(claimed, str) or _digest(unhashed) != claimed:
        raise ValueError("load-soak qualification result hash mismatch")
    return value
