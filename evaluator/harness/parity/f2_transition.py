"""F2 fast cross-source discarded-transition clock/mask classifier v2."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any

CLASSIFIER = "f2-cross-source-fast-transition-v2"
RESULT_SCHEMA = "m3-f2-fast-transition-result-v2"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate_f2_fast_transition(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != CLASSIFIER:
        raise ValueError("unknown F2 fast transition classifier version")
    fields = {
        "schema_version", "preassigned_branch", "transition_start_ns",
        "transition_end_ns", "scored_start_ns", "scored_end_ns",
        "transition_samples", "scored_samples",
    }
    if set(value) != fields:
        raise ValueError("F2 fast transition input fields are not exact")
    branch = value["preassigned_branch"]
    if branch not in {"original", "power-cap"}:
        raise ValueError("unknown F2 preassigned branch")
    transition = value["transition_samples"]
    scored = value["scored_samples"]
    reasons: list[str] = []
    if not isinstance(transition, list) or len(transition) < 2:
        transition = transition if isinstance(transition, list) else []
        reasons.append("transition-coverage-failure")
    if not isinstance(scored, list) or len(scored) < 2:
        scored = scored if isinstance(scored, list) else []
        reasons.append("scored-coverage-failure")
    if any(row.get("query_ok") is not True for row in transition + scored):
        reasons.append("query-failure")
    timing_valid = (
        isinstance(value["transition_start_ns"], int)
        and isinstance(value["transition_end_ns"], int)
        and isinstance(value["scored_start_ns"], int)
        and isinstance(value["scored_end_ns"], int)
        and value["transition_start_ns"] < value["transition_end_ns"]
        <= value["scored_start_ns"] < value["scored_end_ns"]
        and all(
            value["transition_start_ns"] <= row["time_ns"] < value["transition_end_ns"]
            for row in transition
        )
        and all(
            value["scored_start_ns"] <= row["time_ns"] <= value["scored_end_ns"]
            for row in scored
        )
    )
    if not timing_valid:
        reasons.append("transition-scored-timestamp-overlap-or-gap")
    good_scored = [row for row in scored if row.get("query_ok") is True]
    clocks = [float(row["clock_mhz"]) for row in good_scored]
    median = float(statistics.median(clocks)) if clocks else None
    lower = median * 0.97 if median is not None else None
    upper = median * 1.03 if median is not None else None
    scored_clock_valid = median is not None and all(
        lower <= clock <= upper and math.isfinite(clock) for clock in clocks
    )
    if not scored_clock_valid:
        reasons.append("scored-clock-outside-own-3-percent")
    transition_in_band = [
        row.get("query_ok") is True
        and median is not None
        and lower <= float(row["clock_mhz"]) <= upper
        for row in transition
    ]
    first_contact = (
        transition_in_band.index(True) if any(transition_in_band) else None
    )
    clock_contact_valid = first_contact is not None
    if not clock_contact_valid:
        reasons.append("transition-clock-band-contact-missing")
    exits: list[int] = []
    reentries: list[int] = []
    if first_contact is not None:
        previous = True
        for index in range(first_contact + 1, len(transition_in_band)):
            current = transition_in_band[index]
            if previous and not current:
                exits.append(index)
            if not previous and current:
                reentries.append(index)
            previous = current
    scored_masks = [row.get("reason_mask") for row in good_scored]
    if branch == "original":
        scored_branch_valid = all(
            isinstance(mask, int) and (mask & ~0x1) == 0
            for mask in scored_masks
        )
    else:
        scored_branch_valid = all(mask == 0x4 for mask in scored_masks)
    if not scored_branch_valid:
        reasons.append("scored-mask-branch-mismatch")
    transition_masks = [
        row.get("reason_mask") for row in transition
        if row.get("query_ok") is True
    ]
    mask_entry = None
    mask_irreversible = True
    if branch == "original":
        mask_irreversible = all(
            isinstance(mask, int) and (mask & ~0x1) == 0
            for mask in transition_masks
        )
    else:
        invalid_vocabulary = any(mask not in {0x0, 0x1, 0x4} for mask in transition_masks)
        if invalid_vocabulary or 0x4 not in transition_masks:
            mask_irreversible = False
        else:
            mask_entry = transition_masks.index(0x4)
            if any((mask & ~0x1) != 0 for mask in transition_masks[:mask_entry]):
                mask_irreversible = False
            if any(mask != 0x4 for mask in transition_masks[mask_entry:]):
                mask_irreversible = False
    if not mask_irreversible:
        reasons.append("transition-mask-one-way-entry-failed")
    final_clock = (
        float(transition[-1]["clock_mhz"])
        if transition and transition[-1].get("query_ok") is True else None
    )
    if final_clock is None or lower is None or upper is None:
        final_signed_distance = None
    elif final_clock < lower:
        final_signed_distance = final_clock - lower
    elif final_clock > upper:
        final_signed_distance = final_clock - upper
    else:
        final_signed_distance = 0.0
    reasons = list(dict.fromkeys(reasons))
    result = {
        "schema_version": RESULT_SCHEMA,
        "classifier_version": CLASSIFIER,
        "valid": not reasons,
        "reasons": reasons,
        "preassigned_branch": branch,
        "scored_median_clock_mhz": median,
        "scored_band_mhz": [lower, upper],
        "scored_clock_valid": scored_clock_valid,
        "scored_branch_valid": scored_branch_valid,
        "clock_contact_valid": clock_contact_valid,
        "first_clock_contact_index": first_contact,
        "first_clock_contact_time_ns": (
            transition[first_contact]["time_ns"] if first_contact is not None else None
        ),
        "transition_in_band_count": sum(transition_in_band),
        "transition_out_of_band_count": len(transition_in_band) - sum(transition_in_band),
        "clock_exit_indices": exits,
        "clock_exit_times_ns": [transition[index]["time_ns"] for index in exits],
        "clock_reentry_indices": reentries,
        "clock_reentry_times_ns": [transition[index]["time_ns"] for index in reentries],
        "final_transition_clock_mhz": final_clock,
        "final_transition_signed_distance_from_band_mhz": final_signed_distance,
        "mask_branch_entry_index": mask_entry,
        "mask_branch_entry_time_ns": (
            transition[mask_entry]["time_ns"] if mask_entry is not None else None
        ),
        "mask_irreversible_entry_valid": mask_irreversible,
        "timing_nonoverlap_valid": timing_valid,
        "transition_samples": transition,
        "scored_samples": scored,
    }
    result["result_sha256"] = _digest(result)
    return read_f2_fast_transition(result)


def read_f2_fast_transition(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("unknown F2 fast transition result schema")
    if value.get("classifier_version") != CLASSIFIER:
        raise ValueError("F2 transition v1/v2 mismatch")
    if "clock_irreversible_entry_valid" in value:
        raise ValueError("superseded clock irreversible decision field is forbidden")
    claimed = value.get("result_sha256")
    unhashed = {key: value[key] for key in value if key != "result_sha256"}
    if not isinstance(claimed, str) or _digest(unhashed) != claimed:
        raise ValueError("F2 transition result hash mismatch")
    return value
