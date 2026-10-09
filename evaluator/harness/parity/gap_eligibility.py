"""Ratio-blind ex-ante solo eligibility and post-seal scoring for gap rows."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
import subprocess
from typing import Any

from .constants import (
    GAP_CLOCK_WINDOW_FRACTION,
    GAP_ROW_CASE_IDS,
    MIN_COVERING_SAMPLES_PER_SLOT,
    ORIGINAL_ALLOWED_REASON_MASK,
    SW_POWER_CAP_REASON_MASK,
)

POLICY_SCHEMA = "m3-parity-gap-policy-v1"
SOLO_SCHEMA = "m3-parity-gap-solo-v1"
ELIGIBILITY_INPUT_SCHEMA = "m3-parity-gap-eligibility-input-v3"
ELIGIBILITY_LEDGER_SCHEMA = "m3-parity-gap-eligibility-ledger-v3"
THROUGHPUT_SCHEMA = "m3-parity-gap-throughput-v1"
SCORE_SCHEMA = "m3-parity-gap-score-v1"
BRANCH_MASKS = {
    "original": {0x0, 0x1},
    "power-cap": {SW_POWER_CAP_REASON_MASK},
}
POLICY_FIELDS = {
    "gpu_uuid",
    "pci_bus_id",
    "configured_power_limit_w",
    "enforced_power_limit_w",
    "application_graphics_clock_mhz",
    "application_memory_clock_mhz",
    "default_graphics_clock_mhz",
    "default_memory_clock_mhz",
    "persistence_mode",
    "compute_mode",
}
SAMPLE_FIELDS = {
    "time_ns",
    "query_ok",
    "clock_mhz",
    "temperature_c",
    "reason_mask",
    "raw",
    "power_draw_w",
    "power_limit_w",
}


class GapSoloError(RuntimeError):
    """A required solo or bound-policy prerequisite failed the whole route."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _exact(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _reject_scoring_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if (
                "throughput" in lowered
                or lowered == "ratio"
                or lowered.startswith("ratio_")
                or lowered.endswith("_ratio")
                or "_ratio_" in lowered
            ):
                raise ValueError(
                    f"ratio-blind gap eligibility input contains scoring field {key!r}"
                )
            _reject_scoring_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_scoring_fields(child)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def query_gap_policy(gpu: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-gpu=uuid,pci.bus_id,power.limit,enforced.power.limit,"
        "clocks.applications.graphics,clocks.applications.memory,"
        "clocks.default_applications.graphics,clocks.default_applications.memory,"
        "persistence_mode,compute_mode",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode:
        raise GapSoloError(f"gap policy query failed: {completed.stderr.strip()}")
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 10:
        raise GapSoloError(f"gap policy query returned malformed row: {completed.stdout!r}")
    try:
        return {
            "gpu_uuid": fields[0],
            "pci_bus_id": fields[1],
            "configured_power_limit_w": float(fields[2]),
            "enforced_power_limit_w": float(fields[3]),
            "application_graphics_clock_mhz": int(fields[4]),
            "application_memory_clock_mhz": int(fields[5]),
            "default_graphics_clock_mhz": int(fields[6]),
            "default_memory_clock_mhz": int(fields[7]),
            "persistence_mode": fields[8],
            "compute_mode": fields[9],
        }
    except ValueError as exc:
        raise GapSoloError(f"gap policy query contains nonnumeric fields: {fields}") from exc


def freeze_policy(policy_tuple: object) -> dict[str, Any]:
    policy = _exact(policy_tuple, POLICY_FIELDS, "gap policy tuple")
    if any(
        not isinstance(policy[field], str) or not policy[field]
        for field in ("gpu_uuid", "pci_bus_id", "persistence_mode", "compute_mode")
    ):
        raise ValueError("gap policy identity/mode fields must be non-empty strings")
    for field in (
        "configured_power_limit_w",
        "enforced_power_limit_w",
        "application_graphics_clock_mhz",
        "application_memory_clock_mhz",
        "default_graphics_clock_mhz",
        "default_memory_clock_mhz",
    ):
        value = policy[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"gap policy {field} must be finite numeric")
    payload = {
        "schema_version": POLICY_SCHEMA,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "policy": dict(policy),
        "policy_sha256": digest(policy),
    }
    payload["policy_record_sha256"] = digest(payload)
    return read_policy(payload)


def read_policy(value: object) -> dict[str, Any]:
    fields = {
        "schema_version", "captured_utc", "policy", "policy_sha256",
        "policy_record_sha256",
    }
    record = _exact(value, fields, "gap policy record")
    if record["schema_version"] != POLICY_SCHEMA:
        raise ValueError("unknown gap policy schema")
    _exact(record["policy"], POLICY_FIELDS, "gap policy tuple")
    if digest(record["policy"]) != _require_sha256(record["policy_sha256"], "policy_sha256"):
        raise ValueError("gap policy tuple hash mismatch")
    unhashed = {key: record[key] for key in record if key != "policy_record_sha256"}
    if digest(unhashed) != _require_sha256(record["policy_record_sha256"], "policy_record_sha256"):
        raise ValueError("gap policy record hash mismatch")
    return record


def _sample(value: object, label: str) -> dict[str, Any]:
    row = _exact(value, SAMPLE_FIELDS, label)
    if not isinstance(row["time_ns"], int) or isinstance(row["time_ns"], bool):
        raise ValueError(f"{label}.time_ns must be integer")
    if not isinstance(row["query_ok"], bool):
        raise ValueError(f"{label}.query_ok must be boolean")
    if row["query_ok"]:
        for field in ("clock_mhz", "reason_mask"):
            if not isinstance(row[field], int) or isinstance(row[field], bool):
                raise ValueError(f"{label}.{field} must be integer")
        for field in ("power_draw_w", "power_limit_w"):
            if isinstance(row[field], bool) or not isinstance(row[field], (int, float)) or not math.isfinite(float(row[field])):
                raise ValueError(f"{label}.{field} must be finite")
    return row


def make_solo(
    *,
    case_id: str,
    role: str,
    image_id: str,
    image_sha256: str,
    policy_record: dict[str, Any],
    slot_start_ns: int,
    series_id: str,
    duration_leg: str,
    session_id: str,
    solo_block_id: str,
    eligibility_window_start_ns: int,
    eligibility_window_end_ns: int,
    cuda_scoring_window_start_ns: int,
    cuda_scoring_window_end_ns: int,
    slot_end_ns: int,
    samples: list[dict[str, Any]],
    leading_discard_ns: int = 5_000_000_000,
) -> dict[str, Any]:
    if case_id not in GAP_ROW_CASE_IDS:
        raise GapSoloError(f"case {case_id!r} is outside the closed gap-row list")
    policy = read_policy(policy_record)
    rows = [_sample(row, "solo sample") for row in samples]
    reasons: list[str] = []
    timing_identical = (
        eligibility_window_start_ns == cuda_scoring_window_start_ns
        and eligibility_window_end_ns == cuda_scoring_window_end_ns
        and eligibility_window_start_ns >= slot_start_ns + leading_discard_ns
        and eligibility_window_end_ns <= slot_end_ns
        and eligibility_window_end_ns > eligibility_window_start_ns
    )
    if not timing_identical:
        reasons.append("eligibility-cuda-scoring-window-mismatch")
    if len(rows) < MIN_COVERING_SAMPLES_PER_SLOT:
        reasons.append("uncovered-solo")
    if any(not row["query_ok"] for row in rows):
        reasons.append("solo-query-failure")
    if any(
        not (eligibility_window_start_ns <= row["time_ns"] <= eligibility_window_end_ns)
        for row in rows
    ):
        reasons.append("solo-sample-outside-window")
    good = [row for row in rows if row["query_ok"]]
    clocks = [float(row["clock_mhz"]) for row in good]
    median = float(statistics.median(clocks)) if clocks else None
    if median is None or any(
        abs(clock - median) > GAP_CLOCK_WINDOW_FRACTION * median for clock in clocks
    ):
        reasons.append("solo-clock-outside-own-3-percent")
    masks = {row["reason_mask"] for row in good}
    if masks and all((mask & ~ORIGINAL_ALLOWED_REASON_MASK) == 0 for mask in masks):
        branch = "original"
    elif masks == {SW_POWER_CAP_REASON_MASK}:
        branch = "power-cap"
    else:
        branch = "invalid"
        reasons.append("solo-mixed-other-or-composite-mask")
    bound_policy = policy["policy"]
    power_limits = {float(row["power_limit_w"]) for row in good}
    if (
        power_limits != {float(bound_policy["configured_power_limit_w"])}
        or float(bound_policy["configured_power_limit_w"])
        != float(bound_policy["enforced_power_limit_w"])
    ):
        reasons.append("solo-policy-power-limit-mismatch")
    if reasons:
        raise GapSoloError("gap solo failed: " + ", ".join(dict.fromkeys(reasons)))
    powers = [float(row["power_draw_w"]) for row in good]
    payload = {
        "schema_version": SOLO_SCHEMA,
        "case_id": case_id,
        "role": role,
        "image_id": image_id,
        "series_id": series_id,
        "duration_leg": duration_leg,
        "session_id": session_id,
        "solo_block_id": solo_block_id,
        "image_sha256": _require_sha256(image_sha256, "solo image_sha256"),
        "gpu_uuid": bound_policy["gpu_uuid"],
        "pci_bus_id": bound_policy["pci_bus_id"],
        "policy_sha256": policy["policy_sha256"],
        "leading_discard_ns": leading_discard_ns,
        "policy_record_sha256": policy["policy_record_sha256"],
        "assigned_branch": branch,
        "slot_start_ns": slot_start_ns,
        "eligibility_window_start_ns": eligibility_window_start_ns,
        "eligibility_window_end_ns": eligibility_window_end_ns,
        "cuda_scoring_window_start_ns": cuda_scoring_window_start_ns,
        "cuda_scoring_window_end_ns": cuda_scoring_window_end_ns,
        "slot_end_ns": slot_end_ns,
        "timestamp_coverage_identical": True,
        "median_clock_mhz": median,
        "clock_min_mhz": min(clocks),
        "clock_max_mhz": max(clocks),
        "configured_power_limit_w": bound_policy["configured_power_limit_w"],
        "power_draw_w": {
            "sample_count": len(powers),
            "p10": _percentile(powers, 0.10),
            "median": _percentile(powers, 0.50),
            "p90": _percentile(powers, 0.90),
        },
        "covering_sample_count": len(rows),
        "samples": rows,
    }
    payload["solo_sha256"] = digest(payload)
    return read_solo(payload)


def read_solo(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SOLO_SCHEMA:
        raise ValueError("unknown gap solo schema")
    claimed = _require_sha256(value.get("solo_sha256"), "solo_sha256")
    unhashed = {key: value[key] for key in value if key != "solo_sha256"}
    if digest(unhashed) != claimed:
        raise ValueError("gap solo hash mismatch")
    if value.get("assigned_branch") not in BRANCH_MASKS:
        raise ValueError("gap solo has invalid assigned branch")
    if value.get("timestamp_coverage_identical") is not True:
        raise ValueError("gap solo eligibility/scoring timestamps differ")
    return value


def _half(
    raw: object,
    solo: dict[str, Any],
    policy: dict[str, Any],
    label: str,
    leading_discard_ns: int,
    identity_gating: bool,
) -> dict[str, Any]:
    fields = {
        "role", "series_id", "duration_leg", "session_id", "solo_block_id",
        "image_id", "image_sha256", "gpu_uuid", "pci_bus_id",
        "policy_sha256", "policy_record_sha256", "solo_sha256",
        "slot_start_ns", "eligibility_window_start_ns",
        "eligibility_window_end_ns", "cuda_scoring_window_start_ns",
        "cuda_scoring_window_end_ns", "slot_end_ns", "samples",
    }
    half = _exact(raw, fields, label)
    reasons: list[str] = []
    for field in ("image_sha256", "policy_sha256", "policy_record_sha256", "solo_sha256"):
        _require_sha256(half[field], f"{label}.{field}")
    if half["image_id"] != solo["image_id"] or half["image_sha256"] != solo["image_sha256"]:
        reasons.append("image-identity-mismatch")
    if half["role"] != solo["role"]:
        reasons.append("role-mismatch")
    for field in ("series_id", "duration_leg", "session_id", "solo_block_id"):
        if half[field] != solo[field]:
            reasons.append(f"{field}-scope-mismatch")
    if half["solo_sha256"] != solo["solo_sha256"]:
        reasons.append("solo-hash-mismatch")
    if (
        half["policy_sha256"] != policy["policy_sha256"]
        or solo["policy_sha256"] != policy["policy_sha256"]
        or half["policy_record_sha256"] != policy["policy_record_sha256"]
    ):
        reasons.append("policy-hash-mismatch")
    bound = policy["policy"]
    if (
        half["gpu_uuid"] != bound["gpu_uuid"]
        or half["pci_bus_id"] != bound["pci_bus_id"]
        or solo["gpu_uuid"] != bound["gpu_uuid"]
        or solo["pci_bus_id"] != bound["pci_bus_id"]
    ):
        reasons.append("gpu-policy-identity-mismatch")
    timestamps = [
        half["slot_start_ns"], half["eligibility_window_start_ns"],
        half["eligibility_window_end_ns"], half["cuda_scoring_window_start_ns"],
        half["cuda_scoring_window_end_ns"], half["slot_end_ns"],
    ]
    if any(not isinstance(item, int) or isinstance(item, bool) for item in timestamps):
        raise ValueError(f"{label} timestamps must be integer")
    timing_valid = (
        half["eligibility_window_start_ns"] == half["cuda_scoring_window_start_ns"]
        and half["eligibility_window_end_ns"] == half["cuda_scoring_window_end_ns"]
        and half["eligibility_window_start_ns"] >= half["slot_start_ns"] + leading_discard_ns
        and half["eligibility_window_end_ns"] <= half["slot_end_ns"]
        and half["eligibility_window_end_ns"] > half["eligibility_window_start_ns"]
    )
    if not timing_valid:
        reasons.append("eligibility-cuda-scoring-window-mismatch")
    rows = [_sample(row, label + ".sample") for row in half["samples"]]
    if len(rows) < MIN_COVERING_SAMPLES_PER_SLOT:
        reasons.append("uncovered")
    if any(not row["query_ok"] for row in rows):
        reasons.append("query-failure")
    if any(
        not (
            half["eligibility_window_start_ns"]
            <= row["time_ns"]
            <= half["eligibility_window_end_ns"]
        )
        for row in rows
    ):
        reasons.append("sample-outside-eligibility-window")
    good = [row for row in rows if row["query_ok"]]
    clocks = [float(row["clock_mhz"]) for row in good]
    median = float(statistics.median(clocks)) if clocks else None
    clock_valid = median is not None and all(
        abs(clock - median) <= GAP_CLOCK_WINDOW_FRACTION * median for clock in clocks
    )
    if not clock_valid:
        reasons.append("clock-outside-own-3-percent")
    masks = {row["reason_mask"] for row in good}
    if masks and all((mask & ~ORIGINAL_ALLOWED_REASON_MASK) == 0 for mask in masks):
        observed_branch = "original"
    elif masks == {SW_POWER_CAP_REASON_MASK}:
        observed_branch = "power-cap"
    else:
        observed_branch = "invalid"
    preassigned_branch = solo["assigned_branch"]
    branch_valid = observed_branch == preassigned_branch
    if not branch_valid:
        reasons.append("within-half-branch-deviation")
    solo_median = float(solo["median_clock_mhz"])
    identity_delta_fraction = (
        None if median is None else (median - solo_median) / solo_median
    )
    identity_valid = (
        identity_delta_fraction is not None
        and abs(identity_delta_fraction) <= 0.01
    )
    if not identity_valid and identity_gating:
        reasons.append("solo-clock-identity-mismatch")
    power_limits = {float(row["power_limit_w"]) for row in good}
    policy_valid = (
        power_limits == {float(bound["configured_power_limit_w"])}
        and float(bound["configured_power_limit_w"])
        == float(bound["enforced_power_limit_w"])
        and "policy-hash-mismatch" not in reasons
        and "gpu-policy-identity-mismatch" not in reasons
    )
    if not policy_valid:
        reasons.append("policy-mismatch")
    powers = [float(row["power_draw_w"]) for row in good]
    reasons = list(dict.fromkeys(reasons))
    return {
        "valid": not reasons,
        "reasons": reasons,
        "preassigned_branch": preassigned_branch,
        "observed_branch": observed_branch,
        "branch_valid": branch_valid,
        "clock_valid": clock_valid,
        "identity_valid": identity_valid,
        "identity_gating": identity_gating,
        "policy_valid": policy_valid,
        "timing_valid": timing_valid,
        "timestamp_coverage_identical": timing_valid,
        "solo_sha256": solo["solo_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "policy_record_sha256": policy["policy_record_sha256"],
        "solo_median_clock_mhz": solo_median,
        "paired_median_clock_mhz": median,
        "identity_delta_fraction": identity_delta_fraction,
        "identity_delta_percent": (
            None if identity_delta_fraction is None else 100.0 * identity_delta_fraction
        ),
        "clock_min_mhz": min(clocks) if clocks else None,
        "clock_max_mhz": max(clocks) if clocks else None,
        "observed_masks": sorted(masks),
        "power_draw_w": {
            "sample_count": len(powers),
            "p10": _percentile(powers, 0.10),
            "median": _percentile(powers, 0.50),
            "p90": _percentile(powers, 0.90),
        },
        "configured_power_limit_w": bound["configured_power_limit_w"],
        "eligibility_window_start_ns": half["eligibility_window_start_ns"],
        "eligibility_window_end_ns": half["eligibility_window_end_ns"],
        "cuda_scoring_window_start_ns": half["cuda_scoring_window_start_ns"],
        "cuda_scoring_window_end_ns": half["cuda_scoring_window_end_ns"],
        "covering_sample_count": len(rows),
    }


def evaluate_eligibility(value: object) -> dict[str, Any]:
    _reject_scoring_fields(value)
    fields = {
        "schema_version", "case_id", "gpu_uuid", "pci_bus_id",
        "policy_before", "policy_frozen", "policy_after", "target_pairs",
        "attempt_cap", "leading_discard_ns", "identity_gating",
        "solo_blocks", "attempts",
    }
    source = _exact(value, fields, "gap eligibility input")
    if source["schema_version"] != ELIGIBILITY_INPUT_SCHEMA:
        raise ValueError("unknown gap eligibility input schema")
    if source["case_id"] not in GAP_ROW_CASE_IDS:
        raise ValueError("gap eligibility case is outside the closed allowlist")
    policies = {
        name: read_policy(source[name])
        for name in ("policy_before", "policy_frozen", "policy_after")
    }
    policy = policies["policy_frozen"]
    if any(record["policy"] != policy["policy"] for record in policies.values()):
        raise GapSoloError("gap policy changed during solo/paired epoch")
    if source["gpu_uuid"] != policy["policy"]["gpu_uuid"] or source["pci_bus_id"] != policy["policy"]["pci_bus_id"]:
        raise ValueError("gap eligibility GPU identity differs from policy")
    blocks_raw = source["solo_blocks"]
    if not isinstance(blocks_raw, dict) or not blocks_raw:
        raise ValueError("gap eligibility requires sealed solo blocks")
    solo_blocks = {
        block_id: {
            role: read_solo(solos[role])
            for role in ("reference", "linked")
        }
        for block_id, solos in blocks_raw.items()
    }
    identity_gating = source["identity_gating"]
    if not isinstance(identity_gating, bool):
        raise ValueError("identity_gating must be boolean")
    target = source["target_pairs"]
    cap = source["attempt_cap"]
    if not isinstance(target, int) or target <= 0 or not isinstance(cap, int) or cap < target:
        raise ValueError("invalid gap eligibility target/cap")
    leading_discard_ns = source["leading_discard_ns"]
    if (
        not isinstance(leading_discard_ns, int)
        or isinstance(leading_discard_ns, bool)
        or leading_discard_ns <= 0
    ):
        raise ValueError("gap leading_discard_ns must be positive integer")
    attempts = source["attempts"]
    if not isinstance(attempts, list) or len(attempts) > cap:
        raise ValueError("gap attempt list exceeds cap")
    rows: list[dict[str, Any]] = []
    selected: list[int] = []
    for expected, raw in enumerate(attempts):
        attempt = _exact(
            raw,
            {"attempt", "order", "solo_block_id", "halves", "strict_classification", "superseded_same_branch_classification"},
            f"gap attempt {expected}",
        )
        if attempt["attempt"] != expected:
            raise ValueError("gap attempts must be contiguous chronological order")
        expected_order = ["reference", "linked"] if expected % 2 == 0 else ["linked", "reference"]
        if attempt["order"] != expected_order:
            raise ValueError("deterministic gap pair order drift")
        block_id = attempt["solo_block_id"]
        if block_id not in solo_blocks:
            raise ValueError("attempt references unknown solo block")
        solos = solo_blocks[block_id]
        raw_halves = _exact(attempt["halves"], {"reference", "linked"}, "gap halves")
        halves = {
            role: _half(
                raw_halves[role], solos[role], policy,
                f"attempt {expected}.{role}", leading_discard_ns,
                identity_gating,
            )
            for role in ("reference", "linked")
        }
        valid = halves["reference"]["valid"] and halves["linked"]["valid"]
        cross_branch = (
            halves["reference"]["preassigned_branch"]
            != halves["linked"]["preassigned_branch"]
        )
        reasons = [
            f"{role}:{reason}"
            for role in ("reference", "linked")
            for reason in halves[role]["reasons"]
        ]
        reference_median = halves["reference"]["paired_median_clock_mhz"]
        linked_median = halves["linked"]["paired_median_clock_mhz"]
        rows.append({
            "attempt": expected,
            "solo_block_id": block_id,
            "order": expected_order,
            "gap_pair_valid": valid,
            "reasons": reasons,
            "halves": halves,
            "cross_branch": cross_branch,
            "strict_classification": attempt["strict_classification"],
            "superseded_same_branch_classification": attempt["superseded_same_branch_classification"],
            "clock_median_delta_linked_minus_reference_mhz": (
                None if reference_median is None or linked_median is None
                else linked_median - reference_median
            ),
        })
        if valid and len(selected) < target:
            selected.append(expected)
    payload = {
        "schema_version": ELIGIBILITY_LEDGER_SCHEMA,
        "case_id": source["case_id"],
        "gpu_uuid": source["gpu_uuid"],
        "pci_bus_id": source["pci_bus_id"],
        "policy_sha256": policy["policy_sha256"],
        "policy_record_sha256": policy["policy_record_sha256"],
        "policy_before_record_sha256": policies["policy_before"]["policy_record_sha256"],
        "policy_after_record_sha256": policies["policy_after"]["policy_record_sha256"],
        "solo_block_sha256": {
            block_id: {
                role: solo["solo_sha256"] for role, solo in block.items()
            }
            for block_id, block in solo_blocks.items()
        },
        "identity_gating": identity_gating,
        "target_pairs": target,
        "attempt_cap": cap,
        "leading_discard_ns": leading_discard_ns,
        "attempts": rows,
        "selected_attempts": selected,
        "target_met": len(selected) == target,
        "ratio_blind": True,
    }
    payload["eligibility_ledger_sha256"] = digest(payload)
    return read_eligibility_ledger(payload)


def read_eligibility_ledger(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != ELIGIBILITY_LEDGER_SCHEMA:
        raise ValueError("unknown gap eligibility-ledger schema")
    claimed = _require_sha256(value.get("eligibility_ledger_sha256"), "eligibility_ledger_sha256")
    unhashed = {key: value[key] for key in value if key != "eligibility_ledger_sha256"}
    if digest(unhashed) != claimed:
        raise ValueError("gap eligibility ledger hash mismatch")
    if value.get("ratio_blind") is not True:
        raise ValueError("gap eligibility ledger is not ratio blind")
    return value


def score(ledger_value: object, throughput_value: object) -> dict[str, Any]:
    ledger = read_eligibility_ledger(ledger_value)
    fields = {"schema_version", "case_id", "eligibility_ledger_sha256", "attempts"}
    throughputs = _exact(throughput_value, fields, "gap throughput input")
    if throughputs["schema_version"] != THROUGHPUT_SCHEMA:
        raise ValueError("unknown gap throughput schema")
    if throughputs["case_id"] != ledger["case_id"] or throughputs["eligibility_ledger_sha256"] != ledger["eligibility_ledger_sha256"]:
        raise ValueError("gap throughput/eligibility identity mismatch")
    by_attempt: dict[int, dict[str, Any]] = {}
    for raw in throughputs["attempts"]:
        row = _exact(raw, {"attempt", "reference_throughput", "linked_throughput"}, "gap throughput attempt")
        if row["attempt"] in by_attempt:
            raise ValueError("duplicate gap throughput attempt")
        for field in ("reference_throughput", "linked_throughput"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError("gap throughput must be finite positive")
        by_attempt[row["attempt"]] = row
    if not ledger["target_met"]:
        raise ValueError("gap eligibility target was not met")
    selected = ledger["selected_attempts"]
    if any(attempt not in by_attempt for attempt in selected):
        raise ValueError("selected gap throughput is missing")
    ratios = [
        float(by_attempt[attempt]["linked_throughput"])
        / float(by_attempt[attempt]["reference_throughput"])
        for attempt in selected
    ]
    result = {
        "schema_version": SCORE_SCHEMA,
        "case_id": ledger["case_id"],
        "eligibility_ledger_sha256": ledger["eligibility_ledger_sha256"],
        "selected_attempts": selected,
        "gap_pair_valid_ratios_linked_over_reference": ratios,
        "median_ratio_linked_over_reference": float(statistics.median(ratios)),
        "estimand": "realized throughput ratio under the observed operating points",
        "iso_frequency_claim": False,
    }
    result["score_sha256"] = digest(result)
    return read_score(result)


def read_score(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SCORE_SCHEMA:
        raise ValueError("unknown gap score schema")
    claimed = _require_sha256(value.get("score_sha256"), "score_sha256")
    unhashed = {key: value[key] for key in value if key != "score_sha256"}
    if digest(unhashed) != claimed:
        raise ValueError("gap score hash mismatch")
    return value
