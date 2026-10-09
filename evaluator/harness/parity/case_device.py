"""Pre-pair all-role case/device qualification and deterministic fallback."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

from .constants import GAP_ROW_CASE_IDS

F2_SCHEMA_V1 = "f2-case-device-branch-qualification-v1"
F2_SCHEMA_V2 = "f2-case-device-branch-qualification-v2"
F2_SCHEMA_V3 = "f2-case-device-branch-qualification-v3"
F2_SCHEMA = F2_SCHEMA_V3
GAP_SCHEMA = "gap-case-device-branch-qualification-v1"
RESULT_SCHEMA_V1 = "m3-case-device-branch-qualification-result-v1"
RESULT_SCHEMA_V2 = "m3-case-device-branch-qualification-result-v2"
RESULT_SCHEMA_V3 = "m3-case-device-branch-qualification-result-v3"
RESULT_SCHEMA = RESULT_SCHEMA_V3
STRICT_OWN_CLOCK_CLASSIFIER_V1 = "strict-own-clock-all-samples-v1"
F2_SOLO_ANCHOR_CLASSIFIER_V2 = "f2-solo-anchor-percentile-envelope-v2"
F2_SOLO_ANCHOR_CLASSIFIER_V3 = "f2-solo-anchor-isolated-tail-v3"
STRICT_CLOCK_RESULT_SCHEMA_V1 = "m3-strict-own-clock-result-v1"
F2_ANCHOR_RESULT_SCHEMA_V2 = "m3-f2-solo-anchor-result-v2"
F2_ANCHOR_RESULT_SCHEMA_V3 = "m3-f2-solo-anchor-result-v3"
PASS = "PASS"
INELIGIBLE = "CASE_DEVICE_INELIGIBLE_TELEMETRY"
FAIL = "CASE_DEVICE_QUALIFICATION_FAIL"
ROLE_CLASSES = {
    "stable-original",
    "stable-power-cap",
    "mixed-cap-boundary",
    "fatal",
}
F2_CASE_IDS = frozenset(f"F2-{index:02d}" for index in range(18))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _reject_scoring(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if "throughput" in lowered or lowered == "ratio" or "_ratio_" in lowered:
                raise ValueError("case/device qualification is telemetry-only")
            _reject_scoring(child)
    elif isinstance(value, list):
        for child in value:
            _reject_scoring(child)


def _sealed_clock_result(
    schema_version: str,
    classifier_version: str,
    valid: bool,
    reasons: list[str],
    statistics_record: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": schema_version,
        "classifier_version": classifier_version,
        "valid": bool(valid),
        "reasons": reasons,
        **statistics_record,
    }
    result["result_sha256"] = _digest(result)
    return result


def _clock_classifications(clocks: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    finite = bool(clocks) and all(
        math.isfinite(clock) and clock > 0.0 for clock in clocks
    )
    ordered = sorted(clocks) if finite else []
    median = float(statistics.median(ordered)) if ordered else None
    n = len(clocks)
    if ordered:
        q05 = ordered[math.ceil(0.05 * n) - 1]
        q95 = ordered[math.ceil(0.95 * n) - 1]
        minimum = ordered[0]
        maximum = ordered[-1]
        outside_3 = sum(
            abs(clock - median) > 0.03 * median for clock in ordered
        )
        outside_5 = sum(
            abs(clock - median) > 0.05 * median for clock in ordered
        )
    else:
        q05 = q95 = minimum = maximum = None
        outside_3 = outside_5 = n
    statistics_record = {
        "n": n,
        "minimum_mhz": minimum,
        "q05_mhz": q05,
        "median_mhz": median,
        "q95_mhz": q95,
        "maximum_mhz": maximum,
        "count_outside_3_percent": outside_3,
        "count_outside_5_percent": outside_5,
    }
    strict_valid = finite and outside_3 == 0
    strict = _sealed_clock_result(
        STRICT_CLOCK_RESULT_SCHEMA_V1,
        STRICT_OWN_CLOCK_CLASSIFIER_V1,
        strict_valid,
        [] if strict_valid else ["own-clock-window-failure"],
        statistics_record,
    )
    percentile_valid = bool(
        finite
        and abs(q05 - median) <= 0.03 * median
        and abs(q95 - median) <= 0.03 * median
    )
    tail_valid = finite and outside_5 == 0
    anchor_reasons = []
    if not finite:
        anchor_reasons.append(
            "f2-solo-anchor-nonfinite-or-nonpositive-clock"
        )
    if finite and not percentile_valid:
        anchor_reasons.append("f2-solo-anchor-percentile-envelope-failure")
    if finite and not tail_valid:
        anchor_reasons.append("f2-solo-anchor-tail-envelope-failure")
    anchor = _sealed_clock_result(
        F2_ANCHOR_RESULT_SCHEMA_V2,
        F2_SOLO_ANCHOR_CLASSIFIER_V2,
        percentile_valid and tail_valid,
        anchor_reasons,
        statistics_record,
    )
    return strict, anchor
def _read_clock_result(
    value: object,
    *,
    schema_version: str,
    classifier_version: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("clock classification result must be an object")
    expected = {
        "schema_version", "classifier_version", "valid", "reasons", "n",
        "minimum_mhz", "q05_mhz", "median_mhz", "q95_mhz",
        "maximum_mhz", "count_outside_3_percent",
        "count_outside_5_percent", "result_sha256",
    }
    if set(value) != expected:
        raise ValueError("clock classification result fields drift")
    if (
        value["schema_version"] != schema_version
        or value["classifier_version"] != classifier_version
    ):
        raise ValueError("unknown clock classification schema/classifier")
    if not isinstance(value["valid"], bool) or not isinstance(
        value["reasons"], list
    ):
        raise ValueError("clock classification validity/reasons malformed")
    for field in (
        "n", "count_outside_3_percent", "count_outside_5_percent"
    ):
        if (
            not isinstance(value[field], int)
            or isinstance(value[field], bool)
            or value[field] < 0
        ):
            raise ValueError(f"clock classification {field} malformed")
    claimed = value["result_sha256"]
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or _digest({
            key: item for key, item in value.items()
            if key != "result_sha256"
        }) != claimed
    ):
        raise ValueError("clock classification hash mismatch")
    return value


def read_strict_own_clock(value: object) -> dict[str, Any]:
    return _read_clock_result(
        value,
        schema_version=STRICT_CLOCK_RESULT_SCHEMA_V1,
        classifier_version=STRICT_OWN_CLOCK_CLASSIFIER_V1,
    )


def read_f2_solo_anchor(value: object) -> dict[str, Any]:
    return _read_clock_result(
        value,
        schema_version=F2_ANCHOR_RESULT_SCHEMA_V2,
        classifier_version=F2_SOLO_ANCHOR_CLASSIFIER_V2,
    )


def evaluate_f2_solo_anchor(clocks_mhz: object) -> dict[str, Any]:
    if not isinstance(clocks_mhz, list):
        raise ValueError("F2 solo anchor clocks must be a list")
    if any(
        not isinstance(clock, (int, float)) or isinstance(clock, bool)
        for clock in clocks_mhz
    ):
        raise ValueError("F2 solo anchor clocks must be numeric")
    _strict, anchor = _clock_classifications(
        [float(clock) for clock in clocks_mhz]
    )
    return read_f2_solo_anchor(anchor)


def evaluate_strict_own_clock(clocks_mhz: object) -> dict[str, Any]:
    if not isinstance(clocks_mhz, list):
        raise ValueError("strict own-clock clocks must be a list")
    if any(
        not isinstance(clock, (int, float)) or isinstance(clock, bool)
        for clock in clocks_mhz
    ):
        raise ValueError("strict own-clock clocks must be numeric")
    strict, _anchor = _clock_classifications(
        [float(clock) for clock in clocks_mhz]
    )
    return read_strict_own_clock(strict)

def _clock_statistics(clocks: list[float]) -> dict[str, Any]:
    strict, _anchor = _clock_classifications(clocks)
    return {
        key: strict[key]
        for key in (
            "n", "minimum_mhz", "q05_mhz", "median_mhz", "q95_mhz",
            "maximum_mhz", "count_outside_3_percent",
            "count_outside_5_percent",
        )
    }


def _sample_proof(
    samples: list[dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    return {
        "index_zero_based": index,
        "chronological_index_one_based": index + 1,
        "time_ns": samples[index]["time_ns"],
        "clock_mhz": samples[index]["clock_mhz"],
    }


def evaluate_f2_solo_anchor_v3(samples: object) -> dict[str, Any]:
    if not isinstance(samples, list):
        raise ValueError("F2 v3 solo anchor samples must be a list")
    normalized: list[dict[str, Any]] = []
    for sample in samples:
        if (
            not isinstance(sample, dict)
            or not isinstance(sample.get("time_ns"), int)
            or isinstance(sample.get("time_ns"), bool)
            or not isinstance(sample.get("clock_mhz"), (int, float))
            or isinstance(sample.get("clock_mhz"), bool)
        ):
            raise ValueError("F2 v3 anchor requires timestamped clock samples")
        normalized.append({
            "time_ns": sample["time_ns"],
            "clock_mhz": float(sample["clock_mhz"]),
        })
    clocks = [sample["clock_mhz"] for sample in normalized]
    full = _clock_statistics(clocks)
    reasons: list[str] = []
    timestamp_order_valid = all(
        normalized[index - 1]["time_ns"] < normalized[index]["time_ns"]
        for index in range(1, len(normalized))
    )
    if not timestamp_order_valid:
        reasons.append("f2-solo-anchor-v3-timestamp-order-failure")
    median0 = full["median_mhz"]
    finite_positive = (
        median0 is not None
        and math.isfinite(median0)
        and median0 > 0
        and all(math.isfinite(clock) and clock > 0 for clock in clocks)
    )
    if not finite_positive:
        reasons.append("f2-solo-anchor-v3-nonfinite-or-nonpositive-clock")
    full_percentile_valid = bool(
        finite_positive
        and abs(full["q05_mhz"] - median0) <= 0.03 * median0
        and abs(full["q95_mhz"] - median0) <= 0.03 * median0
    )
    if finite_positive and not full_percentile_valid:
        reasons.append("f2-solo-anchor-v3-full-percentile-failure")
    outliers = (
        [
            index for index, clock in enumerate(clocks)
            if abs(clock - median0) > 0.05 * median0
        ]
        if finite_positive else []
    )
    retained = full
    excluded_sample = predecessor = successor = None
    predecessor_gap_ns = successor_gap_ns = None
    excluded_count = 0
    if len(outliers) > 1:
        reasons.append("f2-solo-anchor-v3-multiple-tail-outliers")
    elif len(outliers) == 1:
        index = outliers[0]
        excluded_count = 1
        excluded_sample = _sample_proof(normalized, index)
        retained_clocks = clocks[:index] + clocks[index + 1:]
        retained = _clock_statistics(retained_clocks)
        interior = 0 < index < len(clocks) - 1
        if not interior:
            reasons.append("f2-solo-anchor-v3-endpoint-outlier")
        else:
            predecessor = _sample_proof(normalized, index - 1)
            successor = _sample_proof(normalized, index + 1)
            predecessor_gap_ns = (
                normalized[index]["time_ns"]
                - normalized[index - 1]["time_ns"]
            )
            successor_gap_ns = (
                normalized[index + 1]["time_ns"]
                - normalized[index]["time_ns"]
            )
        median1 = retained["median_mhz"]
        if median1 != median0:
            reasons.append("f2-solo-anchor-v3-retained-median-changed")
        retained_percentile_valid = bool(
            median1 is not None
            and median1 > 0
            and abs(retained["q05_mhz"] - median1) <= 0.03 * median1
            and abs(retained["q95_mhz"] - median1) <= 0.03 * median1
        )
        if not retained_percentile_valid:
            reasons.append("f2-solo-anchor-v3-retained-percentile-failure")
        if retained["count_outside_5_percent"] != 0:
            reasons.append("f2-solo-anchor-v3-retained-tail-failure")
        if interior and (
            abs(predecessor["clock_mhz"] - median1) > 0.03 * median1
            or abs(successor["clock_mhz"] - median1) > 0.03 * median1
        ):
            reasons.append("f2-solo-anchor-v3-neighbor-envelope-failure")
    result = {
        "schema_version": F2_ANCHOR_RESULT_SCHEMA_V3,
        "classifier_version": F2_SOLO_ANCHOR_CLASSIFIER_V3,
        "valid": not reasons,
        "reasons": reasons,
        **full,
        "full_outlier_count_over_5_percent": len(outliers),
        "excluded_sample_count": excluded_count,
        "excluded_sample": excluded_sample,
        "predecessor_sample": predecessor,
        "successor_sample": successor,
        "predecessor_gap_ns": predecessor_gap_ns,
        "successor_gap_ns": successor_gap_ns,
        "retained_n": retained["n"],
        "retained_minimum_mhz": retained["minimum_mhz"],
        "retained_q05_mhz": retained["q05_mhz"],
        "retained_median_mhz": retained["median_mhz"],
        "retained_q95_mhz": retained["q95_mhz"],
        "retained_maximum_mhz": retained["maximum_mhz"],
        "retained_count_outside_3_percent": retained[
            "count_outside_3_percent"
        ],
        "retained_count_outside_5_percent": retained[
            "count_outside_5_percent"
        ],
        "anchor_median_mhz": retained["median_mhz"],
        "timestamp_order_valid": timestamp_order_valid,
    }
    result["result_sha256"] = _digest(result)
    return read_f2_solo_anchor_v3(result)


def read_f2_solo_anchor_v3(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("F2 v3 solo anchor result must be an object")
    if (
        value.get("schema_version") != F2_ANCHOR_RESULT_SCHEMA_V3
        or value.get("classifier_version")
        != F2_SOLO_ANCHOR_CLASSIFIER_V3
    ):
        raise ValueError("unknown F2 v3 solo anchor schema/classifier")
    claimed = value.get("result_sha256")
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or _digest({
            key: item for key, item in value.items()
            if key != "result_sha256"
        }) != claimed
    ):
        raise ValueError("F2 v3 solo anchor hash mismatch")
    if not isinstance(value.get("valid"), bool):
        raise ValueError("F2 v3 solo anchor validity malformed")
    return value





def _classify_role(
    role: dict[str, Any],
    *,
    qualifier_version: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    samples = role.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        samples = samples if isinstance(samples, list) else []
        reasons.append("coverage-gap")
    if any(
        not isinstance(row, dict) or row.get("query_ok") is not True
        for row in samples
    ):
        reasons.append("query-failure")
    good = [
        row for row in samples
        if isinstance(row, dict) and row.get("query_ok") is True
    ]
    clocks: list[float] = []
    try:
        clocks = [float(row["clock_mhz"]) for row in good]
    except (KeyError, TypeError, ValueError):
        reasons.append("query-failure")
    strict_clock, f2_anchor = _clock_classifications(clocks)
    f2_anchor_v3 = (
        evaluate_f2_solo_anchor_v3(good)
        if qualifier_version == F2_SCHEMA_V3 else None
    )
    if qualifier_version == F2_SCHEMA_V3:
        if not f2_anchor_v3["valid"]:
            reasons.append("f2-solo-anchor-v3-failure")
    elif qualifier_version == F2_SCHEMA_V2:
        if not f2_anchor["valid"]:
            reasons.append("f2-solo-anchor-failure")
    elif not strict_clock["valid"]:
        reasons.append("own-clock-window-failure")
    expected_policy = role.get("policy_sha256")
    if not isinstance(expected_policy, str) or len(expected_policy) != 64:
        reasons.append("policy-mismatch")
    if any(
        row.get("policy_sha256", expected_policy) != expected_policy
        for row in good
    ):
        reasons.append("policy-mismatch")
    masks = [row.get("reason_mask") for row in good]
    allowed = all(mask in {0x0, 0x1, 0x4} for mask in masks)
    has_original = any(mask in {0x0, 0x1} for mask in masks)
    has_cap = any(mask == 0x4 for mask in masks)
    if not allowed:
        reasons.append("fatal-unknown-composite-or-thermal-mask")
    fatal = bool(reasons)
    if fatal:
        classification = "fatal"
    elif has_original and has_cap:
        classification = "mixed-cap-boundary"
    elif has_cap:
        classification = "stable-power-cap"
    else:
        classification = "stable-original"
    result = {
        "role_id": role.get("role_id"),
        "classification": classification,
        "reasons": list(dict.fromkeys(reasons)),
        "median_clock_mhz": strict_clock["median_mhz"],
        "clock_min_mhz": strict_clock["minimum_mhz"],
        "clock_max_mhz": strict_clock["maximum_mhz"],
        "mask_set": sorted(
            set(masks), key=lambda value: -1 if value is None else value
        ),
        "covering_sample_count": len(samples),
    }
    if qualifier_version in {F2_SCHEMA_V2, F2_SCHEMA_V3}:
        result.update({
            "strict_own_clock_valid_v1": strict_clock["valid"],
            "f2_solo_anchor_valid_v2": f2_anchor["valid"],
            "strict_own_clock_v1": strict_clock,
            "f2_solo_anchor_v2": f2_anchor,
        })
    if qualifier_version == F2_SCHEMA_V3:
        result.update({
            "f2_solo_anchor_valid_v3": f2_anchor_v3["valid"],
            "f2_solo_anchor_v3": f2_anchor_v3,
        })
    return result


def evaluate_case_device_qualification(value: object) -> dict[str, Any]:
    _reject_scoring(value)
    if not isinstance(value, dict):
        raise ValueError("case/device qualification input must be an object")
    schema = value.get("schema_version")
    if schema not in {
        F2_SCHEMA_V1, F2_SCHEMA_V2, F2_SCHEMA_V3, GAP_SCHEMA
    }:
        raise ValueError("unknown case/device qualification schema")
    case_id = value.get("case_id")
    is_f2 = schema in {F2_SCHEMA_V1, F2_SCHEMA_V2, F2_SCHEMA_V3}
    expected_roles = 3 if is_f2 else 2
    if is_f2 and case_id not in F2_CASE_IDS:
        raise ValueError("F2 case/device disposition is closed to F2-00..F2-17")
    if schema == GAP_SCHEMA and case_id not in GAP_ROW_CASE_IDS:
        raise ValueError("gap case/device disposition is closed to seven gap rows")
    if value.get("phase") != "fast-prepair":
        raise ValueError("case/device fallback is legal only before fast pair attempt 0")
    if value.get("pair_attempt_directories") != []:
        raise ValueError("rejected/pass qualification root must precede every pair directory")
    roles = value.get("roles")
    role_order = value.get("role_order")
    if not isinstance(roles, list) or len(roles) != expected_roles:
        raise ValueError(f"qualification requires exactly {expected_roles} roles")
    role_ids = [role.get("role_id") for role in roles]
    if (
        any(not isinstance(role_id, str) or not role_id for role_id in role_ids)
        or len(set(role_ids)) != expected_roles
        or role_order != sorted(role_ids, key=lambda item: item.encode("utf-8"))
        or role_ids != role_order
    ):
        raise ValueError("qualification roles must execute in frozen bytewise role-ID order")
    for role in roles:
        for field in ("image_sha256", "input_sha256", "stream_map_sha256", "policy_sha256"):
            item = role.get(field)
            if not isinstance(item, str) or len(item) != 64:
                raise ValueError(f"role {role.get('role_id')} lacks frozen {field}")
    classifications = [
        _classify_role(role, qualifier_version=schema)
        for role in roles
    ]
    if any(row["classification"] == "fatal" for row in classifications):
        disposition = FAIL
    elif any(row["classification"] == "mixed-cap-boundary" for row in classifications):
        disposition = INELIGIBLE
    else:
        disposition = PASS
    result = {
        "schema_version": (
            RESULT_SCHEMA_V3 if schema == F2_SCHEMA_V3
            else RESULT_SCHEMA_V2 if schema == F2_SCHEMA_V2
            else RESULT_SCHEMA_V1
        ),
        "qualifier_version": schema,
        "case_id": case_id,
        "assignment_epoch_id": value.get("assignment_epoch_id"),
        "physical_index": value.get("physical_index"),
        "gpu_uuid": value.get("gpu_uuid"),
        "pci_bus_id": value.get("pci_bus_id"),
        "policy_sha256": value.get("policy_sha256"),
        "role_order": role_order,
        "roles": classifications,
        "disposition": disposition,
        "sealed_before_pairs": True,
        "pair_attempt_directories": [],
        "telemetry_only": True,
    }
    if schema in {F2_SCHEMA_V2, F2_SCHEMA_V3}:
        result.update({
            "compatibility_only": True,
            "may_supply_pair_branch_or_identity": False,
        })
    result["qualification_sha256"] = _digest(result)
    return read_case_device_qualification(result)


def read_case_device_qualification(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("case/device qualification result must be an object")
    result_schema = value.get("schema_version")
    qualifier = value.get("qualifier_version")
    if result_schema == RESULT_SCHEMA_V1:
        if qualifier not in {F2_SCHEMA_V1, GAP_SCHEMA}:
            raise ValueError("unknown v1 case/device qualifier")
    elif result_schema in {RESULT_SCHEMA_V2, RESULT_SCHEMA_V3}:
        expected_qualifier = (
            F2_SCHEMA_V3
            if result_schema == RESULT_SCHEMA_V3 else F2_SCHEMA_V2
        )
        if qualifier != expected_qualifier:
            raise ValueError("unknown F2 case/device qualifier")
        if (
            value.get("compatibility_only") is not True
            or value.get("may_supply_pair_branch_or_identity") is not False
        ):
            raise ValueError("F2 case qualification may not alias operational solo")
        roles = value.get("roles")
        if not isinstance(roles, list) or len(roles) != 3:
            raise ValueError("F2 result requires three roles")
        for role in roles:
            strict = read_strict_own_clock(role.get("strict_own_clock_v1"))
            anchor = read_f2_solo_anchor(role.get("f2_solo_anchor_v2"))
            if (
                role.get("strict_own_clock_valid_v1") is not strict["valid"]
                or role.get("f2_solo_anchor_valid_v2") is not anchor["valid"]
            ):
                raise ValueError("F2 dual clock-classification validity mismatch")
            if result_schema == RESULT_SCHEMA_V3:
                anchor_v3 = read_f2_solo_anchor_v3(
                    role.get("f2_solo_anchor_v3")
                )
                if (
                    role.get("f2_solo_anchor_valid_v3")
                    is not anchor_v3["valid"]
                ):
                    raise ValueError(
                        "F2 v3 clock-classification validity mismatch"
                    )
    else:
        raise ValueError("unknown case/device qualification result schema")
    _reject_scoring(value)
    if value.get("disposition") not in {PASS, INELIGIBLE, FAIL}:
        raise ValueError("unknown case/device disposition")
    claimed = value.get("qualification_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("case/device qualification hash missing")
    unhashed = {
        key: value[key] for key in value if key != "qualification_sha256"
    }
    if _digest(unhashed) != claimed:
        raise ValueError("case/device qualification hash mismatch")
    return value


def rotated_fallback_order(ordered_devices: list[int], assigned: int) -> list[int]:
    if (
        not ordered_devices
        or len(set(ordered_devices)) != len(ordered_devices)
        or ordered_devices != sorted(ordered_devices)
        or assigned not in ordered_devices
    ):
        raise ValueError("fallback order requires unique numeric qualified list and assigned member")
    offset = ordered_devices.index(assigned)
    return ordered_devices[offset:] + ordered_devices[:offset]


class CaseDeviceFallback:
    """One-pass dispatcher over a presealed rotated case fallback order."""

    def __init__(
        self,
        ordered_devices: list[int],
        assigned: int,
        *,
        wait_for_lock: Callable[[int], None],
    ) -> None:
        self.order = rotated_fallback_order(ordered_devices, assigned)
        self.wait_for_lock = wait_for_lock
        self.position = 0
        self.tried: list[int] = []

    def current(self) -> int:
        if self.position >= len(self.order):
            raise RuntimeError("case fallback order exhausted")
        device = self.order[self.position]
        if device in self.tried:
            raise AssertionError("case/device tuple revisited")
        self.wait_for_lock(device)
        self.tried.append(device)
        return device

    def consume(self, result: object) -> int | None:
        record = read_case_device_qualification(result)
        if record["physical_index"] != self.tried[-1]:
            raise ValueError("qualification result differs from current fallback device")
        if record["disposition"] == PASS:
            return self.tried[-1]
        if record["disposition"] == FAIL:
            raise RuntimeError("fatal CASE_DEVICE_QUALIFICATION_FAIL; fallback forbidden")
        self.position += 1
        if self.position >= len(self.order):
            raise RuntimeError("case fallback order exhausted")
        return None


def require_case_qualification_pass(record: object, pair_root: Path) -> None:
    value = read_case_device_qualification(record)
    if value["disposition"] != PASS:
        raise RuntimeError("paired launcher blocked until case/device qualification PASS")
    if pair_root.exists() and any(pair_root.iterdir()):
        raise RuntimeError("pair root must be empty before qualification handoff")
