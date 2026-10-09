"""Fail-closed can-implement/public-dispatcher preflight records."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import (
    PREFLIGHT_COMPLETED_BEFORE_FREEZE,
    PREFLIGHT_HASH_CATEGORIES,
    PREFLIGHT_NO_TIMED_SLOT_FIRST,
    SCHEMA_VERSIONS,
)
from .util import require_sha256, utc_now, write_json_atomic


def _validate_time(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("utc_time must be an ISO-8601 UTC string ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("utc_time must be an ISO-8601 UTC string ending in Z") from exc
    return value


def _validate_hash_inventory(hashes: object) -> dict[str, Any]:
    if not isinstance(hashes, dict) or set(hashes) != set(PREFLIGHT_HASH_CATEGORIES):
        raise ValueError(f"hashes must contain exactly {'/'.join(PREFLIGHT_HASH_CATEGORIES)}")
    result: dict[str, Any] = {}
    for category, inventory in hashes.items():
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError(f"hashes.{category} must be a non-empty object")
        result[category] = {
            name: require_sha256(value, f"hashes.{category}.{name}")
            for name, value in sorted(inventory.items())
            if isinstance(name, str) and name
        }
        if len(result[category]) != len(inventory):
            raise ValueError(f"hashes.{category} has an invalid name")
    return result


def _validate_entry(entry: object) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("preflight entry must be an object")
    required = {"entry_id", "case_id", "image_id", "shape", "arguments", "eligibility", "hashes"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"preflight entry misses {sorted(missing)}")
    for key in ("entry_id", "case_id", "image_id"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise ValueError(f"preflight entry {key} must be non-empty")
    shape = entry["shape"]
    if not isinstance(shape, list) or not shape or any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError(f"{entry['entry_id']}: shape must contain positive integers")
    if not isinstance(entry["arguments"], dict):
        raise ValueError(f"{entry['entry_id']}: arguments must be an object")
    hashes = entry["hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{entry['entry_id']}: hashes must be non-empty")
    checked_hashes = {
        name: require_sha256(value, f"entry.{entry['entry_id']}.hashes.{name}")
        for name, value in sorted(hashes.items())
        if isinstance(name, str) and name
    }
    if len(checked_hashes) != len(hashes):
        raise ValueError(f"{entry['entry_id']}: invalid hash name")
    eligibility = entry["eligibility"]
    if not isinstance(eligibility, dict):
        raise ValueError(f"{entry['entry_id']}: eligibility must be an object")
    kind = eligibility.get("kind")
    if kind == "cutlass-can-implement":
        expected = {"kind", "eligible", "status_code", "status_string"}
        if set(eligibility) != expected:
            raise ValueError(f"{entry['entry_id']}: CUTLASS eligibility fields must be {sorted(expected)}")
        if not isinstance(eligibility["status_code"], int) or not isinstance(eligibility["status_string"], str):
            raise ValueError(f"{entry['entry_id']}: CUTLASS status code/string required")
        if eligibility["eligible"] != (eligibility["status_code"] == 0):
            raise ValueError(f"{entry['entry_id']}: CUTLASS eligible/status-code mismatch")
    elif kind == "fa4-public-dispatcher-jit":
        expected = {"kind", "eligible", "dispatcher_eligible", "jit_eligible", "status_string"}
        if set(eligibility) != expected:
            raise ValueError(f"{entry['entry_id']}: FA4 eligibility fields must be {sorted(expected)}")
        if not isinstance(eligibility["dispatcher_eligible"], bool) or not isinstance(eligibility["jit_eligible"], bool):
            raise ValueError(f"{entry['entry_id']}: dispatcher/JIT eligibility must be boolean")
        if eligibility["eligible"] != (eligibility["dispatcher_eligible"] and eligibility["jit_eligible"]):
            raise ValueError(f"{entry['entry_id']}: FA4 aggregate eligibility mismatch")
    else:
        raise ValueError(f"{entry['entry_id']}: unsupported eligibility kind {kind!r}")
    if not isinstance(eligibility.get("eligible"), bool):
        raise ValueError(f"{entry['entry_id']}: eligible must be boolean")
    result = deepcopy(entry)
    result["hashes"] = checked_hashes
    return result


def build_preflight_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and close a complete preflight matrix without deleting false rows."""
    if not isinstance(record, dict):
        raise ValueError("preflight record must be an object")
    result = deepcopy(record)
    result["schema_version"] = SCHEMA_VERSIONS["preflight"]
    if result.get("completed_before_measurement_freeze") is not PREFLIGHT_COMPLETED_BEFORE_FREEZE:
        raise ValueError("completed_before_measurement_freeze must be true")
    if result.get("fail_closed_no_timed_slot_ran_first") is not PREFLIGHT_NO_TIMED_SLOT_FIRST:
        raise ValueError("fail_closed_no_timed_slot_ran_first must be true")
    result["utc_time"] = _validate_time(result.get("utc_time", utc_now()))
    device = result.get("device")
    if not isinstance(device, dict):
        raise ValueError("device must be an object")
    for field in ("physical_index", "uuid", "name"):
        if field not in device:
            raise ValueError(f"device.{field} is required")
    if (
        not isinstance(device["physical_index"], int)
        or isinstance(device["physical_index"], bool)
        or device["physical_index"] <= 0
    ):
        raise ValueError("device.physical_index must be a positive nonzero physical index")
    if not isinstance(device["uuid"], str) or not device["uuid"].startswith("GPU-"):
        raise ValueError("device.uuid must be a physical GPU UUID")
    environment = result.get("environment")
    if not isinstance(environment, dict) or set(environment) != {"toolkit", "driver", "target"}:
        raise ValueError("environment must contain exactly toolkit/driver/target")
    if any(not isinstance(environment[field], str) or not environment[field] for field in environment):
        raise ValueError("toolkit/driver/target values must be non-empty strings")
    if environment["target"] not in {"sm90", "sm100", "sm103", "sm120"}:
        raise ValueError("target must be one of sm90, sm100, sm103, sm120")
    result["hashes"] = _validate_hash_inventory(result.get("hashes"))
    entries = result.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    checked_entries = [_validate_entry(entry) for entry in entries]
    entry_ids = [entry["entry_id"] for entry in checked_entries]
    if len(set(entry_ids)) != len(entry_ids):
        raise ValueError("preflight entry IDs must be unique")
    expected_count = result.get("summary", {}).get("expected_count") if isinstance(result.get("summary"), dict) else None
    if not isinstance(expected_count, int) or expected_count < 0:
        raise ValueError("summary.expected_count must be a non-negative integer")
    observed_count = len(checked_entries)
    all_true = observed_count == expected_count and all(entry["eligibility"]["eligible"] for entry in checked_entries)
    result["entries"] = checked_entries
    result["summary"] = {
        "expected_count": expected_count,
        "observed_count": observed_count,
        "all_true": all_true,
        "fail_closed_no_timed_slot_ran_first": True,
    }
    return result


def write_preflight(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    result = build_preflight_record(record)
    write_json_atomic(path, result)
    return result
