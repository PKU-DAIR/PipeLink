"""Explicit strict-v2/gap-v6 result dispatch for gates and aggregators."""
from __future__ import annotations

from typing import Any, Iterable

from .constants import SCHEMA_VERSIONS


def pair_is_primary_valid(record: dict[str, Any]) -> bool:
    """Read the primary validity field only after explicit schema dispatch."""
    version = record.get("schema_version")
    if version in {
        SCHEMA_VERSIONS["pair"],
        SCHEMA_VERSIONS["fast_pair"],
        SCHEMA_VERSIONS["original_pair"],
        SCHEMA_VERSIONS["negative_pair"],
    }:
        value = record.get("pair_valid")
    elif version == SCHEMA_VERSIONS["gap_pair"]:
        classification = record.get("gap_classification")
        if not isinstance(classification, dict):
            raise ValueError("gap-v6 pair lacks gap_classification")
        value = classification.get("gap_pair_valid")
    else:
        raise ValueError(f"unsupported parity pair schema: {version!r}")
    if not isinstance(value, bool):
        raise ValueError(f"{version} primary validity must be boolean")
    return value


def chronological_primary_pairs(
    records: Iterable[dict[str, Any]], target: int
) -> list[dict[str, Any]]:
    """Return at most the first target primary-valid records by attempt number."""
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        raise ValueError("pair target must be a positive integer")
    ordered = sorted(records, key=lambda record: record.get("attempt", -1))
    attempts = [record.get("attempt") for record in ordered]
    if any(not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0 for attempt in attempts):
        raise ValueError("every pair record needs a non-negative integer attempt")
    if len(set(attempts)) != len(attempts):
        raise ValueError("pair attempt numbers must be unique")
    return [record for record in ordered if pair_is_primary_valid(record)][:target]


def summary_primary_valid_count(record: dict[str, Any]) -> int:
    """Read the primary valid count only after explicit summary-schema dispatch."""
    version = record.get("schema_version")
    if version in {
        SCHEMA_VERSIONS["summary"],
        SCHEMA_VERSIONS["fast_summary"],
        SCHEMA_VERSIONS["original_summary"],
        SCHEMA_VERSIONS["negative_summary"],
    }:
        value = record.get("pair_valid_count")
    elif version == SCHEMA_VERSIONS["gap_summary"]:
        value = record.get("gap_pair_valid_count")
    else:
        raise ValueError(f"unsupported parity summary schema: {version!r}")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{version} primary valid count must be a non-negative integer")
    return value
