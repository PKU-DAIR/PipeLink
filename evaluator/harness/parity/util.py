"""Small stdlib-only helpers shared by the parity harness."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest") from exc
    if len(parsed) != 32 or value != value.lower():
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value
