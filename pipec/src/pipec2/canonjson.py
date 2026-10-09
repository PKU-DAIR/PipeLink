from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(obj: Any) -> str:
    """Serialize a JSON value deterministically, terminated by one line feed."""

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def canonical_bytes(obj: Any) -> bytes:
    """Return :func:`canonical_dumps` encoded as UTF-8."""

    return canonical_dumps(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    return hashlib.sha256(data).hexdigest()


def canonical_digest(value: Any) -> str:
    """Hash integer-only semantic JSON, treating tuples as ordered lists."""
    def check(item: Any) -> None:
        if item is None or type(item) in (str, int, bool):
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif isinstance(item, dict) and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
        else:
            raise ValueError("semantic digest requires JSON with integer-only numeric payloads")

    check(value)
    return sha256_hex(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
