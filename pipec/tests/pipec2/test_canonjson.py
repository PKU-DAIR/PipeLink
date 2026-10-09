from __future__ import annotations

import hashlib

import pytest

from pipec2.canonjson import canonical_bytes, canonical_dumps, sha256_hex


def test_canonical_json_is_sorted_compact_and_lf_terminated() -> None:
    obj = {"z": [3, 2, 1], "a": {"y": True, "x": None}}
    expected = '{"a":{"x":null,"y":true},"z":[3,2,1]}\n'
    assert canonical_dumps(obj) == expected
    assert canonical_bytes(obj) == expected.encode("utf-8")


def test_sha256_hex_hashes_bytes() -> None:
    data = b"PipeLink ABI v2\n"
    assert sha256_hex(data) == hashlib.sha256(data).hexdigest()


def test_canonical_json_rejects_non_json_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_dumps({"bad": float("nan")})
