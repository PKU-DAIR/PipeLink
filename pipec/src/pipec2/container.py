from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any
import zipfile

from .canonjson import canonical_bytes, sha256_hex
from .model import ClauseFailure, PackError, register_clause


MEMBER_NAMES = ("manifest.json", "stage.o")
CANONICAL_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
CANONICAL_CREATE_SYSTEM = 3
CANONICAL_EXTERNAL_ATTR = 0o600 << 16
CANONICAL_COMPRESS_LEVEL = 9

register_clause(
    "P-CONTAINER",
    phase="pack",
    description="artifact containers and payload declarations are readable and well formed",
)


def _raise_pack_error(path: str, message: str) -> None:
    raise PackError([ClauseFailure("P-CONTAINER", path, message, {})])


def _member_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=CANONICAL_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = CANONICAL_CREATE_SYSTEM
    info.external_attr = CANONICAL_EXTERNAL_ATTR
    info.internal_attr = 0
    info.flag_bits = 0
    info.extra = b""
    info.comment = b""
    return info


def _build_container(manifest_bytes: bytes, object_bytes: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=CANONICAL_COMPRESS_LEVEL,
    ) as archive:
        archive.comment = b""
        archive.writestr(
            _member_info(MEMBER_NAMES[0]),
            manifest_bytes,
            compresslevel=CANONICAL_COMPRESS_LEVEL,
        )
        archive.writestr(
            _member_info(MEMBER_NAMES[1]),
            object_bytes,
            compresslevel=CANONICAL_COMPRESS_LEVEL,
        )
    return output.getvalue()


def _source_manifest(manifest_obj: dict[str, Any], object_bytes: bytes) -> dict[str, Any]:
    if not isinstance(manifest_obj, dict):
        _raise_pack_error("$", "manifest root must be an object")
    if not isinstance(object_bytes, bytes):
        _raise_pack_error("stage.o", "object payload must be bytes")

    manifest = copy.deepcopy(manifest_obj)
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        _raise_pack_error("$.payload", "source manifest payload must be an object")
    unknown = sorted(set(payload) - {"file", "sha256"})
    if unknown:
        _raise_pack_error(f"$.payload.{unknown[0]}", "unknown payload field")
    if payload.get("file") != MEMBER_NAMES[1]:
        _raise_pack_error("$.payload.file", "payload file must be 'stage.o'")
    if "sha256" in payload:
        _raise_pack_error(
            "$.payload.sha256",
            "source-supplied payload digest is forbidden; pack computes it",
        )
    payload["sha256"] = sha256_hex(object_bytes)
    return manifest


def pack(manifest_obj: dict[str, Any], object_bytes: bytes, out: Path) -> dict[str, Any]:
    """Pack a source manifest and object into the canonical two-member ZIP."""

    manifest = _source_manifest(manifest_obj, object_bytes)
    try:
        manifest_bytes = canonical_bytes(manifest)
        container_bytes = _build_container(manifest_bytes, object_bytes)
    except (TypeError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        _raise_pack_error("$", f"cannot encode canonical artifact: {exc}")

    output_path = Path(out)
    try:
        output_path.write_bytes(container_bytes)
    except OSError as exc:
        _raise_pack_error(str(output_path), f"cannot write artifact: {exc}")

    payload_digest = manifest["payload"]["sha256"]
    return {
        "artifact_sha256": sha256_hex(container_bytes),
        "manifest_sha256": sha256_hex(manifest_bytes),
        "payload": {
            "file": MEMBER_NAMES[1],
            "sha256": payload_digest,
            "size": len(object_bytes),
        },
    }


def read(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read the manifest and object; pack owns canonicalization and hashing."""

    source = Path(path)
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            manifest_bytes = archive.read("manifest.json")
            object_bytes = archive.read("stage.o")
    except (zipfile.BadZipFile, OSError, RuntimeError, ValueError, KeyError) as exc:
        _raise_pack_error("zip", f"invalid ZIP container: {exc}")

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _raise_pack_error("$.manifest.json", f"manifest is not UTF-8 JSON: {exc}")
    if not isinstance(manifest, dict):
        _raise_pack_error("$", "manifest root must be an object")
    return manifest, object_bytes


__all__ = [
    "CANONICAL_COMPRESS_LEVEL",
    "CANONICAL_CREATE_SYSTEM",
    "CANONICAL_EXTERNAL_ATTR",
    "CANONICAL_TIMESTAMP",
    "MEMBER_NAMES",
    "pack",
    "read",
]
