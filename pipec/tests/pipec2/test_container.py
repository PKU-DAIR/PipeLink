from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from pipec2.canonjson import sha256_hex
from pipec2.container import (
    CANONICAL_CREATE_SYSTEM,
    CANONICAL_EXTERNAL_ATTR,
    CANONICAL_TIMESTAMP,
    MEMBER_NAMES,
    pack,
    read,
)
from pipec2.model import PackError


def _manifest() -> dict:
    return {
        "target": {"toolkit": "13.3", "lto_arch": "lto_103"},
        "stage": {"version": "1.0", "id": "org.example.stage"},
        "payload": {"file": "stage.o"},
    }




def test_pack_is_byte_deterministic_and_does_not_mutate_input(tmp_path: Path) -> None:
    object_bytes = (b"ELF-LTO\0" + bytes(range(256))) * 16
    manifest = _manifest()
    original = json.loads(json.dumps(manifest))
    first = tmp_path / "first.plink"
    second = tmp_path / "second.plink"

    first_report = pack(manifest, object_bytes, first)
    second_report = pack(manifest, object_bytes, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_report == second_report
    assert first_report["artifact_sha256"] == sha256_hex(first.read_bytes())
    assert first_report["payload"] == {
        "file": "stage.o",
        "sha256": sha256_hex(object_bytes),
        "size": len(object_bytes),
    }
    assert manifest == original

    packed_manifest, packed_object = read(first)
    assert packed_object == object_bytes
    assert packed_manifest["payload"] == {
        "file": "stage.o",
        "sha256": sha256_hex(object_bytes),
    }


def test_manifest_insertion_order_does_not_change_container_bytes(tmp_path: Path) -> None:
    first_manifest = _manifest()
    second_manifest = {
        "payload": {"file": "stage.o"},
        "stage": {"id": "org.example.stage", "version": "1.0"},
        "target": {"lto_arch": "lto_103", "toolkit": "13.3"},
    }
    object_bytes = b"same object"
    first = tmp_path / "first.plink"
    second = tmp_path / "second.plink"

    pack(first_manifest, object_bytes, first)
    pack(second_manifest, object_bytes, second)

    assert first.read_bytes() == second.read_bytes()


def test_pack_writes_all_canonical_zip_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "stage.plink"
    pack(_manifest(), b"object", artifact)

    with zipfile.ZipFile(artifact) as archive:
        assert archive.comment == b""
        infos = archive.infolist()
        assert [info.filename for info in infos] == list(MEMBER_NAMES)
        for info in infos:
            assert info.date_time == CANONICAL_TIMESTAMP
            assert info.compress_type == zipfile.ZIP_DEFLATED
            assert info.create_system == CANONICAL_CREATE_SYSTEM
            assert info.external_attr == CANONICAL_EXTERNAL_ATTR
            assert info.internal_attr == 0
            assert info.flag_bits == 0
            assert info.extra == b""
            assert info.comment == b""


def test_pack_rejects_source_supplied_digest_and_bad_payload_shape(tmp_path: Path) -> None:
    supplied = _manifest()
    supplied["payload"]["sha256"] = "0" * 64
    with pytest.raises(PackError, match="source-supplied payload digest is forbidden"):
        pack(supplied, b"object", tmp_path / "digest.plink")

    bad_file = _manifest()
    bad_file["payload"]["file"] = "other.o"
    with pytest.raises(PackError, match="payload file must be 'stage.o'"):
        pack(bad_file, b"object", tmp_path / "file.plink")

    extra = _manifest()
    extra["payload"]["size"] = 6
    with pytest.raises(PackError, match="unknown payload field"):
        pack(extra, b"object", tmp_path / "extra.plink")


