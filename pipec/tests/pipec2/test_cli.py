from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pipec2 import cli
from pipec2.canonjson import canonical_bytes, canonical_dumps
from pipec2.model import NVCC_DEFAULT_MATH
from pipec2.profile import get_profile
from pipec2.symbols import cuda_home


ZERO_HASH = "0" * 64


def _manifest() -> dict[str, object]:
    return {
        "pipelink_abi": {
            "schema": "2.0-draft",
            "target": {
                "profile": "sm_103a",
                "lto_arch": "lto_103",
                "toolkit": "cuda-13.3.0",
            },
        },
        "stage": {
            "id": "org.example",
            "version": "1.0.0",
            "symbol_prefix": "org_example_",
            "entry_abi": "v2",
            "roles": {"work": {"entry": "org_example_work", "warps": 4}},
            "exports": {},
            "entry_signature": "void(const void* params, unsigned variant)",
            "variants": [{"key": {}, "index": 0}],
        },
        "endpoints": [],
        "claims": [],
        "topology": {
            "cta_group": {"fixed": 1},
            "cluster": {
                "compiled": "dynamic",
                "launch": {"x_multiple_of": 1, "max": [1, 1, 1]},
            },
            "multicast": [],
            "persistent": False,
            "scheduler": "static",
        },
        "params": {
            "bytes": 8,
            "align": 8,
            "type": "OpaqueParams",
            "sha256_of_layout_probe": ZERO_HASH,
        },
        "hints": {},
        "payload": {"file": "stage.o"},
    }








def test_raw_pack_inspect_preserves_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from pipec2.container import pack

    manifest = _manifest()
    manifest["provenance"] = {
        "mechanism": "pack-cu-v1",
        "inputs": {"tu": "0123456789abcdef" * 4},
    }
    artifact = tmp_path / "provenance.plink"
    pack(manifest, b"lto-object", artifact)
    assert cli.main(["inspect", str(artifact), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["provenance"] == manifest["provenance"]

def test_inspect_explicitly_rehashes_payload(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import zipfile
    from pipec2.container import read

    manifest = _manifest()
    manifest["payload"]["sha256"] = "0" * 64
    artifact = tmp_path / "tampered.plink"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("stage.o", b"unverified payload")
    assert read(artifact) == (manifest, b"unverified payload")
    assert cli.main(["inspect", str(artifact)]) == 1
    assert "payload SHA-256 mismatch" in capsys.readouterr().err















def test_clauses_are_deterministic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pipec2.model import CLAUSES

    assert cli.main(["clauses", "--json"]) == 0
    assert capsys.readouterr().out == canonical_dumps(CLAUSES)
    assert cli.main(["clauses"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{clause_id}: {CLAUSES[clause_id]}" for clause_id in sorted(CLAUSES)
    ]


def test_cli_rejects_duplicate_blob_stage_ids(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        cli.main(
            [
                "launch",
                "link.json",
                "--blob",
                "same=a.bin",
                "--blob",
                "same=b.bin",
            ]
        )
        == 1
    )
    assert "duplicate --blob stage ID" in capsys.readouterr().err


_STAGE_ANNOTATIONS = (
    '#include <pipelink/stage.hpp>\n'
    'struct Params { unsigned count; };\n'
    'PL_STAGE("org.example", "1.0.0");\n'
    'PL_PARAMS(Params, count);\n'
    'PL_ROLE(work, org_example_work, 4);\n'
)


def _require_cuda_toolchain() -> None:
    try:
        cuda = cuda_home()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if (
        not (cuda / "bin" / "nvcc").is_file()
        or not (cuda / "lib64" / "libnvJitLink.so.13").is_file()
        or shutil.which("nm") is None
    ):
        pytest.skip("configured CUDA compilation tools are unavailable")


def test_toolchain_pack_link_rejects_undeclared_import_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _require_cuda_toolchain()
    source = tmp_path / "stage.cu"
    artifact_path = tmp_path / "stage.plink"
    source.write_text(
        'extern "C" __device__ void org_example_work(const void*, unsigned) {}\n'
        + _STAGE_ANNOTATIONS,
        encoding="utf-8",
    )
    assert cli.main(["pack", str(source), "--out", str(artifact_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads(Path(str(artifact_path) + ".report.json").read_text()) == report
    assert cli.main(["inspect", str(artifact_path), "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["stage"]["id"] == "org.example"
    assert manifest["payload"]["sha256"] == report["payload"]["sha256"]
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_bytes(canonical_bytes({
        "pipelink_abi": manifest["pipelink_abi"],
        "pipeline": {
            "name": "pack-link-smoke",
            "stages": [{"stage_id": "org.example", "artifact_path": "stage.plink"}],
            "handles": {
                "pipelink_work_warps": {
                    "class": "base", "binding": "bind", "domain": "warp",
                    "owner": "claim:work",
                },
            },
            "phases": ["work"],
            "exclusive": [],
            "domains": {},
            "depth_priority": [],
            "regs_priority": [],
            "forced_bindings": {},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }))
    assert cli.main(["link", str(pipeline_path), "--out-dir", str(tmp_path / "linked")]) == 0
    assert json.loads(capsys.readouterr().out)["kernel_symbol"] == "pipelink_kernel"

    source.write_text(
        'extern "C" __device__ void undeclared_external();\n'
        'extern "C" __device__ void org_example_work(const void*, unsigned) { undeclared_external(); }\n'
        + _STAGE_ANNOTATIONS,
        encoding="utf-8",
    )
    rejected_artifact = tmp_path / "undeclared.plink"
    assert cli.main(["pack", str(source), "--out", str(rejected_artifact)]) == 1
    error = capsys.readouterr().err
    assert "P-IMPORT" in error and "undeclared_external" in error
    assert not rejected_artifact.exists()


@pytest.mark.parametrize("profile_name", ["sm_90a", "sm_120a"])
def test_toolchain_pack_stamps_target_from_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], profile_name: str
) -> None:
    _require_cuda_toolchain()
    arch = get_profile(profile_name).arch
    source = tmp_path / "stage.cu"
    artifact_path = tmp_path / "stage.plink"
    source.write_text(
        'extern "C" __device__ void org_example_work(const void*, unsigned) {}\n'
        + _STAGE_ANNOTATIONS,
        encoding="utf-8",
    )
    assert cli.main(["pack", str(source), "--out", str(artifact_path), "--profile", profile_name]) == 0, capsys.readouterr().err
    report = json.loads(capsys.readouterr().out)
    assert [flag for flag in report["compile"]["emit"] if flag.startswith("arch=")] == [f"arch={arch.compute},code={arch.sm}"]
    assert [flag for flag in report["compile"]["device"] if flag.startswith("arch=")] == [f"arch={arch.compute},code={arch.lto}"]
    assert cli.main(["inspect", str(artifact_path), "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["pipelink_abi"]["target"] == {"profile": profile_name, "lto_arch": arch.lto, "toolkit": "cuda-13.3.0",
                                             "math": NVCC_DEFAULT_MATH.to_json_obj()}
    assert manifest["symbols"]["mechanism"] == report["symbol_mechanism"]
    assert manifest["symbols"]["defined_strong"] == ["org_example_work"]
