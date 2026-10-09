"""Executable tutorial acceptance; GPU execution is explicitly owner-selected."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipec2.profile import PROFILES
from tools.pipelink_env import load

UV = ("uv", "run", "--project", "pipec", "--locked")
EPILOGUE = REPO / "examples/one-epilogue-many-producers"


def command(*args, env):
    result = subprocess.run(list(map(str, args)), cwd=REPO, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Command failed ({result.returncode}): {' '.join(map(str, args))}\n"
        f"{result.stdout}\n{result.stderr}"
    )
    return result.stdout


@pytest.fixture(scope="session")
def build_env():
    return {
        **os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": "",
        "CUTLASS_INCLUDE_DIR": str(REPO / "third_party/cutlass/include"),
    }


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_t0_pack_inspect_link(profile, tmp_path, build_env):
    manifests = {}
    for stage in ("provider", "caller"):
        package = tmp_path / f"{stage}.plink"
        command(*UV, "pipec2", "pack", REPO / f"examples/hello-callable/src/{stage}.cu",
                "--profile", profile, "-o", package, env=build_env)
        manifests[stage] = json.loads(command(
            *UV, "pipec2", "inspect", package, "--json", env=build_env))
    pipeline = {
        "pipelink_abi": manifests["caller"]["pipelink_abi"],
        "pipeline": {
            "name": "hello-callable",
            "stages": [
                {"stage_id": "org.hello.provider", "artifact_path": "provider.plink"},
                {"stage_id": "org.hello.caller", "artifact_path": "caller.plink"},
            ],
            "handles": {"pipelink_call_warps": {
                "class": "base", "binding": "bind", "domain": "warp", "owner": "claim:call",
            }},
            "phases": ["call"],
            "exclusive": [], "domains": {}, "depth_priority": [], "regs_priority": [],
            "forced_bindings": {},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline) + "\n")
    command(*UV, "pipec2", "link", pipeline_path, "--out-dir", tmp_path / "link",
            env=build_env)
    link = json.loads((tmp_path / "link/link.json").read_text())
    assert link["threads"] == 128
    assert link["dyn_smem_bytes"] == 0


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_t3_contract_gallery(profile, build_env):
    command(*UV, "python", "examples/contract-gallery/check.py", "--profile", profile,
            env=build_env)


@pytest.fixture(scope="session")
def tutorial_gpu(build_env):
    physical = os.environ.get("PIPELINK_TUTORIAL_GPU")
    if not physical:
        pytest.skip("set PIPELINK_TUTORIAL_GPU to a physical Hopper or Blackwell GPU index")
    if not physical.isdecimal():
        pytest.fail("PIPELINK_TUTORIAL_GPU must be a non-negative physical GPU index")
    capability = command("nvidia-smi", "-i", physical, "--query-gpu=compute_cap",
                         "--format=csv,noheader", env=build_env).strip()
    profile = f"sm_{capability.replace('.', '')}a"
    if profile not in PROFILES:
        pytest.fail(f"no registered profile for compute_cap={capability}; registered: {sorted(PROFILES)}")
    return physical, profile


def run_acceptance(example, profile, physical, build_env, *, runner="run.py", output=None):
    root = REPO / "examples" / example
    results_path = (output or root / "out" / profile) / "results.json"
    results_path.unlink(missing_ok=True)
    command(*UV, "python", root / runner, "--profile", profile, "--gpu", physical,
            env={**build_env, "CUDA_VISIBLE_DEVICES": physical})
    assert json.loads(results_path.read_text())["status"] == "PASS"


@pytest.mark.gpu
@pytest.mark.parametrize("example", ("hello-callable", "ring-pipeline"), ids=("T1", "T2"))
def test_callable_and_ring(example, tutorial_gpu, build_env):
    physical, profile = tutorial_gpu
    command(*UV, "python", REPO / "examples" / example / "build.py", "--profile", profile,
            env=build_env)
    run_acceptance(example, profile, physical, build_env)


@pytest.fixture(scope="session")
def hand_epilogue(tutorial_gpu, build_env):
    _, profile = tutorial_gpu
    command(*UV, "python", EPILOGUE / "build.py", "--profile", profile, env=build_env)


@pytest.mark.gpu
def test_t4_hand_epilogue(tutorial_gpu, build_env, request):
    if not build_env.get("PIPELINK_TILELANG_PYTHON"):
        pytest.fail(
            "T4 hand-CUDA cells require PIPELINK_TILELANG_PYTHON pointing to the pinned, "
            "patched TileLang backend for non-alias tensor adapters; see T4 prerequisites "
            "and frontends/tilelang/env/README.md"
        )
    request.getfixturevalue("hand_epilogue")
    physical, profile = tutorial_gpu
    run_acceptance(EPILOGUE.name, profile, physical, build_env)


@pytest.mark.gpu
def test_t4_tilelang_cells(tutorial_gpu, build_env, request):
    if not build_env.get("PIPELINK_TILELANG_PYTHON"):
        pytest.skip("TileLang cells require PIPELINK_TILELANG_PYTHON (T4)")
    if not build_env.get("TILELANG_INCLUDE_DIR"):
        pytest.fail("T4 TileLang cells require TILELANG_INCLUDE_DIR; see T4 environment setup")
    request.getfixturevalue("hand_epilogue")
    physical, profile = tutorial_gpu
    out = EPILOGUE / "out" / profile
    python = build_env["PIPELINK_TILELANG_PYTHON"]
    includes = ("-I", REPO / "third_party/cutlass/include",
                "-I", build_env["TILELANG_INCLUDE_DIR"])
    command(python, "-m", "pipelink_tilelang", "--kernel",
            EPILOGUE / "providers/tilelang/gemm_provider.py", "--profile", profile,
            "--out", out / "tilelang/provider", env=build_env)
    command(*UV, "pipec2", "pack", out / "tilelang/provider/roles.cu",
            "-o", out / "packed/tl-reg.plink", *includes, "--profile", profile, env=build_env)
    command(python, "-m", "pipelink_tilelang", "--kernel",
            EPILOGUE / "consumer-tilelang/epilogue_contract.py", "--profile", profile,
            "--out", out / "tilelang/consumer", env={
                **build_env,
                "PIPELINK_ACC_PARTNER": str(out / "packed/tl-reg.plink"),
                "PIPELINK_SCALE_PARTNER": str(out / "packed/rowscale.plink"),
            })
    command(*UV, "pipec2", "pack", out / "tilelang/consumer/generation/tilelang-epilogue/roles.cu",
            "-o", out / "packed/tl-epilogue.plink", *includes, "--profile", profile, env=build_env)
    command(*UV, "python", EPILOGUE / "pipelines/extra_cells.py", profile, env=build_env)
    command(*UV, "python", EPILOGUE / "build.py", "--profile", profile,
            "--extra-cells", out / "extra-cells.json", env=build_env)
    run_acceptance(EPILOGUE.name, profile, physical, build_env)


@pytest.mark.gpu
def test_t5_host_abi(tutorial_gpu, build_env):
    physical, profile = tutorial_gpu
    if profile == "sm_90a":
        example = "hopper-gemm-split"
        command(*UV, "python", REPO / "examples" / example / "build_host.py",
                "--profile", profile, env=build_env)
        run_acceptance(example, profile, physical, build_env, runner="run_host.py",
                       output=REPO / "examples" / example / "out" / f"host-{profile}")
    else:
        example = "profile-consumer"
        command(*UV, "python", REPO / "examples" / example / "build.py",
                "--profile", profile, env=build_env)
        run_acceptance(example, profile, physical, build_env,
                       output=REPO / "examples" / example / "build" / profile)
