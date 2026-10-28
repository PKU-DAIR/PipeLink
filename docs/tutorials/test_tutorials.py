"""Executable tutorial acceptance; GPU execution is explicitly owner-selected."""
from __future__ import annotations

import ctypes as C
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
CODA = REPO / "campaigns/coda"


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
    results = json.loads(results_path.read_text())
    assert results["status"] == "PASS"


@pytest.mark.gpu
@pytest.mark.parametrize("example", ("hello-callable", "ring-pipeline"), ids=("T1", "T2"))
def test_callable_and_ring(example, tutorial_gpu, build_env):
    physical, profile = tutorial_gpu
    command(*UV, "python", REPO / "examples" / example / "build.py", "--profile", profile,
            env=build_env)
    run_acceptance(example, profile, physical, build_env)


@pytest.mark.gpu
def test_t4_t6_coda(tutorial_gpu, build_env, tmp_path):
    if not (build_env.get("PIPELINK_TILELANG_PYTHON") and build_env.get("TILELANG_INCLUDE_DIR")):
        pytest.fail("T4/T6 require PIPELINK_TILELANG_PYTHON and TILELANG_INCLUDE_DIR pointing at the "
                    "pinned, patched TileLang environment; see T4 section 3")
    physical, profile = tutorial_gpu
    coda_profile = "sm" + profile.removeprefix("sm_").removesuffix("a")
    if coda_profile not in ("sm103", "sm90", "sm120"):
        pytest.skip(f"CODA has no producer for {profile}")
    # Separate outputs: campaigns/coda/out/<profile> may hold the owner's receipts.
    out = tmp_path / coda_profile
    command(*UV, "python", CODA / "build.py", "--profile", coda_profile, "--out", out, env=build_env)
    command(*UV, "python", CODA / "run.py", "--profile", coda_profile, "--gpu", physical,
            "--build-dir", out, env={**build_env, "CUDA_VISIBLE_DEVICES": physical})
    assert json.loads((out / "results.json").read_text())["status"] == "PASS"


def test_native_sequence_allocates_once_and_times_complete_pairs(monkeypatch, tmp_path):
    import numpy as np
    from evaluator.harness.parity import native

    calls = []
    handles = iter(range(100, 1000))

    def driver_call(self, name, types, *args):
        if name in ("cuCtxGetCurrent", "cuStreamCreate", "cuMemAlloc_v2",
                    "cuModuleLoadData", "cuModuleGetFunction", "cuEventCreate"):
            args[0]._obj.value = next(handles)
        elif name in ("cuDeviceGet", "cuCtxGetDevice"):
            args[0]._obj.value = 0
        elif name == "cuEventElapsedTime":
            args[0]._obj.value = 6.0
        if name == "cuLaunchKernelEx":
            config = C.cast(args[0], C.POINTER(native._Config)).contents
            calls.append((name, args[1].value, config.stream))
        else:
            calls.append((name,))

    monkeypatch.setattr(native.C, "CDLL", lambda name: object())
    monkeypatch.setattr(native.NativeSession, "_call", driver_call)
    cubin = tmp_path / "plain.cubin"
    cubin.write_bytes(b"test driver fixture")
    seen = []

    def arguments(addresses):
        seen.append(addresses)
        return [C.c_uint64(addresses["intermediate"])]

    buffers = {name: np.zeros(4, dtype=np.uint16) for name in ("a", "intermediate", "output")}
    with native.open_native(cubin, "gemm", buffers=buffers, arguments=arguments, grid=(1, 1, 1),
                            following=({"cubin": cubin, "kernel": "epilogue",
                                        "arguments": arguments, "grid": (1, 1, 1)},)) as session:
        assert seen[0] is seen[1] is session.addresses
        assert sum(call[0] == "cuMemAlloc_v2" for call in calls) == len(buffers)
        assert sum(call[0] == "cuStreamCreate" for call in calls) == 1
        session.launch()
        first_pair = [call for call in calls if call[0] == "cuLaunchKernelEx"]
        assert len(first_pair) == 2 and first_pair[0][1] != first_pair[1][1]
        assert first_pair[0][2] == first_pair[1][2] == session.stream
        calls.clear()
        assert session.time(3) == 6.0
        assert [call for call in calls if call[0] == "cuLaunchKernelEx"] == first_pair * 3
        event_indices = [i for i, call in enumerate(calls) if call[0] == "cuEventRecord"]
        launch_indices = [i for i, call in enumerate(calls) if call[0] == "cuLaunchKernelEx"]
        assert event_indices[0] < min(launch_indices) < max(launch_indices) < event_indices[1]
    assert sum(call[0] == "cuMemFree_v2" for call in calls) == len(buffers)
    assert sum(call[0] == "cuModuleUnload" for call in calls) == 2


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
