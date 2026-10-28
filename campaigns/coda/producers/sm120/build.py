#!/usr/bin/env python3
"""Pack the sm120 owner and compile its host provider and native witnesses."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from tools.pipelink_env import load
from pipec2.container import read as read_package
from pipec2.host.abi import ABI_TYPES, header_path
from pipec2.host.provider import build_provider


def main():
    env = {**os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": ""}
    os.environ.update(env)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    includes = [REPO / "third_party/cutlass/include", REPO / "pipec/src/pipec2/pack/include",
                header_path().parent]
    command = [sys.executable, "-m", "pipec2.cli", "pack", str(HERE / "producer.cu"),
               "--profile", "sm_120a", "-o", str(out / "producer.plink")]
    for root in includes:
        command += ["-I", str(root)]
    result = subprocess.run(command, env=env, cwd=REPO, capture_output=True, text=True)
    (out / "pack.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    manifest, _ = read_package(out / "producer.plink")
    host = manifest["host"]
    inventory = host["provider"]["abi_types"]
    local = [r for r in inventory if r["name"] not in ABI_TYPES]
    build_provider([HERE / "provider.cpp"], out / "provider.so", owner="coda_sm120",
        owner_contract_digest=host["provider"]["owner_contract_digest"], profile="sm_120a",
        dtypes_digest=host["target"]["dtypes"],
        type_ids={r["name"]: r["id"] for r in local},
        extra_types={r["name"]: r["fields"] for r in local}, extra_headers=[str(HERE / "abi.h")],
        compile_flags=["-O3", f"-I{header_path().parent}", f"-I{includes[1]}"])
    source = REPO / "campaigns/coda/natives/sm120/kernels.cu"
    native = source.parent / "out"
    native.mkdir(exist_ok=True)
    command = [str(Path(env["PIPELINK_CUDA_HOME"]) / "bin/nvcc"), "-std=c++17", "-O3", "-DNDEBUG",
               "-arch=sm_120a", "--cubin", "-Xptxas=-v", f"-I{includes[0]}", str(source),
               "-o", str(native / "kernels.cubin")]
    result = subprocess.run(command, env=env, cwd=REPO, capture_output=True, text=True)
    (native / "compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    command = [str(Path(env["PIPELINK_CUDA_HOME"]) / "bin/nvcc"), "-std=c++17", "-O3", "-DNDEBUG",
               "-arch=sm_120a", "--shared", "-Xcompiler=-fPIC", f"-I{includes[0]}",
               str(source.parent / "reference.cu"), "-o", str(native / "reference.so")]
    result = subprocess.run(command, env=env, cwd=REPO, capture_output=True, text=True)
    (native / "reference.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    print(json.dumps({"producer": str(out / "producer.plink"), "provider": str(out / "provider.so"),
                      "natives": str(native / "kernels.cubin")}))


if __name__ == "__main__":
    main()
