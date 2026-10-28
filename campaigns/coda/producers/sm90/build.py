"""GPU-free owner package, provider and native build."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tools.pipelink_env import load


def main():
    from pipec2.container import read
    from pipec2.host.abi import ABI_TYPES, header_path
    from pipec2.host.provider import build_provider
    env = {**os.environ, **load(ROOT), "CUDA_VISIBLE_DEVICES": ""}
    os.environ.update(env)
    here = Path(__file__).resolve().parent
    recipe = json.loads((here / "build.json").read_text())
    out = ROOT / "campaigns/coda/out/sm_90a/producer"
    out.mkdir(parents=True, exist_ok=True)
    flags = [*[v for p in recipe["includes"] for v in ("-I", str(ROOT / p))],
             *[v for k, x in recipe["defines"].items() for v in ("-D", f"{k}={x}")]]
    cmd = [sys.executable, "-m", "pipec2.cli", "pack", str(here / "producer.cu"),
           "--profile", "sm_90a", f"--nvcc-flag=-std={recipe['std']}",
           *[f"--nvcc-flag={x}" for x in recipe["compile_flags"]],
           "-o", str(out / "producer.plink"), *flags]
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    manifest, _ = read(out / "producer.plink")
    host = manifest["host"]
    local = [x for x in host["provider"]["abi_types"] if x["name"] not in ABI_TYPES]
    cuda = Path(env["PIPELINK_CUDA_HOME"])
    inventory = build_provider([here / "provider.cpp"], out / "provider.so", owner="coda_sm90",
        owner_contract_digest=host["provider"]["owner_contract_digest"], profile="sm_90a",
        dtypes_digest=host["target"]["dtypes"], type_ids={x["name"]: x["id"] for x in local},
        extra_types={x["name"]: x["fields"] for x in local}, extra_headers=[str(here / "params.h")],
        compile_flags=["-O3", f"-I{cuda / 'include'}", f"-I{header_path().parent}",
                       f"-L{cuda / 'lib64/stubs'}", "-Wl,--no-as-needed", "-lcuda"])
    if inventory != host["provider"]["abi_types"]:
        raise RuntimeError("Provider ABI differs from package")
    cmd = [str(cuda / "bin/nvcc"), "-O3", "-std=c++20", "-arch=sm_90a", "--cubin",
           "-maxrregcount=168", "-Xptxas=-v", *recipe["compile_flags"], *flags,
           str(ROOT / "campaigns/coda/natives/sm90/native.cu"), "-o", str(out / "native.cubin")]
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    (out / "native-build.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    print(result.stderr)


if __name__ == "__main__":
    main()
