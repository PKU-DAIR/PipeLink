"""Pack the owner, build its ABI-v3 provider, and compile plain native cubins."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tools.pipelink_env import load
from pipec2.container import read
from pipec2.host.abi import ABI_TYPES, header_path
from pipec2.host.provider import build_provider


def main():
    os.environ.update({**load(ROOT), **os.environ})
    here = Path(__file__).resolve().parent
    out = here / "out"
    out.mkdir(exist_ok=True)
    cuda = Path(os.environ["PIPELINK_CUDA_HOME"])
    recipe = json.loads((here / "build.json").read_text())
    includes = [ROOT / p for p in recipe["includes"]]
    flags = ["-std=" + recipe["std"], *recipe["compile_flags"]]
    defines = [f"{k}={v}" for k, v in recipe["defines"].items()]
    cmd = [sys.executable, "-m", "pipec2.cli", "pack", str(here / "producer.cu"),
           "--profile", "sm_103a", "-o", str(out / "producer.plink")]
    cmd += [a for p in includes for a in ("-I", str(p))]
    cmd += [a for d in defines for a in ("-D", d)]
    cmd += ["--nvcc-flag=" + f for f in flags]
    subprocess.run(cmd, check=True, cwd=ROOT)
    manifest, _ = read(out / "producer.plink")
    host = manifest["host"]
    inventory = host["provider"]["abi_types"]
    local = [r for r in inventory if r["name"] not in ABI_TYPES]
    observed = build_provider([here / "provider.cpp"], out / "provider.so", owner="coda_sm103",
        owner_contract_digest=host["provider"]["owner_contract_digest"], profile="sm_103a",
        dtypes_digest=host["target"]["dtypes"],
        type_ids={r["name"]:r["id"] for r in local}, extra_types={r["name"]:r["fields"] for r in local},
        extra_headers=[str(here / "params.h")],
        compile_flags=["-O3",f"-I{cuda/'include'}",f"-I{header_path().parent}",
                       f"-L{cuda/'lib64/stubs'}","-Wl,--no-as-needed","-lcuda"])
    if observed != inventory:
        raise RuntimeError("Provider ABI does not match package")
    native = ROOT / "campaigns/coda/natives/sm103"
    (native / "out").mkdir(exist_ok=True)
    subprocess.run([str(cuda/"bin/nvcc"),*flags,"-O3","-gencode=arch=compute_103a,code=sm_103a",
        *["-I"+str(p) for p in includes],*["-D"+d for d in defines],
        "--ptxas-options=-v","--cubin",str(native/"native.cu"),"-o",str(native/"out/native.cubin")],check=True,cwd=ROOT)
    print(json.dumps({"package":str(out/"producer.plink"),"provider":str(out/"provider.so"),
                      "native":str(native/"out/native.cubin")}))

if __name__ == "__main__":
    main()
