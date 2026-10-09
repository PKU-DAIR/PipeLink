#!/usr/bin/env python3
"""Build the GEMM host owner, source-withheld consumer, and same-source control."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from tools.pipelink_env import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("sm_90a",), required=True)
    parser.add_argument("--owner-only", action="store_true", help="produce only the four-file owner distribution")
    args = parser.parse_args()
    from pipec2.container import read as read_package
    from pipec2.host.abi import ABI_TYPES, header_path
    from pipec2.host.provider import build_provider

    env = {**os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": ""}
    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "hopper-gemm-split" / "host-build"
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    os.environ.update(env)
    tempfile.tempdir = str(scratch)
    cuda = Path(env["PIPELINK_CUDA_HOME"])
    out = HERE / "out/host-sm_90a"
    owner = out / "owner"
    logs = out / "logs"
    owner.mkdir(parents=True, exist_ok=True)
    logs.mkdir(exist_ok=True)
    sdk = header_path().parent.parent.parent
    includes = [HERE / "src", REPO / "third_party/cutlass/include", sdk / "pack/include", header_path().parent]

    def command(argv, log, *, cwd=REPO):
        completed = subprocess.run(list(map(str, argv)), cwd=cwd, env=env, capture_output=True, text=True)
        (logs / log).write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"Command failed: {logs / log}\n{completed.stderr}")
        return completed.stdout

    cli = [sys.executable, "-m", "pipec2.cli"]
    manifests = {}
    for name in ("load", "mma"):
        command([*cli, "pack", HERE / "src" / f"{name}.cu", "--profile", args.profile,
                 "-D", "GEMM_HOST_ABI=1", "-o", owner / f"{name}.plink",
                 *[flag for root in includes for flag in ("-I", root)]], name + ".pack.log")
        manifests[name], _ = read_package(owner / f"{name}.plink")
    host = manifests["load"]["host"]
    inventory = host["provider"]["abi_types"]
    local = [row for row in inventory if row["name"] not in ABI_TYPES]
    observed = build_provider(
        [HERE / "host/owner/provider.cpp"], owner / "provider.so", owner="gemm",
        owner_contract_digest=host["provider"]["owner_contract_digest"], profile=args.profile,
        dtypes_digest=host["target"]["dtypes"],
        type_ids={row["name"]: row["id"] for row in local},
        extra_types={row["name"]: row["fields"] for row in local},
        extra_headers=[str(HERE / "host/owner/abi.h")],
        compile_flags=["-O3", f"-I{cuda / 'include'}", f"-I{header_path().parent}",
                       f"-L{cuda / 'lib64/stubs'}", "-Wl,--no-as-needed", "-lcuda"],
    )
    if observed != inventory:
        raise RuntimeError("Owner's independently compiled ABI differs from its package")
    shutil.copy2(HERE / "include/gemm.h", owner / "gemm.h")
    if args.owner_only:
        print(json.dumps({"status": "PASS", "owner": str(owner),
                          "files": ["load.plink", "mma.plink", "provider.so", "gemm.h"]}))
        return

    # This tree has only the published owner artifacts and the consumer's code.
    # Compile/link inside it, and audit compiler dependencies against both private roots.
    with tempfile.TemporaryDirectory(prefix="withheld-", dir=scratch) as temporary:
        tree = Path(temporary)
        shutil.copytree(owner, tree / "owner")
        shutil.copytree(HERE / "host/consumer", tree / "consumer-src")
        command([sys.executable, tree / "consumer-src/build.py", "--owner", tree / "owner",
                 "--out", tree / "consumer", "--forbid-root", HERE / "src",
                 "--forbid-root", HERE / "host/owner"], "consumer-withheld.log", cwd=tree)
        shutil.copytree(tree / "consumer", out / "consumer", dirs_exist_ok=True)
        shutil.copytree(tree / "link", out / "link", dirs_exist_ok=True)

    # As in dequant-splice's control: structural role-view types in generated
    # dispatch are namespaced, while exact owner/consumer bodies retain C symbols.
    link_dir = out / "link"
    link = json.loads((link_dir / "link.json").read_text())
    generated = link_dir / "generated"
    control = link_dir / "one-tu.cu"
    control.write_text(
        f'#include "{HERE / "src/load.cu"}"\n'
        f'#include "{HERE / "src/mma.cu"}"\n'
        f'#include "{HERE / "host/consumer/scaled_store.cu"}"\n'
        f'#include "{generated / "binding_stub.cu"}"\n'
        'namespace gemm_control_dispatch {\n'
        f'#include "{generated / "dispatch.cu"}"\n'
        '}\n'
    )
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-DGEMM_HOST_ABI=1",
             f"-maxrregcount={link['regs_cap']}",
             *[f"-I{root}" for root in [*includes, owner, HERE / "host/consumer"]]]
    nvcc = cuda / "bin/nvcc"
    command([nvcc, *flags, "-arch=sm_90a", "--cubin", control,
             "-o", link_dir / "one-tu.cubin"], "one-tu.log")
    command([nvcc, *flags, "-gencode=arch=compute_90a,code=lto_90a", "-dc", "-rdc=true", control,
             "-o", link_dir / "one-tu-lto.o"], "one-tu-lto.log")
    command([nvcc, "-dlink", "-dlto", "-arch=sm_90a", f"-maxrregcount={link['regs_cap']}",
             "-cubin", link_dir / "one-tu-lto.o", "-o", link_dir / "one-tu-lto.cubin"], "one-tu-lto-link.log")
    print(json.dumps({"status": "PASS", "owner": str(owner), "consumer_source_withheld": True,
                      "link": str(link_dir / "link.json"), "threads": link["threads"],
                      "dynamic_smem_bytes": link["dyn_smem_bytes"],
                      "owner_local_abi_types": len(local), "neutral_restatements": 0}, indent=2))


if __name__ == "__main__":
    main()
