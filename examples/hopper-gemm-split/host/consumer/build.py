#!/usr/bin/env python3
"""Build a new consumer from its sources and the four-file opaque GEMM distribution."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import pipec2
from pipec2.container import read as read_package
from pipec2.host.abi import ABI_TYPES
from pipec2.host.provider import build_provider

HERE = Path(__file__).resolve().parent
SDK = Path(pipec2.__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--forbid-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    owner, out = args.owner.resolve(), args.out.resolve()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    cuda = Path(env["PIPELINK_CUDA_HOME"])
    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "hopper-gemm-split" / "consumer-compile"
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    os.environ.update(env)
    tempfile.tempdir = str(scratch)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    includes = [HERE, owner, SDK / "pack/include", SDK / "host/include"]

    def command(argv, log):
        completed = subprocess.run(list(map(str, argv)), cwd=HERE, env=env,
                                   capture_output=True, text=True)
        (logs / log).write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"Command failed: {logs / log}\n{completed.stderr}")
        return completed.stdout

    cli = [sys.executable, "-m", "pipec2.cli"]
    package = out / "epilogue.plink"
    command([*cli, "pack", HERE / "scaled_store.cu", "--profile", "sm_90a", "-o", package,
             *[flag for root in includes for flag in ("-I", root)]], "pack.log")
    consumer, _ = read_package(package)
    load, _ = read_package(owner / "load.plink")
    mma, _ = read_package(owner / "mma.plink")
    host = consumer["host"]
    inventory = host["provider"]["abi_types"]
    local = [row for row in inventory if row["name"] not in ABI_TYPES]
    observed = build_provider(
        [HERE / "provider.cpp"], out / "user.so", owner="user",
        owner_contract_digest=host["provider"]["owner_contract_digest"],
        profile="sm_90a", dtypes_digest=host["target"]["dtypes"],
        type_ids={row["name"]: row["id"] for row in local},
        extra_types={row["name"]: row["fields"] for row in local},
        extra_headers=[str(HERE / "epilogue.h")],
        compile_flags=["-O3", *[f"-I{root}" for root in includes]],
    )
    if observed != inventory:
        raise RuntimeError("Consumer's independently compiled ABI differs from its package")

    handles = {
        "pipelink_gemm_depth": {"class": "depth", "binding": "bind", "values": [4, 2],
                                "owner": "channel:operands"},
        "pipelink_launch_regs": {"class": "regs", "binding": "bind", "domain": "regs",
                                 "values": [168], "owner": "reg_group:launch"},
    }
    for name, domain in (("operands", "smem"), ("full", "mbarrier"), ("empty", "mbarrier"),
                         ("output", "smem"), ("ready", "mbarrier"), ("drained", "mbarrier"),
                         ("warp_load", "warp"), ("warp_mma", "warp"), ("warp_epilogue", "warp")):
        handle = "pipelink_gemm_" + name
        handles[handle] = {"class": "base", "binding": "bind", "domain": domain,
                           "owner": "claim:" + handle}
    pipeline = {"pipelink_abi": load["pipelink_abi"], "pipeline": {
        "name": "hopper-gemm-host-scaled-store", "stages": [
            {"stage_id": manifest["stage"]["id"], "artifact_path": os.path.relpath(path, out)}
            for manifest, path in ((load, owner / "load.plink"), (mma, owner / "mma.plink"),
                                   (consumer, package))],
        "handles": handles, "phases": ["steady"], "exclusive": [], "domains": {},
        "depth_priority": ["pipelink_gemm_depth"], "regs_priority": ["pipelink_launch_regs"],
        "forced_bindings": {"pipelink_gemm_depth": 4},
        "topology": {"persistent": False, "scheduler": "static"}}}
    spec = out / "pipeline.json"
    spec.write_text(json.dumps(pipeline, indent=2) + "\n")
    command([*cli, "link", spec, "--out-dir", out.parent / "link"], "link.log")

    dependencies = {}
    for name in ("scaled_store.cu", "provider.cpp"):
        compiler = cuda / "bin/nvcc" if name.endswith(".cu") else Path("g++")
        extra = ["-arch=sm_90a"] if name.endswith(".cu") else []
        text = command([compiler, "-std=c++17", *extra, *[f"-I{root}" for root in includes],
                        "-M", HERE / name], name + ".dependencies")
        paths = sorted(set(re.findall(r"(?:/[^\s\\]+)", text)))
        forbidden = [path for path in paths if any(
            Path(path).is_relative_to(root.resolve()) for root in args.forbid_root)]
        if forbidden:
            raise RuntimeError(f"Consumer accessed withheld owner source: {forbidden}")
        dependencies[name] = paths
    report = {"status": "PASS", "owner_artifacts": ["load.plink", "mma.plink", "provider.so", "gemm.h"],
              "consumer_sources": ["scaled_store.cu", "provider.cpp", "epilogue.h", "build.py"],
              "dependencies": dependencies, "neutral_inventory_restatements": 0,
              "local_abi_types": len(local), "total_abi_types": len(inventory)}
    (out / "source-withheld.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "consumer": str(package),
                      "provider": str(out / "user.so"), "link": str(out.parent / "link/link.json")}))


if __name__ == "__main__":
    main()
