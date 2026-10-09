#!/usr/bin/env python3
"""GPU-free pack, providers, links and same-source one-TU controls for comm-overlap.

Compositions:
  ag   comm.gather + gemm.load.ag + gemm.mma (repacked under gemm.owner.ag) + gemm.epilogue.store
  rs   gemm.load + gemm.mma (hopper-gemm-split host packages, unchanged) + comm.epilogue.rs
  base gemm.load + gemm.mma (unchanged) + gemm.epilogue.store: the GEMM of the sequential baselines

Selecting ag or rs also builds base for sequential-baseline timings. Selected
local packages/providers and controls are rebuilt; the frozen GEMM owner
distribution is copied unchanged (built at its source only if absent).
--out selects a self-contained candidate root. RS width applies equally to the
package and both same-source controls and is recorded in index.json. RS always
builds the shared owner-local follow-on reduction kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEMM = REPO / "examples/hopper-gemm-split"
sys.path.insert(0, str(REPO))
from tools.pipelink_env import load  # noqa: E402

MAX_ROW_BLOCKS = 512  # gmem_counter capacity = the comm owner's slot claim


def handles(extra: dict[str, str]) -> dict:
    result = {
        "pipelink_gemm_depth": {"class": "depth", "binding": "bind", "values": [4, 2],
                                "owner": "channel:operands"},
        "pipelink_launch_regs": {"class": "regs", "binding": "bind", "domain": "regs",
                                 "values": [168], "owner": "reg_group:launch"},
    }
    names = {"operands": "smem", "full": "mbarrier", "empty": "mbarrier", "output": "smem",
             "ready": "mbarrier", "drained": "mbarrier", "warp_load": "warp", "warp_mma": "warp",
             "warp_epilogue": "warp"}
    for name, domain in [*((f"pipelink_gemm_{k}", v) for k, v in names.items()), *extra.items()]:
        result[name] = {"class": "base", "binding": "bind", "domain": domain, "owner": f"claim:{name}"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("sm_90a",), required=True)
    parser.add_argument("--rs-warps", type=int, choices=(1, 2, 3), default=2,
                        help="RS communication warps (default: 2)")
    parser.add_argument("--out", type=Path, help="artifact root (default: out/<profile>)")
    parser.add_argument("--composition", choices=("all", "ag", "rs", "base"), default="all")
    args = parser.parse_args()
    selected = {"ag", "rs", "base"} if args.composition == "all" else {args.composition, "base"}
    rs_defines = {"COMM_RS_WARPS": str(args.rs_warps)}
    env = {**os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": ""}
    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "comm-overlap" / "build"
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    os.environ.update(env)
    from pipec2.container import read as read_package
    from pipec2.host.abi import ABI_TYPES, header_path
    from pipec2.host.provider import build_provider

    cuda = Path(env["PIPELINK_CUDA_HOME"])
    out = args.out.resolve() if args.out is not None else HERE / "out" / args.profile
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sdk = header_path().parent.parent.parent
    includes = [REPO / "third_party/cutlass/include", GEMM / "src", sdk / "pack/include",
                header_path().parent, GEMM, HERE / "comm", HERE / "gemm_ag"]

    def command(argv, log):
        completed = subprocess.run(list(map(str, argv)), cwd=REPO, env=env, capture_output=True, text=True)
        (logs / log).write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"Command failed: {logs / log}\n{completed.stderr[-4000:]}")
        return completed.stdout

    cli = [sys.executable, "-m", "pipec2.cli"]

    # The GEMM owner's existing four-file host distribution (load, mma, provider, gemm.h).
    owner = GEMM / "out/host-sm_90a/owner"
    if not all((owner / name).exists() for name in ("load.plink", "mma.plink", "provider.so")):
        command([sys.executable, GEMM / "build_host.py", "--profile", args.profile, "--owner-only"],
                "gemm-owner.log")
    # The links consume a copy of the shipped distribution, so out/ is self-contained.
    shutil.copytree(owner, out / "gemm-owner", dirs_exist_ok=True)
    owner = out / "gemm-owner"

    # The repacked MMA is hopper-gemm-split's src/mma.cu, byte for byte. Staging it beside a
    # host/owner/host_annotations.cuh that forwards to the AG annotations lets its own relative
    # include select the gemm.owner.ag host group without editing the owner's source.
    staged = out / "gemm-ag-src"
    if "ag" in selected:
        (staged / "src").mkdir(parents=True, exist_ok=True)
        (staged / "host/owner").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(GEMM / "src/mma.cu", staged / "src/mma.cu")
        (staged / "host/owner/host_annotations.cuh").write_text(
            f'#pragma once\n#include "{HERE / "gemm_ag/host_annotations.cuh"}"\n')
    mma_digest = hashlib.sha256((GEMM / "src/mma.cu").read_bytes()).hexdigest()

    host_abi = {"GEMM_HOST_ABI": "1"}
    recipes = {
        "comm_gather": (HERE / "comm/ag_gather.cu", {}),
        "epilogue_rs": (HERE / "comm/epilogue_rs.cu", rs_defines),
        "load_ag": (HERE / "gemm_ag/load_ag.cu", host_abi),
        "mma_ag": (staged / "src/mma.cu", host_abi),
        "epilogue_store": (GEMM / "src/epilogue_store.cu", {}),
    }
    needed = {"epilogue_store"}
    if "ag" in selected:
        needed.update(("comm_gather", "load_ag", "mma_ag"))
    if "rs" in selected:
        needed.add("epilogue_rs")
    manifests = {}
    for name, (source, defines) in recipes.items():
        if name not in needed:
            continue
        command([*cli, "pack", source, "--profile", args.profile, "-o", out / f"{name}.plink",
                 *[f for d in defines.items() for f in ("-D", "=".join(d))],
                 *[f for root in includes for f in ("-I", root)]], f"{name}.pack.log")
        manifests[name], _ = read_package(out / f"{name}.plink")
    for name in ("load", "mma"):
        manifests[name], _ = read_package(owner / f"{name}.plink")

    def provider(package, sources, so, suffix, headers, flags):
        host = manifests[package]["host"]
        inventory = host["provider"]["abi_types"]
        local = [row for row in inventory if row["name"] not in ABI_TYPES]
        observed = build_provider(
            sources, out / so, owner=suffix, owner_contract_digest=host["provider"]["owner_contract_digest"],
            profile=args.profile, dtypes_digest=host["target"]["dtypes"],
            type_ids={row["name"]: row["id"] for row in local},
            extra_types={row["name"]: row["fields"] for row in local},
            extra_headers=headers, compile_flags=["-O3", f"-I{header_path().parent}", *flags])
        if observed != inventory:
            raise RuntimeError(f"{so}: independently compiled ABI differs from its package")
        return len(local)

    cuda_flags = [f"-I{cuda / 'include'}", f"-L{cuda / 'lib64/stubs'}", "-Wl,--no-as-needed", "-lcuda"]
    local_types = {}
    if "ag" in selected:
        local_types["comm.owner.ag"] = provider(
            "comm_gather", [HERE / "comm/provider_ag.cpp"], "comm_ag.so", "comm_ag",
            [HERE / "comm/comm.h"], [])
        local_types["gemm.owner.ag"] = provider(
            "load_ag", [HERE / "gemm_ag/provider.cpp"], "gemm_ag.so", "gemm_ag",
            [HERE / "gemm_ag/abi.h"], [f"-I{GEMM}", *cuda_flags])
    if "rs" in selected:
        local_types["comm.owner.rs"] = provider(
            "epilogue_rs", [HERE / "comm/provider_rs.cpp"], "comm_rs.so", "comm_rs",
            [HERE / "comm/comm.h"], [])

    compositions = {
        "ag": {"stages": [("comm_gather", out / "comm_gather.plink"), ("load_ag", out / "load_ag.plink"),
                          ("mma_ag", out / "mma_ag.plink"), ("epilogue_store", out / "epilogue_store.plink")],
               "extra": {"pipelink_comm_warp": "warp", "pipelink_comm_gather_reads_done": "mbarrier",
                         "pipelink_comm_ready_slot": "gmem_counter"},
               "domains": {"gmem_counter": {"capacity": MAX_ROW_BLOCKS}},
               "providers": {"comm.owner.ag": "comm_ag.so", "gemm.owner.ag": "gemm_ag.so"},
               "blobs": ["gemm.epilogue.store"]},
        "rs": {"stages": [("load", owner / "load.plink"), ("mma", owner / "mma.plink"),
                          ("epilogue_rs", out / "epilogue_rs.plink")],
               "extra": {"pipelink_rs_barrier": "named_barrier"},
               "domains": {},
               "providers": {"gemm.owner.params": str(owner / "provider.so"), "comm.owner.rs": "comm_rs.so"},
               "blobs": []},
        "base": {"stages": [("load", owner / "load.plink"), ("mma", owner / "mma.plink"),
                            ("epilogue_store", out / "epilogue_store.plink")],
                 "extra": {}, "domains": {},
                 "providers": {"gemm.owner.params": str(owner / "provider.so")},
                 "blobs": ["gemm.epilogue.store"]},
    }
    one_tu_sources = {"comm_gather": recipes["comm_gather"][0], "epilogue_rs": recipes["epilogue_rs"][0],
                      "load_ag": recipes["load_ag"][0], "mma_ag": recipes["mma_ag"][0],
                      "epilogue_store": recipes["epilogue_store"][0],
                      "load": GEMM / "src/load.cu", "mma": GEMM / "src/mma.cu"}
    nvcc = cuda / "bin/nvcc"
    index = {"profile": args.profile, "mma_source_sha256": mma_digest, "compositions": {},
             "provider_local_abi_types": local_types, "rs_warps": args.rs_warps,
             "rs_route": "follow-on"}
    for name, spec in compositions.items():
        if name not in selected:
            continue
        link_dir = out / name
        link_dir.mkdir(parents=True, exist_ok=True)
        pipeline = {"pipelink_abi": manifests["load"]["pipelink_abi"], "pipeline": {
            "name": f"comm-overlap-{name}",
            "stages": [{"stage_id": manifests[stage]["stage"]["id"], "artifact_path": os.path.relpath(path, link_dir)}
                       for stage, path in spec["stages"]],
            "handles": handles(spec["extra"]), "phases": ["steady"], "exclusive": [],
            "domains": spec["domains"], "depth_priority": ["pipelink_gemm_depth"],
            "regs_priority": ["pipelink_launch_regs"], "forced_bindings": {"pipelink_gemm_depth": 4},
            "topology": {"persistent": False, "scheduler": "static"}}}
        (link_dir / "pipeline.json").write_text(json.dumps(pipeline, indent=2) + "\n")
        command([*cli, "link", link_dir / "pipeline.json", "--out-dir", link_dir], f"{name}.link.log")
        link = json.loads((link_dir / "link.json").read_text())
        # Same-source control: every stage source in one TU with the generated binding stub;
        # generated dispatch is namespaced because its role-view structs restate owner types.
        generated = link_dir / "generated"
        control = link_dir / "one-tu.cu"
        control.write_text(
            "".join(f'#include "{one_tu_sources[stage]}"\n' for stage, _ in spec["stages"])
            + f'#include "{generated / "binding_stub.cu"}"\n'
            + "namespace comm_overlap_control_dispatch {\n"
            + f'#include "{generated / "dispatch.cu"}"\n' + "}\n")
        flags = ["-std=c++17", "-O3", "-DNDEBUG", "-DGEMM_HOST_ABI=1", f"-maxrregcount={link['regs_cap']}",
                 *[f"-I{root}" for root in includes]]
        if name == "rs":
            flags.extend(f"-D{key}={value}" for key, value in rs_defines.items())
        command([nvcc, *flags, f"-arch={args.profile}", "--cubin", control, "-o", link_dir / "one-tu.cubin"],
                f"{name}.one-tu.log")
        command([nvcc, *flags, "-gencode=arch=compute_90a,code=lto_90a", "-dc", "-rdc=true", control,
                 "-o", link_dir / "one-tu-lto.o"], f"{name}.one-tu-lto.log")
        command([nvcc, "-dlink", "-dlto", f"-arch={args.profile}", f"-maxrregcount={link['regs_cap']}",
                 "-cubin", link_dir / "one-tu-lto.o", "-o", link_dir / "one-tu-lto.cubin"],
                f"{name}.one-tu-lto-link.log")
        index["compositions"][name] = {
            "link": f"{name}/link.json", "linked": f"{name}/pipeline.cubin", "one_tu": f"{name}/one-tu.cubin",
            "one_tu_lto": f"{name}/one-tu-lto.cubin",
            "providers": {group: os.path.relpath(out / so, out) for group, so in spec["providers"].items()},
            "blobs": spec["blobs"], "threads": link["threads"], "dyn_smem_bytes": link["dyn_smem_bytes"],
            "regs_cap": link["regs_cap"], "kernel_symbol": link.get("kernel_symbol"),
        }
        if name == "rs":
            index["compositions"][name].update(rs_warps=args.rs_warps, rs_route="follow-on")
            command([nvcc, "-std=c++17", "-O3", f"-arch={args.profile}", "--cubin",
                     *[f"-I{root}" for root in (header_path().parent, HERE / "comm", cuda / "include")],
                     *[f"-D{key}={value}" for key, value in rs_defines.items()],
                     HERE / "comm/reduce_rs.cu", "-o", link_dir / "reduce.cubin"], "rs.reduce.log")
            index["compositions"][name]["rs_completion"] = {
                "cubin": "rs/reduce.cubin", "symbol": "comm_rs_reduce", "threads": 128,
            }
    command([nvcc, "-std=c++17", "-O3", f"-arch={args.profile}", "--cubin", HERE / "helpers.cu",
             "-o", out / "helpers.cubin"], "helpers.log")
    index["annotations"] = annotation_counts()
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


def annotation_counts() -> dict:
    """Authoring burden: annotation statements per new source (device vs host families)."""
    import re
    files = {"comm.gather": HERE / "comm/ag_gather.cu", "comm.epilogue.rs": HERE / "comm/epilogue_rs.cu",
             "gemm.load.ag": HERE / "gemm_ag/load_ag.cu",
             "gemm.owner.ag host group": HERE / "gemm_ag/host_annotations.cuh"}
    counts = {}
    for name, path in files.items():
        text = re.sub(r"//[^\n]*", "", path.read_text())
        macros = re.findall(r"^\s*(PL_[A-Z_]+|GEMM_[A-Z_]+)\(", text, re.MULTILINE)
        counts[name] = {
            "device": sum(not m.startswith("PL_HOST_") and not m.startswith("GEMM_") for m in macros),
            "host": sum(m.startswith("PL_HOST_") for m in macros),
            "shared_gemm_macros": sum(m.startswith("GEMM_") for m in macros),
        }
    return counts


if __name__ == "__main__":
    main()
