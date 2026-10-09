#!/usr/bin/env python3
"""GPU-free pack, link, and same-source one-TU controls for Hopper split GEMM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from tools.pipelink_env import load


CELLS = (("store", 4), ("bias_relu", 4), ("rowscale", 4), ("store", 2))


def pipeline_spec(abi: dict, stages: list[dict], *, name: str, depth: int) -> dict:
    handles = {
        "pipelink_gemm_depth": {
            "class": "depth", "binding": "bind", "values": [4, 2],
            "owner": "channel:operands",
        },
        "pipelink_launch_regs": {
            "class": "regs", "binding": "bind", "domain": "regs",
            "values": [168], "owner": "reg_group:launch",
        },
    }
    for handle, domain in (
        ("pipelink_gemm_operands", "smem"),
        ("pipelink_gemm_full", "mbarrier"),
        ("pipelink_gemm_empty", "mbarrier"),
        ("pipelink_gemm_output", "smem"),
        ("pipelink_gemm_ready", "mbarrier"),
        ("pipelink_gemm_drained", "mbarrier"),
        ("pipelink_gemm_warp_load", "warp"),
        ("pipelink_gemm_warp_mma", "warp"),
        ("pipelink_gemm_warp_epilogue", "warp"),
    ):
        handles[handle] = {
            "class": "base", "binding": "bind", "domain": domain,
            "owner": f"claim:{handle}",
        }
    return {
        "pipelink_abi": abi,
        "pipeline": {
            "name": f"hopper-gemm-split-{name}",
            "stages": stages,
            "handles": handles,
            "phases": ["steady"],
            "exclusive": [], "domains": {},
            "depth_priority": ["pipelink_gemm_depth"],
            "regs_priority": ["pipelink_launch_regs"],
            "forced_bindings": {"pipelink_gemm_depth": depth},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("sm_90a",), required=True)
    args = parser.parse_args()
    env = {**os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": ""}
    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "hopper-gemm-split" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)

    from evaluator.harness.parity.example import onetu_control
    from pipec2.container import read as read_package

    out = HERE / "out" / args.profile
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    includes = [
        REPO / "third_party/cutlass/include",
        HERE / "src",
        REPO / "pipec/src/pipec2/pack/include",
    ]

    def command(arguments: list[str], log: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pipec2.cli", *arguments], cwd=REPO, env=env,
            capture_output=True, text=True,
        )
        (logs / log).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Command failed; see {logs / log}\n{result.stderr}")

    manifests = {}
    recipes = {}
    # Pack each owner once; every cell links the same load and MMA packages.
    for name in ("load", "mma", "epilogue_store", "epilogue_bias_relu", "epilogue_rowscale"):
        source = HERE / "src" / f"{name}.cu"
        package = out / f"{name}.plink"
        command(
            ["pack", str(source), "--profile", args.profile, "-o", str(package),
             *[flag for include in includes for flag in ("-I", str(include))]],
            f"{name}.pack.log",
        )
        manifests[name], _ = read_package(package)
        recipes[name] = {"source": source, "includes": includes, "defines": {}}

    cells = []
    summaries = []
    for epilogue, depth in CELLS:
        name = f"{epilogue}-d{depth}"
        cell_dir = out / name
        cell_dir.mkdir(parents=True, exist_ok=True)
        stage_names = ("load", "mma", f"epilogue_{epilogue}")
        stages = [
            {"stage_id": manifests[stage]["stage"]["id"], "artifact_path": f"../{stage}.plink"}
            for stage in stage_names
        ]
        pipeline = pipeline_spec(manifests["load"]["pipelink_abi"], stages, name=name, depth=depth)
        pipeline_path = cell_dir / "pipeline.json"
        pipeline_path.write_text(json.dumps(pipeline, indent=2) + "\n")
        command(["link", str(pipeline_path), "--out-dir", str(cell_dir)], f"{name}.link.log")
        controls = onetu_control(
            cell_dir, [recipes[stage] for stage in stage_names], profile=args.profile,
            env=env, log_dir=logs / name,
        )
        cell = {
            "name": name, "epilogue": epilogue, "depth": depth,
            "link": f"{name}/link.json", "linked": f"{name}/pipeline.cubin",
            **{key: str(path.relative_to(out)) for key, path in controls.items()},
        }
        cells.append(cell)
        link = json.loads((cell_dir / "link.json").read_text())
        summaries.append({
            **cell, "threads": link["threads"], "dyn_smem_bytes": link["dyn_smem_bytes"],
            "regs_cap": link["regs_cap"], "params": link["params"],
        })

    index_path = out / "index.json"
    index_path.write_text(json.dumps({"profile": args.profile, "cells": cells}, indent=2) + "\n")
    print(json.dumps({
        "profile": args.profile, "index": str(index_path.relative_to(REPO)), "cells": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
