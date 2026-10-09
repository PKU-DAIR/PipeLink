#!/usr/bin/env python3
"""Pack two CUDA stages once, then bind ring depths two and four without repacking."""
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

from pipec2.profile import PROFILES
from tools.pipelink_env import load


def pipeline(abi: dict, stages: list[dict], depth: int) -> dict:
    handles = {
        "pipelink_ring_depth": {
            "class": "depth", "binding": "bind", "values": [2, 4], "owner": "channel:tile_ring",
        },
    }
    for name, domain in (("base", "smem"), ("full", "mbarrier"), ("empty", "mbarrier"),
                         ("warp_producer", "warp"), ("warp_consumer", "warp")):
        handle = f"pipelink_ring_{name}"
        handles[handle] = {"class": "base", "binding": "bind", "domain": domain,
                           "owner": f"claim:{handle}"}
    return {
        "pipelink_abi": abi,
        "pipeline": {
            "name": f"ring-pipeline-depth-{depth}", "stages": stages, "handles": handles,
            "phases": ["steady"], "exclusive": [], "domains": {},
            "depth_priority": ["pipelink_ring_depth"], "regs_priority": [],
            "forced_bindings": {"pipelink_ring_depth": depth},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    args = parser.parse_args()
    from evaluator.harness.parity.example import onetu_control
    from pipec2.container import read as read_package

    out = HERE / "out" / args.profile
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": ""}
    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "ring-pipeline" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)

    def command(arguments: list[str], log: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pipec2.cli", *arguments], cwd=REPO, env=env,
            capture_output=True, text=True,
        )
        (logs / log).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Command failed; see {logs / log}\n{result.stderr}")

    stages, pipeline_stages = [], []
    for name in ("producer", "consumer"):
        source = HERE / "src" / f"{name}.cu"
        package = out / f"{name}.plink"
        command(["pack", str(source), "--profile", args.profile, "-o", str(package)], f"{name}.pack.log")
        manifest, _ = read_package(package)
        pipeline_stages.append({"stage_id": manifest["stage"]["id"], "artifact_path": package.name})
        stages.append({"source": source, "includes": [REPO / "pipec/src/pipec2/pack/include"], "defines": {}})

    records = []
    for depth in (2, 4):
        pipeline_path = out / f"pipeline-depth-{depth}.json"
        pipeline_path.write_text(json.dumps(pipeline(manifest["pipelink_abi"], pipeline_stages, depth), indent=2) + "\n")
        link_dir = out / f"depth-{depth}"
        command(["link", str(pipeline_path), "--out-dir", str(link_dir)], f"depth-{depth}.link.log")
        controls = onetu_control(link_dir, stages, profile=args.profile, env=env, log_dir=logs / f"depth-{depth}")
        index = {name: str(Path(path).relative_to(out)) for name, path in controls.items()}
        (link_dir / "controls.json").write_text(json.dumps(index, indent=2) + "\n")
        link = json.loads((link_dir / "link.json").read_text())
        records.append({"depth": depth, "link": str(link_dir / "link.json"), "threads": link["threads"],
                        "dyn_smem_bytes": link["dyn_smem_bytes"], "params": link["params"], "controls": index})
    print(json.dumps({"profile": args.profile, "links": records}, indent=2))


if __name__ == "__main__":
    main()
