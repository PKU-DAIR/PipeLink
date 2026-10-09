#!/usr/bin/env python3
"""Launch one MK image (linked pipeline, one-TU control, or upstream mono) through mk_runner.

Derives every launch fact from the pipeline's link.json (kernel symbol, threads, dynamic smem,
params/residue layout, owner identity, cluster, opcode bindings). The provider query supplies grid.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = HERE.parent / "port" / "build"
MONO_KERNEL = (
    "_ZN10megakernel2mkINS_14default_configE9globals_tILi16ELi2048ELi8192ELi64ELi32ELi8ELi16ELi16ELi148EE"
    "J17attention_partialIS1_S3_E19attention_reductionIS1_S3_E19rms_qkv_rope_appendIS1_S3_E8downprojIS1_S3_E"
    "6o_projIS1_S3_E15rms_upgate_siluIS1_S3_E11rms_lm_headIS1_S3_EEEEvT0_"
)


def launch_facts(pipeline: str, image: str) -> dict:
    link = json.loads((PORT / "linked" / pipeline / "link.json").read_text())
    host = link["host"]
    group = host["groups"][0]
    residue = group["residue"]
    static = host["static_launch"]
    facts = dict(params_bytes=host["params"]["bytes"], params_align=host["params"]["align"],
                 residue_offset=residue["offset"], residue_bytes=residue["bytes"],
                 owner_id=group["owner_id"], dtypes_digest=host["target"]["dtypes"],
                 owner_contract=group["provider"]["owner_contract_digest"],
                 threads=static["block"][0], cluster=static["cluster"])
    opcode_map = ",".join(
        f"{k}:{link['bindings'][f'pipelink_megakernel_opcode_slot{k}'][0]}" for k in range(8)
    )
    if image == "linked":
        cubin, kernel, dyn = PORT / "linked" / pipeline / "pipeline.cubin", link["kernel_symbol"], static["dynamic_smem_bytes"]
    elif image == "onetu":
        cubin, kernel, dyn = PORT / "onetu" / f"{pipeline}.cubin", link["kernel_symbol"], static["dynamic_smem_bytes"]
    elif image == "mono":
        # Upstream kernel: raw llama_1b_globals, upstream opcode numbering (slot k -> k).
        cubin, kernel, dyn = PORT / "mono.cubin", MONO_KERNEL, 216504
        opcode_map = ",".join(f"{k}:{k}" for k in range(8))
        facts.update(params_bytes=residue["bytes"], params_align=residue["align"],
                     residue_offset=0, threads=640, cluster=[1, 1, 1])
    else:
        raise SystemExit(f"unknown image {image}")
    return dict(cubin=cubin, kernel=kernel, dyn=dyn, opcode_map=opcode_map, **facts)


def launch_command(
    pipeline: str,
    image: str,
    bundle: str | Path,
    out: str | Path,
    mode: str,
    runner_args: tuple[str, ...] = (),
) -> list[str]:
    f = launch_facts(pipeline, image)
    return [
        str(HERE / "build" / "mk_runner"), "--cubin", str(f["cubin"]), "--kernel", f["kernel"],
        "--dyn-smem", str(f["dyn"]), "--threads", str(f["threads"]),
        "--cluster", ",".join(map(str, f["cluster"])),
        "--params-bytes", str(f["params_bytes"]), "--params-align", str(f["params_align"]),
        "--residue-offset", str(f["residue_offset"]), "--residue-bytes", str(f["residue_bytes"]),
        "--owner-id", str(f["owner_id"]), "--dtypes-digest", f["dtypes_digest"],
        "--owner-contract", f["owner_contract"], "--provider", str(HERE / "build" / "provider.so"),
        "--bundle", str(bundle), "--out", str(out), "--mode", mode,
        "--opcode-map", f["opcode_map"], *runner_args,
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="a")
    ap.add_argument("--image", choices=["linked", "onetu", "mono"], required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--mode", default="check")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("runner_args", nargs="*")
    a = ap.parse_args()
    cmd = launch_command(a.pipeline, a.image, a.bundle, a.out, a.mode, tuple(a.runner_args))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    if a.print_only:
        print(" ".join(cmd))
        return 0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "launch.json").write_text(json.dumps({"cmd": cmd, "gpu": a.gpu}, indent=2))
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
