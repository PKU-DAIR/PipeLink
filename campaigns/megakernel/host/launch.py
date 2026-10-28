#!/usr/bin/env python3
"""Launch an MK image or a persistent native paired comparison.

Each arm's launch facts (symbol, threads, dynamic smem, params/residue, owner,
cluster, opcodes) come from link.json; the provider query supplies the grid.
Paired timing goes through the shared clock/window owner; check and exploratory
bench modes launch directly.
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


def launch_facts(pipeline: str, image: str, *, port_build: Path = PORT) -> dict:
    link = json.loads((port_build / "linked" / pipeline / "link.json").read_text())
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
        cubin, kernel, dyn = port_build / "linked" / pipeline / "pipeline.cubin", link["kernel_symbol"], static["dynamic_smem_bytes"]
    elif image == "onetu":
        cubin, kernel, dyn = port_build / "onetu" / f"{pipeline}.cubin", link["kernel_symbol"], static["dynamic_smem_bytes"]
    elif image == "mono":
        # Upstream kernel: raw llama_1b_globals, upstream opcode numbering (slot k -> k).
        cubin, kernel, dyn = port_build / "mono.cubin", MONO_KERNEL, 216504
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
    *,
    port_build: Path = PORT,
    host_build: Path = HERE / "build",
    runner: Path | None = None,
    provider: Path | None = None,
) -> list[str]:
    f = launch_facts(pipeline, image, port_build=port_build)
    return [
        str((runner or host_build / "mk_runner").resolve()), "--cubin", str(f["cubin"].resolve()), "--kernel", f["kernel"],
        "--dyn-smem", str(f["dyn"]), "--threads", str(f["threads"]),
        "--cluster", ",".join(map(str, f["cluster"])),
        "--params-bytes", str(f["params_bytes"]), "--params-align", str(f["params_align"]),
        "--residue-offset", str(f["residue_offset"]), "--residue-bytes", str(f["residue_bytes"]),
        "--owner-id", str(f["owner_id"]), "--dtypes-digest", f["dtypes_digest"],
        "--owner-contract", f["owner_contract"], "--provider", str((provider or host_build / "provider.so").resolve()),
        "--bundle", str(bundle), "--out", str(out), "--mode", mode,
        "--opcode-map", f["opcode_map"], *runner_args,
    ]


def paired_command(
    pipeline: str,
    image: str,
    reference_pipeline: str,
    reference_image: str,
    bundle: str | Path,
    out: str | Path,
    pos_id: int,
    *,
    mode: str = "pair",
    runner: Path | None = None,
    provider: Path | None = None,
    port_build: Path = PORT,
) -> list[str]:
    """One native process, two manifest-derived images, one persistent fixture."""
    if mode not in ("pair", "pair-check"):
        raise ValueError("paired_command mode must be pair or pair-check")
    if image not in ("linked", "onetu") or reference_image not in ("linked", "onetu"):
        raise ValueError("paired protocol is limited to maintained linked/one-TU images")
    keywords = dict(port_build=port_build, runner=runner, provider=provider)
    args = ("--pos-id", str(pos_id))
    candidate = launch_command(pipeline, image, bundle, out, mode, args, **keywords)
    reference = launch_command(reference_pipeline, reference_image, bundle, out, mode, args, **keywords)
    return [*candidate, "--reference", *reference[1:]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="a")
    ap.add_argument("--image", choices=["linked", "onetu", "mono"], required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--mode", default="check")
    ap.add_argument("--reference-pipeline")
    ap.add_argument("--reference-image", choices=["linked", "onetu"], default="onetu")
    ap.add_argument("--pos-id", type=int)
    ap.add_argument("--runner", type=Path, help="isolated host-only runner binary")
    ap.add_argument("--provider", type=Path, help="reuse an existing provider library")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("runner_args", nargs="*")
    a = ap.parse_args()
    if a.mode in ("pair", "pair-check"):
        if a.pos_id is None or a.runner_args:
            ap.error("paired modes require --pos-id and accept no legacy runner arguments")
        cmd = paired_command(
            a.pipeline, a.image, a.reference_pipeline or a.pipeline, a.reference_image,
            Path(a.bundle).resolve(), Path(a.out).resolve(), a.pos_id, mode=a.mode,
            runner=a.runner, provider=a.provider,
        )
    else:
        if a.pos_id is not None:
            a.runner_args += ["--pos-id", str(a.pos_id)]
        cmd = launch_command(a.pipeline, a.image, a.bundle, a.out, a.mode, tuple(a.runner_args),
                             runner=a.runner, provider=a.provider)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    if a.print_only:
        print(" ".join(cmd))
        return 0
    Path(a.out).mkdir(parents=True, exist_ok=a.mode not in ("pair", "pair-check"))
    (Path(a.out) / "launch.json").write_text(json.dumps({"cmd": cmd, "gpu": a.gpu}, indent=2))
    if a.mode == "pair":
        sys.path.insert(0, str(HERE.parents[2]))
        from evaluator.harness.parity.fixed_count import run_native_pair
        # Empty or 0 PIPELINK_LOCK_SM_MHZ runs with unlocked clocks.
        clock_mhz = int(os.environ.get("PIPELINK_LOCK_SM_MHZ") or 0)
        execution = run_native_pair(
            cmd, env=env, gpus=[int(a.gpu)], clock_mhz=clock_mhz,
            out_dir=Path(a.out).resolve() / "execution", timeout=600,
        )
        print(json.dumps(execution))
        return 0 if execution["admitted"] else 1
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
