#!/usr/bin/env python3
"""Host-less verification, paired timing, sanitizers, diagnostic SASS, and native-fused timing."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipec/src"))
from evaluator.harness.parity.example import compare, sanitize, sass_diagnostic, time_pair
from pipec2.launcher import open_hostless, run_hostless
from pipec2.profile import get_profile
from tools.pipelink_env import load as load_environment
from params import PROFILE_SOURCES, build_params, inputs, oracle

BENCH_SHAPES = ((4096, 8192, 4096),)  # Grid 32x64; K long enough that the mainloop dominates.
# K=64 covers prologue/drain; K=512 wraps a three-slot K64 operand ring more than twice.
SANITIZE_SHAPES = ((128, 128, 64), (256, 256, 512))
# Per CTA the driver reserves 1 KiB of shared memory; per SM capacity is the per-block opt-in
# maximum plus that reservation (CUDA Programming Guide: 228/227 KiB on sm90/sm10x, 100/99 KiB on sm12x).
SMEM_RESERVED_PER_CTA = 1024
REGS_PER_SM = 65536


def launch_arguments(record, shape):
    link_path = REPO / record["link"]
    link = json.loads(link_path.read_text())
    return {"link_json_path": link_path, "buffers": inputs(*shape),
            "params": partial(build_params, entries=link["params"]["entries"], shape=shape),
            "grid": (shape[0] // 128, shape[1] // 128, 1), "device": 0}


def verify(record, shape, images=("linked", "one-tu")):
    arguments = launch_arguments(record, shape)
    expected = oracle(arguments["buffers"])
    outputs, comparisons = {}, {}
    for image in images:
        outputs[image] = run_hostless(**arguments, cubin=REPO / record[image.replace("-", "_")],
                                      outputs=("output",))["output"]
        comparisons[image] = compare(outputs[image], expected, rtol=1e-3, atol=1e-2)
    bitwise = (compare(outputs["linked"], outputs["one-tu"], rtol=0, atol=0)["bitwise"]
               if len(outputs) == 2 else None)
    passed = all(item["allclose"] for item in comparisons.values()) and bitwise is not False
    return {"shape": list(shape), "status": "PASS" if passed else "FAIL",
            "returncode": 0 if passed else 1,
            "oracle_mismatches": sum(item["mismatches"] for item in comparisons.values()),
            "bitwise": bitwise, "comparisons": comparisons}


def resources(record, env, profile):
    """Dynamic smem from the link, registers from the cubin, and the resulting CTA/SM bound."""
    link = json.loads((REPO / record["link"]).read_text())
    dump = subprocess.run([Path(env["PIPELINK_CUDA_HOME"]) / "bin/cuobjdump", "--dump-resource-usage",
                           REPO / record["linked"]], capture_output=True, text=True, check=True).stdout
    regs, stack, static = map(int, re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+)", dump).groups())
    smem = link["dyn_smem_bytes"] + static + SMEM_RESERVED_PER_CTA
    smem_per_sm = get_profile(profile).caps.smem_capacity_dynamic + SMEM_RESERVED_PER_CTA
    return {"dyn_smem_bytes": link["dyn_smem_bytes"], "regs": regs, "stack": stack,
            "ctas_per_sm_bound": min(smem_per_sm // smem, REGS_PER_SM // (regs * link["threads"]))}


def _arm_ms(timing, label):
    """Median per-launch milliseconds of one arm across the balanced blocks."""
    return float(np.median([sum(ms for arm, ms in zip(block["order"], block["milliseconds"]) if arm == label)
                            for block in timing["blocks"]])) / (2 * timing["repetitions"])


def benchmark(cell, native, gpu, shape):
    arguments = launch_arguments(cell, shape)
    flops = 2 * shape[0] * shape[1] * shape[2]
    with open_hostless(**arguments, cubin=REPO / cell["linked"]) as linked, \
            linked.with_image(REPO / cell["one_tu"]) as control:
        # Both images run on the same buffers, params and stream: allocation placement is a
        # controlled variable of the parity measurement, as in the original runner.
        poison = np.full_like(linked.read("output"), np.nan)
        linked.write("output", poison)
        linked.launch()
        actual = linked.read("output")
        control.write("output", poison)
        control.launch()
        parity = compare(actual, control.read("output"), rtol=0, atol=0)
        if not parity["bitwise"] or not np.isfinite(actual).all():
            raise RuntimeError("Benchmark images must be finite and bitwise equal")
        timing = time_pair(linked, control, repetitions=20, gpu=gpu)
        result = {**timing, "shape": list(shape), "returncode": 0 if timing["status"] == "PASS" else 1,
                  "bitwise": True, "boundary_result": timing["ratio_onetu_over_linked"] < 0.97,
                  "linked_ms": _arm_ms(timing, "A"), "linked_tflops": flops / _arm_ms(timing, "A") / 1e9}
        if native is None:
            return result
        # Native-fused characterization (not a same-source one-TU parity ratio or gate). The native
        # kernel has its own params layout, so it runs on its own allocation of identical inputs.
        with open_hostless(**launch_arguments(native, shape), cubin=REPO / native["linked"]) as fused:
            fused.write("output", poison)
            fused.launch()
            check = compare(fused.read("output"), actual, rtol=1e-3, atol=1e-2)
            if not check["allclose"]:
                raise RuntimeError(f"Native output disagrees with linked: {check}")
            pair = time_pair(linked, fused, repetitions=20, gpu=gpu)
    native_ms = _arm_ms(pair, "B")
    result["native"] = {
        "status": pair["status"], "blocks": pair["blocks"], "repetitions": pair["repetitions"],
        "ratio_linked_over_native": pair["ratio_linked_over_onetu"],
        "linked_ms": _arm_ms(pair, "A"), "native_ms": native_ms, "native_tflops": flops / native_ms / 1e9,
        "max_abs_vs_linked": check["max_abs"], "clocks": pair["clocks"]}
    result["returncode"] |= 0 if pair["status"] == "PASS" else 1
    return result


def main():
    env = {**os.environ, **load_environment(REPO)}
    os.environ.update(env)  # The shared clock report observes the same .env configuration.
    default_gpu = int(env["PIPELINK_GPUS"].split(",")[0]) if env["PIPELINK_GPUS"] else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_SOURCES), required=True)
    parser.add_argument("--gpu", type=int, default=default_gpu,
                        help="physical GPU index (default: first PIPELINK_GPUS entry)")
    parser.add_argument("--single", nargs=2, metavar=("NAME", "IMAGE"),
                        help="verify one cell's linked/one-tu image or a native's native image "
                             "(sanitizer subprocess)")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"), default=(128, 128, 64),
                        help="shape for --single")
    args = parser.parse_args()
    if args.gpu is None:
        parser.error("set --gpu or PIPELINK_GPUS in .env")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out = HERE / "out" / args.profile
    index = json.loads((out / "index.json").read_text())
    natives = index.get("natives", {})
    if args.single:
        name, image = args.single
        if image == "native" and name in natives:
            result = verify(natives[name], tuple(args.shape), ("linked",))
        elif name in index["cells"] and image in ("linked", "one-tu"):
            result = verify(index["cells"][name], tuple(args.shape), (image,))
        else:
            parser.error("--single requires a cell with linked/one-tu, or a native with native")
        print(json.dumps(result, sort_keys=True))
        return result["returncode"]

    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "one-epilogue" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    results = {"profile": args.profile, "gpu": args.gpu, "clock_mutation": False,
               "cells": {}, "natives": {}}
    passed = True

    def sanitize_image(name, image, evidence):
        records = []
        for shape in SANITIZE_SHAPES:
            records += [{**record, "image": image, "shape": list(shape)} for record in sanitize(
                [sys.executable, str(HERE / "run.py"), "--profile", args.profile, "--gpu", str(args.gpu),
                 "--single", name, image, "--shape", *map(str, shape)],
                env=env, log_dir=evidence / "sanitize" / image / "x".join(map(str, shape)),
                tools=("memcheck", "racecheck", "synccheck"))]
        return records

    for name, native in natives.items():
        evidence = out / "evidence" / name
        row = results["natives"][name] = {"verify": [], "resources": resources(native, env, args.profile)}
        for k in (64, 256, 1024):
            result = verify(native, (256, 256, k), ("linked",))
            row["verify"].append(result)
            passed &= result["returncode"] == 0
        row["sanitize"] = sanitize_image(name, "native", evidence)
        passed &= all(record["status"] == "PASS" for record in row["sanitize"])
        print(f"{name}: " + json.dumps({key: row[key] for key in ("resources",)}
                                       | {"verify": [item["status"] for item in row["verify"]],
                                          "sanitize": [item["status"] for item in row["sanitize"]]}),
              flush=True)
    for name, cell in index["cells"].items():
        evidence = out / "evidence" / name
        evidence.mkdir(parents=True, exist_ok=True)
        row = results["cells"][name] = {"verify": [], "sanitize": [], "bench": [],
                                        "resources": resources(cell, env, args.profile)}
        for k in (64, 256, 1024):
            try:
                result = verify(cell, (256, 256, k))
            except Exception as error:
                result = {"shape": [256, 256, k], "returncode": 1, "status": "ERROR", "error": str(error)}
            (evidence / f"verify-k{k}.json").write_text(json.dumps(result, indent=2) + "\n")
            row["verify"].append(result)
            passed &= result["returncode"] == 0
        if all(item["returncode"] == 0 for item in row["verify"]):
            for shape in BENCH_SHAPES:
                try:
                    bench = benchmark(cell, natives.get(cell.get("native")), args.gpu, shape)
                except Exception as error:
                    bench = {"shape": list(shape), "returncode": 1, "status": "ERROR", "error": str(error)}
                row["bench"].append(bench)
                passed &= bench["returncode"] == 0
                if bench.get("status") == "PASS":
                    fused = bench.get("native")
                    note = (f"; linked/native {fused['ratio_linked_over_native']:.4f} "
                            f"({fused['linked_ms']:.4f} vs {fused['native_ms']:.4f} ms)" if fused else "")
                    print(f"{name} {shape}: one-TU/linked {bench['ratio_onetu_over_linked']:.6f}, "
                          f"{bench['linked_tflops']:.1f} TFLOP/s{note}", flush=True)
        (evidence / "bench.json").write_text(json.dumps(row["bench"], indent=2) + "\n")
        # Always run both tools over both images, even after a numerical failure.
        for image in ("linked", "one-tu"):
            row["sanitize"] += sanitize_image(name, image, evidence)
        passed &= all(record["status"] == "PASS" for record in row["sanitize"])
        row["sass"] = sass_diagnostic(REPO / cell["linked"], REPO / cell["one_tu"],
                                      env=env, out_dir=evidence,
                                      backend_matched=REPO / cell["one_tu_lto"])
        print(f"{name} SASS (diagnostic only): " + json.dumps(row["sass"], sort_keys=True), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    results["status"] = "PASS" if passed else "FAIL"
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
