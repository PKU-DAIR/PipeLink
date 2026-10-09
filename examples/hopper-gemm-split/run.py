#!/usr/bin/env python3
"""Check Hopper GEMM composition, same-source timing, and sanitizer evidence."""
from __future__ import annotations

import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
from statistics import median
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from params import build
from oracle import Cublas, _inputs, RTOL, ATOL
from tools.pipelink_env import load
from evaluator.harness.parity.example import compare, sanitize, sass_diagnostic, time_pair

SMALL_SHAPE = (256, 384, 512)  # Rectangular grid, eight K tiles: wraps both ring depths.
PERFORMANCE_SHAPES = ((4096, 4096, 4096), (8192, 8192, 4096))
CELLS = {
    "store-d4": ("store", 4),
    "bias_relu-d4": ("bias_relu", 4),
    "rowscale-d4": ("rowscale", 4),
    "store-d2": ("store", 2),
}




def _epilogue(product, inputs, kind):
    if kind == "bias_relu":
        return np.maximum(product + inputs["bias"][None, :], np.float32(0))
    if kind == "rowscale":
        return product * inputs["scale"][:, None]
    return product


def _open(out, cell, shape, inputs, image, *, oracle=False):
    from pipec2.launcher import open_hostless

    M, N, K = shape
    buffers = dict(inputs)
    if oracle:
        buffers["oracle"] = ((M, N), np.float32)
    return open_hostless(
        out / cell["link"], buffers=buffers,
        params=partial(build, M=M, N=N, K=K, epilogue=cell["epilogue"]),
        grid=(M // 128, N // 128, 1), device=0,
        cubin=out / cell["linked" if image == "linked" else "one_tu"],
    )


def _baseline(blas, shape):
    blas.time(8)
    repetitions = 1
    for _ in range(24):
        elapsed = blas.time(repetitions)
        if elapsed >= 15.625:
            break
        repetitions = max(repetitions + 1, math.ceil(repetitions * 15.625 / elapsed))
        if repetitions > 1 << 20:
            raise RuntimeError("cuBLAS calibration exceeded the repetition cap")
    else:
        raise RuntimeError("cuBLAS calibration exhausted 24 rounds")
    batches = [blas.time(repetitions) for _ in range(8)]
    milliseconds = median(batches) / repetitions
    M, N, K = shape
    return {
        "descriptive_only": True, "epilogue_included": False,
        "operation": "cublasGemmEx BF16 inputs, FP32 accumulation/output",
        "library": blas.library_name, "version": blas.version,
        "repetitions": repetitions, "batch_milliseconds": batches,
        "milliseconds": milliseconds, "tflops": 2 * M * N * K / (milliseconds * 1e9),
    }


def _measure_cell(out, cell, shape, inputs, config, gpu):
    with _open(out, cell, shape, inputs, "linked", oracle=True) as linked, \
            _open(out, cell, shape, inputs, "one-tu") as control, \
            Cublas(linked, shape, config["PIPELINK_CUDA_HOME"]) as blas:
        linked.launch()
        actual = linked.read("output")
        control.launch()
        combined = control.read("output")
        blas.launch()
        expected = _epilogue(linked.read("oracle"), inputs, cell["epilogue"])
        numeric = {
            "linked_cublas": compare(actual, expected, rtol=RTOL, atol=ATOL),
            "one_tu_cublas": compare(combined, expected, rtol=RTOL, atol=ATOL),
            "linked_vs_one_tu": compare(actual, combined, rtol=0.0, atol=0.0),
        }
        passed = (numeric["linked_cublas"]["allclose"]
                  and numeric["one_tu_cublas"]["allclose"]
                  and numeric["linked_vs_one_tu"]["bitwise"])
        result = {"name": cell["name"], "epilogue": cell["epilogue"],
                  "depth": cell["depth"], "numeric": numeric}
        del expected
        if passed:
            # Keep independent numerical outputs, but remove allocation/descriptor
            # differences from the paired timing denominator.
            with linked.with_image(out / cell["one_tu"]) as timed_control:
                timing = time_pair(linked, timed_control, repetitions=20, gpu=gpu)
            baseline = _baseline(blas, shape)
            M, N, K = shape
            for label, name in (("A", "linked"), ("B", "one_tu")):
                milliseconds = median(
                    ms / timing["repetitions"]
                    for block in timing["blocks"]
                    for leg, ms in zip(block["order"], block["milliseconds"])
                    if leg == label
                )
                timing[f"{name}_milliseconds"] = milliseconds
                timing[f"{name}_tflops"] = 2 * M * N * K / (milliseconds * 1e9)
            timing["cublas"] = baseline
            timing["descriptive_ratio_linked_over_cublas_throughput"] = (
                timing["linked_tflops"] / baseline["tflops"]
            )
            result["timing"] = timing
            passed = timing["status"] == "PASS"
        else:
            result["timing"] = {"status": "NOT_RUN_NUMERICAL_FAILURE"}
        result["status"] = "PASS" if passed else "FAIL"
    return result, actual, combined


def _single(out, cell, shape, config, image):
    inputs = _inputs(shape)
    with _open(out, cell, shape, inputs, image, oracle=True) as session, \
            Cublas(session, shape, config["PIPELINK_CUDA_HOME"]) as blas:
        session.launch()
        actual = session.read("output")
        blas.launch()
        expected = _epilogue(session.read("oracle"), inputs, cell["epilogue"])
        numeric = compare(actual, expected, rtol=RTOL, atol=ATOL)
    print(json.dumps({"cell": cell["name"], "image": image, "shape": shape,
                      "cublas_oracle": numeric, "status": "PASS" if numeric["allclose"] else "FAIL"}))
    return 0 if numeric["allclose"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("sm_90a",), required=True)
    parser.add_argument("--gpu", type=int, help="physical GPU index; default: first PIPELINK_GPUS entry")
    parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"),
                        help="replace performance shapes; the small phase-wrap check still runs")
    parser.add_argument("--single", choices=("linked", "one-tu"),
                        help="one image plus cuBLAS oracle only; do not recurse into sanitizers")
    parser.add_argument("--cell", choices=tuple(CELLS), help="required with --single")
    args = parser.parse_args()
    if bool(args.single) != bool(args.cell):
        parser.error("--single and --cell must be used together")
    if args.shape:
        M, N, K = args.shape
        if min(args.shape) <= 0 or max(args.shape) > 2**31 - 1 or M % 128 or N % 128 or K % 64:
            parser.error("M,N must be positive multiples of 128; K a positive multiple of 64; all must fit int32")
    config = load(REPO)
    gpu = args.gpu
    if gpu is None:
        if not config["PIPELINK_GPUS"]:
            parser.error("set PIPELINK_GPUS in .env or pass --gpu")
        gpu = int(config["PIPELINK_GPUS"].split(",")[0])
    if gpu < 0:
        parser.error("--gpu must be nonnegative")
    if not config["PIPELINK_CUDA_HOME"]:
        parser.error("set PIPELINK_CUDA_HOME in .env")
    os.environ.update(config)
    # Neither the launcher nor cuBLAS has initialized CUDA at this point.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    scratch = Path(config["PIPELINK_SCRATCH_ROOT"]) / "hopper-gemm-split" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["CUDA_CACHE_PATH"] = str(scratch / "cuda-cache")

    out = HERE / "out" / args.profile
    index = json.loads((out / "index.json").read_text())
    cells = {cell["name"]: cell for cell in index["cells"]}
    if index["profile"] != args.profile or len(index["cells"]) != len(CELLS) or set(cells) != set(CELLS):
        raise ValueError("index.json must describe the four sm_90a GEMM cells")
    for name, (epilogue, depth) in CELLS.items():
        if (cells[name]["epilogue"], cells[name]["depth"]) != (epilogue, depth):
            raise ValueError(f"index.json has the wrong epilogue/depth for {name}")
    if args.single:
        return _single(out, cells[args.cell], tuple(args.shape or SMALL_SHAPE), config, args.single)

    performance_shapes = (tuple(args.shape),) if args.shape else PERFORMANCE_SHAPES
    shapes = list(dict.fromkeys((SMALL_SHAPE, *performance_shapes)))
    results = {
        "profile": args.profile, "gpu": gpu, "device_ordinal": 0,
        "hostless_only": True, "rtol": RTOL, "atol": ATOL,
        "small_shape": list(SMALL_SHAPE), "scoped_shape": args.shape,
        "timing_policy": "shared buffers, params and stream; time_pair PASS required; ratios have no threshold",
        "throughput_policy": "2*M*N*K FLOPs; cuBLAS ratios descriptive, cuBLAS excludes epilogues",
        "sanitize": [], "sass": {}, "shapes": [],
    }
    # Always run both tools on both images of every cell, even when one fails.
    # Do this first so later numerical/timing errors cannot omit sanitizer legs.
    for name in CELLS:
        for image in ("linked", "one-tu"):
            reports = sanitize(
                [sys.executable, str(HERE / "run.py"), "--profile", args.profile,
                 "--gpu", str(gpu), "--single", image, "--cell", name,
                 "--shape", *map(str, SMALL_SHAPE)],
                env=dict(os.environ), log_dir=out / name / "sanitize" / image,
            )
            for report in reports:
                report["log"] = str(Path(report["log"]).relative_to(HERE))
                results["sanitize"].append({"cell": name, "image": image, **report})
    passed = all(report["status"] == "PASS" for report in results["sanitize"])
    for name, cell in cells.items():
        report = sass_diagnostic(
            out / cell["linked"], out / cell["one_tu"], env=dict(os.environ),
            out_dir=out / name / "sass", backend_matched=out / cell["one_tu_lto"],
        )
        report["comparison"] = str(Path(report["comparison"]).relative_to(HERE))
        results["sass"][name] = report
    for shape in shapes:
        inputs = _inputs(shape)
        row = {"shape": list(shape), "grid": [shape[0] // 128, shape[1] // 128, 1], "cells": []}
        store_d4 = None
        for name in CELLS:
            result, actual, combined = _measure_cell(out, cells[name], shape, inputs, config, gpu)
            row["cells"].append(result)
            passed &= result["status"] == "PASS"
            if name == "store-d4":
                store_d4 = (actual, combined)
            elif name == "store-d2":
                depth = {
                    "linked": compare(actual, store_d4[0], rtol=0.0, atol=0.0),
                    "one_tu": compare(combined, store_d4[1], rtol=0.0, atol=0.0),
                }
                row["store_d4_vs_d2"] = depth
                passed &= all(check["bitwise"] for check in depth.values())
            del actual, combined
        results["shapes"].append(row)
        del inputs, store_d4
    results["status"] = "PASS" if passed else "FAIL"
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
