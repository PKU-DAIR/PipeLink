"""Absolute per-call CUDA-event timing shared by all public adapters."""
from __future__ import annotations

import math
import statistics
import subprocess

import torch


def read_clocks(gpu_uuid) -> dict:
    gpu_uuid = str(gpu_uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_uuid}", "--query-gpu=clocks.sm,clocks.max.sm", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    sm, maximum = result.stdout.strip().split(",")
    return {"gpu_uuid": gpu_uuid, "sm_mhz": int(sm), "max_sm_mhz": int(maximum)}


def measure(run, *, reps=10, min_interval_ms=20.0, smoke=False) -> dict:
    """Time N complete calls per interval, no host sync inside the interval.

    prepare must have covered compile/autotune/plan. Additional warmup calls warm
    L2 and lazy launches. Only smoke=True permits fewer than ten repetitions.
    If an interval falls below the floor, increase N and restart all repetitions.
    """
    if reps < (2 if smoke else 10) or min_interval_ms < 20:
        raise ValueError("Require >=10 reps (>=2 smoke) and >=20ms intervals")
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def interval(n):
        start.record()
        for _ in range(n):
            run()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    n = 1
    while True:
        elapsed = interval(n)
        if elapsed >= min_interval_ms:
            break
        n = max(n + 1, math.ceil(n * min_interval_ms / max(elapsed, 0.001) * 1.1))
    samples = []
    while len(samples) < reps:
        elapsed = interval(n)
        if elapsed < min_interval_ms:
            n = max(n + 1, math.ceil(n * min_interval_ms / max(elapsed, 0.001) * 1.1))
            samples.clear()
            continue
        samples.append(elapsed / n)
    return {
        "median_ms": statistics.median(samples), "reps": reps, "n": n,
        "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples,
        "min_interval_ms": min_interval_ms, "cuda_graphs": False,
    }
