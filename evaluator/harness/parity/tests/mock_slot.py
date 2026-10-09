#!/usr/bin/env python3
"""CPU-only subprocess fixture for exercising the slot protocol."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

EXCLUDED = [
    "compilation", "fa4-jit", "resolver-solve-validation", "stub-compile",
    "nvjitlink", "cubin-retrieval-write", "module-load", "allocation",
    "initialization", "one-time-prepack",
]
trace_file: Path | None = None
if "--trace-file" in sys.argv:
    trace_index = sys.argv.index("--trace-file")
    trace_file = Path(sys.argv[trace_index + 1])


request = json.load(sys.stdin)
common = {
    "schema_version": "m3-parity-slot-result-v1",
    "action": request["action"],
    "case_id": request["case_id"],
    "image_id": request["image"]["image_id"],
    "image_role": request["image_role"],
    "correctness": {
        "all_true": True,
        "checked_before_warmup": True,
        "gate_a": True,
        "gate_b": True,
        "fixture": True,
    },
}
if request["action"] == "warmup":
    common.update(timing=False, warmups_completed=request["protocol"]["warmups_per_image"])
else:
    protocol = request["protocol"]
    slot_start = time.time_ns()
    time.sleep(protocol["leading_discard_seconds"])
    window_start = time.time_ns()
    time.sleep(protocol["steady_seconds"])
    window_end = time.time_ns()
    slot_end = time.time_ns()
    throughput = 100.0 if request["image_role"] == "reference" else 99.8
    common.update({
        "timing": True,
        "iterations": 100,
        "throughput": throughput,
        "throughput_unit": "fixture-work/s",
        "steady_cuda_event_milliseconds": protocol["steady_seconds"] * 1000.0,
        "slot_start_ns": slot_start,
        "sampling_window_start_ns": window_start,
        "sampling_window_end_ns": window_end,
        "cuda_scoring_window_start_ns": window_start,
        "cuda_scoring_window_end_ns": window_end,
        "slot_end_ns": slot_end,
        "slot_duration_seconds": protocol["slot_seconds"],
        "leading_discard_seconds": protocol["leading_discard_seconds"],
        "steady_duration_seconds": protocol["steady_seconds"],
        "configured_batch_quantum_ms": protocol["configured_batch_quantum_ms"],
        "last_batch_start_before_nominal_end": True,
        "declared_slot_seconds": protocol["slot_seconds"],
        "observed_slot_seconds": (slot_end - slot_start) / 1.0e9,
        "declared_scored_seconds": protocol["steady_seconds"],
        "observed_scored_seconds": (window_end - window_start) / 1.0e9,
        "single_launch_calibration_ms": 1.0,
        "inner_batch_launches": 1,
        "maximum_observed_batch_ms": 1.0,
        "timing_method": "cuda-events-repeated-batches",
        "steady_timing_excludes": EXCLUDED,
    })
    if trace_file is not None:
        with trace_file.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps({
                "role": request["image_role"],
                "window_start_ns": window_start,
                "window_end_ns": window_end,
            }, sort_keys=True) + "\n")
json.dump(common, sys.stdout, sort_keys=True)
sys.stdout.write("\n")
