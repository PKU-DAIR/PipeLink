#!/usr/bin/env python3
"""Megakernels llama-1b adapter for the shared parity slot protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HOST = ROOT / "campaigns" / "megakernel" / "host"
sys.path.insert(0, str(ROOT / "tools"))
from pipelink_env import load

TMP_ROOT = Path(load(ROOT)["PIPELINK_SCRATCH_ROOT"])
MAX_BATCH_TOKENS = 32
WORK_DEFINITION = "one llama-1b decode token at fixed pos_id={pos_id} (148 CTAs, Megakernel schedule rr)"
STEADY_TIMING_EXCLUDES = [
    "compilation",
    "fa4-jit",
    "resolver-solve-validation",
    "stub-compile",
    "nvjitlink",
    "cubin-retrieval-write",
    "module-load",
    "allocation",
    "initialization",
    "one-time-prepack",
]
OUTPUT_NAMES = (
    "hidden_states",
    "logits",
    "q_post_rope",
    "attn_out",
    "silu_out",
    "k_row",
    "v_row",
    "attn_out_p0",
    "attn_lse_p0",
)

sys.path.insert(0, str(HOST))
from compare import ATOL_RMS_FACTOR, RTOL, decode, load_raw, parse_manifest
from launch import launch_command


class SlotError(RuntimeError):
    """A fail-closed slot request or execution error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=("a", "rebind-smem", "rebind-regs", "rebind-opcode", "cute"),
    )
    parser.add_argument("--image", required=True, choices=("linked", "onetu", "mono"))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--pos-id", required=True, type=int)
    return parser.parse_args()


def require_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SlotError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise SlotError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def validate_request(request: Any, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if not isinstance(request, dict):
        raise SlotError("request must be one JSON object")
    if request.get("schema_version") != "m3-parity-slot-request-v1":
        raise SlotError("unsupported request schema")
    if request.get("action") not in ("warmup", "slot"):
        raise SlotError("unsupported action")
    if request.get("cuda_visible_device") != 0:
        raise SlotError("visible CUDA device must be 0")
    physical_gpu = request.get("physical_gpu")
    if isinstance(physical_gpu, bool) or not isinstance(physical_gpu, int) or physical_gpu <= 0:
        raise SlotError("physical_gpu must be a positive nonzero integer")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise SlotError("CUDA_VISIBLE_DEVICES does not match request physical_gpu")
    if not isinstance(request.get("case_id"), str) or not request["case_id"]:
        raise SlotError("case_id must be a non-empty string")

    bundle = args.bundle.resolve()
    if not bundle.is_dir():
        raise SlotError(f"bundle directory is missing: {bundle}")
    if args.pos_id < 0:
        raise SlotError("pos_id must be nonnegative")
    return request, bundle


def run_checked(command: list[str], label: str, *, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise SlotError(
            f"{label} failed rc={completed.returncode}: stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    return completed


def correctness_recheck(args: argparse.Namespace, bundle: Path, workspace: Path) -> dict[str, Any]:
    out = workspace / "correctness"
    command = launch_command(args.pipeline, args.image, bundle, out, "check")
    command.extend(["--ntok", "1", "--pos-id-first", str(args.pos_id)])
    run_checked(command, "Gate P2 correctness recheck")

    specs = parse_manifest(bundle)
    tensor_results: dict[str, Any] = {}
    all_finite = True
    for name in OUTPUT_NAMES:
        reference_name = f"ref_tok0_{name}"
        if reference_name not in specs:
            raise SlotError(f"bundle manifest is missing {reference_name}")
        spec = specs[reference_name]
        observed = decode(load_raw(out / f"tok0_{name}.bin", spec), spec.dtype)
        reference = decode(load_raw(spec.path, spec), spec.dtype)
        observed_finite = bool(np.isfinite(observed).all())
        reference_finite = bool(np.isfinite(reference).all())
        all_finite &= observed_finite and reference_finite
        tensor_results[name] = {
            "observed_finite": observed_finite,
            "reference_finite": reference_finite,
            "elements": int(observed.size),
        }

    for name in ("k_row", "v_row"):
        spec = specs[f"ref_tok0_{name}"]
        observed = decode(load_raw(out / f"tok0_{name}.bin", spec), spec.dtype)[0]
        reference = decode(load_raw(spec.path, spec), spec.dtype)[0]
        rms_reference = float(np.sqrt(np.mean(np.square(reference.astype(np.float64)))))
        atol = ATOL_RMS_FACTOR * rms_reference
        finite_pair = np.isfinite(observed) & np.isfinite(reference)
        within = finite_pair & (
            np.abs(observed.astype(np.float64) - reference.astype(np.float64))
            <= atol + RTOL * np.abs(reference.astype(np.float64))
        )
        violations = int(np.count_nonzero(~within))
        tensor_results[f"{name}[0]"] = {
            "pass": violations == 0,
            "elements": int(reference.size),
            "violations": violations,
            "rms_reference": rms_reference,
            "rtol": RTOL,
            "atol": atol,
        }
        if violations:
            raise SlotError(f"Gate P2 strict predicate failed for {name}[0]: {violations} violations")

    if not all_finite:
        failed = [name for name in OUTPUT_NAMES if not tensor_results[name]["observed_finite"]]
        raise SlotError(f"Gate P2 correctness recheck found non-finite outputs: {failed}")
    result = {"all_true": True, "checked_before_warmup": True}
    print("MK Gate P2 recheck " + json.dumps(tensor_results, sort_keys=True), file=sys.stderr)
    return result


def perform_warmup(
    request: dict[str, Any], args: argparse.Namespace, bundle: Path, workspace: Path, correctness: dict[str, Any]
) -> dict[str, Any]:
    warmups = request["protocol"]["warmups_per_image"]
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise SlotError("warmups_per_image must be a nonnegative integer")
    for index in range(warmups):
        out = workspace / f"warmup-{index}"
        command = launch_command(args.pipeline, args.image, bundle, out, "check")
        command.extend(["--ntok", "1", "--pos-id-first", str(args.pos_id)])
        run_checked(command, f"untimed warmup {index + 1}/{warmups}")
    return {
        "schema_version": "m3-parity-slot-result-v1",
        "action": "warmup",
        "case_id": request["case_id"],
        "image_id": request["image"]["image_id"],
        "image_role": request["image_role"],
        "correctness": correctness,
        "timing": False,
        "warmups_completed": warmups,
    }


def parse_bench_output(
    completed: subprocess.CompletedProcess[str],
    *,
    expected_tokens: int,
    configured_quantum_ms: float,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SlotError(f"bench runner did not emit one JSON object: {completed.stdout!r}") from exc
    if not isinstance(payload, dict) or set(payload) != {"batches"} or not isinstance(payload["batches"], list):
        raise SlotError("bench runner output must contain exactly one batches array")
    batches: list[dict[str, Any]] = []
    previous_end = 0
    for index, batch in enumerate(payload["batches"]):
        if not isinstance(batch, dict) or set(batch) != {"start_ns", "end_ns", "cuda_ms", "tokens"}:
            raise SlotError(f"bench batch {index} has unexpected fields")
        start_ns, end_ns = batch["start_ns"], batch["end_ns"]
        tokens, cuda_ms = batch["tokens"], batch["cuda_ms"]
        if (
            isinstance(start_ns, bool)
            or not isinstance(start_ns, int)
            or isinstance(end_ns, bool)
            or not isinstance(end_ns, int)
            or start_ns <= 0
            or end_ns <= start_ns
            or start_ns < previous_end
        ):
            raise SlotError(f"bench batch {index} has invalid epoch-ns bounds")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens != expected_tokens:
            raise SlotError(
                f"bench batch {index} does not contain exactly {expected_tokens} tokens"
            )
        cuda_value = require_number(cuda_ms, f"bench batch {index} cuda_ms", positive=True)
        if cuda_value > configured_quantum_ms:
            raise SlotError(
                f"bench batch {index} exceeds {configured_quantum_ms} ms: cuda_ms={cuda_value}"
            )
        previous_end = end_ns
        batches.append({"start_ns": start_ns, "end_ns": end_ns, "cuda_ms": cuda_value, "tokens": tokens})
    if not batches:
        raise SlotError("bench runner returned no batches")
    return batches


def perform_slot(
    request: dict[str, Any], args: argparse.Namespace, bundle: Path, workspace: Path, correctness: dict[str, Any]
) -> dict[str, Any]:
    protocol = request["protocol"]
    slot_seconds = float(protocol["slot_seconds"])
    discard_seconds = float(protocol["leading_discard_seconds"])
    steady_seconds = float(protocol["steady_seconds"])
    configured_quantum_ms = float(protocol["configured_batch_quantum_ms"])

    probe_out = workspace / "batch-probe"
    probe_command = launch_command(args.pipeline, args.image, bundle, probe_out, "bench")
    probe_command.extend(
        [
            "--bench-seconds",
            "0.001",
            "--bench-discard",
            "0",
            "--bench-batch-tokens",
            "1",
            "--pos-id",
            str(args.pos_id),
        ]
    )
    probe_completed = run_checked(probe_command, "single-token batch probe")
    probe_batches = parse_bench_output(
        probe_completed,
        expected_tokens=1,
        configured_quantum_ms=configured_quantum_ms,
    )
    single_launch_ms = max(batch["cuda_ms"] for batch in probe_batches)
    target_batch_ms = 0.5 * configured_quantum_ms
    batch_tokens = max(
        1, min(MAX_BATCH_TOKENS, int(target_batch_ms / single_launch_ms))
    )

    out = workspace / "bench"
    command = launch_command(args.pipeline, args.image, bundle, out, "bench")
    command.extend(
        [
            "--bench-seconds",
            str(slot_seconds),
            "--bench-discard",
            str(discard_seconds),
            "--bench-batch-tokens",
            str(batch_tokens),
            "--pos-id",
            str(args.pos_id),
        ]
    )
    completed = run_checked(command, "timed bench slot", timeout=slot_seconds + discard_seconds + 120.0)
    batches = parse_bench_output(
        completed,
        expected_tokens=batch_tokens,
        configured_quantum_ms=configured_quantum_ms,
    )

    slot_start_ns = batches[0]["start_ns"]
    nominal_slot_end_ns = slot_start_ns + round(slot_seconds * 1.0e9)
    nominal_scoring_start_ns = slot_start_ns + round(discard_seconds * 1.0e9)
    if any(batch["start_ns"] >= nominal_slot_end_ns for batch in batches):
        raise SlotError("bench runner started a batch at/after nominal slot end")
    steady_batches = [
        batch
        for batch in batches
        if batch["start_ns"] >= nominal_scoring_start_ns
        and batch["end_ns"] <= nominal_slot_end_ns
    ]
    if not steady_batches:
        raise SlotError("no complete CUDA-event batch falls inside the steady scoring window")

    iterations = sum(batch["tokens"] for batch in steady_batches)
    elapsed_ms = sum(batch["cuda_ms"] for batch in steady_batches)
    if iterations <= 0 or elapsed_ms <= 0.0:
        raise SlotError("steady batch selection has no positive timed work")
    throughput = iterations / (elapsed_ms * 1.0e-3)
    sampling_start_ns = steady_batches[0]["start_ns"]
    sampling_end_ns = steady_batches[-1]["end_ns"]
    slot_end_ns = batches[-1]["end_ns"]
    observed_slot_seconds = (slot_end_ns - slot_start_ns) / 1.0e9
    observed_scored_seconds = (sampling_end_ns - sampling_start_ns) / 1.0e9
    cuda_values = [batch["cuda_ms"] for batch in batches]
    wall_values = [(batch["end_ns"] - batch["start_ns"]) / 1.0e6 for batch in batches]

    return {
        "schema_version": "m3-parity-slot-result-v1",
        "action": "slot",
        "case_id": request["case_id"],
        "image_id": request["image"]["image_id"],
        "image_role": request["image_role"],
        "correctness": correctness,
        "timing": True,
        "iterations": iterations,
        "throughput": throughput,
        "throughput_unit": "tokens/s",
        "steady_cuda_event_milliseconds": elapsed_ms,
        "slot_start_ns": slot_start_ns,
        "sampling_window_start_ns": sampling_start_ns,
        "sampling_window_end_ns": sampling_end_ns,
        "cuda_scoring_window_start_ns": sampling_start_ns,
        "cuda_scoring_window_end_ns": sampling_end_ns,
        "slot_end_ns": slot_end_ns,
        "slot_duration_seconds": slot_seconds,
        "leading_discard_seconds": discard_seconds,
        "steady_duration_seconds": steady_seconds,
        "configured_batch_quantum_ms": configured_quantum_ms,
        "last_batch_start_before_nominal_end": True,
        "declared_slot_seconds": slot_seconds,
        "observed_slot_seconds": observed_slot_seconds,
        "declared_scored_seconds": steady_seconds,
        "observed_scored_seconds": observed_scored_seconds,
        "protocol_version": protocol["protocol_variant"],
        "nominal_slot_end_ns": nominal_slot_end_ns,
        "inner_batch_launches": batch_tokens,
        "single_launch_calibration_ms": single_launch_ms,
        "batch_sizing_target_fraction": 0.5,
        "inner_batch_cap": MAX_BATCH_TOKENS,
        "maximum_observed_batch_ms": max(cuda_values),
        "minimum_observed_batch_ms": min(cuda_values),
        "maximum_observed_batch_wall_ms": max(wall_values),
        "minimum_observed_batch_wall_ms": min(wall_values),
        "measured_batch_count": len(batches),
        "steady_batch_count": len(steady_batches),
        "batch_boundaries_ns": [
            {
                "start_ns": batch["start_ns"],
                "end_ns": batch["end_ns"],
                "cuda_ms": batch["cuda_ms"],
                "tokens": batch["tokens"],
                "steady": batch in steady_batches,
            }
            for batch in batches
        ],
        "timing_method": "cuda-events-repeated-batches",
        "steady_timing_excludes": STEADY_TIMING_EXCLUDES,
        "work_definition": WORK_DEFINITION.format(pos_id=args.pos_id),
    }


def main() -> int:
    args = parse_args()
    request = json.load(sys.stdin)
    request, bundle = validate_request(request, args)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="megakernel-parity-slot-", dir=TMP_ROOT) as temporary:
        workspace = Path(temporary)
        correctness = correctness_recheck(args, bundle, workspace)
        if request["action"] == "warmup":
            result = perform_warmup(request, args, bundle, workspace, correctness)
        else:
            result = perform_slot(request, args, bundle, workspace, correctness)
    json.dump(result, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        json.dump(
            {
                "schema_version": "m3-parity-slot-error-v1",
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            sys.stdout,
            separators=(",", ":"),
        )
        sys.stdout.write("\n")
        print(f"MK parity slot: {exc}", file=sys.stderr)
        raise SystemExit(2)
