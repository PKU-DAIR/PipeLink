#!/usr/bin/env python3
"""Run definitive MK parity cases through the shared evaluator.

Suite `reference` compares each all-ThunderKittens linked image with its same-source
one-TU build. Suite `cute` does the same for the CuTe-attention composition and adds
cross-library pairs (all-ThunderKittens linked reference, CuTe linked candidate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "campaigns" / "megakernel" / "evidence"
SLOT = EVIDENCE / "parity" / "mk_slot.py"
BUNDLES = ROOT / "campaigns" / "megakernel" / "host" / "bundles"
RUNS = BUNDLES.parent / "runs"
PORT_BUILD = ROOT / "campaigns" / "megakernel" / "port" / "build"
# Suite-specific locations; select_suite() rebinds them (default: reference).
ARCHIVE_ROOT = EVIDENCE / "archive" / "submission-2026-09-17"
SANITIZER_RUNS = RUNS / "submission-2026-09-17" / "sanitizer"
RESULTS_PATH = EVIDENCE / "results.json"
RESULT_MARKDOWN_PATH = EVIDENCE / "RESULT.md"
SUITE = "reference"
GPU = int(os.environ["MK_GPU"]) if os.environ.get("MK_GPU") else None
MAX_BATCH_TOKENS = 32
RATIO_FLOOR = 0.97


def build_record(pipeline: str) -> Path:
    # The CuTe images come from `build.sh --cute` with their own build generation.
    return PORT_BUILD / ("build-record-cute.json" if pipeline == "cute" else "build-record.json")


def sanitize_receipt(pipeline: str) -> Path:
    return EVIDENCE / ("sanitize-cute.json" if pipeline == "cute" else "sanitize.json")


sys.path.insert(0, str(ROOT))
from evaluator.harness.parity import ImageSpec, PairCapError, Protocol, run_case
from evaluator.harness.parity.constants import (
    REQUIRED_RACECHECK_HAZARDS,
    REQUIRED_SYNCCHECK_ERRORS,
)

CASES = (
    {
        "case_id": "MK-A-16",
        "pipeline": "a",
        "bundle": "p16-n2",
        "pos_id": 16,
        "reference_image": "onetu",
        "linked_image": "linked",
        "kind": "seam-parity",
    },
    {
        "case_id": "MK-A-1024",
        "pipeline": "a",
        "bundle": "p1024-n8",
        "pos_id": 1024,
        "reference_image": "onetu",
        "linked_image": "linked",
        "kind": "seam-parity",
    },
    {
        "case_id": "MK-A-4096",
        "pipeline": "a",
        "bundle": "p4096-n2",
        "pos_id": 4096,
        "reference_image": "onetu",
        "linked_image": "linked",
        "kind": "seam-parity",
    },
    {
        "case_id": "MK-REGS-1024",
        "pipeline": "rebind-regs",
        "bundle": "p1024-n8",
        "pos_id": 1024,
        "reference_image": "onetu",
        "linked_image": "linked",
        "kind": "seam-parity",
    },
    {
        "case_id": "MK-A-1024-red",
        "pipeline": "a",
        "bundle": "p1024-n8-red",
        "pos_id": 1024,
        "reference_image": "onetu",
        "linked_image": "linked",
        "kind": "seam-parity",
    },
)

CUTE_CONTEXTS = (("16", "p16-n2", 16), ("1024-red", "p1024-n8-red", 1024), ("4096", "p4096-n2", 4096))
CUTE_CASES = tuple(
    {
        "case_id": f"MK-CUTE-{label}",
        "pipeline": "cute",
        "bundle": bundle,
        "pos_id": pos_id,
        "reference_image": "onetu",
        "linked_image": "linked",
        "kind": "seam-parity",
    }
    for label, bundle, pos_id in CUTE_CONTEXTS
) + tuple(
    # Candidate/reference is CuTe throughput over all-ThunderKittens throughput:
    # the paired speedup of the composition. Not a seam-parity gate.
    {
        "case_id": f"MK-CUTE-VS-TK-{label}",
        "pipeline": "cute",
        "reference_pipeline": "a",
        "bundle": bundle,
        "pos_id": pos_id,
        "reference_image": "linked",
        "linked_image": "linked",
        "kind": "cross-library",
    }
    for label, bundle, pos_id in CUTE_CONTEXTS
)

SUITES = {
    "reference": {
        "cases": CASES, "pipelines": ("a", "rebind-smem", "rebind-regs", "rebind-opcode"),
        "archive": EVIDENCE / "archive" / "submission-2026-09-17",
        "sanitizer_runs": RUNS / "submission-2026-09-17" / "sanitizer",
        "results": EVIDENCE / "results.json", "markdown": EVIDENCE / "RESULT.md",
    },
    "cute": {
        "cases": CUTE_CASES, "pipelines": ("cute",),
        "archive": EVIDENCE / "archive" / "cute",
        "sanitizer_runs": RUNS / "cute-20261007" / "p3-sanitizer",
        "results": EVIDENCE / "cute-parity.json", "markdown": EVIDENCE / "CUTE-RESULT.md",
    },
}


def select_suite(name: str) -> None:
    global SUITE, CASES, ARCHIVE_ROOT, SANITIZER_RUNS, RESULTS_PATH, RESULT_MARKDOWN_PATH
    suite = SUITES[name]
    SUITE, CASES = name, suite["cases"]
    ARCHIVE_ROOT, SANITIZER_RUNS = suite["archive"], suite["sanitizer_runs"]
    RESULTS_PATH, RESULT_MARKDOWN_PATH = suite["results"], suite["markdown"]


def role_pipeline(case: dict[str, Any], role: str) -> str:
    return case.get(f"{role}_pipeline", case["pipeline"])


def sanitizer_bundle(case: dict[str, Any]) -> Path:
    # Red cells use the P2-validated p1024-n8-red bundle (one token): the
    # short-prompt p16-n1-red export stalls natively (2 partitions over a
    # 16-token prompt; recorded 2026-09-17, deviation 9) and is not evidence.
    return BUNDLES / ("p1024-n8-red" if case["bundle"].endswith("-red") else "p16-n2")


class EvidenceError(RuntimeError):
    """A fail-closed parity orchestration error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_scalars(bundle: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    path = bundle / "scalars.txt"
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 2:
            raise EvidenceError(f"{path}:{line_number}: malformed scalar row")
        key, value = fields
        if key in result:
            raise EvidenceError(f"{path}:{line_number}: duplicate scalar {key}")
        result[key] = value
    return result


def image_path(pipeline: str, image: str) -> Path:
    if image == "linked":
        return (PORT_BUILD / "linked" / pipeline / "pipeline.cubin").resolve()
    if image == "onetu":
        return (PORT_BUILD / "onetu" / f"{pipeline}.cubin").resolve()
    raise EvidenceError(f"unknown image {image}")


def build_generation(pipeline: str) -> str:
    return json.loads(build_record(pipeline).read_text(encoding="utf-8"))["build_generation"]


def image_spec(case: dict[str, Any], role: str, image_hashes: dict[Path, str]) -> ImageSpec:
    image = case[f"{role}_image"]
    pipeline = role_pipeline(case, role)
    path = image_path(pipeline, image)
    command = [
        sys.executable,
        str(SLOT),
        "--pipeline",
        pipeline,
        "--image",
        image,
        "--bundle",
        str((BUNDLES / case["bundle"]).resolve()),
        "--pos-id",
        str(case["pos_id"]),
    ]
    return ImageSpec(
        image_id=f"{case['case_id']}-{role}-{image}-{pipeline}",
        command=tuple(command),
        image_path=path,
        image_sha256=image_hashes[path],
        cwd=ROOT,
        metadata={
            "family": "megakernel-llama-1b",
            "pipeline": pipeline,
            "image": image,
            "case_kind": case["kind"],
        },
    )


def case_config(case: dict[str, Any]) -> dict[str, Any]:
    bundle = (BUNDLES / case["bundle"]).resolve()
    manifest = bundle / "manifest.txt"
    if not manifest.is_file() or not (bundle / "scalars.txt").is_file():
        raise EvidenceError(f"required bundle is incomplete: {bundle}")
    scalars = parse_scalars(bundle)
    if int(scalars.get("pos_id_first", "-1")) != case["pos_id"]:
        raise EvidenceError(f"{case['case_id']}: bundle pos_id_first differs from case pos_id")
    return {
        "stream_map_sha256": sha256_bytes(manifest.read_bytes()),
        "stream_classes": [f"megakernel/llama-1b/{case['bundle']}"],
        "inputs": {
            "bundle": str(bundle),
            "pos_id": case["pos_id"],
            "scalars": scalars,
        },
        "arguments": {
            "pipelines": {
                "reference": role_pipeline(case, "reference"),
                "linked": role_pipeline(case, "linked"),
            },
            "images": {
                "reference": case["reference_image"],
                "linked": case["linked_image"],
            },
        },
    }


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def marker_path() -> Path:
    state_dir = os.environ.get("PIPELINK_LOCK_STATE_DIR")
    if not state_dir:
        raise EvidenceError("PIPELINK_LOCK_STATE_DIR is not set; source tools/env.sh")
    return Path(state_dir).expanduser() / f"gpu{GPU}.json"


def read_marker() -> dict[str, Any]:
    path = marker_path()
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read gpulockd marker {path}: {exc}") from exc
    if not isinstance(marker, dict):
        raise EvidenceError(f"gpulockd marker is not an object: {path}")
    return marker


def lock_command(action: str, target_mhz: int | None = None) -> list[str]:
    command = [sys.executable, str(ROOT / "tools" / "gpulockd.py"), action, "--gpus", str(GPU)]
    if action == "start":
        if target_mhz is None:
            raise EvidenceError("lock start requires target MHz")
        command.extend(["--sm-mhz", str(target_mhz)])
    return command


def check_lock(label: str, target_mhz: int) -> dict[str, Any]:
    record = run_command(lock_command("status"))
    record["label"] = label
    if record["returncode"] != 0:
        raise EvidenceError(
            f"gpulockd status failed during {label}: rc={record['returncode']} "
            f"stdout={record['stdout']!r} stderr={record['stderr']!r}"
        )
    marker = read_marker()
    if (
        marker.get("gpu") != GPU
        or marker.get("sm_mhz") != target_mhz
        or marker.get("violations") != 0
    ):
        raise EvidenceError(f"gpulockd marker violates held-clock contract during {label}: {marker}")
    record["marker"] = marker
    record["honored"] = True
    return record


def start_lock(target_mhz: int) -> dict[str, Any]:
    record = run_command(lock_command("start", target_mhz))
    if record["returncode"] == 3:
        raise EvidenceError(
            "gpulockd returned exit code 3: requested SM clock lock was not honored; "
            f"stdout={record['stdout']!r} stderr={record['stderr']!r}"
        )
    if record["returncode"] != 0:
        raise EvidenceError(
            f"gpulockd start failed rc={record['returncode']}: "
            f"stdout={record['stdout']!r} stderr={record['stderr']!r}"
        )
    record["target_sm_mhz"] = target_mhz
    return record


def stop_lock() -> dict[str, Any]:
    return run_command(lock_command("stop"))


SANITIZER_ADMISSION = (
    "memcheck-zero + recorded racecheck/synccheck counts (owner review at Gate 2). "
    "racecheck/synccheck report hazards on the same-source one-TU control in the "
    "same class and magnitude as on the linked image, so they characterize the "
    "upstream Megakernels/ThunderKittens source, not the link seam; their strict "
    "verdicts are kept verbatim and are not admission predicates for this epoch."
)


def sanitizer_images() -> list[tuple[str, str, Path]]:
    """Unique (pipeline, image, bundle) runs covering this suite's own images."""
    owned = SUITES[SUITE]["pipelines"]
    return sorted({
        (role_pipeline(case, role), case[f"{role}_image"], sanitizer_bundle(case))
        for case in CASES for role in ("reference", "linked")
        if role_pipeline(case, role) in owned
    })


def run_sanitizers() -> int:
    host = BUNDLES.parent
    sys.path.insert(0, str(host))
    from launch import launch_command

    required = {
        "memcheck": 0,
        "racecheck": REQUIRED_RACECHECK_HAZARDS,
        "synccheck": REQUIRED_SYNCCHECK_ERRORS,
    }
    gating = {"memcheck"}
    sanitizer = str(Path(os.environ["PIPELINK_CUDA_HOME"]) / "bin" / "compute-sanitizer")
    records = []
    owner = SUITES[SUITE]["pipelines"][0]
    generation = build_generation(owner)
    for pipeline, image, bundle in sanitizer_images():
        path = image_path(pipeline, image)
        for tool, required_count in required.items():
            out = SANITIZER_RUNS / f"{pipeline}-{image}-{bundle.name}" / tool
            out.mkdir(parents=True, exist_ok=True)
            command = launch_command(
                pipeline, image, bundle, out, "check",
                runner_args=("--ntok", "1"),
            )
            record = run_command(
                [sanitizer, "--tool", tool, "--error-exitcode", "99", *command],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), MK_WATCHDOG_SECONDS="1800"),
            )
            text = record.pop("stdout") + "\n" + record.pop("stderr")
            log = out / "sanitizer.log"
            log.write_text(text, encoding="utf-8")
            # The total line is "ERROR SUMMARY: N errors" at end of line; the
            # "M errors were not printed" line is a print-limit notice.
            pattern = (
                r"RACECHECK SUMMARY:\s*(\d+)\s+hazards?"
                if tool == "racecheck" else r"ERROR SUMMARY:\s*(\d+)\s+errors?[ \t]*$"
            )
            counts = re.findall(pattern, text, re.MULTILINE)
            count = int(counts[0]) if len(counts) == 1 else None
            passed = record["returncode"] == 0 and count == required_count
            records.append({
                **record, "pipeline": pipeline, "image": str(path),
                "image_role": image, "tool": tool, "count": count,
                "required": required_count, "passed": passed, "gating": tool in gating,
                "log": str(log), "bundle": str(bundle), "ntok": 1, "watchdog_seconds": 1800,
            })
            print(f"{pipeline} {image} {bundle.name} {tool}: {'PASS' if passed else 'FAIL'}", flush=True)
    if build_generation(owner) != generation:
        raise EvidenceError("Build changed during sanitizers; rerun P2 sanitizers")
    admitted = all(
        record["passed"] if record["gating"] else isinstance(record["count"], int)
        for record in records
    )
    result = {
        "generated_at": utc_now(), "gpu": GPU, "required": required,
        "build_generation": generation,
        "passed": all(record["passed"] for record in records),
        "admission": {"policy": SANITIZER_ADMISSION, "gating_tools": sorted(gating),
                      "admitted": admitted},
        "records": records,
    }
    sanitize_receipt(owner).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if admitted else 1


def require_sanitized_images() -> dict[str, dict[str, Any]]:
    """Admission per SANITIZER_ADMISSION; returns per-case recorded counts for the results.

    Each image is admitted by the receipt of the build that produced it, so a
    cross-library case reads both the reference and the CuTe receipts.
    """
    receipts: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for pipeline in sorted({role_pipeline(case, role) for case in CASES for role in ("reference", "linked")}):
        path = sanitize_receipt(pipeline)
        if str(path) in receipts:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("build_generation") != build_generation(pipeline):
            raise EvidenceError(f"P3 requires sanitizers from the current build generation: {path}")
        if not result.get("admission", {}).get("admitted"):
            raise EvidenceError(f"P3 requires admitted sanitizers: {path}")
        receipts[str(path)] = {
            (row["image"], row["bundle"], row["tool"]): row for row in result["records"]
        }
    per_case: dict[str, dict[str, Any]] = {}
    for case in CASES:
        counts: dict[str, Any] = {}
        for role in ("reference", "linked"):
            pipeline = role_pipeline(case, role)
            image_file = image_path(pipeline, case[f"{role}_image"])
            records = receipts[str(sanitize_receipt(pipeline))]
            for tool in ("memcheck", "racecheck", "synccheck"):
                row = records.get((str(image_file), str(sanitizer_bundle(case)), tool))
                if row is None:
                    raise EvidenceError(f"P3 sanitizer coverage missing: {case['case_id']} {role} {tool}")
                if row["gating"] and not row["passed"]:
                    raise EvidenceError(f"P3 refuses: {tool} not clean for {case['case_id']} {role}")
                if not isinstance(row["count"], int):
                    raise EvidenceError(f"P3 refuses: {tool} count not recorded for {case['case_id']} {role}")
                counts[f"{role}_{tool}"] = {"count": row["count"], "strict_passed": row["passed"]}
        per_case[case["case_id"]] = counts
    return per_case


def selected_pairs(result: dict[str, Any]) -> list[dict[str, Any]]:
    target = result["summary"]["pair_valid_target"]
    return [pair for pair in result["pairs"] if pair["pair_valid"]][:target]


def collect_case(case: dict[str, Any], result: dict[str, Any], pair_cap_error: str | None) -> dict[str, Any]:
    summary = result["summary"]
    selected = selected_pairs(result)
    slots: list[dict[str, Any]] = []
    valid_ms: dict[str, list[float]] = {"reference": [], "linked": []}
    all_cuda_min: list[float] = []
    all_cuda_max: list[float] = []
    batch_sizes: set[int] = set()
    for pair in result["pairs"]:
        for role in ("reference", "linked"):
            output = pair["outputs"][role]
            throughput = float(output["throughput"])
            ms_per_token = 1000.0 / throughput
            slot = {
                "attempt": pair["attempt"],
                "role": role,
                "pair_valid": pair["pair_valid"],
                "validity_class": pair["validity_class"],
                "throughput_tokens_per_second": throughput,
                "milliseconds_per_token": ms_per_token,
                "iterations": output["iterations"],
                "steady_cuda_event_milliseconds": output["steady_cuda_event_milliseconds"],
                "batch_tokens": output["inner_batch_launches"],
                "observed_batch_cuda_ms_min": output["minimum_observed_batch_ms"],
                "observed_batch_cuda_ms_max": output["maximum_observed_batch_ms"],
                "observed_batch_wall_ms_min": output["minimum_observed_batch_wall_ms"],
                "observed_batch_wall_ms_max": output["maximum_observed_batch_wall_ms"],
            }
            slots.append(slot)
            batch_sizes.add(output["inner_batch_launches"])
            all_cuda_min.append(float(output["minimum_observed_batch_ms"]))
            all_cuda_max.append(float(output["maximum_observed_batch_ms"]))
    for pair in selected:
        for role in ("reference", "linked"):
            valid_ms[role].append(1000.0 / float(pair["outputs"][role]["throughput"]))

    invariant = summary["validity_count_invariant"]
    if invariant.get("equal") is not True:
        raise EvidenceError(f"{case['case_id']}: harness validity invariant failed")
    gate_outcome = "NOT_GATED"
    if case["kind"] == "seam-parity":
        if summary["median_ratio_linked_over_reference"] is None:
            gate_outcome = "EXECUTION_FAIL"
        else:
            gate_outcome = (
                "PASS"
                if summary["median_ratio_linked_over_reference"] >= RATIO_FLOOR
                else "MISS"
            )
    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "pipeline": case["pipeline"],
        "reference_pipeline": role_pipeline(case, "reference"),
        "bundle": case["bundle"],
        "pos_id": case["pos_id"],
        "reference_image": case["reference_image"],
        "linked_image": case["linked_image"],
        "status": "EXECUTION_FAIL" if pair_cap_error else summary["status"],
        "pair_cap_error": pair_cap_error,
        "pair_valid_target": summary["pair_valid_target"],
        "pair_valid_count": summary["pair_valid_count"],
        "pair_valid_pairs_collected": summary["pair_valid_pairs_collected"],
        "attempted_pairs": summary["attempted_pairs"],
        "original_valid_count": summary["original_valid_count"],
        "amended_valid_count": summary["amended_valid_count"],
        "validity_count_invariant": invariant,
        "operating_regime": summary["operating_regime"],
        "pair_valid_ratios_linked_over_reference": summary[
            "pair_valid_ratios_linked_over_reference"
        ],
        "median_ratio_linked_over_reference": summary[
            "median_ratio_linked_over_reference"
        ],
        "ratio_floor": summary["ratio_floor"],
        "gate_outcome": gate_outcome,
        "reference_milliseconds_per_token_median": (
            statistics.median(valid_ms["reference"]) if valid_ms["reference"] else None
        ),
        "linked_milliseconds_per_token_median": (
            statistics.median(valid_ms["linked"]) if valid_ms["linked"] else None
        ),
        "batch_token_counts": sorted(batch_sizes),
        "observed_batch_cuda_ms_min": min(all_cuda_min) if all_cuda_min else None,
        "observed_batch_cuda_ms_max": max(all_cuda_max) if all_cuda_max else None,
        "slots": slots,
        "archive": str((ARCHIVE_ROOT / case["case_id"]).relative_to(ROOT)),
    }


def collect_from_archive(case: dict[str, Any], pair_cap_error: str | None = None) -> dict[str, Any]:
    archive = ARCHIVE_ROOT / case["case_id"]
    summary_path = archive / "summary.json"
    if not summary_path.is_file():
        raise EvidenceError(f"{case['case_id']}: harness did not write summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pairs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((archive / "pairs").glob("pair-*.json"))
    ]
    return collect_case(case, {"summary": summary, "pairs": pairs}, pair_cap_error)


def json_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError("non-finite number cannot be rendered")
    return str(value)


def write_result_markdown(payload: dict[str, Any]) -> None:
    rows = []
    for case in payload["cases"]:
        comparison = f"{case['linked_image']} / {case['reference_image']}"
        validity = (
            f"{case['pair_valid_count']}/{case['attempted_pairs']} pair-valid; "
            f"{case['operating_regime']}"
        )
        rows.append(
            "| "
            + " | ".join(
                (
                    case["case_id"],
                    comparison,
                    json_number(case["median_ratio_linked_over_reference"]),
                    str(case["pair_valid_count"]),
                    json_number(case["reference_milliseconds_per_token_median"]),
                    json_number(case["linked_milliseconds_per_token_median"]),
                    ",".join(str(value) for value in case["batch_token_counts"]),
                    validity,
                    case["gate_outcome"],
                )
            )
            + " |"
        )
    seam_pass = (
        len(payload["cases"]) == len(CASES)
        and all(case["gate_outcome"] == "PASS" for case in payload["cases"])
    )
    cuda_min = min(
        case["observed_batch_cuda_ms_min"]
        for case in payload["cases"]
        if case["observed_batch_cuda_ms_min"] is not None
    )
    cuda_max = max(
        case["observed_batch_cuda_ms_max"]
        for case in payload["cases"]
        if case["observed_batch_cuda_ms_max"] is not None
    )
    p2_note = (
        f"**P2 numerical gate PASS:** all {len(payload['p2_receipts'])} receipts "
        "(five bundles × four pipelines) pass; P3 admission requires it.\n\n"
    )
    tools = ("memcheck", "racecheck", "synccheck")
    sanitizer_rows = "\n".join(
        f"| {case['case_id']} | "
        + " | ".join(
            f"{case['sanitizers'][f'linked_{tool}']['count']} / {case['sanitizers'][f'reference_{tool}']['count']}"
            for tool in tools
        )
        + " |"
        for case in payload["cases"]
    )
    text = f"""# Megakernel llama-1b port — seam-parity evidence

Rendered by `parity/run_parity.py` from `results.json` (campaign generated {payload['generated_at']}).

## Verdict

**Seam-parity gate {'PASS' if seam_pass else 'MISS'}** — all {len(CASES)} linked / same-source one-TU cases {'meet' if seam_pass else 'do not all meet'} the median ratio floor of {RATIO_FLOOR}.

{p2_note}| case | linked / reference | median ratio | pair-valid count | reference median ms/token | linked median ms/token | batch tokens | clock validity | 0.97 seam gate |
|---|---|---:|---:|---:|---:|---:|---|---|
{chr(10).join(rows)}

Each slot calibrated a one-token probe and selected an integer batch at a 50 ms target, capped at {MAX_BATCH_TOKENS} tokens. The per-case batch sizes are in the table. Across every slot, observed CUDA-event batch time ranged from {json_number(cuda_min)} ms to {json_number(cuda_max)} ms, below the 100 ms harness quantum. gpulockd held the selected GPU at {payload['lock']['target_sm_mhz']} MHz; its pre-case, per-case, and post-case status checks were healthy with zero recorded violations.

## Sanitizers

Counts on the timed images, linked / one-TU. Admission requires memcheck zero; racecheck and synccheck counts are recorded, not admission predicates. Logs cap reported hazards, so racecheck counts are not exhaustive tallies.

| case | memcheck | racecheck | synccheck |
|---|---:|---:|---:|
{sanitizer_rows}

## Images

`port/build.sh` builds the host-ABI-v3 linked, same-source one-TU and monolithic images with fast math (FTZ and FMA enabled, precise division and square root disabled) from upstream Megakernels `7309cec` plus the minimal correctness patches in `port/patches/` (partial-store `laneid`, partial KV stage index, tensor-allocator barrier, reduction page lifetime); without them the multi-partition attention path is dead or racy upstream. The reference is therefore patched upstream, not stock upstream.

## Interpretation

Each PipeLink-linked cubin is compared with its same-source one-TU build, isolating the ABI/link seam. MK-A-1024-red additionally executes two attention partials per KV head followed by terminal attention reductions in all sixteen layers. No upstream performance comparisons are part of this campaign.

## Method

Each case was executed by `evaluator.harness.parity.run_case` with `Protocol.definitive`: five untimed single-token check-mode warmups per image, 3 s nominal slots, 1 s leading discard, first eight chronological primary-valid pairs, and the shared ±3% clock predicate. Before every warmup request and every timed slot request, the adapter ran a separate single-token Gate P2 recheck: all nine outputs finite and layer-0 `k_row`/`v_row` within `rtol=2e-2`, `atol=2e-2*RMS(reference)`. Throughput is complete steady-window tokens divided by summed CUDA-event batch time. Pair orientation is always linked/reference; no ratios were imputed.

Running the campaign also writes machine-readable results (`results.json`) and sanitizer receipts (`sanitize.json`) next to this report; they are not published. Harness-owned requests, slot outputs, telemetry, pair decisions, and summaries are written to the git-ignored `{ARCHIVE_ROOT.relative_to(EVIDENCE)}/<case-id>/`.
"""
    RESULT_MARKDOWN_PATH.write_text(text, encoding="utf-8")


CUTE_CONTEXT_LABELS = {"p16-n2": "16", "p1024-n8-red": "1024, 2 KV partitions + reduction", "p4096-n2": "4096"}


def write_cute_markdown(payload: dict[str, Any]) -> None:
    def row(case: dict[str, Any], *cells: str) -> str:
        return "| " + " | ".join((case["case_id"], CUTE_CONTEXT_LABELS[case["bundle"]], *cells)) + " |"

    def common(case: dict[str, Any]) -> tuple[str, ...]:
        return (
            json_number(case["median_ratio_linked_over_reference"]),
            f"{case['pair_valid_count']}/{case['attempted_pairs']}",
            json_number(case["reference_milliseconds_per_token_median"]),
            json_number(case["linked_milliseconds_per_token_median"]),
            ",".join(str(value) for value in case["batch_token_counts"]),
        )

    seam = [case for case in payload["cases"] if case["kind"] == "seam-parity"]
    cross = [case for case in payload["cases"] if case["kind"] == "cross-library"]
    seam_rows = "\n".join(row(case, *common(case), case["gate_outcome"]) for case in seam)
    cross_rows = "\n".join(row(case, *common(case), case["status"]) for case in cross)
    tools = ("memcheck", "racecheck", "synccheck")
    sanitizer_rows = "\n".join(
        row(case, *(f"{case['sanitizers'][f'linked_{tool}']['count']} / "
                    f"{case['sanitizers'][f'reference_{tool}']['count']}" for tool in tools))
        for case in payload["cases"]
    )
    images = "\n".join(f"- `{path}`: `{digest[:12]}`" for path, digest in sorted(payload.get("images", {}).items()))
    generations = ", ".join(f"{name} `{value[:8]}`" for name, value in payload.get("build_generations", {}).items())
    status = "complete" if payload["execution_error"] is None and len(payload["cases"]) == len(CASES) else "INCOMPLETE"
    text = f"""# Megakernel CuTe-attention composition — definitive evidence

Rendered by `parity/run_parity.py --suite cute` (campaign generated {payload['generated_at']}; run {status}).

Pipeline `cute` replaces only the attention-partial package of pipeline `a` with a CUTLASS/CuTe
implementation; the other eight packages, the host provider and the instruction schedule are unchanged.

## Seam parity: CuTe linked / CuTe same-source one-TU

| case | context | median ratio | pair-valid / attempted | one-TU median ms/token | linked median ms/token | batch tokens | 0.97 seam gate |
|---|---|---:|---:|---:|---:|---:|---|
{seam_rows}

## Cross-library speedup: CuTe linked / all-ThunderKittens linked

The median ratio is the paired throughput speedup of the CuTe composition over the all-ThunderKittens
linked kernel. It is descriptive; the seam gate does not apply.

| case | context | paired speedup | pair-valid / attempted | ThunderKittens median ms/token | CuTe median ms/token | batch tokens | harness status |
|---|---|---:|---:|---:|---:|---:|---|
{cross_rows}

## Sanitizers

Counts on the timed images, candidate / reference (one token; short contexts use the 16-token bundle, the
reduction case its own bundle). Admission requires memcheck zero; racecheck and synccheck counts are recorded,
not admission predicates. Logs cap reported hazards, so racecheck counts are not exhaustive tallies.

| case | context | memcheck | racecheck | synccheck |
|---|---|---:|---:|---:|
{sanitizer_rows}

## Method

Each case was executed by `evaluator.harness.parity.run_case` with `Protocol.definitive`, exactly as the
[all-ThunderKittens evidence](RESULT.md#method): five untimed single-token check-mode warmups per image,
3 s nominal slots, 1 s leading discard, first eight chronological primary-valid pairs, the shared ±3% clock
predicate, and a single-token correctness recheck before every warmup and timed slot. Each slot sized its
batch from a one-token probe (50 ms target, at most {MAX_BATCH_TOKENS} tokens). gpulockd held the GPU at
{payload['lock']['target_sm_mhz']} MHz. P3 admission required all {len(payload['p2_receipts'])} P2 numerical
receipts (ThunderKittens and CuTe) to pass.

Images (build generations {generations}), SHA-256 prefixes:

{images}
"""
    RESULT_MARKDOWN_PATH.write_text(text, encoding="utf-8")


def write_results(payload: dict[str, Any]) -> None:
    temporary = RESULTS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESULTS_PATH)
    if payload["cases"]:
        (write_cute_markdown if SUITE == "cute" else write_result_markdown)(payload)


P2_BUNDLES = ("p10-n8-l1", "p1024-n8-l1", "p10-n8", "p1024-n8", "p1024-n8-red")
P2_PIPELINES = ("a", "rebind-smem", "rebind-regs", "rebind-opcode")
# CuTe P2: both layer-1 fixtures plus full depth at the three timed contexts.
CUTE_P2 = RUNS / "cute-20261007" / "final-check"
CUTE_P2_FULL_DEPTH = ("p16-n2", "p1024-n8-red", "p4096-n2")


def require_p2_pass() -> list[dict[str, Any]]:
    """P3 admission: every saved P2 numerical receipt must PASS (the CuTe suite also
    needs the ThunderKittens receipts: its cross-library reference is pipeline `a`)."""
    paths = []
    for name in P2_BUNDLES:
        directory = RUNS / "submission-2026-09-17" if name.endswith("-red") else RUNS
        paths += [(name, pipeline, directory / name / f"compare-{pipeline}.json") for pipeline in P2_PIPELINES]
    if SUITE == "cute":
        paths += [(name, "cute", CUTE_P2 / name / "comparison.json") for name in P2_BUNDLES[:2] + CUTE_P2_FULL_DEPTH]
    receipts = []
    for name, pipeline, path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "PASS":
            raise EvidenceError(f"P3 requires every P2 receipt to PASS: {path}")
        receipts.append({
            "bundle": name, "pipeline": pipeline, "status": result["status"],
            "path": str(path.relative_to(ROOT)),
        })
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="reference")
    parser.add_argument("--sanitize", action="store_true", help="P2 sanitizer pass on the P3 images")
    args = parser.parse_args()
    if GPU is None or GPU <= 0:
        parser.error("set MK_GPU to the positive physical GPU index used for this campaign")
    select_suite(args.suite)
    if args.sanitize:
        return run_sanitizers()
    p2_receipts = require_p2_pass()
    target_text = os.environ.get("PIPELINK_LOCK_SM_MHZ")
    if not target_text:
        raise EvidenceError("PIPELINK_LOCK_SM_MHZ is not set; source tools/env.sh")
    try:
        target_mhz = int(target_text)
    except ValueError as exc:
        raise EvidenceError("PIPELINK_LOCK_SM_MHZ must be an integer") from exc
    if target_mhz <= 0:
        raise EvidenceError("PIPELINK_LOCK_SM_MHZ must be positive")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "pipelink-megakernel-parity-results-v1",
        "suite": SUITE,
        "generated_at": utc_now(),
        "gpu": GPU,
        "protocol": "definitive",
        "ratio_floor": RATIO_FLOOR,
        "batch_token_cap": MAX_BATCH_TOKENS,
        "batch_sizing": "one-token CUDA-event probe; floor(50 ms / probe ms), clamped to [1,32]",
        "lock": {
            "target_sm_mhz": target_mhz,
            "start": None,
            "initial_status": None,
            "case_status_checks": [],
            "final_status": None,
            "stop": None,
        },
        "cases": [],
        "execution_error": None,
        "p2_receipts": p2_receipts,
    }
    lock_started = False
    execution_failed = False
    try:
        image_paths = {
            image_path(role_pipeline(case, role), case[f"{role}_image"])
            for case in CASES
            for role in ("reference", "linked")
        }
        image_hashes = {path: sha256_file(path) for path in sorted(image_paths)}
        payload["images"] = {str(path.relative_to(ROOT)): digest for path, digest in image_hashes.items()}
        payload["build_generations"] = {
            pipeline: build_generation(pipeline)
            for pipeline in sorted({role_pipeline(case, role) for case in CASES for role in ("reference", "linked")})
        }
        sanitizer_counts = require_sanitized_images()
        payload["sanitizer_admission"] = SANITIZER_ADMISSION
        payload["lock"]["start"] = start_lock(target_mhz)
        lock_started = True
        payload["lock"]["initial_status"] = check_lock("before-cases", target_mhz)
        for case in CASES:
            before = check_lock(f"before-{case['case_id']}", target_mhz)
            payload["lock"]["case_status_checks"].append(before)
            archive = ARCHIVE_ROOT / case["case_id"]
            if archive.exists():
                shutil.rmtree(archive)
            reference = image_spec(case, "reference", image_hashes)
            linked = image_spec(case, "linked", image_hashes)
            protocol = Protocol.definitive(archive, case_config(case))
            try:
                result = run_case(case["case_id"], reference, linked, GPU, protocol)
                case_result = collect_case(case, result, None)
            except PairCapError as exc:
                case_result = collect_from_archive(case, str(exc))
                execution_failed = True
            case_result["sanitizers"] = sanitizer_counts[case["case_id"]]
            payload["cases"].append(case_result)
            after = check_lock(f"after-{case['case_id']}", target_mhz)
            payload["lock"]["case_status_checks"].append(after)
            write_results(payload)
        payload["lock"]["final_status"] = check_lock("after-cases", target_mhz)
    except Exception as exc:
        execution_failed = True
        payload["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if lock_started:
            if payload["lock"]["final_status"] is None:
                try:
                    payload["lock"]["final_status"] = check_lock("finally-before-stop", target_mhz)
                except Exception as exc:
                    execution_failed = True
                    payload["lock"]["final_status"] = {
                        "label": "finally-before-stop",
                        "honored": False,
                        "error": str(exc),
                    }
            stop = stop_lock()
            payload["lock"]["stop"] = stop
            if stop["returncode"] != 0:
                execution_failed = True
                if payload["execution_error"] is None:
                    payload["execution_error"] = {
                        "type": "EvidenceError",
                        "message": f"gpulockd stop failed: {stop}",
                    }
        payload["generated_at"] = utc_now()
        write_results(payload)

    if payload["execution_error"] is not None:
        print(json.dumps(payload["execution_error"], sort_keys=True), file=sys.stderr)
    return 1 if execution_failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MK parity driver: {exc}", file=sys.stderr)
        raise SystemExit(1)
