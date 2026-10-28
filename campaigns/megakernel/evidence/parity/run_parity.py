#!/usr/bin/env python3
"""Run the local MK comparisons with the paired-reload-v2 protocol.

Suite `reference` compares each all-ThunderKittens linked image with its same-source
one-TU build. Suite `cute` does the same for the CuTe-attention composition and adds
cross-library pairs (all-ThunderKittens linked reference, CuTe linked candidate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import socket
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
BUNDLES = ROOT / "campaigns" / "megakernel" / "host" / "bundles"
PORT_BUILD = ROOT / "campaigns" / "megakernel" / "port" / "build"
# Set by select_suite under --output-root, outside the published evidence directory:
# <root>/p2/<bundle>/compare-<pipeline>.json holds the numerical receipts and
# <root>/<suite>/ the sanitizer receipts, paired checks and timing results.
RECEIPT_ROOT: Path
P2_ROOT: Path
OUTPUT_ROOT: Path
SANITIZER_RUNS: Path
RESULTS_PATH: Path
RESULT_MARKDOWN_PATH: Path
SUITE = "reference"
# MK_GPU selects one of the configured devices; by default the first of PIPELINK_GPUS.
_GPU = os.environ.get("MK_GPU") or os.environ.get("PIPELINK_GPUS", "").split(",")[0]
GPU = int(_GPU) if _GPU.strip() else None
HOST = BUNDLES.parent
RUNNER = HOST / "build" / "mk_runner"
PROVIDER = HOST / "build" / "provider.so"
RATIO_FLOOR = 0.97


def build_record(pipeline: str) -> Path:
    # The CuTe images come from `build.sh --cute` with their own build generation.
    return PORT_BUILD / ("build-record-cute.json" if pipeline == "cute" else "build-record.json")


def sanitize_receipt(pipeline: str) -> Path:
    return (RECEIPT_ROOT / "cute" / "sanitize-cute.json" if pipeline == "cute"
            else RECEIPT_ROOT / "reference" / "sanitize.json")


sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HOST))
from launch import launch_command, paired_command
from evaluator.harness.parity.fixed_count import PROTOCOL, run_native_pair
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
    # Paired speedup of CuTe over all-ThunderKittens throughput; reported, not gated.
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
    },
    "cute": {
        "cases": CUTE_CASES, "pipelines": ("cute",),
    },
}


def select_suite(name: str, output_root: Path) -> None:
    global SUITE, CASES, RECEIPT_ROOT, P2_ROOT, OUTPUT_ROOT, SANITIZER_RUNS, RESULTS_PATH, RESULT_MARKDOWN_PATH
    SUITE, CASES = name, SUITES[name]["cases"]
    RECEIPT_ROOT = output_root.resolve()
    P2_ROOT = RECEIPT_ROOT / "p2"
    OUTPUT_ROOT = RECEIPT_ROOT / name
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SANITIZER_RUNS = OUTPUT_ROOT / "image-sanitizers"
    RESULTS_PATH = OUTPUT_ROOT / "results.json"
    RESULT_MARKDOWN_PATH = OUTPUT_ROOT / "RESULT.md"


def role_pipeline(case: dict[str, Any], role: str) -> str:
    return case.get(f"{role}_pipeline", case["pipeline"])


def sanitizer_bundle(case: dict[str, Any]) -> Path:
    # Reduction cells use the P2-validated p1024-n8-red bundle: the short-prompt
    # p16-n1-red export (2 partitions over a 16-token prompt) stalls natively.
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




def sanitizer_images() -> list[tuple[str, str, Path]]:
    """Unique (pipeline, image, bundle) runs covering this suite's own images."""
    owned = SUITES[SUITE]["pipelines"]
    return sorted({
        (role_pipeline(case, role), case[f"{role}_image"], sanitizer_bundle(case))
        for case in CASES for role in ("reference", "linked")
        if role_pipeline(case, role) in owned
    })


def run_sanitizers() -> int:
    required = {
        "memcheck": 0,
        "racecheck": REQUIRED_RACECHECK_HAZARDS,
        "synccheck": REQUIRED_SYNCCHECK_ERRORS,
    }
    sanitizer = str(Path(os.environ["PIPELINK_CUDA_HOME"]) / "bin" / "compute-sanitizer")
    records = []
    owner = SUITES[SUITE]["pipelines"][0]
    generation = build_generation(owner)
    for pipeline, image, bundle in sanitizer_images():
        path = image_path(pipeline, image)
        for tool, required_count in required.items():
            out = SANITIZER_RUNS / f"{pipeline}-{image}-{bundle.name}" / tool
            out.mkdir(parents=True, exist_ok=False)
            command = launch_command(
                pipeline, image, bundle, out, "check",
                runner_args=("--ntok", "1"), runner=RUNNER, provider=PROVIDER,
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
                "required": required_count, "passed": passed,
                "log": str(log), "bundle": str(bundle), "ntok": 1, "watchdog_seconds": 1800,
            })
            print(f"{pipeline} {image} {bundle.name} {tool}: {'PASS' if passed else 'FAIL'}", flush=True)
    if build_generation(owner) != generation:
        raise EvidenceError("Build changed during sanitizers; rerun P2 sanitizers")
    passed = all(record["passed"] for record in records)
    result = {
        "generated_at": utc_now(), "gpu": GPU, "required": required,
        "build_generation": generation,
        "passed": passed,
        "records": records,
    }
    destination = OUTPUT_ROOT / ("sanitize-cute.json" if owner == "cute" else "sanitize.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if passed else 1


def require_sanitized_images() -> dict[str, dict[str, Any]]:
    """Require every tool clean on every timed image; returns per-case counts for the results.

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
        if not result.get("passed"):
            raise EvidenceError(f"P3 requires clean sanitizers: {path}")
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
                if not row["passed"] or row["returncode"] != 0 or row["count"] != 0:
                    raise EvidenceError(f"P3 refuses: {tool} not clean for {case['case_id']} {role}")
                counts[f"{role}_{tool}"] = {"count": row["count"]}
        per_case[case["case_id"]] = counts
    return per_case


SNAPSHOT_PHASES = ("initial", "before_reload", "after_reload", "final")
TOOLS = ("memcheck", "racecheck", "synccheck")


def physical_images(case: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for arm, role in (("candidate", "linked"), ("reference", "reference")):
        pipeline = role_pipeline(case, role)
        image = case[f"{role}_image"]
        path = image_path(pipeline, image)
        result[arm] = {
            "pipeline": pipeline, "image": image, "path": str(path),
            "sha256": sha256_file(path), "build_generation": build_generation(pipeline),
            "manifest": str(PORT_BUILD / "linked" / pipeline / "link.json"),
            "manifest_sha256": sha256_file(PORT_BUILD / "linked" / pipeline / "link.json"),
        }
    return result


def executable_identity() -> dict[str, Any]:
    return {name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (("runner", RUNNER), ("provider", PROVIDER))}


def pair_command(case: dict[str, Any], out: Path, mode: str) -> list[str]:
    return paired_command(
        role_pipeline(case, "linked"), case["linked_image"],
        role_pipeline(case, "reference"), case["reference_image"],
        (BUNDLES / case["bundle"]).resolve(), out.resolve(), case["pos_id"],
        mode=mode, runner=RUNNER, provider=PROVIDER,
    )


def check_native_payload(case: dict[str, Any], payload: dict[str, Any], out: Path,
                         mode: str) -> dict[str, Any]:
    """Apply compare.py's fresh-slot predicates to the retained snapshots."""
    import numpy as np
    from compare import OUTPUT_NAMES, compare_to_reference, decode, load_raw, parse_manifest

    if payload.get("protocol") != PROTOCOL or payload.get("mode") != mode:
        raise EvidenceError("native paired protocol/mode differs")
    numeric = payload.get("numerical", {})
    if (numeric.get("status") != "PASS" or
            numeric.get("snapshot_phases") != list(SNAPSHOT_PHASES) or
            numeric.get("root") != str(out.resolve())):
        raise EvidenceError("native paired numerical snapshots incomplete")
    images = physical_images(case)
    provenance = payload.get("provenance", {})
    expected = {
        "candidate_image": images["candidate"]["path"],
        "reference_image": images["reference"]["path"],
        "bundle": str((BUNDLES / case["bundle"]).resolve()),
        "visible_device": 0, "shared_runtime_buffers": True,
        "initial_load_order": ["candidate", "reference"],
        "reload_load_order": ["reference", "candidate"], "reload_count": 1,
        "pos_id": case["pos_id"],
        "embedding_rows": int(parse_scalars(BUNDLES / case["bundle"])["ntok"]),
        "queued_probe_repetitions": 2 if mode == "pair-check" else 0,
        "provider_sessions_per_arm": 1,
        "provider_sessions_persist_through_reload": True,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise EvidenceError("native physical images, fixture, or reload provenance differs")
    interval = provenance.get("reload_interval_ns", [])
    if len(interval) != 2 or not 0 < interval[0] < interval[1]:
        raise EvidenceError("native reload interval is missing")
    specs = parse_manifest(BUNDLES / case["bundle"])
    checks = {}
    for phase in SNAPSHOT_PHASES:
        for arm in ("candidate", "reference"):
            tensors = {}
            for name in OUTPUT_NAMES:
                spec = specs[f"ref_tok0_{name}"]
                observed = decode(load_raw(out / phase / arm / f"tok0_{name}.bin", spec), spec.dtype)
                reference = decode(load_raw(spec.path, spec), spec.dtype)
                finite = bool(np.isfinite(observed).all() and np.isfinite(reference).all())
                tensors[name] = {"finite": finite}
                if not finite:
                    raise EvidenceError(f"{phase}/{arm}/{name}: nonfinite snapshot")
                if name in ("k_row", "v_row"):
                    strict = compare_to_reference(observed[0], reference[0])
                    tensors[name]["layer0"] = strict
                    if not strict["pass"]:
                        raise EvidenceError(f"{phase}/{arm}/{name}: strict layer-zero KV mismatch")
            checks[f"{phase}/{arm}"] = tensors
    return {"status": "PASS", "predicate": numeric["predicate"], "snapshots": checks}


def sanitizer_pairs() -> list[dict[str, Any]]:
    """Predeclared coverage: one entry per unique arm pair and sanitizer bundle."""
    pairs = {}
    for case in CASES:
        bundle = sanitizer_bundle(case)
        key = (role_pipeline(case, "linked"), case["linked_image"],
               role_pipeline(case, "reference"), case["reference_image"], bundle.name)
        pairs[key] = {
            **case, "case_id": "-".join(key), "bundle": bundle.name,
            "pos_id": int(parse_scalars(bundle)["pos_id_first"]),
        }
    return [pairs[key] for key in sorted(pairs)]


def pair_coverage(case: dict[str, Any]) -> dict[str, Any]:
    bundle = sanitizer_bundle(case).resolve()
    return {"images": physical_images(case), "bundle": str(bundle),
            "manifest_sha256": sha256_file(bundle / "manifest.txt"),
            "scalars_sha256": sha256_file(bundle / "scalars.txt")}


def run_pair_checks(*, sanitize: bool) -> int:
    # These checks run inside gpu_window.sh; timing must not, because
    # run_native_pair owns its own window and clock lock.
    if not os.environ.get("PIPELINK_GPU_WINDOW"):
        raise EvidenceError("wrap --pair-check / --sanitize-pair in tools/gpu_window.sh")
    root = OUTPUT_ROOT / ("paired-sanitizers" if sanitize else "pair-check")
    root.mkdir(parents=True, exist_ok=False)
    identity = executable_identity()
    records = []
    cases = sanitizer_pairs() if sanitize else CASES
    sanitizer = str(Path(os.environ["PIPELINK_CUDA_HOME"]) / "bin" / "compute-sanitizer") if sanitize else None
    for case in cases:
        coverage = pair_coverage(case) if sanitize else {"images": physical_images(case)}
        for tool in (TOOLS if sanitize else ("native",)):
            out = root / case["case_id"] / tool
            out.mkdir(parents=True, exist_ok=False)
            native_command = pair_command(case, out, "pair-check")
            sanitizer_log = out / "sanitizer.log"
            command = ([sanitizer, "--tool", tool, "--error-exitcode", "99",
                        "--log-file", str(sanitizer_log), *native_command]
                       if sanitize else native_command)
            record = run_command(command, env=dict(
                os.environ, CUDA_VISIBLE_DEVICES=str(GPU), MK_WATCHDOG_SECONDS="1800"))
            stdout, stderr = record.pop("stdout"), record.pop("stderr")
            (out / "native.stdout").write_text(stdout)
            (out / "native.stderr").write_text(stderr)
            count = None
            if sanitize:
                pattern = (r"RACECHECK SUMMARY:\s*(\d+)\s+hazards?" if tool == "racecheck"
                           else r"ERROR SUMMARY:\s*(\d+)\s+errors?[ \t]*$")
                text = sanitizer_log.read_text() if sanitizer_log.exists() else ""
                counts = re.findall(pattern, text, re.MULTILINE)
                count = int(counts[0]) if len(counts) == 1 else None
            passed = record["returncode"] == 0 and (not sanitize or count == 0)
            numeric = None
            error = None
            try:
                native_payload = json.loads(stdout)
                numeric = check_native_payload(case, native_payload, out, "pair-check")
            except (ValueError, KeyError, OSError, EvidenceError) as exc:
                passed, error = False, str(exc)
            records.append({**record, "case_id": case["case_id"], "coverage": coverage,
                            "tool": tool, "count": count, "passed": passed, "numeric": numeric,
                            "error": error, "out": str(out),
                            "sanitizer_log": str(sanitizer_log) if sanitize else None})
            print(f"{case['case_id']} {tool}: {'PASS' if passed else 'FAIL'}", flush=True)
    unchanged = identity == executable_identity() and all(
        record["coverage"] == (pair_coverage(case) if sanitize else {"images": physical_images(case)})
        for case in cases for record in records if record["case_id"] == case["case_id"])
    payload = {"protocol": PROTOCOL, "action": "sanitize-pair" if sanitize else "pair-check",
               "generated_at": utc_now(), "hostname": socket.gethostname(), "gpu": GPU,
               "executables": identity, "records": records,
               "passed": unchanged and all(row["passed"] for row in records)}
    (root / "receipt.json").write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if payload["passed"] else 1


def require_pair_sanitizers(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if (receipt.get("protocol") != PROTOCOL or receipt.get("action") != "sanitize-pair" or
            not receipt.get("passed") or receipt.get("executables") != executable_identity()):
        raise EvidenceError(f"current runner/provider paired sanitizer evidence required: {path}")
    for case in CASES:
        coverage = pair_coverage(case)
        for tool in TOOLS:
            matches = [row for row in receipt["records"]
                       if row["coverage"] == coverage and row["tool"] == tool]
            if (len(matches) != 1 or not matches[0]["passed"] or
                    matches[0]["returncode"] != 0 or matches[0]["count"] != 0 or
                    matches[0].get("numeric", {}).get("status") != "PASS"):
                raise EvidenceError(f"paired sanitizer coverage missing/failed: {case['case_id']} {tool}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "status": "PASS"}


def collect_case(case: dict[str, Any], execution: dict[str, Any],
                 numerical: dict[str, Any] | None) -> dict[str, Any]:
    native = execution.get("payload") or {}
    admitted = bool(execution["admitted"] and numerical and numerical["status"] == "PASS")
    ratio = native.get("pair", {}).get("ratio_onetu_over_linked") if admitted else None
    gate = ("PASS" if ratio >= RATIO_FLOOR else "MISS") if admitted and case["kind"] == "seam-parity" else (
        "NOT_GATED" if admitted else "EXECUTION_FAIL")
    return {**case, "protocol": PROTOCOL, "status": "PASS" if admitted else "EXECUTION_FAIL",
            "ratio_onetu_over_linked": ratio,
            "ratio_floor": RATIO_FLOOR if case["kind"] == "seam-parity" else None,
            "gate_outcome": gate, "pair": native.get("pair"), "numerical": numerical,
            "provenance": native.get("provenance"), "execution": execution}


C1_BANNER = ("> Unlocked or unrecorded SM clocks: timings are valid as relative comparisons only, "
             "not comparable with the paper's locked-clock numbers.")


def arm_ms_per_token(pair: dict[str, Any]) -> dict[str, float]:
    """Median over blocks of each arm's two batch totals (2 x 10 decodes) per token."""
    sums: dict[str, list[float]] = {"candidate": [], "reference": []}
    for block in pair["blocks"]:
        for arm, letter in (("candidate", "A"), ("reference", "B")):
            sums[arm].append(sum(ms for order, ms in zip(block["order"], block["milliseconds"])
                                 if order == letter) / (2 * pair["repetitions"]))
    return {arm: statistics.median(values) for arm, values in sums.items()}


def write_results(payload: dict[str, Any]) -> None:
    temporary = RESULTS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS_PATH)
    rows = []
    for case in payload["cases"]:
        images = case["images"]
        ratio = case["ratio_onetu_over_linked"]
        ms = arm_ms_per_token(case["pair"]) if case["pair"] and case["pair"].get("blocks") else {}
        rows.append(f"| {case['case_id']} | {images['candidate']['pipeline']}:{images['candidate']['image']} | "
                    f"{images['reference']['pipeline']}:{images['reference']['image']} | "
                    f"{ratio if ratio is not None else 'n/a'} | "
                    f"{ms.get('reference', float('nan')):.6f} | {ms.get('candidate', float('nan')):.6f} | "
                    f"{case['status']} | {case['gate_outcome']} |")
    clock = (f"SM clock locked at {payload['clock_mhz']} MHz" if payload["clock_locked"]
             else "SM clock unlocked")
    RESULT_MARKDOWN_PATH.write_text(
        (C1_BANNER + "\n\n" if not payload["clock_locked"] else "") +
        f"# Megakernel {SUITE} — paired reload evidence\n\n"
        f"Protocol `{PROTOCOL}`; local execution on one GPU; "
        f"{clock}. Generated {payload['generated_at']}.\n\n"
        "| case | candidate A | reference B | speedup B/A | reference ms/token | candidate ms/token | admission | seam gate |\n"
        "|---|---|---|---:|---:|---:|---|---|\n" + "\n".join(rows) + "\n\n"
        "Each case runs 8 blocks × 4 batches × 10 complete fixed-position decodes, ordered "
        "ABBA, BAAB, BAAB, ABBA | BAAB, ABBA, ABBA, BAAB. The result is "
        "sqrt(median(first four block B/A ratios) × median(last four)); each block ratio "
        "is the sum of its two reference times divided by the sum of its two candidate times. "
        "An arm's ms/token is the median over the eight blocks of its two batch totals divided by 20. "
        "Only same-source seam comparisons have the 0.97 gate; cross-library ratios are reported ungated. "
        "Each case runs once, including on a MISS, with no calibration, duration gate, "
        "time slots or per-arm telemetry coverage.\n\n"
        "One context, the shared fixture and buffers, immutable pinned embedding rows and "
        "per-image opaque params persist. Both modules unload at the midpoint and reload "
        "reference-first with refreshed function attributes. Each event interval covers Bar "
        "reset, embedding HtoD and the complete decode. Each arm runs one provider query/build "
        "before all batches; both sessions and their opaque parameter storage persist "
        "through the reload and are released after the final drain. The common helper runs "
        "eight warmup calls per arm before each half. Four numerical snapshot "
        "phases per arm run outside scored timing. results.json and its local execution "
        "directories keep all 32 raw batch totals, epoch intervals, image identities, "
        "lock/window receipts and snapshots.\n\n"
        "Admission requires the P2 predicates, sanitizer receipts for the current image "
        "generation, runner/provider-bound paired reload sanitizer receipts and "
        "finite-nine/strict-layer0-KV snapshot checks through compare.py.\n"
    )


P2_BUNDLES = ("p10-n8-l1", "p1024-n8-l1", "p10-n8", "p1024-n8", "p1024-n8-red")
P2_PIPELINES = ("a", "rebind-smem", "rebind-regs", "rebind-opcode")
# CuTe P2: both layer-1 fixtures plus full depth at the three timed contexts.
CUTE_P2_BUNDLES = ("p10-n8-l1", "p1024-n8-l1", "p16-n2", "p1024-n8-red", "p4096-n2")


def require_p2_pass() -> list[dict[str, Any]]:
    """P3 admission: every saved P2 numerical receipt must PASS. The CuTe suite also
    needs the ThunderKittens receipts because its cross-library reference is pipeline `a`."""
    paths = [(name, pipeline, P2_ROOT / name / f"compare-{pipeline}.json")
             for name in P2_BUNDLES for pipeline in P2_PIPELINES]
    if SUITE == "cute":
        paths += [(name, "cute", P2_ROOT / name / "compare-cute.json") for name in CUTE_P2_BUNDLES]
    receipts = []
    for name, pipeline, path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "PASS":
            raise EvidenceError(f"P3 requires every P2 receipt to PASS: {path}")
        receipts.append({
            "bundle": name, "pipeline": pipeline, "status": result["status"],
            "path": str(path),
        })
    return receipts


def main() -> int:
    global RUNNER, PROVIDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="reference")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--sanitize", action="store_true", help="per-image P2 sanitizer coverage")
    actions.add_argument("--sanitize-pair", action="store_true", help="bounded paired/reload sanitizer coverage")
    actions.add_argument("--pair-check", action="store_true", help="bounded numerical/reload check; no scored blocks")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="local staging root: p2/ numerical receipts, one subdirectory per suite")
    parser.add_argument("--runner", type=Path, default=RUNNER)
    parser.add_argument("--provider", type=Path, default=PROVIDER)
    parser.add_argument("--paired-sanitizers", type=Path,
                        help="paired sanitizer receipt; default OUTPUT_ROOT/SUITE/paired-sanitizers/receipt.json")
    args = parser.parse_args()
    if GPU is None:
        parser.error("set MK_GPU or PIPELINK_GPUS (source tools/env.sh)")
    RUNNER, PROVIDER = args.runner.resolve(), args.provider.resolve()
    select_suite(args.suite, args.output_root)
    for case in CASES:
        case_config(case)  # Resolve the local bundles.
    if args.sanitize:
        if not os.environ.get("PIPELINK_GPU_WINDOW"):
            raise EvidenceError("wrap --sanitize in tools/gpu_window.sh")
        return run_sanitizers()
    p2_receipts = require_p2_pass()
    if args.pair_check or args.sanitize_pair:
        return run_pair_checks(sanitize=args.sanitize_pair)
    if os.environ.get("PIPELINK_GPU_WINDOW"):
        raise EvidenceError("normal timing owns its window; remove the outer gpu_window.sh")
    if RESULTS_PATH.exists() or RESULT_MARKDOWN_PATH.exists():
        raise EvidenceError("timing output already exists; choose a fresh --output-root, never overwrite evidence")
    # Unset, empty or 0 PIPELINK_LOCK_SM_MHZ runs with unlocked clocks.
    target_mhz = int(os.environ.get("PIPELINK_LOCK_SM_MHZ") or 0)
    sanitizer_counts = require_sanitized_images()
    paired_sanitizers = require_pair_sanitizers(
        args.paired_sanitizers or OUTPUT_ROOT / "paired-sanitizers" / "receipt.json")
    executables = executable_identity()
    payload: dict[str, Any] = {
        "schema_version": "pipelink-megakernel-parity-results-v2",
        "protocol": PROTOCOL, "suite": SUITE, "generated_at": utc_now(),
        "hostname": socket.gethostname(), "execution_location": "local",
        "gpu": GPU, "clock_mhz": target_mhz or None, "clock_locked": bool(target_mhz),
        "executables": executables,
        "ratio_floor": RATIO_FLOOR, "repetitions": 10,
        "cases": [], "execution_error": None, "p2_receipts": p2_receipts,
        "paired_sanitizers": paired_sanitizers,
    }
    execution_failed = False
    try:
        for case in CASES:
            archive = OUTPUT_ROOT / "cases" / case["case_id"]
            snapshots = archive / "snapshots"
            images = physical_images(case)
            execution = run_native_pair(
                pair_command(case, snapshots, "pair"),
                env=dict(os.environ), gpus=[GPU], clock_mhz=target_mhz,
                out_dir=archive, timeout=600,
            )
            numerical = None
            if execution.get("command_returncode") == 0 and execution.get("payload"):
                try:
                    numerical = check_native_payload(case, execution["payload"], snapshots, "pair")
                except (ValueError, KeyError, OSError, EvidenceError) as exc:
                    execution["numerical_error"] = str(exc)
            if images != physical_images(case) or executables != executable_identity():
                raise EvidenceError("images, manifest, provider or runner changed during the case")
            case_result = collect_case(case, execution, numerical)
            case_result.update(images=images, fixture=case_config(case),
                               sanitizers=sanitizer_counts[case["case_id"]])
            payload["cases"].append(case_result)
            execution_failed |= case_result["status"] != "PASS"
            write_results(payload)
    except Exception as exc:
        execution_failed = True
        payload["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
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
