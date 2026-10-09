"""Same-source controls and evidence for public examples.

CUDA tools are invoked only by the build, sanitizer, and SASS helpers. Timing
accepts host-less sessions by behavior, without importing the CUDA launcher.
"""
from __future__ import annotations

import json
from math import ceil
import os
from pathlib import Path
from statistics import median
import subprocess

import numpy as np

from evaluator.gates.comparison import compare_streams
from evaluator.gates.normalization import canonicalize, sections


_ORDERS = ("ABBA", "BAAB", "BAAB", "ABBA", "BAAB", "ABBA", "ABBA", "BAAB")


def _run(argv, *, env, log, check):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(list(map(str, argv)), env=env, capture_output=True)
    log.write_bytes(completed.stdout + completed.stderr)
    if check and completed.returncode:
        raise RuntimeError(
            f"Command failed: {log}\n{completed.stderr.decode(errors='replace')}"
        )
    return completed


def onetu_control(link_dir, stages, *, profile, env, extra_includes=(), log_dir):
    """Build whole-program and backend-matched LTO controls for a link.

    Each stage supplies ``source``, ``includes``, and ``defines`` from its pack
    recipe. Paths are relative to the caller's working directory. Stage sources
    precede dispatch, bindings, and the remaining generated CUDA sources; include
    roots retain stage precedence, with ``extra_includes`` appended last.
    Conflicting stage definitions raise ValueError. Build failures raise
    RuntimeError with the log path. The LTO image is a diagnostic control, not a
    replacement for the scored whole-program one-TU image.
    """
    link_dir = Path(link_dir).resolve()
    log_dir = Path(log_dir)
    link = json.loads((link_dir / "link.json").read_text())
    generated = link_dir / "generated"
    ordered = [generated / "dispatch.cu", generated / "binding_stub.cu"]
    ordered += sorted(
        generated / filename
        for filename in link["generated_hashes"]
        if filename.endswith(".cu") and generated / filename not in ordered
    )
    includes = list(dict.fromkeys(str(p) for stage in stages for p in stage["includes"]))
    includes.extend(map(str, extra_includes))
    definitions = {}
    for stage in stages:
        for key, value in stage["defines"].items():
            if definitions.setdefault(key, value) != value:
                raise ValueError(f"One-TU define conflict: {key}")

    source = link_dir / "one-tu.cu"
    source.write_text(
        "".join(f'#include "{Path(stage["source"]).resolve()}"\n' for stage in stages)
        + "".join(f'#include "{path}"\n' for path in ordered)
    )
    nvcc = Path(env["PIPELINK_CUDA_HOME"]) / "bin/nvcc"
    regs = f"-maxrregcount={link['regs_cap']}"
    recipe_flags = [*[f"-I{p}" for p in includes], *[f"-D{k}={v}" for k, v in definitions.items()]]
    controls = {"one_tu": link_dir / "one-tu.cubin", "one_tu_lto": link_dir / "one-tu-lto.cubin"}
    _run(
        [nvcc, "-std=c++17", "-O3", "-DNDEBUG", f"-arch={profile}", regs,
         *recipe_flags, "--cubin", source, "-o", controls["one_tu"]],
        env=env, log=log_dir / "one-tu.log", check=True,
    )
    lto = profile.replace("sm_", "lto_")
    compute = profile.replace("sm_", "compute_")
    lto_object = link_dir / "one-tu-lto.o"
    _run(
        [nvcc, "-std=c++17", "-O3", "-DNDEBUG", f"-gencode=arch={compute},code={lto}",
         "-dc", "-rdc=true", regs, *recipe_flags, source, "-o", lto_object],
        env=env, log=log_dir / "one-tu-lto.log", check=True,
    )
    _run(
        [nvcc, "-dlink", "-dlto", f"-arch={profile}", regs, "-cubin",
         lto_object, "-o", controls["one_tu_lto"]],
        env=env, log=log_dir / "one-tu-lto-link.log", check=True,
    )
    return controls


def _sm_clocks(gpu):
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm", "--format=csv,noheader", "-i", str(gpu)],
        capture_output=True, text=True, check=True,
    )
    current, maximum = completed.stdout.strip().split(",")
    return {"sm": current.strip(), "max_sm": maximum.strip()}


def time_pair(linked, control, *, repetitions=20, gpu=None):
    """Calibrate and measure eight balanced blocks in both ratio directions.

    Sessions expose ``launch()`` and ``time(n) -> float``: total milliseconds
    for n back-to-back launches bracketed by one synchronized event pair. Each
    session first times eight warmup launches, discarded. Shared calibration
    starts at one launch and targets a fastest batch of at least 15.625 ms,
    within 24 rounds and a 2**20 repetition cap; failure raises RuntimeError.
    The scored repetition count is at least the caller's ``repetitions``.

    A is linked and B is one-TU. ``blocks[].milliseconds`` contains four batch
    totals. Each ratio divides the two corresponding totals by the other two;
    each final ratio is independently the median of its eight block ratios.
    Status is PASS only if every scored batch lasts at least 12.5 ms, otherwise
    BATCH_TOO_SHORT. Timing neither substitutes a control nor gates on a ratio.

    With ``gpu`` set, clocks are observed before warmup and after measurement.
    ``clock_locked`` is unknown (None): current/max clock observations and an
    environment target cannot attest a lock. The requested target is recorded
    verbatim as ``PIPELINK_LOCK_SM_MHZ``. This helper never changes clocks.
    """
    before = _sm_clocks(gpu) if gpu is not None else None
    linked.time(8)
    control.time(8)
    batch_repetitions = 1
    for _ in range(24):
        fastest = min(linked.time(batch_repetitions), control.time(batch_repetitions))
        if fastest >= 15.625:
            break
        next_repetitions = ceil(batch_repetitions * 15.625 / fastest)
        if next_repetitions > 1 << 20:
            raise RuntimeError("Calibration failed: repetition cap exceeded")
        batch_repetitions = max(batch_repetitions + 1, next_repetitions)
    else:
        raise RuntimeError("Calibration failed: 24 rounds exhausted")
    repetitions = max(batch_repetitions, repetitions)
    sessions = {"A": linked, "B": control}
    blocks = []
    duration_ok = True
    for order in _ORDERS:
        milliseconds = [sessions[label].time(repetitions) for label in order]
        duration_ok &= all(ms >= 12.5 for ms in milliseconds)
        a = sum(ms for label, ms in zip(order, milliseconds) if label == "A")
        b = sum(ms for label, ms in zip(order, milliseconds) if label == "B")
        blocks.append({
            "order": order,
            "milliseconds": milliseconds,
            "ratio_linked_over_onetu": a / b,
            "ratio_onetu_over_linked": b / a,
        })
    after = _sm_clocks(gpu) if gpu is not None else None
    return {
        "repetitions": repetitions,
        "blocks": blocks,
        "ratio_linked_over_onetu": median(block["ratio_linked_over_onetu"] for block in blocks),
        "ratio_onetu_over_linked": median(block["ratio_onetu_over_linked"] for block in blocks),
        "status": "PASS" if duration_ok else "BATCH_TOO_SHORT",
        "gpu": gpu,
        "clocks": {"before": before, "after": after},
        "PIPELINK_LOCK_SM_MHZ": os.environ.get("PIPELINK_LOCK_SM_MHZ") or None,
        "clock_locked": None,
        "clock_mutation": False,
    }


def compare(actual, expected, *, rtol, atol):
    """Report NumPy tolerance agreement and exact logical-array byte identity.

    Tolerances use ``abs(actual - expected) <= atol + rtol * abs(expected)``.
    NaNs are mismatches, including paired NaNs, and propagate to ``max_abs``.
    Bitwise comparison includes shape and dtype, distinguishes signed zero, and
    compares elements in C order regardless of storage layout. Empty arrays
    have zero maximum error.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    close = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    difference = np.abs(np.subtract(actual, expected, dtype=np.result_type(actual.dtype, expected.dtype, np.float64)))
    return {
        "allclose": bool(np.all(close)),
        "bitwise": bool(actual.shape == expected.shape and actual.dtype == expected.dtype
                        and actual.tobytes() == expected.tobytes()),
        "max_abs": float(np.max(difference, initial=0)),
        "mismatches": int(np.count_nonzero(~close)),
    }


def sanitize(argv, *, env, log_dir, tools=("racecheck", "synccheck")):
    """Run every requested sanitizer, including after a preceding tool fails.

    Call for each image being accepted, using a distinct log directory per
    image. Nonzero tool exits are returned as FAIL records, not skipped or
    promoted to exceptions, so callers can retain all acceptance evidence.
    """
    sanitizer = Path(env["PIPELINK_CUDA_HOME"]) / "bin/compute-sanitizer"
    log_dir = Path(log_dir)
    command = list(argv)
    records = []
    for tool in tools:
        log = log_dir / f"{tool}.log"
        completed = _run(
            [sanitizer, "--tool", tool, "--error-exitcode", "1", *command],
            env=env, log=log, check=False,
        )
        records.append({
            "tool": tool, "returncode": completed.returncode,
            "status": "PASS" if completed.returncode == 0 else "FAIL", "log": str(log),
        })
    return records


def sass_diagnostic(linked_cubin, control_cubin, *, env, out_dir, backend_matched=None):
    """Save normalized SASS comparisons without turning deltas into a gate.

    When instruction counts differ, an available backend-matched one-TU image
    is compared too, attributing backend construction versus seam differences.
    Extraction errors still raise: missing evidence is not a clean diagnostic.
    """
    out_dir = Path(out_dir)
    cuobjdump = Path(env["PIPELINK_CUDA_HOME"]) / "bin/cuobjdump"

    def extract(image, label):
        path = out_dir / f"{label}.sass"
        _run([cuobjdump, "-sass", image], env=env, log=path, check=True)
        streams = [(function, canonicalize(body)) for function, body in sections(path.read_text())]
        if not streams:
            raise RuntimeError(f"No SASS functions extracted: {path}")
        return streams

    linked = extract(linked_cubin, "linked")
    control = extract(control_cubin, "one_tu")
    comparison = compare_streams(control, linked)
    path = out_dir / "sass-comparison.json"
    path.write_text(json.dumps(comparison, indent=2) + "\n")
    result = {
        "diagnostic_only": True, "comparison": str(path),
        "checks": comparison["checks"],
        "instruction_counts": comparison["instruction_counts"],
        "memory_op_class_counts": comparison["memory_op_class_counts"],
    }
    if not comparison["checks"]["instruction_counts_equal"] and backend_matched is not None:
        matched = extract(backend_matched, "one_tu_lto")
        backend = compare_streams(matched, linked)
        (out_dir / "sass-comparison-backend-matched.json").write_text(json.dumps(backend, indent=2) + "\n")
        result["backend_matched"] = {
            "checks": backend["checks"],
            "instruction_counts": backend["instruction_counts"],
            "memory_op_class_counts": backend["memory_op_class_counts"],
        }
    return result
