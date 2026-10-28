"""Same-source controls and evidence for public examples.

Only the build, sanitizer and SASS helpers invoke CUDA tools. Timing works with
any session object that has ``launch()`` and ``time(n)``; it does not import the
CUDA launcher.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from math import ceil, isfinite, sqrt
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

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


def onetu_control(link_dir, stages, *, profile, env, extra_includes=(), log_dir,
                  std=None, compile_flags=()):
    """Build whole-program and backend-matched LTO controls for a link.

    Each stage supplies ``source``, ``includes`` and ``defines`` from its pack
    recipe, and optionally ``std`` and ``compile_flags``. The combined TU uses the
    highest requested C++ standard unless ``std`` is given. Both controls share
    the flags, including the final LTO link. Paths are relative to the caller's
    working directory.

    Stage sources come first, then dispatch, bindings and the other generated
    CUDA sources. Include roots keep the same stage order, with
    ``extra_includes`` last. As in the host-ABI example, host-group dispatch puts
    its role-view types in their own namespace and keeps its extern-C kernel and
    role symbols.

    Conflicting stage definitions raise ValueError; build failures raise
    RuntimeError with the log path. The whole-program one-TU image is the timed
    control; the LTO image is for diagnosis only.
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
    standards = [stage.get("std", "c++17") for stage in stages]
    standard = std or max(standards, key=lambda value: int(value.removeprefix("c++")))
    flags = list(dict.fromkeys([*compile_flags,
                               *(flag for stage in stages for flag in stage.get("compile_flags", ())) ]))
    standard_flags = [flag for flag in flags if flag.startswith(("-std=", "--std="))]
    if standard_flags:
        standard = max([standard, *(flag.split("=", 1)[1] for flag in standard_flags)],
                       key=lambda value: int(value.removeprefix("c++")))
        flags = [flag for flag in flags if flag not in standard_flags]

    def generated_include(path):
        include = f'#include "{path}"\n'
        if link.get("host") and path.name == "dispatch.cu":
            return "namespace pipelink_control_dispatch {\n" + include + "}\n"
        return include

    source = link_dir / "one-tu.cu"
    source.write_text(
        "".join(f'#include "{Path(stage["source"]).resolve()}"\n' for stage in stages)
        + "".join(generated_include(path) for path in ordered)
    )
    nvcc = Path(env["PIPELINK_CUDA_HOME"]) / "bin/nvcc"
    regs = f"-maxrregcount={link['regs_cap']}"
    recipe_flags = [*[f"-I{p}" for p in includes], *[f"-D{k}={v}" for k, v in definitions.items()]]
    controls = {"one_tu": link_dir / "one-tu.cubin", "one_tu_lto": link_dir / "one-tu-lto.cubin"}
    _run(
        [nvcc, f"-std={standard}", "-O3", "-DNDEBUG", f"-arch={profile}", regs, *flags,
         *recipe_flags, "--cubin", source, "-o", controls["one_tu"]],
        env=env, log=log_dir / "one-tu.log", check=True,
    )
    lto = profile.replace("sm_", "lto_")
    compute = profile.replace("sm_", "compute_")
    lto_object = link_dir / "one-tu-lto.o"
    _run(
        [nvcc, f"-std={standard}", "-O3", "-DNDEBUG", f"-gencode=arch={compute},code={lto}", *flags,
         "-dc", "-rdc=true", regs, *recipe_flags, source, "-o", lto_object],
        env=env, log=log_dir / "one-tu-lto.log", check=True,
    )
    _run(
        [nvcc, "-dlink", "-dlto", f"-arch={profile}", regs, *flags, "-cubin",
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


@contextmanager
def clock_lease(gpu, *, env, log_dir):
    """Hold a ``tools/gpulockd.py`` lease on ``gpu`` at ``PIPELINK_LOCK_SM_MHZ``.

    When ``PIPELINK_LOCK_SM_MHZ`` is unset, empty or 0 this yields an unlocked
    record (``locked`` False, ``sm_mhz`` None, healthy and stopped) without
    calling gpulockd. Otherwise it yields the lease record. ``healthy`` is True only if ``status`` (live daemon,
    clock readback within tolerance, zero violations) passes after acquisition
    and again after the body. If start fails, this raises and leaves other
    leases alone. Otherwise the lease is always stopped, and a failed stop raises.
    """
    lock_mhz = str(env.get("PIPELINK_LOCK_SM_MHZ") or "").strip()
    if lock_mhz in {"", "0"}:
        # Unlocked mode: no daemon, no sudo, nothing to check or release.
        yield {"gpu": gpu, "locked": False, "sm_mhz": None, "healthy": True, "stopped": True,
               "operations": []}
        return
    tool = [sys.executable, str(Path(__file__).resolve().parents[3] / "tools/gpulockd.py")]
    log_dir = Path(log_dir)
    record = {"gpu": gpu, "sm_mhz": int(env["PIPELINK_LOCK_SM_MHZ"]), "healthy": False, "operations": []}

    def call(operation, *extra):
        completed = _run([*tool, operation, "--gpus", str(gpu), *extra], env=env,
                         log=log_dir / f"gpulockd-{operation}-{len(record['operations'])}.log", check=False)
        record["operations"].append({"operation": operation, "returncode": completed.returncode})
        return completed.returncode == 0

    if not call("start", "--sm-mhz", str(record["sm_mhz"])):
        raise RuntimeError(f"gpulockd start failed on GPU {gpu}; see {log_dir}")
    try:
        before = call("status")
        yield record
        record["healthy"] = before and call("status")
    finally:
        record["stopped"] = call("stop")
    if not record["stopped"]:
        raise RuntimeError(f"gpulockd stop failed on GPU {gpu}; see {log_dir}")


def _gpu_pids(gpu):
    completed = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    return sorted({int(line.strip()) for line in completed.stdout.splitlines() if line.strip()})


def _own_process(pid):
    while pid > 1:
        if pid == os.getpid():
            return True
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[1])
        except FileNotFoundError:
            return False
    return False


def _sampled_pair(linked, control, repetitions, gpu, reload=None):
    from evaluator.harness.parity.telemetry import ClockSampler

    if gpu is None:
        raise ValueError("sample_clocks requires a physical GPU index")
    allowed = {os.getpid()}
    state = os.environ.get("PIPELINK_LOCK_STATE_DIR")
    if state:
        marker = Path(state) / f"gpu{gpu}.json"
        if marker.exists():
            allowed.add(json.loads(marker.read_text())["pid"])
    pids_before = _gpu_pids(gpu)
    allowed.update(pid for pid in pids_before if _own_process(pid))
    with ClockSampler(gpu) as sampler:
        result = time_pair(linked, control, repetitions=repetitions, gpu=gpu, reload=reload)
    pids_after = _gpu_pids(gpu)
    allowed.update(pid for pid in pids_after if _own_process(pid))
    result["pids"] = {"before": pids_before, "after": pids_after, "allowed": sorted(allowed)}
    result["foreign_pids"] = sorted((set(pids_before) | set(pids_after)) - allowed)
    if result["foreign_pids"]:
        result["status"] = "FOREIGN_GPU_PROCESS"
    result["telemetry"] = {}
    for arm in ("A", "B"):
        windows = [window for block in result["blocks"]
                   for label, window in zip(block["order"], block["wall_intervals_ns"]) if label == arm]
        samples = [s for s in sampler.samples() if s.query_ok and
                   any(start <= s.time_ns <= end for start, end in windows)]
        clocks = [s.clock_mhz for s in samples]
        result["telemetry"][arm] = {
            "samples": len(samples), "sw_power_cap_samples": sum(bool(s.reason_mask & 4) for s in samples),
            "clock_min_mhz": min(clocks) if clocks else None,
            "clock_median_mhz": median(clocks) if clocks else None,
            "trace": [s.to_json() for s in samples],
        }
    return result


def time_pair(linked, control, *, repetitions=20, gpu=None, sample_clocks=False, reload=None):
    """Time linked (A) against one-TU (B) in eight ABBA/BAAB blocks; report B/A speedup.

    Sessions expose ``launch()`` and ``time(n) -> float``, the total milliseconds
    of n back-to-back launches between one synchronized event pair. Each session
    first runs eight discarded warmup launches.

    Without ``reload``, a shared calibration starts at one launch and grows the
    batch until the fastest one lasts at least 15.625 ms, within 24 rounds and a
    2**20 repetition cap (RuntimeError otherwise). The timed batch size is at
    least ``repetitions``. Each block holds four batch totals in
    ``blocks[].milliseconds``; its speedup is B's two totals over A's two. The
    result is the median of the eight block speedups. Status is PASS if every
    timed batch lasts at least 12.5 ms, else BATCH_TOO_SHORT; the speedup value
    never affects status.

    With a no-argument ``reload`` callback (protocol ``paired-reload-v2``),
    calibration is skipped and every batch uses exactly ``repetitions``. After
    four blocks the callback runs outside the timing events, then each session
    runs eight more discarded warmups; a callback failure propagates before any
    further measurement. The result is the geometric mean of the two four-block
    medians. All raw blocks are kept and every elapsed time must be positive and
    finite; there is no minimum batch duration or per-arm telemetry check. The
    caller sets the initial module order before each call; the callback reverses
    it while keeping the logical arms and their fixture resources.

    With ``gpu`` set, SM clocks are read before warmup and after measurement.
    This helper never changes clocks and cannot attest a lock, so
    ``clock_locked`` is None; callers holding ``clock_lease`` record its health.
    ``PIPELINK_LOCK_SM_MHZ`` is recorded as given. ``sample_clocks`` also records
    per-arm telemetry over the timed batches and compute PIDs before and after;
    a foreign PID invalidates the timing. SW_POWER_CAP samples are counted and
    reported but do not affect status.
    """
    if sample_clocks:
        return _sampled_pair(linked, control, repetitions, gpu, reload=reload)
    before = _sm_clocks(gpu) if gpu is not None else None
    linked.time(8)
    control.time(8)
    if reload is None:
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
    for index, order in enumerate(_ORDERS):
        if reload is not None and index == 4:
            reload()
            linked.time(8)
            control.time(8)
        milliseconds, intervals = [], []
        for label in order:
            start = time.time_ns()
            milliseconds.append(sessions[label].time(repetitions))
            intervals.append((start, time.time_ns()))
        if reload is None:
            duration_ok &= all(ms >= 12.5 for ms in milliseconds)
        elif any(not isfinite(ms) or ms <= 0 for ms in milliseconds):
            raise ValueError("Fixed-count timing requires positive finite elapsed time")
        a = sum(ms for label, ms in zip(order, milliseconds) if label == "A")
        b = sum(ms for label, ms in zip(order, milliseconds) if label == "B")
        blocks.append({
            "order": order,
            "milliseconds": milliseconds,
            "wall_intervals_ns": intervals,
            "ratio_onetu_over_linked": b / a,
        })
    after = _sm_clocks(gpu) if gpu is not None else None
    if reload is None:
        speedup = median(block["ratio_onetu_over_linked"] for block in blocks)
    else:
        speedup = sqrt(
            median(block["ratio_onetu_over_linked"] for block in blocks[:4])
            * median(block["ratio_onetu_over_linked"] for block in blocks[4:])
        )
    result = {
        "repetitions": repetitions,
        "blocks": blocks,
        "ratio_onetu_over_linked": speedup,
        "status": "PASS" if duration_ok else "BATCH_TOO_SHORT",
        "gpu": gpu,
        "clocks": {"before": before, "after": after},
        "PIPELINK_LOCK_SM_MHZ": os.environ.get("PIPELINK_LOCK_SM_MHZ") or None,
        "clock_locked": None,
        "clock_mutation": False,
    }
    if reload is not None:
        result.update(
            protocol="paired-reload-v2",
            statistic="geometric mean of per-order median speedups",
        )
    return result


def compare(actual, expected, *, rtol, atol):
    """Report tolerance agreement and exact byte identity of two arrays.

    The tolerance test is ``abs(actual - expected) <= atol + rtol * abs(expected)``.
    Every NaN is a mismatch, even when both sides are NaN, and propagates to
    ``max_abs``. The bitwise check also compares shape and dtype, distinguishes
    signed zero and reads elements in C order whatever the storage layout. Empty
    arrays have zero maximum error.
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
    """Run every requested sanitizer, even after an earlier one fails.

    Call once per image, with a separate log directory for each. A nonzero tool
    exit becomes a FAIL record, so the caller keeps the evidence from every tool.
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
    """Save normalized SASS comparisons for diagnosis; differences do not fail anything.

    If the normalized streams differ and a backend-matched one-TU image is given,
    it is compared too, to separate backend differences from seam differences.
    Extraction errors raise, since a missing stream cannot count as a match.
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
    same_stream = all(value for key, value in comparison["checks"].items()
                      if key != "no_local_traffic_or_calls")
    if not same_stream and backend_matched is not None:
        matched = extract(backend_matched, "one_tu_lto")
        backend = compare_streams(matched, linked)
        (out_dir / "sass-comparison-backend-matched.json").write_text(json.dumps(backend, indent=2) + "\n")
        result["backend_matched"] = {
            "checks": backend["checks"],
            "instruction_counts": backend["instruction_counts"],
            "memory_op_class_counts": backend["memory_op_class_counts"],
        }
    return result
