"""10 Hz covering clock sampler and the original/amended §2.4 predicates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
import subprocess
import threading
import time
from typing import Any, Callable

from .constants import (
    CLOCK_SAMPLE_INTERVAL_SECONDS,
    GAP_CLOCK_WINDOW_FRACTION,
    CLOCK_WINDOW_FRACTION,
    MIN_COVERING_SAMPLES_PER_SLOT,
    ORIGINAL_ALLOWED_REASON_MASK,
    SW_POWER_CAP_REASON_MASK,
)

CLOCK_REASON_FIELD = "clocks_event_reasons.active"


@dataclass(frozen=True)
class ClockSample:
    time_ns: int
    query_ok: bool
    clock_mhz: int | None
    temperature_c: int | None
    reason_mask: int | None
    raw: str
    power_draw_w: float | None = None
    power_limit_w: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _parse_sample(line: str, timestamp_ns: int) -> ClockSample:
    fields = [field.strip() for field in line.strip().split(",")]
    try:
        if len(fields) != 5:
            raise ValueError("wrong field count")
        clock = int(fields[0])
        temperature = int(fields[1])
        reason = int(fields[2], 0)
        if clock <= 0:
            raise ValueError("non-positive SM clock")
    except (ValueError, TypeError):
        return ClockSample(timestamp_ns, False, None, None, None, line.rstrip("\n"))
    try:
        power_draw = float(fields[3])
        power_limit = float(fields[4])
    except (ValueError, TypeError):
        power_draw = None
        power_limit = None
    return ClockSample(
        timestamp_ns, True, clock, temperature, reason, line.rstrip("\n"),
        power_draw, power_limit,
    )


def query_clock_support(gpu: int) -> dict[str, Any]:
    """Fail closed unless the exact §2.4 active-reasons field is queryable."""
    if gpu == 0:
        raise ValueError("physical GPU 0 is forbidden by M3 G0/§2.5")
    command = [
        "nvidia-smi",
        "-i",
        str(gpu),
        f"--query-gpu=clocks.sm,temperature.gpu,{CLOCK_REASON_FIELD},power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sample = _parse_sample(completed.stdout, time.time_ns()) if completed.returncode == 0 else None
    if completed.returncode != 0 or sample is None or not sample.query_ok:
        raise RuntimeError(
            f"GPU {gpu} does not expose required {CLOCK_REASON_FIELD}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return {"field": CLOCK_REASON_FIELD, "probe": sample.to_json()}


class ClockSampler:
    """One long-lived ``nvidia-smi --loop-ms=100`` process per case."""

    def __init__(self, gpu: int, field: str = CLOCK_REASON_FIELD) -> None:
        if gpu == 0:
            raise ValueError("physical GPU 0 is forbidden by M3 G0/§2.5")
        if field != CLOCK_REASON_FIELD:
            raise ValueError(f"M3 requires exact telemetry field {CLOCK_REASON_FIELD}")
        self.gpu = gpu
        self.field = field
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[ClockSample] = []
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stderr = ""

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("clock sampler already started")
        command = [
            "nvidia-smi",
            "-i",
            str(self.gpu),
            f"--query-gpu=clocks.sm,temperature.gpu,{self.field},power.draw,power.limit",
            "--format=csv,noheader,nounits",
            f"--loop-ms={round(CLOCK_SAMPLE_INTERVAL_SECONDS * 1000)}",
        ]
        self._process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_loop, name=f"m3-clock-gpu-{self.gpu}", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            sample = _parse_sample(line, time.time_ns())
            with self._lock:
                self._samples.append(sample)
            self._ready.set()

    def wait_ready(self, timeout: float = 3.0) -> None:
        if not self._ready.wait(timeout):
            return_code = self._process.poll() if self._process is not None else None
            raise RuntimeError(f"10 Hz clock sampler produced no first sample (returncode={return_code})")
        first = self.samples()[0]
        if not first.query_ok:
            raise RuntimeError(f"10 Hz clock sampler first query failed: {first.raw}")

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if process.stderr is not None:
            self._stderr = process.stderr.read()
        self._process = None

    def samples(self) -> list[ClockSample]:
        with self._lock:
            return list(self._samples)

    @property
    def stderr(self) -> str:
        return self._stderr

    def __enter__(self) -> "ClockSampler":
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop()


SamplerFactory = Callable[[int, str], ClockSampler]


def _between(samples: list[ClockSample], start_ns: int, end_ns: int) -> list[ClockSample]:
    return [sample for sample in samples if start_ns <= sample.time_ns <= end_ns]


def _clock_window_valid(rows: list[ClockSample], run_median: float | None) -> bool:
    if run_median is None or run_median <= 0 or len(rows) < MIN_COVERING_SAMPLES_PER_SLOT:
        return False
    return all(
        row.query_ok
        and row.clock_mhz is not None
        and abs(row.clock_mhz - run_median) <= CLOCK_WINDOW_FRACTION * run_median
        for row in rows
    )


def evaluate_clock_predicates(
    windows: list[dict[str, tuple[int, int]]], samples: list[ClockSample]
) -> tuple[float | None, list[dict[str, Any]]]:
    """Evaluate every attempt against one case-wide run median.

    ``windows`` has one item per attempted pair and exactly ``reference`` and
    ``linked`` steady CUDA-event window bounds.  Only samples inside these
    windows contribute to the run median, matching the covering-sample wording
    of M3 §2.4.
    """
    covering_by_attempt: list[dict[str, list[ClockSample]]] = []
    all_covering: list[ClockSample] = []
    for pair_windows in windows:
        if set(pair_windows) != {"reference", "linked"}:
            raise ValueError("pair windows must contain reference and linked")
        by_slot: dict[str, list[ClockSample]] = {}
        for slot in ("reference", "linked"):
            start_ns, end_ns = pair_windows[slot]
            if end_ns <= start_ns:
                raise ValueError(f"{slot} has a non-positive steady window")
            rows = _between(samples, start_ns, end_ns)
            by_slot[slot] = rows
            all_covering.extend(rows)
        covering_by_attempt.append(by_slot)
    clocks = [sample.clock_mhz for sample in all_covering if sample.query_ok and sample.clock_mhz is not None]
    run_median = float(statistics.median(clocks)) if clocks else None
    evaluations: list[dict[str, Any]] = []
    for by_slot in covering_by_attempt:
        common_reasons: list[str] = []
        original_mask_reasons: list[str] = []
        amended_mask_reasons: list[str] = []
        slot_medians: dict[str, float | None] = {}
        for slot in ("reference", "linked"):
            rows = by_slot[slot]
            good_clocks = [row.clock_mhz for row in rows if row.query_ok and row.clock_mhz is not None]
            slot_medians[slot] = float(statistics.median(good_clocks)) if good_clocks else None
            if len(rows) < MIN_COVERING_SAMPLES_PER_SLOT:
                common_reasons.append(f"{slot}:uncovered")
            if any(not row.query_ok for row in rows):
                common_reasons.append(f"{slot}:query-failure")
            if not _clock_window_valid(rows, run_median):
                common_reasons.append(f"{slot}:clock-outside-3-percent")
            if any(
                row.reason_mask is None
                or (row.reason_mask & ~ORIGINAL_ALLOWED_REASON_MASK) != 0
                for row in rows
            ):
                original_mask_reasons.append(f"{slot}:reason-not-none-or-idle-only-0x0-or-0x1")
            if any(row.reason_mask != SW_POWER_CAP_REASON_MASK for row in rows):
                amended_mask_reasons.append(f"{slot}:reason-not-steady-sw-power-cap-0x4")
        reference_median = slot_medians["reference"]
        linked_median = slot_medians["linked"]
        if (
            run_median is None
            or reference_median is None
            or linked_median is None
            or abs(reference_median - linked_median) > CLOCK_WINDOW_FRACTION * run_median
        ):
            common_reasons.append("asymmetric-slot-medians-over-3-percent")
        common_reasons = list(dict.fromkeys(common_reasons))
        original_reasons = list(dict.fromkeys([*common_reasons, *original_mask_reasons]))
        amended_reasons = list(dict.fromkeys([*common_reasons, *amended_mask_reasons]))
        original_valid = not original_reasons
        amended_valid = not amended_reasons
        if original_valid and amended_valid:
            raise AssertionError("§2.4 original-valid and amended-valid branches must be mutually exclusive")
        pair_valid = original_valid or amended_valid
        evaluations.append({
            "run_median_clock_mhz": run_median,
            "slot_median_clock_mhz": slot_medians,
            "covering_sample_counts": {slot: len(rows) for slot, rows in by_slot.items()},
            "common_clock_condition": not common_reasons,
            "common_clock_discard_reasons": common_reasons,
            "original_valid": original_valid,
            "original_discard_reasons": original_reasons,
            "amended_valid": amended_valid,
            "amended_discard_reasons": amended_reasons,
            "pair_valid": pair_valid,
            "pair_discard_reasons": [] if pair_valid else [
                "original-valid-branch-failed",
                "amended-valid-branch-failed",
            ],
            "traces": {
                slot: [sample.to_json() for sample in rows]
                for slot, rows in by_slot.items()
            },
        })
    return run_median, evaluations


class GapPowerTelemetryError(RuntimeError):
    """Required report-only power telemetry is absent or inconsistent."""


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise GapPowerTelemetryError("gap-row power telemetry has no covering samples")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _gap_half(slot: str, rows: list[ClockSample]) -> dict[str, Any]:
    common_reasons: list[str] = []
    if len(rows) < MIN_COVERING_SAMPLES_PER_SLOT:
        common_reasons.append(f"{slot}:uncovered")
    if any(not row.query_ok for row in rows):
        common_reasons.append(f"{slot}:query-failure")
    clocks = [float(row.clock_mhz) for row in rows if row.query_ok and row.clock_mhz is not None]
    median_clock = float(statistics.median(clocks)) if clocks else None
    if (
        median_clock is None
        or median_clock <= 0
        or len(clocks) != len(rows)
        or any(abs(clock - median_clock) > GAP_CLOCK_WINDOW_FRACTION * median_clock for clock in clocks)
    ):
        common_reasons.append(f"{slot}:own-median-clock-outside-3-percent")
    original_mask_reasons: list[str] = []
    if any(
        row.reason_mask is None
        or (row.reason_mask & ~ORIGINAL_ALLOWED_REASON_MASK) != 0
        for row in rows
    ):
        original_mask_reasons.append(f"{slot}:reason-not-none-or-idle-only-0x0-or-0x1")
    amended_mask_reasons: list[str] = []
    if any(row.reason_mask != SW_POWER_CAP_REASON_MASK for row in rows):
        amended_mask_reasons.append(f"{slot}:reason-not-steady-sw-power-cap-0x4")
    original_reasons = list(dict.fromkeys([*common_reasons, *original_mask_reasons]))
    amended_reasons = list(dict.fromkeys([*common_reasons, *amended_mask_reasons]))
    original_valid = not original_reasons
    amended_valid = not amended_reasons
    if original_valid and amended_valid:
        raise AssertionError("gap original/amended half branches must be mutually exclusive")
    power_values = [float(row.power_draw_w) for row in rows]
    result = {
        "gap_original_half_valid": original_valid,
        "gap_original_half_discard_reasons": original_reasons,
        "gap_amended_half_valid": amended_valid,
        "gap_amended_half_discard_reasons": amended_reasons,
        "gap_validity_branch": "original" if original_valid else "amended" if amended_valid else "none",
        "covering_sample_count": len(rows),
        "clock_median_mhz": median_clock,
        "clock_min_mhz": min(clocks) if clocks else None,
        "clock_max_mhz": max(clocks) if clocks else None,
        "power_draw_w": {
            "sample_count": len(power_values),
            "p10": _percentile(power_values, 0.10),
            "median": _percentile(power_values, 0.50),
            "p90": _percentile(power_values, 0.90),
        },
    }
    return result


def evaluate_gap_row_predicates(
    windows: list[dict[str, tuple[int, int]]],
    samples: list[ClockSample],
) -> list[dict[str, Any]]:
    """Classify allowlisted gap rows using per-half own-median clock windows."""
    evaluations: list[dict[str, Any]] = []
    for pair_windows in windows:
        if set(pair_windows) != {"reference", "linked"}:
            raise ValueError("gap-row windows must contain reference and linked")
        by_slot = {
            slot: _between(samples, pair_windows[slot][0], pair_windows[slot][1])
            for slot in ("reference", "linked")
        }
        covering = [row for rows in by_slot.values() for row in rows]
        if any(
            row.power_draw_w is None
            or row.power_limit_w is None
            or not math.isfinite(float(row.power_draw_w))
            or not math.isfinite(float(row.power_limit_w))
            for row in covering
        ):
            raise GapPowerTelemetryError(
                "gap-row route requires power.draw and power.limit in every covering 10-Hz sample"
            )
        limits = {float(row.power_limit_w) for row in covering}
        if len(limits) != 1:
            raise GapPowerTelemetryError("gap-row configured power limit changed or is unavailable")
        halves = {slot: _gap_half(slot, rows) for slot, rows in by_slot.items()}
        reference = halves["reference"]
        linked = halves["linked"]
        original_pair = (
            reference["gap_original_half_valid"]
            and linked["gap_original_half_valid"]
        )
        amended_pair = (
            reference["gap_amended_half_valid"]
            and linked["gap_amended_half_valid"]
        )
        if original_pair and amended_pair:
            raise AssertionError("gap pair branches must be mutually exclusive")
        cross_branch = (
            reference["gap_original_half_valid"]
            and linked["gap_amended_half_valid"]
        ) or (
            reference["gap_amended_half_valid"]
            and linked["gap_original_half_valid"]
        )
        gap_pair_valid = original_pair or amended_pair
        reference_median = reference["clock_median_mhz"]
        linked_median = linked["clock_median_mhz"]
        if reference_median is None or linked_median is None:
            delta_mhz = None
            delta_percent = None
        else:
            delta_mhz = linked_median - reference_median
            delta_percent = 100.0 * (linked_median / reference_median - 1.0)
        power_delta = (
            linked["power_draw_w"]["median"]
            - reference["power_draw_w"]["median"]
        )
        evaluations.append({
            "halves": halves,
            "gap_original_pair_valid": original_pair,
            "gap_amended_pair_valid": amended_pair,
            "gap_pair_valid": gap_pair_valid,
            "gap_same_branch_failure": cross_branch,
            "gap_pair_discard_reasons": [] if gap_pair_valid else (
                ["cross-branch-halves"] if cross_branch else ["one-or-both-halves-invalid"]
            ),
            "configured_power_limit_w": next(iter(limits)) if limits else None,
            "clock_median_delta_linked_minus_reference_mhz": delta_mhz,
            "clock_median_delta_percent_linked_over_reference": delta_percent,
            "power_draw_median_delta_linked_minus_reference_w": power_delta,
        })
    return evaluations
