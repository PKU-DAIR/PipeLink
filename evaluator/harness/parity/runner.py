"""Family-neutral interleaved paired executor with smoke and escalation protocols."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Sequence

from .case_device import PASS as CASE_DEVICE_PASS, read_case_device_qualification
from .assign import query_device_identity
from .constants import (
    ATTEMPT_CAP,
    CLOCK_SAMPLE_HZ,
    GAP_RESULT_WORDING,
    GAP_ROW_CASE_IDS,
    GAP_ROW_CLASSIFIER,
    EXCLUDED_STEADY_PHASES,
    LEADING_DISCARD_SECONDS,
    RATIO_FLOOR,
    FAST_SOLO_DISCARD_SECONDS,
    FAST_SOLO_SECONDS,
    F2_ORIGINAL_CROSS_DISCARD_SECONDS,
    F2_ORIGINAL_CROSS_SLOT_SECONDS,
    F2_ORIGINAL_CROSS_STEADY_SECONDS,
    MAX_BATCH_QUANTUM_MS,
    REHEARSAL_ATTEMPT_CAP,
    REHEARSAL_DISCARD_SECONDS,
    NEGATIVE_LAUNCH_ATTEMPT_CAP,
    NEGATIVE_LAUNCH_BATCH_QUANTUM_MS,
    NEGATIVE_LAUNCH_DISCARD_SECONDS,
    NEGATIVE_LAUNCH_SLOT_SECONDS,
    NEGATIVE_LAUNCH_STEADY_SECONDS,
    NEGATIVE_LAUNCH_VALID_PAIRS,
    NEGATIVE_LAUNCH_WARMUPS,
    ORIGINAL_ATTEMPT_CAP,
    ORIGINAL_DISCARD_SECONDS,
    ORIGINAL_SLOT_SECONDS,
    ORIGINAL_STEADY_SECONDS,
    ORIGINAL_VALID_PAIRS,
    ORIGINAL_SOLO_DISCARD_SECONDS,
    ORIGINAL_SOLO_SECONDS,
    REHEARSAL_SLOT_SECONDS,
    REHEARSAL_STEADY_SECONDS,
    REHEARSAL_VALID_PAIRS,
    SCHEMA_VERSIONS,
    SLOT_SECONDS,
    SMOKE_ATTEMPT_CAP,
    SMOKE_VALID_PAIRS,
    STEADY_SECONDS,
    SWEEP_PROTOCOL_NAME,
    SWEEP_PROTOCOL_VARIANT,
    STRICT_PAIR_CLASSIFIER,
    VALID_PAIRS,
    WARMUPS_PER_IMAGE,
)
from .fast_protocol import decide as decide_fast, seal_leg, seal_trigger
from .gap_eligibility import (
    ELIGIBILITY_INPUT_SCHEMA,
    THROUGHPUT_SCHEMA,
    GapSoloError,
    evaluate_eligibility,
    freeze_policy,
    make_solo,
    query_gap_policy,
    score as score_gap,
)
from .telemetry import (
    CLOCK_REASON_FIELD,
    ClockSampler,
    GapPowerTelemetryError,
    SamplerFactory,
    evaluate_clock_predicates,
    evaluate_gap_row_predicates,
    query_clock_support,
)
from .reader import chronological_primary_pairs, pair_is_primary_valid
from .util import canonical_json_bytes, require_sha256, sha256_bytes, sha256_file, utc_now, write_json_atomic


class PairCapError(RuntimeError):
    """The case reached the fixed attempt cap before its pair-valid target."""


class SlotExecutionError(RuntimeError):
    """A slot plugin failed or violated the closed subprocess protocol."""


@dataclass(frozen=True)
class ImageSpec:
    image_id: str
    command: tuple[str, ...]
    image_path: Path
    image_sha256: str
    cwd: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImageSpec":
        command = value.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("image command must be a non-empty JSON string array")
        image_path = Path(value.get("image_path", "")).resolve()
        if not image_path.is_file():
            raise ValueError(f"image path is missing: {image_path}")
        digest = require_sha256(value.get("image_sha256"), "image_sha256")
        actual = sha256_file(image_path)
        if actual != digest:
            raise ValueError(f"image hash mismatch for {image_path}: expected {digest}, got {actual}")
        image_id = value.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image_id must be a non-empty string")
        cwd_value = value.get("cwd")
        cwd = Path(cwd_value).resolve() if cwd_value is not None else None
        if cwd is not None and not cwd.is_dir():
            raise ValueError(f"image cwd is not a directory: {cwd}")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("image metadata must be an object")
        return cls(image_id, tuple(command), image_path, digest, cwd, metadata)

    def archive_record(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "command": list(self.command),
            "image_path": str(self.image_path),
            "image_sha256": self.image_sha256,
            "cwd": str(self.cwd) if self.cwd else None,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Protocol:
    name: str
    archive: Path
    case_config: dict[str, Any]
    valid_pairs: int
    attempt_cap: int
    slot_seconds: float
    discard_seconds: float
    steady_seconds: float
    warmups_per_image: int = WARMUPS_PER_IMAGE
    ratio_floor: float = RATIO_FLOOR
    _allow_nonstandard: bool = False
    gap_row_allowlist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.archive, Path):
            raise TypeError("protocol.archive must be a pathlib.Path")
        if not isinstance(self.case_config, dict):
            raise TypeError("protocol.case_config must be an object")
        is_sweep = self.name == SWEEP_PROTOCOL_NAME
        if is_sweep:
            if (
                isinstance(self.warmups_per_image, bool)
                or not isinstance(self.warmups_per_image, int)
                or self.warmups_per_image < 0
            ):
                raise ValueError("sweep warmups must be an integer >= 0")
            if (
                isinstance(self.slot_seconds, bool)
                or not isinstance(self.slot_seconds, (int, float))
                or not math.isfinite(float(self.slot_seconds))
                or self.slot_seconds <= 0
            ):
                raise ValueError("sweep slot seconds must be finite and > 0")
            if (
                isinstance(self.valid_pairs, bool)
                or not isinstance(self.valid_pairs, int)
                or self.valid_pairs < 1
            ):
                raise ValueError("sweep valid pairs must be an integer >= 1")
            if (
                isinstance(self.attempt_cap, bool)
                or not isinstance(self.attempt_cap, int)
                or self.attempt_cap < self.valid_pairs
            ):
                raise ValueError(
                    "sweep attempt cap must be an integer >= valid pairs"
                )
            expected_discard = self.slot_seconds / 3.0
            expected_steady = self.slot_seconds - expected_discard
            if (
                self.discard_seconds != expected_discard
                or self.steady_seconds != expected_steady
            ):
                raise ValueError(
                    "sweep timing requires discard=slot/3 and steady=slot-slot/3"
                )
            if self.ratio_floor != RATIO_FLOOR:
                raise ValueError("sweep ratio floor must remain frozen")
        if self.valid_pairs <= 0 or self.attempt_cap < self.valid_pairs:
            raise ValueError("pair target/cap are invalid")
        if self.slot_seconds <= 0 or self.discard_seconds < 0 or self.steady_seconds <= 0:
            raise ValueError("slot timing must be positive")
        if not math.isclose(self.discard_seconds + self.steady_seconds, self.slot_seconds):
            raise ValueError("discard + steady duration must equal slot duration")
        official = {
            "definitive": (VALID_PAIRS, ATTEMPT_CAP, SLOT_SECONDS, LEADING_DISCARD_SECONDS, STEADY_SECONDS),
            "smoke": (SMOKE_VALID_PAIRS, SMOKE_ATTEMPT_CAP, SLOT_SECONDS, LEADING_DISCARD_SECONDS, STEADY_SECONDS),
            "rehearsal": (REHEARSAL_VALID_PAIRS, REHEARSAL_ATTEMPT_CAP, REHEARSAL_SLOT_SECONDS, REHEARSAL_DISCARD_SECONDS, REHEARSAL_STEADY_SECONDS),
            "original-duration": (ORIGINAL_VALID_PAIRS, ORIGINAL_ATTEMPT_CAP, ORIGINAL_SLOT_SECONDS, ORIGINAL_DISCARD_SECONDS, ORIGINAL_STEADY_SECONDS),
            "f2-fast-cross-source": (VALID_PAIRS, ATTEMPT_CAP, SLOT_SECONDS, LEADING_DISCARD_SECONDS, STEADY_SECONDS),
            "f2-original-cross-source": (ORIGINAL_VALID_PAIRS, ORIGINAL_ATTEMPT_CAP, F2_ORIGINAL_CROSS_SLOT_SECONDS, F2_ORIGINAL_CROSS_DISCARD_SECONDS, F2_ORIGINAL_CROSS_STEADY_SECONDS),
            "negative-launch-scale-definitive": (NEGATIVE_LAUNCH_VALID_PAIRS, NEGATIVE_LAUNCH_ATTEMPT_CAP, NEGATIVE_LAUNCH_SLOT_SECONDS, NEGATIVE_LAUNCH_DISCARD_SECONDS, NEGATIVE_LAUNCH_STEADY_SECONDS),
            "negative-launch-scale-rehearsal": (3, 9, NEGATIVE_LAUNCH_SLOT_SECONDS, NEGATIVE_LAUNCH_DISCARD_SECONDS, NEGATIVE_LAUNCH_STEADY_SECONDS),
        }
        values = (self.valid_pairs, self.attempt_cap, self.slot_seconds, self.discard_seconds, self.steady_seconds)
        if self.name in official and values != official[self.name]:
            raise ValueError(f"{self.name} protocol drifts from M3 plan")
        if self.name not in official and not is_sweep and not self._allow_nonstandard:
            raise ValueError("unsupported public protocol")
        if (
            self.name.startswith("negative-launch-scale-")
            and self.warmups_per_image != NEGATIVE_LAUNCH_WARMUPS
        ):
            raise ValueError("negative launch-scale protocol requires 1000 warmups")
        classes = self.case_config.get("stream_classes")
        if not isinstance(classes, list) or not classes:
            raise ValueError("case_config.stream_classes must be non-empty")
        require_sha256(self.case_config.get("stream_map_sha256"), "case_config.stream_map_sha256")
        if not isinstance(self.case_config.get("inputs"), dict) or not isinstance(self.case_config.get("arguments"), dict):
            raise ValueError("case_config inputs/arguments must be objects")
        allowlist = tuple(self.gap_row_allowlist)
        if len(set(allowlist)) != len(allowlist) or set(allowlist) - GAP_ROW_CASE_IDS:
            raise ValueError("gap-row allowlist differs from closed case IDs")
        object.__setattr__(self, "gap_row_allowlist", allowlist)

    @classmethod
    def definitive(cls, archive: Path, case_config: dict[str, Any], *, gap_row_allowlist: Sequence[str] = ()) -> "Protocol":
        return cls("definitive", archive, case_config, VALID_PAIRS, ATTEMPT_CAP, SLOT_SECONDS, LEADING_DISCARD_SECONDS, STEADY_SECONDS, gap_row_allowlist=tuple(gap_row_allowlist))

    @classmethod
    def smoke(cls, archive: Path, case_config: dict[str, Any], *, gap_row_allowlist: Sequence[str] = ()) -> "Protocol":
        return cls("smoke", archive, case_config, SMOKE_VALID_PAIRS, SMOKE_ATTEMPT_CAP, SLOT_SECONDS, LEADING_DISCARD_SECONDS, STEADY_SECONDS, gap_row_allowlist=tuple(gap_row_allowlist))

    @classmethod
    def rehearsal(cls, archive: Path, case_config: dict[str, Any], *, gap_row_allowlist: Sequence[str] = ()) -> "Protocol":
        return cls("rehearsal", archive, case_config, REHEARSAL_VALID_PAIRS, REHEARSAL_ATTEMPT_CAP, REHEARSAL_SLOT_SECONDS, REHEARSAL_DISCARD_SECONDS, REHEARSAL_STEADY_SECONDS, gap_row_allowlist=tuple(gap_row_allowlist))

    @classmethod
    def sweep(
        cls,
        archive: Path,
        case_config: dict[str, Any],
        *,
        warmups: int,
        slot_seconds: float,
        valid_pairs: int,
        attempt_cap: int,
        gap_row_allowlist: Sequence[str] = (),
    ) -> "Protocol":
        if (
            isinstance(slot_seconds, bool)
            or not isinstance(slot_seconds, (int, float))
            or not math.isfinite(float(slot_seconds))
            or slot_seconds <= 0
        ):
            raise ValueError("sweep slot seconds must be finite and > 0")
        slot = float(slot_seconds)
        discard = slot / 3.0
        return cls(
            SWEEP_PROTOCOL_NAME,
            archive,
            case_config,
            valid_pairs,
            attempt_cap,
            slot,
            discard,
            slot - discard,
            warmups,
            gap_row_allowlist=tuple(gap_row_allowlist),
        )

    @classmethod
    def original_duration(cls, archive: Path, case_config: dict[str, Any], *, gap_row_allowlist: Sequence[str] = ()) -> "Protocol":
        return cls("original-duration", archive, case_config, ORIGINAL_VALID_PAIRS, ORIGINAL_ATTEMPT_CAP, ORIGINAL_SLOT_SECONDS, ORIGINAL_DISCARD_SECONDS, ORIGINAL_STEADY_SECONDS, gap_row_allowlist=tuple(gap_row_allowlist))
    @classmethod
    def f2_cross_source(
        cls, archive: Path, case_config: dict[str, Any], *, mode: str
    ) -> "Protocol":
        if mode == "fast":
            return cls("f2-fast-cross-source", archive, case_config, VALID_PAIRS, ATTEMPT_CAP, SLOT_SECONDS, LEADING_DISCARD_SECONDS, STEADY_SECONDS)
        if mode == "original":
            return cls("f2-original-cross-source", archive, case_config, ORIGINAL_VALID_PAIRS, ORIGINAL_ATTEMPT_CAP, F2_ORIGINAL_CROSS_SLOT_SECONDS, F2_ORIGINAL_CROSS_DISCARD_SECONDS, F2_ORIGINAL_CROSS_STEADY_SECONDS)
        raise ValueError("F2 cross-source mode must be fast or original")
    @classmethod
    def negative_launch_scale(
        cls, archive: Path, case_config: dict[str, Any], *, mode: str
    ) -> "Protocol":
        if mode == "definitive":
            return cls(
                "negative-launch-scale-definitive", archive, case_config,
                NEGATIVE_LAUNCH_VALID_PAIRS, NEGATIVE_LAUNCH_ATTEMPT_CAP,
                NEGATIVE_LAUNCH_SLOT_SECONDS, NEGATIVE_LAUNCH_DISCARD_SECONDS,
                NEGATIVE_LAUNCH_STEADY_SECONDS, NEGATIVE_LAUNCH_WARMUPS,
            )
        if mode == "rehearsal":
            return cls(
                "negative-launch-scale-rehearsal", archive, case_config,
                3, 9, NEGATIVE_LAUNCH_SLOT_SECONDS,
                NEGATIVE_LAUNCH_DISCARD_SECONDS,
                NEGATIVE_LAUNCH_STEADY_SECONDS, NEGATIVE_LAUNCH_WARMUPS,
            )
        raise ValueError("negative launch-scale mode must be rehearsal or definitive")



    @classmethod
    def _self_test(cls, archive: Path, case_config: dict[str, Any], *, gap_row_allowlist: Sequence[str] = ()) -> "Protocol":
        return cls("selftest", archive, case_config, 2, 3, 0.08, 0.02, 0.06, 5, RATIO_FLOOR, True, tuple(gap_row_allowlist))

    def classifier_for_case(self, case_id: str) -> str:
        if not self.gap_row_allowlist:
            return STRICT_PAIR_CLASSIFIER
        if case_id not in self.gap_row_allowlist:
            raise ValueError(f"case {case_id!r} is not in caller gap allowlist")
        return GAP_ROW_CLASSIFIER

    def protocol_variant(self) -> str:
        solo_variants = {
            "f2-fast-cross-solo": "f2-fast-cross-solo-v1",
            "f2-original-cross-solo": "f2-original-cross-solo-v1",
            "fast-gap-solo": "gap-fast-solo-v1",
            "original-gap-solo": "gap-original-solo-v1",
            "selftest-gap-solo": "selftest-gap-solo-v1",
        }
        if self.name in solo_variants:
            return solo_variants[self.name]
        if self.name == SWEEP_PROTOCOL_NAME:
            return SWEEP_PROTOCOL_VARIANT
        if self.name == "f2-fast-cross-source":
            return "f2-fast-cross-source-v1"
        if self.name == "f2-original-cross-source":
            return "f2-original-cross-source-v1"
        if self.name.startswith("negative-launch-scale-"):
            return "negative-launch-scale-v1"
        if self.name == "original-duration":
            return "gap-original-v1" if self.gap_row_allowlist else "original-escalation-strict-v1"
        if self.name in {"definitive", "smoke"}:
            return "gap-fast-v1" if self.gap_row_allowlist else "fast-strict-v1"
        return "g7-route-rehearsal-v1" if self.name == "rehearsal" else "selftest-v1"

    def solo_protocol(self) -> "Protocol":
        if self._allow_nonstandard:
            return Protocol(
                "selftest-gap-solo", self.archive, self.case_config,
                self.valid_pairs, self.attempt_cap, self.slot_seconds,
                self.discard_seconds, self.steady_seconds,
                WARMUPS_PER_IMAGE, self.ratio_floor, True,
                self.gap_row_allowlist,
            )
        f2_original = self.name == "f2-original-cross-source"
        original = self.name == "original-duration"
        slot = ORIGINAL_SOLO_SECONDS if (original or f2_original) else FAST_SOLO_SECONDS
        discard = ORIGINAL_SOLO_DISCARD_SECONDS if (original or f2_original) else FAST_SOLO_DISCARD_SECONDS
        label = (
            "f2-original-cross-solo"
            if f2_original else
            "original-gap-solo"
            if original else
            "f2-fast-cross-solo"
            if self.name == "f2-fast-cross-source" else
            "fast-gap-solo"
        )
        return Protocol(label, self.archive, self.case_config, self.valid_pairs, self.attempt_cap, slot, discard, slot - discard, WARMUPS_PER_IMAGE, self.ratio_floor, True, self.gap_row_allowlist)

    def record(self) -> dict[str, Any]:
        record = {
            "name": self.name,
            "protocol_variant": self.protocol_variant(),
            "pair_valid_target": self.valid_pairs,
            "attempt_cap": self.attempt_cap,
            "warmups_per_image": self.warmups_per_image,
            "slot_seconds": self.slot_seconds,
            "leading_discard_seconds": self.discard_seconds,
            "steady_seconds": self.steady_seconds,
            "clock_sample_hz": CLOCK_SAMPLE_HZ,
            "configured_batch_quantum_ms": (
                NEGATIVE_LAUNCH_BATCH_QUANTUM_MS
                if self.name.startswith("negative-launch-scale-")
                else MAX_BATCH_QUANTUM_MS
            ),
            "ratio_floor": self.ratio_floor,
            "selection_classifier": GAP_ROW_CLASSIFIER if self.gap_row_allowlist else STRICT_PAIR_CLASSIFIER,
            "gap_row_allowlist": list(self.gap_row_allowlist),
        }
        if self.name == SWEEP_PROTOCOL_NAME:
            record["experiment_only"] = True
        return record


DeviceIdentityProvider = Callable[[int], dict[str, Any]]
ClockSupportProvider = Callable[[int], dict[str, Any]]
GapPolicyProvider = Callable[[int], dict[str, Any]]


def _request(
    action: str,
    case_id: str,
    role: str,
    image: ImageSpec,
    gpu: int,
    protocol: Protocol,
    *,
    attempt: int | None = None,
    order_position: int | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema_version": SCHEMA_VERSIONS["slot_request"],
        "action": action,
        "case_id": case_id,
        "image_role": role,
        "image": image.archive_record(),
        "physical_gpu": gpu,
        "cuda_visible_device": 0,
        "case_config": protocol.case_config,
        "protocol": protocol.record(),
    }
    if attempt is not None:
        request["attempt"] = attempt
    if order_position is not None:
        request["order_position"] = order_position
    return request


def _validate_correctness(result: dict[str, Any], action: str) -> None:
    correctness = result.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("all_true") is not True:
        raise SlotExecutionError(f"{action} plugin correctness recheck failed: {correctness!r}")
    if correctness.get("checked_before_warmup") is not True:
        raise SlotExecutionError("plugin must check Gate A/B before warmup/timing")


def _validate_result(
    result: Any,
    request: dict[str, Any],
    protocol: Protocol,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise SlotExecutionError("slot plugin output must be one JSON object")
    fixed = {
        "schema_version": SCHEMA_VERSIONS["slot_result"],
        "action": request["action"],
        "case_id": request["case_id"],
        "image_id": request["image"]["image_id"],
        "image_role": request["image_role"],
    }
    for field, expected in fixed.items():
        if result.get(field) != expected:
            raise SlotExecutionError(f"slot plugin field {field} is {result.get(field)!r}, expected {expected!r}")
    _validate_correctness(result, request["action"])
    if request["action"] == "warmup":
        if result.get("timing") is not False or result.get("warmups_completed") != protocol.warmups_per_image:
            raise SlotExecutionError(
                f"warmup plugin must perform exactly {protocol.warmups_per_image} untimed warmups "
                f"and no timed slot; got warmups_completed="
                f"{result.get('warmups_completed')!r} timing={result.get('timing')!r}"
            )
        return result
    if request["action"] != "slot":
        raise SlotExecutionError(f"unknown plugin action {request['action']!r}")
    if result.get("timing") is not True:
        raise SlotExecutionError("slot plugin must mark timing=true")
    positive_numbers = (
        ("steady_cuda_event_milliseconds",)
        if protocol.gap_row_allowlist
        else ("throughput", "steady_cuda_event_milliseconds")
    )
    if any(
        isinstance(result.get(field), bool)
        or not isinstance(result.get(field), (int, float))
        or not math.isfinite(float(result[field]))
        or result[field] <= 0
        for field in positive_numbers
    ):
        raise SlotExecutionError("slot plugin omitted a finite positive timing/throughput field")
    if isinstance(result.get("iterations"), bool) or not isinstance(result.get("iterations"), int) or result["iterations"] <= 0:
        raise SlotExecutionError("slot iterations must be a positive integer")
    timestamp_fields = ("slot_start_ns", "sampling_window_start_ns", "sampling_window_end_ns", "slot_end_ns")
    if any(isinstance(result.get(field), bool) or not isinstance(result.get(field), int) or result[field] <= 0 for field in timestamp_fields):
        raise SlotExecutionError("slot timestamps must be positive integer epoch nanoseconds")
    if not (
        result["slot_start_ns"] <= result["sampling_window_start_ns"]
        < result["sampling_window_end_ns"] <= result["slot_end_ns"]
    ):
        raise SlotExecutionError("slot plugin returned inconsistent window timestamps")
    if protocol.gap_row_allowlist:
        scoring_fields = (
            "cuda_scoring_window_start_ns",
            "cuda_scoring_window_end_ns",
        )
        if any(
            isinstance(result.get(field), bool)
            or not isinstance(result.get(field), int)
            or result[field] <= 0
            for field in scoring_fields
        ):
            raise SlotExecutionError(
                "gap slot requires positive integer CUDA scoring-window timestamps"
            )
        if (
            result["cuda_scoring_window_start_ns"]
            != result["sampling_window_start_ns"]
            or result["cuda_scoring_window_end_ns"]
            != result["sampling_window_end_ns"]
        ):
            raise SlotExecutionError(
                "gap eligibility and CUDA scoring must cover identical intervals"
            )
    for field, expected in (
        ("slot_duration_seconds", protocol.slot_seconds),
        ("leading_discard_seconds", protocol.discard_seconds),
        ("steady_duration_seconds", protocol.steady_seconds),
    ):
        value = result.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected, abs_tol=0.25)
        ):
            raise SlotExecutionError(f"slot plugin {field} differs from protocol: {value!r} versus {expected}")
    observed_wall = (result["slot_end_ns"] - result["slot_start_ns"]) / 1.0e9
    observed_window = (result["sampling_window_end_ns"] - result["sampling_window_start_ns"]) / 1.0e9
    if observed_wall + 0.25 < protocol.slot_seconds or observed_window + 0.25 < protocol.steady_seconds:
        raise SlotExecutionError("slot plugin did not sustain the full wall/steady durations")
    batch_quantum = result.get("configured_batch_quantum_ms")
    if (
        isinstance(batch_quantum, bool)
        or not isinstance(batch_quantum, (int, float))
        or not math.isfinite(float(batch_quantum))
        or float(batch_quantum) <= 0
    ):
        raise SlotExecutionError("configured batch quantum must be finite positive")
    batch_quantum_ms = float(batch_quantum)
    if batch_quantum_ms != float(
        protocol.record()["configured_batch_quantum_ms"]
    ):
        raise SlotExecutionError(
            "slot batch quantum differs from shared protocol request"
        )
    maximum_batch = result.get("maximum_observed_batch_ms")
    calibration = result.get("single_launch_calibration_ms")
    batch_iterations = result.get("inner_batch_launches")
    for field, value in (
        ("maximum_observed_batch_ms", maximum_batch),
        ("single_launch_calibration_ms", calibration),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise SlotExecutionError(f"{field} must be finite positive")
    if (
        isinstance(batch_iterations, bool)
        or not isinstance(batch_iterations, int)
        or batch_iterations <= 0
    ):
        raise SlotExecutionError("inner_batch_launches must be a positive integer")
    batch_diagnostics = (
        f"observed_batch_ms={float(maximum_batch):.6f}, "
        f"quantum_ms={batch_quantum_ms:.6f}, "
        f"single_launch_calibration_ms={float(calibration):.6f}, "
        f"resulting_batch_iterations={batch_iterations}"
    )
    if float(maximum_batch) > batch_quantum_ms:
        raise SlotExecutionError(
            "observed launch batch exceeds configured quantum "
            f"({batch_diagnostics})"
        )
    if result.get("last_batch_start_before_nominal_end") is not True:
        raise SlotExecutionError("slot engine started a batch at/after nominal slot end")
    declared_observed = {
        "declared_slot_seconds": protocol.slot_seconds,
        "declared_scored_seconds": protocol.steady_seconds,
    }
    for field, expected in declared_observed.items():
        if result.get(field) != expected:
            raise SlotExecutionError(f"{field} differs from protocol")
    for field in ("observed_slot_seconds", "observed_scored_seconds"):
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SlotExecutionError(f"{field} must be finite numeric")
    overshoot_bound = batch_quantum_ms / 1000.0 + 0.01
    slot_overshoot = result["observed_slot_seconds"] - protocol.slot_seconds
    if slot_overshoot > overshoot_bound:
        raise SlotExecutionError(
            "observed slot overshoot exceeds one batch quantum "
            f"(overshoot_ms={slot_overshoot * 1000.0:.6f}, {batch_diagnostics})"
        )
    scored_overshoot = result["observed_scored_seconds"] - protocol.steady_seconds
    if scored_overshoot > overshoot_bound:
        raise SlotExecutionError(
            "observed scored overshoot exceeds one batch quantum "
            f"(overshoot_ms={scored_overshoot * 1000.0:.6f}, {batch_diagnostics})"
        )
    if result.get("timing_method") != "cuda-events-repeated-batches":
        raise SlotExecutionError("steady timing must use repeated CUDA-event batches")
    if result.get("steady_timing_excludes") != list(EXCLUDED_STEADY_PHASES):
        raise SlotExecutionError("slot plugin timing-boundary declaration differs from the required exclusions")
    if not protocol.gap_row_allowlist:
        if not isinstance(result.get("throughput_unit"), str) or not result["throughput_unit"]:
            raise SlotExecutionError("slot plugin must name a throughput unit")
    return result


def _invoke_plugin(
    image: ImageSpec,
    request: dict[str, Any],
    protocol: Protocol,
    gpu: int,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    completed = subprocess.run(
        list(image.command),
        cwd=image.cwd,
        env=environment,
        input=json.dumps(request, sort_keys=True) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(60.0, protocol.slot_seconds + 120.0),
    )
    if completed.returncode:
        raise SlotExecutionError(
            f"slot plugin failed rc={completed.returncode}: {list(image.command)!r}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SlotExecutionError(f"slot plugin did not emit exactly one JSON object: {completed.stdout!r}") from exc
    result = _validate_result(result, request, protocol)
    result["plugin_stderr"] = completed.stderr
    return result


def _pair_records(
    case_id: str,
    attempts: list[dict[str, Any]],
    samples: list[Any],
    protocol: Protocol,
    device: dict[str, Any],
) -> tuple[float | None, list[dict[str, Any]]]:
    windows = [
        {
            role: (
                int(attempt["outputs"][role]["sampling_window_start_ns"]),
                int(attempt["outputs"][role]["sampling_window_end_ns"]),
            )
            for role in ("reference", "linked")
        }
        for attempt in attempts
    ]
    run_median, strict_evaluations = evaluate_clock_predicates(windows, samples)
    classifier = protocol.classifier_for_case(case_id)
    if classifier == GAP_ROW_CLASSIFIER:
        raise AssertionError(
            "gap rows must use the ex-ante solo/ratio-blind v4 path"
        )
    pairs: list[dict[str, Any]] = []
    ratio_blind = protocol.name in {"definitive", "smoke", "original-duration", SWEEP_PROTOCOL_NAME} or protocol.name.startswith("negative-launch-scale-")
    pair_schema = (
        SCHEMA_VERSIONS["original_pair"]
        if protocol.name == "original-duration"
        else SCHEMA_VERSIONS["negative_pair"]
        if protocol.name.startswith("negative-launch-scale-")
        else SCHEMA_VERSIONS["fast_pair"]
        if protocol.name in {"definitive", "smoke"}
        else SCHEMA_VERSIONS["pair"]
    )
    for attempt, strict_evaluation in zip(
        attempts, strict_evaluations, strict=True
    ):
        reference = attempt["outputs"]["reference"]
        linked = attempt["outputs"]["linked"]
        if not ratio_blind and reference["throughput_unit"] != linked["throughput_unit"]:
            raise SlotExecutionError("reference/linked throughput units differ")
        common = {
            "case_id": case_id,
            "attempt": attempt["attempt"],
            "order": attempt["order"],
            "device": device,
            "protocol": protocol.record(),
            "stream_map_sha256": protocol.case_config["stream_map_sha256"],
            "stream_classes": protocol.case_config["stream_classes"],
            "inputs": protocol.case_config["inputs"],
            "arguments": protocol.case_config["arguments"],
            "correctness": {
                role: attempt["outputs"][role]["correctness"]
                for role in ("reference", "linked")
            },
            "outputs": attempt["outputs"],
        }
        original_valid = strict_evaluation["original_valid"]
        amended_valid = strict_evaluation["amended_valid"]
        pair_valid = strict_evaluation["pair_valid"]
        if original_valid and amended_valid:
            raise AssertionError("pair validity branches must be mutually exclusive")
        if pair_valid != (original_valid or amended_valid):
            raise AssertionError("pair-valid disjunction invariant failed")
        ratio = (
            None
            if ratio_blind
            else linked["throughput"] / reference["throughput"]
            if pair_valid
            else None
        )
        validity_class = (
            "original-valid" if original_valid
            else "amended-valid" if amended_valid
            else "discarded"
        )
        pairs.append({
            "schema_version": pair_schema,
            **common,
            "telemetry": strict_evaluation,
            "original_valid": original_valid,
            "original_discard_reasons": strict_evaluation["original_discard_reasons"],
            "amended_valid": amended_valid,
            "amended_discard_reasons": strict_evaluation["amended_discard_reasons"],
            "pair_valid": pair_valid,
            "pair_discard_reasons": strict_evaluation["pair_discard_reasons"],
            "validity_class": validity_class,
            "ratio_linked_over_reference": ratio,
        })
    return run_median, pairs


def _write_pairs(archive: Path, pairs: list[dict[str, Any]]) -> None:
    for pair in pairs:
        write_json_atomic(archive / "pairs" / f"pair-{pair['attempt']:02d}.json", pair)


def _summary(
    case_id: str,
    protocol: Protocol,
    device: dict[str, Any],
    pairs: list[dict[str, Any]],
    run_median: float | None,
    *,
    cap_hit: bool,
) -> dict[str, Any]:
    selected = [pair for pair in pairs if pair["pair_valid"]][: protocol.valid_pairs]
    ratios = [pair["ratio_linked_over_reference"] for pair in selected]
    median_ratio = float(statistics.median(ratios)) if len(ratios) == protocol.valid_pairs else None
    if cap_hit:
        status = "EXECUTION_FAIL"
        verdict = "PAIR_CAP_HIT"
    elif median_ratio is not None and median_ratio >= protocol.ratio_floor:
        status = "PASS"
        verdict = "PASS"
    else:
        status = "MISS"
        verdict = "BELOW_RATIO_FLOOR"
    original_count = sum(pair["original_valid"] for pair in pairs)
    amended_count = sum(pair["amended_valid"] for pair in pairs)
    pair_count = sum(pair["pair_valid"] for pair in pairs)
    if pair_count != original_count + amended_count:
        raise AssertionError("count invariant pair_valid == original_valid + amended_valid failed")
    operating_regime = (
        "mixed-original-and-amended"
        if original_count and amended_count
        else "original-only"
        if original_count
        else "amended-only"
        if amended_count
        else "none"
    )
    absolute_placement: dict[str, Any] = {}
    if protocol.name in {"definitive", "smoke"} and pairs:
        slot = pairs[0]["outputs"]["linked"]
        if "tflops" in slot:
            absolute_placement = {
                "linked_tflops_median": (
                    float(statistics.median(pair["outputs"]["linked"]["tflops"] for pair in selected))
                    if selected else None
                ),
                "control_tflops_median": (
                    float(statistics.median(pair["outputs"]["reference"]["tflops"] for pair in selected))
                    if selected else None
                ),
                "flop_convention": slot["flop_convention"],
                "dtype": slot["dtype"],
                "arch": slot["arch"],
            }
    return {
        "schema_version": SCHEMA_VERSIONS["summary"],
        "utc_time": utc_now(),
        "case_id": case_id,
        "status": status,
        "verdict": verdict,
        "device": device,
        "protocol": protocol.record(),
        "stream_map_sha256": protocol.case_config["stream_map_sha256"],
        "stream_classes": protocol.case_config["stream_classes"],
        "attempted_pairs": len(pairs),
        "pair_valid_target": protocol.valid_pairs,
        "pair_valid_pairs_collected": len(selected),
        "original_valid_count": original_count,
        "amended_valid_count": amended_count,
        "pair_valid_count": pair_count,
        "discarded_pairs": sum(not pair["pair_valid"] for pair in pairs),
        "validity_count_invariant": {
            "pair_valid_count": pair_count,
            "original_plus_amended_count": original_count + amended_count,
            "equal": True,
        },
        "operating_regime": operating_regime,
        "mixed_validity_classes": bool(original_count and amended_count),
        "pair_cap_hit": cap_hit,
        "run_median_clock_mhz": run_median,
        **absolute_placement,
        "pair_valid_ratios_linked_over_reference": ratios,
        "median_ratio_linked_over_reference": median_ratio,
        "ratio_floor": protocol.ratio_floor,
        "equality_at_floor_passes": True,
        "statistic": "median of the first chronological pair-valid throughput_linked/throughput_reference ratios",
        "telemetry": {
            "field": CLOCK_REASON_FIELD,
            "sample_hz": CLOCK_SAMPLE_HZ,
            "common_clock_condition": "query success and coverage in both slots; every clock +/-3% run median; slot medians within 3%",
            "original_valid_branch": "common condition and every reason mask satisfies (mask & ~0x1) == 0",
            "amended_valid_branch": "common condition and every reason mask is exactly steady SW_POWER_CAP=0x4",
            "pair_valid_disjunction": "original_valid OR amended_valid; branches mutually exclusive",
        },
    }


def _numeric_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "max": ordered[-1],
    }




def _policy_tuple_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["policy"] == right["policy"]


def _run_gap_solo(
    *,
    case_id: str,
    role: str,
    image: ImageSpec,
    gpu: int,
    protocol: Protocol,
    policy: dict[str, Any],
    archive: Path,
    sampler_factory: SamplerFactory,
    series_id: str,
    duration_leg: str,
    session_id: str,
    solo_block_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sampler = sampler_factory(gpu, CLOCK_REASON_FIELD)
    sampler.start()
    try:
        sampler.wait_ready()
        request = _request(
            "slot", case_id, role, image, gpu, protocol,
            attempt=0, order_position=0,
        )
        request["solo_corroboration"] = True
        output = _invoke_plugin(image, request, protocol, gpu)
    finally:
        sampler.stop()
    rows = [
        sample.to_json()
        for sample in sampler.samples()
        if output["sampling_window_start_ns"]
        <= sample.time_ns
        <= output["sampling_window_end_ns"]
    ]
    solo = make_solo(
        case_id=case_id,
        role=role,
        image_id=image.image_id,
        image_sha256=image.image_sha256,
        policy_record=policy,
        series_id=series_id,
        duration_leg=duration_leg,
        session_id=session_id,
        solo_block_id=solo_block_id,
        slot_start_ns=output["slot_start_ns"],
        eligibility_window_start_ns=output["sampling_window_start_ns"],
        eligibility_window_end_ns=output["sampling_window_end_ns"],
        cuda_scoring_window_start_ns=output["cuda_scoring_window_start_ns"],
        cuda_scoring_window_end_ns=output["cuda_scoring_window_end_ns"],
        slot_end_ns=output["slot_end_ns"],
        samples=rows,
        leading_discard_ns=round(protocol.discard_seconds * 1_000_000_000),
    )
    write_json_atomic(archive / "solos" / f"{role}.json", solo)
    return solo, output


def _gap_half_input(
    *,
    role: str,
    image: ImageSpec,
    output: dict[str, Any],
    trace: list[dict[str, Any]],
    solo: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    bound = policy["policy"]
    return {
        "role": role,
        "series_id": solo["series_id"],
        "duration_leg": solo["duration_leg"],
        "session_id": solo["session_id"],
        "solo_block_id": solo["solo_block_id"],
        "image_id": image.image_id,
        "image_sha256": image.image_sha256,
        "gpu_uuid": bound["gpu_uuid"],
        "pci_bus_id": bound["pci_bus_id"],
        "policy_sha256": policy["policy_sha256"],
        "policy_record_sha256": policy["policy_record_sha256"],
        "solo_sha256": solo["solo_sha256"],
        "slot_start_ns": output["slot_start_ns"],
        "eligibility_window_start_ns": output["sampling_window_start_ns"],
        "eligibility_window_end_ns": output["sampling_window_end_ns"],
        "cuda_scoring_window_start_ns": output["cuda_scoring_window_start_ns"],
        "cuda_scoring_window_end_ns": output["cuda_scoring_window_end_ns"],
        "slot_end_ns": output["slot_end_ns"],
        "samples": trace,
    }


def _gap_v4_summary(
    *,
    case_id: str,
    protocol: Protocol,
    device: dict[str, Any],
    ledger: dict[str, Any],
    score_result: dict[str, Any] | None,
    solos: dict[str, dict[str, Any]],
    solo_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = ledger["attempts"]
    selected = set(ledger["selected_attempts"])
    ratios = (
        score_result["gap_pair_valid_ratios_linked_over_reference"]
        if score_result is not None else []
    )
    median_ratio = (
        score_result["median_ratio_linked_over_reference"]
        if score_result is not None else None
    )
    cap_hit = not ledger["target_met"]
    if cap_hit:
        status, verdict = "EXECUTION_FAIL", "GAP_PAIR_CAP_HIT"
    elif median_ratio is not None and median_ratio >= protocol.ratio_floor:
        status, verdict = "PASS", "PASS"
    else:
        status, verdict = "MISS", "BELOW_RATIO_FLOOR"
    strict_original = sum(row["strict_classification"]["original_valid"] for row in rows)
    strict_amended = sum(row["strict_classification"]["amended_valid"] for row in rows)
    strict_pair = sum(row["strict_classification"]["pair_valid"] for row in rows)
    superseded_same = sum(
        row["superseded_same_branch_classification"]["gap_pair_valid"]
        for row in rows
    )
    branch_counts: dict[str, int] = {}
    for row in rows:
        key = (
            row["halves"]["reference"]["preassigned_branch"]
            + "/"
            + row["halves"]["linked"]["preassigned_branch"]
        )
        branch_counts[key] = branch_counts.get(key, 0) + 1
    selected_rows = [row for row in rows if row["attempt"] in selected]
    clock_deltas = [
        row["clock_median_delta_linked_minus_reference_mhz"]
        for row in selected_rows
    ]
    identity_deltas = {
        role: [
            row["halves"][role]["identity_delta_percent"]
            for row in selected_rows
        ]
        for role in ("reference", "linked")
    }
    power_medians = {
        role: [
            row["halves"][role]["power_draw_w"]["median"]
            for row in selected_rows
        ]
        for role in ("reference", "linked")
    }
    return {
        "schema_version": SCHEMA_VERSIONS["gap_summary"],
        "utc_time": utc_now(),
        "case_id": case_id,
        "status": status,
        "verdict": verdict,
        "device": device,
        "protocol": protocol.record(),
        "protocol_variant": protocol.protocol_variant(),
        "selection_classifier": "gap-ex-ante-solo-v2",
        "eligibility_ledger_sha256": ledger["eligibility_ledger_sha256"],
        "score_sha256": score_result["score_sha256"] if score_result else None,
        "ratio_blind_eligibility": True,
        "throughput_read_after_eligibility_seal": True,
        "opening_solo_block_id": next(iter(ledger["solo_block_sha256"])),
        "opening_solo_sha256": next(
            iter(ledger["solo_block_sha256"].values())
        ),
        "mid_series_solo_refresh_count": 0,
        "identity_gating": ledger["identity_gating"],
        "policy_sha256": ledger["policy_sha256"],
        "policy_record_sha256": ledger["policy_record_sha256"],
        "attempted_pairs": len(rows),
        "gap_pair_valid_target": protocol.valid_pairs,
        "gap_pair_valid_pairs_collected": len(ledger["selected_attempts"]),
        "gap_pair_valid_count": sum(row["gap_pair_valid"] for row in rows),
        "discarded_pairs": sum(not row["gap_pair_valid"] for row in rows),
        "strict_original_valid_count": strict_original,
        "strict_amended_valid_count": strict_amended,
        "strict_pair_valid_count": strict_pair,
        "superseded_same_branch_valid_count": superseded_same,
        "cross_branch_valid_count": sum(
            row["gap_pair_valid"] and row["cross_branch"] for row in rows
        ),
        "branch_combination_counts": branch_counts,
        "pair_cap_hit": cap_hit,
        "gap_pair_valid_ratios_linked_over_reference": ratios,
        "median_ratio_linked_over_reference": median_ratio,
        "ratio_floor": protocol.ratio_floor,
        "cross_half_clock_delta_linked_minus_reference_mhz": {
            "values": clock_deltas,
            "distribution": _numeric_distribution(clock_deltas),
        },
        "identity_delta_percent": {
            role: {
                "values": values,
                "distribution": _numeric_distribution(values),
            }
            for role, values in identity_deltas.items()
        },
        "per_half_power_draw_median_w": {
            role: {
                "values": values,
                "distribution": _numeric_distribution(values),
            }
            for role, values in power_medians.items()
        },
        "timestamp_coverage_identical": all(
            half["timestamp_coverage_identical"]
            for row in rows for half in row["halves"].values()
        ),
        "solo_clock_median_mhz": {
            role: solo["median_clock_mhz"] for role, solo in solos.items()
        },
        "solo_throughput": {
            role: {
                "value": output["throughput"],
                "unit": output["throughput_unit"],
            }
            for role, output in solo_outputs.items()
        },

        "result_wording": GAP_RESULT_WORDING,
        "clock_normalized_efficiency_claimed": False,
    }
def _gap_identity_is_gating(protocol: Protocol) -> bool:
    """Return the adjudicated identity interpretation for a gap duration leg."""
    return protocol.name not in {"definitive", "smoke", SWEEP_PROTOCOL_NAME}



def _run_gap_case_v4(
    *,
    case_id: str,
    images: dict[str, ImageSpec],
    gpu: int,
    protocol: Protocol,
    archive: Path,
    device: dict[str, Any],
    policy_before: dict[str, Any],
    sampler_factory: SamplerFactory,
    gap_policy_provider: GapPolicyProvider,
) -> dict[str, Any]:
    solos: dict[str, dict[str, Any]] = {}
    solo_outputs: dict[str, dict[str, Any]] = {}
    solo_protocol = protocol.solo_protocol()
    series_id = (
        f"{case_id}:{images['reference'].image_id}-over-"
        f"{images['linked'].image_id}"
    )
    duration_leg = protocol.protocol_variant()
    session_id = sha256_bytes(
        f"{series_id}:{duration_leg}:{utc_now()}:{os.getpid()}".encode()
    )
    solo_block_id = sha256_bytes(
        f"{session_id}:attempt-0-adjacent-solos".encode()
    )
    for role in ("reference", "linked"):
        solo, output = _run_gap_solo(
            case_id=case_id,
            role=role,
            image=images[role],
            gpu=gpu,
            protocol=solo_protocol,
            policy=policy_before,
            archive=archive,
            sampler_factory=sampler_factory,
            series_id=series_id,
            duration_leg=duration_leg,
            session_id=session_id,
            solo_block_id=solo_block_id,
        )
        solos[role] = solo
        solo_outputs[role] = output
    solo_block = {
        "schema_version": "m3-gap-solo-block-v3",
        "case_id": case_id,
        "series_id": series_id,
        "duration_leg": duration_leg,
        "session_id": session_id,
        "solo_block_id": solo_block_id,
        "gpu_uuid": device["uuid"],
        "pci_bus_id": device["pci_bus_id"],
        "policy_sha256": policy_before["policy_sha256"],
        "role_order": ["reference", "linked"],
        "solo_sha256": {
            role: solo["solo_sha256"] for role, solo in solos.items()
        },
        "opening_block": True,
        "sealed_before_attempt_0": True,
        "mid_series_refresh_permitted": False,
        "unrelated_gpu_work_permitted": False,
    }
    solo_block["solo_block_sha256"] = sha256_bytes(
        canonical_json_bytes(solo_block)
    )
    identity_gating = _gap_identity_is_gating(protocol)
    solo_blocks = {solo_block_id: solos}
    post_solo_policy = freeze_policy(gap_policy_provider(gpu))
    write_json_atomic(archive / "policy-post-solos.json", post_solo_policy)
    if not _policy_tuple_equal(policy_before, post_solo_policy):
        raise GapSoloError("gap policy changed during solo corroboration")
    attempts: list[dict[str, Any]] = []
    opaque_outputs: list[dict[str, Any]] = []
    latest_policy = post_solo_policy
    for attempt_index in range(protocol.attempt_cap):
        order = (
            ("reference", "linked")
            if attempt_index % 2 == 0
            else ("linked", "reference")
        )
        sampler = sampler_factory(gpu, CLOCK_REASON_FIELD)
        sampler.start()
        outputs: dict[str, dict[str, Any]] = {}
        try:
            sampler.wait_ready()
            for position, role in enumerate(order):
                request = _request(
                    "slot", case_id, role, images[role], gpu, protocol,
                    attempt=attempt_index, order_position=position,
                )
                outputs[role] = _invoke_plugin(
                    images[role], request, protocol, gpu
                )
        finally:
            sampler.stop()
        windows = [{
            role: (
                outputs[role]["sampling_window_start_ns"],
                outputs[role]["sampling_window_end_ns"],
            )
            for role in ("reference", "linked")
        }]
        strict_median, strict_rows = evaluate_clock_predicates(
            windows, sampler.samples()
        )
        same_branch = evaluate_gap_row_predicates(
            windows, sampler.samples()
        )[0]
        halves = {
            role: _gap_half_input(
                role=role,
                image=images[role],
                output=outputs[role],
                trace=strict_rows[0]["traces"][role],
                solo=solos[role],
                policy=policy_before,
            )
            for role in ("reference", "linked")
        }
        attempts.append({
            "attempt": attempt_index,
            "order": list(order),
            "halves": halves,
            "solo_block_id": solo_block_id,
            "strict_classification": strict_rows[0],
            "superseded_same_branch_classification": same_branch,
        })
        opaque_outputs.append({
            "attempt": attempt_index,
            "order": list(order),
            "outputs": outputs,
            "strict_run_median_clock_mhz": strict_median,
        })
        latest_policy = freeze_policy(gap_policy_provider(gpu))
        if not _policy_tuple_equal(policy_before, latest_policy):
            write_json_atomic(archive / "execution-error.json", {
                "phase": "gap-policy-mismatch",
                "attempt": attempt_index,
                "route_failed_without_retry": True,
            })
            raise GapSoloError("gap policy changed during paired eligibility")
        provisional_input = {
            "schema_version": ELIGIBILITY_INPUT_SCHEMA,
            "case_id": case_id,
            "gpu_uuid": device["uuid"],
            "pci_bus_id": device["pci_bus_id"],
            "policy_before": policy_before,
            "policy_frozen": policy_before,
            "policy_after": latest_policy,
            "target_pairs": protocol.valid_pairs,
            "attempt_cap": protocol.attempt_cap,
            "leading_discard_ns": round(protocol.discard_seconds * 1_000_000_000),
            "identity_gating": identity_gating,
            "solo_blocks": solo_blocks,
            "attempts": attempts,
        }
        provisional = evaluate_eligibility(provisional_input)
        write_json_atomic(
            archive / "eligibility" / f"attempt-{attempt_index:02d}.json",
            provisional["attempts"][-1],
        )
        if provisional["target_met"]:
            break
    policy_after = freeze_policy(gap_policy_provider(gpu))
    write_json_atomic(archive / "policy-after.json", policy_after)
    eligibility_input = {
        "schema_version": ELIGIBILITY_INPUT_SCHEMA,
        "case_id": case_id,
        "gpu_uuid": device["uuid"],
        "pci_bus_id": device["pci_bus_id"],
        "policy_before": policy_before,
        "policy_frozen": policy_before,
        "policy_after": policy_after,
        "target_pairs": protocol.valid_pairs,
        "attempt_cap": protocol.attempt_cap,
        "leading_discard_ns": round(protocol.discard_seconds * 1_000_000_000),
        "identity_gating": identity_gating,
        "solo_blocks": solo_blocks,
        "attempts": attempts,
    }
    write_json_atomic(archive / "eligibility-input.json", eligibility_input)
    ledger = evaluate_eligibility(eligibility_input)
    write_json_atomic(archive / "eligibility-ledger.json", ledger)
    throughput_units = {
        raw["outputs"][role].get("throughput_unit")
        for raw in opaque_outputs
        for role in ("reference", "linked")
    }
    if (
        len(throughput_units) != 1
        or not all(isinstance(unit, str) and unit for unit in throughput_units)
    ):
        raise SlotExecutionError(
            "post-seal gap scorer requires one non-empty shared throughput unit"
        )
    throughput_rows = [
        {
            "attempt": raw["attempt"],
            "reference_throughput": raw["outputs"]["reference"]["throughput"],
            "linked_throughput": raw["outputs"]["linked"]["throughput"],
        }
        for raw in opaque_outputs
    ]
    score_result: dict[str, Any] | None = None
    if ledger["target_met"]:
        throughput_input = {
            "schema_version": THROUGHPUT_SCHEMA,
            "case_id": case_id,
            "eligibility_ledger_sha256": ledger["eligibility_ledger_sha256"],
            "attempts": throughput_rows,
        }
        write_json_atomic(archive / "throughput.json", throughput_input)
        score_result = score_gap(ledger, throughput_input)
        write_json_atomic(archive / "score.json", score_result)
    selected_ratio = {}
    if score_result is not None:
        selected_ratio = dict(zip(
            score_result["selected_attempts"],
            score_result["gap_pair_valid_ratios_linked_over_reference"],
            strict=True,
        ))
    pairs: list[dict[str, Any]] = []
    for eligibility_row, raw in zip(
        ledger["attempts"], opaque_outputs, strict=True
    ):
        attempt_index = eligibility_row["attempt"]
        pairs.append({
            "schema_version": SCHEMA_VERSIONS["gap_pair"],
            "case_id": case_id,
            "attempt": attempt_index,
            "order": eligibility_row["order"],
            "device": device,
            "protocol": protocol.record(),
            "protocol_variant": protocol.protocol_variant(),
            "selection_classifier": "gap-ex-ante-solo-v2",
            "stream_map_sha256": protocol.case_config["stream_map_sha256"],
            "stream_classes": protocol.case_config["stream_classes"],
            "inputs": protocol.case_config["inputs"],
            "arguments": protocol.case_config["arguments"],
            "correctness": {
                role: raw["outputs"][role]["correctness"]
                for role in ("reference", "linked")
            },
            "outputs": raw["outputs"],
            "strict_classification": eligibility_row["strict_classification"],
            "superseded_same_branch_classification": eligibility_row[
                "superseded_same_branch_classification"
            ],
            "gap_classification": {
                key: eligibility_row[key]
                for key in (
                    "gap_pair_valid", "reasons", "halves", "cross_branch",
                    "clock_median_delta_linked_minus_reference_mhz",
                )
            },
            "eligibility_ledger_sha256": ledger[
                "eligibility_ledger_sha256"
            ],
            "opening_solo_block_id": eligibility_row["solo_block_id"],
            "opening_solo_sha256": ledger["solo_block_sha256"][
                eligibility_row["solo_block_id"]
            ],
            "mid_series_solo_refresh_count": 0,
            "identity_gating": ledger["identity_gating"],
            "policy_sha256": ledger["policy_sha256"],
            "ratio_linked_over_reference": selected_ratio.get(attempt_index),
            "result_wording": GAP_RESULT_WORDING,
            "clock_normalized_efficiency_claimed": False,
        })
    _write_pairs(archive, pairs)
    for role, output in solo_outputs.items():
        write_json_atomic(archive / "solos" / f"{role}-slot-output.json", output)
    summary = _gap_v4_summary(
        case_id=case_id,
        protocol=protocol,
        device=device,
        ledger=ledger,
        score_result=score_result,
        solos=solos,
        solo_outputs=solo_outputs,
    )
    write_json_atomic(archive / "summary.json", summary)
    if not ledger["target_met"]:
        raise PairCapError(
            f"{case_id}: only {len(ledger['selected_attempts'])} gap-valid pairs "
            f"after {len(ledger['attempts'])} attempts (cap {protocol.attempt_cap})"
        )
    return {"pairs": pairs, "summary": summary}


def _seal_strict_eligibility(
    case_id: str,
    protocol: Protocol,
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    attempts = [
        {
            "attempt": pair["attempt"],
            "order": pair["order"],
            "original_valid": pair["original_valid"],
            "original_discard_reasons": pair["original_discard_reasons"],
            "amended_valid": pair["amended_valid"],
            "amended_discard_reasons": pair["amended_discard_reasons"],
            "pair_valid": pair["pair_valid"],
            "pair_discard_reasons": pair["pair_discard_reasons"],
            "telemetry": pair["telemetry"],
        }
        for pair in pairs
    ]
    selected = [
        row["attempt"] for row in attempts if row["pair_valid"]
    ][: protocol.valid_pairs]
    payload = {
        "schema_version": "m3-strict-eligibility-ledger-v1",
        "case_id": case_id,
        "protocol_variant": protocol.protocol_variant(),
        "target_pairs": protocol.valid_pairs,
        "attempt_cap": protocol.attempt_cap,
        "attempts": attempts,
        "selected_attempts": selected,
        "target_met": len(selected) == protocol.valid_pairs,
        "ratio_blind": True,
    }
    payload["eligibility_ledger_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def run_case(
    case_id: str,
    image_a: ImageSpec,
    image_b: ImageSpec,
    gpu: int,
    protocol: Protocol,
    *,
    sampler_factory: SamplerFactory = ClockSampler,
    device_identity_provider: DeviceIdentityProvider = query_device_identity,
    clock_support_provider: ClockSupportProvider = query_clock_support,
    gap_policy_provider: GapPolicyProvider = query_gap_policy,
) -> dict[str, Any]:
    """Run one complete comparison; ``image_a`` is reference, ``image_b`` linked.

    The first five arguments form the family API.
    Dependency-injection keywords exist only for CPU self-tests.
    """
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    classifier = protocol.classifier_for_case(case_id)
    if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu <= 0:
        raise ValueError("physical GPU index must be positive and nonzero")
    if image_a.image_id == image_b.image_id:
        raise ValueError("reference and linked image IDs must differ")
    archive = protocol.archive.resolve()
    if archive.exists() and (not archive.is_dir() or any(archive.iterdir())):
        raise ValueError(f"case archive must be a fresh empty directory: {archive}")
    archive.mkdir(parents=True, exist_ok=True)
    device = device_identity_provider(gpu)
    support = clock_support_provider(gpu)
    if (
        classifier == GAP_ROW_CLASSIFIER
        and not protocol._allow_nonstandard
    ):
        qualification_source = protocol.case_config.get(
            "case_device_qualification"
        )
        if isinstance(qualification_source, str):
            qualification_source = json.loads(
                Path(qualification_source).read_text(encoding="utf-8")
            )
        qualification = read_case_device_qualification(
            qualification_source
        )
        if (
            qualification["disposition"] != CASE_DEVICE_PASS
            or qualification["case_id"] != case_id
            or qualification["physical_index"] != gpu
            or qualification["gpu_uuid"] != device["uuid"]
            or qualification["pci_bus_id"] != device["pci_bus_id"]
        ):
            raise RuntimeError(
                "gap paired launcher requires matching sealed case/device PASS"
            )
    policy_before: dict[str, Any] | None = None
    if classifier == GAP_ROW_CLASSIFIER:
        policy_before = freeze_policy(gap_policy_provider(gpu))
        if (
            policy_before["policy"]["gpu_uuid"] != device["uuid"]
            or policy_before["policy"]["pci_bus_id"] != device["pci_bus_id"]
        ):
            raise GapSoloError("gap policy identity differs from assigned device")
        write_json_atomic(archive / "policy-before.json", policy_before)
    metadata = {
        "schema_version": "m3-parity-run-metadata-v1",
        "utc_time": utc_now(),
        "case_id": case_id,
        "device": device,
        "clock_support": support,
        "selection_classifier": classifier,
        "protocol": protocol.record(),
        "reference": image_a.archive_record(),
        "linked": image_b.archive_record(),
        "case_config": protocol.case_config,
        "gap_policy_record_sha256": (
            policy_before["policy_record_sha256"] if policy_before else None
        ),
    }
    write_json_atomic(archive / "run-metadata.json", metadata)

    images = {"reference": image_a, "linked": image_b}
    for role in ("reference", "linked"):
        request = _request("warmup", case_id, role, images[role], gpu, protocol)
        try:
            output = _invoke_plugin(images[role], request, protocol, gpu)
        except Exception as exc:
            write_json_atomic(archive / "execution-error.json", {"phase": f"warmup-{role}", "error": str(exc)})
            raise
        write_json_atomic(archive / f"warmup-{role}.json", {"request": request, "result": output})
    if classifier == GAP_ROW_CLASSIFIER:
        assert policy_before is not None
        return _run_gap_case_v4(
            case_id=case_id,
            images=images,
            gpu=gpu,
            protocol=protocol,
            archive=archive,
            device=device,
            policy_before=policy_before,
            sampler_factory=sampler_factory,
            gap_policy_provider=gap_policy_provider,
        )

    sampler = sampler_factory(gpu, CLOCK_REASON_FIELD)
    attempts: list[dict[str, Any]] = []
    final_pairs: list[dict[str, Any]] = []
    run_median: float | None = None
    ratio_blind_leg = protocol.name in {"definitive", "smoke", "original-duration", SWEEP_PROTOCOL_NAME} or protocol.name.startswith("negative-launch-scale-")
    sampler.start()
    try:
        sampler.wait_ready()
        for attempt in range(protocol.attempt_cap):
            order = ("reference", "linked") if attempt % 2 == 0 else ("linked", "reference")
            outputs: dict[str, dict[str, Any]] = {}
            for position, role in enumerate(order):
                request = _request(
                    "slot", case_id, role, images[role], gpu, protocol,
                    attempt=attempt, order_position=position,
                )
                try:
                    outputs[role] = _invoke_plugin(images[role], request, protocol, gpu)
                except Exception as exc:
                    write_json_atomic(archive / "execution-error.json", {
                        "phase": "slot", "attempt": attempt, "role": role,
                        "request": request, "error": str(exc),
                    })
                    raise
            attempts.append({"attempt": attempt, "order": list(order), "outputs": outputs})
            try:
                run_median, final_pairs = _pair_records(
                    case_id, attempts, sampler.samples(), protocol, device
                )
            except GapPowerTelemetryError as exc:
                write_json_atomic(archive / "execution-error.json", {
                    "phase": "required-gap-power-telemetry",
                    "attempt": attempt,
                    "error": str(exc),
                    "route_failed_without_retry": True,
                })
                raise
            if ratio_blind_leg:
                provisional = _seal_strict_eligibility(case_id, protocol, final_pairs)
                write_json_atomic(
                    archive / "eligibility" / f"attempt-{attempt:02d}.json",
                    provisional["attempts"][-1],
                )
            else:
                _write_pairs(archive, final_pairs)
            if len(chronological_primary_pairs(final_pairs, protocol.valid_pairs)) >= protocol.valid_pairs:
                break
    finally:
        sampler.stop()

    try:
        run_median, final_pairs = _pair_records(
            case_id, attempts, sampler.samples(), protocol, device
        )
    except GapPowerTelemetryError as exc:
        write_json_atomic(archive / "execution-error.json", {
            "phase": "required-gap-power-telemetry",
            "error": str(exc),
            "route_failed_without_retry": True,
        })
        raise
    eligibility_ledger = None
    if ratio_blind_leg:
        eligibility_ledger = _seal_strict_eligibility(
            case_id, protocol, final_pairs
        )
        write_json_atomic(archive / "eligibility-ledger.json", eligibility_ledger)
        selected_attempts = set(eligibility_ledger["selected_attempts"])
        units = {
            pair["outputs"][role].get("throughput_unit")
            for pair in final_pairs for role in ("reference", "linked")
        }
        if len(units) != 1 or not all(isinstance(unit, str) and unit for unit in units):
            raise SlotExecutionError("post-seal strict scorer requires one throughput unit")
        for pair in final_pairs:
            pair["ratio_linked_over_reference"] = (
                pair["outputs"]["linked"]["throughput"]
                / pair["outputs"]["reference"]["throughput"]
                if pair["attempt"] in selected_attempts
                else None
            )
    _write_pairs(archive, final_pairs)
    selected_count = len(chronological_primary_pairs(final_pairs, protocol.valid_pairs))
    cap_hit = selected_count < protocol.valid_pairs
    summary = _summary(
        case_id, protocol, device, final_pairs, run_median, cap_hit=cap_hit
    )
    summary["schema_version"] = (
        SCHEMA_VERSIONS["original_summary"]
        if protocol.name == "original-duration"
        else SCHEMA_VERSIONS["negative_summary"]
        if protocol.name.startswith("negative-launch-scale-")
        else SCHEMA_VERSIONS["fast_summary"]
        if protocol.name in {"definitive", "smoke"}
        else SCHEMA_VERSIONS["summary"]
    )
    summary["protocol_variant"] = protocol.protocol_variant()
    summary["eligibility_ledger_sha256"] = (
        eligibility_ledger["eligibility_ledger_sha256"]
        if eligibility_ledger else None
    )
    summary["ratio_blind_eligibility"] = bool(eligibility_ledger)
    write_json_atomic(archive / "summary.json", summary)
    if cap_hit:
        raise PairCapError(
            f"{case_id}: only {summary['pair_valid_pairs_collected']} primary-valid pairs after "
            f"{summary['attempted_pairs']} attempts (cap {protocol.attempt_cap})"
        )
    return {"pairs": final_pairs, "summary": summary}


def run_decision_case(
    case_id: str,
    image_a: ImageSpec,
    image_b: ImageSpec,
    gpu: int,
    archive: Path,
    case_config: dict[str, Any],
    *,
    gap_row_allowlist: Sequence[str] = (),
    direct_f2_audit_only: bool = False,
    sampler_factory: SamplerFactory = ClockSampler,
    device_identity_provider: DeviceIdentityProvider = query_device_identity,
    clock_support_provider: ClockSupportProvider = query_clock_support,
    gap_policy_provider: GapPolicyProvider = query_gap_policy,
) -> dict[str, Any]:
    archive = archive.resolve()
    if archive.exists() and (not archive.is_dir() or any(archive.iterdir())):
        raise ValueError("decision archive must be fresh and empty")
    archive.mkdir(parents=True, exist_ok=True)
    fast_protocol = Protocol.definitive(
        archive / "fast",
        case_config,
        gap_row_allowlist=gap_row_allowlist,
    )
    fast_run = run_case(
        case_id, image_a, image_b, gpu, fast_protocol,
        sampler_factory=sampler_factory,
        device_identity_provider=device_identity_provider,
        clock_support_provider=clock_support_provider,
        gap_policy_provider=gap_policy_provider,
    )
    fast_summary = fast_run["summary"]
    fast_leg = seal_leg(
        protocol_variant=fast_protocol.protocol_variant(),
        case_id=case_id,
        gpu=fast_summary["device"],
        eligibility_ledger_sha256=fast_summary["eligibility_ledger_sha256"],
        median_ratio_linked_over_reference=fast_summary[
            "median_ratio_linked_over_reference"
        ],
        result_path=str((archive / "fast" / "summary.json").resolve()),
        escalation_depth=0,
    )
    write_json_atomic(archive / "fast-result.json", fast_leg)
    trigger_evaluation = seal_trigger(fast_leg)
    write_json_atomic(
        archive / "trigger-evaluation.json", trigger_evaluation
    )
    original_leg = None
    if trigger_evaluation["triggered"]:
        original_protocol = Protocol.original_duration(
            archive / "original-duration",
            case_config,
            gap_row_allowlist=gap_row_allowlist,
        )
        original_run = run_case(
            case_id, image_a, image_b, gpu, original_protocol,
            sampler_factory=sampler_factory,
            device_identity_provider=device_identity_provider,
            clock_support_provider=clock_support_provider,
            gap_policy_provider=gap_policy_provider,
        )
        original_summary = original_run["summary"]
        original_leg = seal_leg(
            protocol_variant=original_protocol.protocol_variant(),
            case_id=case_id,
            gpu=original_summary["device"],
            eligibility_ledger_sha256=original_summary[
                "eligibility_ledger_sha256"
            ],
            median_ratio_linked_over_reference=original_summary[
                "median_ratio_linked_over_reference"
            ],
            result_path=str(
                (archive / "original-duration" / "summary.json").resolve()
            ),
            escalation_depth=1,
        )
        write_json_atomic(archive / "original-duration-result.json", original_leg)
    trigger, decision = decide_fast(
        fast_leg,
        original_leg,
        trigger_evaluation=trigger_evaluation,
        direct_f2_audit_only=direct_f2_audit_only,
    )
    write_json_atomic(archive / "trigger.json", trigger)
    write_json_atomic(archive / "decision.json", decision)
    return {
        "fast": fast_run,
        "original": original_leg,
        "trigger": trigger,
        "decision": decision,
    }
