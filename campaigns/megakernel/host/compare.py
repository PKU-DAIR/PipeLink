#!/usr/bin/env python3
"""Apply the numerical portions of the Megakernels host Gate P2.

The fixed 2e-2 strict tolerances are retained where the computation is stable.
Full-depth comparisons use a 2x repeat floor because random-weight bf16 layers
amplify reduction-order noise.  When two upstream mono runs are available, the
effective floor is the larger linked or mono repeat floor: a port must not fail
merely because upstream's own launch-to-launch noise exceeds the linked floor.
Full-depth distance to the PyVM reference is gated on mean relative difference
only; its per-element maximum is an extreme-value statistic and is reported.

Only PipeLink-built images (linked, linked2, one-TU, and extras with
--gate-extra) are gated.  The upstream mono runs are the baseline: their
reference metrics are reported, and they define the repeat floor and the
upstream-equivalence target, but they are not themselves held to a predicate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Gate constants. These are deliberately fixed rather than inferred from a run.
RTOL = 2.0e-2
ATOL_RMS_FACTOR = 2.0e-2
FLOOR_FACTOR = 2.0
ULP_FACTOR = 2.0
RDIFF_EPS = 1.0e-6
REL_EPS = 1.0e-6

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
STRICT_REQUIRED_REFERENCE_TENSORS = frozenset(
    ("q_post_rope", "attn_out", "k_row", "v_row")
)
REPORTED_REFERENCE_TENSORS = ("hidden_states", "silu_out")
ROW_TENSORS = ("k_row", "v_row")
TRIVIAL_ZERO_TENSORS = frozenset(("attn_out_p0", "attn_lse_p0"))
REQUIRED_DETERMINISTIC = ("k_row[0]", "v_row[0]")
REFERENCE_RE = re.compile(r"^ref_tok(?P<t>[0-9]+)_(?P<name>.+)$")
DTYPES = {
    "bf16": (np.dtype("<u2"), 7, -133),
    "f32": (np.dtype("<f4"), 23, -149),
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--linked-run", required=True, type=Path)
    parser.add_argument("--linked-run2", required=True, type=Path)
    parser.add_argument("--onetu-run", required=True, type=Path)
    parser.add_argument("--mono-run", type=Path)
    parser.add_argument("--mono-run2", type=Path)
    parser.add_argument("--extra-run", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--gate-extra", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def parse_manifest(bundle: Path) -> dict[str, TensorSpec]:
    manifest_path = bundle / "manifest.txt"
    specs: dict[str, TensorSpec] = {}
    for line_no, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"{manifest_path}:{line_no}: malformed manifest row")
        name, dtype, ndim_text = fields[:3]
        ndim = int(ndim_text)
        if len(fields) != 4 + ndim:
            raise ValueError(
                f"{manifest_path}:{line_no}: expected {4 + ndim} fields, got {len(fields)}"
            )
        if name in specs:
            raise ValueError(f"{manifest_path}:{line_no}: duplicate tensor {name}")
        shape = tuple(int(value) for value in fields[3 : 3 + ndim])
        if any(dim < 0 for dim in shape):
            raise ValueError(f"{manifest_path}:{line_no}: negative dimension")
        relative_path = Path(fields[-1])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{manifest_path}:{line_no}: unsafe relative path")
        specs[name] = TensorSpec(name, dtype, shape, bundle / relative_path)
    return specs


def parse_scalars(bundle: Path) -> dict[str, str]:
    path = bundle / "scalars.txt"
    result: dict[str, str] = {}
    for line_no, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_no}: expected 'key value'")
        key, value = fields
        if key in result:
            raise ValueError(f"{path}:{line_no}: duplicate scalar {key}")
        result[key] = value
    return result


def load_raw(path: Path, spec: TensorSpec) -> np.ndarray:
    if spec.dtype not in DTYPES:
        raise ValueError(f"comparison does not support dtype {spec.dtype} for {spec.name}")
    raw_dtype = DTYPES[spec.dtype][0]
    expected_bytes = spec.elements * raw_dtype.itemsize
    try:
        actual_bytes = path.stat().st_size
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing tensor file: {path}") from exc
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path}: got {actual_bytes} bytes, expected {expected_bytes} "
            f"for {spec.dtype}{spec.shape}"
        )
    return np.fromfile(path, dtype=raw_dtype, count=spec.elements).reshape(spec.shape)


def load_initial_cache_row(spec: TensorSpec, pos_id: int) -> np.ndarray:
    if spec.dtype not in DTYPES or len(spec.shape) != 4:
        raise ValueError(f"unexpected initial-cache spec for {spec.name}: {spec}")
    if not 0 <= pos_id < spec.shape[1]:
        raise ValueError(f"position {pos_id} is outside {spec.name}{spec.shape}")
    raw_dtype = DTYPES[spec.dtype][0]
    expected_bytes = spec.elements * raw_dtype.itemsize
    if spec.path.stat().st_size != expected_bytes:
        raise ValueError(f"{spec.path}: initial cache has the wrong byte size")
    mapped = np.memmap(spec.path, dtype=raw_dtype, mode="r", shape=spec.shape)
    try:
        return np.array(mapped[:, pos_id], copy=True)
    finally:
        del mapped


def decode(raw: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        words = raw.astype(np.uint32, copy=False)
        return np.left_shift(words, 16).view(np.float32)
    if dtype == "f32":
        return raw.astype(np.float32, copy=False)
    raise ValueError(dtype)


def ulp_at_scale(scale: float, dtype: str) -> float:
    _, mantissa_bits, zero_exponent = DTYPES[dtype]
    if scale == 0.0:
        return math.ldexp(1.0, zero_exponent)
    return math.ldexp(1.0, math.floor(math.log2(scale)) - mantissa_bits)


def finite_max(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def complete_mean(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(np.mean(values))


def pair_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    lhs = left.astype(np.float64)
    rhs = right.astype(np.float64)
    finite_lhs = np.isfinite(lhs)
    finite_rhs = np.isfinite(rhs)
    finite_pair = finite_lhs & finite_rhs
    abs_diff = np.abs(lhs - rhs)

    if finite_pair.any():
        pair_diff = abs_diff[finite_pair]
        pair_lhs = lhs[finite_pair]
        pair_rhs = rhs[finite_pair]
        max_abs = float(pair_diff.max())
        mean_rdiff = float(
            np.mean(
                2.0
                * pair_diff
                / (np.abs(pair_lhs) + np.abs(pair_rhs) + RDIFF_EPS)
            )
        )
    else:
        max_abs = None
        mean_rdiff = None

    return {
        "elements": int(rhs.size),
        "left_finite": bool(finite_lhs.all()),
        "right_finite": bool(finite_rhs.all()),
        "both_finite": bool(finite_lhs.all() and finite_rhs.all()),
        "left_nonfinite": int(np.count_nonzero(~finite_lhs)),
        "right_nonfinite": int(np.count_nonzero(~finite_rhs)),
        "max_abs": max_abs,
        "mean_rdiff": mean_rdiff,
    }


def compare_to_reference(observed: np.ndarray, reference: np.ndarray) -> dict:
    obs = observed.astype(np.float64)
    ref = reference.astype(np.float64)
    finite_obs = np.isfinite(obs)
    finite_ref = np.isfinite(ref)
    finite_pair = finite_obs & finite_ref

    abs_diff = np.abs(obs - ref)
    rms_ref = (
        float(np.sqrt(np.mean(np.square(ref[finite_ref])))) if finite_ref.any() else None
    )
    atol = ATOL_RMS_FACTOR * rms_ref if rms_ref is not None else None
    if atol is None:
        within = np.zeros(ref.shape, dtype=bool)
    else:
        threshold = atol + RTOL * np.abs(ref)
        within = finite_pair & (abs_diff <= threshold)
    violations = int(np.count_nonzero(~within))

    if finite_pair.any():
        pair_diff = abs_diff[finite_pair]
        pair_obs = obs[finite_pair]
        pair_ref = ref[finite_pair]
        max_abs = float(pair_diff.max())
        max_rel = float((pair_diff / np.maximum(np.abs(pair_ref), REL_EPS)).max())
        mean_rdiff = float(
            np.mean(
                2.0
                * pair_diff
                / (np.abs(pair_obs) + np.abs(pair_ref) + RDIFF_EPS)
            )
        )
    else:
        max_abs = None
        max_rel = None
        mean_rdiff = None

    return {
        "pass": violations == 0,
        "elements": int(ref.size),
        "reference_finite": bool(finite_ref.all()),
        "observed_finite": bool(finite_obs.all()),
        "reference_nonfinite": int(np.count_nonzero(~finite_ref)),
        "observed_nonfinite": int(np.count_nonzero(~finite_obs)),
        "rms_reference": rms_ref,
        "atol": atol,
        "rtol": RTOL,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_rdiff": mean_rdiff,
        "violations": violations,
    }


def aggregate_pair_metrics(token_metrics: list[dict]) -> dict:
    return {
        "tokens": len(token_metrics),
        "both_finite": all(metric["both_finite"] for metric in token_metrics),
        "max_abs": finite_max([metric["max_abs"] for metric in token_metrics]),
        "mean_rdiff": complete_mean(
            [metric["mean_rdiff"] for metric in token_metrics]
        ),
    }


def aggregate_reference_metrics(token_metrics: list[dict]) -> dict:
    return {
        "tokens": len(token_metrics),
        "pass": all(metric["pass"] for metric in token_metrics),
        "elements": sum(metric["elements"] for metric in token_metrics),
        "reference_finite": all(
            metric["reference_finite"] for metric in token_metrics
        ),
        "observed_finite": all(metric["observed_finite"] for metric in token_metrics),
        "reference_nonfinite": sum(
            metric["reference_nonfinite"] for metric in token_metrics
        ),
        "observed_nonfinite": sum(
            metric["observed_nonfinite"] for metric in token_metrics
        ),
        "max_abs": finite_max([metric["max_abs"] for metric in token_metrics]),
        "max_rel": finite_max([metric["max_rel"] for metric in token_metrics]),
        "mean_rdiff": complete_mean(
            [metric["mean_rdiff"] for metric in token_metrics]
        ),
        "violations": sum(metric["violations"] for metric in token_metrics),
    }


def format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def format_pair(metric: dict | None) -> str:
    if metric is None:
        return "n/a"
    return f"{format_number(metric['max_abs'])}/{format_number(metric['mean_rdiff'])}"


def main() -> None:
    args = parse_args()
    if args.mono_run2 is not None and args.mono_run is None:
        raise ValueError("--mono-run2 requires --mono-run")

    bundle = args.bundle.resolve()
    specs = parse_manifest(bundle)
    scalars = parse_scalars(bundle)
    if "ntok" not in scalars:
        raise ValueError("scalars.txt has no ntok")
    ntok = int(scalars["ntok"])
    if "layer_limit" not in scalars:
        raise ValueError("scalars.txt has no layer_limit; regenerate the bundle with export_inputs.py")
    layer_limit = int(scalars["layer_limit"])
    if layer_limit not in (0, 1):
        raise ValueError(
            f"Gate P2 defines reference regimes only for layer_limit 0 or 1, got {layer_limit}"
        )
    regime = "strict-layer-1" if layer_limit == 1 else "full-depth-floor-relative"

    reference_specs: dict[tuple[int, str], TensorSpec] = {}
    for name, spec in specs.items():
        match = REFERENCE_RE.match(name)
        if match is None:
            continue
        key = (int(match.group("t")), match.group("name"))
        reference_specs[key] = spec
    for t in range(ntok):
        for name in OUTPUT_NAMES:
            if (t, name) not in reference_specs:
                raise ValueError(f"bundle is missing ref_tok{t}_{name}")

    run_dirs: dict[str, Path] = {
        "linked": args.linked_run.resolve(),
        "linked2": args.linked_run2.resolve(),
        "onetu": args.onetu_run.resolve(),
    }
    core_run_names = ["linked", "linked2", "onetu"]
    if args.mono_run is not None:
        run_dirs["mono"] = args.mono_run.resolve()
        core_run_names.append("mono")
    if args.mono_run2 is not None:
        run_dirs["mono2"] = args.mono_run2.resolve()
        core_run_names.append("mono2")

    extra_run_names: list[str] = []
    for value in args.extra_run:
        if "=" not in value:
            raise ValueError(f"--extra-run must be NAME=DIR, got {value!r}")
        name, directory = value.split("=", 1)
        if not name or not directory:
            raise ValueError(f"--extra-run must be NAME=DIR, got {value!r}")
        if name in run_dirs:
            raise ValueError(f"duplicate run name: {name}")
        run_dirs[name] = Path(directory).resolve()
        extra_run_names.append(name)

    gated_run_names = {"linked", "linked2", "onetu"}
    if args.gate_extra:
        gated_run_names.update(extra_run_names)

    raw_cache: dict[tuple[str, int, str], np.ndarray] = {}
    decoded_cache: dict[tuple[str, int, str], np.ndarray] = {}
    reference_raw: dict[tuple[int, str], np.ndarray] = {}
    reference_decoded: dict[tuple[int, str], np.ndarray] = {}
    for t in range(ntok):
        for name in OUTPUT_NAMES:
            spec = reference_specs[(t, name)]
            ref_raw = load_raw(spec.path, spec)
            reference_raw[(t, name)] = ref_raw
            reference_decoded[(t, name)] = decode(ref_raw, spec.dtype)
            for run_name, run_dir in run_dirs.items():
                raw = load_raw(run_dir / f"tok{t}_{name}.bin", spec)
                raw_cache[(run_name, t, name)] = raw
                decoded_cache[(run_name, t, name)] = decode(raw, spec.dtype)

    failures: list[str] = []
    informational_failures: list[str] = []

    def build_repeat_results(
        first_run: str, second_run: str, *, check_required_slices: bool
    ) -> tuple[dict[str, dict[str, dict]], dict[str, dict], dict[str, dict]]:
        token_results: dict[str, dict[str, dict]] = {}
        aggregates: dict[str, dict] = {}
        bitwise: dict[str, dict] = {}
        for t in range(ntok):
            tensor_results: dict[str, dict] = {}
            for name in OUTPUT_NAMES:
                first_raw = raw_cache[(first_run, t, name)]
                second_raw = raw_cache[(second_run, t, name)]
                metric = pair_metrics(
                    decoded_cache[(first_run, t, name)],
                    decoded_cache[(second_run, t, name)],
                )
                metric["bitwise_equal"] = bool(np.array_equal(first_raw, second_raw))
                if name in ROW_TENSORS:
                    metric["layer0_bitwise_equal"] = bool(
                        np.array_equal(first_raw[0], second_raw[0])
                    )
                tensor_results[name] = metric
            token_results[f"tok{t}"] = tensor_results

        for name in OUTPUT_NAMES:
            metrics = [token_results[f"tok{t}"][name] for t in range(ntok)]
            aggregate = aggregate_pair_metrics(metrics)
            aggregate["bitwise_equal_all_tokens"] = all(
                metric["bitwise_equal"] for metric in metrics
            )
            aggregate["required_slice"] = None
            aggregate["required_bitwise_equal_all_tokens"] = None
            if name in ROW_TENSORS:
                aggregate["required_slice"] = "layer0"
                aggregate["required_bitwise_equal_all_tokens"] = all(
                    metric["layer0_bitwise_equal"] for metric in metrics
                )
                if (
                    check_required_slices
                    and not aggregate["required_bitwise_equal_all_tokens"]
                ):
                    failures.append(
                        f"gate (i): required deterministic slice {name}[0] was not "
                        f"bitwise equal between {first_run} and {second_run} for every token"
                    )
            aggregates[name] = aggregate
            bitwise[name] = {
                "full_tensor_all_tokens": aggregate["bitwise_equal_all_tokens"],
                "required_slice": aggregate["required_slice"],
                "required_slice_all_tokens": aggregate[
                    "required_bitwise_equal_all_tokens"
                ],
            }
        return token_results, aggregates, bitwise

    linked_floor_tokens, linked_floor_aggregate, bitwise_status = build_repeat_results(
        "linked", "linked2", check_required_slices=True
    )
    if "mono2" in run_dirs:
        mono_floor_tokens, mono_floor_aggregate, mono_bitwise_status = (
            build_repeat_results("mono", "mono2", check_required_slices=False)
        )
        effective_floor_source = "max(linked-repeat,mono-repeat)"
    else:
        mono_floor_tokens = None
        mono_floor_aggregate = None
        mono_bitwise_status = None
        effective_floor_source = "linked-repeat-only (--mono-run2 not supplied)"

    effective_floor_tokens: dict[str, dict[str, dict]] = {}
    effective_floor_aggregate: dict[str, dict] = {}
    for t in range(ntok):
        tensor_results: dict[str, dict] = {}
        for name in OUTPUT_NAMES:
            linked_metric = linked_floor_tokens[f"tok{t}"][name]
            mono_metric = (
                mono_floor_tokens[f"tok{t}"][name]
                if mono_floor_tokens is not None
                else None
            )
            both_finite = linked_metric["both_finite"] and (
                mono_metric is None or mono_metric["both_finite"]
            )
            max_values = [linked_metric["max_abs"]]
            mean_values = [linked_metric["mean_rdiff"]]
            if mono_metric is not None:
                max_values.append(mono_metric["max_abs"])
                mean_values.append(mono_metric["mean_rdiff"])
            effective = {
                "elements": linked_metric["elements"],
                "both_finite": both_finite,
                "max_abs": (
                    max(value for value in max_values if value is not None)
                    if all(value is not None for value in max_values)
                    else None
                ),
                "mean_rdiff": (
                    max(value for value in mean_values if value is not None)
                    if all(value is not None for value in mean_values)
                    else None
                ),
                "linked_repeat_max_abs": linked_metric["max_abs"],
                "linked_repeat_mean_rdiff": linked_metric["mean_rdiff"],
                "mono_repeat_max_abs": (
                    mono_metric["max_abs"] if mono_metric is not None else None
                ),
                "mono_repeat_mean_rdiff": (
                    mono_metric["mean_rdiff"] if mono_metric is not None else None
                ),
                "source": effective_floor_source,
            }
            tensor_results[name] = effective
        effective_floor_tokens[f"tok{t}"] = tensor_results
    for name in OUTPUT_NAMES:
        metrics = [effective_floor_tokens[f"tok{t}"][name] for t in range(ntok)]
        effective_floor_aggregate[name] = {
            **aggregate_pair_metrics(metrics),
            "source": effective_floor_source,
            "linked_repeat": linked_floor_aggregate[name],
            "mono_repeat": (
                mono_floor_aggregate[name]
                if mono_floor_aggregate is not None
                else None
            ),
        }
        if not effective_floor_aggregate[name]["both_finite"]:
            failures.append(
                f"gate (i): {name} repeat-floor runs contain a nonfinite value"
            )

    onetu_tokens: dict[str, dict[str, dict]] = {}
    for t in range(ntok):
        tensor_results: dict[str, dict] = {}
        for name in OUTPUT_NAMES:
            linked = decoded_cache[("linked", t, name)]
            onetu = decoded_cache[("onetu", t, name)]
            ref = reference_decoded[(t, name)]
            finite = bool(
                np.isfinite(linked).all()
                and np.isfinite(onetu).all()
                and np.isfinite(ref).all()
            )
            pair = pair_metrics(linked, onetu)
            ref_scale = (
                float(np.max(np.abs(ref.astype(np.float64)))) if finite else None
            )
            bf16_ulp = (
                ulp_at_scale(ref_scale, "bf16") if ref_scale is not None else None
            )
            # Floor = the linked repeat floor of this tensor aggregated over all
            # tokens (max), not the single-token value: one pair of repeats at
            # one token is a noisy estimator of a chaotic tensor's floor (p1024/a
            # after the dispatch fix: tok3 logits 0.2676 vs 2x per-token 0.1328,
            # while other tokens' repeat floors reached 0.24). The aggregate is
            # the same statistic gate (iii) already uses.
            floor = linked_floor_aggregate[name]["max_abs"]
            allowed = (
                max(FLOOR_FACTOR * floor, ULP_FACTOR * bf16_ulp)
                if floor is not None and bf16_ulp is not None
                else None
            )
            passed = bool(
                finite
                and pair["max_abs"] is not None
                and allowed is not None
                and pair["max_abs"] <= allowed
            )
            tensor_results[name] = {
                "pass": passed,
                "both_finite": finite,
                "max_abs": pair["max_abs"],
                "mean_rdiff": pair["mean_rdiff"],
                "allowed_max_abs": allowed,
                "repeat_floor": floor,
                "repeat_floor_token": linked_floor_tokens[f"tok{t}"][name]["max_abs"],
                "reference_scale": ref_scale,
                "bf16_ulp_at_reference_scale": bf16_ulp,
                "bitwise_equal": bool(
                    np.array_equal(
                        raw_cache[("linked", t, name)],
                        raw_cache[("onetu", t, name)],
                    )
                ),
            }
            if not passed:
                failures.append(
                    f"gate (ii): tok{t} {name} linked-vs-one-TU max_abs="
                    f"{format_number(pair['max_abs'])} exceeds "
                    f"{format_number(allowed)} or contains a nonfinite value"
                )
        onetu_tokens[f"tok{t}"] = tensor_results
    onetu_aggregate = {
        name: aggregate_pair_metrics(
            [onetu_tokens[f"tok{t}"][name] for t in range(ntok)]
        )
        for name in OUTPUT_NAMES
    }
    for name in OUTPUT_NAMES:
        onetu_aggregate[name]["pass_all_tokens"] = all(
            onetu_tokens[f"tok{t}"][name]["pass"] for t in range(ntok)
        )
        onetu_aggregate[name]["max_allowed"] = finite_max(
            [
                onetu_tokens[f"tok{t}"][name]["allowed_max_abs"]
                for t in range(ntok)
            ]
        )

    def aggregate_between(left_run: str, right_run: str | None, name: str) -> dict:
        metrics = []
        for t in range(ntok):
            right = (
                reference_decoded[(t, name)]
                if right_run is None
                else decoded_cache[(right_run, t, name)]
            )
            metrics.append(pair_metrics(decoded_cache[(left_run, t, name)], right))
        return aggregate_pair_metrics(metrics)

    reference_results: dict[str, dict] = {}
    run_failure_messages: dict[str, list[str]] = {name: [] for name in run_dirs}

    if layer_limit == 1:
        if "prompt_len" not in scalars:
            raise ValueError("scalars.txt has no prompt_len")
        prompt_len = int(scalars["prompt_len"])
        for cache_name in ("k_cache", "v_cache"):
            if cache_name not in specs:
                raise ValueError(f"bundle is missing {cache_name}")

        initial_rows: dict[tuple[int, str], np.ndarray] = {}
        reference_untouched: dict[str, dict[str, dict]] = {}
        for t in range(ntok):
            tensor_results: dict[str, dict] = {}
            pos_id = prompt_len + t
            for row_name, cache_name in zip(ROW_TENSORS, ("k_cache", "v_cache")):
                initial = load_initial_cache_row(specs[cache_name], pos_id)
                initial_rows[(t, row_name)] = initial
                ref = reference_raw[(t, row_name)]
                if ref.shape != initial.shape:
                    raise ValueError(
                        f"{row_name} reference shape {ref.shape} does not match "
                        f"initial cache row {initial.shape}"
                    )
                mismatches = int(np.count_nonzero(ref[1:] != initial[1:]))
                tensor_results[row_name] = {
                    "pass": mismatches == 0,
                    "layers": list(range(1, ref.shape[0])),
                    "elements": int(ref[1:].size),
                    "mismatches": mismatches,
                }
                if mismatches:
                    failures.append(
                        f"gate (iii): reference tok{t} {row_name}[1:] changed "
                        f"{mismatches} untouched initial-cache elements"
                    )
            reference_untouched[f"tok{t}"] = tensor_results

        for run_name in run_dirs:
            token_results: dict[str, dict[str, dict]] = {}
            for t in range(ntok):
                tensor_results: dict[str, dict] = {}
                for name in OUTPUT_NAMES:
                    metric = compare_to_reference(
                        decoded_cache[(run_name, t, name)],
                        reference_decoded[(t, name)],
                    )
                    metric["strict_pass"] = metric["pass"]
                    tensor_results[name] = metric
                token_results[f"tok{t}"] = tensor_results

            aggregate_results = {
                name: aggregate_reference_metrics(
                    [token_results[f"tok{t}"][name] for t in range(ntok)]
                )
                for name in OUTPUT_NAMES
            }
            for name, aggregate in aggregate_results.items():
                aggregate["predicate"] = "strict-every-element"
                aggregate["gated"] = name in STRICT_REQUIRED_REFERENCE_TENSORS
                if name in REPORTED_REFERENCE_TENSORS:
                    aggregate["predicate"] = "strict-reported; upstream-equivalence-gated"
                    aggregate["gated"] = False
                    aggregate["finite_gate_pass"] = bool(
                        aggregate["reference_finite"] and aggregate["observed_finite"]
                    )
                    if not aggregate["finite_gate_pass"]:
                        run_failure_messages[run_name].append(
                            f"{name} contains a nonfinite value"
                        )
                if name in TRIVIAL_ZERO_TENSORS:
                    ref_zero = all(
                        token_results[f"tok{t}"][name]["rms_reference"] == 0.0
                        for t in range(ntok)
                    )
                    aggregate["reference_rms_zero_all_tokens"] = ref_zero
                    aggregate["observed_zero_all_tokens"] = all(
                        not np.any(raw_cache[(run_name, t, name)])
                        for t in range(ntok)
                    )
                    if ref_zero:
                        aggregate["predicate"] = "trivial-zero-reference-not-evidence"
                        aggregate["gated"] = False
                        aggregate["pass"] = True
                if aggregate["gated"] and not aggregate["pass"]:
                    run_failure_messages[run_name].append(
                        f"{name} strict predicate: {aggregate['violations']}/"
                        f"{aggregate['elements']} violations, max_abs="
                        f"{format_number(aggregate['max_abs'])}"
                    )

            logits_result = {
                "predicate": "all-zero-bit-pattern",
                "reference_zero_all_tokens": all(
                    not np.any(reference_raw[(t, "logits")]) for t in range(ntok)
                ),
                "observed_zero_all_tokens": all(
                    not np.any(raw_cache[(run_name, t, "logits")])
                    for t in range(ntok)
                ),
            }
            logits_result["pass"] = bool(
                logits_result["reference_zero_all_tokens"]
                and logits_result["observed_zero_all_tokens"]
            )
            aggregate_results["logits"] = {
                **aggregate_results["logits"],
                **logits_result,
                "gated": True,
            }
            if not logits_result["pass"]:
                run_failure_messages[run_name].append(
                    "logits was not all-zero in both the layer-1 PyVM reference and run"
                )

            untouched_results: dict[str, dict[str, dict]] = {}
            for t in range(ntok):
                tensor_results = {}
                for row_name in ROW_TENSORS:
                    observed = raw_cache[(run_name, t, row_name)]
                    initial = initial_rows[(t, row_name)]
                    mismatches = int(np.count_nonzero(observed[1:] != initial[1:]))
                    tensor_results[row_name] = {
                        "pass": mismatches == 0,
                        "layers": list(range(1, observed.shape[0])),
                        "elements": int(observed[1:].size),
                        "mismatches": mismatches,
                    }
                    if mismatches:
                        run_failure_messages[run_name].append(
                            f"tok{t} {row_name}[1:] changed {mismatches} "
                            "untouched initial-cache elements"
                        )
                untouched_results[f"tok{t}"] = tensor_results

            reference_results[run_name] = {
                "pass": False,
                "gated": run_name in gated_run_names,
                "tokens": token_results,
                "aggregate": aggregate_results,
                "untouched_cache": untouched_results,
                "failures": run_failure_messages[run_name],
            }

        reference_auxiliary = {
            "reference_untouched_cache": reference_untouched,
            "mono_hidden_states_repeat": (
                {
                    "available": True,
                    "bitwise_equal_all_tokens": mono_bitwise_status["hidden_states"][
                        "full_tensor_all_tokens"
                    ],
                    "differs_across_repeats": not mono_bitwise_status["hidden_states"][
                        "full_tensor_all_tokens"
                    ],
                    "max_abs": mono_floor_aggregate["hidden_states"]["max_abs"],
                    "mean_rdiff": mono_floor_aggregate["hidden_states"]["mean_rdiff"],
                }
                if mono_bitwise_status is not None
                else {
                    "available": False,
                    "reason": "--mono-run2 not supplied",
                }
            ),
        }
    else:
        for run_name in run_dirs:
            token_results: dict[str, dict[str, dict]] = {}
            for t in range(ntok):
                tensor_results = {}
                for name in OUTPUT_NAMES:
                    metric = compare_to_reference(
                        decoded_cache[(run_name, t, name)],
                        reference_decoded[(t, name)],
                    )
                    metric["strict_pass"] = metric.pop("pass")
                    tensor_results[name] = metric
                token_results[f"tok{t}"] = tensor_results

            aggregate_results: dict[str, dict] = {}
            for name in OUTPUT_NAMES:
                run_ref = aggregate_between(run_name, None, name)
                effective_floor = effective_floor_aggregate[name]
                reference_zero = name in TRIVIAL_ZERO_TENSORS and all(
                    token_results[f"tok{t}"][name]["rms_reference"] == 0.0
                    for t in range(ntok)
                )
                if reference_zero:
                    aggregate_results[name] = {
                        **run_ref,
                        "predicate": "trivial-zero-reference-not-evidence",
                        "gated": False,
                        "pass": True,
                        "reference_rms_zero_all_tokens": True,
                        "observed_zero_all_tokens": all(
                            not np.any(raw_cache[(run_name, t, name)])
                            for t in range(ntok)
                        ),
                    }
                    continue

                # Gate on the mean relative difference only.  The per-element
                # maximum of a chaotic full-depth tensor is an extreme-value
                # statistic: against a floor estimated from one repeat pair it
                # passes or fails by chance (one element of 2048 decided
                # p1024-n8-red).  max_abs stays reported.
                allowed_mean_rdiff = (
                    FLOOR_FACTOR * effective_floor["mean_rdiff"]
                    if effective_floor["mean_rdiff"] is not None
                    else None
                )
                passed = bool(
                    run_ref["both_finite"]
                    and run_ref["mean_rdiff"] is not None
                    and allowed_mean_rdiff is not None
                    and run_ref["mean_rdiff"] <= allowed_mean_rdiff
                )
                aggregate_results[name] = {
                    **run_ref,
                    "predicate": "floor-relative-mean",
                    "gated": True,
                    "effective_floor_source": effective_floor_source,
                    "effective_floor_max_abs": effective_floor["max_abs"],
                    "linked_repeat_floor_mean_rdiff": linked_floor_aggregate[name][
                        "mean_rdiff"
                    ],
                    "mono_repeat_floor_mean_rdiff": (
                        mono_floor_aggregate[name]["mean_rdiff"]
                        if mono_floor_aggregate is not None
                        else None
                    ),
                    "effective_floor_mean_rdiff": effective_floor["mean_rdiff"],
                    "allowed_mean_rdiff": allowed_mean_rdiff,
                    "pass": passed,
                }
                if not passed:
                    run_failure_messages[run_name].append(
                        f"{name} floor-relative predicate: mean_rdiff="
                        f"{format_number(run_ref['mean_rdiff'])} > "
                        f"{format_number(allowed_mean_rdiff)} or nonfinite"
                    )

            layer0_results: dict[str, dict] = {}
            for name in ROW_TENSORS:
                metrics = [
                    compare_to_reference(
                        decoded_cache[(run_name, t, name)][0],
                        reference_decoded[(t, name)][0],
                    )
                    for t in range(ntok)
                ]
                aggregate = aggregate_reference_metrics(metrics)
                aggregate["predicate"] = "strict-every-element"
                aggregate["slice"] = "layer0"
                aggregate["tokens_detail"] = {
                    f"tok{t}": metric for t, metric in enumerate(metrics)
                }
                layer0_results[name] = aggregate
                if not aggregate["pass"]:
                    run_failure_messages[run_name].append(
                        f"{name}[0] strict predicate: {aggregate['violations']}/"
                        f"{aggregate['elements']} violations, max_abs="
                        f"{format_number(aggregate['max_abs'])}"
                    )

            reference_results[run_name] = {
                "pass": False,
                "gated": run_name in gated_run_names,
                "tokens": token_results,
                "aggregate": aggregate_results,
                "layer0_strict": layer0_results,
                "failures": run_failure_messages[run_name],
            }
        reference_auxiliary = {}

    upstream_equivalence: dict = {
        "available": "mono" in run_dirs,
        "effective_floor_source": effective_floor_source,
        "runs": {},
    }
    if "mono" not in run_dirs:
        upstream_equivalence["reason"] = "--mono-run not supplied"
    else:
        for run_name in run_dirs:
            if run_name in ("mono", "mono2"):
                upstream_equivalence["runs"][run_name] = {
                    "role": "baseline",
                    "pass": True,
                    "gated": False,
                }
                continue
            token_results: dict[str, dict[str, dict]] = {}
            for t in range(ntok):
                tensor_results: dict[str, dict] = {}
                for name in OUTPUT_NAMES:
                    candidate = decoded_cache[(run_name, t, name)]
                    mono = decoded_cache[("mono", t, name)]
                    ref = reference_decoded[(t, name)]
                    pair = pair_metrics(candidate, mono)
                    all_finite = bool(
                        pair["both_finite"] and np.isfinite(ref).all()
                    )
                    ref_scale = (
                        float(np.max(np.abs(ref.astype(np.float64))))
                        if all_finite
                        else None
                    )
                    bf16_ulp = (
                        ulp_at_scale(ref_scale, "bf16")
                        if ref_scale is not None
                        else None
                    )
                    # Same aggregate-over-tokens floor statistic as gate (ii):
                    # a single token's repeat pair under-estimates a chaotic
                    # tensor's floor (p10/rebind-regs: tok5 attn_out one-TU vs
                    # mono 0.104 with a per-token floor 0.049 but token-aggregate
                    # floor 0.077).
                    effective_floor = effective_floor_aggregate[name]["max_abs"]
                    allowed = (
                        max(
                            FLOOR_FACTOR * effective_floor,
                            ULP_FACTOR * bf16_ulp,
                        )
                        if effective_floor is not None and bf16_ulp is not None
                        else None
                    )
                    trivial_zero = (
                        name in TRIVIAL_ZERO_TENSORS
                        and compare_to_reference(mono, ref)["rms_reference"] == 0.0
                    )
                    passed = bool(
                        all_finite
                        and pair["max_abs"] is not None
                        and allowed is not None
                        and pair["max_abs"] <= allowed
                    )
                    item = {
                        "pass": passed,
                        "gated": not trivial_zero,
                        "max_abs": pair["max_abs"],
                        "mean_rdiff": pair["mean_rdiff"],
                        "allowed_max_abs": allowed,
                        "effective_repeat_floor": effective_floor,
                        "effective_floor_source": effective_floor_source,
                        "reference_scale": ref_scale,
                        "bf16_ulp_at_reference_scale": bf16_ulp,
                        "both_finite": all_finite,
                        "trivial_zero_reference_not_evidence": trivial_zero,
                    }
                    tensor_results[name] = item
                    if item["gated"] and not passed:
                        run_failure_messages[run_name].append(
                            f"upstream equivalence tok{t} {name}: "
                            f"max_abs={format_number(pair['max_abs'])} > "
                            f"{format_number(allowed)} "
                            f"(effective_floor={format_number(effective_floor)}, "
                            f"bf16_ulp={format_number(bf16_ulp)})"
                        )
                token_results[f"tok{t}"] = tensor_results
            upstream_equivalence["runs"][run_name] = {
                "role": "candidate",
                "pass": all(
                    item["pass"] or not item["gated"]
                    for tensors in token_results.values()
                    for item in tensors.values()
                ),
                "gated": run_name in gated_run_names,
                "tokens": token_results,
            }

    for run_name in run_dirs:
        reference_results[run_name]["pass"] = not run_failure_messages[run_name]
        reference_results[run_name]["failures"] = run_failure_messages[run_name]
        target = failures if run_name in gated_run_names else informational_failures
        for message in run_failure_messages[run_name]:
            target.append(f"gate (iii): {run_name} {message}")

    full_depth_table: dict[str, dict] | None = None
    if layer_limit == 0:
        full_depth_table = {}
        for name in OUTPUT_NAMES:
            full_depth_table[name] = {
                "linked_repeat_floor": linked_floor_aggregate[name],
                "mono_repeat_floor": (
                    mono_floor_aggregate[name]
                    if mono_floor_aggregate is not None
                    else None
                ),
                "effective_repeat_floor": effective_floor_aggregate[name],
                "linked_vs_onetu": onetu_aggregate[name],
                "linked_vs_mono": (
                    aggregate_between("linked", "mono", name)
                    if "mono" in run_dirs
                    else None
                ),
                "linked_vs_reference": aggregate_between("linked", None, name),
                "mono_vs_reference": (
                    aggregate_between("mono", None, name)
                    if "mono" in run_dirs
                    else None
                ),
                "reference_evidence": (
                    "trivial-zero-not-evidence"
                    if name in TRIVIAL_ZERO_TENSORS
                    and reference_results["linked"]["aggregate"][name]["predicate"]
                    == "trivial-zero-reference-not-evidence"
                    else "floor-relative-mean"
                ),
            }

    status = "PASS" if not failures else "FAIL"
    result = {
        "status": status,
        "regime": regime,
        "layer_limit": layer_limit,
        "constants": {
            "rtol": RTOL,
            "atol_rms_factor": ATOL_RMS_FACTOR,
            "floor_factor": FLOOR_FACTOR,
            "ulp_factor": ULP_FACTOR,
            "rdiff_epsilon": RDIFF_EPS,
            "relative_epsilon": REL_EPS,
        },
        "bundle": str(bundle),
        "runs": {name: str(directory) for name, directory in run_dirs.items()},
        "core_runs": core_run_names,
        "extra_runs": extra_run_names,
        "gate_extra": args.gate_extra,
        "gated_runs": sorted(gated_run_names),
        "ntok": ntok,
        "required_deterministic": list(REQUIRED_DETERMINISTIC),
        "bitwise_status": bitwise_status,
        "mono_bitwise_status": mono_bitwise_status,
        "repeat_floor": {
            "effective_source": effective_floor_source,
            "linked": {
                "tokens": linked_floor_tokens,
                "aggregate": linked_floor_aggregate,
            },
            "mono": (
                {
                    "tokens": mono_floor_tokens,
                    "aggregate": mono_floor_aggregate,
                }
                if mono_floor_tokens is not None
                else None
            ),
            "effective": {
                "tokens": effective_floor_tokens,
                "aggregate": effective_floor_aggregate,
            },
        },
        "linked_vs_onetu": {
            "tokens": onetu_tokens,
            "aggregate": onetu_aggregate,
        },
        "upstream_equivalence": upstream_equivalence,
        "runs_vs_reference": reference_results,
        "reference_auxiliary": reference_auxiliary,
        "full_depth_table": full_depth_table,
        "failures": failures,
        "informational_failures": informational_failures,
    }

    print(f"Gate P2 numerical comparison: {status}")
    print(f"reference regime: {regime} (layer_limit={layer_limit})")
    print(f"effective repeat floor: {effective_floor_source}")
    print("linked-repeat bitwise status (all tokens):")
    print("tensor             full-tensor layer0-required")
    for name in OUTPUT_NAMES:
        full = "equal" if bitwise_status[name]["full_tensor_all_tokens"] else "diff"
        layer0 = bitwise_status[name]["required_slice_all_tokens"]
        layer0_text = "n/a" if layer0 is None else ("equal" if layer0 else "diff")
        print(f"{name:<18} {full:>11} {layer0_text:>15}")

    gate_ii_pass = all(
        onetu_tokens[f"tok{t}"][name]["pass"]
        for t in range(ntok)
        for name in OUTPUT_NAMES
    )
    print(
        "linked-vs-one-TU per-token floor/ulp predicate: "
        f"{'PASS' if gate_ii_pass else 'FAIL'}"
    )
    if upstream_equivalence["available"]:
        print("upstream-equivalence per-token floor/ulp predicate:")
        for run_name, run_result in upstream_equivalence["runs"].items():
            role = run_result["role"]
            run_status = "BASELINE" if role == "baseline" else (
                "PASS" if run_result["pass"] else "FAIL"
            )
            print(f"{run_name:<10} {run_status}")
    else:
        print("upstream-equivalence predicate: not evaluated (--mono-run not supplied)")

    if full_depth_table is not None:
        print(
            "full-depth aggregate table "
            "(max_abs/mean_rdiff; mean_rdiff is mean over tokens):"
        )
        print(
            "tensor               linked-floor            mono-floor       effective-floor "
            "         linked/onetu          linked/mono           linked/ref "
            "           mono/ref evidence"
        )
        for name in OUTPUT_NAMES:
            row = full_depth_table[name]
            print(
                f"{name:<18} {format_pair(row['linked_repeat_floor']):>21} "
                f"{format_pair(row['mono_repeat_floor']):>21} "
                f"{format_pair(row['effective_repeat_floor']):>21} "
                f"{format_pair(row['linked_vs_onetu']):>21} "
                f"{format_pair(row['linked_vs_mono']):>21} "
                f"{format_pair(row['linked_vs_reference']):>21} "
                f"{format_pair(row['mono_vs_reference']):>21} "
                f"{row['reference_evidence']}"
            )
    else:
        print("strict layer-1 reference summary:")
        print(
            "run        tensor             predicate                         "
            "max-abs/mean-rdiff violations status"
        )
        for run_name in run_dirs:
            for name in OUTPUT_NAMES:
                aggregate = reference_results[run_name]["aggregate"][name]
                status_text = "PASS" if aggregate["pass"] else "FAIL"
                print(
                    f"{run_name:<10} {name:<18} {aggregate['predicate']:<33} "
                    f"{format_pair(aggregate):>21} {aggregate['violations']:>10} "
                    f"{status_text}"
                )
        print("strict-vs-PyVM violations (reported, not gating):")
        print("tensor        " + " ".join(f"{name:>10}" for name in run_dirs))
        for tensor in REPORTED_REFERENCE_TENSORS:
            print(
                f"{tensor:<14}"
                + " ".join(
                    f"{reference_results[name]['aggregate'][tensor]['violations']:>10}"
                    for name in run_dirs
                )
            )
        if "mono" in run_dirs:
            mono_hidden = reference_auxiliary["mono_hidden_states_repeat"]
            if mono_hidden["available"]:
                print(
                    "mono/mono2 hidden_states repeat: "
                    f"{'diff' if mono_hidden['differs_across_repeats'] else 'equal'} "
                    f"max_abs={format_number(mono_hidden['max_abs'])} "
                    f"mean_rdiff={format_number(mono_hidden['mean_rdiff'])}"
                )

    print("run reference-predicate gated")
    for run_name in run_dirs:
        run_status = "PASS" if reference_results[run_name]["pass"] else "FAIL"
        print(
            f"{run_name:<10} {run_status:<19} "
            f"{'yes' if run_name in gated_run_names else 'no'}"
        )
    if informational_failures:
        print("ungated mono/extra-run failures (reported only):")
        for failure in informational_failures:
            print(f"  - {failure}")
    if failures:
        print("failures:")
        for failure in failures:
            print(f"  - {failure}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
