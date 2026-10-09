#!/usr/bin/env python3
"""Correctness-first absolute-throughput comparison of public sm90 products."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from adapters import load_registry
from cells import check, load_cells, load_selection, oracle_rows, reference, save_selection, select, useful_flops
from env_report import configure_caches
from timing import measure, read_clocks

PROJECT = Path(__file__).resolve().parent


def make_inputs(cell, device):
    generator = torch.Generator(device=device).manual_seed(cell["numeric_case"])
    common = (cell["batch"], cell["seqlen_q"])
    def tensor(heads):
        # Sample directly in bf16 so rounding cannot produce the excluded +2.
        return torch.empty((*common, heads, cell["head_dim"]), device=device, dtype=torch.bfloat16).uniform_(
            -2, 2, generator=generator)
    return tensor(cell["heads_q"]), tensor(cell["heads_kv"]), tensor(cell["heads_kv"])


def markdown(report):
    lines = ["# sm90 public attention", "", f"Mode: {report['mode']}. Ratios are Flex-Triton latency / implementation latency (higher is faster).",
             "Ceiling diagnostics use their own causal oracle/work and never enter sparse comparisons.", ""]
    if report["import_errors"]:
        lines += ["## Unavailable adapter modules", ""]
        lines += [f"- `{name}`: {error}" for name, error in report["import_errors"].items()]
        lines.append("")
    for cell in report["cells"]:
        rows = cell["results"]
        passed = [r for r in rows if r["status"] == "ok"]
        experts = [r for r in passed if r["role"] == "expert"]
        flex = next((r for r in passed if r["role"] == "modular-lower"), None)
        best = min(experts, key=lambda r: r["timing"]["median_ms"]) if experts else None
        lines += [f"## {cell['id']}", "", f"Visible pairs: {cell['pairs']}; useful GFLOP: {cell['useful_flops'] / 1e9:.9f}.",
                  f"Best expert: {best['impl'] if best else 'none measured' }.", "",
                  "| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |",
                  "|---|---|---|---|---:|---:|---:|---:|"]
        for result in sorted(rows, key=lambda r: r.get("timing", {}).get("median_ms", math.inf)):
            if result["status"] == "ok":
                ms = result["timing"]["median_ms"]
                ratio = f"{flex['timing']['median_ms'] / ms:.3f}" if flex else "—"
                metrics = f"{ms * 1000:.3f} | {result['useful_tflops']:.3f} | {ratio}"
            else:
                metrics = "— | — | —"
            reason = result.get("reason", "").replace("|", "\\|").replace("\n", " ")
            clock = result.get("clock", report.get("clock", "locked (legacy)"))
            lines.append(f"| {result['impl']} | {result['role']} | {result['status']} {reason} | {clock} | {metrics} | {result.get('prep_ms', '—')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_report(report, path):
    for cell in report["cells"]:
        cell["results"].sort(key=lambda r: r.get("timing", {}).get("median_ms", math.inf))
        passed = [r for r in cell["results"] if r["status"] == "ok"]
        experts = [r for r in passed if r["role"] == "expert"]
        cell["best_expert"] = experts[0]["impl"] if experts else None
        flex = next((r for r in passed if r["role"] == "modular-lower"), None)
        for result in cell["results"]:
            result["flex_triton_speedup"] = (
                flex["timing"]["median_ms"] / result["timing"]["median_ms"]
                if flex is not None and result["status"] == "ok" else None
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    path.with_suffix(".md").write_text(markdown(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default="all")
    parser.add_argument("--impls", default="all", help="Comma-separated names; all excludes opt-in harness-equivalence products")
    parser.add_argument("--mode", choices=("smoke", "definitive"), default="smoke")
    parser.add_argument("--gpu", type=int, required=True, help="CUDA device ordinal in this process")
    parser.add_argument("--clock-mhz", type=int, help="check an existing lease at N MHz (±3%%); default: unlocked")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selection-dir", type=Path, help="Shared raw W6 files; default results/sm90-selections")
    args = parser.parse_args(argv)
    if args.clock_mhz is not None and args.clock_mhz <= 0:
        parser.error("--clock-mhz must be positive")
    configure_caches()
    destination = args.out or PROJECT.parent / "results" / f"sm90-public-{args.mode}.json"
    selection_dir = args.selection_dir or destination.parent / "sm90-selections"
    cells = load_cells()
    if args.cells != "all":
        requested = set(args.cells.split(","))
        unknown = requested - {c["id"] for c in cells}
        if unknown:
            parser.error(f"Unknown cells: {sorted(unknown)}")
        cells = [c for c in cells if c["id"] in requested]
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("Public baseline requires an sm90 GPU")
    torch.set_float32_matmul_precision("highest")
    impls, errors = load_registry()
    requested_impls = None if args.impls == "all" else set(args.impls.split(","))
    if requested_impls is not None:
        missing = requested_impls - {impl.name for impl in impls}
        for name in sorted(missing):
            errors[f"requested:{name}"] = "Implementation unavailable or unknown; see adapter import errors"
        impls = [impl for impl in impls if impl.name in requested_impls]
    else:
        impls = [impl for impl in impls if impl.role != "equivalence"]
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    report = {
        "schema": "pipelink-sm90-public-v1", "mode": args.mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": uuid, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "clock": "unlocked" if args.clock_mhz is None else "locked",
        "expected_sm_mhz": args.clock_mhz, "import_errors": errors, "cells": [],
        "timing_contract": "full product call, warm L2, no graphs, >=20ms per interval; median/N; 10 reps (2 smoke)",
    }
    with torch.inference_mode():
        for cell in cells:
            cpu_sel, selection_path = None, None
            if cell["mask"]["kind"] == "msa_selected":
                selection_path = selection_dir / (cell["id"] + ".i32")
                if not selection_path.exists():
                    save_selection(cell, select(cell), selection_path)
                cpu_sel = load_selection(cell, selection_path)
            flops = useful_flops(cell, cpu_sel)
            entry = {"id": cell["id"], "cell": cell, "pairs": flops // (4 * cell["head_dim"]),
                     "useful_flops": flops, "oracle_rows": oracle_rows(cell),
                     "selection_file": str(selection_path) if selection_path else None, "results": []}
            report["cells"].append(entry)
            q, k, v = make_inputs(cell, device)
            sel = cpu_sel.to(device) if cpu_sel is not None else None
            ref = None
            for impl in impls:
                result = {"impl": impl.name, "role": impl.role, "clock": report["clock"],
                          "clock_mhz": args.clock_mhz}
                entry["results"].append(result)
                prepared = None
                try:
                    reason = impl.supports(cell)
                    if bool(cell.get("diagnostic")) != (impl.role == "ceiling"):
                        reason = "Ceiling implementations only run diagnostic cells; diagnostics are not placement rows"
                    if reason is not None:
                        result.update(status="unsupported", reason=reason)
                        continue
                    if ref is None:
                        ref = reference(cell, q, k, v, sel)
                    prepared = impl.prepare(cell, q, k, v, sel)
                    result.update(prep_ms=prepared.prep_ms, config=prepared.config)
                    prepared.run()
                    o, lse = prepared.outputs()
                    correctness = check(cell, o, lse, ref)
                    result.update(correctness=correctness, lse_returned=lse is not None)
                    del o, lse
                    if not correctness["pass"]:
                        result["status"] = "incorrect"
                        continue
                    result["clocks_before"] = read_clocks(uuid)
                    timing = measure(prepared.run, reps=2 if args.mode == "smoke" else 10, smoke=args.mode == "smoke")
                    result["clocks_after"] = read_clocks(uuid)
                    # A lease may read back at a nearby hardware clock step.
                    if args.clock_mhz is not None and any(
                            abs(result[key]["sm_mhz"] - args.clock_mhz) > 0.03 * args.clock_mhz
                            for key in ("clocks_before", "clocks_after")):
                        result.update(status="clock_violation",
                                      reason=f"Discarded; rerun at locked {args.clock_mhz} MHz", rejected_timing=timing)
                    else:
                        result.update(status="ok", timing=timing, useful_tflops=flops / (timing["median_ms"] * 1e9))
                except Exception as exc:
                    result.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                finally:
                    del prepared
                    write_report(report, destination)
                print(f"{cell['id']} {impl.name}: {result['status']} ({result['clock']})", flush=True)
            del q, k, v, sel, ref, cpu_sel
            write_report(report, destination)
    write_report(report, destination)
    print(f"Wrote {destination} and {destination.with_suffix('.md')}")


if __name__ == "__main__":
    main()
