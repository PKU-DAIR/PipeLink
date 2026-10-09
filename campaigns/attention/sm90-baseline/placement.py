#!/usr/bin/env python3
"""Join public and slot absolute timings, with independently measured R1 parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cells import load_cells

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(path, optional=False):
    if optional and not path.exists():
        return {"cases": []}
    return json.loads(path.read_text())


def clock_record(row, report):
    timing = report.get("timing")
    target = (timing.get("clock_mhz") if isinstance(timing, dict)
              else report.get("clock_mhz", report.get("expected_sm_mhz")))
    mhz = row.get("clock_mhz", target if "clock" not in report else None)
    # Rows recorded before per-row clock modes were measured under a clock lease;
    # a partial rerun's merge must not relabel them as unlocked.
    mode = row.get("clock", "locked" if row else "pending")
    return {"clock": mode, "clock_mhz": mhz,
            "clocks_before": row.get("clocks_before"),
            "clocks_after": row.get("clocks_after")}


def clock_label(row):
    mode = row["clock"]
    return f"{mode} ({row['clock_mhz']} MHz)" if row.get("clock_mhz") is not None else mode


def absolute_measurement(case, mode, flops, report):
    measurement = case.get("measurements", {}).get(mode, {})
    ms = measurement.get("absolute_median_ms")
    status = measurement.get("status", "pending")
    if status != "PASS":
        ms = None
    return {
        "impl": f"pipelink-{mode}", "role": mode, "status": status,
        "latency_us": ms * 1000 if ms is not None else None,
        "useful_tflops": flops / (ms * 1e9) if ms is not None else None,
        "config": {"pipeline": case.get("pipeline")}, **clock_record(case, report),
    }


def public_measurement(result, report):
    ms = result.get("timing", {}).get("median_ms") if result["status"] == "ok" else None
    return {
        "impl": result["impl"], "role": result["role"], "status": result["status"],
        "latency_us": ms * 1000 if ms is not None else None,
        "useful_tflops": result.get("useful_tflops") if ms is not None else None,
        # The JIT workspace is a machine-local cache path, not part of the measured configuration.
        "config": {key: value for key, value in result.get("config", {}).items() if key != "jit_workspace"},
        "reason": result.get("reason", result.get("error", "")),
        "prep_ms": result.get("prep_ms"), "lse_returned": result.get("lse_returned"),
        **clock_record(result, report),
    }


def ratio(reference, linked):
    if reference is None or linked is None:
        return None
    return reference / linked


def fastest(rows, *, role=None, impl=None):
    candidates = [r for r in rows if r["latency_us"] is not None
                  and (role is None or r["role"] == role)
                  and (impl is None or r["impl"] == impl)]
    return min(candidates, key=lambda r: r["latency_us"], default=None)


def build_report(public, absolute, definitive, equivalence, stock):
    public_cells = {c["id"]: c for c in public["cells"]}
    absolute_cells = {c["case_id"]: c for c in absolute["cases"]}
    r1_cases = {(c["case_id"], c["pipeline"]): c for c in definitive["cases"]}
    report = {"schema": "pipelink-sm90-placement-v1", "cells": [], "ceilings": [],
              "harness_equivalence": [], "public_environment": {
                  k: v for k, v in public.items() if k != "cells"}}
    for cell in load_cells():
        if cell["id"] not in public_cells:
            continue
        source = public_cells[cell["id"]]
        rows = [public_measurement(r, public) for r in source["results"]]
        # Public results already count the exact W6 selections. Never substitute
        # the slot's dense/half-dense FLOP count or regenerate a different selection.
        flops = source["useful_flops"]
        if cell["workload"] == "W6dense":
            report["ceilings"].append({"id": cell["id"], "pairs": source["pairs"],
                                       "useful_flops": flops, "implementations": rows,
                                       "fa3": fastest(rows, impl="fa3-dense-ceiling")})
            continue
        case = absolute_cells.get(cell["id"], {})
        linked = absolute_measurement(case, "linked", flops, absolute)
        control = absolute_measurement(case, "control", flops, absolute)
        r1_case = r1_cases.get((cell["id"], case.get("pipeline")), {})
        r1 = r1_case.get("ratios", {}).get("R1", {})
        expert = None if cell["workload"] == "W6" else fastest(rows, role="expert")
        references = {"flex_triton": fastest(rows, impl="flex-triton"),
                      "best_expert": expert, "best_production": fastest(rows, role="production"),
                      "flex_flash": fastest(rows, impl="flex-flash")}
        report["cells"].append({
            "id": cell["id"], "pipeline": case.get("pipeline"),
            "pairs": source["pairs"], "useful_flops": flops,
            "linked": linked, "control": control,
            "r1": {"ratio": r1.get("ratio"), "status": r1.get("status", "pending"),
                   "pair_valid_count": r1.get("pair_valid_count"), **clock_record(r1_case, definitive)},
            "best_expert_label": expert["impl"] if expert else (
                "none (exception)" if cell["workload"] == "W6" else "none measured"),
            "references": references,
            "ratios": {name: ratio(ref["latency_us"] if ref else None, linked["latency_us"])
                       for name, ref in references.items()},
            "implementations": [linked, control, *rows],
        })
    stock_cases = {c["case_id"]: c for c in stock["cases"]}
    equivalence_cells = {c["id"]: c for c in equivalence["cells"]}
    for cell_id in ("W1-8k", "W1-32k", "W2a-8k", "W2a-32k"):
        py = fastest([public_measurement(r, equivalence) for r in equivalence_cells.get(cell_id, {}).get("results", [])],
                     impl="fa3-slotcfg")
        slot = absolute_measurement(stock_cases.get(cell_id, {}), "stock", 0, stock)
        python_us, stock_us = py["latency_us"] if py else None, slot["latency_us"]
        diff = 100 * (python_us / stock_us - 1) if python_us is not None and stock_us is not None else None
        report["harness_equivalence"].append({
            "id": cell_id, "stock_us": stock_us, "python_us": python_us,
            "stock_clock": clock_label(slot), "python_clock": clock_label(py) if py else "pending",
            "percent_diff": diff, "bound_percent": 3,
            "within_bound": abs(diff) <= 3 if diff is not None else None,
        })
    checks = [r["within_bound"] for r in report["harness_equivalence"]]
    report["cross_harness_status"] = "supported" if all(c is True for c in checks) else (
        "blocked" if False in checks else "pending")
    return report


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def text(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def summary_table(report):
    lines = [
        "| Cell | Clock modes | Linked µs | Useful TFLOP/s | Control µs | R1 | Flex-Triton µs | L/Flex-Triton | Best expert | Expert µs | L/expert | Best production | Production µs | L/production | L/Flex-FLASH |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for c in report["cells"]:
        refs, ratios = c["references"], c["ratios"]
        latency = lambda name: fmt(refs[name]["latency_us"] if refs[name] else None)
        r1 = fmt(c["r1"]["ratio"]) if c["r1"]["ratio"] is not None else "pending"
        clocks = sorted({clock_label(r) for r in c["implementations"] if r["latency_us"] is not None}
                        | ({clock_label(c["r1"])} if c["r1"]["ratio"] is not None else set()))
        lines.append("| " + " | ".join([
            c["id"], ", ".join(clocks) or "pending", fmt(c["linked"]["latency_us"]), fmt(c["linked"]["useful_tflops"]),
            fmt(c["control"]["latency_us"]), r1, latency("flex_triton"), fmt(ratios["flex_triton"]),
            c["best_expert_label"], latency("best_expert"), fmt(ratios["best_expert"]),
            refs["best_production"]["impl"] if refs["best_production"] else "none measured",
            latency("best_production"), fmt(ratios["best_production"]), fmt(ratios["flex_flash"]),
        ]) + " |")
    return "\n".join(lines)


def markdown(report):
    lines = ["# sm90 attention placement", "", summary_table(report), "",
             "## Reading the columns", "",
             "Linked and control are absolute median per-call latencies, not parity-timer latencies. "
             "Useful TFLOP/s = the same cell's public useful_flops / (linked milliseconds × 10⁹). "
             "L/reference is a throughput ratio = reference latency / linked latency; higher is faster.", "",
             "R1 is the independently measured linked/control throughput ratio from sm90-definitive.json, "
             "joined by case ID and pipeline. Missing files or matching rows are pending, never inferred "
             "from the absolute latencies. Its parity floor remains 0.97.", "",
             "Clock modes label every timing source; unlocked observations are not clock leases. "
             "Mixed modes or different requested MHz are not like-for-like performance comparisons.", "",
             "Exceeding Flex-Triton (L/Flex-Triton > 1) and approaching the best expert "
             "(L/expert near 1) are reported targets, never placement gates. "
             "No numeric pass/fail threshold is assigned to 'approach'. "
             "W6 has best expert 'none (exception)' by the reporting policy, even when an adapter "
             "labels a measured W6 implementation expert; all such results remain visible below.", "",
             "## Harness equivalence", "",
             f"Cross-harness placement: {report['cross_harness_status']}. "
             "Difference = 100 × (Python fa3-slotcfg / slot stock − 1); bound is |difference| ≤ 3 %. "
             "A failed check blocks interpretation of cross-harness placement until explained, "
             "not because of a public-performance target.", "",
             "| Cell | Slot clock | Python clock | Slot stock µs | Python fa3-slotcfg µs | Difference % | 3 % bound |",
             "|---|---|---|---:|---:|---:|---|"]
    for row in report["harness_equivalence"]:
        status = "pending" if row["within_bound"] is None else ("within" if row["within_bound"] else "outside")
        lines.append(f"| {row['id']} | {row['stock_clock']} | {row['python_clock']} | "
                     f"{fmt(row['stock_us'])} | {fmt(row['python_us'])} | {fmt(row['percent_diff'])} | {status} |")
    lines += ["", "## W6dense throughput ceiling", "",
              "FA3 dense causal uses its own visible-pair count and oracle. This is throughput context "
              "only, not a W6 sparse expert, latency bound, or speedup denominator.", ""]
    for c in report["ceilings"]:
        lines.append(f"- {c['id']}: {fmt(c['fa3']['useful_tflops'] if c['fa3'] else None)} useful TFLOP/s "
                     f"(FA3 dense causal; {clock_label(c['fa3']) if c['fa3'] else 'pending'}; "
                     f"{c['pairs']} pairs, {c['useful_flops']} useful FLOPs).")
    lines += ["", "## Per-cell implementation details", "",
              "Every public entry is listed: run or failed candidates in the table, unsupported ones "
              "below it with their reasons. Best references use only successful timed results. "
              "Prep is excluded from headline latency; LSE=false means O-only. Each implementation's "
              "full configuration (API, dispatch, tiles, autotune choices) is in sm90-placement.json.", ""]
    for c in [*report["cells"], *report["ceilings"]]:
        lines += [f"### {c['id']}", "", f"Visible pairs: {c['pairs']}; useful FLOPs: {c['useful_flops']}.", ""]
        if "r1" in c:
            lines += [f"R1 status: {c['r1']['status']}; clock: {clock_label(c['r1'])}; "
                      f"valid pairs: {c['r1']['pair_valid_count']}.", ""]
        rows = [r for r in c["implementations"] if r["status"] != "unsupported"]
        clocks = {clock_label(r) for r in rows}
        shared_clock = len(clocks) == 1
        with_reason = any(r.get("reason") for r in rows)
        if shared_clock:
            lines += [f"All implementations below ran {clocks.pop()}.", ""]
        headings = ["Implementation", "Role", "Status", *([] if shared_clock else ["Clock"]),
                    "Latency µs", "Useful TFLOP/s", "Prep ms", "LSE", *(["Reason"] if with_reason else [])]
        aligns = ["---", "---", "---", *([] if shared_clock else ["---"]),
                  "---:", "---:", "---:", "---", *(["---"] if with_reason else [])]
        lines += ["| " + " | ".join(headings) + " |", "|" + "|".join(aligns) + "|"]
        for r in rows:
            cells = [r["impl"], r["role"], r["status"], *([] if shared_clock else [clock_label(r)]),
                     fmt(r["latency_us"]), fmt(r["useful_tflops"]), fmt(r.get("prep_ms")),
                     r.get("lse_returned", "—"), *([r.get("reason", "")] if with_reason else [])]
            lines.append("| " + " | ".join(map(text, cells)) + " |")
        lines.append("")
        unsupported = {}
        for r in c["implementations"]:
            if r["status"] == "unsupported":
                unsupported.setdefault(str(r.get("reason", "")), []).append(r["impl"])
        lines += [f"- Unsupported: {', '.join(impls)} ({reason or 'no reason recorded'})" for reason, impls in unsupported.items()]
        lines.append("")
    env = report["public_environment"]
    lines += ["## Sources and reproduction", "",
              f"Timing environment: {env.get('gpu_name')}, torch {env.get('torch_version')}, "
              f"CUDA {env.get('cuda_version')}, expected SM clock {env.get('expected_sm_mhz')} MHz. "
              f"Timing contract: {env.get('timing_contract')}.", "",
              "Inputs (receipts written locally by the collection workflow, not published):", ""]
    for name, source in report["sources"].items():
        lines.append(f"- {name}: `{source['file']}`" + (" (pending: absent)" if not source["present"] else ""))
    lines += ["", "This CPU-only join writes sm90-placement.json and sm90-placement.md; see "
              "[Placement join](../sm90-baseline/README.md#placement-join) for the command.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS, help="Input and output directory")
    args = parser.parse_args(argv)
    names = {"public": "sm90-public-definitive.json", "absolute": "sm90-absolute-final.json",
             "definitive": "sm90-definitive.json", "equivalence": "sm90-public-equivalence.json",
             "stock": "sm90-absolute-stock-baseline.json"}
    report = build_report(**{key: load(args.results / name, optional=key == "definitive")
                             for key, name in names.items()})
    report["sources"] = {key: {"file": name, "present": (args.results / name).exists()}
                         for key, name in names.items()}
    destination = args.results / "sm90-placement.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    destination.with_suffix(".md").write_text(markdown(report))
    print(summary_table(report))
    print(f"\nWrote {destination} and {destination.with_suffix('.md')}")


if __name__ == "__main__":
    main()
