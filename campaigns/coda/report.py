#!/usr/bin/env python3
"""Render one CODA results JSON as Markdown.

A report reads exactly one file and never merges other runs.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOCK_BANNER = ("> Unlocked or unrecorded SM clocks: timings are valid as relative comparisons only, "
                "not comparable with the paper's locked-clock numbers.")


def lease_status(lease):
    """``locked``, ``unlocked`` or ``unknown`` for one clock-lease record of any receipt schema."""
    if not isinstance(lease, dict):
        return "unknown"
    if lease.get("unlocked") is True or lease.get("locked") is False:
        return "unlocked"
    if lease.get("locked") is True or lease.get("sm_mhz") is not None:
        return "locked"
    return "unknown"


def clock_status(results):
    """Clock status of a results receipt; never assumes a lock that no field records."""
    if isinstance(results.get("clock_locked"), bool):
        return "locked" if results["clock_locked"] else "unlocked"
    if results.get("mode") == "unlocked" or "unlocked" in (results.get("measurement_label") or ""):
        return "unlocked"
    return lease_status(results.get("clock"))


def has_timing(results):
    public = results.get("public", results if "cases" in results else None) or {}
    return (any(cell.get("timings") for cell in results.get("cells", {}).values())
            or any(case.get("milliseconds") for case in public.get("cases", [])))


def clock_text(sample):
    if not sample:
        return "not sampled"
    return (f"{sample['clock_min_mhz']}/{sample['clock_median_mhz']}; "
            f"{sample['sw_power_cap_samples']}/{sample['samples']} cap")


def admission(record, clock):
    admitted = record.get("admitted", clock.get("healthy"))
    return "ADMITTED" if admitted is True else "NON-ADMITTED" if admitted is False else "unlocked/external"


def render(results):
    clock = results.get("clock") or {}
    status = clock_status(results)
    clock_line = {"locked": f"Clock lock: `{clock.get('sm_mhz')}` MHz, healthy `{clock.get('healthy')}`.",
                  "unlocked": "Clock: unlocked (no SM clock lock).",
                  "unknown": "Clock: not recorded in this receipt."}[status]
    lines = ["# CODA results", ""]
    if status != "locked" and has_timing(results):
        lines += [CLOCK_BANNER, ""]
    lines += [f"Profile: `{results['profile']}`. Status: **{results['status']}**.",
              clock_line, "",
             "Same-source parity passes when one-TU / linked ≥ 0.97. All speedups are higher-is-better: "
             "placement columns show baseline / linked time, producer columns cuBLAS / producer time.", ""]
    if results.get("measurement_label"):
        lines += [f"Measurement regime: **{results['measurement_label']}**.", ""]
    lines += ["Clock columns give sampled min/median MHz and how many samples carried the "
              "SW_POWER_CAP bit; those samples are informational and do not affect admission.", ""]
    protocols = {row["pair"].get("protocol") for cell in results.get("cells", {}).values()
                 for row in cell.get("timings", [])}
    if protocols == {"paired-reload-v2"}:
        lines += ["Same-source parity uses paired-reload-v2: eight blocks with a module reload at the midpoint, "
                  "four batches per block and ten fixed repetitions. The parity speedup is the geometric "
                  "mean of the median speedups for the two module orders. Timings are admitted when "
                  "every duration is positive and finite, the qualified clock lease is healthy and no "
                  "foreign GPU process runs. There is no minimum-duration or per-arm coverage requirement.", ""]
    lines += ["Latency columns show median per-arm times. Speedups from ordinary paired timing are "
              "medians of per-block speedups, so they can differ from the ratio of the displayed latencies.", ""]
    if clock.get("healthy") is False:
        lines += [f"**NON-ADMITTED timings: the requested clock lease was unhealthy.**", ""]
    if "cells" in results:
        lines += ["## Same-source parity", "", "| Cell | M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Parity | Admission |", "|---|---:|---:|---|---:|---|---:|---|---|"]
        for name, cell in results["cells"].items():
            for row in cell["timings"]:
                telemetry = row["pair"].get("telemetry", {})
                lines.append(f"| {name} | {row['shape'][0]} | {row['linked_ms']:.6f} | {clock_text(telemetry.get('A'))} | {row['one_tu_ms']:.6f} | {clock_text(telemetry.get('B'))} | {row['pair']['ratio_onetu_over_linked']:.4f} | {row['status']} | {admission(row, clock)} |")
        lines += ["", "## Placement (diagnostic only)", "", "| Cell | M | Baseline | Baseline ms | Baseline clocks | Baseline / linked time | Admission |", "|---|---:|---|---:|---|---:|---|"]
        for name, cell in results["cells"].items():
            for row in cell["timings"]:
                natives = {kind: row["placements"][kind] for kind in ("fused", "unfused") if kind in row["placements"]}
                for baseline, record in {**natives, **row.get("public_placement", {})}.items():
                    if "milliseconds" in record:
                        telemetry = record.get("telemetry", record.get("pair", {}).get("telemetry", {}).get("B"))
                        lines.append(f"| {name} | {row['shape'][0]} | {baseline} | {record['milliseconds']:.6f} | {clock_text(telemetry)} | {record['speedup_over_baseline']:.4f} | {admission(record, clock)} |")
                    else:
                        lines.append(f"| {name} | {row['shape'][0]} | {baseline} | not measured ({record['status']}) | — | — | not timed |")
        lines += ["", "## GEMM-only producer quality (diagnostic)", "",
                  "| Cell | M | Producer GEMM ms | Producer clocks | cuBLAS / producer time | Admission |", "|---|---:|---:|---|---:|---|"]
        for name, cell in results["cells"].items():
            for row in cell["timings"]:
                gemm = row.get("producer_gemm")
                if gemm:
                    arm = gemm.get("timing_arm", "B")
                    lines.append(f"| {name} | {row['shape'][0]} | {gemm['milliseconds']:.6f} | {clock_text(gemm['pair'].get('telemetry', {}).get(arm))} | {gemm['speedup_over_cublas']:.4f} | {admission(gemm, clock)} |")
        lines += ["", "## Correctness and evidence", "", "| Cell | Matrix | Sanitizers | Source-withheld | SASS counts equal | Normalized identity | Backend-matched identity |", "|---|---|---|---|---|---|---|"]
        for name, cell in results["cells"].items():
            matrix = all(r["status"] == "PASS" for r in cell["correctness"])
            sanitizers = all(r["status"] == "PASS" for r in cell["sanitizers"])
            identity = all(v for k, v in cell["sass"]["checks"].items() if k != "no_local_traffic_or_calls")
            matched = cell["sass"].get("backend_matched")
            matched_identity = (all(v for k, v in matched["checks"].items() if k != "no_local_traffic_or_calls")
                                if matched else "not run")
            lines.append(f"| {name} | {'PASS' if matrix else 'FAIL'} | {'PASS' if sanitizers else 'FAIL'} | {cell['source_withheld']['status']} | {cell['sass']['checks']['instruction_counts_equal']} | {identity} | {matched_identity} |")
        for name, cell in results["cells"].items():
            if cell.get("stagger_sensitivity"):
                lines += ["", f"### {name}: zero-stagger sensitivity", "", "| M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Admission |", "|---:|---:|---|---:|---|---:|---|"]
                for row in cell["stagger_sensitivity"]:
                    telemetry = row["pair"].get("telemetry", {})
                    lines.append(f"| {row['shape'][0]} | {row['linked_ms']:.6f} | {clock_text(telemetry.get('A'))} | {row['one_tu_ms']:.6f} | {clock_text(telemetry.get('B'))} | {row['pair']['ratio_onetu_over_linked']:.4f} | {admission(row, clock)} |")
    public = results.get("public", results if "cases" in results else None)
    if public:
        lines += ["", "## Public end-to-end sequences", "", "| M | Sequence | ms | Clocks | Numeric check | Boundary-rule uses | Admission |", "|---:|---|---:|---|---|---:|---|"]
        for row in public["cases"]:
            if row["shape"][1:] != [51200, 5120]:
                continue
            for name, numeric in row["checks"].items():
                value = row.get("milliseconds", {}).get(name)
                ms = f"{value:.6f}" if value is not None else "not measured"
                arm = "A" if name == "compile" else "B"
                sampled = row.get("pair", {}).get("telemetry", {}).get(arm)
                lines.append(f"| {row['shape'][0]} | {public['labels'][name]} | {ms} | {clock_text(sampled)} | {numeric['status']} | {numeric.get('boundary_rule_used', 0)} | {admission(row, clock) if value is not None else 'not timed'} |")
        lines += ["", "The offline weight permutation is untimed; the Liger sequence times its runtime split-layout adaptation.",
                  "Both public sequences round the GEMM output to bf16, so they are checked against the unfused reference.",
                  "A boundary-rule use is a nominal mismatch that is accepted only because the adjacent bf16 rounding "
                  "of the intermediate lies within the f32 dot-product error bound."]
        lines += ["", "## Public GEMM-only placement", "",
                  "| M | Weight/output layout | ms | Clocks | Admission |", "|---:|---|---:|---|---|"]
        for row in public["cases"]:
            for name, ms in row.get("gemm_only_ms", {}).items():
                arm = "A" if name == "interleaved" else "B"
                sampled = row.get("gemm_pair", {}).get("telemetry", {}).get(arm)
                lines.append(f"| {row['shape'][0]} | {name} | {ms:.6f} | {clock_text(sampled)} | {admission(row, clock)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results JSON written by run.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = render(json.loads(args.results.read_text()))
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
