#!/usr/bin/env python3
"""Fold raw MoK run receipts into the summary files that port/report.py renders.

Every subcommand reads raw receipts written by the campaign's own drivers and writes one
summary into the report input layout (see port/report.py):

  numeric  --runs DIR --check TAG [TAG ...] --sanitize TAG  -> numeric-summary.json
           run.py check/sanitize receipts: <tag>-check-both-rank<N>.json, <tag>-sanitize-summary.json
  timing   --runs DIR [DIR ...] (later wins)                 -> timing-summary.json
           run.py time receipts: <tag>-time-summary.json, -time-window.json, -time-both-rank<N>.json
  swap     --checks DIR --timing DIR                         -> owner-swap/gpu-results.json
           owner_swap/check_swap.py receipts <tag>-rank<N>.json plus one run.py time receipt
  facts-gpu --runs DIR --tag LABEL=TAG [...]                 -> checked-facts/gpu-runs.json
           run.py check --image linked receipts of the checked-facts GPU runs
  placement --bench DIR [DIR ...]                            -> placement-summary.json
           host/benchmark_upstream.sh LOG_DIRs; window.log holds tools/gpu_window.sh stderr

A timed run is admitted when run.py reports PASS, its GPU window is uncontended and, if it
requested a clock lock, the lease stayed healthy. With unlocked clocks (PIPELINK_LOCK_SM_MHZ
unset, empty or 0) runs are admitted without a lease and recorded as clock_locked=false.
GPU-free; run with any Python 3.10+.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import median

WINDOW = re.compile(r"gpu_window: waited=(\d+)s ran=(\d+)s contended=([01]) rc=(-?\d+)")


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(path)


def ranks(directory, prefix):
    """Rank receipts <prefix>-rank<N>.json in rank order."""
    found = [p for p in Path(directory).glob(f"{prefix}-rank[0-9]*.json") if p.stem.rsplit("rank", 1)[1].isdigit()]
    return [load(p) for p in sorted(found, key=lambda p: int(p.stem.rsplit("rank", 1)[1]))]


def window(directory, tag, mode):
    path = Path(directory) / f"{tag}-{mode}-window.json"
    return load(path) if path.exists() else None


def numeric(args):
    rows = []
    windows = {}
    for tag in args.check:
        for report in ranks(args.runs, f"{tag}-check-both"):
            cells = report["cells"]
            steps = [s for c in cells for s in c.get("steps", [])]
            graphs = [g for c in cells for g in c.get("graphs", [])]
            exact = [v for r in (*steps, *graphs) if "recompute_exact" in r
                     for v in (r["recompute_exact"].values() if isinstance(r["recompute_exact"], dict)
                               else [r["recompute_exact"]])]
            rows.append({"world": report["world"], "rank": report["rank"], "configurations": len(cells),
                         "statuses": dict(Counter(c["status"] for c in cells)),
                         "direct_generations": len(steps), "graph_image_generations": len(graphs),
                         "bitwise_failures": sum(s.get("linked_onetu_bitwise") is False for s in steps),
                         "recompute_exact_failures": sum(v is False for v in exact)})
            observed = window(args.runs, tag, "check")
            if observed is not None:
                windows[str(report["world"])] = {k: observed[k] for k in ("returncode", "contended", "timings_valid")}
    if not rows:
        raise SystemExit(f"no check receipts for tags {args.check} in {args.runs}")
    sanitizers = load(Path(args.runs) / f"{args.sanitize}-sanitize-summary.json")
    save(Path(args.evidence) / "numeric-summary.json",
         {"ranks": sorted(rows, key=lambda r: (r["world"], r["rank"])), "sanitizers": sanitizers, "windows": windows})


def pair_medians(pair):
    """Median per-launch milliseconds of arms A and B over all scored batches."""
    arms = {"A": [], "B": []}
    for block in pair["blocks"]:
        for arm, milliseconds in zip(block["order"], block["milliseconds"]):
            arms[arm].append(milliseconds / pair["repetitions"])
    return median(arms["A"]), median(arms["B"])


def timed_runs(directories):
    """Every run.py time invocation with its admission verdict."""
    runs = []
    for directory in map(Path, directories):
        for summary_path in sorted(directory.glob("*-time-summary.json")):
            tag = summary_path.name.removesuffix("-time-summary.json")
            summary = load(summary_path)
            observed = window(directory, tag, "time")
            contended = bool(observed and observed["contended"])
            locked = bool(summary.get("clock_locked"))
            lease_valid = locked or not summary.get("lock_mhz")
            admitted = summary["status"] == "PASS" and not contended and lease_valid and observed is not None
            reason = ("admitted" if admitted else "contention" if contended else
                      "unhealthy clock lease" if not lease_valid else
                      "no GPU window record" if observed is None else "run status " + summary["status"])
            receipts = ranks(directory, f"{tag}-time-both")
            runs.append({"tag": tag, "directory": directory, "summary": summary, "admitted": admitted,
                         "reason": reason, "contended": contended, "clock_locked": locked,
                         "lock_mhz": summary.get("lock_mhz") or None, "rank0": receipts[0] if receipts else None,
                         })
    return runs


def timing_cell(run, cell):
    pair = cell["whole_step"]
    placement = cell.get("upstream_pair")
    linked_ms = pair_medians(pair)[0]
    upstream_ms = pair_medians(placement)[1] if placement else None
    return {"tag": run["tag"], "records_root": run["directory"].name, "cell": cell["cell"],
            "tokens": cell["shape"]["tokens"], "admitted": True, "status": cell["status"],
            "clock_locked": run["clock_locked"], "lock_mhz": run["lock_mhz"],
            "speedup_vs_one_tu": pair["ratio_onetu_over_linked"], "linked_ms": linked_ms,
            "min_batch_ms": min(ms for block in pair["blocks"] for ms in block["milliseconds"]),
            "speedup_vs_upstream": placement["ratio_onetu_over_linked"] if placement else None,
            "upstream_ms": upstream_ms, "minimum_speedup": cell["parity_gate"]["minimum_onetu_over_linked"],
            "images": cell.get("images"), "config": cell["config"], "protocol": pair.get("protocol"),
            "statistic": pair.get("statistic"), "repetitions": pair["repetitions"]}


def timing(args):
    runs = timed_runs(args.runs)
    chosen, excluded = {}, []
    # Directories are scanned in the given order and tags in name order; a later admitted
    # run of the same cell and token count supersedes an earlier one.
    for run in runs:
        cells = run["rank0"]["cells"] if run["rank0"] else []
        keys = [(c["cell"], c["shape"]["tokens"]) for c in cells]
        if not run["admitted"]:
            excluded.append({"tag": run["tag"], "cells": [f"{c},{t}" for c, t in keys], "reason": run["reason"],
                             "contended": run["contended"], "clock_locked": run["clock_locked"]})
            continue
        for cell in cells:
            chosen[(cell["cell"], cell["shape"]["tokens"])] = timing_cell(run, cell)
    if not chosen:
        raise SystemExit(f"no admitted time receipts in {args.runs}")
    order = {p: i for i, p in enumerate(("fwd", "bwd", "rec"))}
    cells = sorted(chosen.values(), key=lambda c: (order[c["cell"].split("-")[1]], c["cell"], c["tokens"]))
    save(Path(args.evidence) / "timing-summary.json", {"cells": cells, "excluded": excluded})


def swap(args):
    numeric_rows = []
    for report in ranks(args.checks, args.tag):
        cases = report["cells"]
        numeric_rows.append({"rank": report["rank"], "status": report["status"], "cases": len(cases),
                             "numeric_pass": sum(all(r["passed"] for image in c["reference"].values()
                                                     for r in image.values()) for c in cases),
                             "bitwise_pass": sum(all(c["linked_onetu_bitwise"].values()) for c in cases)})
    if not numeric_rows:
        raise SystemExit(f"no {args.tag}-rank*.json receipts in {args.checks}")
    # The tag may be a prefix (retried runs append -a<attempt>); the last admitted attempt wins.
    admitted = [r for r in timed_runs([args.timing]) if r["admitted"] and r["tag"].startswith(args.timing_tag)]
    if not admitted:
        raise SystemExit(f"no admitted {args.timing_tag}* time receipt in {args.timing}")
    run = admitted[-1]
    cell = run["rank0"]["cells"][0]
    timed = timing_cell(run, cell)
    parity = {"cell": timed["cell"], "tokens_per_rank": timed["tokens"], "status": timed["status"],
              "lock_mhz": run["lock_mhz"], "clock_locked": run["clock_locked"], "contended": run["contended"],
              "images": timed["images"], "ratio_onetu_over_linked": timed["speedup_vs_one_tu"],
              "minimum_speedup": timed["minimum_speedup"],
              "placement_note": "Original upstream SwiGLU is not the replacement owner correctness oracle; "
                                "only linked/one-TU is the timing gate."}
    save(Path(args.evidence) / "owner-swap/gpu-results.json",
         {"status": "PASS" if all(r["status"] == "PASS" for r in numeric_rows) and timed["status"] == "PASS" else "FAIL",
          "scope": "EP4 forward/backward, both precisions and clamp branches, "
                   + " plus ".join(sorted({c["shape"]["name"] for c in cases}, key=lambda n: n != "sanity"))
                   + ", three input generations.",
          "numeric": numeric_rows, "parity": parity})


def facts_gpu(args):
    rows = []
    for item in args.tag:
        label, _, tag = item.partition("=")
        tag = tag or label
        summary = load(Path(args.runs) / f"{tag}-check-summary.json")
        observed = window(args.runs, tag, "check")
        reports = ranks(args.runs, f"{tag}-check-linked")
        worker = summary["runs"][0]["returncode"] if summary["runs"] else None
        errors = [r["error"] for r in reports if r.get("error")]
        if worker == 124:
            status = "WATCHDOG TIMEOUT"
        elif errors and all("provider mok.gemm query failed" in e for e in errors) and len(errors) == len(reports):
            status = "EXPECTED PROVIDER REJECTION"
        else:
            status = summary["status"]
        row = {"label": label, "returncode": observed["returncode"] if observed else None,
               "gpu_window": [line for line in observed["stderr"].splitlines() if WINDOW.search(line)] if observed else [],
               "rank_statuses": [r["status"] for r in reports], "cells_per_rank": [len(r["cells"]) for r in reports],
               "worker_returncode": worker, "status": status}
        if errors:
            row["rank_errors"] = errors
        rows.append(row)
    save(Path(args.evidence) / "checked-facts/gpu-runs.json", {"runs": rows})


def placement(args):
    cells = []
    for directory in map(Path, args.bench):
        run = load(directory / "run.json")
        if "attempt" in run:
            # host/measure_upstream.py layout: bench-N/run.json names the admitted attempt-K/.
            directory = directory / f"attempt-{run['attempt']}"
        log = (directory / "torchrun.log").read_text()
        window_log = directory / "window.log"
        summaries = WINDOW.findall(window_log.read_text()) if window_log.exists() else []
        contended = bool(summaries) and summaries[-1][2] == "1"
        run = {**load(directory / "run.json"), **run}
        locked = bool(run.get("clock_locked"))
        admitted = run["exit_code"] == 0 and bool(summaries) and not contended and (locked or run.get("requested_sm_mhz") is None)
        tokens = int(re.search(r"tokens/rank=(\d+)", log).group(1))
        for name, forward, backward in re.findall(
                r"^(.+?): forward ([0-9.]+) ms, [0-9.]+ TFLOP/s; backward ([0-9.]+) ms", log, re.MULTILINE):
            cells.append({"implementation": name if name.startswith("NCCL") else f"Upstream MoK {name}",
                          "tokens": tokens, "forward_ms": float(forward), "backward_ms": float(backward),
                          "admitted": admitted, "status": "PASS" if admitted else "NOT ADMITTED",
                          "clock_locked": locked, "lock_mhz": run.get("requested_sm_mhz"),
                          "power_cap_samples": None, "telemetry": []})
    if not cells:
        raise SystemExit(f"no benchmark lines in {args.bench}")
    order = {"Upstream MoK BF16": 0, "Upstream MoK MXFP8": 1}
    cells.sort(key=lambda c: (order.get(c["implementation"], 2), c["implementation"], c["tokens"]))
    save(Path(args.evidence) / "placement-summary.json", {"cells": cells})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, required=True, help="report input directory to write into")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("numeric")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--check", nargs="+", required=True)
    p.add_argument("--sanitize", required=True)
    p.set_defaults(func=numeric)
    p = sub.add_parser("timing")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.set_defaults(func=timing)
    p = sub.add_parser("swap")
    p.add_argument("--checks", type=Path, required=True)
    p.add_argument("--tag", default="swap-check")
    p.add_argument("--timing", type=Path, required=True, help="run.py time receipt directory")
    p.add_argument("--timing-tag", default="swap-time", help="run.py --tag (or prefix) of the timed run")
    p.set_defaults(func=swap)
    p = sub.add_parser("facts-gpu")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--tag", nargs="+", required=True, metavar="LABEL[=TAG]")
    p.set_defaults(func=facts_gpu)
    p = sub.add_parser("placement")
    p.add_argument("--bench", type=Path, nargs="+", required=True,
                   help="benchmark_upstream.sh LOG_DIRs (run.json, torchrun.log, window.log with the "
                        "gpu_window.sh summary) or host/measure_upstream.py bench-N directories")
    p.set_defaults(func=placement)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
