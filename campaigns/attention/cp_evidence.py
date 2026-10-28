"""Zigzag context-parallel (CP) causal attention evidence on sm90 (pipeline d128-sm90-c/cp).

Runs the multi-device CP slot (out/bin/attention-slot-cp-sm90) in one process over
the physical GPUs given by --gpus (world = number of GPUs; rank r = r-th GPU) and
writes results/sm90-cp-<stage>-tp<W>-<cell>.json:

  gate       at least five changing payloads (linked then control, with skewed rank
             starts) plus graph replays. Every launch's O/LSE is checked against the
             FP32 boundary-row oracle over the full sequence, and linked/control
             must match bitwise (Gates A/B/C).
  sanitize   memcheck, racecheck and synccheck on the linked and control images.
  parity     paired-reload-v2: eight four-batch blocks of exactly ten all-rank calls
             per batch, timed as max_r(elapsed_r), with a same-context module reload
             halfway; R1 must reach 0.97.
  placement  descriptive: CP linked against a pruned peer-copy gather followed by two
             `selected` segment launches, and against single-GPU full-sequence
             `selected` on rank 0.
  te         descriptive: a new `placement` run plus Transformer Engine
             sequence-only CP (p2p and all_gather) on the same GPUs under the same
             clock lease; needs cp-baseline/setup.sh.
  report     renders the zigzag records in --results, balance and ideal-speedup
             figures, and the analytical payload bytes into <results>/sm90-cp.md
             (CPU only).

--clock-mhz defaults to $PIPELINK_LOCK_SM_MHZ; unset, empty or 0 means unlocked
clocks (observed, not leased), and receipts record `clock_locked`. parity needs
passing numerical gates for the same cell and world from the current build, and
sanitizer coverage: the W1-8k run at the same world (TP2, TP4) is the declared
representative for W1-32k (W1-32k is too slow under racecheck). Each comparison holds its own GPU window (and lease, when
locked). --results and --out-root keep new evidence and build artifacts in
separate directories.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sm90_evidence as ev  # noqa: E402  (shared commands, clock lease, sanitizer parsing)

SLOT = HERE / "out" / "bin" / "attention-slot-cp-sm90"
LINKS = HERE / "out" / "links" / ev.SPEC
CP = {"link_json": LINKS / "cp" / "link.json", "linked": LINKS / "cp" / "pipeline.cubin",
      "control": HERE / "out" / "controls" / ev.SPEC / "cp.cubin"}
SELECTED = {"link_json": LINKS / "selected" / "link.json", "linked": LINKS / "selected" / "pipeline.cubin"}


def slot_command(action: str, cell: str, world: int, extra: list[str] | None = None) -> list[str]:
    command = [str(SLOT), f"--action={action}", f"--world={world}", f"--cell={cell}",
               f"--cells-json={ev.CELLS_JSON}", f"--link-json={CP['link_json']}",
               f"--linked-cubin={CP['linked']}", f"--control-cubin={CP['control']}"]
    if action == "placement":
        command += [f"--selected-link-json={SELECTED['link_json']}", f"--selected-cubin={SELECTED['linked']}"]
    return command + (extra or [])


def run_slot(command: list[str], gpus: list[int]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    run = ev.run_command(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))})
    result = ev.slot_result(run)
    if result is not None and result.get("sharding") != "zigzag":
        raise ev.EvidenceError("CP slot is not zigzag; rebuild cp-load,kv-gather and the slot")
    return run, result


def write(stage: str, world: int, cell: str, payload: dict[str, Any]) -> Path:
    ev.RESULTS.mkdir(parents=True, exist_ok=True)
    path = ev.RESULTS / f"sm90-cp-{stage}-tp{world}-{cell}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def require_inputs(stage: str) -> None:
    needed = [SLOT, *CP.values(), *(SELECTED.values() if stage in ("placement", "te") else ())]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise ev.EvidenceError(f"missing inputs (run build.py): {missing}")

def sanitizer_scope(world: int) -> dict[str, Any]:
    return {"kind": "cp-executable-role-representative", "representative_cell": "W1-8k",
            "representative_world": world, "covered_cells": ["W1-32k"], "covered_worlds": [world],
            "images": ["linked", "control"], "tools": list(ev.SANITIZER_TOOLS),
            "native_action": "paired-check", "measured_worlds": [world],
            "qualification": "executable-role representative at the same world size"}


SCOPE_IDENTITY = ("kind", "representative_cell", "images", "tools", "native_action")


def scope_matches(coverage: Any, world: int) -> bool:
    expected = sanitizer_scope(world)
    return (isinstance(coverage, dict) and coverage.get("representative_world") == world
            and all(coverage.get(key) == expected[key] for key in SCOPE_IDENTITY))



def stage_gate(cell: str, gpus: list[int]) -> dict[str, Any]:
    run, result = run_slot(slot_command("gate", cell, len(gpus)), gpus)
    passed = run["returncode"] == 0 and bool(result and result.get("correctness", {}).get("all_true"))
    return {"status": "PASS" if passed else "FAIL", "slot_result": result, "command": run}


def stage_sanitize(cell: str, gpus: list[int], *, resume: bool = False) -> dict[str, Any]:
    sanitizer = Path(os.environ["PIPELINK_CUDA_HOME"]) / "bin" / "compute-sanitizer"
    checks = {}
    checkpoint = ev.SanitizerCheckpoint(
        ev.RESULTS / f"sm90-cp-sanitize-tp{len(gpus)}-{cell}.partial.json",
        {**artifact_provenance(), "cell": cell, "gpus": gpus, "sanitizer_protocol": "paired-reload-v2-check"},
        resume=resume)
    for tool in ev.SANITIZER_TOOLS:
        command = [str(sanitizer), "--tool", tool, "--error-exitcode", "99",
                   *slot_command("paired-check", cell, len(gpus))]
        key = f"paired-control-{tool}"
        previous = checkpoint.get(key, command)
        if previous is not None:
            checks[f"paired-control/{tool}"] = previous
            continue
        run, result = run_slot(command, gpus)
        count = ev.sanitizer_count(run["stdout"] + "\n" + run["stderr"], tool)
        passed = (run["returncode"] == 0 and count == 0 and ev.paired_check_pass(result, cell, "control")
                  and result.get("world") == len(gpus))
        checks[f"paired-control/{tool}"] = {"passed": passed, "count": count, "slot_result": result, "command": run}
        checkpoint.save(key, checks[f"paired-control/{tool}"])
    return {"status": "PASS" if all(row["passed"] for row in checks.values()) else "FAIL", "checks": checks,
            "sanitizer_protocol": "paired-reload-v2-check",
            **({"coverage": sanitizer_scope(len(gpus))} if cell == "W1-8k" else {})}


def artifact_provenance() -> dict[str, Any]:
    inputs = {"slot": SLOT, **CP, "cells": ev.CELLS_JSON}
    return {"build_generation": ev.build_generation(),
            "input_paths": {key: str(path.resolve()) for key, path in inputs.items()},
            "cells_document": ev.load_json(ev.CELLS_JSON),
            "binary_sha256": {key: ev.sha256(inputs[key]) for key in ("slot", "linked", "control")}}


def require_prior(stage: str, cell: str, world: int) -> str:
    path = ev.RESULTS / f"sm90-cp-{stage}-tp{world}-{cell}.json"
    expected_cell, expected_world = cell, world
    representative = False
    if not path.is_file() and stage == "sanitize" and cell == "W1-32k" and world in (2, 4):
        path = ev.RESULTS / f"sm90-cp-sanitize-tp{world}-W1-8k.json"
        expected_cell, representative = "W1-8k", True
    if not path.is_file():
        raise ev.EvidenceError(f"paired parity requires fresh {stage}: {path}")
    prior = ev.load_json(path)
    if (prior.get("status") != "PASS" or prior.get("cell") != expected_cell
            or prior.get("world") != expected_world
            or (stage == "sanitize" and prior.get("sanitizer_protocol") != "paired-reload-v2-check")
            or (representative and not scope_matches(prior.get("coverage"), world))
            or any(prior.get(key) != value for key, value in artifact_provenance().items())):
        raise ev.EvidenceError(f"stale or failing CP {stage}: {path}")
    return path.name


def stage_paired(cell: str, gpus: list[int], clock_mhz: int | None) -> dict[str, Any]:
    sys.path.insert(0, str(ev.REPO))
    from evaluator.harness.parity.fixed_count import run_native_pair
    prerequisites = [require_prior(stage, cell, len(gpus)) for stage in ("gate", "sanitize")]
    execution = run_native_pair(slot_command("paired", cell, len(gpus)), env=os.environ.copy(),
                                gpus=gpus, clock_mhz=clock_mhz, timeout=1800,
                                out_dir=ev.RESULTS / f"sm90-cp-paired-tp{len(gpus)}-{cell}")
    result = execution.get("payload") or {}
    correct = result.get("correctness", {})
    valid = (execution["admitted"] and result.get("cell") == cell and result.get("world") == len(gpus)
             and result.get("sharding") == "zigzag" and result.get("reference_kind") == "control"
             and all(correct.get(key) is True for key in
                     ("all_true", "checked_before_warmup", "checked_after_reload", "gate_a_bitwise",
                      "gate_b_linked", "gate_b_control", "gate_c_lse")))
    ratio = result.get("pair", {}).get("ratio_onetu_over_linked")
    verdict = "INVALID" if not valid else "PASS" if ratio >= 0.97 else "MISS"
    return {"status": "PASS" if verdict == "PASS" else "FAIL", "verdict": verdict,
            "protocol": "paired-reload-v2", "slot_result": result, "execution": execution,
            "clock": "locked" if clock_mhz else "unlocked", "clock_locked": bool(clock_mhz),
            "clock_mhz": clock_mhz, "prerequisites": prerequisites, "sanitizer_representative_scope": sanitizer_scope(len(gpus))}


def stage_timed(stage: str, cell: str, gpus: list[int], clock_mhz: int | None) -> dict[str, Any]:
    with ev.timed_run(gpus, clock_mhz) as (clock, commands):
        run, result = run_slot(slot_command(stage, cell, len(gpus)), gpus)
    ok = run["returncode"] == 0 and isinstance(result, dict)
    status = "COMPLETE" if ok else "FAIL"
    return {"status": status, "slot_result": result, "command": run, "lock_commands": commands, **clock}


TE_PROJECT = HERE / "cp-baseline"
TE_TRANSPORTS = ("p2p", "all_gather")


def stage_te(cell: str, gpus: list[int], clock_mhz: int | None) -> dict[str, Any]:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    te: dict[str, Any] = {}
    with ev.timed_run(gpus, clock_mhz) as (clock, commands), tempfile.TemporaryDirectory() as scratch:
        run, result = run_slot(slot_command("placement", cell, len(gpus)), gpus)
        for transport in TE_TRANSPORTS:
            output = Path(scratch) / f"{transport}.json"
            command = ["uv", "run", "--project", str(TE_PROJECT), "--no-sync", "torchrun", "--standalone",
                       f"--nproc-per-node={len(gpus)}", str(TE_PROJECT / "bench_te.py"), "--cell", cell,
                       f"--cells-json={ev.CELLS_JSON}", "--transport", transport, "--out", str(output)]
            te_run = ev.run_command(command, env=env)
            te[transport] = {"command": command[:-2], "returncode": te_run["returncode"]}  # minus temp --out
            if te_run["returncode"] == 0:
                te[transport]["measurement"] = json.loads(output.read_text(encoding="utf-8"))
            else:
                te[transport]["stderr"] = te_run["stderr"][-8000:]
    ok = run["returncode"] == 0 and isinstance(result, dict) and all("measurement" in row for row in te.values())
    return {"status": "PASS" if ok else "FAIL", "slot_result": result, "command": run, "te": te,
            "lock_commands": commands, **clock}


def te_report_lines(data: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Transformer Engine sequence-only CP (descriptive)", "",
             "Each record runs PipeLink `placement` and both Transformer Engine transports on the same GPUs",
             "under one clock lease. TE time is the median CUDA-graph replay of its full forward pass",
             "(communication, attention and correction). Every interval starts after an NCCL collective aligns",
             "the ranks and takes the slowest rank. TE output is checked against the FP32 oracle on both sides of",
             "every chunk edge, for all heads. The last column divides the faster TE transport's time by the",
             "PipeLink CP linked time, so values above 1 mean PipeLink is faster.", "",
             "| Cell | TP | Clock | Status | PipeLink ms | TE p2p ms | TE all-gather ms | TE / PipeLink |",
             "|---|---:|---|---|---:|---:|---:|---:|"]
    for row in data:
        if row["stage"] != "te":
            continue
        cp = ((row.get("slot_result") or {}).get("cp_linked") or {}).get("median_step_ms")
        te = [((row["te"][t].get("measurement") or {}).get("timing") or {}).get("median_ms") for t in TE_TRANSPORTS]
        ratio = min(te) / cp if row["status"] == "PASS" else None
        cells = [fmt(value, 3) if value is not None else "—" for value in (cp, *te, ratio)]
        lines.append(f"| {row['cell']} | {row['world']} | {clock_label(row)} | {row['status']} | "
                     + " | ".join(cells) + " |")
    return lines


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def clock_label(row: dict[str, Any]) -> str:
    status = ev.clock_status(row)
    return f"{row.get('clock_mhz')}" if status == "locked" and row.get("clock_mhz") else status


def report() -> Path:
    rows = sorted(path for path in ev.RESULTS.glob("sm90-cp-*-tp*-*.json")
                  if not path.name.endswith(".partial.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in rows]
    data = [row for row in records if row.get("sharding") == "zigzag"]
    parity_rows = [row for row in data if row["stage"] == "parity" and row.get("protocol") == "paired-reload-v2"]
    current_gates = {(row["cell"], row["world"], row.get("build_generation")) for row in parity_rows}
    current_generations = {row.get("build_generation") for row in parity_rows}
    timed = [ev.clock_status(row) for row in data if row["stage"] in ("parity", "placement", "te")]
    lines = ["# SM90 zigzag context-parallel attention (pipeline `cp`)", "", *ev.banner(timed),
             "`cp_evidence.py report` renders this page from the zigzag-sharded `sm90-cp-*.json` records; "
             f"{len(records) - len(data)} records without zigzag sharding were excluded. A row missing "
             "from a table has no qualifying record and counts as unproven.", "",
             "The sequence is split into 2W equal chunks for W ranks. Rank r owns chunks r and 2W−1−r and "
             "stores its rows early chunk first. The sequential baseline gathers only the chunks each rank "
             "needs and then runs the unmodified `selected` pipeline twice per rank; both scheduler resets "
             "fall inside its timed region. TP3 W1-8k is skipped because 8192 is not divisible by 6·128.", ""]
    lines += ["## Correctness", "",
              "The gate rows below come from the same build as the paired timing and cover the same cells and",
              "rank counts.",
              "The sanitizers ran on W1-8k at two and four ranks; each run stands in for W1-32k at the same",
              "rank count because it executes the same images in the same roles (W1-32k is too slow under",
              "racecheck).", "",
              "| Stage | Cell | TP | Status | Detail |", "|---|---|---:|---|---|"]
    for row in data:
        if row["stage"] == "gate":
            if (row["cell"], row["world"], row.get("build_generation")) not in current_gates:
                continue
            result = row.get("slot_result") or {}
            seq = result.get("async_sequences") or {}
            detail = (f"{result.get('synchronized_launches_per_rank')} synchronized launches/rank "
                      f"({result.get('payloads')} poisoned payloads x linked+control, "
                      f"{result.get('graph_replay_payloads')} graph-replay payloads; skewed ranks "
                      f"{result.get('skewed_ranks')}) + async direct and graph: {seq.get('launches_per_image')} "
                      f"linked + {seq.get('launches_per_image')} control each, no sync between launches, rank "
                      f"{seq.get('delayed_rank')} delayed {seq.get('delay_ms_per_launch')} ms per launch; "
                      f"{result.get('launches_per_rank')} launches/rank, gen={result.get('generation')}")
        elif row["stage"] == "sanitize":
            if (row.get("build_generation") not in current_generations
                    or row.get("sanitizer_protocol") != "paired-reload-v2-check"):
                continue
            detail = ", ".join(f"{name}={check['count']}" for name, check in row["checks"].items())
            detail += ("; paired-check path: two queued calls per image in each half, shared buffers and "
                       "module reload; numerical checks before and after")
        else:
            continue
        lines.append(f"| {row['stage']} | {row['cell']} | {row['world']} | {row['status']} | {detail} |")
    lines += ["", "## Seam parity (all-rank boundary)", "",
              "This compares the linked `cp` image (A) with its same-source control (B) under `paired-reload-v2`:",
              "eight blocks in balanced ABBA/BAAB order, each of four batches of exactly ten complete calls on",
              "all ranks. A batch's time is the slowest rank's elapsed time, max_r(elapsed_r), including the",
              "scheduler and gather resets; there is no calibrated inner loop. Halfway through, the modules are",
              "reloaded while contexts, fixtures and generation counters persist. The ratio is",
              "sqrt(median of the first four block B/A ratios × median of the last four) and must reach 0.97.",
              "The blocks balance order effects; they are not independent samples.", "",
              "Each comparison holds its own GPU window and qualified clock lease and needs passing gate and",
              "sanitizer records from the same build (see Correctness). The raw blocks and the window and clock",
              "records are kept. Fixed-count batches measure the whole all-rank call, so this section gives no",
              "per-rank launch latency.", "",
              "| Cell | TP | Clock MHz | Ratio | Blocks | Calls/batch | Admitted | Verdict |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for row in data:
        if row["stage"] != "parity" or row.get("protocol") != "paired-reload-v2":
            continue  # Records from older protocols are skipped, never rescored.
        r = row.get("slot_result") or {}
        pair = r.get("pair") or {}
        lines.append(f"| {row['cell']} | {row['world']} | {clock_label(row)} | "
                     f"{fmt(pair.get('ratio_onetu_over_linked'), 9) if pair.get('ratio_onetu_over_linked') is not None else 'unavailable'} | "
                     f"{len(pair.get('blocks', []))} | "
                     f"10 | {row['execution']['admitted']} | {row['verdict']} |")
    lines += ["", "## Placement (descriptive)", "",
              "Placement uses its own timer, separate from the paired parity run above, and has no pass/fail",
              "threshold. Speedup is single-GPU time / CP time, and ideal efficiency is speedup / W. Sequential / CP",
              "is likewise higher when CP is faster. Rank max/min compares the per-rank intervals of single",
              "launches, which include enqueue and scheduling gaps, so it indicates load balance but does not",
              "break down the all-rank time.", "",
              "| Cell | TP | Clock | CP ms | Gather + two selected ms | Single GPU ms | "
              "Speedup | Ideal | Ideal efficiency | Sequential / CP time | Rank max/min | "
              "Per-rank CP ms | Per-rank sequential ms |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in data:
        if row["stage"] != "placement" or not row.get("slot_result"):
            continue
        r = row["slot_result"]
        cp = r["cp_linked"]["median_step_ms"]
        seq = r["sequential_gather_selected"]["median_step_ms"]
        single = r["single_gpu_selected"]["absolute_median_ms"]
        rank_ms = r["rank_launch_ms"]["cp_linked"]
        lines.append(
            f"| {row['cell']} | {row['world']} | {ev.clock_status(row)} {row.get('clock_mhz') or ''} | {fmt(cp, 3)} | "
            f"{fmt(seq, 3)} | {fmt(single, 3)} | {single / cp:.2f}x | {row['world']}x | "
            f"{100 * single / cp / row['world']:.1f}% | {seq / cp:.3f} | "
            f"{max(rank_ms) / min(rank_ms):.3f} | "
            f"{', '.join(fmt(x, 3) for x in rank_ms)} | "
            f"{', '.join(fmt(x, 3) for x in r['rank_launch_ms']['sequential_gather_selected'])} |")
    lines += ["", "## K+V traffic accounting (analytical, per launch)", "",
              "These byte counts are computed from the shapes the slot reports, not measured. They give the",
              "theoretical K+V payload and exclude counters, release flags and fabric overhead. With C tokens",
              "per chunk, each token carries batch·Hkv·128 elements × 2 bytes (bf16) × 2 tensors (K and V):",
              "",
              "- minimum = 1.5·W(W−1)·C tokens;",
              "- full all-gather = 2·W(W−1)·C tokens;",
              "- sender p sends (W−1+p) chunks and receiver r receives (2W−2−r) chunks.", "",
              "| Cell | TP | Chunk tokens | Minimum MiB | Full all-gather MiB | Minimum / full | "
              "Per-rank send MiB | Per-rank receive MiB |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for row in data:
        if row["stage"] != "placement" or not row.get("slot_result"):
            continue
        r = row["slot_result"]
        world, chunk = row["world"], r["chunk_tokens"]
        chunk_bytes = chunk * r["batch"] * r["heads_kv"] * 128 * 2 * 2
        minimum = 3 * world * (world - 1) * chunk_bytes // 2
        full = 2 * world * (world - 1) * chunk_bytes
        send = ", ".join(fmt((world - 1 + p) * chunk_bytes / 2**20, 3) for p in range(world))
        receive = ", ".join(fmt((2 * world - 2 - p) * chunk_bytes / 2**20, 3) for p in range(world))
        lines.append(f"| {row['cell']} | {world} | {chunk} | {minimum / 2**20:.3f} | "
                     f"{full / 2**20:.3f} | {minimum / full:.2f} | {send} | {receive} |")
    lines += te_report_lines(data)
    path = ev.RESULTS / "sm90-cp.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    global SLOT, LINKS, CP, SELECTED
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--results", type=Path, required=True,
                               help="directory holding the sm90-cp-*.json records; sm90-cp.md is written there")
    for name in ("gate", "sanitize", "parity", "placement", "te"):
        command = sub.add_parser(name)
        command.add_argument("--gpus", required=True, help="physical GPU IDs, one per rank")
        command.add_argument("--cell", default="W1-32k", help="causal cell giving the total sequence")
        command.add_argument("--clock-mhz", type=int,
                             help="gpulockd lease at N MHz; 0 = unlocked; "
                                  "default: PIPELINK_LOCK_SM_MHZ (unlocked when unset, empty or 0)")
        command.add_argument("--results", type=Path, default=ev.RESULTS, help="output and prerequisite evidence directory")
        command.add_argument("--out-root", type=Path, default=ev.OUT, help="artifact tree including build-report.json")
        if name == "sanitize":
            command.add_argument("--resume", action="store_true", help="reuse completed checkpoint legs with identical provenance")
    args = parser.parse_args()
    ev.RESULTS = args.results.resolve()
    if args.command == "report":
        print(report())
        return 0
    gpus = [int(item) for item in args.gpus.split(",")]
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        parser.error("--gpus must name at least two distinct GPUs")
    clock_mhz = args.clock_mhz if args.clock_mhz is not None else ev.default_clock_mhz()
    if clock_mhz < 0:
        parser.error("--clock-mhz must be a positive MHz value, or 0 for unlocked")
    clock_mhz = clock_mhz or None
    ev.OUT = args.out_root.resolve()
    SLOT = ev.OUT / "bin" / "attention-slot-cp-sm90"
    LINKS = ev.OUT / "links" / ev.SPEC
    CP = {"link_json": LINKS / "cp" / "link.json", "linked": LINKS / "cp" / "pipeline.cubin",
          "control": ev.OUT / "controls" / ev.SPEC / "cp.cubin"}
    SELECTED = {"link_json": LINKS / "selected" / "link.json", "linked": LINKS / "selected" / "pipeline.cubin"}
    require_inputs(args.command)
    generation = ev.build_generation()
    if args.command in ("gate", "sanitize"):
        with ev.timed_run(gpus, clock_mhz) as (clock, commands):
            payload = (stage_gate(args.cell, gpus) if args.command == "gate"
                       else stage_sanitize(args.cell, gpus, resume=args.resume))
        payload.update(lock_commands=commands, **clock)
    elif args.command == "te":
        payload = stage_te(args.cell, gpus, clock_mhz)
    elif args.command == "parity":
        payload = stage_paired(args.cell, gpus, clock_mhz)
    else:
        payload = stage_timed(args.command, args.cell, gpus, clock_mhz)
    payload.update(stage=args.command, cell=args.cell, world=len(gpus), gpus=gpus,
                   build_generation=generation, sharding="zigzag")
    payload.update(artifact_provenance())
    path = write(args.command, len(gpus), args.cell, payload)
    print(json.dumps({"stage": args.command, "status": payload["status"], "output": str(path)}))
    return 0 if payload["status"] in ("PASS", "COMPLETE") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ev.EvidenceError as error:
        print(f"cp evidence: {error}", file=sys.stderr)
        raise SystemExit(2)
