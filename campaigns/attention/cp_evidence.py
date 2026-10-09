"""Zigzag context-parallel (CP) causal attention evidence on sm90 (pipeline d128-sm90-c/cp).

Runs the multi-device CP slot (out/bin/attention-slot-cp-sm90) in one process over
the physical GPUs given by --gpus (world = number of GPUs; rank r = r-th GPU) and
writes results/sm90-cp-<stage>-tp<W>-<cell>.json:

  gate       >= 5 changing payloads (linked then control, skewed-rank starts) plus
             graph replays; every launch's O/LSE vs the fp32 boundary-row oracle on
             the full sequence, linked/control bitwise (Gate A/B/C).
  sanitize   racecheck + synccheck on the linked and the control image.
  parity     linked/control at the all-rank timing boundary:
             step = max_r(elapsed_r)/n, alternating balanced blocks, pass >= 0.97.
  placement  descriptive: CP linked vs pruned peer-copy gather + two `selected`
             segment launches vs single-GPU full-sequence `selected` (rank 0).
  te         descriptive: a fresh `placement` plus Transformer Engine sequence-only
             CP (p2p and all_gather) on the same GPUs under the same clock lease;
             needs cp-baseline/setup.sh.
  report     renders zigzag receipts, balance/ideal-speedup diagnostics,
             and analytical payload bytes.

GPU commands default to unlocked clocks (recorded before/after) and accept
--clock-mhz N for a monitored gpulockd lease, exactly as sm90_evidence.py does.
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


def stage_gate(cell: str, gpus: list[int]) -> dict[str, Any]:
    run, result = run_slot(slot_command("gate", cell, len(gpus)), gpus)
    passed = run["returncode"] == 0 and bool(result and result.get("correctness", {}).get("all_true"))
    return {"status": "PASS" if passed else "FAIL", "slot_result": result, "command": run}


def stage_sanitize(cell: str, gpus: list[int]) -> dict[str, Any]:
    sanitizer = Path(os.environ["PIPELINK_CUDA_HOME"]) / "bin" / "compute-sanitizer"
    checks = {}
    for mode in ("linked", "control"):
        for tool in ("racecheck", "synccheck"):
            command = [str(sanitizer), "--tool", tool, "--error-exitcode", "99",
                       *slot_command("sanitize", cell, len(gpus), [f"--mode={mode}"])]
            run, result = run_slot(command, gpus)
            count = ev.sanitizer_count(run["stdout"] + "\n" + run["stderr"], tool)
            passed = (run["returncode"] == 0 and count == 0 and isinstance(result, dict)
                      and result.get("images_executed") == [mode])
            checks[f"{mode}/{tool}"] = {"passed": passed, "count": count, "slot_result": result, "command": run}
    return {"status": "PASS" if all(row["passed"] for row in checks.values()) else "FAIL", "checks": checks}


def stage_timed(stage: str, cell: str, gpus: list[int], clock_mhz: int | None) -> dict[str, Any]:
    with ev.timed_run(gpus, clock_mhz) as (clock, commands):
        run, result = run_slot(slot_command(stage, cell, len(gpus)), gpus)
    ok = run["returncode"] == 0 and isinstance(result, dict)
    status = ("PASS" if result.get("verdict") == "PASS" else "FAIL") if ok and stage == "parity" else \
        ("COMPLETE" if ok else "FAIL")
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
             "Each receipt runs `placement` and both TE transports on the same GPUs under one clock lease.",
             "TE time = median CUDA-graph replay of its full forward (communication, attention, correction);",
             "every interval starts after an NCCL collective aligns the ranks and takes the slowest rank.",
             "TE output is checked against the FP32 oracle at both sides of every chunk edge, all heads.",
             "Ratio = faster TE transport / PipeLink CP linked (above 1: PipeLink faster).", "",
             "| Cell | TP | Clock | Status | PipeLink ms | TE p2p ms | TE all-gather ms | TE / PipeLink |",
             "|---|---:|---|---|---:|---:|---:|---:|"]
    for row in data:
        if row["stage"] != "te":
            continue
        cp = ((row.get("slot_result") or {}).get("cp_linked") or {}).get("median_step_ms")
        te = [((row["te"][t].get("measurement") or {}).get("timing") or {}).get("median_ms") for t in TE_TRANSPORTS]
        ratio = min(te) / cp if row["status"] == "PASS" else None
        cells = [fmt(value, 3) if value is not None else "—" for value in (cp, *te, ratio)]
        lines.append(f"| {row['cell']} | {row['world']} | {row.get('clock_mhz') or 'unlocked'} | {row['status']} | "
                     + " | ".join(cells) + " |")
    return lines


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def report() -> Path:
    rows = sorted(ev.RESULTS.glob("sm90-cp-*-tp*-*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in rows]
    data = [row for row in records if row.get("sharding") == "zigzag"]
    lines = ["# SM90 zigzag context-parallel attention (pipeline `cp`)", "",
             "Generated by `cp_evidence.py report` from zigzag-tagged `results/sm90-cp-*.json`.",
             f"Excluded {len(records) - len(data)} untagged/non-zigzag receipts from current tables.",
             "Missing rows are not passes.", "",
             "Rank r owns chunks r and 2W−1−r of 2W equal chunks; local rows are early then late.",
             "The sequential baseline gathers exactly needed chunks and runs unchanged `selected`",
             "twice per rank, including both scheduler resets in the timed boundary.",
             "TP3 W1-8k is skipped: 8192 is not divisible by 6·128.", ""]
    lines += ["## Correctness", "", "| Stage | Cell | TP | Status | Detail |", "|---|---|---:|---|---|"]
    for row in data:
        if row["stage"] == "gate":
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
            detail = ", ".join(f"{name}={check['count']}" for name, check in row["checks"].items())
        else:
            continue
        lines.append(f"| {row['stage']} | {row['cell']} | {row['world']} | {row['status']} | {detail} |")
    lines += ["", "## Seam parity (all-rank boundary)", "",
              "Step = max over ranks of the per-rank CUDA-event interval / n; ratio = median over 10 alternating",
              "pairs of control_step / linked_step (linked/control throughput), floor 0.97.", "",
              "| Cell | TP | Clock | Linked step ms | Control step ms | Wall step ms (L/C) | Ratio | Verdict | Per-rank launch ms (linked) |",
              "|---|---:|---|---:|---:|---|---:|---|---|"]
    for row in data:
        if row["stage"] != "parity" or not row.get("slot_result"):
            continue
        r = row["slot_result"]
        lines.append(
            f"| {row['cell']} | {row['world']} | {row['clock']} {row.get('clock_mhz') or ''} | "
            f"{fmt(r['linked']['median_step_ms'], 3)} | {fmt(r['control']['median_step_ms'], 3)} | "
            f"{fmt(r['linked']['median_wall_step_ms'], 3)} / {fmt(r['control']['median_wall_step_ms'], 3)} | "
            f"{r['ratio']:.4f} | {r['verdict']} | {', '.join(fmt(x, 3) for x in r['rank_launch_ms']['linked'])} |")
    lines += ["", "## Placement (descriptive)", "",
              "Speedup = single / CP; ideal efficiency = speedup / W. Balance = max/min of",
              "single-launch per-rank intervals (includes enqueue/scheduling gaps), not a",
              "decomposition of the back-to-back all-rank timing boundary. No placement gate.", "",
              "| Cell | TP | Clock | CP ms | Gather + two selected ms | Single GPU ms | "
              "Speedup | Ideal | Ideal efficiency | CP / sequential | Rank max/min | "
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
            f"| {row['cell']} | {row['world']} | {row['clock']} {row.get('clock_mhz') or ''} | {fmt(cp, 3)} | "
            f"{fmt(seq, 3)} | {fmt(single, 3)} | {single / cp:.2f}x | {row['world']}x | "
            f"{100 * single / cp / row['world']:.1f}% | {cp / seq:.3f} | "
            f"{max(rank_ms) / min(rank_ms):.3f} | "
            f"{', '.join(fmt(x, 3) for x in rank_ms)} | "
            f"{', '.join(fmt(x, 3) for x in r['rank_launch_ms']['sequential_gather_selected'])} |")
    lines += ["", "## K+V traffic accounting (analytical, per launch)", "",
              "The slot reports shape; this table gives theoretical payload minima.",
              "Counters, release flags and physical fabric overhead are excluded.",
              "Minimum = 1.5·W(W−1)·C·batch·Hkv·128·2 bytes/bf16·2 tensors;",
              "full all-gather = 2·W(W−1)·C times the same bytes/token.",
              "Sender p: (W−1+p) chunks; receiver r: (2W−2−r) chunks.", "",
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report")
    for name in ("gate", "sanitize", "parity", "placement", "te"):
        command = sub.add_parser(name)
        command.add_argument("--gpus", required=True, help="physical GPU IDs, one per rank")
        command.add_argument("--cell", default="W1-32k", help="causal cell giving the total sequence")
        command.add_argument("--clock-mhz", type=int, help="gpulockd lease at N MHz; default: unlocked")
    args = parser.parse_args()
    if args.command == "report":
        print(report())
        return 0
    gpus = [int(item) for item in args.gpus.split(",")]
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        parser.error("--gpus must name at least two distinct GPUs")
    require_inputs(args.command)
    generation = ev.build_generation()
    if args.command in ("gate", "sanitize"):
        with ev.timed_run(gpus, args.clock_mhz) as (clock, commands):
            payload = (stage_gate if args.command == "gate" else stage_sanitize)(args.cell, gpus)
        payload.update(lock_commands=commands, **clock)
    elif args.command == "te":
        payload = stage_te(args.cell, gpus, args.clock_mhz)
    else:
        payload = stage_timed(args.command, args.cell, gpus, args.clock_mhz)
    payload.update(stage=args.command, cell=args.cell, world=len(gpus), gpus=gpus,
                   build_generation=generation, sharding="zigzag")
    path = write(args.command, len(gpus), args.cell, payload)
    print(json.dumps({"stage": args.command, "status": payload["status"], "output": str(path)}))
    return 0 if payload["status"] in ("PASS", "COMPLETE") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ev.EvidenceError as error:
        print(f"cp evidence: {error}", file=sys.stderr)
        raise SystemExit(2)
