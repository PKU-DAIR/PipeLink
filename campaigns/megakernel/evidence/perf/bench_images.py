#!/usr/bin/env python3
"""Exploratory latency comparison of linked and upstream images; writes only to its own output directory.

From the repository root, compare a read-only snapshot of an earlier build (X) with the
working build (Y) on MK_GPU (default: the first of PIPELINK_GPUS)::

    source tools/env.sh
    tools/gpu_window.sh uv run --project campaigns/megakernel/host --locked python \
      campaigns/megakernel/evidence/perf/bench_images.py \
      --x "$SNAPSHOT/port-build" "$SNAPSHOT/host-build" \
      --y campaigns/megakernel/port/build campaigns/megakernel/host/build

Omit --y to record an absolute X baseline. Later, --baseline PATH/results.json
--y PORT_BUILD HOST_BUILD measures only Y; Y versus the saved X is UNPAIRED.
--pristine-cubin PATH adds unpatched upstream, using X's host/link facts (or the
saved X facts in baseline mode). It is checked numerically against the patched
mono before timing and is never launched on the multi-partition case, where
upstream is known to be broken.
For a small smoke test: --cases MK-A-16 --blocks 2 --slot-seconds 0.6
--discard-seconds 0.2 --warmups 1. Default output is
$PIPELINK_SCRATCH_ROOT/megakernel-perf/<UTC timestamp>.

Timing is mk_slot.perform_slot as is: CUDA-event batches, per-slot one-token
calibration, 50 ms target capped at 32 tokens/100 ms, leading discard and complete
steady batches. Protocol.definitive supplies the defaults (8 blocks, 3 s slots, 1 s
discard, 5 check-mode warmups). Blocks alternate forward/reverse order: AB/BA for
two images, ABCD/DCBA for two sets, and likewise with pristine. Every block measures
each admitted image once, and every paired speedup is the median of within-block
reference_ms/candidate_ms, so >1 means the candidate is faster. Absolute medians are
medians of per-slot ms/token.

Limits: this is exploratory and gates nothing. There is no P2/sanitizer admission,
per-slot correctness recheck, 10 Hz telemetry/eligibility classifier, retry,
escalation or PASS threshold. Clocks are locked and checked through
tools/gpulockd.py at PIPELINK_LOCK_SM_MHZ; unset, empty or 0 runs with unlocked
clocks (no gpulockd lease; results record clock_locked false and the report says
the timings are relative only). The MK_GPU device must be exclusive: any resident
process other than this run's clock daemon aborts the run.
Pristine admission uses two full-bundle runs per image and compare.py's repeat-floor
gate, including each run's distance to the PyVM reference. The floor comes from the
patched mono repeats only, so pristine variation cannot widen its own threshold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"),
                str(ROOT / "campaigns/megakernel/host"),
                str(ROOT / "campaigns/megakernel/evidence/parity")]
from pipelink_env import load
from launch import launch_command
import mk_slot
import run_parity as parity
from compare import OUTPUT_NAMES, aggregate_pair_metrics, decode, load_raw, pair_metrics, parse_manifest
from evaluator.harness.parity import Protocol

GPU = parity.GPU
C1_BANNER = parity.C1_BANNER
CASES = {case["case_id"]: case for case in parity.CASES if case["pipeline"] == "a"}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_processes(allowed_pid: int | None = None) -> list[int]:
    result = subprocess.run(["nvidia-smi", "-i", str(GPU),
                             "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    pids = [int(row.strip()) for row in result.stdout.splitlines() if row.strip()]
    if any(pid != allowed_pid for pid in pids):
        raise RuntimeError(f"GPU {GPU} is occupied: {pids}; allowed clock daemon={allowed_pid}")
    return pids


def lockd(action: str, *extra: str) -> dict[str, Any]:
    record = parity.run_command([sys.executable, str(ROOT / "tools/gpulockd.py"), action,
                                 "--gpus", str(GPU), *extra])
    record["action"] = action
    return record


def lock_marker() -> Path:
    return Path(os.environ["PIPELINK_LOCK_STATE_DIR"]).expanduser() / f"gpu{GPU}.json"


def start_lock(mhz: int) -> dict[str, Any]:
    record = lockd("start", "--sm-mhz", str(mhz))
    if record["returncode"]:
        raise RuntimeError(f"gpulockd start failed: {record['stderr'].strip()}")
    return record


def lock_daemon_pid(mhz: int) -> int | None:
    if not mhz:
        return None
    return json.loads(lock_marker().read_text())["pid"]


def check_lock(label: str, mhz: int) -> dict[str, Any]:
    record = lockd("status")
    record["label"] = label
    if record["returncode"]:
        raise RuntimeError(f"gpulockd status failed during {label}: {record['stderr'].strip()}")
    if mhz:
        marker = json.loads(lock_marker().read_text())
        if marker.get("sm_mhz") != mhz or marker.get("violations") != 0:
            raise RuntimeError(f"clock lock not held during {label}: {marker}")
        record["marker"] = marker
    return record


def image_set(paths: list[str]) -> dict[str, Any]:
    port, host = (Path(value).expanduser().resolve() for value in paths)
    files = [port / "linked/a/link.json", port / "linked/a/pipeline.cubin",
             port / "mono.cubin", host / "mk_runner", host / "provider.so"]
    return {"port_build": str(port), "host_build": str(host),
            "sha256": {str(path): parity.sha256_file(path) for path in files}}


def command_for(images: dict[str, Any], *, pristine: Path | None = None):
    launch = partial(launch_command, port_build=Path(images["port_build"]),
                     host_build=Path(images["host_build"]))
    if pristine is None:
        return launch

    def unpatched(*args, **kwargs):
        command = launch(*args, **kwargs)
        command[command.index("--cubin") + 1] = str(pristine)
        return command
    return unpatched


def summary(case: dict[str, Any], saved: dict[str, Any] | None) -> dict[str, Any]:
    blocks = case["blocks"]
    keys = list(blocks[0]["latency_ms_per_token"])
    medians = {key: statistics.median(b["latency_ms_per_token"][key] for b in blocks) for key in keys}
    speedups = {}
    candidates = [(f"{s}.linked", f"{s}.mono") for s in ("x", "y")]
    candidates += [(f"y.{kind}", f"x.{kind}") for kind in ("linked", "mono")]
    candidates += [("pristine.mono", "x.mono")]
    candidates += [(f"{s}.linked", "pristine.mono") for s in ("x", "y")]
    for candidate, reference in candidates:
        if candidate in keys and reference in keys:
            values = [b["latency_ms_per_token"][reference] / b["latency_ms_per_token"][candidate] for b in blocks]
            speedups[f"{candidate}_vs_{reference}"] = {"median": statistics.median(values), "paired": True, "values": values}
    if saved:
        saved_summary = summary(saved, None)
        for key, value in saved_summary["median_latency_ms_per_token"].items():
            if key.startswith("x."):
                medians[key] = value
        speedups.update({name: value for name, value in saved_summary["speedups"].items()
                         if name == "x.linked_vs_x.mono"})
        comparisons = [(f"y.{kind}", f"x.{kind}") for kind in ("linked", "mono")]
        if "pristine.mono" in medians:
            comparisons += [("pristine.mono", "x.mono"), ("x.linked", "pristine.mono")]
        for candidate, reference in comparisons:
            speedups[f"{candidate}_vs_{reference}"] = {
                "median": medians[reference] / medians[candidate], "paired": False}
    return {"median_latency_ms_per_token": medians, "speedups": speedups}


def markdown(payload: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    headings = ["Case", "X linked ms", "X mono ms", "X linked vs mono speedup", "Y linked ms", "Y mono ms", "Y linked vs mono speedup", "Y vs X linked speedup", "Y vs X mono speedup", "Pristine ms", "Pristine vs X mono speedup", "X linked vs pristine speedup", "Y linked vs pristine speedup"]
    rows = ([C1_BANNER, ""] if not payload.get("clock_locked") else []) + ["# Exploratory image latency benchmark", "", "Speedup = reference_ms/candidate_ms; >1 means the candidate is faster. Paired values are medians of per-block speedups; comparisons against a saved baseline are unpaired ratios of absolute medians.", "", "| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
    for case in payload["cases"]:
        if "summary" not in case:
            continue
        historical = next((c for c in baseline["cases"] if c["case_id"] == case["case_id"]), None) if baseline else None
        case_summary = summary(case, historical)
        med = case_summary["median_latency_ms_per_token"]
        speedups = case_summary["speedups"]
        def m(key):
            return f"{med[key]:.6f}" if key in med else "—"
        def s(key):
            if key not in speedups:
                return "—"
            return f"{speedups[key]['median']:.5f}" + (" (unpaired)" if not speedups[key]["paired"] else "")
        rows.append("| " + " | ".join([case["case_id"], m("x.linked"), m("x.mono"), s("x.linked_vs_x.mono"), m("y.linked"), m("y.mono"), s("y.linked_vs_y.mono"), s("y.linked_vs_x.linked"), s("y.mono_vs_x.mono"), m("pristine.mono"), s("pristine.mono_vs_x.mono"), s("x.linked_vs_pristine.mono"), s("y.linked_vs_pristine.mono")]) + " |")
    if "pristine" in payload:
        rows += ["", "## Pristine numerical admission", "",
                 "Two full-bundle runs per image; compare.py uses the patched-mono repeat floor for the upstream-equivalence and PyVM reference checks. Failed cases are not timed."]
        for case in payload["cases"]:
            admission = case.get("pristine_admission")
            if admission is None:
                continue
            failures = "; ".join(admission.get("comparison", {}).get("failures", []))
            detail = admission.get("error") or admission.get("reason") or failures or "repeat-floor-relative gate passed"
            rows.append(f"- {case['case_id']}: {admission['status']} — {detail}")
    rows += ["", "## Protocol deviations", *[f"- {text}" for text in protocol_deviations(payload["protocol"], bool(payload.get("baseline")))], "", f"Wall seconds: {payload.get('wall_seconds', 0):.2f}", f"Error: {payload.get('error')}"]
    return "\n".join(rows) + "\n"


def protocol_deviations(protocol: dict[str, Any], historical: bool) -> list[str]:
    deviations = [
        "Exploratory: no P2/sanitizer admission, per-slot correctness recheck, 10 Hz telemetry eligibility, retries, escalation or PASS threshold.",
        "Forward/reverse multi-image blocks; median paired speedups (reference_ms/candidate_ms), higher is better.",
        "Timing is mk_slot.perform_slot as is, including the runner pre-discard and scoring-window discard.",
        f"Requested {protocol['pair_valid_target']} blocks, {protocol['slot_seconds']}s slots, {protocol['leading_discard_seconds']}s discard, {protocol['warmups_per_image']} warmups; definitive defaults are 8/3/1/5.",
    ]
    if historical:
        deviations.append("Saved baseline: Y versus X speedups are unpaired reference/candidate ratios of absolute medians from separate runs.")
    return deviations


def main() -> int:
    os.environ.update(load(ROOT))
    MHZ = int(os.environ.get("PIPELINK_LOCK_SM_MHZ") or 0)  # 0: unlocked
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--x", nargs=2, metavar=("PORT_BUILD", "HOST_BUILD"))
    source.add_argument("--baseline", type=Path)
    parser.add_argument("--y", nargs=2, metavar=("PORT_BUILD", "HOST_BUILD"))
    parser.add_argument("--pristine-cubin", type=Path)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--slot-seconds", type=float)
    parser.add_argument("--discard-seconds", type=float)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.baseline and not args.y:
        parser.error("--baseline requires --y")
    if GPU is None:
        parser.error("set MK_GPU or PIPELINK_GPUS (source tools/env.sh)")
    out = args.out or Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "megakernel-perf" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = out.resolve()
    saved = json.loads(args.baseline.read_text()) if args.baseline else None
    if saved and (saved.get("error") or any(name not in {c["case_id"] for c in saved["cases"]} for name in args.cases)):
        parser.error("baseline failed or does not contain all requested cases")
    sets = {}
    if args.x:
        sets["x"] = image_set(args.x)
    if args.y:
        sets["y"] = image_set(args.y)
    definitive = Protocol.definitive(out, parity.case_config(CASES[args.cases[0]]))
    blocks = args.blocks if args.blocks is not None else definitive.valid_pairs
    slot = args.slot_seconds if args.slot_seconds is not None else definitive.slot_seconds
    discard = args.discard_seconds if args.discard_seconds is not None else definitive.discard_seconds
    warmups = args.warmups if args.warmups is not None else definitive.warmups_per_image
    if blocks < 2 or blocks % 2 or warmups < 0 or not 0 <= discard < slot:
        parser.error("require positive even --blocks, nonnegative --warmups, and 0 <= discard < slot")
    protocol = definitive.record()
    protocol.update(name="exploratory-image-perf", protocol_variant="exploratory-image-perf-v1",
                    pair_valid_target=blocks, slot_seconds=slot, leading_discard_seconds=discard,
                    steady_seconds=slot-discard, warmups_per_image=warmups,
                    selection_classifier="none", clock_sample_hz=None)
    out.mkdir(parents=True, exist_ok=False)
    payload = {"schema_version": "megakernel-image-perf-v2", "generated_at": parity.utc_now(),
               "gpu": GPU, "target_sm_mhz": MHZ or None, "clock_locked": bool(MHZ),
               "protocol": protocol, "image_sets": dict(sets),
               "baseline": str(args.baseline.resolve()) if args.baseline else None,
               "cases": [], "lock": {}, "error": None,
               "deviations": protocol_deviations(protocol, saved is not None)}
    if saved:
        payload["image_sets"]["historical_x"] = saved["image_sets"]["x"]
    if args.pristine_cubin:
        args.pristine_cubin = args.pristine_cubin.resolve()
        payload["pristine"] = {"cubin": str(args.pristine_cubin), "sha256": parity.sha256_file(args.pristine_cubin)}
        provenance = args.pristine_cubin.parent / "build.json"
        if provenance.exists():
            payload["pristine"]["build"] = json.loads(provenance.read_text())
    parity.GPU = GPU
    daemon_pid = None
    started = False
    start = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    os.environ["MK_WATCHDOG_SECONDS"] = "30"

    def run(command, label, *, timeout=120.0):
        gpu_processes(daemon_pid)
        Path(command[command.index("--out") + 1]).mkdir(parents=True, exist_ok=True)
        completed = mk_slot.run_checked(command, label, timeout=timeout)
        return completed

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)

    def save():
        payload["wall_seconds"] = time.monotonic() - start
        write_json(out / "results.json", payload)
        (out / "RESULT.md").write_text(markdown(payload, saved))

    try:
        gpu_processes()
        payload["lock"]["start"] = start_lock(MHZ)
        started = True
        daemon_pid = lock_daemon_pid(MHZ)
        gpu_processes(daemon_pid)
        payload["lock"]["initial"] = check_lock("initial", MHZ)
        for name in args.cases:
            config = CASES[name]
            parity.case_config(config)
            bundle = parity.BUNDLES / config["bundle"]
            case = {"case_id": name, "bundle": str(bundle), "pos_id": config["pos_id"], "blocks": []}
            payload["cases"].append(case)
            launchers = {f"{s}.{kind}": (kind, command_for(images)) for s, images in sets.items() for kind in ("linked", "mono")}
            case_dir = out / name
            case_dir.mkdir()
            if args.pristine_cubin:
                case["pristine_admission"] = {"status": "SKIP", "reason": "known upstream multi-partition defects"}
                if not config["bundle"].endswith("-red"):
                    x = sets.get("x") or saved["image_sets"]["x"]
                    pristine_launch = command_for(x, pristine=args.pristine_cubin)
                    admission = {"status": "FAIL"}
                    case["pristine_admission"] = admission
                    try:
                        for label, launch in [("patched", command_for(x)), ("pristine", pristine_launch)]:
                            for repeat in (1, 2):
                                run(launch("a", "mono", bundle, case_dir / f"{label}-{repeat}", "check"), f"{label} repeat {repeat} numerical admission")
                        comparison_path = case_dir / "pristine-comparison.json"
                        patched1, patched2 = (str(case_dir / f"patched-{n}") for n in (1, 2))
                        command = [
                            sys.executable, str(ROOT / "campaigns/megakernel/host/compare.py"),
                            "--bundle", str(bundle), "--linked-run", patched1,
                            "--linked-run2", patched2, "--onetu-run", patched1,
                            "--mono-run", patched1, "--mono-run2", patched2,
                            "--extra-run", f"pristine1={case_dir / 'pristine-1'}",
                            "--extra-run", f"pristine2={case_dir / 'pristine-2'}",
                            "--gate-extra", "--out", str(comparison_path),
                        ]
                        compared = subprocess.run(command, capture_output=True, text=True, timeout=120)
                        (case_dir / "pristine-comparison.log").write_text(compared.stdout + compared.stderr)
                        admission["command"] = command
                        if compared.returncode not in (0, 1) or not comparison_path.exists():
                            raise RuntimeError(f"compare.py failed: {compared.stderr}")
                        admission["comparison"] = json.loads(comparison_path.read_text())
                        admission["status"] = admission["comparison"]["status"]
                        admission["floor_source"] = "patched mono repeats only (same runs supplied for linked/mono roles)"
                        specs = parse_manifest(bundle)
                        admission["pristine_repeat"] = {}
                        for tensor in OUTPUT_NAMES:
                            metrics = []
                            for token in range(admission["comparison"]["ntok"]):
                                spec = specs[f"ref_tok{token}_{tensor}"]
                                first, second = (
                                    decode(load_raw(case_dir / f"pristine-{n}" / f"tok{token}_{tensor}.bin", spec), spec.dtype)
                                    for n in (1, 2)
                                )
                                metrics.append(pair_metrics(first, second))
                            admission["pristine_repeat"][tensor] = aggregate_pair_metrics(metrics)
                    except (RuntimeError, subprocess.SubprocessError) as exc:
                        admission["error"] = str(exc)
                    if admission["status"] == "PASS":
                        launchers["pristine.mono"] = ("mono", pristine_launch)
            for key, (kind, launch) in launchers.items():
                for index in range(warmups):
                    run(launch("a", kind, bundle, case_dir / key / f"warmup-{index}", "check", ("--ntok", "1", "--pos-id-first", str(config["pos_id"]))), f"{key} warmup")
            for index in range(blocks):
                order = list(launchers)
                if index % 2:
                    order.reverse()
                block = {"index": index, "order": order, "latency_ms_per_token": {}, "slots": {}}
                case["blocks"].append(block)
                for key in order:
                    kind, launch = launchers[key]
                    request = {"protocol": protocol, "case_id": name, "image": {"image_id": key}, "image_role": key}
                    result = mk_slot.perform_slot(request, argparse.Namespace(pipeline="a", image=kind, pos_id=config["pos_id"]), bundle,
                                                  case_dir / key / f"block-{index}", {"not_checked": True}, launcher=launch, run_command=run)
                    block["latency_ms_per_token"][key] = result["steady_cuda_event_milliseconds"] / result["iterations"]
                    block["slots"][key] = result
                gpu_processes(daemon_pid)
                block["clock"] = check_lock(f"{name}-block-{index}", MHZ)
                save()
            historical = next((c for c in saved["cases"] if c["case_id"] == name), None) if saved else None
            case["summary"] = summary(case, historical)
            save()
            print(name, json.dumps(case["summary"]), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if started:
            try:
                gpu_processes(daemon_pid)
                payload["lock"]["final"] = check_lock("final", MHZ)
            except Exception as exc:
                payload["error"] = payload["error"] or {"type": type(exc).__name__, "message": str(exc)}
            payload["lock"]["stop"] = lockd("stop")
            if payload["lock"]["stop"]["returncode"]:
                payload["error"] = payload["error"] or {"type": "ClockStopError", "message": str(payload["lock"]["stop"])}
        try:
            payload["remaining_gpu_processes"] = gpu_processes()
        except Exception as exc:
            payload["error"] = payload["error"] or {"type": type(exc).__name__, "message": str(exc)}
        save()
    print(markdown(payload, saved))
    print(f"Results: {out}")
    return int(payload["error"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
