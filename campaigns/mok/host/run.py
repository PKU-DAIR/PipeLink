#!/usr/bin/env python3
"""MoK 24-cell forward/backward/recompute numeric, parity and sanitizer driver.

Run under the host uv project after tools/env.sh. Without RANK, this entrypoint
launches torchrun; with RANK, it is the rank worker. No GPU ordinal is persisted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EVIDENCE = Path(os.environ.get("MOK_EVIDENCE_DIR", REPO / "campaigns/mok/evidence"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check", "time", "sanitize"))
    p.add_argument("--build-dir", type=Path, default=REPO / "campaigns/mok/port/build")
    p.add_argument("--source", type=Path, default=None, help="patched upstream source; defaults to MOK_UPSTREAM_SOURCE or scratch mok/harness/upstream-src")
    p.add_argument("--shape", choices=("sanity", "families", "all"), default="all")
    p.add_argument("--image", choices=("linked", "one-tu", "both"), default="both")
    p.add_argument("--world", type=int, choices=(1,4), default=4)
    p.add_argument("--cell", help="canonical matrix cell ID; default selects all applicable cells")
    p.add_argument("--shape-name", help="exact upstream shape name for a focused diagnostic")
    p.add_argument("--parameter-family", help="exact upstream MoK parameter family for a focused diagnostic")
    p.add_argument("--dry-run", action="store_true", help="validate link resource metadata without importing CUDA/torch")
    p.add_argument("--upstream-only", action="store_true")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--tokens", type=int, nargs="+", choices=(2048, 8192), default=(2048, 8192))
    p.add_argument("--timing-config", choices=("upstream", "harness"), default="upstream",
                   help="matched config: upstream precision defaults or legacy harness 36/28")
    p.add_argument("--timing-sections", action="store_true", help="diagnostic nested events around launch regions")
    p.add_argument("--lock-mhz", type=int)
    p.add_argument("--tool", choices=("memcheck", "racecheck", "synccheck"), nargs="+", default=("memcheck", "racecheck", "synccheck"))
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--tag", default="p2", help="evidence filename prefix, not a path")
    p.add_argument("--sanitizer-rank", choices=("memcheck", "racecheck", "synccheck"), help=argparse.SUPPRESS)
    return p


def scrub(value):
    """Keep machine-specific paths out of repository evidence."""
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str):
        for key in ("PIPELINK_SCRATCH_ROOT", "PIPELINK_CUDA_HOME"):
            if os.environ.get(key):
                value = value.replace(os.environ[key], "$" + key)
        return value.replace(str(REPO) + "/", "")
    return value


def save(name, report):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps(scrub(report), indent=2, allow_nan=False) + "\n")


def configure_environment(args):
    """Single pre-torch environment seam, shared by parent and rank processes.

    NCCL_* and TORCH_* settings are inherited unchanged.
    """
    if "PIPELINK_GPUS" not in os.environ:
        raise RuntimeError("source tools/env.sh to select the authorized devices")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["PIPELINK_GPUS"]
    if args.source is None:
        configured = os.environ.get("MOK_UPSTREAM_SOURCE")
        args.source = Path(configured) if configured else Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "mok/harness/upstream-src"
    args.source = args.source.resolve()
    # Put the selected out-of-tree MoK and test helpers before installed packages.
    sys.path[:0] = [str(args.source), str(REPO)]
    os.environ["PYTHONPATH"] = os.pathsep.join((str(args.source), str(REPO), os.environ.get("PYTHONPATH", "")))


def completion_status(rows):
    for status in ("FAIL","FAIL-inherited-crash","FAIL-inherited"):
        if any(row["status"]==status for row in rows):
            return status
    return "PASS"


def worker(args):
    import torch
    import torch.distributed as dist
    from mok_port_host import MoKHost, SANITY, Shape, check_host, time_host, config_for, InheritedScheduleCapacityError
    from tests.utils import shapes, mok_params
    from mok import functional
    from resources import load_cells
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    report = {"rank": rank, "world": int(os.environ["WORLD_SIZE"]), "mode": args.mode,
              "image": args.image, "status": "FAIL", "cells": []}
    suffix = args.sanitizer_rank or args.mode
    filename = f"{args.tag}-{suffix}-{args.image}-rank{rank}.json"
    try:
        if report["world"] != args.world:
            raise ValueError("torchrun world differs from selected link world")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank),
                                timeout=timedelta(seconds=args.timeout))
        if args.mode == "time":
            selected = [Shape(f"bench-{tokens}", 384, 7168, 3072, 6, tokens) for tokens in args.tokens]
        elif args.mode == "sanitize":
            selected = [SANITY]
        else:
            selected = ([SANITY] if args.shape in ("sanity", "all") else [])
            if args.shape in ("families", "all"):
                selected += [Shape(*shape) for shape in shapes(args.world)]
        if args.shape_name:
            selected = [s for s in selected if s.name==args.shape_name]
            if not selected:
                raise ValueError("unknown selected shape name")
        cells = load_cells(args.build_dir)
        chosen = select_cells(cells,args)
        for cell in chosen:
            siblings = [c for c in cells if all(c[k]==cell[k] for k in ("precision","world","clamp"))]
            for shape in selected:
                params = [None] if shape==SANITY or args.mode!="check" else mok_params()
                if args.parameter_family:
                    params = [p for p in params if p is not None and p[0]==args.parameter_family]
                    if not params:
                        raise ValueError("unknown selected parameter family")
                for param in params:
                    config = config_for(shape,benchmark=args.mode=="time") if param is None else functional.MoKConfig(
                        fwd_num_comm_sms=param[1],bwd_num_comm_sms=param[2],minibatch_size=param[3],
                        macrobatch_size=param[4],schedule_capacity_multiplier=1.5)
                    if args.mode=="time" and args.timing_config=="upstream":
                        config = functional.MoKConfig(
                            fwd_num_comm_sms=24 if cell["precision"]=="bf16" else 36,
                            bwd_num_comm_sms=28 if cell["precision"]=="bf16" else 36)
                    report["active_cell"] = {"cell":cell["id"],"shape":asdict(shape),"config":asdict(config),
                                             "parameter_family":param[0] if param else "sanity/default"}
                    save(filename,report)
                    try:
                        host = MoKHost(shape,siblings,config=config,image="one-tu" if args.image=="one-tu" else "linked")
                    except InheritedScheduleCapacityError as exc:
                        report["cells"].append({**report["active_cell"],"status":"FAIL-inherited-crash",
                            "port_verdict":None,"direct_executed":False,"graph_executed":False,
                            "upstream_defect":exc.evidence})
                        save(filename,report)
                        gc.collect()
                        continue
                    with host:
                        if args.mode=="time":
                            result = time_host(host,cell["pass"],args.repetitions,sections=args.timing_sections)
                            # Bind the ratio to the exact image pair it measured (the cubins the sessions loaded).
                            result["images"] = {key: hashlib.sha256((cell["directory"]/cell[key]).read_bytes()).hexdigest()
                                                for key in ("linked","one_tu")}
                            from mok import _C
                            result["images"]["upstream_extension"] = hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest()
                            result["timing_config"] = args.timing_config
                        elif args.mode=="sanitize":
                            for generation in range(3):
                                host.sequence(host.new_inputs(generation+1),cell["pass"])
                            torch.cuda.synchronize(); dist.barrier()
                            result = {"status":"PASS","steps":3,"tool":args.sanitizer_rank}
                        else:
                            images = ("linked","one-tu") if args.image=="both" else (args.image,)
                            result = check_host(host,cell["pass"],upstream_only=args.upstream_only,steps=args.steps,images=images)
                        report["cells"].append({"cell":cell["id"],"shape":asdict(shape),
                            "config":asdict(config),"parameter_family":param[0] if param else "sanity/default",**result})
                        save(filename,report)
                        if result["status"]=="FAIL":
                            raise RuntimeError("port numeric/parity gate failed; stopping for diagnosis")
                    # Drop the closed bucket before constructing its successor;
                    # graph/session ownership cycles must not retain model-sized loans.
                    del host
                    gc.collect()
        report.pop("active_cell",None)
        report["status"] = completion_status(report["cells"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        import traceback
        report["traceback"] = traceback.format_exc()
    finally:
        save(filename, report)
        print(json.dumps({"rank": rank, "status": report["status"], "evidence": filename}), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0 if report["status"]!="FAIL" else 1


def clock_command(action, args):
    argv = [sys.executable, str(REPO / "tools/gpulockd.py"), action, "--gpus", os.environ["PIPELINK_GPUS"]]
    if action == "start":
        argv += ["--sm-mhz", str(args.lock_mhz)]
    result = subprocess.run(argv, capture_output=True, text=True)
    # gpulockd text contains physical ordinals; persist status only.
    return {"action": action, "returncode": result.returncode}


def launch(args):
    if "PIPELINK_GPUS" not in os.environ:
        raise RuntimeError("source tools/env.sh to select the authorized devices")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ["PIPELINK_GPUS"])
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report = {"mode": args.mode, "world": args.world, "status": "FAIL", "runs": [],
              "lock_mhz": args.lock_mhz, "clock_locked": False}
    # Workers inherit the same host-project Python interpreter as this launcher.
    base = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={args.world}",
            str(Path(__file__).resolve()), *sys.argv[1:]]
    locks = []
    try:
        if args.lock_mhz:
            locks.append(clock_command("start", args))
            if locks[-1]["returncode"]:
                raise RuntimeError("gpulockd start failed")
        cells = [(None, args.image)] if args.mode != "sanitize" else [
            (tool, image) for tool in args.tool
            for image in (("linked", "one-tu") if args.image == "both" else (args.image,))
        ]
        for tool, image in cells:
            command = [*base, *(["--sanitizer-rank", tool, "--image", image] if tool else [])]
            process = subprocess.Popen(command, env=env, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                # Kill the whole rank/sanitizer process group, not just torchrun.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
            rank_reports = [json.loads((EVIDENCE/f"{args.tag}-{tool or args.mode}-{image}-rank{rank}.json").read_text())
                            for rank in range(args.world)] if code==0 else []
            status = completion_status(rank_reports) if code==0 else "FAIL"
            counts = []
            if tool and code==0:
                for rank in range(args.world):
                    findings = json.loads((EVIDENCE/f"{args.tag}-{tool}-{image}-rank{rank}-sanitizer.json").read_text())
                    counts.append({key:findings[key] for key in
                                   ("rank","device_error_or_hazard_count","host_api_error_count")})
                if any(row["device_error_or_hazard_count"]!=0 for row in counts):
                    status = "FAIL"
            report["runs"].append({"tool":tool,"image":image,"returncode":code,"status":status,
                                   "sanitizer_rank_counts":counts})
        if args.lock_mhz:
            locks.append(clock_command("status", args))
            report["clock_locked"] = locks[-1]["returncode"] == 0
        report["status"] = completion_status(report["runs"])
        if args.lock_mhz and not report["clock_locked"]:
            report["status"] = "FAIL"
    finally:
        if args.lock_mhz:
            locks.append(clock_command("stop", args))
        report["lock"] = locks
        save(f"{args.tag}-{args.mode}-summary.json", report)
    return 0 if report["status"]!="FAIL" else 1


def sanitizer_counts(text, tool):
    """Separate tool findings from the host API probes; retain every raw report."""
    errors = re.findall(r"^========= ERROR SUMMARY: ([0-9]+) errors?\s*$", text, re.MULTILINE)
    hazards = re.findall(r"RACECHECK SUMMARY: ([0-9]+) hazards?\b", text)
    host_reports = []
    device_reports = []
    for block in text.split("\n========= \n"):
        if "Error: Race reported between " in block:
            accesses = [line.removeprefix("========= ") for line in block.splitlines()
                        if "Error: Race reported between " in line or line.startswith("=========     and ")]
            multiplicity = re.search(r"\[([0-9]+) hazards?\]", block)
            device_reports.append({"accesses": accesses,
                                   "hazard_multiplicity": int(multiplicity.group(1)) if multiplicity else None})
        for line in block.splitlines():
            if not (line.startswith("========= CUDA API Error:")
                    or (line.startswith("========= Program hit ") and "CUDA API call to " in line)):
                continue
            api = re.search(r"CUDA API call to ([A-Za-z0-9_]+)", line)
            host_reports.append({"message": line.removeprefix("========= "),
                                 "api": api.group(1) if api else None,
                                 "source": "torch::isFabricSupported" if "isFabricSupported()" in block else "see raw backtrace"})
    error_count = int(errors[-1]) if errors else None
    hazard_count = int(hazards[-1]) if hazards else None
    # Memcheck/synccheck summaries include CUDA API diagnostic records; the
    # racecheck hazard summary already counts device hazards separately.
    device_count = hazard_count if tool == "racecheck" else (
        error_count - len(host_reports) if error_count is not None else None)
    return {"errors": error_count, "hazards": hazard_count,
            "host_api_error_count": len(host_reports), "host_api_reports": host_reports,
            "device_error_or_hazard_count": device_count,
            "device_reports": device_reports,
            "count_basis": "racecheck hazard summary" if tool == "racecheck" else "error summary minus individually retained host API diagnostics",
            "summaries": [line.strip() for line in text.splitlines() if "SUMMARY:" in line]}


def select_cells(cells,args):
    selected = [c for c in cells if c["world"]==args.world and (not args.cell or c["id"]==args.cell)]
    if args.mode=="time":
        selected = [c for c in selected if c["world"]==4 and c["clamp"]==0]
    elif args.mode=="sanitize":
        selected = [c for c in selected if c["world"]==4 and c["pass"] in ("fwd","bwd")]
    if not selected:
        raise ValueError("selection contains no required matrix cells")
    return selected


def dry_run(args):
    from resources import load_cells,validate_inventory
    cells = select_cells(load_cells(args.build_dir),args)
    report = []
    for cell in cells:
        scalars = dict(num_local_tokens=512,hidden=256,intermediate=256,num_local_experts=4//cell["world"],
                       schedule_capacity=2048,macrobatch_size=256,minibatch_size=256,topk=1)
        specs = validate_inventory(cell,scalars)
        report.append({"cell":cell["id"],"active":sum(s.active for s in specs.values()),
                       "empty":sum(not s.active for s in specs.values()),
                       "peer_entries":cell["world"],"status":"PASS"})
    print(json.dumps({"cuda_used":False,"scope":"manifest inventory/dtype/rank/domain/views and bound axes; no native query/TMA encoding",
                      "cells":report},indent=2))
    return 0


def main():
    args = parser().parse_args()
    if Path(args.tag).name != args.tag or args.tag in (".", ".."):
        raise ValueError("--tag must be a filename prefix")
    if args.steps < 3 or args.repetitions < 1:
        raise ValueError("--steps must be >=3 and --repetitions must be positive")
    if args.upstream_only and args.mode != "check":
        raise ValueError("--upstream-only is available only for check")
    if args.lock_mhz and args.mode != "time":
        raise ValueError("--lock-mhz is available only for time")
    if args.dry_run:
        return dry_run(args)
    if args.mode=="time":
        args.lock_mhz = args.lock_mhz or int(os.environ["PIPELINK_LOCK_SM_MHZ"])
    configure_environment(args)
    if "RANK" in os.environ:
        if args.lock_mhz and not os.environ.get("MOK_CLOCK_PARENT"):
            raise ValueError("--lock-mhz requires launching run.py directly, not through an external torchrun")
        if args.sanitizer_rank and not os.environ.get("MOK_UNDER_SANITIZER"):
            tool = Path(os.environ["PIPELINK_CUDA_HOME"]) / "bin/compute-sanitizer"
            # One instrumented Python worker per torchrun rank, never a sanitizer
            # around torchrun. Findings stay in evidence, not --error-exitcode:
            # otherwise torchrun kills other ranks before their reports flush.
            env = dict(os.environ, MOK_UNDER_SANITIZER="1")
            result = subprocess.run([str(tool), "--tool", args.sanitizer_rank,
                                     sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                                    env=env, capture_output=True, text=True)
            # Store sanitized tool output as JSON rather than leaking paths into logs.
            text = result.stdout + result.stderr
            save(f"{args.tag}-{args.sanitizer_rank}-{args.image}-rank{os.environ['RANK']}-sanitizer.json",
                 {"tool": args.sanitizer_rank, "rank": int(os.environ["RANK"]), "returncode": result.returncode,
                  **sanitizer_counts(text, args.sanitizer_rank),
                  "stdout": result.stdout, "stderr": result.stderr})
            # Findings are adjudicated by the parent only after every rank's
            # reporter has flushed. A rank-level failure here kills its peers.
            return result.returncode
        if args.mode == "sanitize" and not args.sanitizer_rank:
            raise ValueError("launch sanitize directly so every rank gets its own compute-sanitizer")
        return worker(args)
    if not os.environ.get("PIPELINK_GPU_WINDOW"):
        result = subprocess.run([str(REPO/"tools/gpu_window.sh"),sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],
                                capture_output=True,text=True)
        print(result.stdout,end=""); print(result.stderr,end="",file=sys.stderr)
        contended = "contended=1" in result.stderr
        save(f"{args.tag}-{args.mode}-window.json",{"returncode":result.returncode,"contended":contended,
             "timings_valid":not contended,"stdout":result.stdout,"stderr":result.stderr})
        return result.returncode or int(args.mode=="time" and contended)
    if args.lock_mhz:
        os.environ["MOK_CLOCK_PARENT"] = "1"
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
