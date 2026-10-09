"""Timing-only backward ablations; never modify delivered images or endorse unsafe variants."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import subprocess
from collections import Counter
import sys

REPO = Path(__file__).resolve().parents[3]
HOST = REPO / "campaigns/mok/host"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    from run import scrub
    Path(path).write_text(json.dumps(scrub(value), indent=2) + "\n")


def build(args):
    if args.out.exists():
        raise RuntimeError("scratch destination already exists; use a fresh destination")
    args.out.mkdir(parents=True)
    recorded = json.loads((args.delivered / "logs/commands.json").read_text())
    matrix = json.loads((args.delivered / "matrix.json").read_text())
    records = []
    for variant in args.variant:
        root = args.out / variant
        shutil.copytree(REPO / "campaigns/mok/port", root / "port")
        shutil.copytree(args.delivered / "csrc", root / "csrc")
        if variant == "no-act-barrier":
            subprocess.run(["patch", "--batch", "-R", "-p1", "-i", str(REPO / "campaigns/mok/port/patches/mok-f3a8bd2-racefix-act-swiglu-bwd.patch")], cwd=root, check=True)
        elif variant == "no-gemm-init-barrier":
            subprocess.run(["patch", "--batch", "-p1", "-i", str(HOST / "patches/mok-ablate-gemm-init-barrier.patch")], cwd=root / "port", check=True)
        elif variant == "static-framework-facts":
            subprocess.run(["patch", "--batch", "-p1", "-i", str(HOST / "patches/mok-ablate-static-framework-facts.patch")], cwd=root / "port", check=True)
        for precision in (("mxfp8",) if variant == "no-act-barrier" else ("bf16", "mxfp8")):
            cell = f"mok-bwd-{precision}-ep4-c0"
            delivered_cell = args.delivered / "links" / cell
            if variant == "static-framework-facts":
                facts = json.loads((delivered_cell / "link.json").read_text())["host"]["facts"]
                expected = {"gemm.MLP_Mb": 256, "gemm.MLP_Nb": 256, "act.SWIGLU_Mb": 128,
                            "act.SWIGLU_Nb": 128, "act.SWIGLU_FWD_PIPE_DEPTH": 3, "act.SWIGLU_BWD_PIPE_DEPTH": 2,
                            "comm.DISPATCH_Mb": 128, "comm.DISPATCH_Nb": 512, "comm.COMBINE_Mb": 16,
                            "comm.COMBINE_Nb": 1024, "comm.COMBINE_PIPE_DEPTH": 7}
                if any(facts["mok." + key] != value for key, value in expected.items()):
                    raise RuntimeError("static-fact ablation differs from delivered fact values")
            command = next(c for c in recorded if c[-1] == str(delivered_cell / "one-tu.cubin") and "--cubin" in c)
            output = root / f"{cell}.cubin"
            translated = [os.path.expandvars(x).replace(str(args.delivered / "csrc"), str(root / "csrc")).replace("campaigns/mok/port", str(root / "port")) for x in command]
            translated[-1] = str(output)
            if args.line_info:
                translated.insert(1, "-lineinfo")
            with (root / f"{cell}.build.log").open("w") as log:
                subprocess.run(translated, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            view = root / precision
            view.mkdir()
            view_matrix = {**matrix, "cells": []}
            for row in matrix["cells"]:
                original = args.delivered / row["index"]
                if row["id"] != cell:
                    view_matrix["cells"].append({**row, "index": str(original)})
                    continue
                index = json.loads(original.read_text())
                for key in ("link", "linked", "one_tu"):
                    index[key] = str(original.parent / index[key])
                index["providers"] = {k: str(original.parent / v) for k, v in index["providers"].items()}
                # Both image slots use the same ablation, so the normal harness
                # control checks harness noise, not correctness of the unsafe code.
                index["linked"] = index["one_tu"] = str(output)
                # Runtime paths must remain absolute; evidence scrubbing is only
                # for reports, not loader inputs.
                (view / "cell.json").write_text(json.dumps(index, indent=2) + "\n")
                view_matrix["cells"].append({**row, "index": "cell.json"})
            (view / "matrix.json").write_text(json.dumps(view_matrix, indent=2) + "\n")
            (view / "index.json").write_text(json.dumps({"matrix": "matrix.json"}) + "\n")
            records.append({"variant": variant, "precision": precision, "command": translated,
                            "recorded_command": command, "sha256": digest(output),
                            "delivered_onetu_sha256": digest(delivered_cell / "one-tu.cubin"),
                            "matches_delivered": digest(output) == digest(delivered_cell / "one-tu.cubin")})
    save(args.out / "build.json", {"warning": "No-act and no-gemm variants remove required race fixes. Timing only; never deploy.", "builds": records})


def reduce_trace(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    rows = [dict(x) for x in db.execute("SELECT k.start,k.end,k.globalPid,k.gridX,k.blockX,k.clusterX,k.registersPerThread,k.localMemoryPerThread,s.value name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.globalPid,k.start")]
    upstream = [x for x in rows if "globals_bwd" in x["name"]]
    if not upstream:
        raise RuntimeError("trace has no upstream backward kernels")
    starts = {pid: min(x["start"] for x in upstream if x["globalPid"] == pid) for pid in {x["globalPid"] for x in upstream}}
    grids = {x["gridX"] for x in upstream}
    control = [x for x in rows if x["name"] == "pipelink_kernel" and x["gridX"] in grids and x["start"] > starts[x["globalPid"]]]
    result = {}
    for name, values in (("ablation", control), ("upstream", upstream)):
        durations = [(x["end"] - x["start"]) / 1e6 for x in values]
        result[name] = {"calls": len(values), "mean_ms": statistics.mean(durations),
                        "median_ms": statistics.median(durations), "samples_ms": durations,
                        "grid": sorted({x["gridX"] for x in values}), "block": sorted({x["blockX"] for x in values}),
                        "cluster": sorted({x["clusterX"] for x in values}),
                        "registers": sorted({x["registersPerThread"] for x in values}),
                        "local_memory_bytes": sorted({x["localMemoryPerThread"] for x in values})}
    result["ratio"] = result["ablation"]["mean_ms"] / result["upstream"]["mean_ms"]
    return result


def sass(args):
    sys.path.insert(0, str(REPO))
    from evaluator.gates.normalization import sections, canonicalize
    tool = str(Path(os.environ["PIPELINK_CUDA_HOME"]) / "bin/cuobjdump")
    extension, = (args.source / "mok").glob("_C*.so")
    reports = []
    for precision, code in (("bf16", 0), ("mxfp8", 1)):
        symbol = (f"_ZN7kittens2py13global_kernelIN28dispatch_mlp_swiglu_combinerILi4EL15RoutedPrecision{code}EE6configENS4_11globals_bwd"
                  "EXadL_ZNS4_38dispatch_mlp_swiglu_combine_bwd_kernelILb0EEEvRKS6_EEEEvT0_")
        images = [("port", args.delivered / "links" / f"mok-bwd-{precision}-ep4-c0" / "one-tu.cubin"),
                  ("upstream", extension)]
        images += [(variant, args.out / variant / f"mok-bwd-{precision}-ep4-c0.cubin")
                   for variant in args.variant
                   if (args.out / variant / f"mok-bwd-{precision}-ep4-c0.cubin").exists()]
        for arm, image in images:
            select = ["--function", symbol] if arm == "upstream" else []
            dump = subprocess.run([tool, "--dump-sass", *select, str(image)], check=True, text=True, capture_output=True).stdout
            usage = subprocess.run([tool, "--dump-resource-usage", *select, str(image)], check=True, text=True, capture_output=True).stdout
            (args.out / f"{precision}-{arm}.sass").write_text(dump)
            parsed = sections(dump)
            if not parsed:
                raise RuntimeError(f"no SASS for {precision}/{arm}")
            full = Counter()
            for _, body in parsed:
                for instruction in body:
                    fields = instruction.split()
                    opcode = fields[1] if fields[0].startswith("@") else fields[0]
                    full[opcode] += 1
            classes = Counter()
            for opcode, count in full.items():
                classes[opcode.split(".")[0]] += count
            reports.append({"precision": precision, "arm": arm, "image_sha256": digest(image),
                            "functions": [name for name, _ in parsed], "instructions": sum(full.values()),
                            "opcode_classes": dict(sorted(classes.items())), "opcodes": dict(sorted(full.items())),
                            "normalized_sha256": hashlib.sha256(json.dumps([(name, canonicalize(body)) for name,body in parsed]).encode()).hexdigest(),
                            "resources": usage, "sass_sha256": hashlib.sha256(dump.encode()).hexdigest()})
    suffix = "-" + args.variant[0] if len(args.variant) == 1 else ""
    save(REPO / f"campaigns/mok/evidence/p2-upstream-same-harness-backward-sass{suffix}.json",
         {"metric": "static counts, not dynamic issue counts; port includes all called functions, upstream selected EP4 unclamped backward instantiation", "images": reports})


def profile(args):
    evidence = REPO / "campaigns/mok/evidence"
    variants = [(variant, precision) for variant in args.variant
                for precision in (("mxfp8",) if variant == "no-act-barrier" else ("bf16", "mxfp8"))]
    for variant, precision in variants:
        for repetition in range(1, 4):
            attempt = 1
            while True:
                base = f"p2-upstream-same-harness-ablation-{variant}-{precision}-r{repetition}"
                tag = base if attempt == 1 else f"{base}-attempt{attempt}"
                record_path = evidence / f"{tag}-profile.json"
                if record_path.exists():
                    prior = json.loads(record_path.read_text())
                    if "ratio" in prior and prior["clock"]["status"] == "PASS":
                        break
                    if any("contended=1" in line for line in prior.get("gpu_window_lines", [])):
                        attempt += 1
                        continue
                    raise RuntimeError(f"existing incomplete/non-contention failure {tag}; diagnose before resuming")
                trace = args.out / tag
                command = [str(REPO / "tools/gpu_window.sh"), "nsys", "profile", "-t", "cuda", "--sample=none", "--cpuctxsw=none", "-o", str(trace),
                           sys.executable, str(HOST / "run.py"), "time", "--world", "4", "--tokens", "2048", "--cell", f"mok-bwd-{precision}-ep4-c0",
                           "--lock-mhz", "1300", "--build-dir", str(args.out / variant / precision), "--source", str(args.source), "--timeout", "7200", "--tag", tag]
                completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (args.out / f"{tag}.log").write_text(completed.stdout)
                window = re.findall(r"gpu_window:.*", completed.stdout)
                report = {"variant": variant, "precision": precision, "repetition": repetition, "attempt": attempt, "command": command,
                          "returncode": completed.returncode, "gpu_window_lines": window,
                          "timing_only": True, "removes_required_race_fix": variant.startswith("no-")}
                save(record_path, report)
                if completed.returncode or not window:
                    raise RuntimeError(f"invalid ablation run {tag}; retained log, stopping")
                if "contended=1" in window[-1]:
                    attempt += 1
                    continue
                report["clock"] = json.loads((evidence / f"{tag}-time-summary.json").read_text())
                if report["clock"]["status"] != "PASS":
                    save(record_path, report)
                    raise RuntimeError(f"clock/harness failure {tag}; stopping")
                subprocess.run(["nsys", "export", "--type", "sqlite", "--output", str(trace) + ".sqlite", str(trace) + ".nsys-rep"], check=True)
                report.update(reduce_trace(str(trace) + ".sqlite"))
                report["trace_sha256"] = digest(str(trace) + ".nsys-rep")
                report["image_sha256"] = digest(args.out / variant / f"mok-bwd-{precision}-ep4-c0.cubin")
                save(record_path, report)
                print(tag, report["ablation"]["mean_ms"], report["upstream"]["mean_ms"], flush=True)
                break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("build", "profile", "sass"))
    p.add_argument("--delivered", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--variant", nargs="+", choices=("baseline", "no-act-barrier", "no-gemm-init-barrier", "static-framework-facts"),
                   default=("baseline", "no-act-barrier", "no-gemm-init-barrier"))
    p.add_argument("--line-info", action="store_true", help="add source correlation for separate profiling builds")
    args = p.parse_args()
    args.delivered, args.out, args.source = args.delivered.resolve(), args.out.resolve(), args.source.resolve()
    {"build": build, "profile": profile, "sass": sass}[args.mode](args)


if __name__ == "__main__":
    main()
