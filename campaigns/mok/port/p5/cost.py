#!/usr/bin/env python3
"""GPU-free P5 annotation inventory and serialized developer-loop measurements."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[4]
PORT = REPO / "campaigns/mok/port"
EVIDENCE = REPO / "campaigns/mok/evidence/p5"
OWNERS = ("framework", "comm", "gemm", "act")
CELL = "mok-fwd-mxfp8-ep4-c0"
WRAPPERS = {"MOK_COUNTERS_PL_TYPE": "PL_TYPE", "MOK_COMM_TASK_PL_TYPE": "PL_TYPE",
            "MOK_GEMM_TASK_PL_TYPE": "PL_TYPE", "MOK_ACT_TASK_PL_TYPE": "PL_TYPE",
            "MOK_GEMM_COUNTER": "PL_ENDPOINT", "MOK_GEMM_CHANNEL": "PL_HOST_CHANNEL"}
MUTATION_OLD = "compute_group::mul(denominator, gate, -1.0f);"
MUTATION_NEW = "compute_group::mul(denominator, gate, -1.001f);"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def clean_comments(text):
    return re.sub(r'//[^\n]*|/\*.*?\*/', lambda m: "\n" * m[0].count("\n"), text, flags=re.S)


def count_source(path):
    text = path.read_text()
    clean = clean_comments(text)
    kept, definitions, continuing = [], [], False
    for line_number, line in enumerate(clean.splitlines(), 1):
        if continuing or line.lstrip().startswith("#"):
            if "PL_" in line:
                definitions.append({"line": line_number, "text": line.strip()})
            continuing = line.rstrip().endswith("\\")
            kept.append("")
        else:
            kept.append(line)
    counts, occurrences = Counter(), []
    pattern = r'^\s*(PL_\w+|'+ '|'.join(WRAPPERS) +r')\b'
    for line_number, line in enumerate(kept, 1):
        match = re.match(pattern, line)
        if match:
            authored = match[1]
            kind = WRAPPERS.get(authored, authored)
            counts[kind] += 1
            occurrences.append({"line": line_number, "authored": authored, "kind": kind})
    builders = Counter(re.findall(r'\b(pl::(?:\w+::)*\w+)\s*[({]', clean))
    methods = Counter(re.findall(r'\.([a-z_]\w*)\s*\(', clean[clean.find("PL_STAGE("):] if "PL_STAGE(" in clean else ""))
    return {"path": str(path.relative_to(REPO)), "sha256": digest(path),
            "source_lines": len(text.splitlines()), "nonblank_lines": sum(bool(s.strip()) for s in text.splitlines()),
            "statements_by_kind": dict(sorted(counts.items())),
            "device_statements": sum(v for k, v in counts.items() if not k.startswith("PL_HOST_")),
            "host_statements": sum(v for k, v in counts.items() if k.startswith("PL_HOST_")),
            "total_statements": sum(counts.values()), "occurrences": occurrences,
            "nested_pl_builder_calls_by_kind": dict(sorted(builders.items())),
            "nested_builder_methods_by_kind": dict(sorted(methods.items())),
            "macro_definition_lines_not_counted_as_invocations": definitions}


def annotations():
    rows = [count_source(p) for owner in OWNERS for p in sorted((PORT / owner).iterdir())
            if p.suffix in (".cu", ".cuh", ".h")]
    shared = [count_source(p) for p in sorted((PORT / "include").iterdir()) if p.is_file()]
    alternative = [count_source(p) for p in sorted((PORT / "act_gptoss").glob("*"))
                   if p.suffix in (".cu", ".cuh", ".h")]
    owners = {}
    for owner in OWNERS:
        selected = [r for r in rows if Path(r["path"]).parent.name == owner]
        kinds = Counter()
        for row in selected:
            kinds.update(row["statements_by_kind"])
        owners[owner] = {"source_lines": sum(r["source_lines"] for r in selected),
                         "total_statements": sum(r["total_statements"] for r in selected),
                         "statements_by_kind": dict(sorted(kinds.items()))}
    patch_rows = []
    for patch in sorted((PORT / "patches").glob("*.patch")):
        current = None
        files = []
        for line in patch.read_text().splitlines():
            if line.startswith("+++ b/"):
                current = {"path": line[6:], "added": 0, "removed": 0}
                files.append(current)
            elif current is not None and line.startswith("+") and not line.startswith("+++"):
                current["added"] += 1
            elif current is not None and line.startswith("-") and not line.startswith("---"):
                current["removed"] += 1
        owner = next((o for o in OWNERS if patch.stem.endswith("-" + o)), "shared")
        patch_rows.append({"path": str(patch.relative_to(REPO)), "sha256": digest(patch), "owner": owner,
                           "files": files, "added": sum(f["added"] for f in files),
                           "removed": sum(f["removed"] for f in files)})
    for owner, row in owners.items():
        attributed = [p for p in patch_rows if p["owner"] == owner]
        row["upstream_patch_added"] = sum(p["added"] for p in attributed)
        row["upstream_patch_removed"] = sum(p["removed"] for p in attributed)
        row["upstream_patch_churn"] = row["upstream_patch_added"] + row["upstream_patch_removed"]
    packages = []
    for row in rows:
        filename = Path(row["path"]).name
        if filename in ("act.cu", "comm.cu", "gemm.cu") or filename.startswith("interp_"):
            packages.append({"package": "mok.interp." + filename[7:-3] if filename.startswith("interp_") else "mok." + filename[:-3],
                             **{k: row[k] for k in ("path", "source_lines", "device_statements", "host_statements", "total_statements", "statements_by_kind")}})
    return {"method": "Logical top-level PL_* declaration statements, including one expansion per known MOK_* wrapper invocation; macro definitions are not extra invocations. Nested pl:: builders and chained methods are authored call sites (including macro definitions once), reported separately, not double-counted as statements. All physical lines, including comments/blanks, are source-line denominators. Owner totals include providers and ABI headers; package denominators are entry source only. Shared headers and patches are separate, never multiplied across owners.",
            "precedents": ["examples/comm-overlap/README.md:227-236", "examples/hopper-gemm-split/README.md:243-246", "examples/one-epilogue-many-producers/README.md:9-11"],
            "sources": rows, "shared_sources": shared, "owners": owners, "packages": packages,
            "alternative_act_gptoss_sources": alternative,
            "patches": patch_rows, "patch_measure": "Added and removed physical diff lines; churn=added+removed; moves count on both sides. Shared patch is not arbitrarily assigned to one owner. Framework extracted interpreter sources are authored lines, not an upstream-file diff."}


def normalized(value):
    text = str(value)
    for actual, symbolic in ((str(REPO), "$REPO"), (os.environ.get("PIPELINK_SCRATCH_ROOT", ""), "$PIPELINK_SCRATCH_ROOT"),
                             (os.environ.get("PIPELINK_CUDA_HOME", ""), "$PIPELINK_CUDA_HOME")):
        if actual:
            text = text.replace(actual, symbolic)
    return text


def measure(label, argv, cwd, env, records):
    samples, done = [], threading.Event()
    start = time.perf_counter()
    def sample():
        while not done.is_set():
            samples.append({"seconds": time.perf_counter() - start, "loadavg": list(os.getloadavg())})
            done.wait(1)
    thread = threading.Thread(target=sample)
    thread.start()
    result = subprocess.run(list(map(str, argv)), cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    done.set()
    thread.join()
    log = EVIDENCE / "logs" / (label + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(normalized(result.stdout))
    row = {"label": label, "argv": [normalized(v) for v in argv], "cwd": normalized(cwd),
           "wall_seconds": elapsed, "exit_code": result.returncode,
           "log": str(log.relative_to(REPO)), "load_samples": samples,
           "load_end": list(os.getloadavg())}
    records.append(row)
    save(EVIDENCE / "run-records.json", records)
    print(f"{label}: {elapsed:.3f}s exit={result.returncode}", flush=True)
    return row


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    sources = root / "port"
    sources.mkdir()
    for owner in (*OWNERS, "include"):
        shutil.copytree(PORT / owner, sources / owner)
    shutil.copy2(PORT / "one_tu.cu", sources / "one_tu.cu")
    out = root / "out"
    out.mkdir()
    shutil.copytree(PORT / "build/csrc", out / "csrc")
    for package in ("act", "comm", "gemm", "framework.fwd"):
        shutil.copy2(PORT / "build" / (package + ".plink"), out / (package + ".plink"))
    cell = out / "links" / CELL
    shutil.copytree(PORT / "build/links" / CELL, cell)
    source = out / "csrc/megakernel/swiglu.cuh"
    text = source.read_text()
    if text.count(MUTATION_OLD) != 1:
        raise RuntimeError("Activation mutation no longer has exactly one source anchor")
    source.write_text(text.replace(MUTATION_OLD, MUTATION_NEW))
    return sources, out, cell


def timing_run(gates):
    spec = importlib.util.spec_from_file_location("mok_build", PORT / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    cuda = Path(os.environ["PIPELINK_CUDA_HOME"])
    scratch = Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "mok/p5" / time.strftime("run-%Y%m%d-%H%M%S")
    scratch.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", PIPEC_CUDA=str(cuda))
    records, trials = [], []
    inventory = annotations()
    save(EVIDENCE / "annotation-counts.json", inventory)
    provenance = {"repo_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "lscpu": subprocess.check_output(["lscpu", "-J"], text=True),
                  "cpu_affinity": sorted(os.sched_getaffinity(0)), "logical_cpus": os.cpu_count(),
                  "nvcc_version": subprocess.check_output([cuda / "bin/nvcc", "--version"], text=True),
                  "nvcc_realpath": normalized((cuda / "bin/nvcc").resolve()),
                  "ccache_on_path": shutil.which("ccache"),
                  "cache_environment": {k: normalized(v) for k, v in env.items() if "CACHE" in k or k in ("NVCC", "MOK_NVCC")},
                  "gate_messages": gates, "scratch": normalized(scratch),
                  "frozen_packages": {p.name: digest(p) for p in (PORT / "build").glob("*.plink")},
                  "count_sources": {r["path"]: r["sha256"] for r in inventory["sources"]},
                  "build_driver_sha256": digest(PORT / "build.py"),
                  "upstream_build_driver_sha256": digest(REPO / "campaigns/mok/host/build_upstream.sh"),
                  "upstream_build_inputs": json.loads((PORT / "build/build-inputs.json").read_text()),
                  "host_lock_sha256": digest(REPO / "campaigns/mok/host/uv.lock"),
                  "pipec_source_sha256": {str(p.relative_to(REPO)): digest(p)
                      for p in sorted((REPO / "pipec/src/pipec2").rglob("*"))
                      if p.suffix in (".py", ".h", ".hpp", ".cuh") and p.is_file()},
                  "harness_sha256": digest(Path(__file__))}
    for repetition in range(1, 4):
        for leg in ("repack_relink", "one_tu", "upstream"):
            if leg != "upstream":
                sources, out, cell = prepare(scratch / f"r{repetition}-{leg}")
                includes = [out / "csrc", out / "csrc/megakernel", sources / "include", *(sources / o for o in OWNERS),
                            REPO / "third_party/mixture-of-kittens/third_party/ThunderKittens/include", REPO / "pipec/src/pipec2/pack/include"]
                from pipec2.host.abi import header_path
                includes.append(header_path().parent)
            for state in ("cold", "warm"):
                prefix = f"r{repetition}-{leg}-{state}"
                selected = []
                if leg == "repack_relink":
                    selected.append(measure(prefix + "-pack", [sys.executable, "-m", "pipec2.cli", "pack", sources / "act/act.cu", "--profile", "sm_103a", "-o", out / "act.plink",
                        *[v for inc in includes for v in ("-I", inc)], *[f"--nvcc-flag={f}" for f in [*build.FLAGS, "--frandom-seed=mok.act"]]], REPO, env, records))
                    selected.append(measure(prefix + "-link", [sys.executable, "-m", "pipec2.cli", "link", cell / "pipeline.json", "--out-dir", cell], REPO, env, records))
                elif leg == "one_tu":
                    cap = json.loads((cell / "link.json").read_text())["regs_cap"]
                    flags = [*build.FLAGS, "-O3", "--frandom-seed=mok.one_tu", f"-maxrregcount={cap}", *[f"-I{i}" for i in includes],
                             f'-DMOK_BINDING_STUB="{cell / "generated/binding_stub.cu"}"', f'-DMOK_INTERPRETER="{sources / "framework/interp_fwd.cu"}"', f'-DMOK_DISPATCH="{cell / "generated/dispatch.cu"}"']
                    selected.append(measure(prefix + "-plain", [cuda / "bin/nvcc", *flags, "-arch=sm_103a", "--cubin", sources / "one_tu.cu", "-o", cell / "one-tu.cubin"], REPO, env, records))
                    selected.append(measure(prefix + "-lto-compile", [cuda / "bin/nvcc", *flags, "-gencode=arch=compute_103a,code=lto_103a", "-dc", "-rdc=true", sources / "one_tu.cu", "-o", cell / "one-tu-lto.o"], REPO, env, records))
                    selected.append(measure(prefix + "-lto-link", [sys.executable, str(Path(__file__).resolve()), "--lto-link", cell], REPO, env, records))
                else:
                    source = scratch / f"r{repetition}-upstream-{state}"
                    upstream_env = dict(env, MOK_UPSTREAM_SOURCE=str(source), MOK_UPSTREAM_LOG_DIR=str(scratch / f"r{repetition}-upstream-{state}-logs"), MOK_SCHEDULE_OUT="0")
                    selected.append(measure(prefix, ["bash", REPO / "campaigns/mok/host/build_upstream.sh"], REPO, upstream_env, records))
                    selected[-1]["environment"] = {k: normalized(upstream_env[k]) for k in ("MOK_UPSTREAM_SOURCE", "MOK_UPSTREAM_LOG_DIR", "MOK_SCHEDULE_OUT")}
                    symbols = subprocess.run([cuda / "bin/cuobjdump", "--dump-resource-usage", *source.glob("mok/_C*.so")], capture_output=True, text=True)
                    resource_log = EVIDENCE / "logs" / (prefix + "-resources.log")
                    resource_log.write_text(normalized(symbols.stdout + symbols.stderr))
                    selected[-1]["resource_log"] = str(resource_log.relative_to(REPO))
                    selected[-1]["megakernel_instantiations"] = len(re.findall(r"Function\s+[^\n]*dispatch_mlp_swiglu_combiner", symbols.stdout))
                    selected[-1]["resource_exit_code"] = symbols.returncode
                artifacts = {}
                if leg != "upstream":
                    names = ("pipeline.cubin",) if leg == "repack_relink" else ("one-tu.cubin", "one-tu-lto.cubin")
                    artifacts = {name: digest(cell / name) for name in names}
                    artifacts["changed_swiglu_source"] = digest(out / "csrc/megakernel/swiglu.cuh")
                trials.append({"repetition": repetition, "leg": leg, "state": state,
                               "wall_seconds": sum(r["wall_seconds"] for r in selected),
                               "exit_codes": [r["exit_code"] for r in selected], "components": [r["label"] for r in selected],
                               "artifact_sha256": artifacts})
                save(EVIDENCE / "timing-partial.json", {"environment": provenance, "trials": trials, "records": records})
                if any(r["exit_code"] for r in selected):
                    raise RuntimeError(f"{prefix} failed; recorded rather than scored as a successful build")
    summaries = {}
    for leg in ("repack_relink", "one_tu", "upstream"):
        summaries[leg] = {}
        for state in ("cold", "warm"):
            values = [r["wall_seconds"] for r in trials if r["leg"] == leg and r["state"] == state]
            summaries[leg][state] = {"seconds": values, "median_seconds": statistics.median(values), "min_seconds": min(values), "max_seconds": max(values), "spread_seconds": max(values) - min(values)}
    components = {}
    for suffix in ("repack_relink-cold-pack", "repack_relink-warm-pack",
                   "repack_relink-cold-link", "repack_relink-warm-link",
                   "one_tu-cold-plain", "one_tu-warm-plain",
                   "one_tu-cold-lto-compile", "one_tu-warm-lto-compile",
                   "one_tu-cold-lto-link", "one_tu-warm-lto-link"):
        values = [r["wall_seconds"] for r in records if r["label"].endswith("-" + suffix)]
        components[suffix] = {"seconds": values, "median_seconds": statistics.median(values),
                              "min_seconds": min(values), "max_seconds": max(values),
                              "spread_seconds": max(values) - min(values)}
    result = {"claim": "C5", "objective": "O4", "gpu_used": False, "cell": CELL, "annotations": inventory,
              "environment": provenance, "timings": {"trials": trials, "summary": summaries,
              "component_summary": components, "records": records},
              "mutation": {"file": "csrc/megakernel/swiglu.cuh", "old": MUTATION_OLD, "new": MUTATION_NEW,
                           "scope": "Port scratch copies only: forward shared/BF16 activation sigmoid slope. No ABI change, providers reused. Upstream is unmodified 72-instantiation breadth control, not numerical equivalence."},
              "cache_policy": "Cold means fresh scratch source/output paths, with pre-existing other-owner packages and generated cell configuration copied as developer inputs. OS page cache, toolchain, Python/uv environment and any ambient compiler caches are not flushed. Warm is the immediate second unconditional compile/link with all compiler/cache state as-is. Upstream build script refuses existing destinations, so warm uses a second fresh destination but retains all external cache state. nvcc is called directly, not through a ccache wrapper. No make no-op is scored.",
              "gpu_commands_remaining": []}
    save(EVIDENCE / "cost.json", result)
    (EVIDENCE / "timing-partial.json").unlink()
    (EVIDENCE / "run-records.json").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--lto-link", type=Path)
    parser.add_argument("--build-gates", nargs=3, metavar=("P3_MESSAGE", "P4_MESSAGE", "P2_MESSAGE"))
    args = parser.parse_args()
    if args.annotations_only:
        save(EVIDENCE / "annotation-counts.json", annotations())
    elif args.lto_link:
        from pipec2.link import _NvJitLink, _link_options
        from pipec2.pack.emit import math_mode
        from pipec2.profile import get_profile
        cell = args.lto_link
        cap = json.loads((cell / "link.json").read_text())["regs_cap"]
        options = _link_options(get_profile("sm_103a"), cap, math_mode(["--use_fast_math"]))
        image, info, error = _NvJitLink(Path(os.environ["PIPELINK_CUDA_HOME"]) / "lib64/libnvJitLink.so.13").link([("one-tu-lto.o", (cell / "one-tu-lto.o").read_bytes())], options)
        (cell / "one-tu-lto.cubin").write_bytes(image)
        print(json.dumps({"options": options, "bytes": len(image)}))
        print(info + error)
    else:
        if not args.build_gates:
            parser.error("Timings require explicit P3, P4, and P2 build-completion messages")
        timing_run(args.build_gates)


if __name__ == "__main__":
    main()
