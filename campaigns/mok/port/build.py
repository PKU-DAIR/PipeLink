#!/usr/bin/env python3
"""GPU-free MoK packages/providers and 24 links with same-source controls and reuse evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Lock
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OWNERS = ("framework", "comm", "gemm", "act")
OPS = OWNERS[1:]
PASSES = ("fwd", "bwd", "rec")
PACKAGES = (*OPS, *(f"framework.{p}" for p in PASSES))
DEFAULT_CELL = "mok-fwd-mxfp8-ep4-c0"
FLAGS = ["-std=c++20", "--expt-extended-lambda", "--expt-relaxed-constexpr", "--use_fast_math",
         "-DNDEBUG", "-DKITTENS_SM103", "-DMOK_DEVICE_ONLY", "-D__CUDA_NO_HALF_OPERATORS__",
         "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
         "-D__CUDA_NO_HALF2_OPERATORS__"]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    """sha256 over (relative path, file sha256) of every file under root, in path order."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(f"{path.relative_to(root)}\0{sha256(path)}\n".encode())
    return digest.hexdigest()


def dropped_endpoints(pass_name: str) -> list[dict]:
    backward = {
        "comm": "d_y_routed_ready d_x_routed_ready replayed_x_routed_ready routed_buffers_done routed_buffers_done router_weights_ready router_weights_ready",
        "gemm": "d_y_routed_ready d_hidden_ready d_gate_up_ready d_x_routed_ready replayed_x_routed_ready replayed_gate_up_ready replayed_hidden_ready routed_buffers_done",
        "act": "d_hidden_ready d_gate_up_ready replayed_gate_up_ready replayed_hidden_ready routed_buffers_done",
    }
    forward = {
        "comm": "x_routed_ready y_routed_ready y_routed_done",
        "gemm": "x_routed_ready gate_up_tile_ready hidden_row_block_ready y_routed_ready y_routed_done",
        "act": "gate_up_tile_ready hidden_row_block_ready",
    }
    down = {"comm": "y_routed_ready y_routed_done",
            "gemm": "hidden_row_block_ready y_routed_ready y_routed_done"}
    counters = [forward] if pass_name == "bwd" else [backward]
    if pass_name == "rec":
        counters.append(down)
    rows = [{"stage": f"mok.{owner}", "endpoint": channel, "reason": "owned by unreferenced exports"}
            for group in counters for owner, channels in group.items() for channel in channels.split()]
    rows += [{"stage": f"mok.{owner}", "endpoint": f"{owner}_{p}", "reason": "uncalled export channel"}
             for p in PASSES if p != pass_name for owner in OPS]
    return sorted(rows, key=lambda row: (row["stage"], row["endpoint"], row["reason"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only", nargs="+", choices=OWNERS, help="Pack/build only selected owners")
    mode.add_argument("--link-only", action="store_true", help="Link existing package/provider pairs without rebuilding them")
    parser.add_argument("--out", type=Path, default=HERE / "build", help="Output directory; owner checks use private scratch")
    parser.add_argument("--jobs", type=int, default=min(24, os.cpu_count() or 1), help="Concurrent independent link/control cells")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not os.environ.get("PIPELINK_CUDA_HOME"):
        raise RuntimeError("source tools/env.sh before running this build")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PIPEC_CUDA"] = os.environ["PIPELINK_CUDA_HOME"]
    from pipec2.container import read as read_package
    from pipec2.host.abi import ABI_TYPES, header_path
    from pipec2.host.provider import build_provider
    from pipec2.link import _NvJitLink, _link_options
    from pipec2.pack.emit import math_mode
    from pipec2.profile import get_profile

    started = time.monotonic()
    out = args.out.resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cuda = Path(os.environ["PIPELINK_CUDA_HOME"])
    nvcc = cuda / "bin/nvcc"
    cli = [sys.executable, "-m", "pipec2.cli"]
    commands = []
    command_lock = Lock()

    def command(argv, name, cwd=REPO):
        argv = list(map(str, argv))
        with command_lock:
            commands.append([value.replace(str(REPO) + "/", "")
                             .replace(str(cuda), "$PIPELINK_CUDA_HOME") for value in argv])
            write_json(logs / "commands.json", commands)
        result = subprocess.run(argv, cwd=cwd, env=os.environ.copy(), text=True, capture_output=True)
        text = result.stdout + result.stderr
        log = logs / name
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text)
        if result.returncode:
            raise RuntimeError(f"{name}: exit {result.returncode}\n{text[-12000:]}")
        return text

    includes = [out / "csrc", out / "csrc/megakernel", HERE / "include",
                *(HERE / owner for owner in OWNERS),
                REPO / "third_party/mixture-of-kittens/third_party/ThunderKittens/include",
                REPO / "pipec/src/pipec2/pack/include", header_path().parent]
    if not args.link_only:
        # A fresh copy prevents patches accumulating across authoring builds.
        if (out / "csrc").exists():
            shutil.rmtree(out / "csrc")
        shutil.copytree(REPO / "third_party/mixture-of-kittens/csrc", out / "csrc")
        patches = sorted((HERE / "patches").glob("*.patch"))
        for patch in patches:
            if out.is_relative_to(REPO):
                command(["git", "apply", f"--directory={out.relative_to(REPO)}", patch], patch.stem + ".log")
            else:
                command(["git", "apply", patch], patch.stem + ".log", cwd=out)
        provenance = {
            # Content digests work on remote mirrors without git metadata and cover local edits.
            "source_digests": {name: tree_digest(path) for name, path in (
                ("mok_csrc", REPO / "third_party/mixture-of-kittens/csrc"),
                ("tk_include", REPO / "third_party/mixture-of-kittens/third_party/ThunderKittens/include"))},
            "patches": [{"path": str(p.relative_to(REPO)), "sha256": sha256(p)} for p in patches],
            "profile": "sm_103a", "flags": FLAGS,
            "toolchain": command([nvcc, "--version"], "nvcc.version.log").strip(),
        }
        write_json(out / "build-inputs.json", provenance)
        for package in PACKAGES:
            owner, _, pass_name = package.partition(".")
            if args.only and owner not in args.only:
                continue
            source = HERE / owner / (f"interp_{pass_name}.cu" if pass_name else f"{owner}.cu")
            stage_id = f"mok.interp.{pass_name}" if pass_name else f"mok.{owner}"
            command([*cli, "pack", source, "--profile", "sm_103a", "-o", out / f"{package}.plink",
                     *[v for inc in includes for v in ("-I", inc)],
                     *[f"--nvcc-flag={flag}" for flag in [*FLAGS, f"--frandom-seed={stage_id}"]]],
                    f"{package}.pack.log")
            manifest, _ = read_package(out / f"{package}.plink")
            host = manifest["host"]
            inventory = host["provider"]["abi_types"]
            local = [row for row in inventory if row["name"] not in ABI_TYPES]
            type_header = HERE / owner / ("framework_abi.h" if pass_name else f"{owner}.cu")
            type_flags = {"framework": [f"-DMOK_FRAMEWORK_PASS={PASSES.index(pass_name)}"] if pass_name else [],
                          "act": ["-DMOK_ACT_PROVIDER"], "gemm": ["-DMOK_GEMM_TYPES_ONLY"],
                          "comm": ["-DMOK_COMM_PROVIDER"]}[owner]
            provider = out / f"provider_{package}.so"
            try:
                observed = build_provider(
                    [HERE / owner / "provider.cu"], provider, owner=f"mok_{owner}",
                    owner_contract_digest=host["provider"]["owner_contract_digest"], profile="sm_103a",
                    dtypes_digest=host["target"]["dtypes"],
                    type_ids={row["name"]: row["id"] for row in local},
                    extra_types={row["name"]: row["fields"] for row in local},
                    extra_headers=[type_header], compiler=str(nvcc),
                    compile_flags=["-x", "cu", "-O3", *FLAGS, *type_flags, *[f"-I{inc}" for inc in includes],
                                   f"-L{cuda / 'lib64/stubs'}", "-lcuda"])
            except subprocess.CalledProcessError as error:
                (logs / f"{package}.provider.log").write_text((error.stdout or "") + (error.stderr or ""))
                raise RuntimeError(f"{package}: provider compilation failed\n{error.stderr}") from error
            if observed != inventory:
                raise RuntimeError(f"{package}: provider ABI inventory differs from packed owner contract")
            write_json(logs / f"{package}.provider-abi.json", observed)
            write_json(out / f"{package}.provider.json", {
                "inventory": observed, "owner_contract_digest": host["provider"]["owner_contract_digest"],
                "artifact_sha256": sha256(out / f"{package}.plink"), "provider_sha256": sha256(provider),
            })
            print(f"{package}: package and ABI-matched provider built", flush=True)
    if args.only:
        elapsed = time.monotonic() - started
        write_json(out / "timings.json", {"owner_build_seconds": elapsed})
        print(f"Owner build wall time: {elapsed:.3f}s", flush=True)
        return

    provenance = json.loads((out / "build-inputs.json").read_text())
    artifacts = {}
    manifests = {}
    input_hashes = {}
    for package in PACKAGES:
        artifact = out / f"{package}.plink"
        provider = out / f"provider_{package}.so"
        manifest, payload = read_package(artifact)
        manifests[package] = manifest
        proof = json.loads((out / f"{package}.provider.json").read_text())
        artifact_hash, provider_hash = sha256(artifact), sha256(provider)
        if (proof["artifact_sha256"] != artifact_hash or proof["provider_sha256"] != provider_hash
                or proof["inventory"] != manifest["host"]["provider"]["abi_types"]
                or proof["owner_contract_digest"] != manifest["host"]["provider"]["owner_contract_digest"]):
            raise RuntimeError(f"{package}: existing package/provider pair does not match its ABI build record")
        payload_hash = hashlib.sha256(payload).hexdigest()
        if payload_hash != manifest["payload"]["sha256"]:
            raise RuntimeError(f"{package}: payload hash differs from package manifest")
        input_hashes[artifact.name] = artifact_hash
        input_hashes[provider.name] = provider_hash
        artifacts[package] = {
            "path": artifact.name, "stage_id": manifest["stage"]["id"], "sha256": artifact_hash,
            "provider": {"path": provider.name, "sha256": provider_hash,
                         "owner_contract_digest": proof["owner_contract_digest"]},
            "payloads": [{"payload_id": manifest["payload"]["file"], "member": manifest["payload"]["file"],
                          "sha256": payload_hash, "bytes": len(payload), "variants": manifest["stage"]["variants"],
                          "sharing": "all listed variants share this physical payload"}],
        }
    package_seconds = time.monotonic() - started
    cells = [{"id": f"mok-{p}-{precision}-ep{world}-c{clamp}", "pass": p,
              "precision": precision, "world": world, "clamp": clamp,
              "index": f"links/mok-{p}-{precision}-ep{world}-c{clamp}/index.json"}
             for p in PASSES for precision in ("bf16", "mxfp8") for world in (1, 4) for clamp in (0, 1)]

    def build_cell(cell):
        cell_started = time.monotonic()
        cell_id, pass_name = cell["id"], cell["pass"]
        cell_out = out / "links" / cell_id
        cell_out.mkdir(parents=True, exist_ok=True)
        packages = (f"framework.{pass_name}", *OPS)
        pipeline = json.loads((HERE / f"pipelines/{pass_name}.json").read_text())
        pipeline["pipelink_abi"] = manifests[packages[0]]["pipelink_abi"]
        body = pipeline["pipeline"]
        body["name"] = cell_id
        body["stages"] = [{"stage_id": manifests[p]["stage"]["id"], "artifact_path": f"../../{p}.plink"} for p in packages]
        body["forced_bindings"] = {
            "pipelink_mok_pass": PASSES.index(pass_name), "pipelink_mok_precision": int(cell["precision"] == "mxfp8"),
            "pipelink_mok_world": cell["world"], "pipelink_mok_clamp": cell["clamp"],
        }
        write_json(cell_out / "pipeline.json", pipeline)
        command([*cli, "link", cell_out / "pipeline.json", "--out-dir", cell_out], f"{cell_id}/link.log")
        link = json.loads((cell_out / "link.json").read_text())
        expected = dropped_endpoints(pass_name)
        observed = sorted(link.get("dropped_endpoints", []), key=lambda row: (row["stage"], row["endpoint"], row["reason"]))
        if observed != expected:
            write_json(cell_out / "drop-mismatch.json", {"expected": expected, "observed": observed})
            raise RuntimeError(f"{cell_id}: contract §5.2 dropped_endpoints multiset mismatch; see {cell_out / 'drop-mismatch.json'}")
        expansions = {row["channel"]: row for row in link["protocol_expansions"] if row["counter"] is not None}
        slots = {key.removeprefix("pipelink_mok_cnt_") for key in body["handles"] if key.startswith("pipelink_mok_cnt_")}
        if set(expansions) != slots:
            raise RuntimeError(f"{cell_id}: contract §5 live counter graph mismatch: {set(expansions)} != {slots}")
        if pass_name == "rec" and expansions["hidden_row_block_ready"]["counter"][2]:
            raise RuntimeError(f"{cell_id}: contract §5 recompute hidden counter must have no consumers")
        counter_graph = {}
        for owner in OPS:
            for endpoint in manifests[owner]["endpoints"]:
                if (endpoint["protocol"]["kind"] == "counter"
                        and f"mok_{owner}_{pass_name}" in endpoint.get("exports", [])):
                    directions = counter_graph.setdefault(endpoint["name"], {"produce": [], "consume": []})
                    directions[endpoint["direction"]].append(f"mok.{owner}")
        if pass_name == "bwd" and sorted(counter_graph["routed_buffers_done"]["produce"]) != ["mok.act", "mok.comm", "mok.gemm"]:
            raise RuntimeError(f"{cell_id}: contract §5 routed_buffers_done must retain all three producer stages")
        flags = [*FLAGS, "-O3", "--frandom-seed=mok.one_tu", f"-maxrregcount={link['regs_cap']}",
                 *[f"-I{inc}" for inc in includes],
                 f'-DMOK_BINDING_STUB="{cell_out / "generated/binding_stub.cu"}"',
                 f'-DMOK_INTERPRETER="{HERE / f"framework/interp_{pass_name}.cu"}"',
                 f'-DMOK_DISPATCH="{cell_out / "generated/dispatch.cu"}"']
        command([nvcc, *flags, "-arch=sm_103a", "--cubin", HERE / "one_tu.cu", "-o", cell_out / "one-tu.cubin"],
                f"{cell_id}/one-tu.log")
        command([nvcc, *flags, "-gencode=arch=compute_103a,code=lto_103a", "-dc", "-rdc=true",
                 HERE / "one_tu.cu", "-o", cell_out / "one-tu-lto.o"], f"{cell_id}/one-tu-lto.log")
        # Match the package link's LTO backend and dead-export policy. nvcc -dlink
        # retains out-of-line exports and applies the kernel cap to their bodies.
        lto_options = _link_options(get_profile("sm_103a"), link["regs_cap"], math_mode(FLAGS))
        write_json(logs / cell_id / "one-tu-lto-link-options.json", {
            "backend": "nvJitLink", "input": "one-tu-lto.o", "options": lto_options})
        try:
            cubin, info, error = _NvJitLink(cuda / "lib64/libnvJitLink.so.13").link(
                [("one-tu-lto.o", (cell_out / "one-tu-lto.o").read_bytes())], lto_options)
        except Exception as error:
            (logs / cell_id / "one-tu-lto-link.log").write_text(str(error))
            raise
        (cell_out / "one-tu-lto.cubin").write_bytes(cubin)
        (logs / cell_id / "one-tu-lto-link.log").write_text(info + error)
        diagnostics = {}
        for name in ("pipeline", "one-tu", "one-tu-lto"):
            resources = command([cuda / "bin/cuobjdump", "--dump-resource-usage", cell_out / f"{name}.cubin"],
                                f"{cell_id}/{name}.resources.log")
            sass = command([cuda / "bin/cuobjdump", "--dump-sass", cell_out / f"{name}.cubin"], f"{cell_id}/{name}.sass.log")
            diagnostics[name] = {"resource_usage": resources, "sass_counts": {
                op: len(re.findall(r"\b" + op + r"(?:\.[A-Z0-9_]+)*\b", sass)) for op in ("CALL", "STL", "LDL")}}
        index = {**{key: value for key, value in cell.items() if key != "index"},
                 "profile": "sm_103a", "link": "link.json", "linked": "pipeline.cubin",
                 "one_tu": "one-tu.cubin", "one_tu_lto": "one-tu-lto.cubin",
                 "one_tu_lto_linker": "nvJitLink",
                 "providers": {f"mok.{p.split('.')[0]}.params": f"../../provider_{p}.so" for p in packages},
                 "provider_abi_equal": {owner: True for owner in OWNERS},
                 "kernel_symbol": link["kernel_symbol"], "threads": link["threads"],
                 "block": [256, 1, 1], "cluster": [2, 1, 1], "dyn_smem_bytes": link["dyn_smem_bytes"],
                 "regs_cap": link["regs_cap"], "diagnostics": diagnostics}
        write_json(cell_out / "index.json", index)
        selected = []
        for p in packages:
            artifact = artifacts[p]
            stage_id = artifact["stage_id"]
            if link["input_hashes"][stage_id] != artifact["sha256"]:
                raise RuntimeError(f"{cell_id}: contract §11.2 artifact identity changed: {p}")
            variant_key = link["variants"][stage_id]
            variant = next(v for v in manifests[p]["stage"]["variants"] if v["key"] == variant_key)
            selected.append({"stage_id": stage_id, "artifact": artifact["path"], "artifact_sha256": artifact["sha256"],
                             "variant": variant, "payload_id": artifact["payloads"][0]["payload_id"],
                             "payload_sha256": artifact["payloads"][0]["sha256"], "provider": artifact["provider"]})
        elapsed = time.monotonic() - cell_started
        print(f"{cell_id}: link + plain/LTO controls passed, drops={len(observed)}, wall={elapsed:.3f}s", flush=True)
        return {"id": cell_id, "interpreter": selected[0], "ops": selected[1:],
                "link_sha256": sha256(cell_out / "link.json"), "cubin_sha256": sha256(cell_out / "pipeline.cubin"),
                "dropped_endpoints": {"expected": expected, "observed": observed, "match": True},
                "counter_graph": counter_graph, "wall_seconds": elapsed}

    links_started = time.monotonic()
    records, failures = {}, {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(build_cell, cell): cell["id"] for cell in cells}
        for future in as_completed(futures):
            cell_id = futures[future]
            try:
                records[cell_id] = future.result()
            except Exception as error:
                failures[cell_id] = str(error)
                print(f"{cell_id}: FAILED: {error}", file=sys.stderr, flush=True)
    timings = {"package_provider_seconds": package_seconds, "links_controls_seconds": time.monotonic() - links_started,
               "total_seconds": time.monotonic() - started, "jobs": args.jobs,
               "cells": {key: row["wall_seconds"] for key, row in sorted(records.items())}}
    write_json(out / "timings.json", timings)
    for name, digest in input_hashes.items():
        if sha256(out / name) != digest:
            raise RuntimeError(f"contract §11.2 input artifact/provider changed during matrix: {name}")
    if failures:
        write_json(out / "build-failures.json", failures)
        raise RuntimeError(f"{len(failures)}/24 cells failed; see {out / 'build-failures.json'}; wall={timings['total_seconds']:.3f}s")
    (out / "build-failures.json").unlink(missing_ok=True)
    fanout = []
    variant_counts = []
    for package, artifact in artifacts.items():
        consumed = [r["id"] for r in records.values() if any(s["stage_id"] == artifact["stage_id"] for s in [r["interpreter"], *r["ops"]])]
        artifact["links"] = sorted(consumed)
        expected_count = 8 if package.startswith("framework.") else 24
        if len(consumed) != expected_count:
            raise RuntimeError(f"{package}: contract §11.2 package fan-out mismatch")
        for payload in artifact["payloads"]:
            fanout.append({"stage_id": artifact["stage_id"], "payload_id": payload["payload_id"],
                           "payload_sha256": payload["sha256"], "links": sorted(consumed)})
        counts = Counter(s["variant"]["index"] for r in records.values() for s in [r["interpreter"], *r["ops"]]
                         if s["stage_id"] == artifact["stage_id"])
        expected_count = 2 if package.startswith("framework.") else (12 if package == "gemm" else 6)
        for variant in manifests[package]["stage"]["variants"]:
            count = counts[variant["index"]]
            if count != expected_count:
                raise RuntimeError(f"{package}: contract §11.2 variant {variant['index']} fan-out {count} != {expected_count}")
            variant_counts.append({"stage_id": artifact["stage_id"], "variant": variant,
                                   "links": count, "expected": expected_count})
    write_json(out / "payload-fanout.json", {"schema": "pipelink-mok-payload-fanout-v1", **provenance,
               "artifacts": list(artifacts.values()), "links": [records[c["id"]] for c in cells],
               "fanout": fanout, "variant_selection_counts": variant_counts, "inputs_unchanged": True})
    write_json(out / "matrix.json", {"default_cell": DEFAULT_CELL, "cells": cells,
                                    "payload_fanout": "payload-fanout.json", "timings": "timings.json"})
    default_dir = out / "links" / DEFAULT_CELL
    index = json.loads((default_dir / "index.json").read_text())
    for key in ("link", "linked", "one_tu", "one_tu_lto"):
        index[key] = str((default_dir / index[key]).relative_to(out))
    index["providers"] = {key: str((default_dir / value).resolve().relative_to(out)) for key, value in index["providers"].items()}
    index.update(default_cell=DEFAULT_CELL, matrix="matrix.json", payload_fanout="payload-fanout.json")
    write_json(out / "index.json", index)
    print(f"24/24 links + plain/LTO controls; one payload per op reused 24 times; wall={timings['total_seconds']:.3f}s "
          f"(matrix={timings['links_controls_seconds']:.3f}s, jobs={args.jobs})", flush=True)


if __name__ == "__main__":
    main()
