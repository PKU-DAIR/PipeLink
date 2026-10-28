#!/usr/bin/env python3
"""Broker launches, numeric matrix, same-source parity and placement measurements."""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipec/src"))
from params import (BF16, PROFILES, SHAPES, SMALL_SHAPES, FULL_SHAPES,
                    inputs, references, sample_coordinates, check, check_unfused, semantic_difference, self_check)
from evaluator.harness.parity.example import clock_lease, time_pair, sanitize, sass_diagnostic
from evaluator.harness.parity.native import open_native
from pipec2.host.broker import open_host
from pipec2.host.cuda import CudaBackend, reload_modules
from pipec2.host.fixture import Fixture
from pipec2.host.profile import DTYPES
from pipec2.model import LinkResult
from tools.pipelink_env import load

DEFAULT_STAGGER_NS = 22500
_PROGRESS_PATH = None


def clock_locked_mode(env):
    """True when the environment requests a locked SM clock (empty or 0 means unlocked)."""
    return str(env.get("PIPELINK_LOCK_SM_MHZ") or "").strip() not in ("", "0")


def lease_locked(record):
    """Whether a clock-lease record held a locked clock; unlocked records carry no frequency."""
    return bool(record.get("locked", record.get("sm_mhz") is not None))


def progress(operation, **context):
    """Flushed phase evidence outside event-timed launch loops."""
    line = json.dumps({"time_ns": time.time_ns(), "pid": os.getpid(),
                       "operation": operation, **context}, default=str)
    print("[coda-progress] " + line, file=sys.stderr, flush=True)
    if _PROGRESS_PATH is not None:
        with _PROGRESS_PATH.open("a") as stream:
            stream.write(line + "\n")


def checkpoint(out, results):
    temporary = out / "results-partial.json.tmp"
    temporary.write_text(json.dumps({"complete": False, "results": results}, indent=2) + "\n")
    temporary.replace(out / "results-partial.json")


def arm_ms(pair, label):
    return float(np.median([sum(t for a, t in zip(b["order"], b["milliseconds"]) if a == label)
                            for b in pair["blocks"]])) / (2 * pair["repetitions"])


class BrokerImage:
    """Timing adapter over a broker session; two images share resources and params."""
    def __init__(self, session, backend, image, resource, shape):
        self.session, self.backend, self.image = session, backend, image
        self.resource, self.shape = resource, shape

    def launch(self):
        self.session.launch(self.backend.stream_token, image=self.image)

    def time(self, n):
        b = self.backend
        start, end = b.event_create(), b.event_create()
        try:
            b.event_record(start, b.stream_token)
            for _ in range(n):
                self.launch()
            b.event_record(end, b.stream_token)
            b.event_synchronize(end)
            return b.event_elapsed_time(start, end)
        finally:
            b.event_destroy(end)
            b.event_destroy(start)

    def read(self, name="out"):
        self.backend.synchronize()
        return np.frombuffer(self.backend.readback(self.resource.address, self.resource.allocation_bytes),
                             dtype=BF16).reshape(self.shape)

    def poison(self):
        self.backend.synchronize()
        self.backend.upload(self.resource.address, np.full(self.shape, np.nan, dtype=BF16).tobytes())


@contextmanager
def linked_images(record, fixture_data, shape, *, images=("linked", "one_tu"), stagger_ns=DEFAULT_STAGGER_NS):
    progress("linked.open.begin", link=record["link"], shape=shape, images=images)
    result = LinkResult.from_json_obj(json.loads((REPO / record["link"]).read_text()))
    kwargs = dict(device=0, dynamic_smem_bytes=result.dyn_smem_bytes,
                  cluster=tuple(result.cluster), block=(result.threads, 1, 1))
    with ExitStack() as stack:
        backend = stack.enter_context(CudaBackend(REPO / record["linked"], result.kernel_symbol, **kwargs))
        progress("linked.backend.ready", link=record["link"])
        fixture = stack.enter_context(Fixture(backend))
        for name, value in fixture_data.items():
            fixture.tensor(name, data=value.tobytes(), allocation_bytes=value.nbytes,
                           extents=value.shape, bit_strides=tuple(s * 8 for s in value.strides),
                           dtype_id=DTYPES["bf16"]["id"], alignment=128)
        progress("linked.buffers.ready", link=record["link"])
        resources = fixture.resources
        scalars = dict(zip(("M", "N", "K"), shape), stagger_ns=stagger_ns)
        session = stack.enter_context(open_host(result, {k: REPO / v for k, v in record["providers"].items()},
                                               {}, resources, scalars, backend))
        adapters = {}
        for image in images:
            module = backend if image == "linked" else stack.enter_context(
                CudaBackend(REPO / record[image], result.kernel_symbol, **kwargs))
            adapters[image] = BrokerImage(session, backend, module, fixture.resources["out"],
                                         (shape[0], shape[1] // 2))
        progress("linked.open.complete", link=record["link"], geometry=session.geometry)
        yield adapters, session
        progress("linked.close.begin", link=record["link"])
    progress("linked.close.complete", link=record["link"])


def native_session(index, kind, fixture_data, shape, *, stream=None, stagger_ns=None):
    record = index["natives"]["unfused_gemm" if kind == "unfused" else kind]
    spec = importlib.util.spec_from_file_location("coda_native_params", REPO / index["native_builder"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    M, N, K = shape
    driver = C.CDLL("libcuda.so.1")
    driver.cuInit(0)
    num_sms = C.c_int()
    status = driver.cuDeviceGetAttribute(C.byref(num_sms), 16, 0)
    if status:
        raise RuntimeError(f"cuDeviceGetAttribute failed: {status}")
    grid = eval(record["grid"], {"__builtins__": {}},
                {"M": M, "N": N, "K": K, "num_sms": num_sms.value, "min": min, "max": max})
    buffers = dict(fixture_data)
    following = ()
    native_kind = "unfused_gemm" if kind == "unfused" else kind
    if native_kind == "unfused_gemm":
        buffers["d"] = np.full((M, N), np.nan, dtype=BF16)
    if kind == "unfused":
        epilogue = index["standalone"]
        following = ({"cubin": REPO / epilogue["cubin"], "kernel": epilogue["kernel"],
                      "threads": epilogue["threads"], "dyn_smem": epilogue["dyn_smem"],
                      "cluster": tuple(epilogue["cluster"]), "grid": ((M * N + 2047) // 2048, 1, 1),
                      "arguments": lambda a: [C.c_uint64(a["d"]), C.c_uint64(a["out"]), C.c_int(M), C.c_int(N)]},)
    return open_native(REPO / record["cubin"], record["kernel"], buffers=buffers,
                       arguments=lambda a: module.arguments(native_kind, a, M, N, K, stagger_ns=stagger_ns),
                       grid=tuple(grid), threads=record["threads"], cluster=tuple(record["cluster"]),
                       dyn_smem=record["dyn_smem"], device=0, stream=stream, following=following)


def verify(record, shape, images=("linked", "one_tu"), *, stagger_ns=DEFAULT_STAGGER_NS):
    progress("verify.inputs.begin", link=record["link"], shape=shape, images=images)
    buffers = inputs(*shape)
    progress("verify.oracle.begin", link=record["link"], shape=shape)
    coordinates = sample_coordinates(*shape[:2]) if shape in FULL_SHAPES else None
    refs = references(buffers, coordinates)
    progress("verify.oracle.complete", link=record["link"], shape=shape)
    checks, outputs = {}, {}
    with linked_images(record, buffers, shape, images=images, stagger_ns=stagger_ns) as (sessions, session):
        for image, adapter in sessions.items():
            progress("verify.poison.begin", link=record["link"], shape=shape, image=image)
            adapter.poison()
            progress("verify.launch.begin", link=record["link"], shape=shape, image=image)
            adapter.launch()
            progress("verify.read.begin", link=record["link"], shape=shape, image=image)
            outputs[image] = adapter.read()
            checks[image] = check(outputs[image], refs["fused"], coordinates)
            progress("verify.image.complete", link=record["link"], shape=shape, image=image,
                     status=checks[image]["status"])
        geometry = session.geometry
    bitwise = (np.array_equal(outputs["linked"].view(np.uint16), outputs["one_tu"].view(np.uint16))
               if len(outputs) == 2 else None)
    return {"shape": list(shape), "checks": checks, "bitwise_full_output": bitwise,
            "geometry": geometry, "semantic_rounding": semantic_difference(refs),
            "status": "PASS" if bitwise is not False and all(c["status"] == "PASS" for c in checks.values()) else "FAIL"}


def time_placement(linked, index, kind, buffers, shape, refs, coordinates, gpu, *, stagger_ns):
    progress("placement.open.begin", kind=kind, shape=shape, stagger_ns=stagger_ns)
    with native_session(index, kind, buffers, shape, stream=linked.backend.stream_token,
                        stagger_ns=stagger_ns) as native:
        progress("placement.launch.begin", kind=kind, shape=shape)
        native.launch()
        progress("placement.read.begin", kind=kind, shape=shape)
        actual = native.read("out")
        numeric = (check_unfused(actual, refs["unfused"], buffers, coordinates) if kind == "unfused"
                   else check(actual, refs["fused"], coordinates))
        if numeric["status"] != "PASS":
            return {"check": numeric, "status": "FAIL"}
        progress("placement.timing.begin", kind=kind, shape=shape)
        pair = time_pair(linked, native, gpu=gpu, sample_clocks=True)
        progress("placement.timing.complete", kind=kind, shape=shape, status=pair["status"])
        return {"check": numeric, "pair": pair, "milliseconds": arm_ms(pair, "B"),
                "speedup_over_baseline": pair["ratio_onetu_over_linked"],
                "diagnostic_only": True, "status": pair["status"]}


def benchmark(record, index, shape, gpu, *, stagger_ns=DEFAULT_STAGGER_NS,
              placements=("fused", "unfused")):
    progress("benchmark.begin", link=record["link"], shape=shape, stagger_ns=stagger_ns)
    buffers = inputs(*shape)
    coordinates = sample_coordinates(*shape[:2])
    refs = references(buffers, coordinates)
    with linked_images(record, buffers, shape, stagger_ns=stagger_ns) as (sessions, session):
        a, b = sessions["linked"], sessions["one_tu"]
        outputs = []
        for image, adapter in (("linked", a), ("one_tu", b)):
            progress("benchmark.check.begin", link=record["link"], shape=shape, image=image)
            adapter.poison()
            adapter.launch()
            progress("benchmark.read.begin", link=record["link"], shape=shape, image=image)
            outputs.append(adapter.read())
        checks = [check(value, refs["fused"], coordinates) for value in outputs]
        bitwise = np.array_equal(outputs[0].view(np.uint16), outputs[1].view(np.uint16))
        if not bitwise or any(c["status"] != "PASS" for c in checks):
            raise RuntimeError(f"timing prerequisite failed: bitwise={bitwise}, checks={checks}")
        reload_checks = {}

        def reload_pair():
            reload_modules((b.image, a.image))
            for name, adapter in (("linked", a), ("one_tu", b)):
                adapter.poison()
                adapter.launch()
                equal = np.array_equal(adapter.read().view(np.uint16), outputs[0].view(np.uint16))
                reload_checks[name] = bool(equal)
                if not equal:
                    raise RuntimeError(f"post-reload output mismatch: {name}")

        progress("benchmark.parity.begin", link=record["link"], shape=shape)
        pair = time_pair(a, b, repetitions=10, gpu=gpu, sample_clocks=True, reload=reload_pair)
        progress("benchmark.parity.complete", link=record["link"], shape=shape, status=pair["status"])
        result = {"shape": list(shape), "stagger_ns": stagger_ns, "pair": pair,
                  "reload_bitwise": reload_checks,
                  "linked_ms": arm_ms(pair, "A"), "one_tu_ms": arm_ms(pair, "B"),
                  "parity_pass": pair["ratio_onetu_over_linked"] >= .97,
                  "within_2_percent": .98 <= pair["ratio_onetu_over_linked"] <= 1.02,
                  "status": "PASS" if pair["status"] == "PASS" and pair["ratio_onetu_over_linked"] >= .97 else "FAIL",
                  "placements": {}}
        for kind in placements:
            placement = time_placement(a, index, kind, buffers, shape, refs, coordinates, gpu,
                                       stagger_ns=stagger_ns)
            result["placements"][kind] = placement
            if placement["status"] != "PASS":
                result["status"] = "FAIL"
        if result["placements"].get("unfused", {}).get("check", {}).get("status") == "PASS":
            with native_session(index, "unfused_gemm", buffers, shape, stream=a.backend.stream_token,
                                stagger_ns=stagger_ns) as gemm:
                progress("benchmark.gemm.begin", link=record["link"], shape=shape)
                pair = time_pair(a, gemm, gpu=gpu, sample_clocks=True)
                progress("benchmark.gemm.complete", link=record["link"], shape=shape, status=pair["status"])
                result["producer_gemm"] = {"milliseconds": arm_ms(pair, "B"), "pair": pair,
                    "correctness": "same GEMM checked in unfused sequence", "diagnostic_only": True}
                if pair["status"] != "PASS":
                    result["status"] = "FAIL"
    return result


def public_results(args, env, out):
    destination = out / ("public-unlocked.json" if args.unlocked else "public-results.json")
    command = ["uv", "run", "--project", str(HERE / "public"), "--locked", "python",
               str(HERE / "public/run.py"), "--profile", args.profile, "--gpu", str(args.gpu),
               "--output", str(destination), "--unlocked" if args.unlocked else "--external-lease"]
    label = getattr(args, "measurement_label", None)
    if label:
        command.extend(("--measurement-label", label))
    if args.unlocked and not getattr(args, "public_only", False):
        command.extend(("--index", str(args.build_dir / "index.json"),
                        "--stagger-ns", str(args.stagger_ns)))
    progress("public.begin", profile=args.profile, gpu=args.gpu, output=destination)
    with (out / "public.log").open("wb") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    progress("public.complete", profile=args.profile, returncode=completed.returncode)
    if completed.returncode:
        raise RuntimeError(f"public sequence failed; see {out / 'public.log'}")
    return json.loads(destination.read_text())


def measure(args, env, out, index, *, functional_only=False):
    results = {"profile": args.profile, "cells": {}, "natives": {},
               "mode": ("functional-only" if functional_only else "unlocked" if args.unlocked
                        else "locked" if clock_locked_mode(env) else "unlocked-clock")}
    results["measurement_label"] = getattr(args, "measurement_label", None)
    for name, cell in index["cells"].items():
        evidence = out / "evidence" / name
        row = {"correctness": [], "sanitizers": []}
        results["cells"][name] = row
        checkpoint(out, results)
        for shape in SHAPES:
            row["correctness"].append(verify(cell, shape, stagger_ns=args.stagger_ns))
            checkpoint(out, results)
        for shape in SMALL_SHAPES[:3]:
            for image in ("linked", "one_tu"):
                command = [sys.executable, str(HERE / "run.py"), "--profile", args.profile, "--gpu", str(args.gpu),
                           "--build-dir", str(args.build_dir), "--single", name, image, "--shape", *map(str, shape), "--stagger-ns", str(args.stagger_ns)]
                for tool in ("memcheck", "racecheck", "synccheck"):
                    progress("sanitizer.begin", cell=name, image=image, shape=shape, tool=tool)
                    row["sanitizers"] += [{**r, "image": image, "shape": list(shape)} for r in sanitize(
                        command, env=env, log_dir=evidence / "sanitizers" / image / "x".join(map(str, shape)),
                        tools=(tool,))]
                    checkpoint(out, results)
                    progress("sanitizer.complete", cell=name, image=image, shape=shape, tool=tool)
        row["sass"] = sass_diagnostic(REPO / cell["linked"], REPO / cell["one_tu"], env=env,
                                      out_dir=evidence / "sass", backend_matched=REPO / cell["one_tu_lto"])
        distribution = cell["source_withheld"]
        workspace = Path(tempfile.mkdtemp(prefix=f"coda-{name}-launch-", dir=env["PIPELINK_SCRATCH_ROOT"]))
        shutil.unpack_archive(REPO / distribution["bundle"], workspace)
        opaque = {"link": str(workspace / "link/link.json"), "linked": str(workspace / "link/pipeline.cubin"),
                  "providers": {group: str(workspace / "received" / filename)
                                for group, filename in distribution["provider_files"].items()}}
        invocation = workspace / "launch.json"
        invocation.write_text(json.dumps(opaque, indent=2) + "\n")
        completed = subprocess.run(
            [sys.executable, str(HERE / "run.py"), "--profile", args.profile, "--gpu", str(args.gpu),
             "--opaque-record", str(invocation), "--stagger-ns", str(args.stagger_ns)],
            cwd=workspace, env=env, capture_output=True)
        (workspace / "launch.log").write_bytes(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"source-withheld launch failed: {workspace / 'launch.log'}")
        row["source_withheld"] = json.loads(completed.stdout)
        row["source_withheld"]["workspace"] = str(workspace)
        checkpoint(out, results)
        row["timings"] = []
        if not functional_only:
            for shape in FULL_SHAPES:
                row["timings"].append(benchmark(cell, index, shape, args.gpu, stagger_ns=args.stagger_ns))
                checkpoint(out, results)
            if args.profile == "sm103":
                row["stagger_sensitivity"] = []
                for shape in FULL_SHAPES:
                    row["stagger_sensitivity"].append(benchmark(cell, index, shape, args.gpu, stagger_ns=0))
                    checkpoint(out, results)
        row["status"] = "PASS" if all(r["status"] == "PASS" for r in
            [*row["correctness"], *row["sanitizers"], *row["timings"], row["source_withheld"],
             *row.get("stagger_sensitivity", [])]) else "FAIL"
        results["cells"][name] = row
    # The native matrix has its own references (including bf16 D), separate from the fused linked oracle.
    for kind in ("fused", "unfused"):
        rows = []
        results["natives"][kind] = rows
        for shape in SHAPES:
            progress("native.check.begin", kind=kind, shape=shape)
            buffers = inputs(*shape)
            coordinates = sample_coordinates(*shape[:2]) if shape in FULL_SHAPES else None
            refs = references(buffers, coordinates)
            with native_session(index, kind, buffers, shape, stagger_ns=args.stagger_ns) as session:
                session.launch()
                progress("native.read.begin", kind=kind, shape=shape)
                actual = session.read("out")
                numeric = (check_unfused(actual, refs["unfused"], buffers, coordinates) if kind == "unfused"
                           else check(actual, refs["fused"], coordinates))
                rows.append({"shape": list(shape), **numeric})
                checkpoint(out, results)
                progress("native.check.complete", kind=kind, shape=shape, status=numeric["status"])
        results["natives"][kind] = rows
    if functional_only:
        results["status"] = "PASS" if (
            all(c["status"] == "PASS" for c in results["cells"].values()) and
            all(r["status"] == "PASS" for rows in results["natives"].values() for r in rows)) else "FAIL"
        return results
    public = results["public"] = public_results(args, env, out)
    attach_public_placement(results)
    results["status"] = "PASS" if (public["status"] == "PASS" and
        all(c["status"] == "PASS" for c in results["cells"].values()) and
        all(r["status"] == "PASS" for rows in results["natives"].values() for r in rows)) else "FAIL"
    return results


def attach_public_placement(results):
    public = results["public"]
    for cell_name, cell in results["cells"].items():
        for timing in cell["timings"]:
            baseline = next(row for row in public["cases"] if row["shape"] == timing["shape"])
            if "paired_placement" in baseline:
                direct = baseline["paired_placement"]
                timing["public_placement"] = direct["cells"][cell_name]["public"]
                for comparison in timing["public_placement"].values():
                    comparison["telemetry"] = {key: value for key, value in
                        comparison["pair"]["telemetry"]["B"].items() if key != "trace"}
                if "producer_gemm" in timing and timing["producer_gemm"].get("timing_arm", "B") == "B":
                    timing["producer_gemm_linked_diagnostic"] = {
                        key: value for key, value in timing["producer_gemm"].items() if key != "speedup_over_cublas"}
                timing["producer_gemm"] = {**direct["producer_gemm"], "timing_arm": "A"}
                continue
            if results.get("mode") == "unlocked":
                raise ValueError("unlocked placement requires direct linked/public and producer/cuBLAS pairs")
            timing["public_placement"] = {name: {"milliseconds": ms,
                "speedup_over_baseline": ms / timing["linked_ms"], "diagnostic_only": True,
                "telemetry": {key: value for key, value in
                    baseline["pair"]["telemetry"]["A" if name == "compile" else "B"].items() if key != "trace"}}
                for name, ms in baseline["milliseconds"].items()}
            if "producer_gemm" in timing:
                timing["producer_gemm"]["speedup_over_cublas"] = (
                    baseline["gemm_only_ms"]["interleaved"] / timing["producer_gemm"]["milliseconds"])


def leased_timing(args, env, out, index):
    """Run functional checks (or reuse a receipt), then time under one clock lease.

    A reused receipt is valid only for unchanged stage sources and artifacts.
    Without one, the full functional matrix runs first, with no clock claim.
    The lease covers the same-source parity pairs, native placement and the
    public sequences.
    """
    if args.reuse_correctness:
        results = json.loads(args.reuse_correctness.read_text())
        results["correctness_reused_from"] = str(args.reuse_correctness)
    else:
        results = measure(args, env, out, index, functional_only=True)
        receipt = out / "functional-results.json"
        receipt.write_text(json.dumps(results, indent=2) + "\n")
        results["correctness_collected_in"] = str(receipt)
    functional = [r for cell in results["cells"].values()
                  for r in [*cell["correctness"], *cell["sanitizers"], cell["source_withheld"]]]
    functional += [r for rows in results["natives"].values() for r in rows]
    if results["profile"] != args.profile or not all(r["status"] == "PASS" for r in functional):
        raise ValueError("timing requires matching-profile passing numerical/sanitizer/source-withheld evidence")
    results["mode"] = "locked" if clock_locked_mode(env) else "unlocked-clock"
    results["measurement_label"] = getattr(args, "measurement_label", None)
    results["timing_protocol"] = (
        "Same-source parity: paired-reload-v2, eight blocks, midpoint reload, four batches per block, "
        "ten fixed repetitions; positive finite elapsed values, qualified lease and no foreign PIDs. "
        "Placement retains ordinary paired timing.")
    with clock_lease(args.gpu, env=env, log_dir=out / "clock-primary") as primary_clock:
        for name, cell in index["cells"].items():
            entry = results["cells"][name]
            entry["timings"] = []
            for shape in FULL_SHAPES:
                entry["timings"].append(benchmark(cell, index, shape, args.gpu,
                    stagger_ns=args.stagger_ns))
                checkpoint(out, results)
            if args.profile == "sm103":
                entry["stagger_sensitivity"] = []
                for shape in FULL_SHAPES:
                    entry["stagger_sensitivity"].append(benchmark(cell, index, shape, args.gpu,
                        stagger_ns=0))
                    checkpoint(out, results)
            entry["status"] = "PASS" if all(r["status"] == "PASS" for r in
                [*entry["timings"], *entry.get("stagger_sensitivity", [])]) else "FAIL"
        results["public"] = public_results(args, env, out)
    results["clock"] = primary_clock
    results["clock_locked"] = lease_locked(primary_clock)
    attach_public_placement(results)
    rows = [(name, row) for name, cell in results["cells"].items()
            for row in [*cell["timings"], *cell.get("stagger_sensitivity", [])]]
    pairs = [row["pair"] for _, row in rows]
    pairs += [p["pair"] for _, row in rows for p in row["placements"].values() if "pair" in p]
    pairs += [row["producer_gemm"]["pair"] for _, row in rows if "producer_gemm" in row]
    pairs += [case[key] for case in results["public"]["cases"]
              for key in ("pair", "gemm_pair") if key in case]
    primary_admitted = primary_clock["healthy"] and all(not p.get("foreign_pids") for p in pairs)
    results["timing_admission"] = {"primary": primary_admitted}
    for case in results["public"]["cases"]:
        case["admitted"] = primary_admitted and case["status"] == "PASS"
    for _, row in rows:
        row["admitted"] = primary_admitted and row["pair"]["status"] == "PASS"
        for placement in [*row["placements"].values(), *row.get("public_placement", {}).values()]:
            placement["clock_lease"] = "primary"
            placement["admitted"] = primary_admitted and placement.get("status", "PASS") == "PASS"
        if "producer_gemm" in row:
            row["producer_gemm"]["admitted"] = (
                primary_admitted and row["producer_gemm"]["pair"]["status"] == "PASS")
    results["status"] = "PASS" if (primary_admitted and results["public"]["status"] == "PASS"
        and all(cell["status"] == "PASS" for cell in results["cells"].values())) else "FAIL"
    return results


def main():
    global _PROGRESS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="sm103")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--single", nargs=2, metavar=("CELL", "IMAGE"))
    parser.add_argument("--opaque-record", type=Path, help="verify a linked-only invocation record from a distribution without owner sources")
    parser.add_argument("--shape", nargs=3, type=int, default=SMALL_SHAPES[0])
    parser.add_argument("--stagger-ns", type=int, default=DEFAULT_STAGGER_NS)
    parser.add_argument("--lock-sm-mhz", type=int, metavar="MHZ",
                        help="SM clock to lock; overrides PIPELINK_LOCK_SM_MHZ, 0 means unlocked")
    parser.add_argument("--unlocked", action="store_true",
                        help="unlocked sensitivity protocol: no clock lease, public sequences paired directly")
    parser.add_argument("--build-dir", type=Path, help="build directory from build.py (default: out/<profile>)")
    parser.add_argument("--measurement-label", help="measurement regime label, recorded verbatim (it does not assert a clock lock)")
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--reuse-correctness", type=Path,
                        help="passing functional receipt to reuse; valid only for unchanged stage sources and artifacts")
    parser.add_argument("--results-dir", type=Path, help="new output directory for results; build artifacts stay in out/<profile>")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
        return 0
    env = {**os.environ, **load(REPO)}
    if args.lock_sm_mhz is not None:
        env["PIPELINK_LOCK_SM_MHZ"] = str(args.lock_sm_mhz) if args.lock_sm_mhz else ""
    if args.unlocked:
        env["PIPELINK_LOCK_SM_MHZ"] = ""
    args.gpu = args.gpu if args.gpu is not None else int(env["PIPELINK_GPUS"].split(",")[0])
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.update(env)
    build_out = args.build_dir = (args.build_dir or HERE / "out" / args.profile).resolve()
    out = args.results_dir.resolve() if args.results_dir else build_out
    out.mkdir(parents=True, exist_ok=True)
    _PROGRESS_PATH = out / "progress.jsonl"
    progress("run.begin", profile=args.profile, gpu=args.gpu, single=args.single, shape=args.shape)
    if args.opaque_record:
        record = json.loads(args.opaque_record.read_text())
        result = verify(record, tuple(args.shape), ("linked",), stagger_ns=args.stagger_ns)
        print(json.dumps(result))
        return 0 if result["status"] == "PASS" else 1
    if args.single:
        index = json.loads((build_out / "index.json").read_text())
        name, image = args.single
        result = verify(index["cells"][name], tuple(args.shape), (image,), stagger_ns=args.stagger_ns)
        print(json.dumps(result))
        return 0 if result["status"] == "PASS" else 1
    if args.reuse_correctness and (args.unlocked or args.public_only):
        parser.error("--reuse-correctness applies to the leased timing protocol; "
                     "it excludes --unlocked and --public-only")
    if not args.unlocked and not args.public_only:
        index = json.loads((build_out / "index.json").read_text())
        results = leased_timing(args, env, out, index)
    else:
        context = nullcontext({"unlocked": True}) if args.unlocked else clock_lease(args.gpu, env=env, log_dir=out / "clock")
        with context as lease:
            if args.public_only:
                results = public_results(args, env, out)
            else:
                index = json.loads((build_out / "index.json").read_text())
                results = measure(args, env, out, index)
        results["clock"] = lease
        results["clock_locked"] = not args.unlocked and lease_locked(lease)
        if not args.unlocked and not lease["healthy"]:
            results["status"] = "FAIL"
    destination = out / ("public-results.json" if args.public_only else
                         "results-unlocked.json" if args.unlocked else "results.json")
    destination.write_text(json.dumps(results, indent=2) + "\n")
    progress("run.complete", status=results["status"], receipt=destination)
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
