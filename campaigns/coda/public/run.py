#!/usr/bin/env python3
"""Timed end-to-end public sequences; preparation is outside timing, copies inside."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))
from params import (FULL_SHAPES, SMALL_SHAPES, PROFILES, inputs, references,
                    sample_coordinates, check_unfused, semantic_difference)
from evaluator.harness.parity.example import clock_lease, time_pair
from evaluator.harness.parity.telemetry import ClockSampler
from tools.pipelink_env import load

LABELS = {"compile": "cuBLAS + torch.compile, interleaved",
          "liger": "cuBLAS + Liger, concatenated output with timed split-layout adaptation"}


def arm_ms(pair, label):
    import numpy as np
    return float(np.median([sum(t for a, t in zip(b["order"], b["milliseconds"]) if a == label)
                            for b in pair["blocks"]])) / (2 * pair["repetitions"])


class Sequence:
    def __init__(self, launch, torch):
        self.launch, self.torch = launch, torch

    def time(self, count):
        start, end = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(count):
            self.launch()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)


def sequences(x, wi, d, out):
    """Return the exact timed callables on caller-owned CUDA tensors.

    Concatenated weights are prepared once here, outside every launch. The
    returned closures retain them. Warm up compile/Liger before sustained timing.
    """
    import torch
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    wc = torch.cat((wi[0::2], wi[1::2]), dim=0)
    width = d.shape[1] // 2

    @torch.compile(fullgraph=True)
    def epilogue(value, output):
        gate, up = value[:, 0::2].float(), value[:, 1::2].float()
        output.copy_((gate / (1 + torch.exp(-gate))) * up)

    def compiled():
        torch.mm(x, wi.T, out=d)
        epilogue(d, out)
        return out

    def liger():
        torch.mm(x, wc.T, out=d)
        # Both row-major halves have a full-width row stride. These adaptations
        # and Liger's output allocation are part of every measured sequence.
        return LigerSiLUMulFunction.apply(d[:, :width].contiguous(), d[:, width:].contiguous())

    return {"compile": compiled, "liger": liger,
            "gemm_interleaved": lambda: torch.mm(x, wi.T, out=d),
            "gemm_concatenated": lambda: torch.mm(x, wc.T, out=d)}


def measure(shape, *, gpu, timing, index=None, stagger_ns=22500):
    import numpy as np
    import torch
    fixture = inputs(*shape)
    coordinates = sample_coordinates(*shape[:2]) if shape in FULL_SHAPES else None
    refs = references(fixture, coordinates)
    x = torch.from_numpy(fixture["x"].view(np.uint16)).view(torch.bfloat16).cuda()
    wi = torch.from_numpy(fixture["w"].view(np.uint16)).view(torch.bfloat16).cuda()
    d = torch.empty((shape[0], shape[1]), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((shape[0], shape[1] // 2), dtype=torch.bfloat16, device="cuda")
    programs = sequences(x, wi, d, out)

    checks = {}
    for label in ("compile", "liger"):
        function = programs[label]
        d.fill_(float("nan"))
        out.fill_(float("nan"))
        actual = function().detach().view(torch.uint16).cpu().numpy().view(fixture["out"].dtype)
        checks[label] = check_unfused(actual, refs["unfused"], fixture, coordinates)
    result = {"shape": list(shape), "checks": checks, "semantic_rounding": semantic_difference(refs),
              "status": "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "FAIL"}
    if timing and result["status"] == "PASS":
        pair = time_pair(Sequence(programs["compile"], torch), Sequence(programs["liger"], torch),
                         gpu=gpu, sample_clocks=True)
        result["pair"] = pair
        result["milliseconds"] = {"compile": arm_ms(pair, "A"), "liger": arm_ms(pair, "B")}
        gemm_pair = time_pair(Sequence(programs["gemm_interleaved"], torch),
                              Sequence(programs["gemm_concatenated"], torch), gpu=gpu, sample_clocks=True)
        result["gemm_only_ms"] = {"interleaved": arm_ms(gemm_pair, "A"),
                                  "concatenated": arm_ms(gemm_pair, "B")}
        result["gemm_pair"] = gemm_pair
        result["status"] = "PASS" if pair["status"] == gemm_pair["status"] == "PASS" else "FAIL"
        if index is not None:
            result["paired_placement"] = paired_placement(
                index, fixture, shape, refs, coordinates, programs, gpu, stagger_ns=stagger_ns)
            if result["paired_placement"]["status"] != "PASS":
                result["status"] = "FAIL"
    del programs, function, x, wi, d, out
    torch.cuda.empty_cache()
    return result


def paired_placement(index, fixture, shape, refs, coordinates, programs, gpu, *, stagger_ns):
    """Direct linked/public and producer/cuBLAS pairs on the same GPU and stream."""
    import torch
    import run as harness

    result = {"cells": {}, "status": "PASS"}
    for name, cell in index["cells"].items():
        with harness.linked_images(cell, fixture, shape, images=("linked",),
                                   stagger_ns=stagger_ns) as (images, _session):
            linked = images["linked"]
            linked.poison()
            linked.launch()
            numeric = harness.check(linked.read(), refs["fused"], coordinates)
            comparisons = {"check": numeric, "public": {}}
            result["cells"][name] = comparisons
            if numeric["status"] != "PASS":
                result["status"] = "FAIL"
                continue
            with torch.cuda.stream(torch.cuda.ExternalStream(linked.backend.stream_token, device=0)):
                for label in ("compile", "liger"):
                    pair = time_pair(linked, Sequence(programs[label], torch), gpu=gpu, sample_clocks=True)
                    comparisons["public"][label] = {
                        "pair": pair, "linked_ms": arm_ms(pair, "A"),
                        "milliseconds": arm_ms(pair, "B"),
                        "speedup_over_baseline": pair["ratio_onetu_over_linked"], "diagnostic_only": True}
                    if pair["status"] != "PASS":
                        result["status"] = "FAIL"
    stream = torch.cuda.current_stream().cuda_stream
    with harness.native_session(index, "unfused", fixture, shape, stream=stream,
                                stagger_ns=stagger_ns) as sequence:
        sequence.launch()
        numeric = check_unfused(sequence.read("out"), refs["unfused"], fixture, coordinates)
    quality = result["producer_gemm"] = {"check": numeric}
    if numeric["status"] != "PASS":
        result["status"] = "FAIL"
        return result
    with harness.native_session(index, "unfused_gemm", fixture, shape, stream=stream,
                                stagger_ns=stagger_ns) as gemm:
        pair = time_pair(gemm, Sequence(programs["gemm_interleaved"], torch),
                         gpu=gpu, sample_clocks=True)
        quality.update(pair=pair, milliseconds=arm_ms(pair, "A"), cublas_ms=arm_ms(pair, "B"),
                       speedup_over_cublas=pair["ratio_onetu_over_linked"], diagnostic_only=True)
        if pair["status"] != "PASS":
            result["status"] = "FAIL"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="sm103")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-lease", action="store_true", help="caller holds the same GPU's clock lease")
    parser.add_argument("--unlocked", action="store_true", help="no clock lease (sensitivity protocol)")
    parser.add_argument("--lock-sm-mhz", type=int, metavar="MHZ",
                        help="SM clock to lock; overrides PIPELINK_LOCK_SM_MHZ, 0 means unlocked")
    parser.add_argument("--measurement-label", help="measurement regime label, recorded verbatim")
    parser.add_argument("--index", type=Path, help="also directly pair every linked cell/public sequence and producer/cuBLAS")
    parser.add_argument("--stagger-ns", type=int, default=22500)
    parser.add_argument("--merge-results", type=Path,
                        help="add supplemental pairs to an existing results matrix, keeping its same-source parity results")
    args = parser.parse_args()
    env = {**os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    if args.lock_sm_mhz is not None:
        env["PIPELINK_LOCK_SM_MHZ"] = str(args.lock_sm_mhz) if args.lock_sm_mhz else ""
    if args.unlocked:
        env["PIPELINK_LOCK_SM_MHZ"] = ""
    requested_lock = str(env.get("PIPELINK_LOCK_SM_MHZ") or "").strip() not in ("", "0")
    if args.merge_results and (not args.index or not args.unlocked):
        parser.error("--merge-results requires --index and --unlocked")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("OPENBLAS_NUM_THREADS", "8")
    os.environ.update(env)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    results = {"profile": args.profile, "gpu": args.gpu, "labels": LABELS,
               "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "liger-kernel": importlib.metadata.version("liger-kernel")}, "cases": []}
    results["measurement_label"] = args.measurement_label
    index = json.loads(args.index.read_text()) if args.index else None
    context = (nullcontext({"external": args.external_lease, "unlocked": args.unlocked})
               if args.external_lease or args.unlocked else
               clock_lease(args.gpu, env=env, log_dir=args.output.parent / "public-clock"))
    with context as lease:
        for shape in SMALL_SHAPES + FULL_SHAPES:
            with ClockSampler(args.gpu) as sampler:
                row = measure(shape, gpu=args.gpu, timing=shape in FULL_SHAPES,
                              index=index if shape in FULL_SHAPES else None, stagger_ns=args.stagger_ns)
            row["telemetry"] = [sample.to_json() for sample in sampler.samples()]
            results["cases"].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
    results["clock"] = lease
    # An external lease belongs to the caller, which passes its clock setting in the environment.
    results["clock_locked"] = (not args.unlocked and requested_lock if args.external_lease
                               else bool(lease.get("locked", lease.get("sm_mhz") is not None)))
    passed = all(row["status"] == "PASS" for row in results["cases"])
    if not args.external_lease and not args.unlocked:
        passed &= lease["healthy"]
    results["status"] = "PASS" if passed else "FAIL"
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if args.merge_results:
        import run as harness
        existing = json.loads(args.merge_results.read_text())
        devices = {row["pair"]["gpu"] for cell in existing["cells"].values() for row in cell["timings"]}
        if existing["profile"] != args.profile or existing["mode"] != "unlocked" or devices != {args.gpu}:
            raise ValueError("supplemental pairs must match the unlocked matrix profile and physical GPU")
        existing["public"] = results
        existing["public_pairs_from"] = str(args.output)
        existing["measurement_label"] = args.measurement_label
        if results["status"] != "PASS":
            existing["status"] = "FAIL"
        else:
            harness.attach_public_placement(existing)
        args.merge_results.write_text(json.dumps(existing, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
