#!/usr/bin/env python3
"""P4-only GPU check: existing host sessions, activation-owner torch reference.

Invoke with torchrun inside tools/gpu_window.sh; does not run original SwiGLU
as a correctness oracle and grants no inherited-upstream tolerance exceptions.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from types import FunctionType

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "campaigns/mok/host"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tag", default="p4-gptoss")
    args = parser.parse_args()
    from run import configure_environment, scrub
    configure_environment(args)
    import torch
    import torch.distributed as dist
    from tests import utils
    from act_gptoss_reference import gptoss_swiglu
    from mok_port_host import MoKHost, SANITY, Shape, GRAD_NAMES, tolerance_report, bitwise_report
    from resources import load_cells
    # Reuse routing, GEMMs, collectives and backward autograd unchanged. Separate
    # globals bind only the reference's activation, leaving upstream modules intact.
    reference_globals = dict(vars(utils), run_swiglu_reference=gptoss_swiglu)
    reference = FunctionType(utils.run_reference_bf16.__code__, reference_globals,
                             argdefs=utils.run_reference_bf16.__defaults__)
    reference.__kwdefaults__ = utils.run_reference_bf16.__kwdefaults__
    forward_reference = FunctionType(utils.run_forward_reference_bf16.__code__, reference_globals,
                                     argdefs=utils.run_forward_reference_bf16.__defaults__)
    rank = int(os.environ["RANK"])
    report = {"status": "FAIL", "rank": rank, "reference": "GPT-OSS alpha=1.702", "cells": []}
    evidence = REPO / "campaigns/mok/evidence/p4"
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
                                timeout=timedelta(seconds=7200))
        if dist.get_world_size() != 4:
            raise ValueError("P4 requires EP4")
        cells = [c for c in load_cells(args.build_dir) if c["world"] == 4 and c["pass"] in ("fwd", "bwd")]
        if len(cells) != 8:
            raise ValueError("Expected eight EP4 forward/backward precision/clamp cells")
        shapes = (SANITY, Shape(*utils.shapes(4)[0]))
        for precision in ("bf16", "mxfp8"):
            for clamp in (0, 1):
                siblings = [c for c in cells if c["precision"] == precision and c["clamp"] == clamp]
                for shape in shapes:
                    with MoKHost(shape, siblings) as host:
                        host.poison = True
                        for generation in (1, 2, 3):
                            inputs = host.new_inputs(generation)
                            values = reference(inputs.x, inputs.experts, inputs.router, *host.shared_weights,
                                               *host.routed_weights_bf16, host.grad_output, host.limit)
                            _, gate, up, hidden, _ = forward_reference(inputs.x, inputs.experts, *host.shared_weights,
                                                                       *host.routed_weights_bf16, host.limit)
                            expected = {**dict(zip(("output", *GRAD_NAMES), values)),
                                        "gate_shared": gate, "up_shared": up, "hidden_shared": hidden}
                            # Autograd has produced all gradient oracles. Measurement
                            # reductions are inference-only and use in-place buffers.
                            expected = {k: v.detach() for k, v in expected.items()}
                            for pass_name in ("fwd", "bwd"):
                                results = {}
                                outputs = {}
                                for image in ("linked", "one-tu"):
                                    actual = host.sequence(inputs, pass_name, image=image)
                                    outputs[image] = {k: v.clone() for k, v in actual.items() if k in expected}
                                    results[image] = {k: tolerance_report(expected[k], v, precision)
                                                      for k, v in outputs[image].items()}
                                bitwise = {k: bitwise_report(v, outputs["one-tu"][k])
                                           for k, v in outputs["linked"].items()}
                                report["cells"].append({"precision": precision, "clamp": clamp, "pass": pass_name,
                                    "shape": asdict(shape), "generation": generation, "reference": results,
                                    "linked_onetu_bitwise": bitwise,
                                    "passed": all(bitwise.values()) and all(r["passed"] for im in results.values() for r in im.values())})
        report["status"] = "PASS" if all(c["passed"] for c in report["cells"]) else "FAIL"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (evidence / f"{args.tag}-rank{rank}.json").write_text(json.dumps(scrub(report), indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
