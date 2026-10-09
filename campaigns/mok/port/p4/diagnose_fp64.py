#!/usr/bin/env python3
"""Compare failing GPT-OSS BF16 gate-weight gradients to end-to-end FP64 VJPs.

No gates or tolerances change. Routed experts are differentiated one at a time
so the diagnostic does not retain a full FP64 replica of all expert weights.
All matmuls, activation/clamps, router weighting and upstream gradients on the
requested VJP paths are FP64; no low-precision intermediate casts are made.
"""
from __future__ import annotations
import argparse
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
    parser.add_argument("--tag", default="p4-gptoss-fp64")
    args = parser.parse_args()
    from run import configure_environment, scrub
    configure_environment(args)
    import torch
    import torch.distributed as dist
    from tests import utils
    from act_gptoss_reference import gptoss_swiglu
    from mok_port_host import MoKHost, SANITY, Shape, GRAD_NAMES
    from resources import load_cells

    class ErrorMetric:
        def __init__(self):
            self.sums = torch.zeros(3, dtype=torch.float64, device="cuda")
            self.maximum = torch.zeros((), dtype=torch.float64, device="cuda")

        @torch.no_grad()
        def add(self, actual, expected):
            actual, expected = actual.flatten(), expected.detach().flatten()
            for start in range(0, expected.numel(), 1 << 20):
                ref = expected[start:start + (1 << 20)]
                delta = (actual[start:start + (1 << 20)].double() - ref).abs()
                self.sums[0].add_(delta.sum())
                self.sums[1].add_(ref.abs().sum())
                self.sums[2].add_(ref.numel())
                if ref.numel():
                    torch.maximum(self.maximum, delta.max(), out=self.maximum)

        def finish(self):
            dist.all_reduce(self.sums)
            dist.all_reduce(self.maximum, op=dist.ReduceOp.MAX)
            return {"abs_max": self.maximum.item(),
                    "abs_mean": (self.sums[0] / self.sums[2]).item(),
                    "relative_l1": (self.sums[0] / self.sums[1]).item(),
                    "elements": int(self.sums[2].item())}

    class Boundaries:
        def __init__(self, limit):
            self.limit = limit
            positive = torch.tensor(limit, dtype=torch.bfloat16, device="cuda")
            self.ulp = float((torch.nextafter(positive, torch.full_like(positive, float("inf"))).double()
                              - positive.double()).item())
            self.counts = {}

        @torch.no_grad()
        def add(self, scope, gate64, up64, gate_bf16, up_bf16):
            for name, exact, rounded, boundaries in (("gate", gate64, gate_bf16, (self.limit,)),
                                                    ("up", up64, up_bf16, (-self.limit, self.limit))):
                for boundary in boundaries:
                    key = f"{scope}.{name}.boundary_{boundary:+g}"
                    count = self.counts.setdefault(key, torch.zeros(4, dtype=torch.int64, device="cuda"))
                    count[0].add_(exact.numel())
                    count[1].add_(((exact - boundary).abs() <= self.ulp).sum())
                    count[2].add_(((rounded.double() - boundary).abs() <= self.ulp).sum())
                    if name == "gate":
                        count[3].add_(((exact < boundary) != (rounded.double() < boundary)).sum())
                    elif boundary > 0:
                        count[3].add_(((exact < boundary) != (rounded.double() < boundary)).sum())
                    else:
                        count[3].add_(((exact > boundary) != (rounded.double() > boundary)).sum())

        def finish(self):
            result = {}
            for key, counts in sorted(self.counts.items()):
                dist.all_reduce(counts)
                total, exact, rounded, disagreement = counts.tolist()
                result[key] = {"elements": total, "fp64_within_one_bf16_ulp": exact,
                               "bf16_reference_within_one_bf16_ulp": rounded,
                               "fp64_vs_bf16_clamp_mask_disagreements": disagreement}
            return {"bf16_ulp": self.ulp,
                    "definition": "inclusive distance <= nextafter_bf16(+limit,+inf)-limit; same magnitude spacing used at -limit",
                    "counts": result}

    reference = FunctionType(utils.run_reference_bf16.__code__,
                             dict(vars(utils), run_swiglu_reference=gptoss_swiglu),
                             argdefs=utils.run_reference_bf16.__defaults__)
    reference.__kwdefaults__ = utils.run_reference_bf16.__kwdefaults__
    rank = int(os.environ["RANK"])
    report = {"status": "FAIL", "rank": rank, "purpose": "diagnosis only; original numeric gate stays FAIL", "cases": []}
    evidence = REPO / "campaigns/mok/evidence/p4"
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
                                timeout=timedelta(seconds=7200))
        if dist.get_world_size() != 4:
            raise ValueError("P4 diagnosis requires EP4")
        siblings = [c for c in load_cells(args.build_dir)
                    if c["world"] == 4 and c["precision"] == "bf16" and c["clamp"] == 1
                    and c["pass"] in ("fwd", "bwd")]
        if len(siblings) != 2:
            raise ValueError("Expected clamped BF16 forward/backward pair")
        family = Shape(*utils.shapes(4)[0])
        cases = ((SANITY, 1, False), (SANITY, 3, True), (family, 1, False), (family, 2, False))
        for shape, generation, routed in cases:
            with MoKHost(shape, siblings) as host:
                inputs = host.new_inputs(generation)
                names = ("d_w_shared_gate", "d_w_routed_gate") if routed else ("d_w_shared_gate",)
                bf_values = reference(inputs.x, inputs.experts, inputs.router, *host.shared_weights,
                                      *host.routed_weights_bf16, host.grad_output, host.limit)
                all_bf = dict(zip(("output", *GRAD_NAMES), bf_values))
                bf = {name: all_bf[name].detach() for name in names}
                del bf_values, all_bf
                all_actual = host.sequence(inputs, "bwd", image="linked")
                actual = {name: all_actual[name].detach() for name in names}
                del all_actual
                metrics = {name: {"kernel_vs_fp64": ErrorMetric(), "torch_bf16_vs_fp64": ErrorMetric()}
                           for name in names}
                boundaries = Boundaries(host.limit)
                x = inputs.x.double()
                wg = host.shared_weights[0].double().requires_grad_()
                wu, wd = (weight.double() for weight in host.shared_weights[1:])
                gate, up = x @ wg.T, x @ wu.T
                boundaries.add("shared", gate.detach(), up.detach(),
                               inputs.x @ host.shared_weights[0].T, inputs.x @ host.shared_weights[1].T)
                y = gptoss_swiglu(gate, up, host.limit) @ wd.T
                (gradient,) = torch.autograd.grad(y, (wg,), host.grad_output.double())
                metrics["d_w_shared_gate"]["kernel_vs_fp64"].add(actual["d_w_shared_gate"], gradient)
                metrics["d_w_shared_gate"]["torch_bf16_vs_fp64"].add(bf["d_w_shared_gate"], gradient)
                stored_gate = host.buffers["gate_shared"].detach()
                stored_up = host.buffers["up_shared"].detach()
                bf_gate = inputs.x @ host.shared_weights[0].T
                kernel_mask, bf_mask, exact_mask = stored_gate < host.limit, bf_gate < host.limit, gate.detach() < host.limit
                counts = torch.stack(((kernel_mask != bf_mask).sum(),
                                      (kernel_mask != exact_mask).sum(), (bf_mask != exact_mask).sum()))
                positions = (kernel_mask != bf_mask).nonzero()[:16]
                samples = [{"row": int(pos[0]), "column": int(pos[1]),
                            "fp64": float(gate.detach()[pos[0], pos[1]]),
                            "kernel_bf16": float(stored_gate[pos[0], pos[1]]),
                            "oracle_bf16": float(bf_gate[pos[0], pos[1]])} for pos in positions]
                dist.all_reduce(counts)
                same_gate, same_up = stored_gate.detach().requires_grad_(), stored_up.detach().requires_grad_()
                same_hidden = gptoss_swiglu(same_gate, same_up, host.limit)
                same_dgate, same_dup = torch.autograd.grad(same_hidden, (same_gate, same_up),
                                                          host.buffers["d_hidden_shared"].detach())
                stage_metrics = {}
                stage_samples = {}
                for key, oracle in (("d_gate_shared", same_dgate), ("d_up_shared", same_dup)):
                    metric = ErrorMetric()
                    metric.add(host.buffers[key].detach(), oracle.double())
                    stage_metrics[key] = metric.finish()
                    difference = (host.buffers[key].detach().float() - oracle.detach().float()).abs()
                    _, worst = difference.flatten().topk(8)
                    stage_samples[key] = []
                    for position in worst.tolist():
                        row, column = divmod(position, shape.intermediate)
                        stage_samples[key].append({
                            "row": row, "column": column,
                            "gate": float(stored_gate[row, column]), "up": float(stored_up[row, column]),
                            "d_hidden": float(host.buffers["d_hidden_shared"][row, column]),
                            "kernel_gradient": float(host.buffers[key][row, column]),
                            "identical_input_autograd_gradient": float(oracle[row, column])})
                weight_from_saved = host.buffers["d_gate_shared"].detach().T @ inputs.x
                metric = ErrorMetric()
                metric.add(actual["d_w_shared_gate"], weight_from_saved.double())
                stage_metrics["d_w_shared_gate_vs_torch_gemm_of_kernel_d_gate"] = metric.finish()
                localization = {
                    "global_gate_mask_disagreements": dict(zip(
                        ("kernel_vs_bf16_oracle", "kernel_vs_fp64", "bf16_oracle_vs_fp64"), counts.tolist())),
                    "local_gate_mask_disagreement_samples": samples,
                    "activation_autograd_on_identical_saved_inputs": stage_metrics,
                    "local_worst_activation_samples": stage_samples,
                    "interpretation": "These are stage-local diagnostic oracles only, not replacements for the independent gate."}
                del same_gate, same_up, same_hidden, same_dgate, same_dup, weight_from_saved
                del x, wg, wu, wd, gate, up, y, gradient
                if routed:
                    local_experts = host.routed_weights_bf16[0].shape[0]
                    topk = inputs.experts.shape[1]
                    expert_ids = inputs.experts.flatten()
                    destination = expert_ids // local_experts
                    order = torch.argsort(destination, stable=True)
                    sends = torch.bincount(destination, minlength=dist.get_world_size())
                    all_sends = torch.empty(dist.get_world_size(), dist.get_world_size(), dtype=torch.int64, device="cuda")
                    dist.all_gather_into_tensor(all_sends, sends)
                    send_splits, recv_splits = sends.tolist(), all_sends[:, rank].tolist()
                    count = sum(recv_splits)
                    received_x_bf = utils.all_to_all(inputs.x[order // topk], count, recv_splits, send_splits)
                    received_experts = utils.all_to_all((expert_ids[order] % local_experts).contiguous(), count, recv_splits, send_splits)
                    # The VJP through the final sum and router multiply is exact
                    # in FP64; do not copy the BF16 reference's cast here.
                    weighted_grad = host.grad_output.double().unsqueeze(1) * inputs.router.double().unsqueeze(2)
                    received_grad = utils.all_to_all(weighted_grad.reshape(-1, shape.hidden)[order], count, recv_splits, send_splits)
                    for expert in range(local_experts):
                        selected = (received_experts == expert).nonzero().flatten()
                        xb = received_x_bf[selected]
                        x = xb.double()
                        wg = host.routed_weights_bf16[0][expert].double().requires_grad_()
                        wu, wd = (weight[expert].double() for weight in host.routed_weights_bf16[1:])
                        gate, up = x @ wg.T, x @ wu.T
                        boundaries.add("routed", gate.detach(), up.detach(),
                                       xb @ host.routed_weights_bf16[0][expert].T,
                                       xb @ host.routed_weights_bf16[1][expert].T)
                        y = gptoss_swiglu(gate, up, host.limit) @ wd.T
                        (gradient,) = torch.autograd.grad(y, (wg,), received_grad[selected])
                        metrics["d_w_routed_gate"]["kernel_vs_fp64"].add(actual["d_w_routed_gate"][expert], gradient)
                        metrics["d_w_routed_gate"]["torch_bf16_vs_fp64"].add(bf["d_w_routed_gate"][expert], gradient)
                        del x, wg, wu, wd, gate, up, y, gradient
                    del received_x_bf, received_experts, received_grad, weighted_grad
                results = {name: {label: metric.finish() for label, metric in comparisons.items()}
                           for name, comparisons in metrics.items()}
                for comparison in results.values():
                    comparison["kernel_error_le_bf16_reference_error"] = {
                        metric: comparison["kernel_vs_fp64"][metric] <= comparison["torch_bf16_vs_fp64"][metric]
                        for metric in ("abs_max", "abs_mean", "relative_l1")}
                report["cases"].append({"shape": shape.name, "generation": generation,
                                        "tensors": results, "preactivation_boundaries": boundaries.finish(),
                                        "shared_stage_localization": localization})
                (evidence / f"{args.tag}-rank{rank}.json").write_text(json.dumps(scrub(report), indent=2) + "\n")
        report["status"] = "COMPLETE"
    except Exception as exc:
        import traceback
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        (evidence / f"{args.tag}-rank{rank}.json").write_text(json.dumps(scrub(report), indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
