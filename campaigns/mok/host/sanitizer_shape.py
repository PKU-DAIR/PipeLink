"""Upstream sanity-shape workload; backward includes its prerequisite forward."""

import argparse
import json

import torch
import torch.distributed as dist

from benchmarks.utils import init_distributed
from mok import functional, ops
from tests.utils import generate_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="pass_name", choices=("forward", "backward"), required=True)
    parser.add_argument("--precision", choices=("bf16", "mxfp8"), required=True)
    args = parser.parse_args()
    rank, world_size, device = init_distributed()
    # tests/test_functional.py:349-380, including the minimum collective chunk.
    config = functional.MoKConfig(
        fwd_num_comm_sms=2, bwd_num_comm_sms=2, minibatch_size=256,
        macrobatch_size=256, schedule_capacity_multiplier=1.5,
        all_gather_top_experts_chunk_bytes=16,
    )
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=device,
        num_local_tokens=512, hidden_size=256, topk=1,
    )
    x, experts, router, shared_gate, shared_up, shared_down, gate, up, down, dy = generate_inputs(
        rank, device, world_size, 1, 1, 512, 256, 256,
    )
    schedule = functional.build_schedule(workspace, config, experts, num_local_experts=1)
    shared = (shared_gate, shared_up, shared_down)
    if args.precision == "mxfp8":
        gate, up, down = (ops.mxfp8_quantize(w, True, True) for w in (gate, up, down))
        forward_weights = (gate[:2], up[:2], down[:2])
        backward_weights = (gate, up, down[2:])
    else:
        forward_weights = backward_weights = (gate, up, down)
    output, context = functional.forward(
        config, workspace, schedule, x, router, *shared, *forward_weights,
    )
    if args.pass_name == "backward":
        gradients = functional.backward(
            config, workspace, schedule, context, dy, x, router, *shared, *backward_weights,
        )
    torch.cuda.synchronize(device)
    dist.barrier()
    print(json.dumps({"rank": rank, "pass": args.pass_name, "precision": args.precision, "completed": True}), flush=True)
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
