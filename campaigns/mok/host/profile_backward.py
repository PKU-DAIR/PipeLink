"""Profile one EP4 backward step after own-arm forward restoration and warmup."""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

from run import configure_environment, scrub


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--precision", choices=("bf16", "mxfp8"), required=True)
    parser.add_argument("--arm", choices=("port", "upstream"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay-store", type=Path, help="fresh file-store namespace per application replay pass")
    args = parser.parse_args()
    configure_environment(args)
    import torch
    import torch.distributed as dist
    from mok import functional, _C
    from mok_port_host import MoKHost, Shape
    from resources import load_cells

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    rendezvous = {}
    if args.replay_store:
        args.replay_store.mkdir(parents=True, exist_ok=True)
        counter = args.replay_store / f"rank{rank}-pass"
        replay_pass = int(counter.read_text()) + 1 if counter.exists() else 0
        counter.write_text(str(replay_pass))
        rendezvous = {"init_method": f"file://{args.replay_store.resolve()}/pass{replay_pass}",
                      "rank": rank, "world_size": 4}
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=300), **rendezvous)
    cells = [c for c in load_cells(args.build_dir) if c["precision"] == args.precision and c["world"] == 4 and c["clamp"] == 0]
    config = functional.MoKConfig(fwd_num_comm_sms=24 if args.precision == "bf16" else 36,
                                  bwd_num_comm_sms=28 if args.precision == "bf16" else 36)
    try:
        with MoKHost(Shape("profile-2048", 384, 7168, 3072, 6, 2048), cells, config=config) as host:
            inputs = host.inputs
            schedule = context = None
            def prepare():
                nonlocal schedule, context
                if args.arm == "port":
                    host.forward(inputs)
                else:
                    schedule = functional.build_schedule(host.workspace, config, inputs.experts, num_local_experts=96)
                    _, context = functional.forward(config, host.workspace, schedule, inputs.x, inputs.router,
                        *host.shared_weights, *host.routed_weights, host.limit)
                host.barrier()
            def issue():
                if args.arm == "port":
                    return host.backward(inputs)
                return functional.backward(config, host.workspace, schedule, context, host.grad_output, inputs.x,
                    inputs.router, *host.shared_weights, *host.backward_weights, host.limit)
            for _ in range(3):
                prepare()
                issue()
            prepare()
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.cudart().cudaProfilerStart()
            values = issue()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            dist.barrier()
            cell = next(c for c in cells if c["pass"] == "bwd")
            images = {key: hashlib.sha256((cell["directory"] / cell[key]).read_bytes()).hexdigest() for key in ("linked", "one_tu")}
            images["upstream_extension"] = hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest()
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{args.arm}-{args.precision}-rank{rank}.json").write_text(json.dumps(scrub({
                "status": "completed", "rank": rank, "arm": args.arm, "precision": args.precision,
                "shape": "EP4, 2048 tokens/rank, H7168 I3072 experts384 top6 clamp0",
                "images": images, "interval": "one backward step; own forward/context restored outside profiler range",
                "config": {field: getattr(config, field) for field in config.__dataclass_fields__},
                "numeric_validation": "not performed by profiler probe"}), indent=2) + "\n")
            del values
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
