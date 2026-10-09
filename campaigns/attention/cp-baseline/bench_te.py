#!/usr/bin/env python3
"""Sequence-only Transformer Engine context-parallel attention worker (launch with torchrun).

Rank r holds zigzag chunks r and 2W-1-r of the canonical cell's Q/K/V, as PipeLink's
`cp` does. The full TE forward (communication, attention, output correction) is captured
once in a CUDA graph; its output is checked against the FP32 oracle on both sides of every
chunk edge, then replays are timed at an all-rank boundary. Rank 0 writes one JSON record.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import sysconfig

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "sm90-baseline"))
from bench_public import make_inputs  # noqa: E402
from cells import SPEC_PATH, check, load_cells, oracle_rows, reference  # noqa: E402

TRANSPORTS = ("p2p", "all_gather")


def zigzag_rows(length, world, rank):
    """Global query positions of rank's local rows: chunk rank, then chunk 2W-1-rank."""
    chunk = length // (2 * world)
    return [*range(rank * chunk, (rank + 1) * chunk),
            *range((2 * world - 1 - rank) * chunk, (2 * world - rank) * chunk)]


def checked_rows(cell, world, rank):
    """(local, global) owned rows among the oracle rows and both sides of every chunk edge."""
    length = cell["seqlen_q"]
    chunk = length // (2 * world)
    wanted = set(oracle_rows(cell))
    for edge in range(chunk, length, chunk):
        wanted.update((edge - 1, edge))
    owned = zigzag_rows(length, world, rank)
    local = [i for i, row in enumerate(owned) if row in wanted]
    return local, [owned[i] for i in local]


def check_local(cell, output, q, k, v, local, rows):
    """Compare local output rows `local` with the oracle at global positions `rows`."""
    _, expected, lse = reference(cell, q, k, v, rows=rows)
    return check(cell, output, None, (local, expected, lse))


def capture(cell, q, k, v, world, transport):
    # TE resolves cuDNN by soname; load the pinned wheel's library first.
    cudnn = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cudnn/lib/libcudnn.so.9"
    ctypes.CDLL(str(cudnn), mode=ctypes.RTLD_GLOBAL)
    from transformer_engine.pytorch import DotProductAttention

    cp_stream = torch.cuda.Stream()
    attention = DotProductAttention(
        num_attention_heads=cell["heads_q"], kv_channels=cell["head_dim"],
        num_gqa_groups=cell["heads_kv"], attention_dropout=0.0, qkv_format="bshd",
        attn_mask_type="causal", softmax_scale=cell["scale"], cp_group=dist.group.WORLD,
        cp_global_ranks=list(range(world)), cp_stream=cp_stream, cp_comm_type=transport,
    ).eval()

    def forward():
        output = attention(q, k, v)
        torch.cuda.current_stream().wait_stream(cp_stream)
        return output.view(q.shape)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    return graph, output


def measure(graph, reps=10, min_interval_ms=20.0):
    """Median per-forward time; each interval starts after an NCCL collective aligns every
    rank's stream and is the slowest rank's CUDA-event span."""
    token = torch.zeros(1, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def interval(n):
        dist.all_reduce(token)
        start.record()
        for _ in range(n):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed = torch.tensor([start.elapsed_time(end)], device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return elapsed.item()

    for _ in range(10):
        graph.replay()
    n = max(1, math.ceil(min_interval_ms / interval(1) * 1.1))
    samples = [interval(n) / n for _ in range(reps)]
    return {"median_ms": statistics.median(samples), "samples_ms": samples, "n": n,
            "warmup": 10, "cuda_graphs": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--cells-json", type=Path, default=SPEC_PATH)
    parser.add_argument("--transport", choices=TRANSPORTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cell = next(cell for cell in load_cells(args.cells_json) if cell["id"] == args.cell)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    with torch.no_grad():
        q, k, v = make_inputs(cell, "cuda")
        index = torch.tensor(zigzag_rows(cell["seqlen_q"], world, rank), device="cuda")
        shards = [tensor.index_select(1, index).contiguous() for tensor in (q, k, v)]
        graph, output = capture(cell, *shards, world, args.transport)
        graph.replay()
        local, rows = checked_rows(cell, world, rank)
        result = {"rank": rank, "rows_checked": len(rows),
                  **check_local(cell, output, q, k, v, local, rows)}
        del q, k, v
        torch.cuda.empty_cache()
        correctness = [None] * world
        dist.all_gather_object(correctness, result)
        if not all(item["pass"] for item in correctness):
            raise RuntimeError(f"TE output differs from the FP32 oracle: {correctness}")
        timing = measure(graph)
    if rank == 0:
        backends = importlib.import_module(
            "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention")._attention_backends
        args.out.write_text(json.dumps({
            "cell": args.cell, "world": world, "transport": args.transport,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformer-engine", "nvidia-cudnn-cu13")},
            "backend": {key: value for key, value in backends.items()
                        if key in ("use_flash_attention", "use_fused_attention", "fused_attention_backend")},
            "correctness": correctness, "timing": timing,
        }, indent=2, default=str) + "\n")
    # Destroying the NCCL communicator while a graph still references it hangs.
    del graph, output
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
