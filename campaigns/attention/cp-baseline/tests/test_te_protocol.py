"""CPU coverage of zigzag ownership and the packed-row oracle check."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_te import check_local, checked_rows, zigzag_rows  # noqa: E402


def cell_for(length):
    return {"batch": 1, "seqlen_q": length, "seqlen_k": length,
            "heads_q": 4, "heads_kv": 2, "head_dim": 8,
            "scale": 8 ** -0.5, "softcap": None, "mask": {"kind": "causal"}}


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("length", [32, 8192])
def test_zigzag_partition_checks_every_chunk_edge(length, world):
    cell = cell_for(length)
    chunk = length // (2 * world)
    shards = [zigzag_rows(length, world, rank) for rank in range(world)]
    assert sorted(row for shard in shards for row in shard) == list(range(length))
    checked = set()
    for rank, owned in enumerate(shards):
        local, rows = checked_rows(cell, world, rank)
        assert rows == [owned[i] for i in local]
        checked.update(rows)
    # Includes the middle edge, whose two chunks belong to the same rank.
    for edge in range(chunk, length, chunk):
        assert {edge - 1, edge} <= checked


@pytest.mark.parametrize("world", [2, 4])
def test_packed_output_check(world):
    cell = cell_for(32)
    generator = torch.Generator().manual_seed(193)
    q = torch.randn(1, 32, 4, 8, generator=generator).bfloat16()
    k = torch.randn(1, 32, 2, 8, generator=generator).bfloat16()
    v = torch.randn(1, 32, 2, 8, generator=generator).bfloat16()
    dense = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().repeat_interleave(2, dim=2).transpose(1, 2),
        v.float().repeat_interleave(2, dim=2).transpose(1, 2),
        is_causal=True, scale=cell["scale"],
    ).transpose(1, 2).bfloat16()
    for rank in range(world):
        owned = zigzag_rows(32, world, rank)
        packed = dense[:, owned].contiguous()
        local, rows = checked_rows(cell, world, rank)
        assert check_local(cell, packed, q, k, v, local, rows)["pass"]
        assert check_local(cell, packed, q, k, v, list(range(len(owned))), owned)["pass"]
        # Both sides of the local seam between the two owned chunks, every head.
        for head in range(cell["heads_q"]):
            for row in (len(owned) // 2 - 1, len(owned) // 2):
                corrupted = packed.clone()
                corrupted[:, row, head] += 1
                assert not check_local(cell, corrupted, q, k, v, local, rows)["pass"]
        assert not check_local(cell, packed.flip(1), q, k, v, list(range(len(owned))), owned)["pass"]
