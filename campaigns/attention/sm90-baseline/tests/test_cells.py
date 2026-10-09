"""Consumer-visible mask, selection, work-count and oracle invariants (CPU)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cells import check, load_selection, oracle_rows, pair_count, reference, save_selection, select, useful_flops, visible


MASKS = [
    {"kind": "causal"},
    {"kind": "sliding_window", "window": 65},
    {"kind": "chunked", "chunk": 128},
    {"kind": "phi3_blocksparse", "block": 16, "local_blocks": 3, "vert_stride": 8, "homo_head": False, "head_sliding_step": 2},
    {"kind": "phi3_blocksparse", "block": 16, "local_blocks": 3, "vert_stride": 8, "homo_head": True, "head_sliding_step": 2},
    *[{"kind": "msa_selected", "block": 32, "topk": 16, "forced_local": True, "pattern": pattern, "seed": 123} for pattern in ("prefix", "uniform")],
]


def cell_for(mask, length=1027):
    return dict(id="test", batch=2, heads_q=4, heads_kv=2, head_dim=8,
                seqlen_q=length, seqlen_k=length, scale=0.37, softcap=0.7,
                mask=mask, numeric_case=19)


def dense_mask(cell, head, sel):
    """Independent dense construction, deliberately not using visible()."""
    n, m = cell["seqlen_q"], cell["mask"]
    allowed = torch.ones(n, n, dtype=torch.bool).tril()
    q, k = torch.arange(n)[:, None], torch.arange(n)[None, :]
    if m["kind"] == "sliding_window":
        allowed &= torch.ones(n, n, dtype=torch.bool).triu(1 - m["window"])
    elif m["kind"] == "chunked":
        for start in range(0, n, m["chunk"]):
            allowed[start:start + m["chunk"], :start] = False
    elif m["kind"] == "phi3_blocksparse":
        block_allowed = torch.zeros((n + m["block"] - 1) // m["block"], (n + m["block"] - 1) // m["block"], dtype=torch.bool)
        for qb in range(len(block_allowed)):
            for kb in range(qb + 1):
                offset = 0 if m["homo_head"] else head * m["head_sliding_step"]
                block_allowed[qb, kb] = kb >= qb - m["local_blocks"] + 1 or (kb + offset + 1) % m["vert_stride"] == 0
        allowed &= block_allowed.repeat_interleave(m["block"], 0).repeat_interleave(m["block"], 1)[:n, :n]
    elif m["kind"] == "msa_selected":
        membership = torch.zeros(n, n, dtype=torch.bool)
        for row in range(n):
            for block in sel[head // (cell["heads_q"] // cell["heads_kv"]), row].tolist():
                if block >= 0:
                    membership[row, block * m["block"]:(block + 1) * m["block"]] = True
        allowed &= membership
    return allowed


@pytest.mark.parametrize("mask", MASKS)
def test_pair_count_matches_dense_all_heads(mask):
    cell = cell_for(mask)
    sel = select(cell) if mask["kind"] == "msa_selected" else None
    count = 0
    q, k = torch.arange(cell["seqlen_q"])[:, None], torch.arange(cell["seqlen_k"])[None, :]
    for head in range(cell["heads_q"]):
        expected = dense_mask(cell, head, sel)
        assert torch.equal(visible(cell, head, q, k, sel), expected)
        count += int(expected.sum())
    assert pair_count(cell, sel) == count * cell["batch"]
    assert useful_flops(cell, sel) == 4 * cell["head_dim"] * count * cell["batch"]


@pytest.mark.parametrize("pattern", ["prefix", "uniform"])
def test_selection_invariants_and_shared_file(pattern, tmp_path):
    mask = dict(MASKS[-1], pattern=pattern)
    cell = cell_for(mask)
    selection = select(cell)
    assert torch.equal(selection, select(cell))
    for h in range(cell["heads_kv"]):
        for row in range(cell["seqlen_q"]):
            block = row // mask["block"]
            entries = selection[h, row].tolist()
            count = 1 + min(mask["topk"] - 1, block)
            valid = entries[:count]
            assert len(set(valid)) == count
            assert valid == sorted(valid)
            assert valid[-1] == block
            assert all(0 <= b < block for b in valid[:-1])
            assert entries[count:] == [-1] * (mask["topk"] - count)
            if block < mask["topk"] or pattern == "prefix":
                assert valid[:-1] == list(range(min(mask["topk"] - 1, block)))
    if pattern == "uniform":
        # Per-query and per-KV-head, not a shared block-tile selection.
        assert not torch.equal(selection[0, -64:], selection[1, -64:])
        assert not torch.equal(selection[0, -2], selection[0, -1])
    path = tmp_path / "selection.i32"
    save_selection(cell, selection, path)
    assert torch.equal(selection, load_selection(cell, path))
    assert path.read_bytes()[:4] == b"\x00\x00\x00\x00"


@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("requested_rows", [None, [36, 0, 17, 5]])
def test_reference_matches_dense_fp32_with_softcap(mask, requested_rows):
    mask = dict(mask)
    if "block" in mask:
        mask["block"] = 2
    if "chunk" in mask:
        mask["chunk"] = 8
    if "window" in mask:
        mask["window"] = 7
    cell = cell_for(mask, length=37)
    generator = torch.Generator().manual_seed(92)
    def random(heads):
        return torch.randn(cell["batch"], 37, heads, 8, generator=generator).to(torch.bfloat16)
    q, k, v = random(4), random(2), random(2)
    sel = select(cell) if mask["kind"] == "msa_selected" else None
    full_o, full_lse = torch.empty_like(q, dtype=torch.float32), torch.empty(2, 4, 37)
    for b in range(2):
        for h in range(4):
            scores = q[b, :, h].float() @ k[b, :, h // 2].float().T
            scores = torch.tanh(scores * cell["scale"] / cell["softcap"]) * cell["softcap"]
            scores[~dense_mask(cell, h, sel)] = -torch.inf
            full_o[b, :, h] = scores.softmax(-1) @ v[b, :, h // 2].float()
            full_lse[b, h] = scores.logsumexp(-1)
    ref = reference(cell, q, k, v, sel, rows=requested_rows)
    assert ref[0] == (oracle_rows(cell) if requested_rows is None else requested_rows)
    rows, out, lse = ref
    torch.testing.assert_close(out, full_o[:, rows], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(lse, full_lse[:, :, rows], atol=1e-6, rtol=1e-6)
    assert check(cell, full_o.to(torch.bfloat16), full_lse, ref)["pass"]
    assert check(cell, full_o.to(torch.bfloat16), None, ref)["pass"]
    bad_o = full_o.clone()
    bad_o[:, 0] += 1
    assert not check(cell, bad_o, full_lse, ref)["pass"]
    assert not check(cell, full_o, full_lse + 1, ref)["pass"]


def test_oracle_covers_every_block_half_and_chunk_edge():
    for mask in (dict(MASKS[3], block=64), MASKS[2], MASKS[-1]):
        cell = cell_for(mask)
        rows = oracle_rows(cell)
        assert rows == sorted(set(rows))
        step = mask.get("block", mask.get("chunk"))
        for edge in range(step, cell["seqlen_q"], step):
            assert {edge - 1, edge, edge + 1} <= set(rows)
    window_cell = cell_for(MASKS[1])
    assert {64, 65, 66} <= set(oracle_rows(window_cell))
