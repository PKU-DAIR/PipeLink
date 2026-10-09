#!/usr/bin/env python3
"""Exact bf16 sm90 cell masks, useful work, selections and bounded fp32 oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SPEC_PATH = Path(__file__).with_name("cells.json")


def load_cells(path=SPEC_PATH) -> list[dict]:
    spec = json.loads(Path(path).read_text())
    if spec["schema"] != "pipelink-sm90-cells-v1":
        raise ValueError("Unsupported cell schema")
    return spec["cells"]


def visible(cell, h, q_idx_tensor, k_idx_tensor, sel=None):
    """Broadcastable token indices; h is a logical query head (scalar or tensor).

    W6 sel must already be on the indices' device; no implicit copying or RNG.
    """
    q, k = q_idx_tensor, k_idx_tensor
    m = cell["mask"]
    causal = k <= q
    kind = m["kind"]
    if kind == "causal":
        return causal
    if kind == "sliding_window":
        return causal & (k > q - m["window"])
    if kind == "chunked":
        return causal & (k // m["chunk"] == q // m["chunk"])
    if kind == "phi3_blocksparse":
        qb, kb = q // m["block"], k // m["block"]
        offset = 0 if m["homo_head"] else h * m["head_sliding_step"]
        return causal & ((qb - kb < m["local_blocks"]) | ((kb + offset + 1) % m["vert_stride"] == 0))
    if kind == "msa_selected":
        if sel is None:
            raise ValueError("W6 requires the materialized selection tensor")
        chosen = sel[h // (cell["heads_q"] // cell["heads_kv"]), q]
        return causal & (chosen == (k // m["block"])[..., None]).any(dim=-1)
    raise ValueError(f"Unknown mask kind: {kind}")


def select(cell) -> torch.Tensor:
    """CPU int32 [Hkv,S,topk], sorted block ids, -1 padding last.

    Uniform uses batched Floyd sampling, independently per (KV head, query).
    A CPU torch.Generator owns all randomness; no global RNG is modified.
    """
    m = cell["mask"]
    if m["kind"] != "msa_selected":
        raise ValueError("Selections are only defined for W6")
    heads, length, topk = cell["heads_kv"], cell["seqlen_q"], m["topk"]
    blocks = np.arange(length, dtype=np.int64) // m["block"]
    counts = np.minimum(topk - 1, blocks)
    out = np.full((heads, length, topk), -1, dtype=np.int32)
    if m["pattern"] == "prefix":
        for i in range(topk - 1):
            out[:, counts > i, i] = i
    elif m["pattern"] == "uniform":
        generator = torch.Generator().manual_seed(m["seed"])
        # Floyd's collision replacement is exact sampling without replacement;
        # it needs O(topk) random draws, not an S-by-S random-key array.
        for i in range(topk - 1):
            j = blocks - counts + i
            random = torch.rand((heads, length), dtype=torch.float64, generator=generator).numpy()
            candidate = np.floor(random * (j + 1)).astype(np.int32)
            collision = (out[:, :, :i] == candidate[:, :, None]).any(axis=-1)
            candidate = np.where(collision, j[None, :], candidate)
            out[:, counts > i, i] = candidate[:, counts > i]
    else:
        raise ValueError(f"Unknown selection pattern: {m['pattern']}")
    out[:, np.arange(length), counts] = blocks
    # Move padding after valid entries before sorting (ordinary -1 sorts first).
    out[out < 0] = np.iinfo(np.int32).max
    out.sort(axis=-1)
    out[out == np.iinfo(np.int32).max] = -1
    return torch.from_numpy(out)


def save_selection(cell, sel, path):
    """Write the shared raw little-endian int32 selection artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sel.cpu().numpy().astype("<i4", copy=False).tofile(path)


def load_selection(cell, path) -> torch.Tensor:
    shape = (cell["heads_kv"], cell["seqlen_q"], cell["mask"]["topk"])
    return torch.from_numpy(np.fromfile(path, dtype="<i4").reshape(shape).astype(np.int32, copy=False))


def pair_count(cell, sel=None) -> int:
    """Exact integer sum over batch and query heads; no dense token mask."""
    s, m = cell["seqlen_q"], cell["mask"]
    if cell["seqlen_k"] != s:
        raise ValueError("These cells require equal-length causal prefill")
    kind = m["kind"]
    if kind == "causal":
        per_head = s * (s + 1) // 2
    elif kind == "sliding_window":
        w = min(s, m["window"])
        per_head = w * (w + 1) // 2 + (s - w) * w
    elif kind == "chunked":
        full, rem = divmod(s, m["chunk"])
        per_head = full * m["chunk"] * (m["chunk"] + 1) // 2 + rem * (rem + 1) // 2
    elif kind == "phi3_blocksparse":
        block, stride = m["block"], m["vert_stride"]
        total = 0
        # Sum full earlier blocks plus the triangular diagonal, including tails.
        for h in range(cell["heads_q"]):
            offset = 0 if m["homo_head"] else h * m["head_sliding_step"]
            residue = (-offset - 1) % stride
            for qb, start in enumerate(range(0, s, block)):
                rows = min(block, s - start)
                local_start = max(0, qb - m["local_blocks"] + 1)
                vertical = max(0, (local_start - 1 - residue) // stride + 1)
                total += rows * block * (qb - local_start + vertical) + rows * (rows + 1) // 2
        return cell["batch"] * total
    elif kind == "msa_selected":
        if sel is None:
            raise ValueError("W6 pair counting requires the shared selection artifact")
        chosen = sel.detach().cpu().numpy().astype(np.int64, copy=False)
        q = np.arange(s)[None, :, None]
        starts = chosen * m["block"]
        lengths = np.clip(np.minimum(q + 1, s) - starts, 0, m["block"])
        lengths[chosen < 0] = 0
        return int(lengths.sum()) * cell["batch"] * (cell["heads_q"] // cell["heads_kv"])
    else:
        raise ValueError(f"Unknown mask kind: {kind}")
    return cell["batch"] * cell["heads_q"] * per_head


def useful_flops(cell, sel=None) -> int:
    return 4 * cell["head_dim"] * pair_count(cell, sel)


def oracle_rows(cell) -> list[int]:
    s, m = cell["seqlen_q"], cell["mask"]
    rows = {0, 1, 127, 128, s // 2, s - 1}
    if m["kind"] == "sliding_window":
        rows.update((m["window"] - 1, m["window"], m["window"] + 1))
    step = m.get("chunk") if m["kind"] == "chunked" else m.get("block")
    if step:
        for edge in range(step, s, step):
            rows.update((edge - 1, edge, edge + 1))
    return sorted(row for row in rows if 0 <= row < s)


@torch.no_grad()
def reference(cell, q, k, v, sel=None, *, rows=None):
    """Return (rows, fp32 O[B,R,H,D], ln-LSE[B,H,R]).

    At most 32 query rows and one KV head are converted/reduced at once.
    No all-head S-by-S score tensor, even for every 64-token edge at 128K.
    rows (global query positions) defaults to oracle_rows(cell).
    """
    rows = oracle_rows(cell) if rows is None else list(rows)
    device = q.device
    indices = torch.tensor(rows, device=device)
    keys = torch.arange(cell["seqlen_k"], device=device)[None, :]
    out = torch.empty((cell["batch"], len(rows), cell["heads_q"], cell["head_dim"]), device=device, dtype=torch.float32)
    lse = torch.empty((cell["batch"], cell["heads_q"], len(rows)), device=device, dtype=torch.float32)
    groups = cell["heads_q"] // cell["heads_kv"]
    # Explicitly disable TF32 so the spot oracle remains fp32.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for b in range(cell["batch"]):
            for kvh in range(cell["heads_kv"]):
                key, value = k[b, :, kvh].float(), v[b, :, kvh].float()
                for h in range(kvh * groups, (kvh + 1) * groups):
                    for first in range(0, len(rows), 32):
                        qr = indices[first:first + 32]
                        scores = (q[b, qr, h].float() @ key.T) * cell["scale"]
                        if cell["softcap"] is not None:
                            scores = cell["softcap"] * torch.tanh(scores / cell["softcap"])
                        scores.masked_fill_(~visible(cell, h, qr[:, None], keys, sel), -torch.inf)
                        lse[b, h, first:first + len(qr)] = torch.logsumexp(scores, dim=-1)
                        out[b, first:first + len(qr), h] = scores.softmax(dim=-1) @ value
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return rows, out, lse


def check(cell, o, lse, ref) -> dict:
    rows, expected_o, expected_lse = ref
    delta_o = (o[:, rows].float() - expected_o).abs()
    passed = torch.all(delta_o <= 0.02 + 0.02 * expected_o.abs()).item()
    error_lse = None
    if lse is not None:
        delta_lse = (lse[:, :, rows].float() - expected_lse).abs()
        error_lse = delta_lse.max().item()
        passed = passed and torch.all(delta_lse <= 0.02 + 0.02 * expected_lse.abs()).item()
    return {"max_abs_err_o": delta_o.max().item(), "max_abs_err_lse": error_lse, "pass": bool(passed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("select")
    selection.add_argument("cell_id")
    selection.add_argument("--out", type=Path, required=True)
    selection.add_argument("--cells-json", type=Path, default=SPEC_PATH)
    pairs = sub.add_parser("pairs")
    pairs.add_argument("--selection-dir", type=Path, help="Read/write shared W6 .i32 files here")
    args = parser.parse_args(argv)
    cells = load_cells(args.cells_json if args.command == "select" else SPEC_PATH)
    if args.command == "select":
        cell = next(c for c in cells if c["id"] == args.cell_id)
        save_selection(cell, select(cell), args.out)
        print(args.out)
        return
    print(f"{'Cell':24} {'Visible pairs':>16} {'Useful GFLOP':>16}")
    for cell in cells:
        sel = None
        if cell["mask"]["kind"] == "msa_selected":
            if args.selection_dir:
                path = args.selection_dir / (cell["id"] + ".i32")
                if not path.exists():
                    save_selection(cell, select(cell), path)
                sel = load_selection(cell, path)
            else:
                sel = select(cell)
        count = pair_count(cell, sel)
        print(f"{cell['id']:24} {count:16d} {4 * cell['head_dim'] * count / 1e9:16.9f}")


if __name__ == "__main__":
    main()
