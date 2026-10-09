"""Pinned FA4 Hopper forward: native windows/softcap and exact sparse masks.

Sparse lists contain the union over each 128-row tile, with fully visible tiles
separated from partial tiles. W5 retains its 64-token, logical-query-head mask.
W6 converts the supplied selections to a membership bitset during preparation;
it is Q-outer FA4, not a KV-outer MSA implementation. All metadata construction
and the configuration's first CuTe compile/launch are outside attention timing.
"""
from __future__ import annotations

import time

from . import Impl
from ._loaders import PINS


_TILE = 128  # Pinned FA4's H100-tuned d128 SM90 forward tile, in both dimensions.


def supports(cell):
    if cell.get("diagnostic") or cell["workload"] == "W6dense":
        return "fa4 is an expert placement row, not the W6dense diagnostic ceiling"
    if cell["dtype"] != "bf16" or cell["head_dim"] != 128 or cell["batch"] != 1:
        return "This FA4 adapter targets batch-1 bf16 d128 SM90 cells"
    if cell["seqlen_q"] != cell["seqlen_k"]:
        return "The cell masks require equal-length prefill, not offset-aligned causal attention"
    mask = cell["mask"]
    kind = mask["kind"]
    if kind not in ("causal", "sliding_window", "chunked", "phi3_blocksparse", "msa_selected"):
        return f"No exact FA4 mask adapter for {kind}"
    if kind in ("chunked", "phi3_blocksparse", "msa_selected"):
        if cell["seqlen_q"] % _TILE:
            return "Sparse metadata construction requires complete 128-token tiles"
        if kind == "chunked" and mask["chunk"] % _TILE:
            return "Chunk boundaries must align to the 128-token sparse tile"
        if kind == "phi3_blocksparse" and mask["block"] != 64:
            return "The Phi retile adapter implements the released 64-token block pattern"
        if kind == "msa_selected" and (mask["block"] != _TILE or mask["topk"] != 16):
            return "The MSA adapter consumes the cell's 128-token, top-16 selection contract"
    return None


def _static_blocks(cell, device):
    """Exact union/intersection from subblock extrema, never a dense token mask."""
    import torch
    from cells import visible

    mask = cell["mask"]
    heads = cell["heads_q"] if mask["kind"] == "phi3_blocksparse" else 1
    nq, nk = cell["seqlen_q"] // _TILE, cell["seqlen_k"] // _TILE
    h = torch.arange(heads, device=device)[:, None, None]
    q = torch.arange(nq, device=device)[None, :, None] * _TILE
    k = torch.arange(nk, device=device)[None, None, :] * _TILE
    active = torch.zeros((heads, nq, nk), dtype=torch.bool, device=device)
    full = torch.ones_like(active)
    step = mask["block"] if mask["kind"] == "phi3_blocksparse" else _TILE
    # Within one model block only causality varies. Largest Q / smallest K
    # tests existence, and smallest Q / largest K tests universal visibility.
    for qo in range(0, _TILE, step):
        for ko in range(0, _TILE, step):
            active |= visible(cell, h, q + qo + step - 1, k + ko)
            full &= visible(cell, h, q + qo, k + ko + step - 1)
    return active, full


def _selected_blocks(cell, sel):
    """Consume selections once: tile unions plus a compact per-row membership map."""
    import torch

    heads, length, topk = sel.shape
    nq, nk = length // _TILE, cell["seqlen_k"] // _TILE
    # Distinct selections make these counts the number of query rows selecting
    # each KV block. Padding goes to a discarded sentinel column, never block 0.
    indices = sel.reshape(heads, nq, _TILE * topk).long()
    indices = torch.where(indices >= 0, indices, nk)
    hits = torch.zeros((heads, nq, nk + 1), dtype=torch.int32, device=sel.device)
    hits.scatter_add_(2, indices, torch.ones_like(indices, dtype=torch.int32))
    hits = hits[:, :, :nk]
    q_start = torch.arange(nq, device=sel.device)[None, :, None] * _TILE
    k_end = (torch.arange(nk, device=sel.device)[None, None, :] + 1) * _TILE - 1
    active = hits > 0
    full = (hits == _TILE) & (k_end <= q_start)

    # One 32-bit load in mask_mod replaces sixteen selection loads per score.
    # Unique block IDs give distinct bits, so integer scatter-add is bitwise OR.
    # Bit 31 is represented by signed int32 on both torch and CuTe sides.
    membership = torch.zeros((heads, length, (nk + 31) // 32), dtype=torch.int32, device=sel.device)
    words = sel.clamp_min(0).long() // 32
    bits = torch.where(sel >= 0, torch.ones_like(sel) << (sel % 32), 0)
    membership.scatter_add_(2, words, bits)
    return active, full, membership


def _ordered_blocks(blocks):
    """Compact sorted indices; only entries below each count are consumed by FA4."""
    import torch

    counts = blocks.sum(dim=-1, dtype=torch.int32)
    width = max(1, int(counts.max().item()))  # Preparation-only synchronization.
    columns = torch.arange(blocks.shape[-1], dtype=torch.int32, device=blocks.device)
    indices = torch.where(blocks, columns, blocks.shape[-1]).sort(dim=-1).values
    return counts[None].contiguous(), indices[..., :width][None].contiguous()


def _sparse_tensors(active, full, groups):
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    mask_count, mask_index = _ordered_blocks(active & ~full)
    full_count, full_index = _ordered_blocks(full)
    if groups != 1:
        # Metadata only: preserve native GQA K/V, with one list per logical Q head.
        mask_count, mask_index, full_count, full_index = (
            tensor.repeat_interleave(groups, dim=1)
            for tensor in (mask_count, mask_index, full_count, full_index)
        )
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_count,
        mask_block_idx=mask_index,
        full_block_cnt=full_count,
        full_block_idx=full_index,
        block_size=(_TILE, _TILE),
    )


def _mask_mod(cell):
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    mask = cell["mask"]
    if mask["kind"] == "chunked":
        chunk = mask["chunk"]

        @cute.jit
        def chunked(batch, head, qi, ki, seqlen_info, aux_tensors):
            size = utils.scalar_to_ssa(chunk, cutlass.Int32)
            return (ki <= qi) & (qi // size == ki // size)

        return chunked
    if mask["kind"] == "phi3_blocksparse":
        block = mask["block"]
        local = mask["local_blocks"]
        stride = mask["vert_stride"]
        head_step = 0 if mask["homo_head"] else mask["head_sliding_step"]

        @cute.jit
        def phi(batch, head, qi, ki, seqlen_info, aux_tensors):
            size = utils.scalar_to_ssa(block, cutlass.Int32)
            qb, kb = qi // size, ki // size
            offset = head * utils.scalar_to_ssa(head_step, cutlass.Int32)
            vertical = (kb + offset + 1) % utils.scalar_to_ssa(stride, cutlass.Int32) == 0
            return (ki <= qi) & ((qb - kb < utils.scalar_to_ssa(local, cutlass.Int32)) | vertical)

        return phi
    groups = cell["heads_q"] // cell["heads_kv"]
    block = mask["block"]

    @cute.jit
    def selected(batch, head, qi, ki, seqlen_info, aux_tensors):
        kb = ki[0] // block
        word = aux_tensors[0][head[0] // groups, qi[0], kb // 32]
        member = ((word >> (kb % 32)) & cutlass.Int32(1)) != 0
        return (ki <= qi) & utils.scalar_to_ssa(member, cutlass.Boolean)

    return selected


class _Prepared:
    def __init__(self, call, q, k, v, kwargs, config):
        self._call = call
        self._q, self._k, self._v = q, k, v
        self._kwargs = kwargs
        self.config = config
        self.prep_ms = 0.0
        self._result = None

    def run(self):
        # flash_attn_func exposes neither out= nor lse=. Its upstream allocation
        # and any internal per-call work stay inside the measured product call.
        self._result = self._call(self._q, self._k, self._v, **self._kwargs)
        return self._result

    def outputs(self):
        # Public FA4 already returns bf16 BSHD and natural-log fp32 BHS.
        return self._result


def prepare(cell, q, k, v, sel=None):
    import torch

    reason = supports(cell)
    if reason is not None:
        raise ValueError(reason)
    torch.cuda.synchronize(q.device)
    started = time.perf_counter()
    from flash_attn.cute.interface import flash_attn_func

    mask = cell["mask"]
    kind = mask["kind"]
    sparse = kind in ("chunked", "phi3_blocksparse", "msa_selected")
    groups = cell["heads_q"] // cell["heads_kv"]
    pack_gqa = groups > 1
    kwargs = {
        "softmax_scale": cell["scale"],
        "softcap": cell["softcap"] or 0.0,
        "causal": not sparse,
        "window_size": (mask["window"] - 1, 0) if kind == "sliding_window" else (None, None),
        "num_splits": 1,  # SM90 rejects SplitKV.
        "return_lse": True,
    }
    config = {
        "backend": "FA4 CuTe-DSL FlashAttentionForwardSm90",
        "pin": PINS["flash-attention"],
        "api": "flash_attn.cute.interface.flash_attn_func",
        "tile_q": _TILE,
        "tile_kv": _TILE,
        "tile_choice": "upstream H100-tuned d128 heuristic; no adapter autotuning",
        "q_stage": 1,
        "num_threads": 384,
        "mma_pv_is_rs": True,
        "intra_wg_overlap": True,
        "num_splits": 1,
        "block_sparsity": sparse,
        "softmax_scale": cell["scale"],
        "softcap": cell["softcap"],
        "window_size": list(kwargs["window_size"]),
        "window_convention": "inclusive left lookback W-1 and right 0; W visible keys including self",
        "kv_replication": False,
        "input_layout_conversion": None,
        "outputs_preallocated": False,
        "output_allocation": "public flash_attn_func has no out/lse parameters; upstream per-call allocation retained",
        "lse_base": "e",
        "compile": "upstream per-configuration CuTe cache; first call and device synchronization in prepare",
    }
    if sparse:
        metadata_groups = 1
        if kind == "msa_selected":
            if sel is None:
                raise ValueError("W6 requires the supplied int32 [Hkv,S,16] selection tensor")
            active, full, membership = _selected_blocks(cell, sel)
            kwargs["aux_tensors"] = [membership]
            # Broadcast only if the supplied selections' actual tile lists agree.
            # The per-row aux lookup still uses the logical head's own KV head.
            shared = bool(((active == active[:1]) & (full == full[:1])).all().item())
            if shared:
                active, full = active[:1], full[:1]
            else:
                metadata_groups = groups
                pack_gqa = False
            config.update(
                selection_source="supplied sel; no regeneration",
                selection_layout_conversion="prep-only int32 membership bitset [Hkv,S,ceil(KV_blocks/32)]",
                membership_bytes=membership.numel() * membership.element_size(),
                sparse_head_layout="broadcast" if shared else "KV-head lists repeated over logical Q heads in prep",
                traversal="Q-outer union over 128 query rows; not KV-outer MSA",
            )
        else:
            active, full = _static_blocks(cell, q.device)
            if kind == "phi3_blocksparse":
                pack_gqa = False  # FA4 requires singleton sparse-head axis for packed GQA.
            config["sparse_head_layout"] = "logical Q head" if kind == "phi3_blocksparse" else "broadcast"
        tensors = _sparse_tensors(active, full, metadata_groups)
        kwargs["block_sparse_tensors"] = tensors
        kwargs["mask_mod"] = _mask_mod(cell)
        config.update(
            sparse_block_size=[_TILE, _TILE],
            sparse_mask_index_shape=list(tensors.mask_block_idx.shape),
            sparse_full_index_shape=list(tensors.full_block_idx.shape),
            full_blocks="exact intersection; mask_mod skipped only on fully visible tiles",
            partial_blocks="exact tile union minus full tiles; exact causal mask_mod on every partial tile",
            mask_mod=kind,
            model_block_size=mask.get("block"),
        )
    kwargs["pack_gqa"] = pack_gqa
    config["pack_gqa"] = pack_gqa
    config["pack_gqa_choice"] = "upstream GQA default when sparse heads broadcast; disabled for head-specific sparse lists"
    prepared = _Prepared(flash_attn_func, q, k, v, kwargs, config)
    prepared.run()
    torch.cuda.synchronize(q.device)
    prepared.prep_ms = (time.perf_counter() - started) * 1000.0
    return prepared


IMPLS = [Impl("fa4", "expert", supports, prepare)]
