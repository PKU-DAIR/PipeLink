"""Pinned FlashInfer FA3 products, with planning and layout work kept untimed.

The variable-sparse wrapper uses compacted, bottom-right causal coordinates.
Phi is exact because every sorted block list ends in its own 64-token diagonal
block: earlier selected blocks remain fully visible and only that last block is
triangular. This does not extend to one-row MSA requests with a full local block.
"""
from __future__ import annotations

from functools import partial
import math
import os
import time

from . import Impl
from ._loaders import PINS


_WORKSPACE_BYTES = 128 * 1024 * 1024


def supports(cell):
    if cell.get("diagnostic"):
        return "W6dense is a separate FA3 ceiling, not a FlashInfer expert placement row"
    if cell["mask"]["kind"] == "msa_selected":
        return (
            "Per-query MSA selections cannot use shared 128-row masks. With one-row "
            "requests, FlashInfer's compacted bottom-right causal mask exposes future "
            "keys in the selected local 128-token block; FA3 has no custom-mask path. "
            "A token-column dense mask is not a practical selected-block adapter."
        )
    if cell["batch"] != 1 or cell["seqlen_q"] != cell["seqlen_k"]:
        return "This adapter requires batch-one equal-length prefill"
    if cell["dtype"] != "bf16" or cell["head_dim"] != 128:
        return "This baseline uses FlashInfer's bf16 d128 FA3 specializations"
    kind = cell["mask"]["kind"]
    if kind == "phi3_blocksparse":
        if cell["mask"]["block"] != 64 or cell["seqlen_q"] % 64:
            return "Phi adaptation requires complete 64-token diagonal blocks"
        return None
    if kind in ("causal", "sliding_window", "chunked"):
        return None
    return f"No FlashInfer FA3 adaptation for mask {kind}"


class _Prepared:
    def __init__(self, call, config, *, head_major=False):
        self._call = call
        self.config = config
        self._head_major = head_major
        self._result = None
        self.prep_ms = 0.0

    def run(self):
        self._result = self._call()
        return self._result

    def outputs(self):
        out, lse = self._result
        if self._head_major:
            out = out.transpose(0, 1)
        else:
            lse = lse.transpose(0, 1)
        # Do not mutate the native buffers: outputs() may be called repeatedly.
        return out.unsqueeze(0), (lse * math.log(2.0)).unsqueeze(0)


def _ragged(flashinfer, torch, cell, q, k, v, config):
    length = cell["seqlen_q"]
    chunk = cell["mask"].get("chunk", length)
    boundaries = list(range(0, length, chunk)) + [length]
    indptr = torch.tensor(boundaries, dtype=torch.int32)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device),
        kv_layout="NHD", backend="fa3", use_cuda_graph=False,
    )
    wrapper.plan(
        indptr, indptr, cell["heads_q"], cell["heads_kv"], cell["head_dim"],
        causal=True, window_left=config["window_left"],
        logits_soft_cap=config["logits_soft_cap"], sm_scale=cell["scale"],
        q_data_type=q.dtype, kv_data_type=k.dtype,
    )
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    return _Prepared(
        partial(wrapper.run, q, k, v, out=out, lse=lse, return_lse=True, enable_pdl=False),
        {
            **config,
            "api": "BatchPrefillWithRaggedKVCacheWrapper.plan/run",
            "ragged_requests": len(boundaries) - 1,
            "chunk": chunk if cell["mask"]["kind"] == "chunked" else None,
            "input_layout": "SHD",
            "native_output_layout": "SHD",
            "native_lse_layout": "SH",
            "preallocated_outputs": True,
            "workspace_bytes": _WORKSPACE_BYTES,
            "scheduler": "FlashInfer FA3 planner",
        },
    )


def _phi(flashinfer, torch, cell, q, k, v, config):
    from cells import visible

    heads, length, dim = cell["heads_q"], cell["seqlen_q"], cell["head_dim"]
    block = cell["mask"]["block"]
    starts = torch.arange(0, length, block, device=q.device)
    head_ids = torch.arange(heads, device=q.device)[:, None, None]
    block_mask = visible(cell, head_ids, starts[None, :, None], starts[None, None, :])
    sizes = torch.full((heads, starts.numel()), block, dtype=torch.int32, device=q.device)
    wrapper = flashinfer.VariableBlockSparseAttentionWrapper(
        torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device), backend="fa3",
    )
    # Mask axis is KV-head, but Phi's vertical stripes depend on QUERY head.
    # MHA replication is a one-time input layout conversion, not per-call work.
    group = heads // cell["heads_kv"]
    qh = q.transpose(0, 1).contiguous()
    kh = k.transpose(0, 1).repeat_interleave(group, dim=0).contiguous()
    vh = v.transpose(0, 1).repeat_interleave(group, dim=0).contiguous()
    wrapper.plan(
        block_mask, sizes, sizes, heads, heads, dim, causal=True,
        logits_soft_cap=config["logits_soft_cap"], sm_scale=cell["scale"],
        q_data_type=q.dtype, kv_data_type=k.dtype,
    )
    # Although run returns HSD/HS, supplied destinations use its internal packed
    # NHD ABI. With MHA (group=1), all of run's einops/contiguous calls are views.
    out = torch.empty((heads * length, 1, dim), dtype=q.dtype, device=q.device)
    lse = torch.empty((heads * length, 1), dtype=torch.float32, device=q.device)
    return _Prepared(
        partial(wrapper.run, qh, kh, vh, out=out, lse=lse, return_lse=True, enable_pdl=False),
        {
            **config,
            "api": "VariableBlockSparseAttentionWrapper.plan/run",
            "input_layout": "HSD",
            "native_output_layout": "HSD",
            "native_lse_layout": "HS",
            "preallocated_outputs": True,
            "output_buffer_layout": "(H*S),1,D; LSE (H*S),1",
            "workspace_bytes": _WORKSPACE_BYTES,
            "block_row_size": block,
            "block_col_size": block,
            "page_size": 1,
            # The page-size-one FA3 gather kernel has no 64/128 block-size
            # restriction; its physical tile is independent of mask granularity.
            "tile_mn": [128, 96],
            "kv_replicated": True,
            "kv_replication_factor": group,
            "kernel_heads_kv": heads,
            "kv_replication_timing": "one-time layout conversion in prepare; excluded from run",
            "causal_alignment": "sorted selected blocks end with the complete diagonal block",
            "scheduler": "FlashInfer FA3 page-size-one sparse planner",
        },
        head_major=True,
    )


def prepare(cell, q, k, v, sel=None):
    import torch
    from env_report import configure_caches
    from timing import measure

    torch.cuda.synchronize(q.device)
    started = time.perf_counter()
    # Must precede importing FlashInfer: its JIT workspace is fixed at import.
    configure_caches()
    import flashinfer

    mask = cell["mask"]
    # Hopper mainloop_mma.cuh masks k < q-window_left, so W visible keys
    # including self require window_left=W-1, not W.
    window_left = mask["window"] - 1 if mask["kind"] == "sliding_window" else -1
    config = {
        "backend": "fa3",
        "upstream_commit": PINS["flashinfer"],
        "causal": True,
        "window_left": window_left,
        "sm_scale": cell["scale"],
        "logits_soft_cap": cell["softcap"] or 0.0,
        "tile_mn": [128, 128],
        "num_stages": 2,
        "num_splits": 1,
        "pack_gqa": "not exposed by FlashInfer FA3",
        "enable_pdl": False,
        "cuda_graphs": False,
        "kv_replicated": False,
        "lse_base_native": 2,
        "lse_conversion": "multiply by ln(2) in outputs; untimed",
        "jit_workspace": os.environ["FLASHINFER_WORKSPACE_BASE"],
    }
    qs, ks, vs = q[0], k[0], v[0]
    if mask["kind"] == "phi3_blocksparse":
        prepared = _phi(flashinfer, torch, cell, qs, ks, vs, config)
    elif mask["kind"] == "chunked":
        prepared = _ragged(flashinfer, torch, cell, qs, ks, vs, config)
    else:
        single = _Prepared(
            partial(
                flashinfer.single_prefill_with_kv_cache, qs, ks, vs,
                backend="fa3", causal=True, kv_layout="NHD", window_left=window_left,
                logits_soft_cap=config["logits_soft_cap"], sm_scale=cell["scale"],
                return_lse=True,
            ),
            {
                **config,
                "api": "single_prefill_with_kv_cache",
                "input_layout": "SHD",
                "native_output_layout": "SHD",
                "native_lse_layout": "SH",
                "preallocated_outputs": False,
                "allocation_note": "single-prefill API allocates O/LSE/temporary workspace each call",
                "scheduler": "FlashInfer FA3 single-request scheduler",
            },
        )
        ragged = _ragged(flashinfer, torch, cell, qs, ks, vs, config)
        candidates = {"single": single, "ragged": ragged}
        # Both are native FA3/TMA paths. Pick the measured B=1 winner rather
        # than assuming the single-request or persistent ragged schedule wins.
        timings = {
            name: measure(candidate.run, reps=2, smoke=True)
            for name, candidate in candidates.items()
        }
        winner = min(timings, key=lambda name: timings[name]["median_ms"])
        prepared = candidates[winner]
        prepared.config["autotune"] = {
            "choice": winner,
            "candidates": timings,
            "timing": "prepare-only CUDA events, two >=20ms intervals, no graphs",
        }
    prepared.run()
    torch.cuda.synchronize(q.device)
    prepared.prep_ms = (time.perf_counter() - started) * 1000.0
    return prepared


IMPLS = [Impl("flashinfer-fa3", "expert", supports, prepare)]
