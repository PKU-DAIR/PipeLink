"""Pinned model-native sparse products, with preparation outside event timing.

Phi's padded wrapper performs a GPU-to-CPU metadata copy on every invocation.
We exercise that API in preparation, then launch its unchanged inference kernel
with the same precomputed metadata and output-zeroing operation. Flash-MSA has
no output/workspace argument: its internal gathers, allocations, resets and
online merges remain part of each complete run_main_forward call.
"""
from __future__ import annotations

from time import perf_counter

import torch

from . import Impl
from ._loaders import PINS, load_flash_msa, load_phi3, load_vllm_m3


def _supports(cell, workload, kind, block):
    if cell["workload"] != workload or cell["mask"]["kind"] != kind:
        return f"Only {workload} {kind} cells are implemented by this model-native adapter"
    if cell["batch"] != 1 or cell["dtype"] != "bf16" or cell["head_dim"] != 128:
        return "Requires batch 1, bf16, head dimension 128"
    if cell["softcap"] is not None:
        return "The native sparse product does not implement logit softcap"
    if cell["seqlen_q"] != cell["seqlen_k"]:
        return "Only equal-length prefill is implemented"
    if cell["mask"]["block"] != block or cell["seqlen_q"] % block:
        return f"Requires {block}-token blocks and a block-aligned sequence length"
    if cell["heads_q"] % cell["heads_kv"]:
        return "Requires an integral GQA ratio"
    return None


def _supports_phi3(cell):
    reason = _supports(cell, "W5", "phi3_blocksparse", 64)
    if reason is not None:
        return reason
    mask = cell["mask"]
    native_step = max(1, mask["vert_stride"] // cell["heads_q"])
    if not mask["homo_head"] and mask["head_sliding_step"] != native_step:
        return "Phi's mask builder fixes head_sliding_step=max(1,vert_stride//heads_q)"
    return None


def _supports_m3(cell):
    reason = _supports(cell, "W6", "msa_selected", 128)
    if reason is not None:
        return reason
    if not cell["mask"]["forced_local"] or cell["mask"]["topk"] != 16:
        return "Requires W6's top-16 selected set including the forced local block"
    return None


def _supports_flash_msa(cell):
    reason = _supports_m3(cell)
    if reason is not None:
        return reason
    if cell["heads_q"] // cell["heads_kv"] > 32:
        return "Flash-MSA's online merge has storage for at most 32 heads per proxy head"
    return None


class _Prepared:
    def __init__(self, invoke, config, *, output_bhsd=False):
        self._invoke = invoke
        self.config = config
        self._output_bhsd = output_bhsd
        self.prep_ms = 0.0

    def run(self):
        self._result = self._invoke()
        return self._result

    def outputs(self):
        out, lse = self._result
        if self._output_bhsd:
            out = out.transpose(1, 2).contiguous()
        return out, lse


def _warm(prepared, start, device):
    prepared.run()
    torch.cuda.synchronize(device)
    prepared.prep_ms = (perf_counter() - start) * 1000
    return prepared


def _prepare_phi3(cell, q, k, v, sel):
    torch.cuda.synchronize(q.device)
    start = perf_counter()
    module = load_phi3()
    mask = cell["mask"]
    length, heads, dim = cell["seqlen_q"], cell["heads_q"], cell["head_dim"]
    block = mask["block"]
    layout, block_mask, _ = module._get_sparse_attn_mask(
        heads, length, length, q.dtype, q.device,
        BLOCK=block, local_blocks=mask["local_blocks"],
        vert_stride=mask["vert_stride"], homo_head=mask["homo_head"],
    )
    del block_mask

    # The shipped builder indexes h on the logical Q-head axis. In the kernel,
    # off_h indexes both CSR tensors, while off_h // q_k_ratio indexes K/V.
    # Thus GQA does not collapse the heterogeneous vertical stripes to KV heads.
    # Exercise the public inference API once, including its JIT, before timing.
    out = module.blocksparse_flash_attn_padded_fwd(
        q, k, v, cell["scale"], layout, block_size=block, max_seqlen=length,
    )

    # Hoist only padded-wrapper metadata. Its unconditional q_lens.cpu() cannot
    # appear in an asynchronous event interval. This is the exact unmodified
    # kernel invocation for equal-length, unpadded, batch-one prefill.
    starts = torch.zeros(1, dtype=torch.int32, device=q.device)
    ends = torch.full((1,), length, dtype=torch.int32, device=q.device)
    batch_ids = torch.zeros(length // block, dtype=torch.int32, device=q.device)
    start_sids = torch.arange(0, length, block, dtype=torch.int32, device=q.device)
    crow, col = layout
    grid = (length // block, heads)
    launch = module._fwd_kernel_batch_inference[grid]
    args = (
        q, k, v, out, cell["scale"],
        starts, ends, starts, ends, batch_ids, start_sids,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        crow, col, *crow.stride(), *col.stride(),
        heads // cell["heads_kv"],
    )
    options = dict(
        HAS_BATCH_DIM=True, D_HEAD=dim, BLOCK_M=block, BLOCK_N=block,
        BLOCK_D=dim, BLOCK_M_LOADING=block, EVEN_D=True,
        num_warps=4, num_stages=3,
    )

    def run():
        # Retain the public wrapper's per-call zeroing even though these cells
        # contain no padding. No resets or attention launches are moved to prep.
        out.zero_()
        launch(*args, **options)
        return out, None

    config = {
        "backend": "Phi-3-small shipped Triton inference kernel",
        "pin": PINS["phi3-small"],
        "api": "blocksparse_flash_attn_padded_fwd (prepare); _fwd_kernel_batch_inference (run)",
        "wrapper_adaptation": "Hoist padded-wrapper GPU-to-CPU q_lens copy and static launch metadata into prep; retain output zeroing in run; unchanged upstream kernel",
        "layout": "BSHD, native GQA",
        "layout_conversion": "none",
        "kv_replication": False,
        "mask_head_axis": "logical query head, not KV head",
        "mask": mask.copy(),
        "scale": cell["scale"],
        "tile_mn": [block, block],
        "block_d": dim,
        "num_warps": 4,
        "num_stages": 3,
        "grid": list(grid),
        "pack_gqa": False,
        "num_splits": 1,
        "autotune": "none; shipped padded-prefill launch configuration",
        "output_preallocated": True,
        "per_call_sequence": ["output.zero_", "_fwd_kernel_batch_inference"],
        "lse_base": None,
    }
    return _warm(_Prepared(run, config), start, q.device)


def _prepare_vllm_m3(cell, q, k, v, sel):
    torch.cuda.synchronize(q.device)
    start = perf_counter()
    module = load_vllm_m3()
    length, hkv, dim = cell["seqlen_q"], cell["heads_kv"], cell["head_dim"]
    block = cell["mask"]["block"]
    pages = length // block
    # One page is one sparse block; K and V occupy the two halves of the last
    # axis. This packing is a one-time input-layout conversion, not replication.
    kv = torch.empty((pages, hkv, block, 2 * dim), dtype=k.dtype, device=k.device)
    kv[..., :dim].copy_(k[0].view(pages, block, hkv, dim).permute(0, 2, 1, 3))
    kv[..., dim:].copy_(v[0].view(pages, block, hkv, dim).permute(0, 2, 1, 3))
    table = torch.arange(pages, dtype=torch.int32, device=q.device)[None]
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=q.device)
    seq_lens = torch.tensor([length], dtype=torch.int32, device=q.device)
    prefix_lens = torch.zeros(1, dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)
    args = (
        q[0], kv, sel, table, cu_seqlens, seq_lens, prefix_lens,
        length, hkv, cell["scale"], out[0],
    )

    def run():
        # Prefill always launches one _gqa_sparse_fwd_kernel. The top-k split
        # and _merge_topk_attn_out_kernel belong to the separate decode API.
        module.minimax_m3_sparse_attn(*args)
        return out, None

    config = {
        "backend": "vLLM MiniMax-M3 Triton prefill",
        "pin": PINS["vllm"],
        "api": "minimax_m3_sparse_attn",
        "layout": "Q SHD; KV [pages,Hkv,128,2D]",
        "layout_conversion": "KV page packing once in untimed prep",
        "kv_replication": False,
        "selection": "shared sel consumed unchanged; valid entries first, -1 suffix",
        "valid_selection_count": "min(topk, floor(q/128)+1); native kernel does not inspect padding",
        "selection_pattern": cell["mask"]["pattern"],
        "scale": cell["scale"],
        "block_size_q": 1,
        "block_size_k": block,
        "block_size_h": 1 << (cell["heads_q"] // hkv - 1).bit_length(),
        "block_size_d": dim,
        "num_q_loop": 1,
        "grid": [length, hkv, 1],
        "pack_gqa": "native grouped query-head tile",
        "num_splits": 1,
        "topk_chunk_merge": False,
        "num_warps": "Triton CUDA default (upstream does not override)",
        "num_stages": "Triton CUDA default (upstream does not override)",
        "autotune": "none; shipped prefill heuristics",
        "output_preallocated": True,
        "per_call_sequence": ["_gqa_sparse_fwd_kernel"],
        "lse_base": None,
    }
    return _warm(_Prepared(run, config), start, q.device)


def _prepare_flash_msa(cell, q, k, v, sel):
    torch.cuda.synchronize(q.device)
    start = perf_counter()
    module, metadata_module = load_flash_msa()
    from flash_msa._flash_attn_compat import _fa4_varlen_func
    from flash_msa.sparse_flash_varlen import TASKS_PER_FLASH_CHUNK

    # The reverse index rejects key_block >= q//128 for remote tasks, including
    # the already-selected local block. run_main_forward adds the causal local
    # block exactly once. Keep sel unchanged: dropping/appending local entries
    # is unnecessary, and duplicating earlier blocks would double-count them.
    metadata = metadata_module.build_sparse_attention_metadata_cuda(
        sel.unsqueeze(0), backward_query_chunk=32, remote_query_chunk=1024,
    )
    q_native = q.transpose(1, 2).contiguous()
    k_native = k.transpose(1, 2).contiguous()
    v_native = v.transpose(1, 2).contiguous()
    remote_backend = "FA4 paged-varlen" if _fa4_varlen_func() is not None else "FA3 varlen (gathered KV fallback)"

    def run():
        out, lse, _ = module.run_main_forward(
            q_native, k_native, v_native, scale=cell["scale"], metadata=metadata,
        )
        return out, lse

    config = {
        "backend": "FA3 local varlen + " + remote_backend,
        "pin": PINS["flash-msa"],
        "flash_attention_pin": PINS["flash-attention"],
        "api": "build_sparse_attention_metadata_cuda (prep); run_main_forward (run)",
        "layout": "BHSD",
        "layout_conversion": "BSHD to contiguous BHSD once in untimed prep; native internal gathers/copies remain timed; O to BSHD only in outputs",
        "kv_replication": False,
        "selection": "shared sel unchanged with added batch axis; local excluded from remote index and supplied once by causal local attention",
        "selection_pattern": cell["mask"]["pattern"],
        "proxy_heads": cell["heads_kv"],
        "heads_per_proxy": cell["heads_q"] // cell["heads_kv"],
        "scale": cell["scale"],
        "block_size": 128,
        "backward_query_chunk": 32,
        "remote_query_chunk": 1024,
        "tasks_per_flash_chunk": TASKS_PER_FLASH_CHUNK,
        "num_remote_tasks": metadata.num_remote_tasks,
        "remote_flash_calls_and_merges": (metadata.num_remote_tasks + TASKS_PER_FLASH_CHUNK - 1) // TASKS_PER_FLASH_CHUNK,
        "local_attention": {"causal": True, "max_seqlen_q": 128, "max_seqlen_k": 128},
        "remote_attention": {"causal": False, "max_seqlen_q": 1024, "max_seqlen_k": 128},
        "num_splits": 1,
        "pack_gqa": "native FlashAttention auto dispatch",
        "tile": "native FlashAttention SM90 d128 dispatch; no overrides",
        "autotune": "none; native flash-msa chunking and FlashAttention heuristics",
        "output_preallocated": False,
        "workspace": "native run_main_forward has no output/workspace argument; allocations and accumulator initialization remain timed",
        "per_call_sequence": [
            "native local layout/sequence setup and causal FA3 varlen",
            "FP32 output and natural-log LSE accumulator initialization",
            "each remote chunk: metadata arithmetic, Q gather, remote attention, online CUDA O/LSE merge",
            "native bf16 output/LSE finalization and zero KL-loss scalar",
        ],
        "lse_base": "e",
    }
    return _warm(_Prepared(run, config, output_bhsd=True), start, q.device)


IMPLS = (
    Impl("phi3-small-triton", "production", _supports_phi3, _prepare_phi3),
    Impl("vllm-m3-triton", "production", _supports_m3, _prepare_vllm_m3),
    Impl("flash-msa", "production", _supports_flash_msa, _prepare_flash_msa),
)
