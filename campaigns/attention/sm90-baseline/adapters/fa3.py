"""Pinned FA3 native, virtual-chunk, and stock-slot equivalence products.

The public APIs allocate O/LSE and scheduler storage themselves; they expose no
output/workspace arguments. Keep their complete per-call sequence (including
scheduler resets and varlen preparation) inside run. Input chunking is only a
zero-copy layout view plus untimed cu_seqlens construction. FA3 is AOT-built;
prepare also pays its first launch/module-loading cost, not a Python kernel JIT.
"""
from __future__ import annotations

from functools import partial
from time import perf_counter

from . import Impl
from ._loaders import PINS, load_fa3


_NATIVE_WORKLOADS = ("W1", "W2a", "W2b", "W3a", "W3b", "W4")


def _supports(cell, *, workloads):
    if cell["workload"] not in workloads:
        if cell["mask"]["kind"] in ("phi3_blocksparse", "msa_selected"):
            return "FA3 has no exact public per-head block mask or per-query selected-block API"
        return "This FA3 product is restricted to " + ", ".join(workloads)
    if cell["dtype"] != "bf16" or cell["head_dim"] != 128:
        return "The pinned FA3 build is bf16 head_dim=128 only"
    if cell["batch"] != 1 or cell["seqlen_q"] != cell["seqlen_k"]:
        return "These adapters implement batch-one, equal-length prefill cells"
    if cell["mask"]["kind"] not in ("causal", "sliding_window", "chunked"):
        return "FA3 cannot represent this mask exactly"
    return None


class _Prepared:
    def __init__(self, function, args, kwargs, config, *, varlen):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.config = config
        self.varlen = varlen

    def run(self):
        self.result = self.function(*self.args, **self.kwargs)
        return self.result

    def outputs(self):
        out, lse = self.result
        if self.varlen:
            # Native varlen O is [total_q,Hq,D], LSE is [Hq,total_q],
            # not [virtual_batch,Hq,max_seqlen]. Tokens remain in original order.
            return out.unsqueeze(0), lse.unsqueeze(0)
        return out, lse


def _prepare(cell, q, k, v, sel, *, varlen=False, slotcfg=False):
    import torch

    torch.cuda.synchronize(q.device)
    started = perf_counter()
    module = load_fa3()
    mask = cell["mask"]
    kwargs = {
        "softmax_scale": cell["scale"],
        "causal": True,
        "window_size": (mask["window"] - 1, 0) if mask["kind"] == "sliding_window" else (-1, -1),
        "attention_chunk": mask["chunk"] if mask["kind"] == "chunked" and not varlen else 0,
        "softcap": cell["softcap"] if cell["softcap"] is not None else 0.0,
        "num_splits": 1,
        "pack_gqa": False if slotcfg else None,
        "sm_margin": 0,
        "return_attn_probs": True,
    }
    function = module.flash_attn_func
    args = (q, k, v)
    config = {
        "backend": "FA3 SM90",
        "source_pin": PINS["flash-attention"],
        "api": "flash_attn_func",
        **kwargs,
        "dispatch": "stock-slot settings" if slotcfg else "shipped FA3 heuristics",
        # Pinned tile_size_fwd_sm90: bf16 d128 causal/local uses 128x128.
        # run_mha_fwd excludes causal/local/varlen from cluster-M=2 dispatch.
        "tile_mn": [128, 128],
        "cluster_m": 1,
        "autotune": False,
        "kv_replication": False,
        "input_layout": "BSHD",
        "input_layout_conversion": "none",
        "lse_base": "e",
        "output_allocation": "per call by public FA3 API (no out/LSE/workspace arguments)",
        "timed_sequence": "native scheduler-semaphore reset + attention",
    }
    if varlen:
        length, chunk = cell["seqlen_q"], mask["chunk"]
        # Consecutive chunks are independent causal sequences. Equal Q/K chunk
        # boundaries preserve the original k<=q relation, including a tail.
        boundaries = [*range(0, length, chunk), length]
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=q.device)
        maximum = min(chunk, length)
        args = (q[0], k[0], v[0], cu_seqlens, cu_seqlens, maximum, maximum)
        function = module.flash_attn_varlen_func
        config.update(
            api="flash_attn_varlen_func",
            input_layout="SHD (zero-copy batch-axis removal)",
            input_layout_conversion="views and cu_seqlens prepared once, untimed",
            chunk_size=chunk,
            virtual_batches=len(boundaries) - 1,
            max_seqlen_q=maximum,
            max_seqlen_k=maximum,
            native_lse_layout="Hq,total_q",
            timed_sequence="native varlen scheduler preparation/reset + attention",
        )
    # Report the pinned get_pack_gqa/should_pack_gqa decision without overriding
    # it: varlen packs GQA; fixed lengths pack only to reduce tile padding.
    length = cell["seqlen_q"]
    groups = cell["heads_q"] // cell["heads_kv"]
    efficiency = length / (((length + 127) // 128) * 128)
    packed_length = length * groups
    packed_efficiency = packed_length / (((packed_length + 127) // 128) * 128)
    config["pack_gqa_resolved"] = not slotcfg and groups > 1 and (
        varlen or efficiency < 0.9 * packed_efficiency
    )
    config["dispatch_facts_source"] = "pinned hopper/{tile_size.h,heuristics.h,flash_fwd_launch_template.h}"
    prepared = _Prepared(function, args, kwargs, config, varlen=varlen)
    prepared.run()
    torch.cuda.synchronize(q.device)
    prepared.prep_ms = (perf_counter() - started) * 1000
    return prepared


IMPLS = [
    Impl("fa3", "expert", partial(_supports, workloads=_NATIVE_WORKLOADS), _prepare),
    Impl(
        "fa3-dense-ceiling", "ceiling",
        partial(_supports, workloads=("W6dense",)), _prepare,
    ),
    Impl(
        "fa3-varlen-chunks", "production",
        partial(_supports, workloads=("W4",)), partial(_prepare, varlen=True),
    ),
    Impl(
        "fa3-slotcfg", "equivalence",
        partial(_supports, workloads=("W1", "W2a")), partial(_prepare, slotcfg=True),
    ),
]
