"""Exact FlexAttention comparisons, with forward-only block-list preparation.

All metadata construction, device transfers and compilation happen in prepare.
The W6 lists are unions of the supplied per-token selections, not regenerated
routing. Partial blocks retain the token predicate, including after Phi's
64-to-128 retile. Full blocks bypass it only when every token pair is visible.
"""
from __future__ import annotations

import ast
from functools import partial
import importlib.util
import time

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from . import Impl

BLOCK = 128
KINDS = {"causal", "sliding_window", "chunked", "phi3_blocksparse", "msa_selected"}


def _supports(cell, *, backend):
    if cell["workload"] == "W6dense" or cell.get("diagnostic", False):
        return "W6dense is an expert throughput ceiling, not a modular placement row"
    if cell["mask"]["kind"] not in KINDS:
        return f"No exact Flex mask for {cell['mask']['kind']}"
    if cell["head_dim"] != 128 or cell["dtype"] != "bf16":
        return "This sm90 adapter is configured for bf16 d128"
    if cell["seqlen_q"] != cell["seqlen_k"] or cell["seqlen_q"] % BLOCK:
        return "Block-list construction requires equal prefill lengths divisible by 128"
    if backend == "FLASH":
        try:
            available = importlib.util.find_spec("flash_attn.cute") is not None
        except (ImportError, ModuleNotFoundError):
            available = False
        if not available:
            return "Flex BACKEND=FLASH requires the pinned flash-attn-4 CuTe package"
        # All these forward-only masks and softcaps are expressible by the
        # pinned FLASH backend. No max_scores, CPU scalar captures or grad
        # buffers are requested (the backend's relevant rejection cases).
    return None


def _mask_mod(cell, sel):
    m = cell["mask"]
    kind = m["kind"]
    if kind == "causal":
        def mask(b, h, q, k):
            return k <= q
    elif kind == "sliding_window":
        window = m["window"]

        def mask(b, h, q, k):
            return (k <= q) & (k > q - window)
    elif kind == "chunked":
        chunk = m["chunk"]

        def mask(b, h, q, k):
            return (k <= q) & (k // chunk == q // chunk)
    elif kind == "phi3_blocksparse":
        block, local, stride = m["block"], m["local_blocks"], m["vert_stride"]
        step = 0 if m["homo_head"] else m["head_sliding_step"]

        def mask(b, h, q, k):
            qb, kb = q // block, k // block
            return (k <= q) & ((qb - kb < local) | ((kb + h * step + 1) % stride == 0))
    else:
        if sel is None:
            raise ValueError("W6 requires the shared materialized selection tensor")
        group = cell["heads_q"] // cell["heads_kv"]
        block, topk = m["block"], m["topk"]

        def mask(b, h, q, k):
            # Scalar loads keep the callback in both backends' pointwise
            # subset; no reduction over an extra dimension is needed.
            member = sel[h // group, q, 0] == k // block
            for slot in range(1, topk):
                member = member | (sel[h // group, q, slot] == k // block)
            return (k <= q) & member
    return mask


def _static_rows(cell):
    """Closed-form block rows; never allocate a token-by-token mask."""
    m = cell["mask"]
    kind, n = m["kind"], cell["seqlen_q"] // BLOCK
    heads = cell["heads_q"] if kind == "phi3_blocksparse" else 1
    partial_rows, full_rows = [], []
    for h in range(heads):
        partial_head, full_head = [], []
        for qb in range(n):
            q0, q1 = qb * BLOCK, (qb + 1) * BLOCK - 1
            ids = np.arange(qb + 1, dtype=np.int32)
            k0, k1 = ids * BLOCK, (ids + 1) * BLOCK - 1
            any_visible = np.ones(ids.shape, dtype=bool)
            all_visible = k1 <= q0
            if kind == "sliding_window":
                any_visible &= k1 > q0 - m["window"]
                all_visible &= k0 > q1 - m["window"]
            elif kind == "chunked":
                any_visible &= k0 // m["chunk"] == q0 // m["chunk"]
                all_visible &= k1 // m["chunk"] == q1 // m["chunk"]
            elif kind == "phi3_blocksparse":
                # Examine the four pairs of 64-token blocks, not 128^2
                # tokens. Causality still makes the diagonal tile partial.
                fine = m["block"]
                qfine = np.arange(q0 // fine, q1 // fine + 1)[:, None, None]
                kfine = ids[None, :, None] * (BLOCK // fine) + np.arange(BLOCK // fine)
                offset = 0 if m["homo_head"] else h * m["head_sliding_step"]
                visible = (kfine <= qfine) & (
                    (qfine - kfine < m["local_blocks"])
                    | ((kfine + offset + 1) % m["vert_stride"] == 0)
                )
                any_visible &= visible.any(axis=(0, 2))
                all_visible &= visible.all(axis=(0, 2))
            partial_head.append(ids[any_visible & ~all_visible])
            full_head.append(ids[all_visible])
        partial_rows.append(partial_head)
        full_rows.append(full_head)
    return partial_rows, full_rows


def _selected_rows(cell, sel):
    """Union/intersection over each 128-query tile, once per KV head."""
    # This copy is reported as preparation. The original GPU tensor, not this
    # host view, remains captured by mask_mod throughout all attention calls.
    chosen = sel.detach().cpu().numpy()
    n = cell["seqlen_q"] // BLOCK
    partial_rows, full_rows = [], []
    for h in range(cell["heads_kv"]):
        partial_head, full_head = [], []
        for qb in range(n):
            row = chosen[h, qb * BLOCK:(qb + 1) * BLOCK]
            ids, counts = np.unique(row[row >= 0], return_counts=True)
            # Slots are distinct within a query, so a count of BLOCK means
            # membership in every query's list. Current-block causal masking
            # must still be evaluated even when all queries select it.
            full = (counts == BLOCK) & (ids < qb)
            partial_head.append(ids[~full])
            full_head.append(ids[full])
        partial_rows.append(partial_head)
        full_rows.append(full_head)
    return partial_rows, full_rows


def _pack_rows(rows, repeat_heads, device):
    heads, nq = len(rows), len(rows[0])
    counts = np.asarray([[len(row) for row in head] for head in rows], dtype=np.int32)
    # Keep one unused slot even for all-empty lists; backend metadata strides
    # and shape checks remain regular. Counts alone determine visibility.
    width = max(1, int(counts.max()))
    indices = np.zeros((heads, nq, width), dtype=np.int32)
    for h, head in enumerate(rows):
        for q, row in enumerate(head):
            indices[h, q, :len(row)] = row
    if repeat_heads != 1:
        # The BlockMask H axis is query-head indexed, not KV-head indexed.
        # Only metadata is repeated; K/V tensors are never replicated.
        counts = np.repeat(counts, repeat_heads, axis=0)
        indices = np.repeat(indices, repeat_heads, axis=0)
    return (
        torch.from_numpy(counts[None]).to(device),
        torch.from_numpy(indices[None]).to(device),
        {"head_dim": len(counts), "max_blocks_per_row": int(counts.max()),
         "mean_blocks_per_row": float(counts.mean()), "index_width": width},
    )


def _block_mask(cell, sel, mask, device):
    selected = cell["mask"]["kind"] == "msa_selected"
    partial_rows, full_rows = _selected_rows(cell, sel) if selected else _static_rows(cell)
    repeat = cell["heads_q"] // cell["heads_kv"] if selected else 1
    if selected and all(
        np.array_equal(row, first)
        for rows in (partial_rows, full_rows)
        for head in rows[1:]
        for row, first in zip(head, rows[0])
    ):
        # In particular, the supplied prefix selections normally share their
        # block rows across KV heads. Preserve broadcast metadata (and FA4's
        # packed GQA) when proven from those rows, not from the pattern label.
        partial_rows, full_rows = partial_rows[:1], full_rows[:1]
        repeat = 1
    pn, pi, partial_stats = _pack_rows(partial_rows, repeat, device)
    fn, fi, full_stats = _pack_rows(full_rows, repeat, device)
    block_mask = BlockMask.from_kv_blocks(
        pn, pi, fn, fi, BLOCK_SIZE=(BLOCK, BLOCK), mask_mod=mask,
        seq_lengths=(cell["seqlen_q"], cell["seqlen_k"]), compute_q_blocks=False,
    )
    return block_mask, {
        "builder": "selection union/intersection" if selected else "closed-form block lists",
        "partial": partial_stats, "full": full_stats,
        "compute_q_blocks": False,
        "selection_host_copy_in_prep": selected,
        "metadata_query_head_repeat": repeat,
    }


def _triton_choices(sources):
    """Read the selected template's constants from pinned Inductor codegen.

    run_and_get_code also captures persistent FX-cache hits. Parsing generated
    code, rather than the candidate timing log, records the actually emitted
    choice without forcing a cache miss or changing the autotuning search.
    """
    choices = []
    for source in sources:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if "def " not in node.value or "BLOCK_M" not in node.value:
                continue
            kernel = ast.parse(node.value)
            values = {}
            for item in ast.walk(kernel):
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if isinstance(item.value, ast.Constant):
                        values[item.target.id] = item.value.value
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
                    if item.func.attr == "template":
                        for kw in item.keywords:
                            if kw.arg in ("num_warps", "num_stages"):
                                values[kw.arg] = ast.literal_eval(kw.value)
            if "BLOCK_M" in values and "BLOCK_N" in values:
                choices.append(values)
    if not choices:
        raise RuntimeError("Pinned Inductor did not expose the selected Flex Triton tile in generated code")
    return choices


class _Prepared:
    def __init__(self, cell, q, k, v, sel, *, backend):
        reason = _supports(cell, backend=backend)
        if reason is not None:
            raise ValueError(reason)
        torch.cuda.synchronize(q.device)
        started = time.perf_counter()
        mask = _mask_mod(cell, sel)
        block_mask, mask_config = _block_mask(cell, sel, mask, q.device)
        cap = cell["softcap"]
        score_mod = None
        if cap is not None:
            def score_mod(score, b, h, qi, ki):
                # Flex applies scale before invoking score_mod; do not scale
                # again (notably Gemma2's non-default 1/sqrt(144)).
                return cap * torch.tanh(score / cap)

        options = {"BACKEND": backend}
        # Keep the default online-softmax safety checks: a row can be empty
        # in the first partial tile of a union even though it includes self
        # later. Tile/warps/stages stay unset for maximum autotuning.
        scale = cell["scale"]

        def attention(qh, kh, vh):
            return flex_attention(
                qh, kh, vh, score_mod=score_mod, block_mask=block_mask,
                scale=scale, enable_gqa=True, return_lse=True,
                kernel_options=options,
            )

        self._inputs = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        compile_config = {"fullgraph": True, "dynamic": False}
        if backend == "FLASH" and cap is not None:
            # torch 2.14 get_flex_flash_fwd_configs enumerates score widths
            # 1..128 under max_autotune, but FA4 Sm90 rejects widths >2.
            # The supported non-autotuned path leaves __vec_size__ unset;
            # FA4 then uses its native width 2 for this tensor-free softcap.
            # This changes only codegen configuration, not score arithmetic.
            compile_config["options"] = {
                "max_autotune": False, "triton.cudagraphs": False,
            }
        else:
            compile_config["mode"] = "max-autotune-no-cudagraphs"
        self._compiled = torch.compile(attention, **compile_config)
        from torch._inductor.utils import run_and_get_code

        with torch.inference_mode():
            self._result, sources = run_and_get_code(self._compiled, *self._inputs)
        torch.cuda.synchronize(q.device)
        self.config = {
            "backend": backend, "kernel_options": options,
            "compile": compile_config,
            "block_size": [BLOCK, BLOCK], "block_mask": mask_config,
            "enable_gqa": True, "kv_replication": False,
            "input_layout": "BHSD transpose views of BSHD (no copy)",
            "output_allocation": "Inductor managed; public Flex API has no out argument",
            "scale": scale, "softcap": cap,
            "score_mod_input": "already scaled logits", "lse_base": "e",
            "return_lse": True, "cuda_graphs": False,
        }
        if backend == "TRITON":
            self.config.update(
                autotune_result=_triton_choices(sources),
                pack_gqa=False, num_splits=1,
                tuning_source="selected generated Inductor Triton template",
            )
        else:
            self.config.update(
                tile=[128, 128], num_stages=2, num_threads=384,
                pack_gqa=mask_config["partial"]["head_dim"] == 1,
                num_splits=1, mma_pv_is_rs=True, intra_wg_overlap=True,
                autotune_result="FA4 sm90 d128 library heuristic (no Flex tile search)",
                tuning_source="pinned FA4 _tile_size_fwd_sm90 and sparse-head pack_gqa guard",
            )
            if cap is not None:
                self.config.update(
                    score_mod_vec_size=2,
                    score_mod_vec_size_source="FA4 default without captured auxiliary tensors",
                    upstream_workaround=(
                        "Disable FLASH score-vector max_autotune: torch 2.14 enumerates "
                        "widths above 2, which FA4 Sm90 rejects with 'score_mod vec_size "
                        "4 not supported on Sm80/90/120 due to accumulator thread "
                        "ownership pattern'. Use the supported library-default width "
                        "2 instead; exact softcap arithmetic and FA4 tile heuristic "
                        "are unchanged. Triton remains maximum-autotuned."
                    ),
                )
        self.prep_ms = (time.perf_counter() - started) * 1000

    def run(self):
        self._result = self._compiled(*self._inputs)
        return self._result

    def outputs(self):
        out, lse = self._result
        return out.transpose(1, 2), lse


IMPLS = [
    Impl("flex-triton", "modular-lower", partial(_supports, backend="TRITON"),
         partial(_Prepared, backend="TRITON")),
    Impl("flex-flash", "modular-flash", partial(_supports, backend="FLASH"),
         partial(_Prepared, backend="FLASH")),
]
