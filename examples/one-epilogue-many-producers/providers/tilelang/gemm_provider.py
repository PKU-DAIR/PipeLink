"""Register accumulator provider, lowered independently for Hopper/sm90 and Blackwell/sm103.

The owner calls this single role with all 128 CTA threads (threadIdx.x=0..127),
a caller-owned 128-float frame per thread, positive M/N multiples of 128, and K
positive and divisible by 64. These shape facts enable 16-byte cp.async loads.
T.Pipelined emits CTA-wide __syncthreads; externalization binds those operations
to one claimed named barrier with the captured 128-participant count. This is
not a warp-specialized kernel and must not be called by only part of this CTA.
The completed fragment is published by the caller's tensor_ready rendezvous;
release is a no-op after import_done. No GPU is needed to lower or package it.

From the repository root, after sourcing tools/env.sh:
  uv run --project frontends/tilelang --locked python -m pipelink_tilelang \
    --kernel examples/one-epilogue-many-producers/providers/tilelang/gemm_provider.py \
    --profile sm_90a --out /path/to/provider
Use sm_103a for the other independently inferred fragment placement.
"""
from pathlib import Path

import tilelang.language as T

from pipelink_tilelang import TensorExportStage, export_fragment
from pipelink_tilelang.declarations import ParamField, ParamsSpec


SEED = 0
PASS_CONFIGS = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
TENSOR_EXPORT = TensorExportStage(
    stage_id="tilelang.gemm.reg", version="1.0.0",
    args=ParamsSpec("pl_gemm_tile_args", (
        ParamField("a", "const void*"), ParamField("b", "const void*"),
        ParamField("M", "int32_t"), ParamField("N", "int32_t"), ParamField("K", "int32_t")), 8),
    args_header=str(Path(__file__).resolve().parents[1] / "abi/gemm_tile_args.h"),
    arguments={"A": "a", "B": "b", "M": "M", "N": "N", "K": "K"},
    fragment="C_local", name="accumulator", work="tensor_work", snapshot="completed_k",
    numerical="fp16_dot_fp32_completed", axes=("m", "n"),
    ready="tensor_ready", done="import_done", owner="tensor_caller",
    produce="tensor_produce", release="tensor_release",
    produce_symbol="plx_produce_source", release_symbol="plx_release",
    capabilities=("reg_copy", "reg_copy_round", "shfl_xor", "smem_store_scalar", "smem_store_round"),
)


def program():
    M, N, K = T.symbolic("M"), T.symbolic("N"), T.symbolic("K")

    @T.prim_func
    def gemm(A: T.Tensor((M, K), "float16"), B: T.Tensor((N, K), "float16")):
        with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
            T.assume(M % 128 == 0)
            T.assume(N % 128 == 0)
            T.assume(K % 64 == 0)
            A_shared = T.alloc_shared((128, 64), "float16")
            B_shared = T.alloc_shared((128, 64), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 64), num_stages=2):
                T.copy(A[bx * 128:bx * 128 + 128, k * 64:k * 64 + 64], A_shared)
                T.copy(B[by * 128:by * 128 + 128, k * 64:k * 64 + 64], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.evaluate(export_fragment(C_local))

    return gemm
