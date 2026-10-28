"""Tensor-export fixture: a register accumulator provider using each target's native GEMM instructions.

The owner calls this single role with all 128 CTA threads (threadIdx.x=0..127),
a caller-owned 128-float frame per thread, positive M/N multiples of 128, and K
positive and divisible by 64. These shape facts enable 16-byte cp.async loads.
T.Pipelined emits CTA-wide __syncthreads; the frontend rebinds them to one claimed
named barrier with a 128-thread count, so the whole CTA must call this role (it is
not warp-specialized). The caller's tensor_ready rendezvous publishes the finished
register fragment. On sm103, tcgen05 accumulates in owned TMEM before copying to the
fragment; release frees TMEM after import_done. The sm90 WGMMA and sm120 mma.sync
paths accumulate directly in registers and need no release work.

Lower independently with --profile sm_90a, sm_103a or sm_120a. Fragment placement,
internal allocation widths and barrier storage are captured by the frontend.
"""
from pathlib import Path

import tilelang.language as T

from pipelink_tilelang import TensorExportStage, context, export_fragment
from pipelink_tilelang.declarations import ParamField, ParamsSpec


SEED = 0
PASS_CONFIGS = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
TENSOR_EXPORT = TensorExportStage(
    stage_id="tilelang.gemm.reg", version="1.0.0",
    args=ParamsSpec("pl_gemm_tile_args", (
        ParamField("a", "const void*"), ParamField("b", "const void*"),
        ParamField("M", "int32_t"), ParamField("N", "int32_t"), ParamField("K", "int32_t")), 8),
    args_header=str(Path(__file__).resolve().parent / "gemm_tile_args.h"),
    arguments={"A": "a", "B": "b", "M": "M", "N": "N", "K": "K"},
    fragment="C_local", name="accumulator", work="tensor_work", snapshot="completed_k",
    numerical="bf16_dot_fp32_completed", axes=("m", "n"),
    ready="tensor_ready", done="import_done", owner="tensor_caller",
    produce="tensor_produce", release="tensor_release",
    produce_symbol="plx_produce_source", release_symbol="plx_release",
    capabilities=("reg_copy", "reg_copy_round", "shfl_xor", "smem_store_scalar", "smem_store_round"),
    internal_storage=("C_tmem", "mma_barrier") if context.profile == "sm_103a" else (),
)


def program():
    M, N, K = T.symbolic("M"), T.symbolic("N"), T.symbolic("K")
    if context.profile == "sm_103a":
        @T.prim_func
        def gemm_tcgen05(A: T.Tensor((M, K), "bfloat16"), B: T.Tensor((N, K), "bfloat16")):
            with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
                T.assume(M % 128 == 0)
                T.assume(N % 128 == 0)
                T.assume(K % 64 == 0)
                A_shared = T.alloc_shared((128, 64), "bfloat16")
                B_shared = T.alloc_shared((128, 64), "bfloat16")
                C_tmem = T.alloc_tmem((128, 128), "float32")
                mma_barrier = T.alloc_barrier(1)
                C_local = T.alloc_fragment((128, 128), "float32")
                for k in T.Pipelined(T.ceildiv(K, 64), num_stages=2):
                    T.copy(A[bx * 128:bx * 128 + 128, k * 64:k * 64 + 64], A_shared)
                    T.copy(B[by * 128:by * 128 + 128, k * 64:k * 64 + 64], B_shared)
                    T.gemm(A_shared, B_shared, C_tmem, transpose_B=True,
                           clear_accum=(k == 0), mbar=mma_barrier)
                T.copy(C_tmem, C_local)
                T.evaluate(export_fragment(C_local))
        return gemm_tcgen05

    @T.prim_func
    def gemm(A: T.Tensor((M, K), "bfloat16"), B: T.Tensor((N, K), "bfloat16")):
        with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
            T.assume(M % 128 == 0)
            T.assume(N % 128 == 0)
            T.assume(K % 64 == 0)
            A_shared = T.alloc_shared((128, 64), "bfloat16")
            B_shared = T.alloc_shared((128, 64), "bfloat16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 64), num_stages=2):
                T.copy(A[bx * 128:bx * 128 + 128, k * 64:k * 64 + 64], A_shared)
                T.copy(B[by * 128:by * 128 + 128, k * 64:k * 64 + 64], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.evaluate(export_fragment(C_local))

    return gemm
