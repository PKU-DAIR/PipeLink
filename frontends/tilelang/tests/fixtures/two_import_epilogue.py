"""Two opaque imports: out = max(acc * scale[m] + bias[m], 0).

The accumulator optionally adopts the provider's register placement; TileLang
infers the scale layout from the accumulator's row holders.
"""
import os

import tilelang.language as T
import pipelink_tilelang as pl

acc = pl.import_tensor(pl.Partner.load(os.environ["PIPELINK_ACC_PARTNER"]), "accumulator")
scale = pl.import_tensor(pl.Partner.load(os.environ["PIPELINK_SCALE_PARTNER"]), "row_scale")


def program():
    @T.prim_func
    def epilogue(bias_handle: T.handle, output_handle: T.handle, M: T.int32, N: T.int32):
        bias = T.match_buffer(bias_handle, (M,), "float32")
        output = T.match_buffer(output_handle, (M, N), "float32")
        with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
            T.assume(M % 128 == 0)
            T.assume(N % 128 == 0)
            acc_fragment = acc.fragment(specialize=os.environ.get("PIPELINK_ACC_SPECIALIZE", "1") != "0")
            scale_fragment = scale.fragment()
            bias_fragment = T.alloc_fragment((128,), "float32")
            with acc.imported(acc_fragment):
                with scale.imported(scale_fragment):
                    # Loaded before any store: output may alias bias.
                    T.copy(bias[bx * 128:bx * 128 + 128], bias_fragment)
                    for m, n in T.Parallel(128, 128):
                        output[bx * 128 + m, by * 128 + n] = T.max(
                            acc_fragment[m, n] * scale_fragment[m] + bias_fragment[m], 0)
    return epilogue


STAGE = pl.contract_stage("tensor.consumer.tilelang-epilogue", "1.0.0", program)
program = STAGE.program
