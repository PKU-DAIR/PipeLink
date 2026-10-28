"""One completed accumulator generation, consumed in eight column chunks."""
import os
from pathlib import Path

import tilelang.language as T
import pipelink_tilelang as pl

partner = pl.Partner.load(os.environ["PIPELINK_ACC_PARTNER"])
acc = pl.import_tensor(partner, "accumulator", outputs=("m0", "n0"), scratch_bytes=0)
THREADS = 384 if partner.profile == "sm_90a" else 256


def program():
    @T.prim_func
    def swiglu(a: T.handle, b: T.handle, output_handle: T.handle,
               M: T.int32, N: T.int32, K: T.int32):
        output = T.match_buffer(output_handle, (M, N // 2), "bfloat16")
        with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 256), threads=THREADS):
            T.assume(M % 128 == 0)
            T.assume(N % 256 == 0)
            frag = acc.fragment(specialize=True, chunk=("n", 32))
            gate = T.alloc_fragment((128, 16), "float32")
            up = T.alloc_fragment((128, 16), "float32")
            with acc.produced(completion="final"):
                m0 = acc.args.m0
                n0 = acc.args.n0
                T.assume(m0 >= 0)
                T.assume(m0 + 128 <= M)
                T.assume(n0 >= 0)
                T.assume(n0 + 256 <= N)
                T.assume(n0 % 256 == 0)
                staging = acc.staging((128, 128), "bfloat16")
                T.annotate_layout({staging: T.Layout(
                    (128, 128), lambda i, j: (i, T.bitwise_xor(j // 8, i % 8) * 8 + j % 8))})
                for chunk in T.serial(8):
                    with acc.imported(frag, chunk=chunk):
                        for i, j in T.Parallel(128, 16):
                            gate[i, j] = frag[i, 2 * j]
                        for i, j in T.Parallel(128, 16):
                            up[i, j] = frag[i, 2 * j + 1]
                        for i, j in T.Parallel(128, 16):
                            staging[i, chunk * 16 + j] = T.cast(
                                T.call_pure_extern("float32", "__fdividef", gate[i, j],
                                    1.0 + T.call_pure_extern("float32", "__expf", -gate[i, j]))
                                * up[i, j], "bfloat16")
                T.copy(staging, output[m0:m0 + 128, n0 // 2:n0 // 2 + 128],
                       coalesced_width=8,
                       loop_layout=T.Fragment(
                           (128, 128),
                           forward_thread_fn=lambda row, column: (row % 16) * 16 + column // 8,
                           forward_index_fn=lambda row, column: (row // 16) * 8 + column % 8))
    return swiglu


STAGE = pl.contract_stage(
    "coda.consumer.tilelang-swiglu", "1.0.0", program,
    host=pl.HostParams(
        str(Path(__file__).resolve().parents[1] / "consumer/host_annotations.cuh"),
        "CodaEpilogueParams",
        {"CODA_READER": '"coda.consumer.tilelang-swiglu:swiglu"',
         "CODA_ROLE_VIEW": "plk_role_view_coda_consumer_tilelang_swiglu_swiglu",
         "CODA_PROFILE": partner.profile.removeprefix("sm_").removesuffix("a")}))
program = STAGE.program
