"""One TileLang epilogue, authored against two opaque tensor providers."""
import os

import tilelang.language as T

from pipelink_tilelang.contract import Partner, TensorDemand, TensorRendezvous, import_tensor
from pipelink_tilelang.declarations import ParamField, ParamsSpec
from pipelink_tilelang.explicit import ContractStage, RoleSpec, emit

acc_partner = Partner.load(os.environ["PIPELINK_ACC_PARTNER"])
scale_partner = Partner.load(os.environ["PIPELINK_SCALE_PARTNER"])


def channel(partner, export, symbol, demand, words, scratch_bytes, arguments, frame=False):
    readiness = next(item for item in partner.raw["tensor"]["exports"] if item["name"] == export)["readiness"]
    lends = [
        dict(kind="smem", name="scratch", extent=scratch_bytes, align=128, handle=f"pipelink_{symbol}_scratch"),
        dict(kind="regs", name="destination", extent=words, handle="fragment"),
        dict(kind="cta_barrier", name="outer", base=0, sync=(2, 1)),
    ]
    if frame:
        lends.extend([
            dict(kind="frame", name="caller_frame", extent=words, handle=f"{symbol}_frame"),
            dict(kind="metadata", name="metadata", extent=2, handle=f"{symbol}_metadata"),
        ])
    return import_tensor(partner, export, role="epilogue", symbol=symbol, demand=demand, lends=lends,
                         rendezvous=tuple(TensorRendezvous(readiness[event], "outer")
                                          for event in ("after", "reuse_after")), arguments=arguments)


acc = channel(acc_partner, "accumulator", "pl_import_acc",
              TensorDemand((("t", 0, 128), ("p", 0, 128)), ("t", "p", "t", "p"),
                           guarantees=("enter_once", "single_completion", "converged", "full_warps", "no_early_return")),
              128, 65536, {name: name for name in ("a", "b", "M", "N", "K")}, frame=True)
scale = channel(scale_partner, "row_scale", "pl_import_scale",
                TensorDemand((("t", 0, 128),), ("t", "t", 0),
                             guarantees=("enter_once", "single_completion", "converged", "full_warps", "no_early_return")), 1, 1024,
                {name: name for name in ("a", "M", "K", "eps")})

SEED = 10390
DEPTHS = ()
PASS_CONFIGS = {"tl.disable_warp_specialized": True}
STAGE = ContractStage(
    stage_id="tensor.consumer.tilelang-epilogue",
    version="1.0.0",
    launch_register_cap=255,
    params=ParamsSpec("EpilogueParams", (
        ParamField("a", "const uint16_t*"), ParamField("b", "const uint16_t*"),
        ParamField("bias", "const float*"), ParamField("output", "float*"),
        ParamField("M", "int32_t"), ParamField("N", "int32_t"), ParamField("K", "int32_t"),
        ParamField("eps", "float"),
    ), alignas=8),
    roles={"epilogue": RoleSpec((0, 4), (acc.args_param, scale.args_param, "bias", "output", "M", "N"))},
    channels=[acc, scale],
)
EMITTER = emit


def program():
    @T.prim_func
    def epilogue(pl_import_acc_args: T.handle, pl_import_scale_args: T.handle,
                 bias_handle: T.handle, output_handle: T.handle, M: T.int32, N: T.int32):
        bias = T.match_buffer(bias_handle, (M,), "float32")
        output = T.match_buffer(output_handle, (M, N), "float32")
        with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
            tx = T.get_thread_binding()
            frame = T.alloc_fragment((128,), "float32")
            fragment = T.alloc_fragment((128,), "float32")
            scales = T.alloc_fragment((1,), "float32")
            acc_scratch = acc.scratch("scratch")
            scale_scratch = scale.scratch("scratch")
            source = T.call_extern("uint64", "pipelink_tilelang_source_word",
                T.call_extern("handle", acc.produce_symbol, acc.args(pl_import_acc_args), T.address_of(frame[0])))
            T.call_extern("void", acc.symbol + "_ready")
            T.call_extern("void", acc.symbol,
                T.call_extern("handle", "pipelink_tilelang_source_from_word", source),
                T.address_of(fragment[0]), T.address_of(acc_scratch[0]))
            scale_source = T.call_extern("uint64", "pipelink_tilelang_source_word",
                T.call_extern("handle", scale.produce_symbol, scale.args(pl_import_scale_args), T.address_of(frame[0])))
            T.call_extern("void", scale.symbol + "_ready")
            T.call_extern("void", scale.symbol,
                T.call_extern("handle", "pipelink_tilelang_source_from_word", scale_source),
                T.address_of(scales[0]), T.address_of(scale_scratch[0]))
            T.assume(N % 128 == 0)
            # 16-byte stores, like the hand consumer; arithmetic is unchanged.
            for q in T.unroll(32):
                for v in T.vectorized(4):
                    p = q * 4 + v
                    output[bx * 128 + tx, by * 128 + p] = T.max(fragment[p] * scales[0] + bias[bx * 128 + tx], 0)
            T.call_extern("void", scale.symbol + "_done")
            T.call_extern("void", scale.release_symbol, scale.args(pl_import_scale_args))
            T.call_extern("void", acc.symbol + "_done")
            T.call_extern("void", acc.release_symbol, acc.args(pl_import_acc_args))
    return epilogue
