"""Runtime-scalar linear combination, P-TL-4 generation lineage (not v3 bias)."""
import tilelang.language as T

from pipelink_tilelang.declarations import (
    Declarations, Endpoint, InitContract, ParamField, ParamsSpec, Role,
    StageDeclaration, Variants,
)

SEED = 10304
KIND = "f1_epilogue"
DEPTHS = (4, 5, 6)
PASS_CONFIGS = {"tl.disable_warp_specialized": True}

PIPELINK = Declarations(stages=(
    StageDeclaration(
        stage_id="org.pipelink.tilelang.f1.epilogue",
        version="1.0.0+pipelink.tilelang",
        roles={
            "epilogue": Role(branch="body"),
            "epilogue_load": Role(branch="family"),
        },
        endpoints=(
            Endpoint("accumulator", "consume", "epilogue",
                     init=InitContract(materializer="stage", phase=0),
                     cluster_leader_policy="mma_pair_leader"),
            Endpoint("tmem_dealloc", "consume", "mma"),
            Endpoint("clc", "consume", "epilogue",
                     init=InitContract(materializer="stage", phase=0)),
            Endpoint("load_order", "consume", "epilogue_load",
                     init=InitContract(materializer="stage", phase=0)),
            Endpoint("tmem_base", "consume", "epilogue"),
        ),
        launch_register_cap=160,
        params=ParamsSpec(
            name="org_pipelink_tilelang_f1_epilogue_Params",
            fields=(
                ParamField("family", "::pipelink::SplitParams<::pipelink::V1FusionOp>"),
                ParamField("alpha", "float"),
                ParamField("beta", "float"),
            ),
            alignas=128,
        ),
        variants=Variants(depth_handle="pipelink_gemm_evt_depth", values=DEPTHS),
    ),
))


def program():
    @T.prim_func
    def linear(A: T.Tensor((4096,), "float32"), C: T.Tensor((4096,), "bfloat16"), D: T.Tensor((4096,), "bfloat16"), alpha: T.float32, beta: T.float32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(4096):
                D[i] = A[i] * alpha + T.Cast("float32", C[i]) * beta
    return linear
