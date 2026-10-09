"""Pinned TileLang WS transform; lineage: generated/ws_gemm/generate.py."""
import tilelang.language as T
from pipelink_tilelang.declarations import (
    Declarations, Endpoint, InitContract, ParamField, ParamsSpec, Role, Sharing,
    StageDeclaration, Variants,
)

SEED = 10302
KIND = "ws_gemm"
DEPTHS = (2, 3)
PASS_CONFIGS = {"tl.disable_warp_specialized": False}
PLANES = {"a_ring": "A_shared", "b_ring": "B_shared"}

PIPELINK = Declarations(stages=tuple(
    StageDeclaration(
        stage_id=f"org.pipelink.tilelang.ws.{role}",
        version="1.0.0+pipelink.tilelang",
        roles={role: Role(branch=branch)},
        endpoints=(
            Endpoint("a_ring", direction, role, sharing=Sharing(tx_bytes=("a_ring", "b_ring")),
                     init=InitContract(materializer="dispatch", phase=0, owner="producer"),
                     full_arrive_by="producer", empty_arrive_by="consumer"),
            Endpoint("b_ring", direction, role, sharing=Sharing(member_of="a_ring"),
                     init=InitContract(materializer="dispatch", phase=0, owner="producer"),
                     full_arrive_by="producer", empty_arrive_by="consumer"),
        ),
        launch_register_cap=255,
        params=ParamsSpec(
            name=f"org_pipelink_tilelang_ws_{role}_Params",
            fields=(ParamField("A_desc", "CUtensorMap"), ParamField("B_desc", "CUtensorMap"))
            if role == "producer" else (ParamField("C", "float*"),),
            alignas=64,
        ),
        variants=Variants(depth_handle="pipelink_tilelang_ring_depth", values=DEPTHS),
    )
    for role, branch, direction in (
        ("producer", "then", "produce"), ("consumer", "else", "consume")
    )
))


def program(depth=3):
    @T.prim_func
    def ws_gemm(A: T.Tensor((256, 512), "bfloat16"), B: T.Tensor((256, 512), "bfloat16"), C: T.Tensor((256, 256), "float32")):
        with T.Kernel(2, 2, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), "bfloat16")
            B_shared = T.alloc_shared((128, 64), "bfloat16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.clear(C_local)
            for k in T.Pipelined(8, num_stages=depth):
                T.copy(A[by * 128, k * 64], A_shared)
                T.copy(B[bx * 128, k * 64], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * 128, bx * 128])
    return ws_gemm
