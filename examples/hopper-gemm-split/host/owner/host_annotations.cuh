#pragma once
#include "abi.h"
#include <pipelink/stage.hpp>

#if defined(PIPELINK_EMIT)
PL_HOST_GROUP("gemm.owner", "gemm.owner.params", "gemm", plk_header_v2,
              PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_ABI_TYPE(100, GemmOwnerResidue, load, mma);
PL_HOST_ABI_TYPE(101, GemmLoadParams, a, b, K);
PL_HOST_ABI_TYPE(102, GemmMmaParams, K);
PL_HOST_ABI_TYPE(103, plk_role_view_gemm_load_load, residue, scalars);
PL_HOST_ABI_TYPE(104, plk_role_view_gemm_mma_mma, residue, scalars);
PL_HOST_RESIDUE(2, "gemm.owner.residue", GemmOwnerResidue, 100, true,
    pl::host::roles({"gemm.load:load", "gemm.mma:mma"}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"gemm.load:load", "gemm.mma:mma"}))
    .field(pl::host::scalar(1, "gemm.M", "i64").caller("M"))
    .field(pl::host::scalar(2, "gemm.N", "i64").caller("N"))
    .field(pl::host::scalar(3, "gemm.K", "i64").caller("K")));
// Load encodes these tensors' descriptors internally and exposes only an smem
// endpoint, so no gmem endpoint is bound and no transfer is reported.
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("gemm.a", "bf16").rank(2).read().caller("a"));
PL_HOST_CHANNEL(2, uint16_t, pl::host::channel("gemm.b", "bf16").rank(2).read().caller("b"));
PL_HOST_SCHEDULER(pl::host::scheduler("gemm.owner", "gemm.static-grid", pl::host::schema()));
// One unkeyed variant per stage; depth stays an ordinary link-time binding.
PL_HOST_VARIANTS("gemm.load", pl::host::variants(std::vector<pl::fixed_bindings>{{}}));
PL_HOST_VARIANTS("gemm.mma", pl::host::variants(std::vector<pl::fixed_bindings>{{}}));
#endif
