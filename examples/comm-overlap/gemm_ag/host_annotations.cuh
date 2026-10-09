#pragma once
// GEMM owner's second host group. Shared by load_ag.cu and the
// repacked, byte-identical hopper-gemm-split src/mma.cu (staged by build.py so that
// its relative "../host/owner/host_annotations.cuh" include resolves here).
#include "abi.h"
#include <pipelink/stage.hpp>

#if defined(PIPELINK_EMIT)
PL_HOST_GROUP("gemm.owner", "gemm.owner.ag", "gemm_ag", plk_header_v2,
              PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "tp");
PL_HOST_ABI_TYPE(100, GemmAgResidue, gemm, ag);
PL_HOST_ABI_TYPE(101, GemmOwnerResidue, load, mma);
PL_HOST_ABI_TYPE(102, GemmLoadParams, a, b, K);
PL_HOST_ABI_TYPE(103, GemmMmaParams, K);
PL_HOST_ABI_TYPE(104, GemmAgLoadParams, ready, gen, rank, rows_per_rank, expected, reserved);
PL_HOST_ABI_TYPE(105, plk_role_view_gemm_load_ag_load, residue, scalars);
PL_HOST_ABI_TYPE(106, plk_role_view_gemm_mma_mma, residue, scalars);
PL_HOST_RESIDUE(2, "gemm.owner.ag.residue", GemmAgResidue, 100, true,
    pl::host::roles({"gemm.load.ag:load", "gemm.mma:mma"}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"gemm.load.ag:load", "gemm.mma:mma"}))
    .field(pl::host::scalar(1, "gemm.M", "i64").caller("M"))
    .field(pl::host::scalar(2, "gemm.N", "i64").caller("N"))
    .field(pl::host::scalar(3, "gemm.K", "i64").caller("K"))
    .field(pl::host::scalar(4, "comm.ag.chunk_rows", "i64").fact("comm.ag.chunk_rows"))
    .field(pl::host::scalar(5, "comm.ag.chunk_cols", "i64").fact("comm.ag.chunk_cols"))
    .field(pl::host::scalar(6, "comm.ag.ready_slot", "i64").binding("pipelink_comm_ready_slot")));
// Read views over the comm owner's caller sources ("gather", "ready", "gen"); the
// A descriptor is encoded over this rank's gathered buffer [M, K].
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("gemm.ag.a", "bf16").rank(2).read().caller("gather").peer("tp"));
PL_HOST_CHANNEL(2, uint16_t, pl::host::channel("gemm.b", "bf16").rank(2).read().caller("b"));
PL_HOST_CHANNEL(3, uint32_t, pl::host::channel("gemm.ag.ready", "u32").rank(1).read().caller("ready").peer("tp"));
PL_HOST_CHANNEL(4, uint32_t, pl::host::channel("gemm.ag.gen", "u32").rank(1).read().caller("gen"));
PL_HOST_SCHEDULER(pl::host::scheduler("gemm.owner", "gemm.static-grid", pl::host::schema()));
PL_HOST_VARIANTS("gemm.load.ag", pl::host::variants(std::vector<pl::fixed_bindings>{{}}));
PL_HOST_VARIANTS("gemm.mma", pl::host::variants(std::vector<pl::fixed_bindings>{{}}));
PL_HOST_TARGET();
#endif
