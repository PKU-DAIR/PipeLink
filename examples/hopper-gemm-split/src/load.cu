#include "channels.cuh"
#include <cute/arch/copy_sm90_tma.hpp>
#if defined(GEMM_HOST_ABI) && GEMM_HOST_ABI
#include "../host/owner/host_annotations.cuh"
#endif

extern "C" __device__ void gemm_load_role(const void* raw_params, unsigned) {
  using namespace hopper_gemm;
#if defined(GEMM_HOST_ABI) && GEMM_HOST_ABI
  auto const& view = *static_cast<const plk_role_view_gemm_load_load*>(raw_params);
  auto const& params = static_cast<const GemmOwnerResidue*>(view.residue)->load;
#else
  auto const& params = *static_cast<const GemmLoadParams*>(raw_params);
#endif
  extern __shared__ __align__(1024) unsigned char gemm_smem[];
  // Dispatch initializes and publishes full/empty before any role starts.
  // The elected producer lane supplies one arrival and both TMA transactions.
  if ((threadIdx.x & 31) != 0) return;
  const int depth = pipelink_gemm_depth();
  for (int k = 0; k < params.K / TileK; ++k) {
    const int slot = k % depth;
    auto* full = barrier(gemm_smem, pipelink_gemm_full(), slot);
    auto* empty = barrier(gemm_smem, pipelink_gemm_empty(), slot);
    cutlass::arch::ClusterBarrier::wait(empty, ((k / depth) & 1) ^ 1);
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(full, SlotBytes);
    auto* operands = gemm_smem + pipelink_gemm_operands() + slot * SlotBytes;
    cute::SM90_TMA_LOAD_2D::copy(&params.a, full, uint64_t(0), operands,
                               k * TileK, int(blockIdx.x) * TileM);
    cute::SM90_TMA_LOAD_2D::copy(&params.b, full, uint64_t(0), operands + OperandBytes,
                               k * TileK, int(blockIdx.y) * TileN);
  }
}

PL_STAGE("gemm.load", "1.0.0");
#if defined(GEMM_HOST_ABI) && GEMM_HOST_ABI
PL_NO_PARAMS();
#else
PL_PARAMS(GemmLoadParams, a, b, K);
#endif
PL_ROLE(load, gemm_load_role, 1);
PL_CLAIM_EX(pl::claim("pipelink_gemm_warp_load", "warp")
    .extent(1).align(1).at(8).roles({"load"}));
GEMM_OPERAND_CLAIMS();
GEMM_OPERANDS(pl::produce, "load");
