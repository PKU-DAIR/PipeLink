#include "channels.cuh"
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/bfloat16.h>
#if defined(GEMM_HOST_ABI) && GEMM_HOST_ABI
#include "../host/owner/host_annotations.cuh"
#endif

#if defined(__CUDA_ARCH__) && !defined(CUTE_ARCH_MMA_SM90A_ENABLED)
#error "hopper-gemm-split requires native Hopper WGMMA; compile for sm_90a"
#endif

namespace hopper_gemm::mma_owner {
using Element = cutlass::bfloat16_t;
using Mma = decltype(cute::make_tiled_mma(cute::SM90::GMMA::ss_op_selector<
    Element, Element, float, cute::Shape<cute::_64, cute::_128, cute::_64>>()));
using ALayout = decltype(cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<Element>{},
                                           cute::Shape<cute::_64, cute::_64>{}));
using BLayout = decltype(cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<Element>{},
                                           cute::Shape<cute::_128, cute::_64>{}));
}

extern "C" __device__ void gemm_mma_role(const void* raw_params, unsigned) {
  using namespace cute;
  using namespace hopper_gemm;
  using namespace hopper_gemm::mma_owner;
#if defined(GEMM_HOST_ABI) && GEMM_HOST_ABI
  auto const& view = *static_cast<const plk_role_view_gemm_mma_mma*>(raw_params);
  auto const& params = static_cast<const GemmOwnerResidue*>(view.residue)->mma;
#else
  auto const& params = *static_cast<const GemmMmaParams*>(raw_params);
#endif
  extern __shared__ __align__(1024) unsigned char gemm_smem[];
  const int thread = int(threadIdx.x) - 32 * pipelink_gemm_warp_mma();
  const int group = thread / 128;
  auto mma = Mma{};
  auto lane_mma = mma.get_slice(thread % 128);
  auto coordinates = lane_mma.partition_C(make_identity_tensor(Shape<_64, _128>{}));
  auto accum = lane_mma.make_fragment_C(coordinates);
  clear(accum);
  const int depth = pipelink_gemm_depth();

  // Each warpgroup owns 64 output rows. Both read B; their A halves are disjoint.
  // A slot remains live until every reader has retired its WGMMA operations.
  for (int k = 0; k < params.K / TileK; ++k) {
    const int slot = k % depth;
    auto* full = barrier(gemm_smem, pipelink_gemm_full(), slot);
    auto* empty = barrier(gemm_smem, pipelink_gemm_empty(), slot);
    cutlass::arch::ClusterBarrier::wait(full, (k / depth) & 1);
    auto* operands = gemm_smem + pipelink_gemm_operands() + slot * SlotBytes;
    auto a = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(
        operands + group * 64 * TileK * sizeof(Element))), ALayout{});
    auto b = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(operands + OperandBytes)), BLayout{});
    auto fa = lane_mma.make_fragment_A(lane_mma.partition_A(a));
    auto fb = lane_mma.make_fragment_B(lane_mma.partition_B(b));
    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    gemm(mma, fa, fb, accum);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);
    cutlass::arch::ClusterBarrier::arrive(empty);
  }

  // Like the sm90 attention core, materialize once into a declared smem channel.
  // Unlike attention, keep f32: no cast or arithmetic is hidden in the seam.
  cutlass::arch::ClusterBarrier::wait(barrier(gemm_smem, pipelink_gemm_drained()), 1);
  auto* output = reinterpret_cast<float*>(gemm_smem + pipelink_gemm_output());
  CUTE_UNROLL
  for (int i = 0; i < size(accum); ++i) {
    auto coord = coordinates(i);
    const int row = group * 64 + int(get<0>(coord));
    const int col = int(get<1>(coord));
    output[row * TileN + col] = accum(i);
  }
  cutlass::arch::ClusterBarrier::arrive(barrier(gemm_smem, pipelink_gemm_ready()));
}

PL_STAGE("gemm.mma", "1.0.0");
#if defined(GEMM_HOST_ABI) && GEMM_HOST_ABI
PL_NO_PARAMS();
#else
PL_PARAMS(GemmMmaParams, K);
#endif
PL_ROLE(mma, gemm_mma_role, 8);
PL_CLAIM_EX(pl::claim("pipelink_gemm_warp_mma", "warp")
    .extent(8).align(4).at(0).roles({"mma"}));
GEMM_OPERAND_CLAIMS();
GEMM_OPERANDS(pl::consume, "mma");
GEMM_OUTPUT_CLAIMS();
GEMM_OUTPUT(pl::produce, "mma");
