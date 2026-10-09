#pragma once
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/half.h>
#include <cutlass/arch/barrier.h>

struct alignas(128) pl_gemm_tma_params {
  CUtensorMap a;
  CUtensorMap b;
};

namespace pl_gemm_sm90 {
#ifndef PL_SM90_STAGES
#define PL_SM90_STAGES 3
#endif
inline constexpr int stages = PL_SM90_STAGES;
static_assert(stages >= 2);
inline constexpr int rows = 128, columns = 128, threads = 128;
struct Configuration {
  using Element = cutlass::half_t;
  using MmaOp = decltype(cute::SM90::GMMA::ss_op_selector<
      Element, Element, float, cute::Shape<cute::_128, cute::_128, cute::_64>>());
  using TiledMma = decltype(cute::make_tiled_mma(MmaOp{}));
  using LogicalShape = cute::Shape<cute::_128, cute::_128>;
  using OperandLayout = decltype(cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<Element>{}, cute::Shape<cute::_128, cute::_64>{}));
  struct alignas(1024) SharedStorage {
    alignas(1024) Element a[stages][128 * 64];
    alignas(1024) Element b[stages][128 * 64];
    uint64_t full[stages];
    uint64_t empty[stages];
  };
  CUTE_HOST_DEVICE static constexpr auto partitioned_coordinates(int thread) {
    return TiledMma{}.get_slice(thread).partition_C(cute::make_identity_tensor(LogicalShape{}));
  }
};

// XOR a permutation of five row bits into the bank index: the WGMMA
// lane's row bits occupy banks 0,3,4 and its column bits occupy banks 1,2.
// Row-per-thread scalar reads also visit all 32 banks.
CUTE_HOST_DEVICE constexpr int export_offset(int m, int n) {
  return m * 128 + (n ^ (((m & 3) << 3) | ((m & 4) >> 2) | ((m & 24) >> 2)));
}

#if defined(__CUDACC__)
CUTE_DEVICE void rendezvous(int barrier) {
  asm volatile("bar.sync %0, 128;" : : "r"(barrier) : "memory");
}

CUTE_DEVICE void load(Configuration::SharedStorage& shared, const CUtensorMap* a,
                      const CUtensorMap* b, int tile) {
  const int slot = tile % stages;
  auto* full = &shared.full[slot];
  cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(full, 32768);
  cute::SM90_TMA_LOAD_2D::copy(a, full, uint64_t(0), shared.a[slot], tile * 64, int(blockIdx.x) * 128);
  cute::SM90_TMA_LOAD_2D::copy(b, full, uint64_t(0), shared.b[slot], tile * 64, int(blockIdx.y) * 128);
}

template<class Accumulator>
CUTE_DEVICE void mainloop(const CUtensorMap* a, const CUtensorMap* b, int K,
                          Configuration::SharedStorage& shared, int barrier,
                          Accumulator& accumulator) {
  using namespace cute;
  using Full = cutlass::arch::ClusterTransactionBarrier;
  using Empty = cutlass::arch::ClusterBarrier;
  if (threadIdx.x == 0) {
    CUTE_UNROLL
    for (int slot = 0; slot < stages; ++slot) {
      Full::init(&shared.full[slot], 1);
      Empty::init(&shared.empty[slot], threads);
    }
    cutlass::arch::fence_barrier_init();
  }
  rendezvous(barrier);
  const int tiles = K / 64;
  if (threadIdx.x == 0) {
    CUTE_UNROLL
    for (int tile = 0; tile < stages; ++tile) {
      if (tile < tiles) load(shared, a, b, tile);
    }
  }
  auto mma = Configuration::TiledMma{};
  auto thread_mma = mma.get_slice(threadIdx.x);
  clear(accumulator);
  warpgroup_fence_operand(accumulator);
  CUTE_NO_UNROLL
  for (int tile = 0; tile < tiles; ++tile) {
    const int slot = tile % stages;
    Full::wait(&shared.full[slot], (tile / stages) & 1);
    auto sa = make_tensor(make_smem_ptr(shared.a[slot]), Configuration::OperandLayout{});
    auto sb = make_tensor(make_smem_ptr(shared.b[slot]), Configuration::OperandLayout{});
    auto fa = thread_mma.make_fragment_A(thread_mma.partition_A(sa));
    auto fb = thread_mma.make_fragment_B(thread_mma.partition_B(sb));
    warpgroup_arrive();
    gemm(mma, fa, fb, accumulator);
    warpgroup_commit_batch();
    // Keep the current batch in flight. Only the preceding batch's operands
    // can be recycled; all 128 arrivals certify its retirement to thread zero.
    warpgroup_wait<1>();
    if (tile > 0) {
      const int retired = tile - 1;
      auto* empty = &shared.empty[retired % stages];
      Empty::arrive(empty);
      if (threadIdx.x == 0 && retired + stages < tiles) {
        Empty::wait(empty, (retired / stages) & 1);
        load(shared, a, b, retired + stages);
      }
    }
  }
  warpgroup_wait<0>();
  warpgroup_fence_operand(accumulator);
  // Callers may reuse operands as epilogue storage after the collective drain.
  rendezvous(barrier);
  // No thread may still wait/arrive when an mbarrier is invalidated.
  if (threadIdx.x == 0) {
    CUTE_UNROLL
    for (int slot = 0; slot < stages; ++slot) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                   :: "r"(cast_smem_ptr_to_uint(&shared.full[slot])) : "memory");
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                   :: "r"(cast_smem_ptr_to_uint(&shared.empty[slot])) : "memory");
    }
  }
}
#endif
}  // namespace pl_gemm_sm90
