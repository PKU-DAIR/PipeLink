#pragma once
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/half.h>
#include <cutlass/arch/barrier.h>

struct alignas(128) pl_gemm_tma_params {
  CUtensorMap a;
  CUtensorMap b;
};

// Consumer Blackwell has TMA and mbarriers but neither WGMMA nor TMEM: the same
// TMA ring as the Hopper mainloop feeds warp-level mma.sync through ldmatrix.
namespace pl_gemm_sm120 {
#ifndef PL_SM120_STAGES
#define PL_SM120_STAGES 3
#endif
inline constexpr int stages = PL_SM120_STAGES;
static_assert(stages >= 2);
inline constexpr int rows = 128, columns = 128, threads = 128;
struct Configuration {
  using Element = cutlass::half_t;
  // Four warps tile 128x128 as 2x2 warps of 64x64; each thread owns 128 f32 values.
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::SM80_16x8x16_F32F16F16F32_TN{}, cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>{},
      cute::Tile<cute::_128, cute::_128, cute::_16>{}));
  using LogicalShape = cute::Shape<cute::_128, cute::_128>;
  // The TMA 128-byte swizzle of a K-major fp16 box row (64 elements): 16-byte chunks XOR row % 8.
  using OperandLayout = decltype(cute::tile_to_shape(
      cute::composition(cute::Swizzle<3, 3, 3>{},
                        cute::Layout<cute::Shape<cute::_8, cute::_64>, cute::Stride<cute::_64, cute::_1>>{}),
      cute::Shape<cute::_128, cute::_64>{}));
  using CopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
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

// Same pair-preserving XOR as the Hopper provider: five row bits enter the bank
// index, so a half-warp's 64-bit fragment stores and row-per-thread scalar reads
// both visit distinct banks.
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
  auto tiled_copy_a = make_tiled_copy_A(Configuration::CopyAtom{}, mma);
  auto tiled_copy_b = make_tiled_copy_B(Configuration::CopyAtom{}, mma);
  auto copy_a = tiled_copy_a.get_slice(threadIdx.x);
  auto copy_b = tiled_copy_b.get_slice(threadIdx.x);
  // Register fragments are shaped once from slot 0; each tile re-partitions its own slot.
  auto sa0 = make_tensor(make_smem_ptr(shared.a[0]), Configuration::OperandLayout{});
  auto sb0 = make_tensor(make_smem_ptr(shared.b[0]), Configuration::OperandLayout{});
  auto fa = thread_mma.partition_fragment_A(sa0);
  auto fb = thread_mma.partition_fragment_B(sb0);
  auto fa_view = copy_a.retile_D(fa);
  auto fb_view = copy_b.retile_D(fb);
  clear(accumulator);
  CUTE_NO_UNROLL
  for (int tile = 0; tile < tiles; ++tile) {
    const int slot = tile % stages;
    Full::wait(&shared.full[slot], (tile / stages) & 1);
    auto sa = copy_a.partition_S(make_tensor(make_smem_ptr(shared.a[slot]), Configuration::OperandLayout{}));
    auto sb = copy_b.partition_S(make_tensor(make_smem_ptr(shared.b[slot]), Configuration::OperandLayout{}));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(fa); ++k) {
      copy(tiled_copy_a, sa(_, _, k), fa_view(_, _, k));
      copy(tiled_copy_b, sb(_, _, k), fb_view(_, _, k));
      gemm(mma, fa(_, _, k), fb(_, _, k), accumulator);
    }
    // mma.sync consumed its operands from registers: every thread's ldmatrix of
    // this slot has completed, so all 128 arrivals certify the slot free.
    auto* empty = &shared.empty[slot];
    Empty::arrive(empty);
    if (threadIdx.x == 0 && tile + stages < tiles) {
      Empty::wait(empty, (tile / stages) & 1);
      load(shared, a, b, tile + stages);
    }
  }
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
}  // namespace pl_gemm_sm120
