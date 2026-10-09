#pragma once
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>

struct alignas(128) pl_gemm_tma_params { CUtensorMap a; CUtensorMap b; };

namespace pl_gemm_sm100 {
#ifndef PL_GEMM_STAGES
#define PL_GEMM_STAGES 3
#endif
inline constexpr int stages = PL_GEMM_STAGES;
inline constexpr int rows = 128, columns = 128, threads = 128;
using Element = cutlass::half_t;
using Mma = decltype(cute::make_tiled_mma(cute::SM100_MMA_F16BF16_SS<
    Element, Element, float, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>{}));
using Partition = decltype(cute::partition_shape_A(Mma{}, cute::Shape<cute::_128, cute::_64>{}));
using OperandLayout = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<Element>{}, Partition{}));
struct SharedStorage {
  // The first two stages become the f32 export tile only after the final MMA.
  alignas(1024) Element operands[stages][2][128 * 64];
  alignas(16) uint64_t full[stages];
  alignas(16) uint64_t empty[stages];
  alignas(16) uint64_t accumulator_full;
};
static_assert(stages >= 2);
CUTE_HOST_DEVICE constexpr auto coordinates() {
  return Mma{}.get_slice(cute::_0{}).partition_C(
      cute::make_identity_tensor(cute::Shape<cute::_128, cute::_128>{}));
}
// Each 16-byte chunk remains contiguous; low-word permutation permits both
// vector stores across eight rows and scalar reads across 32 rows without conflicts.
CUTE_HOST_DEVICE constexpr int export_address(int row, int column) {
  return (row * 128 + (column ^ ((row & 7) * 4) ^ ((row >> 3) & 3))) * 4;
}
#if defined(__CUDACC__)
CUTE_DEVICE void rendezvous(int barrier) {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  asm volatile("bar.sync %0, 128;" :: "r"(barrier) : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
CUTE_DEVICE void mainloop(const CUtensorMap* a, const CUtensorMap* b, int K,
                          SharedStorage& shared, uint32_t base, int barrier) {
  using namespace cute;
  const int warp = int(threadIdx.x) / 32;
  const bool elected = elect_one_sync();
  if (warp == 0 && elected) {
    CUTE_UNROLL
    for (int s = 0; s < stages; ++s) {
      initialize_barrier(shared.full[s], 1);
      initialize_barrier(shared.empty[s], 1);
    }
    initialize_barrier(shared.accumulator_full, 1);
    cutlass::arch::fence_barrier_init();
  }
  rendezvous(barrier);
  if (warp == 0 && elected) {
    for (int k = 0; k < K / 64; ++k) {
      int s = k % stages;
      wait_barrier(shared.empty[s], ((k / stages) & 1) ^ 1);
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&shared.full[s], 32768);
      SM90_TMA_LOAD_2D::copy(a, &shared.full[s], uint64_t(0), shared.operands[s][0],
                            k * 64, int(blockIdx.x) * 128);
      SM90_TMA_LOAD_2D::copy(b, &shared.full[s], uint64_t(0), shared.operands[s][1],
                            k * 64, int(blockIdx.y) * 128);
    }
  }
  // CuTe's MMA and commit elect internally: keep their warp converged.
  if (warp == 1) {
    auto mma = Mma{};
    auto cta = mma.get_slice(_0{});
    auto acc = cta.make_fragment_C(coordinates());
    acc.data() = base;
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    for (int k = 0; k < K / 64; ++k) {
      int s = k % stages;
      wait_barrier(shared.full[s], (k / stages) & 1);
      auto sa = make_tensor(make_smem_ptr(shared.operands[s][0]), OperandLayout{});
      auto sb = make_tensor(make_smem_ptr(shared.operands[s][1]), OperandLayout{});
      auto fa = cta.make_fragment_A(sa);
      auto fb = cta.make_fragment_B(sb);
      CUTE_UNROLL
      for (int q = 0; q < size<2>(fa); ++q) {
        gemm(mma, fa(_, _, q), fb(_, _, q), acc);
        mma.accumulate_ = UMMA::ScaleOut::One;
      }
      // Commit follows every read of this slot, not merely MMA issue.
      cutlass::arch::umma_arrive(&shared.empty[s]);
    }
    cutlass::arch::umma_arrive(&shared.accumulator_full);
  }
  wait_barrier(shared.accumulator_full, 0);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  // Join every final wait before the elected thread invalidates the barriers.
  rendezvous(barrier);
  if (warp == 0 && elected) {
    CUTE_UNROLL
    for (int s = 0; s < stages; ++s) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                   :: "r"(cast_smem_ptr_to_uint(&shared.full[s])) : "memory");
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                   :: "r"(cast_smem_ptr_to_uint(&shared.empty[s])) : "memory");
    }
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                 :: "r"(cast_smem_ptr_to_uint(&shared.accumulator_full)) : "memory");
  }
}
CUTE_DEVICE void load16(uint32_t base, int first, float* out) {
  using Receipt = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x;
  uint32_t bits[16];
  const uint32_t address = base + (int(threadIdx.x) / 32 * 32 << 16) + first;
  Receipt::copy(address, bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7],
                bits[8], bits[9], bits[10], bits[11], bits[12], bits[13], bits[14], bits[15]);
  cutlass::arch::fence_view_async_tmem_load();
  CUTE_UNROLL
  for (int i = 0; i < 16; ++i) out[i] = __uint_as_float(bits[i]);
}
CUTE_DEVICE void export_smem(uint32_t base, float* out) {
  const int row = int(threadIdx.x), low = (row >> 3) & 3;
  CUTE_UNROLL
  for (int first = 0; first < 128; first += 16) {
    float v[16];
    load16(base, first, v);
    CUTE_UNROLL
    for (int i = 0; i < 16; i += 4) {
      auto* dst = reinterpret_cast<float4*>(out + export_address(row, first + i) / 4 - low);
      *dst = make_float4(v[i + (0 ^ low)], v[i + (1 ^ low)],
                        v[i + (2 ^ low)], v[i + (3 ^ low)]);
    }
  }
}
#endif
} // namespace pl_gemm_sm100
