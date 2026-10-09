/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Adapted from CUTLASS examples/cute/tutorial/blackwell/01_mma_sm100.cu.
// OWNER-PRIVATE: compile exactly one MMA-M source per producer package.
// build.py enumerates this exact Configuration before freezing the payload.
// Integration receives the object and public tensor facts, never this source.
#include "../abi/pl_tensor_abi.cuh"
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cutlass/half.h>
#include <cutlass/arch/barrier.h>
#include <pipelink/stage.hpp>
PL_TYPE(pl_gemm_tile_args, const void* a; const void* b; int32_t M; int32_t N; int32_t K;);

#if defined(__CUDA_ARCH__) && \
    (!defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED) || \
     !defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED) || \
     !defined(CUTE_ARCH_TMA_SM90_ENABLED) || \
     !defined(CUTLASS_ARCH_TCGEN_ENABLED))
#error "Real MMA producer requires native FP16 TCGEN05, TMEM and mbarrier frontend support"
#endif

namespace pl_tensor_owner_m128 {

template <int MmaM>
struct Configuration {
  static_assert(MmaM == 64 || MmaM == 128);
  using Element = cutlass::half_t;
  using MmaOp = cute::SM100_MMA_F16BF16_SS<
      Element, Element, float, MmaM, 32, cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using TiledMma = decltype(cute::make_tiled_mma(MmaOp{}));
  // In particular, MmaM=64 repeats twice in M; it does NOT shrink the CTA tile.
  using CtaTile = cute::Shape<cute::_128, cute::_32, cute::_64>;
  using LogicalShape = cute::Shape<cute::_128, cute::_32>;
  using APartition = decltype(cute::partition_shape_A(
      TiledMma{}, cute::Shape<cute::_128, cute::_64>{}));
  using BPartition = decltype(cute::partition_shape_B(
      TiledMma{}, cute::Shape<cute::_32, cute::_64>{}));
  using ALayout = decltype(cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<Element>{}, APartition{}));
  using BLayout = decltype(cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<Element>{}, BPartition{}));

  struct SharedStorage {
    alignas(128) cute::ArrayEngine<Element, cute::cosize_v<ALayout>> a;
    alignas(128) cute::ArrayEngine<Element, cute::cosize_v<BLayout>> b;
    alignas(16) cute::uint64_t mma_barrier;

    CUTE_DEVICE constexpr auto tensor_a() {
      return cute::make_tensor(cute::make_smem_ptr(a.begin()), ALayout{});
    }
    CUTE_DEVICE constexpr auto tensor_b() {
      return cute::make_tensor(cute::make_smem_ptr(b.begin()), BLayout{});
    }
  };
  static_assert(alignof(SharedStorage) == 128);

  // The device producer and the host enumerator use this exact partition.
  // No synthetic physical map or consumer-selected accumulator layout exists.
  CUTE_HOST_DEVICE static constexpr auto partitioned_coordinates() {
    auto mma = TiledMma{};
    auto cta_mma = mma.get_slice(cute::_0{});
    return cta_mma.partition_C(cute::make_identity_tensor(LogicalShape{}));
  }
};

#if defined(__CUDACC__)
template <int MmaM>
CUTE_DEVICE void produce(const pl_gemm_tile_args* params, void* producer_scratch,
                         uint32_t tmem_base) {
  using namespace cute;
  using Config = Configuration<MmaM>;
  using Element = typename Config::Element;
  auto& shared = *reinterpret_cast<typename Config::SharedStorage*>(producer_scratch);
  auto mma = typename Config::TiledMma{};
  auto cta_mma = mma.get_slice(_0{});

  // Runtime dense A[M,K] and B[N,K], both row-major: X = A * B^T.
  auto a = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(params->a)),
                       make_layout(make_shape(params->M, params->K),
                                   make_stride(params->K, _1{})));
  auto b = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(params->b)),
                       make_layout(make_shape(params->N, params->K),
                                   make_stride(params->K, _1{})));
  auto coordinate = make_coord(blockIdx.x, blockIdx.y, _);
  auto ga = local_tile(a, typename Config::CtaTile{}, coordinate, Step<_1, X, _1>{});
  auto gb = local_tile(b, typename Config::CtaTile{}, coordinate, Step<X, _1, _1>{});
  auto partition_a = cta_mma.partition_A(ga);
  auto partition_b = cta_mma.partition_B(gb);
  auto sa = shared.tensor_a();
  auto sb = shared.tensor_b();
  // Native descriptor construction owns swizzles, K steps and both M64 repeats.
  auto fragment_a = cta_mma.make_fragment_A(sa);
  auto fragment_b = cta_mma.make_fragment_B(sb);
  auto coordinates = Config::partitioned_coordinates();
  auto accumulator = cta_mma.make_fragment_C(coordinates);
  accumulator.data() = tmem_base;

  const bool issue_warp = threadIdx.x < 32;
  if (issue_warp && cute::elect_one_sync()) {
    cute::initialize_barrier(shared.mma_barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  int phase = 0;
  mma.accumulate_ = UMMA::ScaleOut::Zero;

  // Each runtime K64 stage is consumed once, with no overlapped operand reuse.
  for (int k_tile = 0; k_tile < size<3>(partition_a); ++k_tile) {
    cooperative_copy<128>(threadIdx.x, partition_a(_, _, _, k_tile), sa);
    cooperative_copy<128>(threadIdx.x, partition_b(_, _, _, k_tile), sb);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    if (issue_warp) {
      CUTE_UNROLL
      for (int k_block = 0; k_block < size<2>(fragment_a); ++k_block) {
        gemm(mma, fragment_a(_, _, k_block), fragment_b(_, _, k_block), accumulator);
        // Both M64 row atoms in the FIRST gemm must see Zero before this change.
        mma.accumulate_ = UMMA::ScaleOut::One;
      }
      // This commits actual tcgen05.mma work to the producer-owned mbarrier.
      cutlass::arch::umma_arrive(&shared.mma_barrier);
    }
    cute::wait_barrier(shared.mma_barrier, phase);
    phase ^= 1;
    __syncthreads();
  }
  // All MMA work is complete on return. The frozen caller supplies tensor_ready
  // with its fenced CTA rendezvous; operand storage and the barrier stay reserved.
}
#endif

}  // namespace pl_tensor_owner_m128

#if defined(__CUDACC__)
extern "C" __device__ pipelink_tensor_source plx_produce(const pl_gemm_tile_args* args, void* frame) {
  (void)frame;
  extern __shared__ __align__(128) unsigned char plx_tensor_owner_smem[];
  auto* tmem_base_cell = reinterpret_cast<uint32_t*>(
      plx_tensor_owner_smem + pipelink_tensor_tmem_base_cell());
  void* producer_scratch =
      plx_tensor_owner_smem + pipelink_tensor_producer_smem();

  if (threadIdx.x < 32) {
    // All lanes of warp0 participate; relinquish is not deallocation.
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(kPlTensorColumns, tmem_base_cell);
    allocator.release_allocation_lock();
  }
  __syncthreads();  // tensor_allocated: publish the base before any MMA use.
  pl_tensor_owner_m128::produce<128>(args, producer_scratch, *tmem_base_cell);
  // The generated ready event publishes this handle before importing.
  return pl::source_of_tmem(*tmem_base_cell);
}

extern "C" __device__ void plx_release(const pl_gemm_tile_args* args) {
  // Same borrowed params invocation; reclaim needs only the published base.
  (void)args;
  extern __shared__ __align__(128) unsigned char plx_tensor_owner_smem[];
  const auto* tmem_base_cell = reinterpret_cast<const uint32_t*>(
      plx_tensor_owner_smem + pipelink_tensor_tmem_base_cell());
  // Precondition: all 128 callers passed import_done, including TMEM load wait
  // and before/after-thread-sync fences. No outstanding reader can survive it.
  if (threadIdx.x < 32) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.free(*tmem_base_cell, kPlTensorColumns);
  }
  __syncthreads();  // tensor_reclaimed: all callers observe completed ownership.
}
#endif

#ifdef PIPELINK_EMIT
namespace pl_tensor_owner_m128 {
PL_STAGE("tensor.producer.m128", "1.0.0");
PL_NO_PARAMS();
PL_EXPORT(produce, plx_produce);
PL_EXPORT(release, plx_release);
PL_CALL(tensor_produce, plx_produce, pl::consume, "produce");
PL_CALL(tensor_release, plx_release, pl::consume, "release");
PL_CLAIM_SMEM("pipelink_tensor_producer_smem", Configuration<128>::SharedStorage, "kernel");
PL_CLAIM("pipelink_tensor_tmem_columns", "tmem", kPlTensorColumns, kPlTensorColumns, "kernel");
PL_CLAIM("pipelink_tensor_tmem_base_cell", "smem", sizeof(uint32_t), alignof(uint32_t), "kernel");
// Every 32-column-aligned base that fits this allocation in 512 TMEM columns.
PL_BINDING("tmem_base", "bits[32]", pl::values{"tmem_base", "bits[32]", {0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480}},
           "tensor_producer", "column", "tensor_import_site", "tensor_allocated_to_tensor_reclaimed");
PL_EXPORT_TENSOR(accumulator,
    pl::identity{"accumulator", "tensor_work", "completed_k", "f32", "fp16_dot_fp32_completed",
                 {{"m", 0, kPlTensorRows}, {"n", 0, kPlTensorColumns}}},
    pl::placement{"tmem", "dp/column", {"dp", "column"}, [](int m, int n, pl::bound base) {
      static const auto addresses = [] {
        using namespace cute;
        using Config = Configuration<128>;
        auto coordinates = Config::partitioned_coordinates();
        auto fragment = Config::TiledMma{}.get_slice(_0{}).make_fragment_C(coordinates);
        std::array<int, kPlTensorRows * kPlTensorColumns> out{};
        for (int i = 0; i < int(size(coordinates)); ++i) {
          auto logical = coordinates(i);
          out[int(get<0>(logical)) * kPlTensorColumns + int(get<1>(logical))] = int(fragment.layout()(i));
        }
        return out;
      }();
      const int address = addresses[m * kPlTensorColumns + n] + int(base);
      return std::array<int, 2>{address >> 16, address & 0xffff};
    }, pl::coord{"m", 0, kPlTensorRows}, pl::coord{"n", 0, kPlTensorColumns},
       pl::bind("tmem_base", "bits[32]")},
    pl::strings{"tmem_packet_receipt"}, pl::readiness{"tensor_ready", "import_done"}, pl::fixed_bindings{},
    pl::lends{{"producer_storage", "smem", sizeof(Configuration<128>::SharedStorage),
               alignof(Configuration<128>::SharedStorage), "cta", "relocatable",
               "pipelink_tensor_producer_smem", std::nullopt, "tensor_producer",
               "tensor_work", "tensor_reclaimed", pl::sync{2, 1}},
              {"producer_tmem", "tmem", kPlTensorColumns, kPlTensorColumns, "cta", "fixed",
               "tmem_base", std::nullopt, "tensor_producer", "tensor_allocated", "tensor_reclaimed"},
              pl::smem("tmem_base_cell", sizeof(uint32_t), alignof(uint32_t),
                       "pipelink_tensor_tmem_base_cell", "tensor_producer", "tensor_work", "tensor_reclaimed")},
    pl::calls{"tensor_produce", "tensor_release"});
} // namespace pl_tensor_owner_m128
#endif
