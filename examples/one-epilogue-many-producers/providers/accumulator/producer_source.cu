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
// CUTLASS-derived M128N128 owner, with a three-slot TMA/UMMA pipeline.
#include "../abi/pl_tensor_source_abi.cuh"
#include "gemm_mainloop_sm100.cuh"
#include <pipelink/stage.hpp>
PL_TYPE(pl_gemm_tile_args, const void* a; const void* b; int32_t M; int32_t N; int32_t K;);

#if PL_TILE_N != 128 || PL_MMA_M != 128
#error "The teaching owner supports only M128N128 fp16"
#endif
#if defined(PL_EXPORT_TMEM)
#define PL_SOURCE_NAMESPACE pl_tensor_owner_m128n128
#define PL_PARAMS_ACCESSOR pipelink_params_tensor_producer_m128n128
#elif defined(PL_EXPORT_SMEM)
#define PL_SOURCE_NAMESPACE pl_tensor_owner_smem128n128
#define PL_PARAMS_ACCESSOR pipelink_params_tensor_producer_smem128n128
#else
#error "Select PL_EXPORT_TMEM or PL_EXPORT_SMEM"
#endif

#if defined(__CUDACC__)
extern "C" __device__ const void* PL_PARAMS_ACCESSOR();
extern "C" __device__ pipelink_tensor_source plx_produce(const pl_gemm_tile_args* args, void* frame) {
  using namespace pl_gemm_sm100;
  (void)frame;
  extern __shared__ __align__(1024) unsigned char plx_tensor_owner_smem[];
  auto* base = reinterpret_cast<uint32_t*>(plx_tensor_owner_smem + pipelink_tensor_tmem_base_cell());
  auto& shared = *reinterpret_cast<SharedStorage*>(plx_tensor_owner_smem + pipelink_tensor_producer_smem());
  if (threadIdx.x < 32) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(128, base);
    allocator.release_allocation_lock();
  }
  const int barrier = pipelink_tensor_producer_barrier();
  rendezvous(barrier);
  const auto* params = static_cast<const pl_gemm_tma_params*>(PL_PARAMS_ACCESSOR());
  mainloop(&params->a, &params->b, args->K, shared, *base, barrier);
#ifdef PL_EXPORT_TMEM
  return pl::source_of_tmem(*base);
#else
  // All operand reads are complete. Reuse exactly the first 64 KiB of the
  // producer's own reservation; barrier cells live beyond the operand ring.
  auto* tile = reinterpret_cast<float*>(shared.operands);
  export_smem(*base, tile);
  __threadfence_block();
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  return pl::source_of(tile);
#endif
}
extern "C" __device__ void plx_release(const pl_gemm_tile_args* args) {
  using namespace pl_gemm_sm100;
  (void)args;
  extern __shared__ __align__(1024) unsigned char plx_tensor_owner_smem[];
  auto* base = reinterpret_cast<uint32_t*>(plx_tensor_owner_smem + pipelink_tensor_tmem_base_cell());
  if (threadIdx.x < 32) cute::TMEM::Allocator1Sm{}.free(*base, 128);
  rendezvous(pipelink_tensor_producer_barrier());
}
#endif

#ifdef PIPELINK_EMIT
namespace PL_SOURCE_NAMESPACE {
using namespace pl_gemm_sm100;
#ifdef PL_EXPORT_TMEM
PL_STAGE("tensor.producer.m128n128", "1.0.0");
#else
PL_STAGE("tensor.producer.smem128n128", "1.0.0");
#endif
PL_PARAMS(pl_gemm_tma_params, a, b);
PL_EXPORT(produce, plx_produce);
PL_CALL(tensor_produce, plx_produce, pl::consume, "produce");
PL_EXPORT(release, plx_release);
PL_CALL(tensor_release, plx_release, pl::consume, "release");
#ifdef PL_EXPORT_TMEM
PL_CLAIM_SMEM("pipelink_tensor_producer_smem", SharedStorage,
              pl::during{"accumulator", "tensor_work", "tensor_ready"});
#else
PL_CLAIM_SMEM("pipelink_tensor_producer_smem", SharedStorage, "kernel");
#endif
PL_CLAIM("pipelink_tensor_tmem_columns", "tmem", 128, 128, "kernel");
PL_CLAIM("pipelink_tensor_tmem_base_cell", "smem", sizeof(uint32_t), alignof(uint32_t), "kernel");
PL_CLAIM("pipelink_tensor_producer_barrier", "named_barrier", 1, 1, "kernel");
#ifdef PL_EXPORT_TMEM
PL_BINDING("tmem_base", "bits[32]", pl::values{"tmem_base", "bits[32]", {0, 128, 256, 384}},
           "tensor_producer", "column", "tensor_import_site", "tensor_allocated_to_tensor_reclaimed");
#endif
PL_EXPORT_TENSOR(accumulator,
    pl::identity{"accumulator", "tensor_work", "completed_k", "f32", "fp16_dot_fp32_completed",
                 {{"m", 0, 128}, {"n", 0, 128}}},
    pl::placement{
#ifdef PL_EXPORT_TMEM
      "tmem", "dp/column", {"dp", "column"}, [](int m, int n, pl::bound base) {
        return std::array<int, 2>{m, n + int(base)};
#else
      "smem", "byte", {"address"}, [](int m, int n) {
        return std::array<int, 1>{export_address(m, n)};
#endif
      }, pl::coord{"m", 0, 128}, pl::coord{"n", 0, 128}
#ifdef PL_EXPORT_TMEM
       , pl::bind("tmem_base", "bits[32]")
#endif
    },
#ifdef PL_EXPORT_TMEM
    pl::strings{"tmem_packet_receipt"},
#else
    pl::strings{"smem_load_scalar", "smem_load_round"},
#endif
    pl::readiness{"tensor_ready", "import_done"}, pl::fixed_bindings{},
    pl::lends{
#ifndef PL_EXPORT_TMEM
      {"producer_storage", "smem", sizeof(SharedStorage), alignof(SharedStorage), "cta", "relocatable",
       "pipelink_tensor_producer_smem", std::nullopt, "tensor_producer",
       "tensor_work", "tensor_reclaimed", pl::sync{2, 1}},
#endif
      pl::smem("tmem_base_cell", sizeof(uint32_t), alignof(uint32_t),
               "pipelink_tensor_tmem_base_cell", "tensor_producer", "tensor_work", "tensor_reclaimed"),
      {"producer_rendezvous", "barrier", 1, 1, "cta", "relocatable",
       "pipelink_tensor_producer_barrier", std::nullopt, "tensor_producer",
       "tensor_work", "tensor_reclaimed", pl::sync{2, 1}}
#ifdef PL_EXPORT_TMEM
      , {"producer_tmem", "tmem", 128, 128, "cta", "fixed", "tmem_base", std::nullopt,
         "tensor_producer", "tensor_allocated", "tensor_reclaimed"}
#endif
    }, pl::calls{"tensor_produce", "tensor_release"});
} // namespace PL_SOURCE_NAMESPACE
#endif
