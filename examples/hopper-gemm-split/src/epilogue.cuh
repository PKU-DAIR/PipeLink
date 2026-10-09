#pragma once
#include "channels.cuh"

namespace hopper_gemm::epilogue_owner {
template <int Operation>
__device__ __forceinline__ void run(const void* raw_params) {
  auto const& params = *static_cast<const GemmEpilogueParams*>(raw_params);
  extern __shared__ __align__(1024) unsigned char gemm_smem[];
  cutlass::arch::ClusterBarrier::wait(barrier(gemm_smem, pipelink_gemm_ready()), 0);
  auto const* accum = reinterpret_cast<const float*>(gemm_smem + pipelink_gemm_output());
  // Consecutive lanes own consecutive columns: coalesced scalar stores, without
  // exposing CuTe's per-thread register-fragment layout to the epilogue owner.
  for (int i = int(threadIdx.x) & 31; i < TileM * TileN; i += 32) {
    const int row = int(blockIdx.x) * TileM + i / TileN;
    const int col = int(blockIdx.y) * TileN + i % TileN;
    float value = accum[i];
    if constexpr (Operation == 1) value = fmaxf(value + params.bias[col], 0.0f);
    if constexpr (Operation == 2) value *= params.scale[row];
    params.output[size_t(row) * params.N + col] = value;
  }
  cutlass::arch::ClusterBarrier::arrive(barrier(gemm_smem, pipelink_gemm_drained()));
}
}

#define GEMM_EPILOGUE_CONTRACT(name, function) \
  PL_STAGE("gemm.epilogue." name, "1.0.0"); \
  PL_PARAMS(GemmEpilogueParams, output, bias, scale, M, N); \
  PL_ROLE(epilogue, function, 1); \
  PL_CLAIM_EX(pl::claim("pipelink_gemm_warp_epilogue", "warp") \
      .extent(1).align(1).at(9).roles({"epilogue"})); \
  GEMM_OUTPUT_CLAIMS(); \
  GEMM_OUTPUT(pl::consume, "epilogue")
