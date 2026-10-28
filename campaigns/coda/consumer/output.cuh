#pragma once
#include <cuda_bf16.h>
#include "geometry.cuh"

namespace coda_epilogue {
struct alignas(128) OutputTile { uint4 vectors[128 * 16]; };

__device__ __forceinline__ float swiglu(float gate, float up) {
  return __fdividef(gate, 1.0f + __expf(-gate)) * up;
}

__device__ __forceinline__ void stage(OutputTile& tile, int t, int chunk, const float* acc) {
  uint4 packed;
  auto* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    pairs[i] = __floats2bfloat162_rn(swiglu(acc[4*i], acc[4*i+1]),
                                   swiglu(acc[4*i+2], acc[4*i+3]));
  const int m = row(t), vector = 2 * chunk + half_row(t);
  tile.vectors[m * 16 + (vector ^ (m & 7))] = packed;
}

// Caller synchronizes the completed tile before all data threads drain it.
__device__ __forceinline__ void store(const OutputTile& tile, void* output,
                                      int m0, int n0, int N, int t) {
  auto* out = static_cast<__nv_bfloat16*>(output);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int linear = t + i * data_threads;
    const int m = linear / 16, vector = linear % 16;
    *reinterpret_cast<uint4*>(out + (int64_t(m0) + m) * (N / 2) + n0 / 2 + vector * 8) =
        tile.vectors[m * 16 + (vector ^ (m & 7))];
  }
}
}
