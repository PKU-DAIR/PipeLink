#pragma once
#include <cuda_fp16.h>

// All four CTA warps participate; the caller synchronizes before reading scale.
__device__ inline void pl_row_scale(const __half* a, int K, int m0, float eps,
                                    float* scale) {
  const int lane = int(threadIdx.x) & 31;
  const int warp = int(threadIdx.x) >> 5;
  for (int r = warp; r < 128; r += 4) {
    const __half* row = a + (m0 + r) * K;
    float sum = 0.0f;
    for (int k = lane * 8; k < K; k += 32 * 8) {
      const uint4 packed = *reinterpret_cast<const uint4*>(row + k);
      const __half* values = reinterpret_cast<const __half*>(&packed);
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float value = __half2float(values[i]);
        sum += value * value;
      }
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, delta);
    if (lane == 0) scale[r] = rsqrtf(sum / float(K) + eps);
  }
}
