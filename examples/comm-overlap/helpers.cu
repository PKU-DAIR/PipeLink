// Caller-side helper kernels for run.py (not stage code): payload fill, rank skew,
// and the separate reduction of the sequential RS baseline. One by-value struct each.
#include <stdint.h>

struct FillArgs {
  uint16_t* out;      // bf16 bits, row-major with leading dimension ld
  uint64_t rows, cols, ld;
  uint32_t seed, reserved;
};

struct SpinArgs {
  uint64_t nanoseconds;
};

struct ReduceArgs {
  const float4* sources[8];  // per-rank partials, already offset to the owner's rows
  float4* out;
  uint64_t vectors;
  uint32_t world, reserved;
};

__device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}

// Uniform(-1, 1) rounded to bf16 (ties to even): products and sums round in f32, so a
// change in reduction order is visible bitwise. Depends only on (seed, row, col).
extern "C" __global__ void comm_fill_bf16(FillArgs a) {
  const uint64_t total = a.rows * a.cols;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < total;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t row = i / a.cols, col = i % a.cols;
    const uint32_t h = mix(a.seed ^ mix(uint32_t(row) * 0x9e3779b9u ^ mix(uint32_t(col) + 0x632be5abu)));
    const uint32_t bits = __float_as_uint(float(h >> 8) * (2.0f / 16777216.0f) - 1.0f);
    a.out[row * a.ld + col] = uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  }
}

extern "C" __global__ void comm_spin(SpinArgs a) {
  uint64_t start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    __nanosleep(1000);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < a.nanoseconds);
}

// Same fixed rank order as the RS last arriver: s0 + s1 + ... + s(W-1).
extern "C" __global__ void comm_reduce(ReduceArgs a) {
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < a.vectors;
       i += uint64_t(gridDim.x) * blockDim.x) {
    float4 sum = a.sources[0][i];
    for (uint32_t s = 1; s < a.world; ++s) {
      const float4 v = a.sources[s][i];
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    a.out[i] = sum;
  }
}
