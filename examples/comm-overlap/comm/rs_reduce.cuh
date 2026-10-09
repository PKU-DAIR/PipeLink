#pragma once
#include "comm.h"

namespace comm_overlap::rs {
// Owner-private implementation of the follow-on reduction kernel.
// All loads and stores are local; each output adds source ranks 0,1,...,W-1.
template <int World, int Threads>
__device__ __forceinline__ void reduce_tile(const CommRsResidue& p, const float4* slots,
                                            size_t source_stride, int tile, unsigned tid) {
  const int col_blocks = p.N / 128;
  const size_t origin = size_t(tile / col_blocks) * 128 * (p.N / 4) + (tile % col_blocks) * 32;
  auto const* input = slots + origin;
  auto* output = reinterpret_cast<float4*>(p.output) + origin;
  for (int i = tid; i < 4096; i += Threads) {
    const size_t offset = size_t(i / 32) * (p.N / 4) + i % 32;
    float4 values[World];
#pragma unroll
    for (int s = 0; s < World; ++s)
      if (s < p.world) values[s] = input[size_t(s) * source_stride + offset];
    float4 sum = values[0];
#pragma unroll
    for (int s = 1; s < World; ++s)
      if (s < p.world) {
        sum.x += values[s].x; sum.y += values[s].y;
        sum.z += values[s].z; sum.w += values[s].w;
      }
    output[offset] = sum;
  }
}

template <int Threads>
__device__ __forceinline__ void reduce(const CommRsResidue& p, const float4* slots,
                                       size_t source_stride, int tile, unsigned tid) {
  switch (p.world) {
    case 2: reduce_tile<2, Threads>(p, slots, source_stride, tile, tid); break;
    case 3: reduce_tile<3, Threads>(p, slots, source_stride, tile, tid); break;
    case 4: reduce_tile<4, Threads>(p, slots, source_stride, tile, tid); break;
    default: reduce_tile<COMM_MAX_WORLD, Threads>(p, slots, source_stride, tile, tid);
  }
}
}  // namespace comm_overlap::rs
