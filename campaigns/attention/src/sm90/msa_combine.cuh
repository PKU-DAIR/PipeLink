#pragma once
#include "msa_params.cuh"

namespace pipelink::attention::sm90 {

// Eight final rows per CTA, sixteen lanes per row, eight bf16 values per lane.
// Partials remain normalized; the inverse map changes only their storage order.
template <bool Short>
__global__ void msa_combine(
    FamilyTypes<true>::Element const* partial_o, float const* partial_m,
    float const* partial_l, int32_t const* inverse, uint8_t const* counts,
    int partial_shift, int seqlen, float scale,
    FamilyTypes<true>::Element* output, float* lse) {
  using Element = FamilyTypes<true>::Element;
  int const lane = threadIdx.x % 16;
  unsigned const mask = 0xffffu << (threadIdx.x & 16);
  int const row = blockIdx.x * 8 + threadIdx.x / 16;
  if (row >= seqlen * 64) return;
  int const query = row / 64, head = row % 64;
  int const count = counts[row];
  bool const valid = lane < count;
  int const group_row = (head / 16) * seqlen + query;
  int const source = valid ? inverse[(group_row << partial_shift) + lane] +
      (head % 16) * (lane == count - 1 ? 128 : 1) : 0;
  float const l = valid ? partial_l[source] : 0.f;
  float const m = l > 0.f ? partial_m[source] : -INFINITY;
  float maximum = m;
  if constexpr (Short) {
    maximum = fmaxf(__shfl_sync(mask, m, 0, 16),
                    __shfl_sync(mask, m, 1, 16));
  } else {
    #pragma unroll
    for (int delta = 8; delta; delta /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, delta, 16));
  }
  float const weight = l > 0.f ? l * expf(scale * (m - maximum)) : 0.f;
  float sum = weight;
  if constexpr (Short) {
    sum = __shfl_sync(mask, weight, 0, 16) +
          __shfl_sync(mask, weight, 1, 16);
  } else {
    #pragma unroll
    for (int delta = 8; delta; delta /= 2)
      sum += __shfl_xor_sync(mask, sum, delta, 16);
  }
  auto load = [&](int rank) {
    int const p = __shfl_sync(mask, source, rank, 16);
    return *reinterpret_cast<uint4 const*>(partial_o + uint64_t(p) * 128 + lane * 8);
  };
  float accum[8] = {};
  auto accumulate = [&](uint4 const& packed, float w) {
    auto const* values = reinterpret_cast<Element const*>(&packed);
    #pragma unroll
    for (int d = 0; d < 8; ++d) accum[d] += w * float(values[d]);
  };
  if constexpr (Short) {
    #pragma unroll
    for (int rank = 0; rank < 2; ++rank) {
      float const w = __shfl_sync(mask, weight, rank, 16);
      if (rank < count && w > 0.f) accumulate(load(rank), w);
    }
  } else {
    float current_weight = __shfl_sync(mask, weight, 0, 16);
    uint4 current = current_weight > 0.f ? load(0) : uint4{};
    #pragma unroll 1
    for (int rank = 0; rank < count; ++rank) {
      float next_weight = 0.f;
      uint4 next{};
      if (rank + 1 < count) {
        next_weight = __shfl_sync(mask, weight, rank + 1, 16);
        if (next_weight > 0.f) next = load(rank + 1);
      }
      // Issue the next vector before consuming the current one.
      accumulate(current, current_weight);
      current = next;
      current_weight = next_weight;
    }
  }
  float const inv_sum = sum > 0.f ? 1.f / sum : 0.f;
  uint4 packed{};
  auto* values = reinterpret_cast<Element*>(&packed);
  #pragma unroll
  for (int d = 0; d < 8; ++d) values[d] = Element(accum[d] * inv_sum);
  *reinterpret_cast<uint4*>(output + uint64_t(row) * 128 + lane * 8) = packed;
  if (lane == 0) lse[uint64_t(head) * seqlen + query] =
      sum > 0.f ? scale * maximum + logf(sum) : -INFINITY;
}

}  // namespace pipelink::attention::sm90
