#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cutlass/bfloat16.h>
#include <cmath>

// The math owner depends only on CUTLASS/CuTe. The VM adapter supplies one page
// per warp and publishes the resulting normalized output and base-2 LSE.
namespace pipelink::cute_attention {
using Element = cutlass::bfloat16_t;
inline constexpr int warps = 4;
inline constexpr int heads = 4;
inline constexpr int head_dim = 64;
inline constexpr int tile_tokens = 32;

struct alignas(128) Workspace {
  Element q[16 * head_dim];
  Element k[tile_tokens * head_dim];
  Element v[tile_tokens * head_dim];
  Element p[16 * tile_tokens];
  float out[heads * head_dim];
  float lse[heads];
};
static_assert(sizeof(Workspace) <= 16384);

using QLayout = decltype(cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_16, cute::_64>,
                 cute::Stride<cute::_64, cute::_1>>{}));
using KVLayout = decltype(cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_32, cute::_64>,
                 cute::Stride<cute::_64, cute::_1>>{}));
using VTransposeLayout = decltype(cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_64, cute::_32>,
                 cute::Stride<cute::_1, cute::_64>>{}));
using PLayout = decltype(cute::composition(
    cute::Swizzle<2, 3, 3>{},
    cute::Layout<cute::Shape<cute::_16, cute::_32>,
                 cute::Stride<cute::_32, cute::_1>>{}));
using Mma = decltype(cute::make_tiled_mma(
    cute::SM80_16x8x16_F32BF16BF16F32_TN{}));

__device__ __forceinline__ void load_kv(
    Workspace& shared, const Element* k, const Element* v, int stride,
    int first, int end, int lane) {
  using Copy = cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
#pragma unroll
  for (int i = 0; i < tile_tokens / 4; ++i) {
    const int row = lane / 8 + 4 * i;
    const int column = (lane % 8) * 8;
    const bool valid = first + row < end;
    // Even a zero-fill copy gets a valid source address; it never reads it.
    const int token = valid ? first + row : first;
    const int offset = KVLayout{}(cute::make_coord(row, column));
    Copy::copy(*reinterpret_cast<const cute::uint128_t*>(k + token * stride + column),
               *reinterpret_cast<cute::uint128_t*>(shared.k + offset), valid);
    Copy::copy(*reinterpret_cast<const cute::uint128_t*>(v + token * stride + column),
               *reinterpret_cast<cute::uint128_t*>(shared.v + offset), valid);
  }
  cute::cp_async_fence();
}

// All 32 lanes participate. Each warp visits a disjoint interleaved subset of
// the external partition, retaining FP32 softmax state and BF16 P for the MMA.
__device__ __forceinline__ void run_warp(
    Workspace& shared, const Element* q, const Element* k, const Element* v,
    int kv_stride, int begin, int end, float scale, int warp, int lane) {
  using namespace cute;
  if (begin + warp * tile_tokens >= end) {
#pragma unroll
    for (int i = lane; i < heads * head_dim; i += 32) shared.out[i] = 0;
    if (lane < heads) shared.lse[lane] = -INFINITY;
    __syncwarp();
    return;
  }
  auto sq = make_tensor(make_smem_ptr(shared.q), QLayout{});
  auto sk = make_tensor(make_smem_ptr(shared.k), KVLayout{});
  auto sv = make_tensor(make_smem_ptr(shared.v), VTransposeLayout{});
  auto sp = make_tensor(make_smem_ptr(shared.p), PLayout{});
  // Q is overwritten by another worker at every layer. Match the VM's
  // producer-consumer protocol by bypassing the noncoherent global L1 cache.
#pragma unroll
  for (int i = lane; i < 16 * head_dim / 8; i += 32) {
    const int row = i / 8;
    const int column = (i % 8) * 8;
    SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>::copy(
        *reinterpret_cast<const uint128_t*>(q + (row % heads) * head_dim + column),
        *reinterpret_cast<uint128_t*>(shared.q + QLayout{}(make_coord(row, column))));
  }
  cp_async_fence();
  cp_async_wait<0>();
  __syncwarp();

  Mma mma;
  auto thread_mma = mma.get_slice(lane);
  auto out_coord = thread_mma.partition_C(make_identity_tensor(Shape<_16, _64>{}));
  auto score_coord = thread_mma.partition_C(make_identity_tensor(Shape<_16, _32>{}));
  auto out = partition_fragment_C(mma, Shape<_16, _64>{});
  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, Element>{}, mma);
  auto copy_v = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, Element>{}, mma);
  auto lane_a = copy_a.get_slice(lane);
  auto lane_b = copy_b.get_slice(lane);
  auto lane_v = copy_v.get_slice(lane);
  clear(out);
  // The M16 atom gives each lane two rows, separated by eight. Four adjacent
  // lanes jointly own a row; all N fragments participate in its reduction.
  float maximum[2] = {-INFINITY, -INFINITY};
  float denominator[2] = {0.0f, 0.0f};
  const float scale_log2 = scale * 1.4426950408889634f;
  for (int first = begin + warp * tile_tokens; first < end;
       first += warps * tile_tokens) {
    load_kv(shared, k, v, kv_stride, first, end, lane);
    cp_async_wait<0>();
    __syncwarp();
    auto scores = partition_fragment_C(mma, Shape<_16, _32>{});
    clear(scores);
    {
      auto rq = thread_mma.partition_fragment_A(sq);
      auto rk = thread_mma.partition_fragment_B(sk);
      auto q_src = lane_a.partition_S(sq);
      auto k_src = lane_b.partition_S(sk);
      auto q_dst = lane_a.retile_D(rq);
      auto k_dst = lane_b.retile_D(rk);
      CUTE_UNROLL
      for (int step = 0; step < size<2>(rq); ++step) {
        copy(copy_a, q_src(_, _, step), q_dst(_, _, step));
        copy(copy_b, k_src(_, _, step), k_dst(_, _, step));
        gemm(mma, rq(_, _, step), rk(_, _, step), scores);
      }
    }
    float next_max[2] = {maximum[0], maximum[1]};
    CUTE_UNROLL
    for (int i = 0; i < size(scores); ++i) {
      const int row = get<0>(score_coord(i)) / 8;
      scores(i) = first + get<1>(score_coord(i)) < end
                      ? scores(i) * scale_log2 : -INFINITY;
      next_max[row] = fmaxf(next_max[row], scores(i));
    }
#pragma unroll
    for (int row = 0; row < 2; ++row) {
      next_max[row] = fmaxf(next_max[row], __shfl_xor_sync(0xffffffff, next_max[row], 1));
      next_max[row] = fmaxf(next_max[row], __shfl_xor_sync(0xffffffff, next_max[row], 2));
    }
    const float correction[2] = {exp2f(maximum[0] - next_max[0]),
                                 exp2f(maximum[1] - next_max[1])};
    CUTE_UNROLL
    for (int i = 0; i < size(out); ++i)
      out(i) *= correction[get<0>(out_coord(i)) / 8];
    float tile_sum[2] = {0.0f, 0.0f};
    CUTE_UNROLL
    for (int i = 0; i < size(scores); ++i) {
      const auto coordinate = score_coord(i);
      const int row = get<0>(coordinate) / 8;
      const float probability = exp2f(scores(i) - next_max[row]);
      tile_sum[row] += probability;
      sp(coordinate) = Element(probability);
    }
#pragma unroll
    for (int row = 0; row < 2; ++row) {
      tile_sum[row] += __shfl_xor_sync(0xffffffff, tile_sum[row], 1);
      tile_sum[row] += __shfl_xor_sync(0xffffffff, tile_sum[row], 2);
      denominator[row] = denominator[row] * correction[row] + tile_sum[row];
      maximum[row] = next_max[row];
    }
    __syncwarp();
    {
      auto rp = thread_mma.partition_fragment_A(sp);
      auto rv = thread_mma.partition_fragment_B(sv);
      auto p_src = lane_a.partition_S(sp);
      auto v_src = lane_v.partition_S(sv);
      auto p_dst = lane_a.retile_D(rp);
      auto v_dst = lane_v.retile_D(rv);
      CUTE_UNROLL
      for (int step = 0; step < size<2>(rp); ++step) {
        copy(copy_a, p_src(_, _, step), p_dst(_, _, step));
        copy(copy_v, v_src(_, _, step), v_dst(_, _, step));
        gemm(mma, rp(_, _, step), rv(_, _, step), out);
      }
    }
    __syncwarp();
  }
  CUTE_UNROLL
  for (int i = 0; i < size(out); ++i) {
    const auto coordinate = out_coord(i);
    const int row = get<0>(coordinate);
    if (row < heads) {
      const float denom = denominator[0];
      shared.out[row * head_dim + get<1>(coordinate)] = denom > 0 ? out(i) / denom : 0;
      if (get<1>(coordinate) == 0)
        shared.lse[row] = denom > 0 ? maximum[0] + log2f(denom) : -INFINITY;
    }
  }
  __syncwarp();
}

// Merge independent normalized outputs, including empty warp partitions.
__device__ __forceinline__ float merge_weights(
    const Workspace* const (&parts)[warps], int head, float (&weights)[warps]) {
  float maximum = -INFINITY;
#pragma unroll
  for (int w = 0; w < warps; ++w) maximum = fmaxf(maximum, parts[w]->lse[head]);
  float sum = 0;
#pragma unroll
  for (int w = 0; w < warps; ++w) {
    weights[w] = maximum == -INFINITY ? 0 : exp2f(parts[w]->lse[head] - maximum);
    sum += weights[w];
  }
#pragma unroll
  for (int w = 0; w < warps; ++w) weights[w] = sum > 0 ? weights[w] / sum : 0;
  return sum > 0 ? maximum + log2f(sum) : -INFINITY;
}

__device__ __forceinline__ float merge_output(
    const Workspace* const (&parts)[warps], const float (&weights)[warps], int index) {
  float value = 0;
#pragma unroll
  for (int w = 0; w < warps; ++w) value += weights[w] * parts[w]->out[index];
  return value;
}
}  // namespace pipelink::cute_attention
