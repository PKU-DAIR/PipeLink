// Hand-fused epilogue over the independently lowered TileLang register fragment.
#include <pipelink/stage.hpp>
#include "../consumer/params.h"
#include "../providers/rowscale/row_scale.cuh"
#include PL_TILELANG_LAYOUT

// Keep the generated device implementation verbatim, but publish only this
// native stage's contract. Its resource handles are claimed by our generator.
#pragma push_macro("PIPELINK_EMIT")
#undef PIPELINK_EMIT
#include PL_TILELANG_PROVIDER_SOURCE
#pragma pop_macro("PIPELINK_EMIT")

extern "C" __device__ int pipelink_native_tilelang_scale();

template <int P>
__device__ __forceinline__ void pl_tl_store_pair(
    const EpilogueParams& params, const float* frame, const float* scale) {
  const int t = int(threadIdx.x);
  const int r = pl_tl_row<P>(t);
  const int row = int(blockIdx.x) * 128 + r;
  const int col = int(blockIdx.y) * 128 + pl_tl_col<P>(t);
  const float x = frame[P] * scale[r] + params.bias[row];
  const float y = frame[P + 1] * scale[r] + params.bias[row];
  *reinterpret_cast<float2*>(params.output + row * params.N + col) =
      make_float2(x > 0.0f ? x : 0.0f, y > 0.0f ? y : 0.0f);
}

extern "C" __device__ void plx_role_native_tilelang(const void* raw_params, unsigned variant) {
  (void)variant;
  const auto& params = *static_cast<const EpilogueParams*>(raw_params);
  pl_gemm_tile_args args{params.a, params.b, params.M, params.N, params.K};
  float frame[128];
  plx_produce_source(&args, frame);
  extern __shared__ __align__(1024) unsigned char pl_tl_native_smem[];
  auto* scale = reinterpret_cast<float*>(pl_tl_native_smem + pipelink_native_tilelang_scale());
  pl_row_scale(reinterpret_cast<const __half*>(params.a), params.K,
               int(blockIdx.x) * 128, params.eps, scale);
  __syncthreads();
  PL_TILELANG_STORE_PAIRS(params, frame, scale);
}
