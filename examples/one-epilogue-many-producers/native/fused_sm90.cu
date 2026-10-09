#include "../providers/accumulator/gemm_mainloop_sm90.cuh"
#include "../providers/rowscale/row_scale.cuh"
#include <pipelink/stage.hpp>

struct alignas(128) Sm90FusedParams {
  CUtensorMap a_map;
  CUtensorMap b_map;
  const uint16_t* a;
  const uint16_t* b;
  const float* bias;
  float* output;
  int32_t M, N, K;
  float eps;
};
static_assert(sizeof(Sm90FusedParams) == 384);
extern "C" __device__ int pipelink_sm90_native_smem();
extern "C" __device__ int pipelink_sm90_native_scale();
extern "C" __device__ int pipelink_sm90_native_barrier();
extern "C" __device__ void plx_role_fused_sm90(const void*, unsigned);

PL_STAGE("native.fused.sm90", "1.0.0");
PL_PARAMS(Sm90FusedParams, a_map, b_map, a, b, bias, output, M, N, K, eps);
PL_ROLE(fused, plx_role_fused_sm90, 4);
PL_CLAIM_SMEM("pipelink_sm90_native_smem", pl_gemm_sm90::Configuration::SharedStorage, "kernel");
PL_CLAIM("pipelink_sm90_native_scale", "smem", 512, 128, "kernel");
PL_CLAIM("pipelink_sm90_native_barrier", "named_barrier", 1, 1, "kernel");

extern "C" __device__ void plx_role_fused_sm90(const void* raw_params, unsigned) {
  using namespace pl_gemm_sm90;
  const auto& params = *static_cast<const Sm90FusedParams*>(raw_params);
  extern __shared__ __align__(1024) unsigned char plx_sm90_native_smem[];
  auto& shared = *reinterpret_cast<Configuration::SharedStorage*>(
      plx_sm90_native_smem + pipelink_sm90_native_smem());
  auto* scale = reinterpret_cast<float*>(plx_sm90_native_smem + pipelink_sm90_native_scale());
  auto coordinates = Configuration::partitioned_coordinates(int(threadIdx.x));
  auto accumulator = Configuration::TiledMma{}.get_slice(threadIdx.x).make_fragment_C(coordinates);
  mainloop(&params.a_map, &params.b_map, params.K, shared, pipelink_sm90_native_barrier(), accumulator);
  pl_row_scale(reinterpret_cast<const __half*>(params.a), params.K, int(blockIdx.x) * 128, params.eps, scale);
  rendezvous(pipelink_sm90_native_barrier());
  CUTE_UNROLL
  for (int i = 0; i < columns; i += 2) {
    const auto logical = coordinates(i);
    const int row = int(cute::get<0>(logical));
    const int column = int(cute::get<1>(logical));
    const int m = int(blockIdx.x) * 128 + row;
    const float s = scale[row], bias = params.bias[m];
    const float x = accumulator(i) * s + bias;
    const float y = accumulator(i + 1) * s + bias;
    *reinterpret_cast<float2*>(params.output + m * params.N + int(blockIdx.y) * 128 + column) =
        make_float2(x > 0.0f ? x : 0.0f, y > 0.0f ? y : 0.0f);
  }
}
