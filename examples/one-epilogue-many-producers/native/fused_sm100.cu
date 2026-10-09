#include "../providers/accumulator/gemm_mainloop_sm100.cuh"
#include "../providers/rowscale/row_scale.cuh"
#include <pipelink/stage.hpp>

extern "C" __device__ void plx_native_sm100(const void* raw_params, unsigned);
extern "C" __device__ int pipelink_native_sm100_smem();
extern "C" __device__ int pipelink_native_sm100_aux();
extern "C" __device__ int pipelink_native_sm100_barrier();

struct alignas(128) NativeSm100Params {
  CUtensorMap map_a;
  CUtensorMap map_b;
  const __half* a;
  const __half* b;
  const float* bias;
  float* output;
  int32_t M, N, K;
  float eps;
};
struct NativeSm100Aux { uint32_t base; float scale[128]; };
PL_STAGE("native.fused.sm100", "1.0.0");
PL_PARAMS(NativeSm100Params, map_a, map_b, a, b, bias, output, M, N, K, eps);
PL_ROLE(fused, plx_native_sm100, 4);
PL_CLAIM_SMEM("pipelink_native_sm100_smem", pl_gemm_sm100::SharedStorage, "kernel");
PL_CLAIM_SMEM("pipelink_native_sm100_aux", NativeSm100Aux, "kernel");
PL_CLAIM("pipelink_native_sm100_tmem", "tmem", 128, 128, "kernel");
PL_CLAIM("pipelink_native_sm100_barrier", "named_barrier", 1, 1, "kernel");

#if defined(__CUDACC__)
extern "C" __device__ void plx_native_sm100(const void* raw_params, unsigned) {
  using namespace pl_gemm_sm100;
  const auto& p = *static_cast<const NativeSm100Params*>(raw_params);
  extern __shared__ __align__(1024) unsigned char plx_native_sm100_shared[];
  auto& shared = *reinterpret_cast<SharedStorage*>(plx_native_sm100_shared + pipelink_native_sm100_smem());
  auto& aux = *reinterpret_cast<NativeSm100Aux*>(plx_native_sm100_shared + pipelink_native_sm100_aux());
  const int barrier = pipelink_native_sm100_barrier();
  if (threadIdx.x < 32) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(128, &aux.base);
    allocator.release_allocation_lock();
  }
  pl_row_scale(p.a, p.K, int(blockIdx.x) * 128, p.eps, aux.scale);
  rendezvous(barrier);
  mainloop(&p.map_a, &p.map_b, p.K, shared, aux.base, barrier);
  const int row = int(blockIdx.x) * 128 + int(threadIdx.x);
  const float scale = aux.scale[threadIdx.x], bias = p.bias[row];
  CUTE_UNROLL
  for (int first = 0; first < 128; first += 16) {
    float v[16];
    load16(aux.base, first, v);
    CUTE_UNROLL
    for (int i = 0; i < 16; ++i) {
      float value = v[i] * scale + bias;
      v[i] = value > 0.0f ? value : 0.0f;
    }
    CUTE_UNROLL
    for (int i = 0; i < 16; i += 4)
      *reinterpret_cast<float4*>(p.output + row * p.N + int(blockIdx.y) * 128 + first + i) =
          make_float4(v[i], v[i+1], v[i+2], v[i+3]);
  }
  rendezvous(barrier);
  if (threadIdx.x < 32) cute::TMEM::Allocator1Sm{}.free(aux.base, 128);
}
#endif
