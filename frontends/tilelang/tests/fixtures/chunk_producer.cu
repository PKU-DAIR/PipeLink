#include <pipelink/stage.hpp>
#include <cstdint>

#ifndef FIRST_THREAD
#define FIRST_THREAD 0
#endif

struct chunk_fixture_args {
  const void* a;
  const void* b;
  int32_t M;
  int32_t N;
  int32_t K;
  int32_t m0;
  int32_t n0;
};
extern "C" __device__ pipelink_tensor_source chunk_fixture_produce(chunk_fixture_args*, void*);
extern "C" __device__ void chunk_fixture_release(chunk_fixture_args*);

#if defined(__CUDACC__)
extern "C" __device__ pipelink_tensor_source chunk_fixture_produce(chunk_fixture_args* args, void* frame) {
  args->m0 = int(blockIdx.x) * 128;
  args->n0 = int(blockIdx.y) * 256;
  int t = int(threadIdx.x) - FIRST_THREAD;
  if (t >= 0 && t < 256) {
    for (int p = 0; p < 128; ++p) {
#ifdef BANDED_PLACEMENT
      int m = ((t / 32) % 2) * 64 + ((p / 4) % 4) * 16 + ((p / 2) % 2) * 8 + (t % 32) / 4;
      int n = (t / 64) * 64 + (p / 16) * 8 + (t % 4) * 2 + p % 2;
#elif defined(WGMMA_PLACEMENT)
      int m = (t / 32) * 16 + (t % 32) / 4 + ((p / 2) % 2) * 8;
      int n = (p / 4) * 8 + (t % 4) * 2 + p % 2;
#else
      int m = t / 2;
      int n = (p / 16) * 32 + (t % 2) * 16 + p % 16;
#endif
      static_cast<float*>(frame)[p] = float(m * 256 + n) / 32768.0f;
    }
  }
  return pl::source_of(frame);
}
extern "C" __device__ void chunk_fixture_release(chunk_fixture_args*) {
  __syncthreads();
}
#endif

PL_STAGE("test.chunk-fixture", "1.0.0");
PL_NO_PARAMS();
PL_TYPE(chunk_fixture_args, a, b, M, N, K, m0, n0);
PL_CALL(produce, chunk_fixture_produce, pl::consume, "produce");
PL_CALL(release, chunk_fixture_release, pl::consume, "release");
PL_EXPORT_TENSOR(accumulator,
  pl::identity{"accumulator", "tile_work", "completed_k", "f32", "bf16_dot_fp32_completed",
               {{"m", 0, 128}, {"n", 0, 256}}},
  pl::placement{"reg", "scalar", {"participant", "position"}, [](int m, int n) {
#ifdef BANDED_PLACEMENT
    return std::array<int, 2>{
      FIRST_THREAD + (n / 64) * 64 + (m / 64) * 32 + (m % 8) * 4 + (n % 8) / 2,
      ((n % 64) / 8) * 16 + ((m % 64) / 16) * 4 + ((m % 16) / 8) * 2 + n % 2};
#elif defined(WGMMA_PLACEMENT)
    return std::array<int, 2>{
      FIRST_THREAD + (m / 16) * 32 + (m % 8) * 4 + (n % 8) / 2,
      (n / 8) * 4 + ((m % 16) / 8) * 2 + n % 2};
#else
    return std::array<int, 2>{FIRST_THREAD + m * 2 + (n / 16) % 2, (n / 32) * 16 + n % 16};
#endif
  }, pl::coord{"m", 0, 128}, pl::coord{"n", 0, 256}},
  pl::strings{"reg_copy", "reg_copy_round", "smem_store_scalar", "smem_store_round"},
  pl::readiness{"tensor_ready", "import_done"}, pl::fixed_bindings{}, pl::lends{},
  pl::calls{"produce", "release"});
