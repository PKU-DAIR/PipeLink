#pragma once
#include <cuda.h>
#include <cuda_bf16.h>

inline constexpr int TileRows = 128;
inline constexpr int TileCols = 64;
inline constexpr int TileElements = TileRows * TileCols;
inline constexpr int TileBytes = TileElements * sizeof(__nv_bfloat16);
inline constexpr int ConsumerWarps = 4;

struct ProducerParams {
  CUtensorMap input;
  int columns;
};
struct ConsumerParams {
  float* output;
  int columns;
  float scale;
  float bias;
};

extern "C" __device__ int pipelink_ring_base();
extern "C" __device__ int pipelink_ring_full();
extern "C" __device__ int pipelink_ring_empty();
extern "C" __device__ int pipelink_ring_depth();
extern "C" __device__ int pipelink_ring_warp_consumer();

__device__ __forceinline__ void ring_wait(unsigned address, unsigned phase) {
  unsigned ready;
  do {
    asm volatile("{ .reg .pred done; "
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%1], %2; "
                 "selp.u32 %0, 1, 0, done; }"
                 : "=r"(ready) : "r"(address), "r"(phase) : "memory");
  } while (!ready);
}
