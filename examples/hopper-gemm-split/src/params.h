#pragma once
#include <cuda.h>
#include <stdint.h>

// The load owner's descriptor-bearing blob is separate from user epilogue args.
// Only load needs the CUDA tensor-map type; no CuTe templates cross this seam.
struct alignas(128) GemmLoadParams {
  CUtensorMap a;
  CUtensorMap b;
  int32_t K;
};
struct GemmMmaParams {
  int32_t K;
};
struct GemmEpilogueParams {
  float* output;
  const float* bias;
  const float* scale;
  int32_t M;
  int32_t N;
};
static_assert(sizeof(GemmLoadParams) == 384);
static_assert(sizeof(GemmMmaParams) == 4);
static_assert(sizeof(GemmEpilogueParams) == 32);
