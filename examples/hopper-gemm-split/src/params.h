#pragma once
#include <cuda.h>
#include <stdint.h>

// Each stage has its own params blob. Only load's carries CUDA tensor maps, and
// no CuTe types cross the seam.
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
