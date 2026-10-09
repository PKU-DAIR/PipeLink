#pragma once
#include <cstdint>

struct EpilogueParams {
  const uint16_t* a;
  const uint16_t* b;
  const float* bias;
  float* output;
  int32_t M, N, K;
  float eps;
};
