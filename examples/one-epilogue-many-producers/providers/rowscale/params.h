#pragma once
#include <cstdint>
#include <pipelink/stage.hpp>

// Public call ABI; row-major f16 A[M,K], tile origin selected by blockIdx.x.
#ifndef PL_ROWSCALE_ARGS_DEFINED
#define PL_ROWSCALE_ARGS_DEFINED
struct pl_rowscale_args {
  const void* a;
  int32_t M, K;
  float eps;
};
#endif
PL_TYPE(pl_rowscale_args, const void* a; int32_t M; int32_t K; float eps;);
