#ifndef PIPELINK_GEMM_TILE_ARGS_H
#define PIPELINK_GEMM_TILE_ARGS_H

#include <stdint.h>

// f16 row-major A[M,K], B[N,K]; tile geometry belongs to the export manifest.
// CTA origin is blockIdx.{x,y} times the provider's tile rows/columns.
// Storage is borrowed through release.
#ifndef PL_GEMM_TILE_ARGS_DEFINED
#define PL_GEMM_TILE_ARGS_DEFINED
typedef struct pl_gemm_tile_args {
  const void* a;
  const void* b;
  int32_t M;
  int32_t N;
  int32_t K;
} pl_gemm_tile_args;
#endif

#endif
