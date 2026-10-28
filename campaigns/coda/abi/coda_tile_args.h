#ifndef CODA_TILE_ARGS_H
#define CODA_TILE_ARGS_H

#include <stdint.h>

// Public argument record shared by every CODA GEMM producer and its epilogues.
// Inputs: bf16 row-major A[M,K] and K-major B[N,K]. Outputs: m0/n0, the
// global origin of the 128x256 tile this CTA exports. The record is private
// to each calling thread; produce() writes m0/n0 in every calling thread, so
// the consumer reads its own copy after produce() returns. Producers export
//   pipelink_tensor_source coda_produce(const void* view, coda_tile_args*, void* frame);
//   void coda_release(const void* view, coda_tile_args*);
// as host-group view calls: the caller passes its own params view.
#ifndef CODA_TILE_ARGS_DEFINED
#define CODA_TILE_ARGS_DEFINED
typedef struct coda_tile_args {
  const void* a;
  const void* b;
  int32_t M;
  int32_t N;
  int32_t K;
  int32_t m0;
  int32_t n0;
} coda_tile_args;
#endif

#endif
