#ifndef GEMM_SCALED_STORE_H
#define GEMM_SCALED_STORE_H
#include <stdint.h>
#include <pipelink_host_abi_v2.h>
#include "gemm.h"

/* These are the consumer's types, not the withheld GEMM owner's params. */
typedef struct ScaledStoreResidue {
  float* output;
  int32_t M;
  int32_t N;
  float alpha;
} ScaledStoreResidue;

typedef struct plk_role_view_gemm_epilogue_scaled_store_epilogue {
  const void* residue;
  const void* scalars;
} plk_role_view_gemm_epilogue_scaled_store_epilogue;
#endif
