#pragma once
// GEMM owner, AllGather host group: the original owner residue (read unchanged by
// the repacked MMA) followed by load_ag's readiness inputs.
#include "host/owner/abi.h"
#include <stdint.h>

struct GemmAgLoadParams {
  const uint32_t* ready;   // this rank's counter slots (index base = gmem_counter binding)
  const uint32_t* gen;     // this rank's completed-launch generation
  int32_t rank;
  int32_t rows_per_rank;   // gathered rows contributed by each rank
  int32_t expected;        // chunk arrivals per row-block per launch
  int32_t reserved;
};

struct GemmAgResidue {
  GemmOwnerResidue gemm;
  GemmAgLoadParams ag;
};

struct plk_role_view_gemm_load_ag_load {
  const void* residue;
  const void* scalars;
};
