#pragma once
#include "../../src/params.h"
#include <pipelink_host_abi_v2.h>

// This descriptor-bearing layout is compiled by the owner, never distributed.
struct GemmOwnerResidue {
  GemmLoadParams load;
  GemmMmaParams mma;
};

struct plk_role_view_gemm_load_load {
  const void* residue;
  const void* scalars;
};
struct plk_role_view_gemm_mma_mma {
  const void* residue;
  const void* scalars;
};
