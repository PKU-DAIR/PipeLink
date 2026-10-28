#pragma once
#include <stdint.h>
#include <pipelink_host_abi_v2.h>

struct CodaEpilogueParams {
  const void* a;
  const void* b;
  void* output;
  int32_t M, N, K;
};

#ifndef CODA_ROLE_VIEW
#define CODA_ROLE_VIEW plk_role_view_coda_swiglu_epilogue
#endif
struct CODA_ROLE_VIEW {
  const void* residue;
  const void* scalars;
};
