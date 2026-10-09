#include "epilogue.cuh"

extern "C" __device__ void gemm_epilogue_rowscale_role(const void* params, unsigned) {
  hopper_gemm::epilogue_owner::run<2>(params);
}
GEMM_EPILOGUE_CONTRACT("rowscale", gemm_epilogue_rowscale_role);
