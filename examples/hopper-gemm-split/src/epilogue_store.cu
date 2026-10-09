#include "epilogue.cuh"

extern "C" __device__ void gemm_epilogue_store_role(const void* params, unsigned) {
  hopper_gemm::epilogue_owner::run<0>(params);
}
GEMM_EPILOGUE_CONTRACT("store", gemm_epilogue_store_role);
