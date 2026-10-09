#include "epilogue.cuh"

extern "C" __device__ void gemm_epilogue_bias_relu_role(const void* params, unsigned) {
  hopper_gemm::epilogue_owner::run<1>(params);
}
GEMM_EPILOGUE_CONTRACT("bias-relu", gemm_epilogue_bias_relu_role);
