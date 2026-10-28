#include "swiglu.cu"
#include <cuda_runtime.h>

extern "C" int coda_swiglu_launch(const void* input, void* output, int M, int N, void* stream) {
  coda_swiglu<<<(int64_t(M) * N / 8 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<__nv_bfloat16*>(output), M, N);
  return int(cudaGetLastError());
}
