#include <cuda_bf16.h>
#include <stdint.h>

// Interleaved bf16 input [M,N] -> bf16 output [M,N/2].
// Launch: block=(256,1,1), grid=(ceil((M*N/2)/1024),1,1), shared=0.
// N is a multiple of 8 for CODA shapes, so every vector is complete.
extern "C" __global__ __launch_bounds__(256)
void coda_swiglu(const __nv_bfloat16* __restrict__ input,
                 __nv_bfloat16* __restrict__ output, int M, int N) {
  const int64_t vector = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vector >= int64_t(M) * N / 8) return;
  const uint4 packed = reinterpret_cast<const uint4*>(input)[vector];
  const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
  uint2 result;
  auto* out = reinterpret_cast<__nv_bfloat162*>(&result);
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float2 a = __bfloat1622float2(pairs[2*i]);
    const float2 b = __bfloat1622float2(pairs[2*i+1]);
    out[i] = __floats2bfloat162_rn(__fdividef(a.x, 1.0f + __expf(-a.x)) * a.y,
                                 __fdividef(b.x, 1.0f + __expf(-b.x)) * b.y);
  }
  reinterpret_cast<uint2*>(output)[vector] = result;
}
