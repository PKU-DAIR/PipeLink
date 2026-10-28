#include "../../producers/sm120/mainloop.cuh"

namespace coda_sm120 {
template<bool Fused, int Rows = 128, int Columns = 256>
__device__ __forceinline__ void execute(void const* a, void const* b, void* out, int M, int N, int K) {
  using C = Configuration<Rows, Columns>;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& shared = *reinterpret_cast<typename C::SharedStorage*>(storage);
  int tiles = (M / Rows) * (N / Columns);
  int tile = int(blockIdx.x);
  if (tile < tiles) {
    coda_tile_args args{a, b, M, N, K, 0, 0};
    C::origin(tile, M, N, args.m0, args.n0);
    typename C::Fragment acc;
    C::run(args, shared, acc);
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < C::Fragment::kElements; p += 2) {
      auto rc = C::coordinate(int(threadIdx.x), p);
      int m = args.m0 + rc.row(), n = args.n0 + rc.column();
      if constexpr (Fused) {
        float g = acc[p], u = acc[p + 1];
        static_cast<__nv_bfloat16*>(out)[size_t(m) * (N / 2) + n / 2] =
            __float2bfloat16_rn(__fdividef(g, 1.0f + __expf(-g)) * u);
      } else {
        reinterpret_cast<__nv_bfloat162*>(out)[(size_t(m) * N + n) / 2] =
            __floats2bfloat162_rn(acc[p], acc[p + 1]);
      }
    }
  }
}
}
extern "C" __global__ __launch_bounds__(256)
void coda_sm120_fused(void const* a, void const* b, void* out, int M, int N, int K) {
  coda_sm120::execute<true>(a, b, out, M, N, K);
}
extern "C" __global__ __launch_bounds__(256)
void coda_sm120_gemm(void const* a, void const* b, void* out, int M, int N, int K) {
  coda_sm120::execute<false>(a, b, out, M, N, K);
}
extern "C" __global__ __launch_bounds__(256)
void coda_sm120_gemm_256x128(void const* a, void const* b, void* out, int M, int N, int K) {
  coda_sm120::execute<false, 256, 128>(a, b, out, M, N, K);
}
