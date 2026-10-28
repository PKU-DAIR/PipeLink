#include "../../producers/sm90/mainloop.cuh"
namespace coda_sm90 {
__device__ __forceinline__ void store_tile(const CUtensorMap* output, Tile& tile, int m, int n) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
               :: "l"(output), "r"(n), "r"(m), "r"(unsigned(__cvta_generic_to_shared(&tile))) : "memory");
}
template<int Columns>
__device__ __forceinline__ void drain(const CUtensorMap& output, Tile* tiles, const coda_tile_args& args) {
  const int wg = int(threadIdx.x) / 128 - 1;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  warpgroup::sync(wg + 4);
  if (warpgroup::laneid() == 0) {
    #pragma unroll
    for (int n = 0; n < Columns / 64; ++n)
      store_tile(&output, tiles[wg * (Columns / 64) + n], args.m0 + wg * 64,
                 args.n0 / (256 / Columns) + n * 64);
    asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;" ::: "memory");
  }
  warpgroup::sync(wg + 4);
}
__device__ __forceinline__ void fused(const CodaSm90Params& params, const CUtensorMap& output, int M, int N, int K) {
  extern __shared__ __align__(1024) unsigned char native_smem[];
  auto& shared = *reinterpret_cast<Storage*>(native_smem);
  // Dedicated finish storage, separate from the operand arena.
  auto* tiles = reinterpret_cast<Tile*>(native_smem + sizeof(Storage));
  coda_tile_args args{nullptr, nullptr, M, N, K, 0, 0};
  Accumulator acc;
  mainloop<false, true>(params, args, shared, acc, [&](const Accumulator& acc, const coda_tile_args& args) {
    const int wg = int(threadIdx.x) / 128 - 1;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      #pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 pair = acc.tiles[0][j].data[p];
        const int m = row(int(threadIdx.x), j * 8 + p * 2) % 64;
        const int n = col(int(threadIdx.x), j * 8 + p * 2) / 2;
        const float value = __fdividef(pair.x, 1.0f + __expf(-pair.x)) * pair.y;
        const unsigned address = Tile::idx(unsigned(__cvta_generic_to_shared(&tiles[wg * 2 + n / 64])), make_int2(m, n % 64));
        asm volatile("{ .reg .b16 packed; cvt.rn.bf16.f32 packed, %1; st.shared.b16 [%0], packed; }"
                     :: "r"(address), "f"(value) : "memory");
      }
    }
    drain<128>(output, tiles, args);
  });
}
}
extern "C" __global__ __launch_bounds__(384, 1) void coda_sm90_fused(
    const __grid_constant__ CodaSm90Params params, const __grid_constant__ CUtensorMap output, int M, int N, int K) {
  coda_sm90::fused(params, output, M, N, K);
}
extern "C" __global__ __launch_bounds__(384, 1) void coda_sm90_unfused_gemm(
    const __grid_constant__ CodaSm90Params params, const __grid_constant__ CUtensorMap output, int M, int N, int K) {
  using namespace coda_sm90;
  extern __shared__ __align__(1024) unsigned char native_smem[];
  auto& shared = *reinterpret_cast<Storage*>(native_smem);
  coda_tile_args args{nullptr, nullptr, M, N, K, 0, 0};
  Accumulator acc;
  mainloop(params, args, shared, acc, [](const Accumulator&, const coda_tile_args&) {});
  // One-shot: all operand readers have drained, so finish reuses their arena.
  if (threadIdx.x >= 128) {
    const int wg = int(threadIdx.x) / 128 - 1;
    auto* tiles = reinterpret_cast<Tile*>(native_smem);
    warpgroup::store(reinterpret_cast<st_bf<64, 256>&>(tiles[wg * 4]), acc);
    drain<256>(output, tiles, args);
  }
}
