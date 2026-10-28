#pragma once
#include "params.h"
#include "../../abi/coda_tile_args.h"
#include "kittens.cuh"

namespace coda_sm90 {
using namespace kittens;
using Accumulator = rt_fl<16, 256>;
using Tile = st_bf<64, 64>;
struct Input { Tile a[2], b[4]; };
struct alignas(1024) Storage {
  Input input[4];
  semaphore full[4], empty[4];
};
static_assert(sizeof(Accumulator) == 128 * sizeof(float));
// The same coordinate function drives native stores and export capture.
__host__ __device__ constexpr int row(int thread, int position) {
  return ((thread - 128) / 32) * 16 + (thread % 32) / 4 + ((position / 2) % 2) * 8;
}
__host__ __device__ constexpr int col(int thread, int position) {
  return (position / 8) * 16 + ((position / 4) % 2) * 8 + (thread % 4) * 2 + position % 2;
}
__device__ __forceinline__ void origin(coda_tile_args& args, int task) {
  const int rows = args.M / 128, cols = args.N / 256;
  const int group = args.M == 4096 ? 12 : 8;
  const int full_rows = rows / group * group;
  int m, n;
  if (task < full_rows * cols) {
    m = group * (task / (group * cols)) + task % group;
    n = (task % (group * cols)) / group;
  } else {
    const int remainder = task - full_rows * cols;
    m = full_rows + remainder % (rows - full_rows);
    n = remainder / (rows - full_rows);
  }
  args.m0 = m * 128;
  args.n0 = n * 256;
}
__device__ __forceinline__ void load(Tile& tile, const CUtensorMap* map,
                                    int r, int k, semaphore& bar) {
  const unsigned dst = __cvta_generic_to_shared(&tile);
  const unsigned sem = __cvta_generic_to_shared(&bar);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(map), "r"(k), "r"(r), "r"(sem) : "memory");
}
template<bool Persistent = false, bool NativeFinishRegisters = false, class Finish>
__device__ __forceinline__ void mainloop(const CodaSm90Params& params,
    coda_tile_args& args, Storage& shared, Accumulator& acc, Finish finish) {
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
      init_semaphore(shared.full[s], 1, 0);
      init_semaphore(shared.empty[s], 8, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const int tiles = (args.M / 128) * (args.N / 256);
  const int step = Persistent ? int(gridDim.x) : tiles;
  int generation = 0;
  if (threadIdx.x < 128) {
    if constexpr (NativeFinishRegisters) warpgroup::decrease_registers<40>();
    for (int task = int(blockIdx.x); task < tiles; task += step) {
      origin(args, task);
      if (threadIdx.x == 0) {
        for (int it = 0; it < args.K / 64; ++it, ++generation) {
          const int s = generation % 4;
          wait(shared.empty[s], ((generation / 4) & 1) ^ 1);
          tma::expect_bytes(shared.full[s], sizeof(Input));
          #pragma unroll
          for (int i = 0; i < 2; ++i) load(shared.input[s].a[i], &params.a, args.m0 + i * 64, it * 64, shared.full[s]);
          #pragma unroll
          for (int i = 0; i < 4; ++i) load(shared.input[s].b[i], &params.b, args.n0 + i * 64, it * 64, shared.full[s]);
        }
      }
    }
    if constexpr (NativeFinishRegisters) warpgroup::increase_registers<168>();
  } else {
    if constexpr (NativeFinishRegisters) warpgroup::increase_registers<232>();
    for (int task = int(blockIdx.x); task < tiles; task += step) {
      origin(args, task);
      warp::zero(acc);
      for (int it = 0; it < args.K / 64; ++it, ++generation) {
        const int s = generation % 4;
        wait(shared.full[s], (generation / 4) & 1);
        warpgroup::mma_ABt(acc, shared.input[s].a[int(threadIdx.x) / 128 - 1],
                          reinterpret_cast<st_bf<256, 64>&>(shared.input[s].b));
        warpgroup::mma_async_wait();
        if (warp::laneid() == 0) arrive(shared.empty[s]);
      }
      finish(acc, args);
    }
    if constexpr (NativeFinishRegisters) warpgroup::decrease_registers<168>();
  }
  // All operand TMA writes and WGMMA reads are retired at tensor_ready.
  __syncthreads();
}
} // namespace coda_sm90
