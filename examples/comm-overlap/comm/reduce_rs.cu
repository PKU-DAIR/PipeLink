// Owner code, NOT part of the linked GEMM+push image. Launched on the same rank
// stream after GEMM+push. Its bounded resident grid persists over owner tiles.
#include "comm.h"
#include "push_gather.cuh"
#include "rs_reduce.cuh"

extern "C" __global__ __launch_bounds__(128) void comm_rs_reduce(const __grid_constant__ CommRsResidue p) {
  using namespace comm_overlap;
  const uint64_t generation = latch(p.generation);
  const int parity = int(generation & 1u);
  const int tiles = (p.M / p.world / 128) * (p.N / 128);
  const size_t source_stride = size_t(p.M / p.world) * (p.N / 4);
  const auto* slots = reinterpret_cast<const float4*>(p.slots[p.rank]) + size_t(parity) * p.world * source_stride;
  const auto* counters = reinterpret_cast<const uint32_t*>(p.counters[p.rank]) + parity * tiles;
  const uint32_t target = uint32_t(((generation >> 1) + 1) * uint64_t(p.world - 1));
  // Current-generation peer producers are GEMM+push kernels with NO current-gen
  // waits. Local own-slot writes completed on this stream before this launch.
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    if (threadIdx.x == 0)
      while (!reached(load_acquire_sys(counters + tile), target)) __nanosleep(32);
    // The acquiring lane hands peer publication to every reader, not via shuffle.
    __syncthreads();
    fence_acq_rel_sys();
    rs::reduce<128>(p, slots, source_stride, tile, threadIdx.x);
  }
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p.done) : "memory");
    if (old == gridDim.x - 1) {
      *reinterpret_cast<volatile uint64_t*>(p.generation) = generation + 1;
      for (int peer = 0; peer < p.world; ++peer) {
        auto* released = reinterpret_cast<uint32_t*>(p.released[peer]) + p.rank;
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(released), "r"(uint32_t(generation + 1)) : "memory");
      }
    }
  }
}
