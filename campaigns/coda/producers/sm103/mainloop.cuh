#pragma once
#include "params.h"
#include "../../abi/coda_tile_args.h"
#include "kittens.cuh"

namespace coda_sm103 {
using namespace kittens;
using Operand = st_bf<128, 64>;
using Accumulator = tt<float, 128, 256>;
template<int Stages> struct alignas(1024) OperandRing { Operand a[Stages]; Operand b[Stages]; };
template<int Stages> struct alignas(16) PipelineState {
  uint32_t base;
  semaphore arrived[Stages], consumed[Stages], output[2], empty[2];
  semaphore provisioned, finished;
};
using Operands = OperandRing<3>;
using State = PipelineState<3>;

__device__ inline unsigned long long clock_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ inline void origin(coda_tile_args& args, int pair) {
  int2 xy = get_swizzled_2d_idx<8>(args.M / 256, args.N / 256, pair);
  args.m0 = 256 * xy.x + 128 * cluster_ctarank();
  args.n0 = 256 * xy.y;
}
__device__ inline void load(Operand& dst, CUtensorMap const* map,
                            int row, int k, semaphore& ready) {
  uint32_t address = uint32_t(__cvta_generic_to_shared(&dst));
  uint32_t bar = uint32_t(__cvta_generic_to_shared(&ready)), remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(remote) : "r"(bar));
  uint16_t mask = uint16_t(1 << cluster_ctarank());
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(address), "l"(map), "r"(remote), "r"(k), "r"(row), "h"(mask) : "memory");
}
template<bool Persistent, int Stages>
__device__ inline void initialize(CodaSm103Params const& p, PipelineState<Stages>& s) {
  if (threadIdx.x == 32) {
    init_semaphore(s.provisioned, 0, 1);
    init_semaphore(s.finished, 0, 1);
    for (int i=0; i<Stages; ++i) {
      init_semaphore(s.arrived[i], 0, 1);
      init_semaphore(s.consumed[i], 0, 1);
    }
    for (int i=0; i<2; ++i) {
      init_semaphore(s.output[i], 0, 1);
      init_semaphore(s.empty[i], 0, 2);
    }
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.a));
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.b));
  }
  everyone::tma::cluster::arrive_aligned();
  everyone::tma::cluster::wait_aligned();
  if (threadIdx.x < 32) {
    tensor_allocator<Persistent ? 1 : 2, 2, false> allocator;
    allocator.provision(s.base);
    warp::arrive(s.provisioned);
  }
  // TMA may fill the operand ring while warp 0 allocates TMEM. Only the
  // MMA issuer needs the allocation before entering its reduction loop.
  if constexpr (Persistent) {
    wait(s.provisioned, 0);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }
}
// Only warp 7 transfers operands; only the leader CTA's warp 4 issues UMMA.
// The same loop is used by the exported one-shot and all native kernels.
template<bool Persistent, int Stages>
__device__ inline void issue(CodaSm103Params const& p, coda_tile_args args,
                            OperandRing<Stages>& op, PipelineState<Stages>& s, int first, int stride, int count) {
  const int warp = threadIdx.x / 32, rank = cluster_ctarank();
  const int steps = args.K / 64;
  if (warp == 7 && warp::elect_leader()) {
    int ring = 0;
    uint32_t phase = 0xffff0000;
    for (int tile=0, pair=first; tile<count; ++tile, pair+=stride) {
      origin(args, pair);
      for (int k=0; k<steps; ++k) {
        wait(s.consumed[ring], get_phasebit<1>(phase, ring));
        load(op.a[ring], &p.a, args.m0, k*64, s.arrived[ring]);
        load(op.b[ring], &p.b, args.n0+rank*128, k*64, s.arrived[ring]);
        update_phasebit<1>(phase, ring);
        ring = ring_advance<Stages>(ring);
      }
    }
  } else if (rank == 0 && warp == 4 && warp::elect_leader()) {
    wait(s.provisioned, 0);
    if (!Persistent && blockIdx.x >= p.num_sms && blockIdx.x < 2*p.num_sms) {
      auto begin=clock_ns();
      while (clock_ns()-begin < p.stagger_ns) {}
    }
    int ring = 0;
    uint32_t phase = 0xffff0000;
    for (int tile=0; tile<count; ++tile) {
      int slot=Persistent ? tile%2 : 0;
      if constexpr (Persistent) wait(s.empty[slot], ((tile/2)&1)^1);
      Accumulator acc(s.base+256*slot);
      for (int k=0; k<steps; ++k) {
        tma::expect_bytes(s.arrived[ring], 4*sizeof(Operand));
        wait(s.arrived[ring], get_phasebit<0>(phase, ring));
        if (k==0) mm2_ABt(acc, op.a[ring], op.b[ring], s.consumed[ring]);
        else mma2_ABt(acc, op.a[ring], op.b[ring], s.consumed[ring]);
        update_phasebit<0>(phase, ring);
        ring = ring_advance<Stages>(ring);
      }
      detail::tcgen05::commit<2>(s.output[slot]);
    }
  }
}
__device__ inline uint32_t produce(CodaSm103Params const& p, coda_tile_args& args,
                                   Operands& op, State& s) {
  origin(args, blockIdx.x/2);
  initialize<false>(p,s);
  issue<false>(p,args,op,s,blockIdx.x/2,1,1);
  wait(s.provisioned, 0);
  wait(s.output[0],0);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  return s.base;
}
template<bool Persistent, int Stages>
__device__ inline void release(PipelineState<Stages>& s) {
  // The caller's final full-CTA done rendezvous precedes this call. An
  // elected peer arrival therefore certifies that both local halves are
  // finished, without another full-cluster all-thread retirement barrier.
  if (threadIdx.x < 32) {
    if (warp::elect_leader()) tma::cluster::arrive(s.finished, 1-cluster_ctarank());
    wait(s.finished, 0);
    tensor_allocator<Persistent ? 1 : 2,2,false> allocator;
    allocator.set_addr(s.base);
    allocator.deprovision();
  }
}
} // namespace coda_sm103
