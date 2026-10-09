#pragma once

#include "fa3_family.cuh"

namespace pipelink::attention::sm90 {

template <class Config>
struct SplitContext {
  using SharedStorage = typename Config::SharedStorage;
  using PipelineStorage = typename Config::PipelineStorage;
  using PipelineQ = typename Config::PipelineQ;
  using PipelineKV = typename Config::PipelineKV;
  Role role;
  SharedStorage& shared;
  PipelineStorage& pipelines;
  PipelineQ q;
  PipelineKV k, v;
  typename Config::WorkRing work_core, work_epilogue;
  typename Config::KvIndexRing kv_index;
  typename Config::OutputRing corrected_o;
  typename Config::FinalStatsRing final_stats;
  cutlass::arch::ClusterBarrier& o_drained;

  template <class Pipeline>
  CUTLASS_DEVICE static typename Pipeline::Params params(Role role, int bytes) {
    typename Pipeline::Params p;
    if (role == Role::Load) p.role = Pipeline::ThreadCategory::Producer;
    if (role == Role::Core) p.role = Pipeline::ThreadCategory::Consumer;
    p.is_leader = role == Role::Load && threadIdx.x % 32 == 0;
    p.num_consumers = 256;
    p.transaction_bytes = bytes;
    p.initializing_warp = pipelink_attention_warp_load();
    return p;
  }

  template <class Ring, class T, int Depth>
  CUTLASS_DEVICE static Ring bind_ring(T* slots, RingBarriers<Depth>& barriers) {
    Ring ring;
    ring.slots = slots;
    ring.full = barriers.full;
    ring.empty = barriers.empty;
    return ring;
  }

  CUTLASS_DEVICE SplitContext(char* storage, Role r)
      : role(r),
        shared(*reinterpret_cast<SharedStorage*>(storage + pipelink_attention_smem_base())),
        pipelines(*reinterpret_cast<PipelineStorage*>(storage + 16 * pipelink_attention_mbarrier_base())),
        q(pipelines.q, params<PipelineQ>(r, AnnotationFacts<Config>::q_bytes),
          typename Config::ClusterShape{}, cute::false_type{}, cute::false_type{}),
        k(pipelines.k, params<PipelineKV>(r, AnnotationFacts<Config>::kv_bytes),
          typename Config::ClusterShape{}, cute::false_type{}, cute::false_type{}),
        v(pipelines.v, params<PipelineKV>(r, AnnotationFacts<Config>::kv_bytes),
          typename Config::ClusterShape{}, cute::false_type{}, cute::false_type{}),
        work_core(bind_ring<typename Config::WorkRing>(shared.work_core, pipelines.work_core)),
        work_epilogue(bind_ring<typename Config::WorkRing>(shared.work_epilogue, pipelines.work_epilogue)),
        kv_index(bind_ring<typename Config::KvIndexRing>(shared.kv_index, pipelines.kv_index)),
        corrected_o(bind_ring<typename Config::OutputRing>(
            reinterpret_cast<typename Config::Element (*)[Config::kBlockM * 128]>(shared.tensors.smem_v.data()),
            pipelines.corrected_o)),
        final_stats(bind_ring<typename Config::FinalStatsRing>(shared.final_stats, pipelines.final_stats)),
        o_drained(pipelines.o_drained) {}

  CUTLASS_DEVICE void initialize() {
    if (role == Role::Load && cute::elect_one_sync()) {
      pipelines.q.full_barrier_[0].init(1);
      pipelines.q.empty_barrier_[0].init(2);
      CUTLASS_PRAGMA_UNROLL
      for (int stage = 0; stage < Config::kStages; ++stage) {
        pipelines.k.full_barrier_[stage].init(1);
        pipelines.k.empty_barrier_[stage].init(2);
        pipelines.v.full_barrier_[stage].init(1);
        pipelines.v.empty_barrier_[stage].init(2);
      }
      work_core.init(1, 8);
      work_epilogue.init(1, 32);
      kv_index.init(1, 8);
      corrected_o.init(256, 1);
      final_stats.init(256, 32);
      o_drained.init(1);
      cutlass::arch::fence_barrier_init();
      // Load first waits on phase 1 (already free); epilogue alone arrives.
      // Empty work must leave Q and O phases alone.
    }
    // Distinct role entry PCs cannot promise CTA-aligned control flow.
    asm volatile("barrier.sync 0, %0;" :: "n"(Config::kThreads) : "memory");
  }
};

} // namespace pipelink::attention::sm90
