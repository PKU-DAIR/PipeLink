#pragma once

#include <cstddef>
#include "family_params.cuh"
#include "work.hpp"
#include "sm90_pipeline_no_cluster.hpp"

extern "C" __device__ int pipelink_attention_smem_base();
extern "C" __device__ int pipelink_attention_mbarrier_base();
extern "C" __device__ int pipelink_attention_warp_load();
extern "C" __device__ int pipelink_attention_warp_core();
extern "C" __device__ int pipelink_attention_warp_epilogue();
extern "C" __device__ int pipelink_attention_warp_empty();
extern "C" __device__ int pipelink_attention_warp_scheduler_wg1();
extern "C" __device__ int pipelink_attention_warp_scheduler_wg2();
extern "C" __device__ int pipelink_attention_epilogue_barrier();
extern "C" __device__ bool pipelink_call_mask_mod(
    int32_t, int32_t, pipelink_attention_sm90_work_desc const&);
#if !defined(PIPELINK_SM90_NO_SCORE_SEAM)
extern "C" __device__ float pipelink_call_score_mod(
    float, pipelink_attention_sm90_work_desc const&);
#endif

namespace pipelink::attention::sm90 {

enum class Role { Load, Core, Epilogue, Empty };

// Optional encoding in the existing flags word. Coordinates are base[0]+col
// and the validity bitmap is a prefix of contiguous_valid_count columns.
// The full base/valid record remains available to generic mask callees.
static constexpr uint32_t contiguous = 1u << 3;
static constexpr int contiguous_valid_shift = 8;

template <int N>
struct KvSlotMeta {
  // Exactly fully_valid needs no coordinates: the core skips all masking.
  // Otherwise every base/valid word is initialized by the producer.
  static_assert(N % 16 == 0);
  int32_t base[N / 16];
  uint32_t valid[(N + 31) / 32];
  uint32_t flags;
};

template <int N>
CUTLASS_DEVICE int32_t kv_coord(KvSlotMeta<N> const& meta, int column) {
  bool valid = (meta.valid[column >> 5] >> (column & 31)) & 1u;
  return valid ? meta.base[column >> 4] + (column & 15) : -1;
}

template <int Depth>
struct RingBarriers {
  cutlass::arch::ClusterBarrier full[Depth];
  cutlass::arch::ClusterBarrier empty[Depth];
};

template <bool Causal, int ClusterM = 1>
struct Family : FamilyTypes<Causal, ClusterM> {
  // This family uses one CTA per cluster; cluster multicast is unsupported.
  static_assert(ClusterM == 1);
  using Types = FamilyTypes<Causal, ClusterM>;
  using Mainloop = typename Types::Mainloop;
  using Epilogue = typename Types::Epilogue;
  using Element = typename Types::Element;
  using TileShape_MNK = typename Mainloop::TileShape_MNK;
  using TiledMmaQK = typename Mainloop::TiledMmaQK;
  using TiledMmaPV = typename Mainloop::TiledMmaPV;
  static constexpr int kBlockM = Types::kBlockM;
  static constexpr int kBlockN = Types::kBlockN;
  static constexpr int kStages = 2;
  static constexpr int kThreads = 384;
  using KvSlotMeta = sm90::KvSlotMeta<kBlockN>;
  using PipelineQ = cutlass::PipelineTmaAsyncNoCluster<1>;
  using PipelineKV = cutlass::PipelineTmaAsyncNoCluster<kStages>;
  using WorkRing = SmemRing<WorkDesc, 2>;
  using KvIndexRing = SmemRing<KvSlotMeta, 4>;
  using FinalStatsRing = SmemRing<FinalStats, 1>;
  using OutputRing = SmemRing<Element[kBlockM * 128], 1>;

  struct SharedStorage {
    typename Mainloop::TensorStorage tensors;
    WorkDesc work_core[WorkRing::depth];
    WorkDesc work_epilogue[WorkRing::depth];
    KvSlotMeta kv_index[KvIndexRing::depth];
    FinalStats final_stats[FinalStatsRing::depth];
  };

  struct PipelineStorage {
    typename PipelineQ::SharedStorage q;
    typename PipelineKV::SharedStorage k;
    typename PipelineKV::SharedStorage v;
    RingBarriers<WorkRing::depth> work_core;
    RingBarriers<WorkRing::depth> work_epilogue;
    RingBarriers<KvIndexRing::depth> kv_index;
    RingBarriers<OutputRing::depth> corrected_o;
    RingBarriers<FinalStatsRing::depth> final_stats;
    cutlass::arch::ClusterBarrier o_drained;
  };

  static_assert(kBlockM == 128 && (kBlockN == 128 || kBlockN == 176));
  static_assert(cute::size(TiledMmaQK{}) == 256 && cute::size(TiledMmaPV{}) == 256);
  static_assert(sizeof(typename Epilogue::TensorStorage) <=
                cute::cosize_v<typename Mainloop::SmemLayoutVt> * sizeof(Element));
};

#if defined(ATTENTION_BUILD_D128_SM90_C) && !defined(ATTENTION_BUILD_D128_SM90_NC)
using Config = Family<true>;
#define PIPELINK_SM90_SPEC "d128-sm90-c"
#define PIPELINK_SM90_SYMBOL_(suffix) org_pipelink_attention_d128_sm90_c_##suffix
#elif defined(ATTENTION_BUILD_D128_SM90_NC) && !defined(ATTENTION_BUILD_D128_SM90_C)
using Config = Family<false>;
#define PIPELINK_SM90_SPEC "d128-sm90-nc"
#define PIPELINK_SM90_SYMBOL_(suffix) org_pipelink_attention_d128_sm90_nc_##suffix
#endif
#define PIPELINK_SM90_SYMBOL(suffix) PIPELINK_SM90_SYMBOL_(suffix)

// These offsets describe the actual owner storage, not independent allocations.
// Every endpoint binds the same base; origin_bits selects its physical view.
template <class F>
struct AnnotationFacts {
  using Mainloop = typename F::Mainloop;
  using Epilogue = typename F::Epilogue;
  using Shared = typename F::SharedStorage;
  using Tensors = typename Mainloop::TensorStorage;
  using Pipelines = typename F::PipelineStorage;
  static constexpr int shared_bytes = sizeof(Shared);
  static constexpr int shared_align = alignof(Shared);
  static constexpr int pipeline_slots = (sizeof(Pipelines) + 15) / 16;
  static constexpr int q_offset = offsetof(Shared, tensors) + offsetof(Tensors, smem_q);
  static constexpr int k_offset = offsetof(Shared, tensors) + offsetof(Tensors, smem_k);
  static constexpr int v_offset = offsetof(Shared, tensors) + offsetof(Tensors, smem_v);
  static constexpr int o_offset = v_offset;
  static constexpr int work_core_offset = offsetof(Shared, work_core);
  static constexpr int work_epilogue_offset = offsetof(Shared, work_epilogue);
  static constexpr int kv_index_offset = offsetof(Shared, kv_index);
  static constexpr int final_stats_offset = offsetof(Shared, final_stats);
  static constexpr int q_bytes = F::kBlockM * 128 * sizeof(typename F::Element);
  static constexpr int kv_bytes = F::kBlockN * 128 * sizeof(typename F::Element);

  // pipec's affine mode checker requires scalar modes. Split the 128-column
  // logical mode into {64,2}; this is the exact FA3 physical layout, not a
  // row-major approximation. V's native (head_dim,N) view is transposed here
  // into the same (N,64,2) endpoint coordinates as K.
  static constexpr auto q_affine = cute::get_nonswizzle_portion(typename Mainloop::SmemLayoutQ{});
  static constexpr auto k_affine = cute::get_nonswizzle_portion(typename Mainloop::SmemLayoutK{});
  static constexpr auto v_affine = cute::get_nonswizzle_portion(typename Mainloop::SmemLayoutVt{});
  static constexpr auto o_affine = cute::get_nonswizzle_portion(typename Epilogue::SmemLayoutO{});
  static_assert(q_affine(cute::make_coord(1, 0)) == 64);
  static_assert(q_affine(cute::make_coord(0, 64)) == 8192);
  static_assert(k_affine(cute::make_coord(1, 0, 0)) == 64);
  static_assert(k_affine(cute::make_coord(0, 64, 0)) == F::kBlockN * 64);
  static_assert(k_affine(cute::make_coord(0, 0, 1)) * sizeof(typename F::Element) == kv_bytes);
  static_assert(v_affine(cute::make_coord(0, 1, 0)) == 64);
  static_assert(v_affine(cute::make_coord(64, 0, 0)) == F::kBlockN * 64);
  static_assert(v_affine(cute::make_coord(0, 0, 1)) * sizeof(typename F::Element) == kv_bytes);
  static_assert(o_affine(cute::make_coord(1, 0)) == 64);
  static_assert(o_affine(cute::make_coord(0, 64)) == 8192);
  static_assert(cute::get_swizzle_portion(typename Mainloop::SmemLayoutQ{}) == cute::Swizzle<3, 4, 3>{});
  static_assert(cute::get_swizzle_portion(typename Mainloop::SmemLayoutK{}) == cute::Swizzle<3, 4, 3>{});
  static_assert(cute::get_swizzle_portion(typename Mainloop::SmemLayoutVt{}) == cute::Swizzle<3, 4, 3>{});
  static_assert(cute::get_swizzle_portion(typename Epilogue::SmemLayoutO{}) == cute::Swizzle<3, 4, 3>{});
};

} // namespace pipelink::attention::sm90
