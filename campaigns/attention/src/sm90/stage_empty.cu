#include "split_context.cuh"
#include <pipelink/stage.hpp>

#if !defined(PIPELINK_SM90_SPEC)
#error "select one sm90 attention specialization"
#endif

extern "C" __device__ void PIPELINK_SM90_SYMBOL(empty_role_empty)(
    void const* raw_params, unsigned variant) {
  using namespace pipelink::attention::sm90;
  int local_warp = cutlass::canonical_warp_idx_sync() - pipelink_attention_warp_empty();
  if (variant != 0 || local_warp < 0 || local_warp >= 2) asm volatile("trap;");
  extern __shared__ char storage[];
  SplitContext<Config> context(storage, Role::Empty);
  context.initialize();
  (void)raw_params;
}

#if defined(PIPELINK_EMIT)
using namespace pipelink::attention;
using namespace pipelink::attention::sm90;
using Facts = sm90::AnnotationFacts<Config>;
using Pipelines = Config::PipelineStorage;
using QStorage = Config::PipelineQ::SharedStorage;
using KVStorage = Config::PipelineKV::SharedStorage;
using StageParams = EmptyParams;
PL_STAGE("org.pipelink.attention." PIPELINK_SM90_SPEC ".empty", "1.0.0+pipelink.attention");
PL_PARAMS(StageParams, reserved);
#define SM90_ROLE(name, fn, warps) PL_ROLE(name, fn, warps)
SM90_ROLE(empty, PIPELINK_SM90_SYMBOL(empty_role_empty), 2);
#undef SM90_ROLE
PL_CLAIM_EX(pl::claim("pipelink_attention_warp_empty", "warp").extent(2).align(1).at(2).roles({"empty"}));
PL_CLAIM_REGS("pipelink_attention_regs_aux_wg", "aux_wg", {40}, "empty");
PL_CLAIM("pipelink_attention_smem_base", "smem", Facts::shared_bytes, Facts::shared_align, "kernel");
PL_CLAIM("pipelink_attention_mbarrier_base", "mbarrier", Facts::pipeline_slots, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_q_full", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_q_empty", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_k_full", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_k_empty", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_v_full", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_v_empty", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_work_core_full", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_work_core_empty", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_work_epilogue_full", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_work_epilogue_empty", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_kv_index_full", "mbarrier", 4, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_kv_index_empty", "mbarrier", 4, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_corrected_o_full", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_corrected_o_empty", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_final_stats_full", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_final_stats_empty", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_o_drained_full", "mbarrier", 1, 1, "kernel");
PL_TOPOLOGY(pl::topology{1, pl::cluster{{1, 1, 1}, pl::launch{1, {1, 1, 1}}}, true, "static", {}});

#endif
