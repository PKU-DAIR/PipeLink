#include "split_context.cuh"
#include "role_load.cuh"
#include <pipelink/stage.hpp>

#if !defined(PIPELINK_SM90_SPEC)
#error "select one sm90 attention specialization"
#endif

extern "C" __device__ void PIPELINK_SM90_SYMBOL(tma_load_role_load)(
    void const* raw_params, unsigned variant) {
  using namespace pipelink::attention::sm90;
  int local_warp = cutlass::canonical_warp_idx_sync() - pipelink_attention_warp_load();
  if (variant != 0 || local_warp < 0 || local_warp >= 1) asm volatile("trap;");
  extern __shared__ char storage[];
  SplitContext<Config> context(storage, Role::Load);
  run_load<Config>(*reinterpret_cast<LoadParams<Config::causal> const*>(raw_params), context);
}

#if defined(PIPELINK_EMIT)
using namespace pipelink::attention;
using namespace pipelink::attention::sm90;
using Facts = sm90::AnnotationFacts<Config>;
using Pipelines = Config::PipelineStorage;
using QStorage = Config::PipelineQ::SharedStorage;
using KVStorage = Config::PipelineKV::SharedStorage;
using StageParams = LoadParams<Config::causal>;
PL_STAGE("org.pipelink.attention." PIPELINK_SM90_SPEC ".tma-load", "1.0.0+pipelink.attention");
PL_PARAMS(StageParams, tma_load_Q, tma_load_K, tma_load_V, shape_Q, shape_K, stride_V, scheduler, seqlen_q, seqlen_k, qhead_per_khead, causal, window_size, score_param, score_param_inv, needs_score_mod);
#define SM90_ROLE(name, fn, warps) PL_ROLE(name, fn, warps)
SM90_ROLE(load, PIPELINK_SM90_SYMBOL(tma_load_role_load), 1);
#undef SM90_ROLE
PL_CLAIM_EX(pl::claim("pipelink_attention_warp_load", "warp").extent(1).align(1).at(0).roles({"load"}));
PL_CLAIM_REGS("pipelink_attention_regs_aux_wg", "aux_wg", {40}, "load");
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

PL_ENDPOINT(q, pl::endpoint("q", pl::produce, "load")
    .space("smem").scope("cta").proxy("async")
    .tile("bf16", {128, 64, 2})
    .layout(pl::affine{{128, 64, 2}, {64, 1, 8192}, 16}.swizzle(3, 4, 3, "byte").origin_bits(8 * Facts::q_offset))
    .protocol(pl::mbarrier{
        pl::side{"pipelink_attention_slots_q_full", {"load", 1}, Facts::q_bytes}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, q) + offsetof(QStorage, full_barrier_), 8}),
        pl::side{"pipelink_attention_slots_q_empty", {"core", 2}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, q) + offsetof(QStorage, empty_barrier_), 8}),
        {"load", 0, "stage", 0}})
    .ring(pl::fixed(1), Facts::q_bytes)
    .bind("base", "pipelink_attention_smem_base")
    .bind("full", "pipelink_attention_slots_q_full")
    .bind("empty", "pipelink_attention_slots_q_empty"));
PL_ENDPOINT(k, pl::endpoint("k", pl::produce, "load")
    .space("smem").scope("cta").proxy("async")
    .tile("bf16", {Config::kBlockN, 64, 2})
    .layout(pl::affine{{Config::kBlockN, 64, 2}, {64, 1, Config::kBlockN * 64}, 16}.swizzle(3, 4, 3, "byte").origin_bits(8 * Facts::k_offset))
    .protocol(pl::mbarrier{
        pl::side{"pipelink_attention_slots_k_full", {"load", 1}, Facts::kv_bytes}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, k) + offsetof(KVStorage, full_barrier_), 8}),
        pl::side{"pipelink_attention_slots_k_empty", {"core", 2}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, k) + offsetof(KVStorage, empty_barrier_), 8}),
        {"load", 0, "stage", 0}})
    .ring(pl::fixed(2), Facts::kv_bytes)
    .bind("base", "pipelink_attention_smem_base")
    .bind("full", "pipelink_attention_slots_k_full")
    .bind("empty", "pipelink_attention_slots_k_empty"));
PL_ENDPOINT(v, pl::endpoint("v", pl::produce, "load")
    .space("smem").scope("cta").proxy("async")
    .tile("bf16", {Config::kBlockN, 64, 2})
    .layout(pl::affine{{Config::kBlockN, 64, 2}, {64, 1, Config::kBlockN * 64}, 16}.swizzle(3, 4, 3, "byte").origin_bits(8 * Facts::v_offset))
    .protocol(pl::mbarrier{
        pl::side{"pipelink_attention_slots_v_full", {"load", 1}, Facts::kv_bytes}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, v) + offsetof(KVStorage, full_barrier_), 8}),
        pl::side{"pipelink_attention_slots_v_empty", {"core", 2}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, v) + offsetof(KVStorage, empty_barrier_), 8}),
        {"load", 0, "stage", 0}})
    .ring(pl::fixed(2), Facts::kv_bytes)
    .bind("base", "pipelink_attention_smem_base")
    .bind("full", "pipelink_attention_slots_v_full")
    .bind("empty", "pipelink_attention_slots_v_empty"));
PL_ENDPOINT(work_core, pl::endpoint("work_core", pl::produce, "load")
    .space("smem").scope("cta").proxy("generic")
    .tile("s32", {sizeof(sm90::WorkDesc) / sizeof(int32_t)})
    .layout(pl::affine{{sizeof(sm90::WorkDesc) / sizeof(int32_t)}, {1}, 32}.origin_bits(8 * Facts::work_core_offset))
    .protocol(pl::mbarrier{
        pl::side{"pipelink_attention_slots_work_core_full", {"load", 1}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, work_core) + offsetof(RingBarriers<2>, full), 8}),
        pl::side{"pipelink_attention_slots_work_core_empty", {"core", 8}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, work_core) + offsetof(RingBarriers<2>, empty), 8}),
        {"load", 0, "stage", 0}})
    .ring(pl::fixed(2), sizeof(sm90::WorkDesc))
    .bind("base", "pipelink_attention_smem_base")
    .bind("full", "pipelink_attention_slots_work_core_full")
    .bind("empty", "pipelink_attention_slots_work_core_empty"));
PL_ENDPOINT(work_epilogue, pl::endpoint("work_epilogue", pl::produce, "load")
    .space("smem").scope("cta").proxy("generic")
    .tile("s32", {sizeof(sm90::WorkDesc) / sizeof(int32_t)})
    .layout(pl::affine{{sizeof(sm90::WorkDesc) / sizeof(int32_t)}, {1}, 32}.origin_bits(8 * Facts::work_epilogue_offset))
    .protocol(pl::mbarrier{
        pl::side{"pipelink_attention_slots_work_epilogue_full", {"load", 1}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, work_epilogue) + offsetof(RingBarriers<2>, full), 8}),
        pl::side{"pipelink_attention_slots_work_epilogue_empty", {"epilogue", 32}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, work_epilogue) + offsetof(RingBarriers<2>, empty), 8}),
        {"load", 0, "stage", 0}})
    .ring(pl::fixed(2), sizeof(sm90::WorkDesc))
    .bind("base", "pipelink_attention_smem_base")
    .bind("full", "pipelink_attention_slots_work_epilogue_full")
    .bind("empty", "pipelink_attention_slots_work_epilogue_empty"));
PL_ENDPOINT(kv_index, pl::endpoint("kv_index", pl::produce, "load")
    .space("smem").scope("cta").proxy("generic")
    .tile("s32", {sizeof(Config::KvSlotMeta) / 4})
    .layout(pl::affine{{sizeof(Config::KvSlotMeta) / 4}, {1}, 32}.origin_bits(8 * Facts::kv_index_offset))
    .protocol(pl::mbarrier{
        pl::side{"pipelink_attention_slots_kv_index_full", {"load", 1}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, kv_index) + offsetof(RingBarriers<4>, full), 8}),
        pl::side{"pipelink_attention_slots_kv_index_empty", {"core", 8}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, kv_index) + offsetof(RingBarriers<4>, empty), 8}),
        {"load", 0, "stage", 0}})
    .ring(pl::fixed(4), sizeof(Config::KvSlotMeta))
    .bind("base", "pipelink_attention_smem_base")
    .bind("full", "pipelink_attention_slots_kv_index_full")
    .bind("empty", "pipelink_attention_slots_kv_index_empty"));
PL_ENDPOINT(o_drained, pl::signal_endpoint("o_drained", pl::consume, "load")
    .scope("cta")
    .protocol(pl::mbarrier{
        pl::side{"pipelink_attention_slots_o_drained_full", {"epilogue", 1}, 0}
            .native({"pipelink_attention_mbarrier_base", offsetof(Pipelines, o_drained), 8}),
        std::nullopt, {"load", 0, "stage", 0}, "external"})
    .ring(pl::fixed(1), 0)
    .bind("full", "pipelink_attention_slots_o_drained_full"));
#endif
