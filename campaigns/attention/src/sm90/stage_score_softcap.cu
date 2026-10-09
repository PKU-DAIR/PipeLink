#include "fa3_family.cuh"
#include <cutlass/fast_math.h>
#include <pipelink/stage.hpp>

#if !defined(PIPELINK_SM90_SPEC)
#error "select one sm90 attention specialization"
#endif

extern "C" __device__ float PIPELINK_SM90_SYMBOL(score_softcap_score_mod)(
    float s, pipelink_attention_sm90_work_desc const& desc) {
  return desc.score_param * cutlass::fast_tanh(s * desc.score_param_inv);
}

#if defined(PIPELINK_EMIT)
PL_STAGE("org.pipelink.attention." PIPELINK_SM90_SPEC ".score-softcap", "1.0.0+pipelink.attention");
PL_NO_PARAMS();
PL_TYPE(pipelink_attention_sm90_work_desc,
    int32_t q_block; int32_t batch; int32_t kv_head; int32_t kv_steps;
    int32_t seqlen_q; int32_t seqlen_k; int32_t q_causal_offset;
    int32_t split_idx; uint32_t flags; uint32_t qhead_per_kvhead;
    int32_t mask_param; float score_param; float score_param_inv;);
#define SM90_EXPORT(name, fn) PL_EXPORT(name, fn)
#define SM90_CALL(name, fn, ...) PL_CALL(name, fn, __VA_ARGS__)
SM90_EXPORT(score_mod, PIPELINK_SM90_SYMBOL(score_softcap_score_mod));
SM90_CALL(score_mod, PIPELINK_SM90_SYMBOL(score_softcap_score_mod), pl::consume, "score_mod");
#undef SM90_EXPORT
#undef SM90_CALL
PL_TOPOLOGY(pl::topology{1, pl::cluster{{1, 1, 1}, pl::launch{1, {1, 1, 1}}}, true, "static", {}});
#endif
