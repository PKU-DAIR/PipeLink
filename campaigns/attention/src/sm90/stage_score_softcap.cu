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
PIPELINK_SM90_WORK_DESC_PL_TYPE;
PL_CALL(score_mod, PIPELINK_SM90_SYMBOL(score_softcap_score_mod), pl::consume, "score_mod");
PL_TOPOLOGY(pl::topology{1, pl::cluster{{1, 1, 1}, pl::launch{1, {1, 1, 1}}}, true, "static", {}});
#endif
