#include "fa3_family.cuh"
#include <pipelink/stage.hpp>

#if !defined(PIPELINK_SM90_SPEC)
#error "select one sm90 attention specialization"
#endif

extern "C" __device__ bool PIPELINK_SM90_SYMBOL(mask_default_mask_mod)(
    int32_t, int32_t, pipelink_attention_sm90_work_desc const&) {
  return true;
}

#if defined(PIPELINK_EMIT)
PL_STAGE("org.pipelink.attention." PIPELINK_SM90_SPEC ".mask-default", "1.0.0+pipelink.attention");
PL_NO_PARAMS();
PIPELINK_SM90_WORK_DESC_PL_TYPE;
PL_CALL(mask_mod, PIPELINK_SM90_SYMBOL(mask_default_mask_mod), pl::consume, "mask_mod");
PL_TOPOLOGY(pl::topology{1, pl::cluster{{1, 1, 1}, pl::launch{1, {1, 1, 1}}}, true, "static", {}});
#endif
