#include "fa3_family.cuh"
#include <pipelink/stage.hpp>

#if !defined(PIPELINK_SM90_SPEC)
#error "select one sm90 attention specialization"
#endif

extern "C" __device__ bool PIPELINK_SM90_SYMBOL(mask_phi_mask_mod)(
    int32_t q_pos, int32_t kv_pos, pipelink_attention_sm90_work_desc const& desc) {
  // Model specialization: block=64, local=16, stride=8, head step=1.
  if (kv_pos < 0 || kv_pos > q_pos) return false;
  int const qb = q_pos / 64, kb = kv_pos / 64;
  return qb - kb < 16 || (kb + desc.kv_head + 1) % 8 == 0;
}

#if defined(PIPELINK_EMIT)
PL_STAGE("org.pipelink.attention." PIPELINK_SM90_SPEC ".mask-phi", "1.0.0+pipelink.attention");
PL_NO_PARAMS();
PIPELINK_SM90_WORK_DESC_PL_TYPE;
PL_CALL(mask_mod, PIPELINK_SM90_SYMBOL(mask_phi_mask_mod), pl::consume, "mask_mod");
PL_TOPOLOGY(pl::topology{1, pl::cluster{{1, 1, 1}, pl::launch{1, {1, 1, 1}}}, true, "static", {}});
#endif
