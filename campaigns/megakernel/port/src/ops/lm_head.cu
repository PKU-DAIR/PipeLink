#include "mk_abi.cuh"
#include "rms_lm_head.cu"

using mk_op_lm_head_t = rms_lm_head<mk_config, mk_globals>;

// RMS partials follow the output stages (utils.cuh::rms_norm).
MK_DEFINE_OP_EXPORTS(lm_head, 7, mk_op_lm_head_t,
    mk_op_lm_head_t::pipeline::WEIGHTS_START_PAGE +
        mk_op_lm_head_t::pipeline::INPUT_PIPELINE_STAGES * mk_op_lm_head_t::pipeline::STAGE_PAGES,
    mk_op_lm_head_t::pipeline::SEM_COUNT,
    mk_op_lm_head_t::pipeline::USED_SCRATCH_BYTES + mk_config::NUM_CONSUMER_WARPS * sizeof(float),
    3, false)
