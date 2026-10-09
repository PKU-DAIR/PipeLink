#include "mk_abi.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"
#include <upgate.cu>

using mk_op_upgate_t = rms_upgate_silu<mk_config, mk_globals>;

// RMS partials follow the output stages (utils.cuh::rms_norm).
MK_DEFINE_OP_EXPORTS(upgate, 5, mk_op_upgate_t,
    mk_op_upgate_t::pipeline::WEIGHTS_START_PAGE +
        mk_op_upgate_t::pipeline::INPUT_PIPELINE_STAGES * mk_op_upgate_t::pipeline::STAGE_PAGES,
    mk_op_upgate_t::pipeline::SEM_COUNT,
    mk_op_upgate_t::pipeline::USED_SCRATCH_BYTES + mk_config::NUM_CONSUMER_WARPS * sizeof(float),
    3, false)
