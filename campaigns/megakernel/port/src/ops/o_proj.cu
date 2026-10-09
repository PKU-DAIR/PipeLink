#include "mk_abi.cuh"

using globals = mk_globals;
#include "matvec_adds.cu"

using mk_op_o_proj_t = o_proj<mk_config, mk_globals>;

MK_DEFINE_OP_EXPORTS(o_proj, 4, mk_op_o_proj_t,
    mk_op_o_proj_t::pipeline::WEIGHTS_START_PAGE +
        mk_op_o_proj_t::pipeline::INPUT_PIPELINE_STAGES * mk_op_o_proj_t::pipeline::STAGE_PAGES,
    mk_op_o_proj_t::pipeline::SEM_COUNT, mk_op_o_proj_t::pipeline::USED_SCRATCH_BYTES,
    4, true)
