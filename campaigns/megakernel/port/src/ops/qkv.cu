#include "mk_abi.cuh"
#include "rms_matvec_rope_append.cu"

using mk_op_qkv_t = rms_qkv_rope_append<mk_config, mk_globals>;

// RoPE occupies the end of scratch; its semaphore follows the RMS pipeline.
MK_DEFINE_OP_EXPORTS(qkv, 1, mk_op_qkv_t,
    mk_op_qkv_t::pipeline::WEIGHTS_START_PAGE +
        mk_op_qkv_t::pipeline::INPUT_PIPELINE_STAGES * mk_op_qkv_t::pipeline::STAGE_PAGES,
    mk_op_qkv_t::pipeline::SEM_COUNT + 1, mk_config::SCRATCH_BYTES, 3, false)
