#include "mk_abi.cuh"
#include "attention_partial.cu"

using mk_op_attn_partial_t = attention_partial<mk_config, mk_globals>;

// Q/O/L plus arrived/finished K/V semaphores per pipeline stage.
MK_DEFINE_OP_EXPORTS(attn_partial, 2, mk_op_attn_partial_t,
    mk_op_attn_partial_t::KV_PAGE + 1,
    3 + 4 * mk_op_attn_partial_t::NUM_STAGES, 0, 0, false)
