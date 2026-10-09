#include "mk_abi.cuh"
#include "attention_reduction.cu"

using mk_op_attn_reduction_t = attention_reduction<mk_config, mk_globals>;

MK_DEFINE_OP_EXPORTS(attn_reduction, 3, mk_op_attn_reduction_t,
    mk_op_attn_reduction_t::SHARED_DATA_PAGE + 1,
    mk_op_attn_reduction_t::Final_O_ready_sem_idx(Q_HEADS_PER_INSTRUCTION - 1) + 1,
    0, 0, false)
