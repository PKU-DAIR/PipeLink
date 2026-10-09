#include "mk_abi.cuh"

using mk_op_noop_t = megakernel::NoOp<mk_config>;

MK_DEFINE_OP_EXPORTS(noop, 0, mk_op_noop_t, 0, 0, 0, 0, false)
