#ifndef MOK_FRAMEWORK_ABI_H
#define MOK_FRAMEWORK_ABI_H
#include <stdint.h>
#include <pipelink_host_abi_v2.h>
#include "mok_abi.h"

/* Framework-private provider residue; dimensions and foreign geometry are host inputs.
 * P2 keeps this common layout across the three interpreter/provider contracts. */
typedef struct MokFrameworkResidue {
    int32_t num_local_tokens, hidden, intermediate, num_local_experts;
    int32_t schedule_capacity, macrobatch_size, minibatch_size, topk, num_comm_sms;
    int32_t DISPATCH_Mb, DISPATCH_Nb, COMBINE_Mb, COMBINE_Nb, COMBINE_PIPE_DEPTH;
    int32_t MLP_Mb, MLP_Nb, SWIGLU_Mb, SWIGLU_Nb;
    int32_t SWIGLU_FWD_PIPE_DEPTH, SWIGLU_BWD_PIPE_DEPTH;
    int32_t* num_tokens;
    int32_t* counters;
} MokFrameworkResidue;
typedef struct plk_role_view_mok_interp_fwd_worker {
    const void* residue;
    const plk_shared_scalars_v2* scalars;
} plk_role_view_mok_interp_fwd_worker;
typedef struct plk_role_view_mok_interp_bwd_worker {
    const void* residue;
    const plk_shared_scalars_v2* scalars;
} plk_role_view_mok_interp_bwd_worker;
typedef struct plk_role_view_mok_interp_rec_worker {
    const void* residue;
    const plk_shared_scalars_v2* scalars;
} plk_role_view_mok_interp_rec_worker;
#endif
