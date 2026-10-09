/* Public whole-CTA comm call ABI. Owner-private descriptors stay in comm.cu. */
#ifndef MOK_COMM_ABI_H
#define MOK_COMM_ABI_H
#include "mok_abi.h"

#ifndef MOKCOMMTASK_DEFINED
#define MOKCOMMTASK_DEFINED
typedef struct MokCommTask {
    int32_t op, num_tokens, macrobatch_size, minibatch_size, macrobatch_idx, task_idx,
        row_divisor, previous_macrobatch_idx, buffer_ready_required_count;
    uint32_t flags;
    int32_t num_workers;
} MokCommTask;
#endif
#define MOK_COMM_TASK_PL_TYPE PL_TYPE(MokCommTask, \
    int32_t op, num_tokens, macrobatch_size, minibatch_size, macrobatch_idx, task_idx, \
        row_divisor, previous_macrobatch_idx, buffer_ready_required_count; uint32_t flags; \
    int32_t num_workers;)

enum {
    MOK_COMM_FWD_DISPATCH = 0, MOK_COMM_FWD_COMBINE = 1,
    MOK_COMM_BWD_REVERSE_COMBINE = 2, MOK_COMM_BWD_REVERSE_DISPATCH = 3,
    MOK_COMM_BWD_REPLAY_DISPATCH = 4, MOK_COMM_BWD_PRELOAD_ROUTER_WEIGHTS = 5,
    MOK_COMM_REC_DISPATCH = 6
};
enum {
    MOK_COMM_TRANSFER_READY = 1u << 0, MOK_COMM_TRANSFER_DONE = 1u << 1,
    MOK_COMM_BUFFER_READY = 1u << 2
};

#ifdef __CUDACC__
extern "C" __device__ __inline_hint__ void mok_comm_init(const void* view);
extern "C" __device__ __inline_hint__ void mok_comm_fwd(const void* view, const MokCommTask*, const MokCounters*, uint32_t* phase);
extern "C" __device__ __inline_hint__ void mok_comm_bwd(const void* view, const MokCommTask*, const MokCounters*, uint32_t* phase);
extern "C" __device__ __inline_hint__ void mok_comm_rec(const void* view, const MokCommTask*, const MokCounters*, uint32_t* phase);
#endif
#endif
