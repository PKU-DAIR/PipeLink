#ifndef MOK_ACT_ABI_H
#define MOK_ACT_ABI_H

#include "mok_abi.h"

#ifndef MOKACTTASK_DEFINED
#define MOKACTTASK_DEFINED
typedef struct MokActTask {
    int32_t op, num_tokens, macrobatch_size, minibatch_size, macrobatch_idx,
        minibatch_idx, task_idx, cta_rank, gate_up_tile_ready_base_index,
        hidden_row_block_ready_base_index;
    uint32_t flags;
    int32_t d_hidden_tile_ready_base_index, replayed_gate_up_tile_ready_base_index,
        d_gate_up_row_block_ready_base_index, buffer_done_index;
} MokActTask;
#endif
#define MOK_ACT_TASK_PL_TYPE PL_TYPE(MokActTask, \
    int32_t op, num_tokens, macrobatch_size, minibatch_size, macrobatch_idx, \
        minibatch_idx, task_idx, cta_rank, gate_up_tile_ready_base_index, \
        hidden_row_block_ready_base_index; uint32_t flags; \
    int32_t d_hidden_tile_ready_base_index, replayed_gate_up_tile_ready_base_index, \
        d_gate_up_row_block_ready_base_index, buffer_done_index;)

enum {
    MOK_ACT_FWD_SHARED = 0, MOK_ACT_FWD_ROUTED = 1,
    MOK_ACT_BWD_SHARED = 2, MOK_ACT_BWD_ROUTED = 3,
    MOK_ACT_BWD_REPLAY_ROUTED = 4, MOK_ACT_REC_SHARED = 5, MOK_ACT_REC_ROUTED = 6
};
enum { MOK_ACT_REPLAYED = 1u << 0, MOK_ACT_BUFFER_DONE = 1u << 1 };

#ifdef __CUDACC__
extern "C" __device__ __inline_hint__ void mok_act_init(const void* view);
extern "C" __device__ __inline_hint__ void mok_act_fwd(const void* view, const MokActTask* task,
    const MokCounters* counters, uint32_t* phase);
extern "C" __device__ __inline_hint__ void mok_act_bwd(const void* view, const MokActTask* task,
    const MokCounters* counters, uint32_t* phase);
extern "C" __device__ __inline_hint__ void mok_act_rec(const void* view, const MokActTask* task,
    const MokCounters* counters, uint32_t* phase);
#endif

#endif
