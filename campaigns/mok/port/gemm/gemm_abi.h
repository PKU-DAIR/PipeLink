#ifndef MOK_GEMM_ABI_H
#define MOK_GEMM_ABI_H
#include "mok_abi.h"

#ifndef MOKGEMMTASK_DEFINED
#define MOKGEMMTASK_DEFINED
typedef struct MokGemmTask {
  int32_t op, num_tokens, macrobatch_size, minibatch_size, macrobatch_idx,
      minibatch_idx, task_idx, cta_rank, input_minibatch_ready_num_cols,
      input_row_block_ready_base_index, input_row_block_ready_required_count,
      output_tile_ready_base_index, buffer_done_index;
  uint32_t flags;
} MokGemmTask;
#endif
#define MOK_GEMM_TASK_PL_TYPE PL_TYPE(MokGemmTask, \
  int32_t op, num_tokens, macrobatch_size, minibatch_size, macrobatch_idx, \
      minibatch_idx, task_idx, cta_rank, input_minibatch_ready_num_cols, \
      input_row_block_ready_base_index, input_row_block_ready_required_count, \
      output_tile_ready_base_index, buffer_done_index; uint32_t flags;)

enum {
  MOK_GEMM_FWD_SHARED_GATE,
  MOK_GEMM_FWD_SHARED_UP,
  MOK_GEMM_FWD_SHARED_DOWN,
  MOK_GEMM_FWD_ROUTED_GATE,
  MOK_GEMM_FWD_ROUTED_UP,
  MOK_GEMM_FWD_ROUTED_DOWN,
  MOK_GEMM_BWD_SHARED_DGRAD_DOWN,
  MOK_GEMM_BWD_SHARED_DGRAD_GATE_UP,
  MOK_GEMM_BWD_SHARED_WGRAD_DOWN,
  MOK_GEMM_BWD_SHARED_WGRAD_GATE,
  MOK_GEMM_BWD_SHARED_WGRAD_UP,
  MOK_GEMM_BWD_REPLAY_GATE,
  MOK_GEMM_BWD_REPLAY_UP,
  MOK_GEMM_BWD_ROUTED_DGRAD_DOWN,
  MOK_GEMM_BWD_ROUTED_DGRAD_GATE_UP,
  MOK_GEMM_BWD_ROUTED_WGRAD_DOWN,
  MOK_GEMM_BWD_ROUTED_WGRAD_GATE,
  MOK_GEMM_BWD_ROUTED_WGRAD_UP,
  MOK_GEMM_REC_SHARED_GATE,
  MOK_GEMM_REC_SHARED_UP,
  MOK_GEMM_REC_ROUTED_GATE,
  MOK_GEMM_REC_ROUTED_UP
};
enum {
  MOK_GEMM_OUTPUT_ROW_READY = 1u << 0,
  MOK_GEMM_BUFFER_DONE = 1u << 1,
  MOK_GEMM_REPLAYED = 1u << 2
};

#ifdef __CUDACC__
extern "C" __device__ __inline_hint__ void mok_gemm_init(const void* view);
extern "C" __device__ __inline_hint__ void mok_gemm_fini(const void* view);
extern "C" __device__ __inline_hint__ void mok_gemm_fwd(const void* view, const MokGemmTask*, const MokCounters*, uint32_t* phase);
extern "C" __device__ __inline_hint__ void mok_gemm_bwd(const void* view, const MokGemmTask*, const MokCounters*, uint32_t* phase);
extern "C" __device__ __inline_hint__ void mok_gemm_rec(const void* view, const MokGemmTask*, const MokCounters*, uint32_t* phase);
#endif
#endif
