// MoK f3a8bd2 csrc/megakernel/backward.cuh:1-443.
// Only P1 owner-call/residue/init substitutions; call argument scalars follow :133-171,228-403.
#include "framework_abi.h"
#include "comm_abi.h"
#include "gemm_abi.h"
#include "act_abi.h"
#include <pipelink/stage.hpp>
#include "mok_port.cuh"
using namespace kittens;
MOK_COUNTERS_PL_TYPE;
MOK_COMM_TASK_PL_TYPE;
MOK_GEMM_TASK_PL_TYPE;
MOK_ACT_TASK_PL_TYPE;
extern "C" __device__ int pipelink_mok_fw_clc();
extern "C" __device__ int pipelink_mok_fw_sems();
extern "C" __device__ int pipelink_mok_precision();
extern "C" __device__ int pipelink_mok_clamp();
extern "C" __device__ void pipelink_call_comm_init(const void* view);
extern "C" __device__ void pipelink_call_gemm_init(const void* view);
extern "C" __device__ void pipelink_call_gemm_fini(const void* view);
extern "C" __device__ void pipelink_call_act_init(const void* view);
extern "C" __device__ void pipelink_call_comm_bwd(const void* view, const MokCommTask*, const MokCounters*, uint32_t*);
extern "C" __device__ void pipelink_call_gemm_bwd(const void* view, const MokGemmTask*, const MokCounters*, uint32_t*);
extern "C" __device__ void pipelink_call_act_bwd(const void* view, const MokActTask*, const MokCounters*, uint32_t*);
struct MokFrameworkClc { clc::handle handle[1], drain[8]; };
struct MokFrameworkSems { semaphore schedule_arrived[1], schedule_finished[1], drain_arrived[8], drain_finished[8]; };
static_assert(sizeof(MokFrameworkClc) == 144 && sizeof(MokFrameworkSems) == 144);
template <mok_port::RoutedPrecision PRECISION, bool IS_CLAMPED>
static __device__ __forceinline__ void mok_framework_run(const void* opaque, const MokFrameworkResidue &g) {
    // Task counts and live targets: upstream backward.cuh:3-61,209; no host capacity substitution.
    const int num_local_experts = g.num_local_experts;
    const int intermediate_dim_col_blocks = g.intermediate / g.MLP_Nb;
    const int hidden_dim_col_blocks = g.hidden / g.MLP_Nb;

    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();

    const int shared_row_blocks = g.num_local_tokens / g.MLP_Mb;
    const int shared_dgrad_down_tasks = shared_row_blocks * intermediate_dim_col_blocks;
    const int shared_swiglu_bwd_tiles = (g.num_local_tokens / g.SWIGLU_Mb) * (g.intermediate / g.SWIGLU_Nb);
    const int shared_swiglu_bwd_tasks = (shared_swiglu_bwd_tiles + 2 * g.SWIGLU_BWD_PIPE_DEPTH - 1) / (2 * g.SWIGLU_BWD_PIPE_DEPTH);
    const int shared_dgrad_gate_up_tasks = shared_row_blocks * hidden_dim_col_blocks;
    const int shared_wgrad_tasks = intermediate_dim_col_blocks * hidden_dim_col_blocks; // per weight matrix
    const int shared_tasks = shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks + 3 * shared_wgrad_tasks;

    const int minibatch_routed_row_blocks = g.minibatch_size / g.MLP_Mb;
    const int minibatch_routed_dgrad_down_tasks = minibatch_routed_row_blocks * intermediate_dim_col_blocks;
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / g.SWIGLU_Mb) * (g.intermediate / g.SWIGLU_Nb);
    const int minibatch_routed_swiglu_bwd_tasks = (minibatch_routed_swiglu_tiles + 2 * g.SWIGLU_BWD_PIPE_DEPTH - 1) / (2 * g.SWIGLU_BWD_PIPE_DEPTH);
    const int minibatch_routed_dgrad_gate_up_tasks = minibatch_routed_row_blocks * hidden_dim_col_blocks;
    const int minibatch_routed_bwd_tasks = minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks + minibatch_routed_dgrad_gate_up_tasks;

    const int wgrad_matrix_tasks = num_local_experts * intermediate_dim_col_blocks * hidden_dim_col_blocks;
    const int wgrad_tasks = 3 * wgrad_matrix_tasks;

    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * intermediate_dim_col_blocks;
    const int minibatch_routed_swiglu_fwd_tasks = (minibatch_routed_swiglu_tiles + 2 * g.SWIGLU_FWD_PIPE_DEPTH - 1) / (2 * g.SWIGLU_FWD_PIPE_DEPTH);
    const int minibatch_routed_replay_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_fwd_tasks;

    const int comm_clusters = g.num_comm_sms / 2;
    const int macrobatch_size = g.macrobatch_size;
    const int num_tokens = g.num_tokens[0];
    const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;
    const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;

    auto num_minibatches_of = [&](int macrobatch_idx) { return (min(num_tokens - macrobatch_idx * macrobatch_size, macrobatch_size) + g.minibatch_size - 1) / g.minibatch_size; };
    auto num_dispatch_tasks_of = [&](int macrobatch_idx) {
        const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
        const int col_blocks = (g.hidden + g.DISPATCH_Nb - 1) / g.DISPATCH_Nb;
        return (macrobatch_tokens / g.DISPATCH_Mb) * col_blocks;
    };
    auto num_combine_tasks_of = [&](int macrobatch_idx) {
        const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
        const int col_blocks = (g.hidden + g.COMBINE_Nb - 1) / g.COMBINE_Nb;
        const int tiles = (macrobatch_tokens / g.COMBINE_Mb) * col_blocks;
        return (tiles + g.COMBINE_PIPE_DEPTH - 1) / g.COMBINE_PIPE_DEPTH;
    };
    auto routed_buffers_done_required_count_of = [&](int macrobatch_idx) {
        return 2 * (num_minibatches_of(macrobatch_idx) * minibatch_routed_bwd_tasks + wgrad_tasks) + num_combine_tasks_of(macrobatch_idx);
    };

    const int num_minibatches = (num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int saved_macrobatch_num_minibatches = num_minibatches_of(0);
    const int saved_macrobatch_tasks = saved_macrobatch_num_minibatches * minibatch_routed_bwd_tasks + wgrad_tasks;
    const int replayed_macrobatch_tasks = minibatches_per_macrobatch * (minibatch_routed_replay_tasks + minibatch_routed_bwd_tasks) + wgrad_tasks;
    const int num_replay_minibatches = num_minibatches - saved_macrobatch_num_minibatches;
    const int true_num_clusters = comm_clusters + shared_tasks + num_minibatches * minibatch_routed_bwd_tasks +
                                  num_replay_minibatches * minibatch_routed_replay_tasks + num_macrobatches * wgrad_tasks;
    if (cluster_idx >= true_num_clusters) return;

    const MokCounters counters{g.counters};
    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_fwd_bitfield = 0xFFFF0000;
    uint32_t swiglu_bwd_bitfield = 0xFFFF0000;
    uint32_t dispatch_bitfield = 0xFFFF0000;
    uint32_t combine_bitfield = 0xFFFF0000;

    auto& clc_state = *reinterpret_cast<MokFrameworkClc*>(mok_port::smem_claim_address(pipelink_mok_fw_clc()));
    auto& sems = *reinterpret_cast<MokFrameworkSems*>(mok_port::smem_claim_address(pipelink_mok_fw_sems()));
    auto& clc_handle = clc_state.handle;
    auto& clc_drain_handle = clc_state.drain;
    auto& schedule_arrived = sems.schedule_arrived;
    auto& schedule_finished = sems.schedule_finished;
    auto& drain_schedule_arrived = sems.drain_arrived;
    auto& drain_schedule_finished = sems.drain_finished;
    if (threadIdx.x == 0) {
        init_semaphore(schedule_arrived[0], 0, 1);
        init_semaphore(schedule_finished[0], 0, 2 * 8);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            init_semaphore(drain_schedule_arrived[i], 0, 1);
            init_semaphore(drain_schedule_finished[i], 0, 2);
        }
    }
    pipelink_call_comm_init(opaque);
    pipelink_call_gemm_init(opaque);
    pipelink_call_act_init(opaque);
    everyone::tma::cluster::sync();

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * 2 + cta_rank;
        auto reverse_combine = [&](int macrobatch_idx, int task_idx) {
            const MokCommTask task{MOK_COMM_BWD_REVERSE_COMBINE, num_tokens, macrobatch_size,
                g.minibatch_size, macrobatch_idx, task_idx, g.topk, -1, 0, 0u, 0};
            pipelink_call_comm_bwd(opaque, &task, &counters, &dispatch_bitfield);
        };
        auto reverse_dispatch = [&](int macrobatch_idx, int task_idx) {
            const MokCommTask task{MOK_COMM_BWD_REVERSE_DISPATCH, num_tokens, macrobatch_size,
                g.minibatch_size, macrobatch_idx, task_idx, 0, 0, 0,
                num_macrobatches > 1 ? uint32_t(MOK_COMM_TRANSFER_DONE) : 0u, 0};
            pipelink_call_comm_bwd(opaque, &task, &counters, &combine_bitfield);
        };
        auto replay_dispatch = [&](int macrobatch_idx, int task_idx) {
            const MokCommTask task{MOK_COMM_BWD_REPLAY_DISPATCH, num_tokens, macrobatch_size,
                g.minibatch_size, macrobatch_idx, task_idx, g.topk, -1, 0, 0u, 0};
            pipelink_call_comm_bwd(opaque, &task, &counters, &dispatch_bitfield);
        };
        const MokCommTask preload{MOK_COMM_BWD_PRELOAD_ROUTER_WEIGHTS, num_tokens, macrobatch_size,
            0, 0, comm_cta_idx, 0, -1, 0, 0u, g.num_comm_sms};
        pipelink_call_comm_bwd(opaque, &preload, &counters, &dispatch_bitfield);
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks_of(0); task_idx += g.num_comm_sms)
            reverse_combine(0, task_idx);
        for (int macrobatch_idx = 0; macrobatch_idx < num_macrobatches; ++macrobatch_idx) {
            // All reverse-dispatch tasks must complete before this CTA moves on: the next macrobatch's pulls
            // wait on routed_buffers_done, which counts every rank's reverse-dispatch arrivals (including this CTA's)
            for (int task_idx = comm_cta_idx; task_idx < num_combine_tasks_of(macrobatch_idx); task_idx += g.num_comm_sms)
                reverse_dispatch(macrobatch_idx, task_idx);
            if (macrobatch_idx + 1 < num_macrobatches) {
                const MokCommTask preload{MOK_COMM_BWD_PRELOAD_ROUTER_WEIGHTS, num_tokens, macrobatch_size,
                    0, macrobatch_idx + 1, comm_cta_idx, 0, macrobatch_idx,
                    routed_buffers_done_required_count_of(macrobatch_idx), MOK_COMM_BUFFER_READY, g.num_comm_sms};
                pipelink_call_comm_bwd(opaque, &preload, &counters, &dispatch_bitfield);
                for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks_of(macrobatch_idx + 1); task_idx += g.num_comm_sms) {
                    reverse_combine(macrobatch_idx + 1, task_idx);
                    replay_dispatch(macrobatch_idx + 1, task_idx);
                }
            }
        }
        pipelink_call_gemm_fini(opaque);
        return;
    }

    // Swiglu (forward and backward) tasks are CTA-local, GEMM is not
    auto is_cta_local_task = [&](int compute_cluster_idx) {
        if (compute_cluster_idx < 0) return false;
        else if (compute_cluster_idx < shared_dgrad_down_tasks) return false; // shared dgrad down
        else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks) return true; // shared swiglu bwd
        else if (compute_cluster_idx < shared_tasks) return false; // shared dgrad/wgrad
        else if (compute_cluster_idx >= true_num_clusters - comm_clusters) return false;

        int idx = compute_cluster_idx - shared_tasks;
        int macrobatch_num_minibatches, macrobatch_task_idx;
        if (idx < saved_macrobatch_tasks) {
            macrobatch_num_minibatches = saved_macrobatch_num_minibatches;
            macrobatch_task_idx = idx;
        } else {
            idx -= saved_macrobatch_tasks;
            const int macrobatch_idx = 1 + idx / replayed_macrobatch_tasks;
            macrobatch_num_minibatches = num_minibatches_of(macrobatch_idx);
            macrobatch_task_idx = idx % replayed_macrobatch_tasks;
            if (macrobatch_task_idx < macrobatch_num_minibatches * minibatch_routed_replay_tasks)
                return macrobatch_task_idx % minibatch_routed_replay_tasks >= 2 * minibatch_routed_gate_up_tasks; // swiglu fwd replay
            macrobatch_task_idx -= macrobatch_num_minibatches * minibatch_routed_replay_tasks;
        }
        if (macrobatch_task_idx >= macrobatch_num_minibatches * minibatch_routed_bwd_tasks) return false; // wgrad
        const int minibatch_task_idx = macrobatch_task_idx % minibatch_routed_bwd_tasks;
        return minibatch_task_idx >= minibatch_routed_dgrad_down_tasks &&
               minibatch_task_idx < minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks; // swiglu bwd
    };

    const int d_gate_up_row_block_ready_required_count = (g.MLP_Mb / g.SWIGLU_Mb) * (g.intermediate / g.SWIGLU_Nb);
    const uint32_t buffer_done_flag = num_macrobatches > 1 ? uint32_t(MOK_GEMM_BUFFER_DONE) : 0u;

    for (int task_iter = 0; cluster_idx >= 0 && cluster_idx < true_num_clusters; ++task_iter) {
        const int clc_stage = task_iter % 1;
        if (warpgroup::groupid() == 1 && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + 1) / 1) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int compute_cluster_idx = cluster_idx - comm_clusters;
        const bool current_is_cta_local = is_cta_local_task(compute_cluster_idx);

        if (compute_cluster_idx < shared_dgrad_down_tasks) {
            // Shared dgrad down: d_hidden_shared = d_y_shared @ w_shared_down
            const int task_idx = compute_cluster_idx;
            const MokGemmTask task{MOK_GEMM_BWD_SHARED_DGRAD_DOWN, num_tokens, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, 0, 0, 0, 0u};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks) {
            // Shared Swiglu bwd: d_gate_shared, d_up_shared = swiglu_bwd(d_hidden_shared, gate_shared, up_shared)
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks;
            const MokActTask task{MOK_ACT_BWD_SHARED, g.num_local_tokens, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, 0u, 0, 0, 0, 0};
            pipelink_call_act_bwd(opaque, &task, &counters, &swiglu_bwd_bitfield);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks) {
            // Shared dgrad gate+up: d_x_shared = d_gate_shared @ w_shared_gate + d_up_shared @ w_shared_up
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks;
            const MokGemmTask task{MOK_GEMM_BWD_SHARED_DGRAD_GATE_UP, num_tokens, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, d_gate_up_row_block_ready_required_count, 0, 0, 0u};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks + shared_wgrad_tasks) {
            // Shared wgrad down: d_w_shared_down += d_y_shared^T @ hidden_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks;
            const MokGemmTask task{MOK_GEMM_BWD_SHARED_WGRAD_DOWN, num_tokens, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, 0, 0, 0, 0u};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
        } else if (compute_cluster_idx < shared_tasks - shared_wgrad_tasks) {
            // Shared wgrad gate: d_w_shared_gate += d_gate_shared^T @ x_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks - shared_wgrad_tasks;
            const MokGemmTask task{MOK_GEMM_BWD_SHARED_WGRAD_GATE, num_tokens, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, d_gate_up_row_block_ready_required_count, 0, 0, 0u};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared wgrad up: d_w_shared_up += d_up_shared^T @ x_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks - 2 * shared_wgrad_tasks;
            const MokGemmTask task{MOK_GEMM_BWD_SHARED_WGRAD_UP, num_tokens, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, d_gate_up_row_block_ready_required_count, 0, 0, 0u};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
        } else {
            // Routed / replay tasks
            const int global_routed_task_idx = compute_cluster_idx - shared_tasks;
            const bool replayed = global_routed_task_idx >= saved_macrobatch_tasks;
            const int replayed_task_idx = global_routed_task_idx - saved_macrobatch_tasks;
            const int macrobatch_idx = replayed ? 1 + replayed_task_idx / replayed_macrobatch_tasks : 0;
            const int macrobatch_task_idx = replayed ? replayed_task_idx % replayed_macrobatch_tasks : global_routed_task_idx;
            const int macrobatch_num_minibatches = num_minibatches_of(macrobatch_idx);
            const int num_replay_tasks = replayed ? macrobatch_num_minibatches * minibatch_routed_replay_tasks : 0;

            if (macrobatch_task_idx < num_replay_tasks) {
                const int minibatch_idx = macrobatch_task_idx / minibatch_routed_replay_tasks;
                const int minibatch_task_idx = macrobatch_task_idx % minibatch_routed_replay_tasks;
                if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                    // Replay gate GEMM refreshes the routed activation.
                    const int task_idx = minibatch_task_idx;
            const MokGemmTask task{MOK_GEMM_BWD_REPLAY_GATE, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, minibatch_idx, task_idx, cta_rank, 0, 0, 0, 0, 0, 0u};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
                } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                    // Replay up GEMM refreshes the routed activation.
                    const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
            const MokGemmTask task{MOK_GEMM_BWD_REPLAY_UP, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, minibatch_idx, task_idx, cta_rank, 0, 0, 0, 0, 0, 0u};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
                } else {
                    // Replay Swiglu refreshes the routed hidden activation.
                    const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
            const MokActTask task{MOK_ACT_BWD_REPLAY_ROUTED, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, minibatch_idx, task_idx, cta_rank, 0, 0, 0u, 0, 0, 0, 0};
            pipelink_call_act_bwd(opaque, &task, &counters, &swiglu_fwd_bitfield);
                }
            } else {
                const int num_routed_tasks = macrobatch_num_minibatches * minibatch_routed_bwd_tasks;
                const int routed_task_idx = macrobatch_task_idx - num_replay_tasks;
                const int minibatch_idx = routed_task_idx / minibatch_routed_bwd_tasks;
                const int minibatch_task_idx = routed_task_idx % minibatch_routed_bwd_tasks;
                if (routed_task_idx < num_routed_tasks && minibatch_task_idx < minibatch_routed_dgrad_down_tasks) {
                    // Dgrad down: d_hidden_routed = d_y_routed @ w_routed_down
                    const int task_idx = minibatch_task_idx;
            const MokGemmTask task{MOK_GEMM_BWD_ROUTED_DGRAD_DOWN, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, minibatch_idx, task_idx, cta_rank, 0, 0, 0, shared_dgrad_down_tasks, macrobatch_idx, buffer_done_flag};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
                } else if (routed_task_idx < num_routed_tasks &&
                           minibatch_task_idx < minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks) {
                    // Routed Swiglu backward
                    const int task_idx = minibatch_task_idx - minibatch_routed_dgrad_down_tasks;
            const MokActTask task{MOK_ACT_BWD_ROUTED, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, minibatch_idx, task_idx, cta_rank, 0, 0, (replayed ? uint32_t(MOK_ACT_REPLAYED) : 0u) | (num_macrobatches > 1 ? uint32_t(MOK_ACT_BUFFER_DONE) : 0u), shared_dgrad_down_tasks, 0, shared_row_blocks, macrobatch_idx};
            pipelink_call_act_bwd(opaque, &task, &counters, &swiglu_bwd_bitfield);
                } else if (routed_task_idx < num_routed_tasks) {
                    // Dgrad gate+up: d_x_routed = d_gate @ w_routed_gate + d_up @ w_routed_up
                    const int task_idx = minibatch_task_idx - minibatch_routed_dgrad_down_tasks - minibatch_routed_swiglu_bwd_tasks;
            const MokGemmTask task{MOK_GEMM_BWD_ROUTED_DGRAD_GATE_UP, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, minibatch_idx, task_idx, cta_rank, 0, shared_row_blocks, d_gate_up_row_block_ready_required_count, 0, macrobatch_idx, buffer_done_flag};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
                } else if (routed_task_idx < num_routed_tasks + wgrad_matrix_tasks) {
                    // Wgrad down: d_w_routed_down += d_y_routed^T @ hidden_routed
                    const int task_idx = routed_task_idx - num_routed_tasks;
            const MokGemmTask task{MOK_GEMM_BWD_ROUTED_WGRAD_DOWN, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, 0, task_idx, cta_rank, g.hidden, 0, d_gate_up_row_block_ready_required_count, 0, macrobatch_idx, buffer_done_flag | (replayed ? uint32_t(MOK_GEMM_REPLAYED) : 0u)};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
                } else if (routed_task_idx < num_routed_tasks + 2 * wgrad_matrix_tasks) {
                    // Wgrad gate: d_w_routed_gate += d_gate_routed^T @ x_routed
                    const int task_idx = routed_task_idx - num_routed_tasks - wgrad_matrix_tasks;
            const MokGemmTask task{MOK_GEMM_BWD_ROUTED_WGRAD_GATE, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, 0, task_idx, cta_rank, g.hidden, shared_row_blocks, d_gate_up_row_block_ready_required_count, 0, macrobatch_idx, buffer_done_flag | (replayed ? uint32_t(MOK_GEMM_REPLAYED) : 0u)};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
                } else {
                    // Wgrad up: d_w_routed_up += d_up_routed^T @ x_routed
                    const int task_idx = routed_task_idx - num_routed_tasks - 2 * wgrad_matrix_tasks;
            const MokGemmTask task{MOK_GEMM_BWD_ROUTED_WGRAD_UP, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, 0, task_idx, cta_rank, g.hidden, shared_row_blocks, d_gate_up_row_block_ready_required_count, 0, macrobatch_idx, buffer_done_flag | (replayed ? uint32_t(MOK_GEMM_REPLAYED) : 0u)};
            pipelink_call_gemm_bwd(opaque, &task, &counters, &gemm_bitfield);
                }
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / 1) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / 2) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_compute_cluster_idx = cluster_idx - comm_clusters;
        if (current_is_cta_local && cluster_idx >= 0 && !is_cta_local_task(next_compute_cluster_idx))
            everyone::tma::cluster::sync();
    }

    everyone::tma::cluster::sync();

    // CLC drain for no-op threadblocks
    if (cluster_idx >= 0 && warp::laneid() == 0) {
        const int stage = warpid();
        int iter = 0;
        if (cta_rank == 0)
            clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
        tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
        while (true) {
            wait(drain_schedule_arrived[stage], iter % 2);
            const auto schedule = clc::query(clc_drain_handle[stage]);
            warp::tma::cluster::arrive(drain_schedule_finished[stage], 0);
            if (cta_rank == 0)
                wait(drain_schedule_finished[stage], iter % 2);
            if (!schedule.success)
                break;
            if (cta_rank == 0)
                clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
            tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
            ++iter;
        }
    }
    pipelink_call_gemm_fini(opaque);
}
extern "C" __device__ void mok_framework_worker(const void* opaque, unsigned) {
    const auto& view = *static_cast<const plk_role_view_mok_interp_bwd_worker*>(opaque);
    const auto& g = *static_cast<const MokFrameworkResidue*>(view.residue);
    if (pipelink_mok_precision() == MOK_PRECISION_MXFP8) {
        if (pipelink_mok_clamp()) mok_framework_run<mok_port::RoutedPrecision::MXFP8, true>(opaque, g);
        else mok_framework_run<mok_port::RoutedPrecision::MXFP8, false>(opaque, g);
    } else {
        if (pipelink_mok_clamp()) mok_framework_run<mok_port::RoutedPrecision::BF16, true>(opaque, g);
        else mok_framework_run<mok_port::RoutedPrecision::BF16, false>(opaque, g);
    }
}
PL_STAGE("mok.interp.bwd", "1.0.0");
PL_NO_PARAMS();
PL_ROLE(worker, mok_framework_worker, 8);
PL_TOPOLOGY(pl::topology{2, pl::cluster{{2,1,1}, pl::launch{2,{2,1,1}}}, true, "clc"});
PL_CLAIM_EX(pl::claim("pipelink_mok_worker_warps", "warp").extent(8).align(4).at(0).roles({"worker"}));
PL_CLAIM("pipelink_mok_fw_clc", "smem", 144, 16, "kernel");
PL_CLAIM("pipelink_mok_fw_sems", "smem", 144, 8, "kernel");
PL_CLAIM("pipelink_mok_cnt_d_y_routed_ready", "gmem_counter", MOK_SLOTS_D_Y_ROUTED_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_d_hidden_ready", "gmem_counter", MOK_SLOTS_D_HIDDEN_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_d_gate_up_ready", "gmem_counter", MOK_SLOTS_D_GATE_UP_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_d_x_routed_ready", "gmem_counter", MOK_SLOTS_D_X_ROUTED_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_replayed_x_routed_ready", "gmem_counter", MOK_SLOTS_REPLAYED_X_ROUTED_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_replayed_gate_up_ready", "gmem_counter", MOK_SLOTS_REPLAYED_GATE_UP_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_replayed_hidden_ready", "gmem_counter", MOK_SLOTS_REPLAYED_HIDDEN_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_routed_buffers_done", "gmem_counter", MOK_SLOTS_ROUTED_BUFFERS_DONE, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_router_weights_ready", "gmem_counter", MOK_SLOTS_ROUTER_WEIGHTS_READY, 1, "kernel");
PL_VARIANT(0, {{"pipelink_mok_precision", 0}, {"pipelink_mok_clamp", 0}, {"pipelink_mok_pass", MOK_PASS_BWD}});
PL_VARIANT(1, {{"pipelink_mok_precision", 0}, {"pipelink_mok_clamp", 1}, {"pipelink_mok_pass", MOK_PASS_BWD}});
PL_VARIANT(2, {{"pipelink_mok_precision", 1}, {"pipelink_mok_clamp", 0}, {"pipelink_mok_pass", MOK_PASS_BWD}});
PL_VARIANT(3, {{"pipelink_mok_precision", 1}, {"pipelink_mok_clamp", 1}, {"pipelink_mok_pass", MOK_PASS_BWD}});
PL_CALL(comm_init, pipelink_call_comm_init, pl::produce, "worker", "mok_comm_init");
PL_CALL(comm_bwd, pipelink_call_comm_bwd, pl::produce, "worker", "mok_comm_bwd");
PL_CALL(gemm_init, pipelink_call_gemm_init, pl::produce, "worker", "mok_gemm_init");
PL_CALL(gemm_fini, pipelink_call_gemm_fini, pl::produce, "worker", "mok_gemm_fini");
PL_CALL(gemm_bwd, pipelink_call_gemm_bwd, pl::produce, "worker", "mok_gemm_bwd");
PL_CALL(act_init, pipelink_call_act_init, pl::produce, "worker", "mok_act_init");
PL_CALL(act_bwd, pipelink_call_act_bwd, pl::produce, "worker", "mok_act_bwd");
PL_HOST_GROUP("mok.framework", "mok.framework.params", "mok_framework", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "ep");
PL_HOST_ABI_EMPTY(100, MokFrameworkResidue);
PL_HOST_ABI_TYPE(101, plk_role_view_mok_interp_bwd_worker, residue, scalars);
PL_HOST_RESIDUE(2, "mok.framework.residue", MokFrameworkResidue, 100, true, pl::host::roles({"mok.interp.bwd:worker"}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"mok.interp.bwd:worker"}))
    .field(pl::host::scalar(1, "mok.num_local_tokens", "i64").caller("num_local_tokens"))
    .field(pl::host::scalar(2, "mok.hidden", "i64").caller("hidden"))
    .field(pl::host::scalar(3, "mok.intermediate", "i64").caller("intermediate"))
    .field(pl::host::scalar(4, "mok.num_local_experts", "i64").caller("num_local_experts"))
    .field(pl::host::scalar(5, "mok.schedule_capacity", "i64").caller("schedule_capacity"))
    .field(pl::host::scalar(6, "mok.macrobatch_size", "i64").caller("macrobatch_size"))
    .field(pl::host::scalar(7, "mok.minibatch_size", "i64").caller("minibatch_size"))
    .field(pl::host::scalar(8, "mok.topk", "i64").caller("topk"))
    .field(pl::host::scalar(9, "mok.num_comm_sms", "i64").caller("num_comm_sms"))
    .field(pl::host::scalar(11, "mok.pass", "i64").binding("pipelink_mok_pass"))
    .field(pl::host::scalar(12, "mok.precision", "i64").binding("pipelink_mok_precision"))
    .field(pl::host::scalar(13, "mok.world", "i64").binding("pipelink_mok_world"))
    .field(pl::host::scalar(14, "mok.clamp", "i64").binding("pipelink_mok_clamp"))
    .field(pl::host::scalar(20, "mok.comm.DISPATCH_Mb", "i64").fact("mok.comm.DISPATCH_Mb"))
    .field(pl::host::scalar(21, "mok.comm.DISPATCH_Nb", "i64").fact("mok.comm.DISPATCH_Nb"))
    .field(pl::host::scalar(22, "mok.comm.COMBINE_Mb", "i64").fact("mok.comm.COMBINE_Mb"))
    .field(pl::host::scalar(23, "mok.comm.COMBINE_Nb", "i64").fact("mok.comm.COMBINE_Nb"))
    .field(pl::host::scalar(24, "mok.comm.COMBINE_PIPE_DEPTH", "i64").fact("mok.comm.COMBINE_PIPE_DEPTH"))
    .field(pl::host::scalar(25, "mok.gemm.MLP_Mb", "i64").fact("mok.gemm.MLP_Mb"))
    .field(pl::host::scalar(26, "mok.gemm.MLP_Nb", "i64").fact("mok.gemm.MLP_Nb"))
    .field(pl::host::scalar(27, "mok.act.SWIGLU_Mb", "i64").fact("mok.act.SWIGLU_Mb"))
    .field(pl::host::scalar(28, "mok.act.SWIGLU_Nb", "i64").fact("mok.act.SWIGLU_Nb"))
    .field(pl::host::scalar(29, "mok.act.SWIGLU_FWD_PIPE_DEPTH", "i64").fact("mok.act.SWIGLU_FWD_PIPE_DEPTH"))
    .field(pl::host::scalar(30, "mok.act.SWIGLU_BWD_PIPE_DEPTH", "i64").fact("mok.act.SWIGLU_BWD_PIPE_DEPTH"))
    .field(pl::host::scalar(45, "mok.slot.d_y_routed_ready", "i64").binding("pipelink_mok_cnt_d_y_routed_ready"))
    .field(pl::host::scalar(46, "mok.slot.d_hidden_ready", "i64").binding("pipelink_mok_cnt_d_hidden_ready"))
    .field(pl::host::scalar(47, "mok.slot.d_gate_up_ready", "i64").binding("pipelink_mok_cnt_d_gate_up_ready"))
    .field(pl::host::scalar(48, "mok.slot.d_x_routed_ready", "i64").binding("pipelink_mok_cnt_d_x_routed_ready"))
    .field(pl::host::scalar(49, "mok.slot.replayed_x_routed_ready", "i64").binding("pipelink_mok_cnt_replayed_x_routed_ready"))
    .field(pl::host::scalar(50, "mok.slot.replayed_gate_up_ready", "i64").binding("pipelink_mok_cnt_replayed_gate_up_ready"))
    .field(pl::host::scalar(51, "mok.slot.replayed_hidden_ready", "i64").binding("pipelink_mok_cnt_replayed_hidden_ready"))
    .field(pl::host::scalar(52, "mok.slot.routed_buffers_done", "i64").binding("pipelink_mok_cnt_routed_buffers_done"))
    .field(pl::host::scalar(53, "mok.slot.router_weights_ready", "i64").binding("pipelink_mok_cnt_router_weights_ready"))
);
PL_HOST_CHANNEL(1, int32_t, pl::host::channel("mok.framework.num_tokens", "i32").rank(1).read().caller("num_tokens"));
PL_HOST_WORKSPACE(1, 256, pl::host::workspace("mok.framework.counters").zero().reset_launch());
PL_HOST_FACT(pl::host::fact("mok.framework.EXPERT_PADDING").i32(256).owner("mok.framework").stage("mok.interp.bwd").consumers({"mok.gemm"}));
PL_HOST_SCHEDULER(pl::host::scheduler("mok.framework", "mok.framework.clc-grid.v1",
    pl::host::schema().required("pass", pl::host::leaf::static_("i64")))
    .input("pass", pl::host::binding_ref("pipelink_mok_pass")));
PL_HOST_TARGET();
