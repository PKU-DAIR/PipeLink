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
extern "C" __device__ void pipelink_call_comm_fwd(const void* view, const MokCommTask*, const MokCounters*, uint32_t*);
extern "C" __device__ void pipelink_call_gemm_fwd(const void* view, const MokGemmTask*, const MokCounters*, uint32_t*);
extern "C" __device__ void pipelink_call_act_fwd(const void* view, const MokActTask*, const MokCounters*, uint32_t*);
struct MokFrameworkClc { clc::handle handle[1], drain[8]; };
struct MokFrameworkSems { semaphore schedule_arrived[1], schedule_finished[1], drain_arrived[8], drain_finished[8]; };
static_assert(sizeof(MokFrameworkClc) == 144 && sizeof(MokFrameworkSems) == 144);
template <mok_port::RoutedPrecision PRECISION, bool IS_CLAMPED>
static __device__ __forceinline__ void mok_framework_run(const void* opaque, const MokFrameworkResidue &g) {
    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int shared_row_blocks = g.num_local_tokens / g.MLP_Mb;
    const int minibatch_routed_row_blocks = g.minibatch_size / g.MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.intermediate / g.MLP_Nb);
    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (g.intermediate / g.MLP_Nb);
    const int shared_swiglu_tiles = (g.num_local_tokens / g.SWIGLU_Mb) * (g.intermediate / g.SWIGLU_Nb);
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / g.SWIGLU_Mb) * (g.intermediate / g.SWIGLU_Nb);
    const int shared_swiglu_tasks = (shared_swiglu_tiles + 2 * g.SWIGLU_FWD_PIPE_DEPTH - 1) / (2 * g.SWIGLU_FWD_PIPE_DEPTH);
    const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + 2 * g.SWIGLU_FWD_PIPE_DEPTH - 1) / (2 * g.SWIGLU_FWD_PIPE_DEPTH);
    const int shared_down_tasks = shared_row_blocks * (g.hidden / g.MLP_Nb);
    const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (g.hidden / g.MLP_Nb);
    const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
    const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
    const int comm_clusters = g.num_comm_sms / 2;
    const int macrobatch_size = g.macrobatch_size;

    const int num_tokens = g.num_tokens[0];
    const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;
    const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;
    const int true_num_global_minibatches = (num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int last_macrobatch_num_minibatches = true_num_global_minibatches - (num_macrobatches - 1) * minibatches_per_macrobatch;
    const int true_num_clusters = comm_clusters + shared_tasks + true_num_global_minibatches * minibatch_tasks;
    if (cluster_idx >= true_num_clusters) return;

    const MokCounters counters{g.counters};

    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_bitfield = 0xFFFF0000;
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
        auto num_dispatch_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int dispatch_col_blocks = (g.hidden + g.DISPATCH_Nb - 1) / g.DISPATCH_Nb;
            return (macrobatch_tokens / g.DISPATCH_Mb) * dispatch_col_blocks;
        };
        auto num_combine_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int combine_col_blocks = (g.hidden + g.COMBINE_Nb - 1) / g.COMBINE_Nb;
            const int combine_tiles = (macrobatch_tokens / g.COMBINE_Mb) * combine_col_blocks;
            return (combine_tiles + g.COMBINE_PIPE_DEPTH - 1) / g.COMBINE_PIPE_DEPTH;
        };
        auto dispatch = [&](int macrobatch_idx, int task_idx) {
            const MokCommTask task{MOK_COMM_FWD_DISPATCH, num_tokens, macrobatch_size,
                g.minibatch_size, macrobatch_idx, task_idx, g.topk, macrobatch_idx + 1, 0,
                macrobatch_idx + 1 < num_macrobatches ? uint32_t(MOK_COMM_TRANSFER_READY) : 0u};
            pipelink_call_comm_fwd(opaque, &task, &counters, &dispatch_bitfield);
        };
        auto combine = [&](int macrobatch_idx, int task_idx) {
            const MokCommTask task{MOK_COMM_FWD_COMBINE, num_tokens, macrobatch_size,
                g.minibatch_size, macrobatch_idx, task_idx, 0, 0, 0,
                macrobatch_idx > 0 ? uint32_t(MOK_COMM_TRANSFER_DONE) : 0u};
            pipelink_call_comm_fwd(opaque, &task, &counters, &combine_bitfield);
        };
        if (num_macrobatches == 0) { pipelink_call_gemm_fini(opaque); return; }
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks(num_macrobatches - 1); task_idx += g.num_comm_sms)
            dispatch(num_macrobatches - 1, task_idx);
        for (int macrobatch_idx = num_macrobatches - 1; macrobatch_idx >= 0; --macrobatch_idx) {
            const int combine_tasks = num_combine_tasks(macrobatch_idx);
            const int dispatch_tasks = macrobatch_idx > 0 ? num_dispatch_tasks(macrobatch_idx - 1) : 0;
            for (int task_idx = comm_cta_idx; task_idx < max(combine_tasks, dispatch_tasks); task_idx += g.num_comm_sms) {
                if (task_idx < combine_tasks)
                    combine(macrobatch_idx, task_idx);
                if (task_idx < dispatch_tasks)
                    dispatch(macrobatch_idx - 1, task_idx);
            }
        }
        pipelink_call_gemm_fini(opaque);
        return;
    }

    // Swiglu tasks are CTA-local, GEMM is not
    auto is_cta_local_task = [&](int compute_cluster_idx) {
        const int minibatch_task_idx = (compute_cluster_idx - shared_tasks) % minibatch_tasks;
        if (compute_cluster_idx < 0) return false;
        else if (compute_cluster_idx < 2 * shared_gate_up_tasks) return false; // shared gate/up
        else if (compute_cluster_idx < 2 * shared_gate_up_tasks + shared_swiglu_tasks) return true; // shared swiglu
        else if (compute_cluster_idx < shared_tasks) return false; // shared down
        else if (minibatch_task_idx < 2 * minibatch_routed_gate_up_tasks) return false; // routed gate/up
        else if (minibatch_task_idx < 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks) return true; // routed swiglu
        else return false; // routed down
    };
    const int hidden_row_block_ready_required_count = (g.MLP_Mb / g.SWIGLU_Mb) * (g.intermediate / g.SWIGLU_Nb);

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

        if (compute_cluster_idx < shared_gate_up_tasks) {
            // Shared gate (BF16)
            const int task_idx = compute_cluster_idx;
            const MokGemmTask task{MOK_GEMM_FWD_SHARED_GATE, num_tokens, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, 0, 0, 0, 0u};
            pipelink_call_gemm_fwd(opaque, &task, &counters, &gemm_bitfield);
        } else if (compute_cluster_idx < shared_gate_up_tasks * 2) {
            // Shared up (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks;
            const MokGemmTask task{MOK_GEMM_FWD_SHARED_UP, num_tokens, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, 0, 0, 0, 0u};
            pipelink_call_gemm_fwd(opaque, &task, &counters, &gemm_bitfield);
        } else if (compute_cluster_idx < shared_gate_up_tasks * 2 + shared_swiglu_tasks) {
            // Shared Swiglu (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks * 2;
            const MokActTask task{MOK_ACT_FWD_SHARED, g.num_local_tokens, macrobatch_size,
                g.minibatch_size, 0, 0, task_idx, cta_rank, 0, 0, 0u};
            pipelink_call_act_fwd(opaque, &task, &counters, &swiglu_bitfield);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared down (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks * 2 - shared_swiglu_tasks;
            const MokGemmTask task{MOK_GEMM_FWD_SHARED_DOWN, num_tokens, macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, hidden_row_block_ready_required_count, 0, 0, 0u};
            pipelink_call_gemm_fwd(opaque, &task, &counters, &gemm_bitfield);
        } else {
            // Routed expert with macro/minibatching
            const int task_ordered_global_minibatch_idx = (compute_cluster_idx - shared_tasks) / minibatch_tasks;
            const int minibatch_task_idx = (compute_cluster_idx - shared_tasks) - task_ordered_global_minibatch_idx * minibatch_tasks;
            int macrobatch_idx, minibatch_idx;
            if (task_ordered_global_minibatch_idx < last_macrobatch_num_minibatches) {
                macrobatch_idx = num_macrobatches - 1;
                minibatch_idx = task_ordered_global_minibatch_idx;
            } else {
                const int idx = task_ordered_global_minibatch_idx - last_macrobatch_num_minibatches;
                macrobatch_idx = num_macrobatches - 2 - idx / minibatches_per_macrobatch;
                minibatch_idx = idx % minibatches_per_macrobatch;
            }

            if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                // Routed gate
                const int task_idx = minibatch_task_idx;
            const MokGemmTask task{MOK_GEMM_FWD_ROUTED_GATE, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, minibatch_idx, task_idx, cta_rank, 0, 0, 0, shared_gate_up_tasks, 0, 0u};
            pipelink_call_gemm_fwd(opaque, &task, &counters, &gemm_bitfield);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                // Routed up
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
            const MokGemmTask task{MOK_GEMM_FWD_ROUTED_UP, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, minibatch_idx, task_idx, cta_rank, 0, 0, 0, shared_gate_up_tasks, 0, 0u};
            pipelink_call_gemm_fwd(opaque, &task, &counters, &gemm_bitfield);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2 + minibatch_routed_swiglu_tasks) {
                // Routed Swiglu
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
                const MokActTask task{MOK_ACT_FWD_ROUTED, num_tokens, macrobatch_size,
                    g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                    shared_gate_up_tasks, shared_row_blocks, 0u};
                pipelink_call_act_fwd(opaque, &task, &counters, &swiglu_bitfield);
            } else {
                // Routed down
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2 - minibatch_routed_swiglu_tasks;
            const MokGemmTask task{MOK_GEMM_FWD_ROUTED_DOWN, num_tokens, macrobatch_size, g.minibatch_size,
                macrobatch_idx, minibatch_idx, task_idx, cta_rank, 0, shared_row_blocks, hidden_row_block_ready_required_count, 0, 0, macrobatch_idx + 1 < num_macrobatches ? uint32_t(MOK_GEMM_OUTPUT_ROW_READY) : 0u};
            pipelink_call_gemm_fwd(opaque, &task, &counters, &gemm_bitfield);
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
    const auto& view = *static_cast<const plk_role_view_mok_interp_fwd_worker*>(opaque);
    const auto& g = *static_cast<const MokFrameworkResidue*>(view.residue);
    if (pipelink_mok_precision() == MOK_PRECISION_MXFP8) {
        if (pipelink_mok_clamp()) mok_framework_run<mok_port::RoutedPrecision::MXFP8, true>(opaque, g);
        else mok_framework_run<mok_port::RoutedPrecision::MXFP8, false>(opaque, g);
    } else {
        if (pipelink_mok_clamp()) mok_framework_run<mok_port::RoutedPrecision::BF16, true>(opaque, g);
        else mok_framework_run<mok_port::RoutedPrecision::BF16, false>(opaque, g);
    }
}
PL_STAGE("mok.interp.fwd", "1.0.0");
PL_NO_PARAMS();
PL_ROLE(worker, mok_framework_worker, 8);
PL_TOPOLOGY(pl::topology{2, pl::cluster{{2,1,1}, pl::launch{2,{2,1,1}}}, true, "clc"});
PL_CLAIM_EX(pl::claim("pipelink_mok_worker_warps", "warp").extent(8).align(4).at(0).roles({"worker"}));
PL_CLAIM("pipelink_mok_fw_clc", "smem", 144, 16, "kernel");
PL_CLAIM("pipelink_mok_fw_sems", "smem", 144, 8, "kernel");
PL_CLAIM("pipelink_mok_cnt_x_routed_ready", "gmem_counter", MOK_SLOTS_X_ROUTED_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_gate_up_tile_ready", "gmem_counter", MOK_SLOTS_GATE_UP_TILE_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_hidden_row_block_ready", "gmem_counter", MOK_SLOTS_HIDDEN_ROW_BLOCK_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_y_routed_ready", "gmem_counter", MOK_SLOTS_Y_ROUTED_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_y_routed_done", "gmem_counter", MOK_SLOTS_Y_ROUTED_DONE, 1, "kernel");
PL_VARIANT(0, {{"pipelink_mok_precision", 0}, {"pipelink_mok_clamp", 0}, {"pipelink_mok_pass", MOK_PASS_FWD}});
PL_VARIANT(1, {{"pipelink_mok_precision", 0}, {"pipelink_mok_clamp", 1}, {"pipelink_mok_pass", MOK_PASS_FWD}});
PL_VARIANT(2, {{"pipelink_mok_precision", 1}, {"pipelink_mok_clamp", 0}, {"pipelink_mok_pass", MOK_PASS_FWD}});
PL_VARIANT(3, {{"pipelink_mok_precision", 1}, {"pipelink_mok_clamp", 1}, {"pipelink_mok_pass", MOK_PASS_FWD}});
PL_CALL(comm_init, pipelink_call_comm_init, pl::produce, "worker", "mok_comm_init");
PL_CALL(comm_fwd, pipelink_call_comm_fwd, pl::produce, "worker", "mok_comm_fwd");
PL_CALL(gemm_init, pipelink_call_gemm_init, pl::produce, "worker", "mok_gemm_init");
PL_CALL(gemm_fini, pipelink_call_gemm_fini, pl::produce, "worker", "mok_gemm_fini");
PL_CALL(gemm_fwd, pipelink_call_gemm_fwd, pl::produce, "worker", "mok_gemm_fwd");
PL_CALL(act_init, pipelink_call_act_init, pl::produce, "worker", "mok_act_init");
PL_CALL(act_fwd, pipelink_call_act_fwd, pl::produce, "worker", "mok_act_fwd");
PL_HOST_GROUP("mok.framework", "mok.framework.params", "mok_framework", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "ep");
PL_HOST_ABI_EMPTY(100, MokFrameworkResidue);
PL_HOST_ABI_TYPE(101, plk_role_view_mok_interp_fwd_worker, residue, scalars);
PL_HOST_RESIDUE(2, "mok.framework.residue", MokFrameworkResidue, 100, true, pl::host::roles({"mok.interp.fwd:worker"}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"mok.interp.fwd:worker"}))
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
    .field(pl::host::scalar(40, "mok.slot.x_routed_ready", "i64").binding("pipelink_mok_cnt_x_routed_ready"))
    .field(pl::host::scalar(41, "mok.slot.gate_up_tile_ready", "i64").binding("pipelink_mok_cnt_gate_up_tile_ready"))
    .field(pl::host::scalar(42, "mok.slot.hidden_row_block_ready", "i64").binding("pipelink_mok_cnt_hidden_row_block_ready"))
    .field(pl::host::scalar(43, "mok.slot.y_routed_ready", "i64").binding("pipelink_mok_cnt_y_routed_ready"))
    .field(pl::host::scalar(44, "mok.slot.y_routed_done", "i64").binding("pipelink_mok_cnt_y_routed_done"))
);
PL_HOST_CHANNEL(1, int32_t, pl::host::channel("mok.framework.num_tokens", "i32").rank(1).read().caller("num_tokens"));
PL_HOST_WORKSPACE(1, 256, pl::host::workspace("mok.framework.counters").zero().reset_launch());
PL_HOST_FACT(pl::host::fact("mok.framework.EXPERT_PADDING").i32(256).owner("mok.framework").stage("mok.interp.fwd").consumers({"mok.gemm"}));
PL_HOST_SCHEDULER(pl::host::scheduler("mok.framework", "mok.framework.clc-grid.v1",
    pl::host::schema().required("pass", pl::host::leaf::static_("i64")))
    .input("pass", pl::host::binding_ref("pipelink_mok_pass")));
PL_HOST_TARGET();
