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
extern "C" __device__ void pipelink_call_comm_rec(const void* view, const MokCommTask*, const MokCounters*, uint32_t*);
extern "C" __device__ void pipelink_call_gemm_rec(const void* view, const MokGemmTask*, const MokCounters*, uint32_t*);
extern "C" __device__ void pipelink_call_act_rec(const void* view, const MokActTask*, const MokCounters*, uint32_t*);
struct MokFrameworkClc { clc::handle handle[1], drain[8]; };
struct MokFrameworkSems { semaphore schedule_arrived[1], schedule_finished[1], drain_arrived[8], drain_finished[8]; };
static_assert(sizeof(MokFrameworkClc) == 144 && sizeof(MokFrameworkSems) == 144);

// MoK f3a8bd2432457c5a2a2ea13915f0dcd0224581e4,
// csrc/megakernel/recompute_forward_context.cuh:1-214. P1 substitutions only:
// descriptor shapes/foreign geometry become residue fields, owner state moves to
// init/fini and claims, and calls use owner task records. CLC topology stays 2 CTAs,
// 8 warps, one schedule stage and eight drain stages; no register-group request.
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
    const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks;
    const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks;
    const int comm_clusters = g.num_comm_sms / 2;
    const int num_tokens = g.num_tokens[0];
    const int routed_num_tokens = min(num_tokens, g.macrobatch_size);
    const int num_routed_minibatches = (routed_num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int true_num_clusters = comm_clusters + shared_tasks + num_routed_minibatches * minibatch_tasks;
    if (cluster_idx >= true_num_clusters) return;

    const MokCounters counters{g.counters};
    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_bitfield = 0xFFFF0000;
    uint32_t dispatch_bitfield = 0xFFFF0000;

    // Upstream :31-75; persistent framework state is disjoint from owner state.
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

    // Upstream :77-90: dispatch only macro zero, with no transfer guards.
    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * 2 + cta_rank;
        const int dispatch_col_blocks = (g.hidden + g.DISPATCH_Nb - 1) / g.DISPATCH_Nb;
        const int num_dispatch_tasks = (routed_num_tokens / g.DISPATCH_Mb) * dispatch_col_blocks;
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks; task_idx += g.num_comm_sms) {
            const MokCommTask task{MOK_COMM_REC_DISPATCH, num_tokens, g.macrobatch_size,
                g.minibatch_size, 0, task_idx, g.topk, -1, 0, 0u, 0};
            pipelink_call_comm_rec(opaque, &task, &counters, &dispatch_bitfield);
        }
        pipelink_call_gemm_fini(opaque);
        return;
    }

    // Upstream :92-111: preserve CTA-local classification and CLC issue order.
    auto is_cta_local_task = [&](int compute_cluster_idx) {
        if (compute_cluster_idx < 0) return false;
        if (compute_cluster_idx < 2 * shared_gate_up_tasks) return false;
        if (compute_cluster_idx < shared_tasks) return true;
        const int minibatch_task_idx = (compute_cluster_idx - shared_tasks) % minibatch_tasks;
        return minibatch_task_idx >= 2 * minibatch_routed_gate_up_tasks;
    };

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

        // Upstream :113-175: all GEMMs and routed act receive live T, not min(T,A).
        if (compute_cluster_idx < shared_gate_up_tasks) {
            // Shared gate (BF16)
            const int task_idx = compute_cluster_idx;
            const MokGemmTask task{MOK_GEMM_REC_SHARED_GATE, num_tokens, g.macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, 0, 0, 0, 0u};
            pipelink_call_gemm_rec(opaque, &task, &counters, &gemm_bitfield);
        } else if (compute_cluster_idx < shared_gate_up_tasks * 2) {
            // Shared up (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks;
            const MokGemmTask task{MOK_GEMM_REC_SHARED_UP, num_tokens, g.macrobatch_size, g.minibatch_size,
                0, 0, task_idx, cta_rank, 0, 0, 0, 0, 0, 0u};
            pipelink_call_gemm_rec(opaque, &task, &counters, &gemm_bitfield);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared Swiglu (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks * 2;
            const MokActTask task{MOK_ACT_REC_SHARED, g.num_local_tokens, g.macrobatch_size,
                g.minibatch_size, 0, 0, task_idx, cta_rank, 0, 0, 0u, 0, 0, 0, 0};
            pipelink_call_act_rec(opaque, &task, &counters, &swiglu_bitfield);
        } else {
            const int routed_task_idx = compute_cluster_idx - shared_tasks;
            const int minibatch_idx = routed_task_idx / minibatch_tasks;
            const int minibatch_task_idx = routed_task_idx % minibatch_tasks;

            if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                // Routed gate
                const int task_idx = minibatch_task_idx;
                const MokGemmTask task{MOK_GEMM_REC_ROUTED_GATE, num_tokens, g.macrobatch_size, g.minibatch_size,
                    0, minibatch_idx, task_idx, cta_rank, 0, 0, 0, shared_gate_up_tasks, 0, 0u};
                pipelink_call_gemm_rec(opaque, &task, &counters, &gemm_bitfield);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                // Routed up
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
                const MokGemmTask task{MOK_GEMM_REC_ROUTED_UP, num_tokens, g.macrobatch_size, g.minibatch_size,
                    0, minibatch_idx, task_idx, cta_rank, 0, 0, 0, shared_gate_up_tasks, 0, 0u};
                pipelink_call_gemm_rec(opaque, &task, &counters, &gemm_bitfield);
            } else {
                // Routed Swiglu
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
                const MokActTask task{MOK_ACT_REC_ROUTED, num_tokens, g.macrobatch_size,
                    g.minibatch_size, 0, minibatch_idx, task_idx, cta_rank,
                    shared_gate_up_tasks, shared_row_blocks, 0u, 0, 0, 0, 0};
                pipelink_call_act_rec(opaque, &task, &counters, &swiglu_bitfield);
            }
        }

        // Upstream :179-191: unchanged schedule completion and CTA-local transition.
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

    // Upstream :193-213: unchanged CLC drain for no-op threadblocks.
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
    const auto& view = *static_cast<const plk_role_view_mok_interp_rec_worker*>(opaque);
    const auto& g = *static_cast<const MokFrameworkResidue*>(view.residue);
    if (pipelink_mok_precision() == MOK_PRECISION_MXFP8) {
        if (pipelink_mok_clamp()) mok_framework_run<mok_port::RoutedPrecision::MXFP8, true>(opaque, g);
        else mok_framework_run<mok_port::RoutedPrecision::MXFP8, false>(opaque, g);
    } else {
        if (pipelink_mok_clamp()) mok_framework_run<mok_port::RoutedPrecision::BF16, true>(opaque, g);
        else mok_framework_run<mok_port::RoutedPrecision::BF16, false>(opaque, g);
    }
}
PL_STAGE("mok.interp.rec", "1.0.0");
PL_NO_PARAMS();
PL_ROLE(worker, mok_framework_worker, 8);
PL_TOPOLOGY(pl::topology{2, pl::cluster{{2,1,1}, pl::launch{2,{2,1,1}}}, true, "clc"});
PL_CLAIM_EX(pl::claim("pipelink_mok_worker_warps", "warp").extent(8).align(4).at(0).roles({"worker"}));
PL_CLAIM("pipelink_mok_fw_clc", "smem", 144, 16, "kernel");
PL_CLAIM("pipelink_mok_fw_sems", "smem", 144, 8, "kernel");
// P2 §5: hidden arrivals remain live even though recompute has no down consumer.
PL_CLAIM("pipelink_mok_cnt_x_routed_ready", "gmem_counter", MOK_SLOTS_X_ROUTED_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_gate_up_tile_ready", "gmem_counter", MOK_SLOTS_GATE_UP_TILE_READY, 1, "kernel");
PL_CLAIM("pipelink_mok_cnt_hidden_row_block_ready", "gmem_counter", MOK_SLOTS_HIDDEN_ROW_BLOCK_READY, 1, "kernel");
PL_VARIANT(0, {{"pipelink_mok_precision", 0}, {"pipelink_mok_clamp", 0}, {"pipelink_mok_pass", 2}});
PL_VARIANT(1, {{"pipelink_mok_precision", 0}, {"pipelink_mok_clamp", 1}, {"pipelink_mok_pass", 2}});
PL_VARIANT(2, {{"pipelink_mok_precision", 1}, {"pipelink_mok_clamp", 0}, {"pipelink_mok_pass", 2}});
PL_VARIANT(3, {{"pipelink_mok_precision", 1}, {"pipelink_mok_clamp", 1}, {"pipelink_mok_pass", 2}});
PL_CALL(comm_init, pipelink_call_comm_init, pl::produce, "worker", "mok_comm_init");
PL_CALL(comm_rec, pipelink_call_comm_rec, pl::produce, "worker", "mok_comm_rec");
PL_CALL(gemm_init, pipelink_call_gemm_init, pl::produce, "worker", "mok_gemm_init");
PL_CALL(gemm_fini, pipelink_call_gemm_fini, pl::produce, "worker", "mok_gemm_fini");
PL_CALL(gemm_rec, pipelink_call_gemm_rec, pl::produce, "worker", "mok_gemm_rec");
PL_CALL(act_init, pipelink_call_act_init, pl::produce, "worker", "mok_act_init");
PL_CALL(act_rec, pipelink_call_act_rec, pl::produce, "worker", "mok_act_rec");
PL_HOST_GROUP("mok.framework", "mok.framework.params", "mok_framework", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "ep");
PL_HOST_ABI_EMPTY(100, MokFrameworkResidue);
PL_HOST_ABI_TYPE(101, plk_role_view_mok_interp_rec_worker, residue, scalars);
PL_HOST_RESIDUE(2, "mok.framework.residue", MokFrameworkResidue, 100, true, pl::host::roles({"mok.interp.rec:worker"}));
// P2 §7: shape 1-9, bound controls 11-14, rec geometry 20-21/25-29,
// and only the three rec slot bases 40-42. No combine/backward geometry reads.
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"mok.interp.rec:worker"}))
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
    .field(pl::host::scalar(25, "mok.gemm.MLP_Mb", "i64").fact("mok.gemm.MLP_Mb"))
    .field(pl::host::scalar(26, "mok.gemm.MLP_Nb", "i64").fact("mok.gemm.MLP_Nb"))
    .field(pl::host::scalar(27, "mok.act.SWIGLU_Mb", "i64").fact("mok.act.SWIGLU_Mb"))
    .field(pl::host::scalar(28, "mok.act.SWIGLU_Nb", "i64").fact("mok.act.SWIGLU_Nb"))
    .field(pl::host::scalar(29, "mok.act.SWIGLU_FWD_PIPE_DEPTH", "i64").fact("mok.act.SWIGLU_FWD_PIPE_DEPTH"))
    .field(pl::host::scalar(40, "mok.slot.x_routed_ready", "i64").binding("pipelink_mok_cnt_x_routed_ready"))
    .field(pl::host::scalar(41, "mok.slot.gate_up_tile_ready", "i64").binding("pipelink_mok_cnt_gate_up_tile_ready"))
    .field(pl::host::scalar(42, "mok.slot.hidden_row_block_ready", "i64").binding("pipelink_mok_cnt_hidden_row_block_ready"))
);
PL_HOST_CHANNEL(1, int32_t, pl::host::channel("mok.framework.num_tokens", "i32").rank(1).read().caller("num_tokens"));
PL_HOST_WORKSPACE(1, 256, pl::host::workspace("mok.framework.counters").zero().reset_launch());
PL_HOST_FACT(pl::host::fact("mok.framework.EXPERT_PADDING").i32(256).owner("mok.framework").stage("mok.interp.rec").consumers({"mok.gemm"}));
PL_HOST_SCHEDULER(pl::host::scheduler("mok.framework", "mok.framework.clc-grid.v1",
    pl::host::schema().required("pass", pl::host::leaf::static_("i64")))
    .input("pass", pl::host::binding_ref("pipelink_mok_pass")));
PL_HOST_TARGET();
