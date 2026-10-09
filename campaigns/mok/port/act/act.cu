#include "act_abi.h"
#include "mok_port.cuh"
#include "mxfp8.cuh"
#include <pipelink/stage.hpp>
#include <pipelink_host_abi_v2.h>

#ifndef MOK_ACT_PROVIDER
extern "C" __device__ int pipelink_mok_precision();
extern "C" __device__ int pipelink_mok_clamp();
extern "C" __device__ int pipelink_mok_act_arena();
extern "C" __device__ int pipelink_mok_act_sems();
extern "C" __device__ int pipelink_mok_cnt_gate_up_tile_ready();
extern "C" __device__ int pipelink_mok_cnt_hidden_row_block_ready();
extern "C" __device__ int pipelink_mok_cnt_d_hidden_ready();
extern "C" __device__ int pipelink_mok_cnt_d_gate_up_ready();
extern "C" __device__ int pipelink_mok_cnt_replayed_gate_up_ready();
extern "C" __device__ int pipelink_mok_cnt_replayed_hidden_ready();
extern "C" __device__ int pipelink_mok_cnt_routed_buffers_done();
#endif

namespace mok_act_owner {
using namespace kittens;
using mok_port::RoutedPrecision;

struct config {
    static constexpr int SWIGLU_Mb = 128;
    static constexpr int SWIGLU_Nb = 128;
    static constexpr int SWIGLU_FWD_PIPE_DEPTH = 3;
    static constexpr int SWIGLU_BWD_PIPE_DEPTH = 2;
    static constexpr int QUANT_Mb = mxfp8::config::Mb;
    static constexpr int QUANT_Nb = mxfp8::config::Nb;
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};
struct Facts { int32_t MLP_Mb, MLP_Nb; };
struct Semaphores {
    semaphore fwd[config::SWIGLU_FWD_PIPE_DEPTH];
    semaphore bwd[config::SWIGLU_BWD_PIPE_DEPTH];
};
static_assert(sizeof(Semaphores) == 40);

template <RoutedPrecision Precision, bool Clamp>
struct Owner {
    static constexpr bool USE_MXFP8 = Precision == RoutedPrecision::MXFP8;
    using config = mok_act_owner::config;
    using quant_bf16_tile = mxfp8::globals::x_bf16_tile;
    using quant_fp8_tile = mxfp8::globals::x_fp8_tile;
    using quant_sc_tile = mxfp8::globals::x_sc_tile;
    using swiglu_tile = st_bf<config::SWIGLU_Mb, config::SWIGLU_Nb>;
    using mlp_bf16_gl = gl<bf16, 1, 1, -1, -1, swiglu_tile>;
    using epi_bf16_gl = gl<bf16, 1, 1, -1, -1, swiglu_tile, quant_bf16_tile>;
    using swiglu_bf16_gl = epi_bf16_gl;
    using mlp_fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, quant_fp8_tile>;
    using routed_bf16_gl = mlp_bf16_gl;
    using routed_activation_gl = std::conditional_t<USE_MXFP8, mlp_fp8_gl, routed_bf16_gl>;
    using routed_gate_up_gl = routed_activation_gl;
    using routed_transposed_gl = routed_activation_gl;
    using sc_gl = gl<fp8e8m0, -1, -1, 32, 16, quant_sc_tile>;
    struct unused_gl {};
    using routed_sc_gl = std::conditional_t<USE_MXFP8, sc_gl, unused_gl>;
    using index_gl = mok_port::index_view;
    using router_weight_gl = gl<float, 1, 1, 1, -1>;
    using d_router_weight_partials_gl = gl<float, 1, 1, -1, -1>;

    struct Residue {
        epi_bf16_gl gate_shared, up_shared, gate_routed, up_routed;
        mlp_bf16_gl hidden_shared;
        routed_activation_gl hidden_routed;
        routed_sc_gl hidden_sc;
        routed_transposed_gl hidden_t;
        routed_sc_gl hidden_sc_t;
        epi_bf16_gl d_hidden_shared, d_hidden_routed;
        mlp_bf16_gl d_gate_shared, d_up_shared;
        routed_gate_up_gl saved_gate, saved_up;
        routed_sc_gl gate_sc, up_sc;
        routed_activation_gl d_gate, d_up;
        routed_sc_gl d_gate_sc, d_up_sc;
        routed_transposed_gl d_gate_t, d_up_t;
        routed_sc_gl d_gate_sc_t, d_up_sc_t;
        router_weight_gl router_weights;
        d_router_weight_partials_gl d_router_weight_partials;
        index_gl schedule_peer_rank;
        Facts facts;
        float swiglu_limit;
    };

#ifndef MOK_ACT_PROVIDER
    static __device__ __forceinline__ const Residue& residue(const void* view) {
        const auto& owner_view = *static_cast<const pipelink_host_view*>(view);
        return *static_cast<const Residue*>(owner_view.residue);
    }
    static __device__ __forceinline__ const Facts& facts(const void* view) { return residue(view).facts; }

#include "megakernel/utils.cuh"
#include "megakernel/swiglu.cuh"

    static __device__ __forceinline__ void forward(const void* view, const MokActTask& task,
            uint32_t& phase, const index_gl& gate_up, const index_gl& hidden, bool shared) {
        const auto& g = residue(view);
        auto& sems = *reinterpret_cast<Semaphores*>(
            mok_port::smem_claim_address(pipelink_mok_act_sems()));
        const uint64_t arena = mok_port::smem_claim_address(pipelink_mok_act_arena());
        if (shared) {
            swiglu_fwd_kernel<true, Clamp>(view, g.gate_shared, g.up_shared, g.hidden_shared,
                nullptr, nullptr, nullptr, gate_up, hidden, sems.fwd, phase,
                task.num_tokens, g.swiglu_limit, task.macrobatch_size, task.minibatch_size,
                task.macrobatch_idx, task.minibatch_idx, task.task_idx, task.cta_rank,
                task.gate_up_tile_ready_base_index, task.hidden_row_block_ready_base_index, arena);
        } else {
            swiglu_fwd_kernel<false, Clamp>(view, g.gate_routed, g.up_routed, g.hidden_routed,
                &g.hidden_sc, &g.hidden_t, &g.hidden_sc_t, gate_up, hidden, sems.fwd, phase,
                task.num_tokens, g.swiglu_limit, task.macrobatch_size, task.minibatch_size,
                task.macrobatch_idx, task.minibatch_idx, task.task_idx, task.cta_rank,
                task.gate_up_tile_ready_base_index, task.hidden_row_block_ready_base_index, arena);
        }
    }
    static __device__ __forceinline__ void fwd(const void* view, const MokActTask& task,
            const MokCounters& counters, uint32_t& phase) {
        const index_gl gate_up{counters.base + pipelink_mok_cnt_gate_up_tile_ready(),
            MOK_SLOTS_GATE_UP_TILE_READY};
        const index_gl hidden{counters.base + pipelink_mok_cnt_hidden_row_block_ready(),
            MOK_SLOTS_HIDDEN_ROW_BLOCK_READY};
        forward(view, task, phase, gate_up, hidden, task.op == MOK_ACT_FWD_SHARED);
    }
    static __device__ __forceinline__ void rec(const void* view, const MokActTask& task,
            const MokCounters& counters, uint32_t& phase) {
        const index_gl gate_up{counters.base + pipelink_mok_cnt_gate_up_tile_ready(),
            MOK_SLOTS_GATE_UP_TILE_READY};
        const index_gl hidden{counters.base + pipelink_mok_cnt_hidden_row_block_ready(),
            MOK_SLOTS_HIDDEN_ROW_BLOCK_READY};
        forward(view, task, phase, gate_up, hidden, task.op == MOK_ACT_REC_SHARED);
    }
    static __device__ __forceinline__ void bwd(const void* view, const MokActTask& task,
            const MokCounters& counters, uint32_t& phase) {
        const index_gl replayed_gate{counters.base + pipelink_mok_cnt_replayed_gate_up_ready(),
            MOK_SLOTS_REPLAYED_GATE_UP_READY};
        if (task.op == MOK_ACT_BWD_REPLAY_ROUTED) {
            const index_gl replayed_hidden{counters.base + pipelink_mok_cnt_replayed_hidden_ready(),
                MOK_SLOTS_REPLAYED_HIDDEN_READY};
            forward(view, task, phase, replayed_gate, replayed_hidden, false);
            return;
        }
        const auto& g = residue(view);
        const index_gl d_hidden{counters.base + pipelink_mok_cnt_d_hidden_ready(),
            MOK_SLOTS_D_HIDDEN_READY};
        const index_gl d_gate_up{counters.base + pipelink_mok_cnt_d_gate_up_ready(),
            MOK_SLOTS_D_GATE_UP_READY};
        const index_gl done{counters.base + pipelink_mok_cnt_routed_buffers_done(),
            MOK_SLOTS_ROUTED_BUFFERS_DONE};
        auto& sems = *reinterpret_cast<Semaphores*>(
            mok_port::smem_claim_address(pipelink_mok_act_sems()));
        const uint64_t arena = mok_port::smem_claim_address(pipelink_mok_act_arena());
        if (task.op == MOK_ACT_BWD_SHARED) {
            swiglu_bwd_kernel<true, Clamp>(view, g.d_hidden_shared, g.gate_shared, g.up_shared,
                g.d_gate_shared, g.d_up_shared, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                d_hidden, nullptr, d_gate_up, nullptr, sems.bwd, phase,
                task.num_tokens, g.swiglu_limit, task.macrobatch_size, task.minibatch_size,
                task.macrobatch_idx, task.minibatch_idx, task.task_idx, task.cta_rank,
                task.d_hidden_tile_ready_base_index, task.replayed_gate_up_tile_ready_base_index,
                task.d_gate_up_row_block_ready_base_index, task.buffer_done_index, arena);
        } else {
            swiglu_bwd_kernel<false, Clamp>(view, g.d_hidden_routed, g.saved_gate, g.saved_up,
                g.d_gate, g.d_up, &g.gate_sc, &g.up_sc, &g.d_gate_sc, &g.d_up_sc,
                &g.d_gate_t, &g.d_gate_sc_t, &g.d_up_t, &g.d_up_sc_t,
                &g.router_weights, &g.d_router_weight_partials, &g.schedule_peer_rank,
                d_hidden, (task.flags & MOK_ACT_REPLAYED) ? &replayed_gate : nullptr,
                d_gate_up, (task.flags & MOK_ACT_BUFFER_DONE) ? &done : nullptr,
                sems.bwd, phase, task.num_tokens, g.swiglu_limit,
                task.macrobatch_size, task.minibatch_size, task.macrobatch_idx,
                task.minibatch_idx, task.task_idx, task.cta_rank,
                task.d_hidden_tile_ready_base_index, task.replayed_gate_up_tile_ready_base_index,
                task.d_gate_up_row_block_ready_base_index, task.buffer_done_index, arena);
        }
    }
#endif
};
} // namespace mok_act_owner

using MokActMXResidue = mok_act_owner::Owner<mok_port::RoutedPrecision::MXFP8, false>::Residue;
using MokActBFResidue = mok_act_owner::Owner<mok_port::RoutedPrecision::BF16, false>::Residue;
union MokActResidue { MokActMXResidue mxfp8; MokActBFResidue bf16; };

#ifndef MOK_ACT_PROVIDER
MOK_ACT_TASK_PL_TYPE;
MOK_COUNTERS_PL_TYPE;

#ifndef PIPELINK_EMIT
extern "C" __device__ __inline_hint__ void mok_act_init(const void* view) {
    auto& sems = *reinterpret_cast<mok_act_owner::Semaphores*>(
        mok_port::smem_claim_address(pipelink_mok_act_sems()));
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < mok_act_owner::config::SWIGLU_FWD_PIPE_DEPTH; ++i)
            kittens::init_semaphore(sems.fwd[i], 0, 1);
        #pragma unroll
        for (int i = 0; i < mok_act_owner::config::SWIGLU_BWD_PIPE_DEPTH; ++i)
            kittens::init_semaphore(sems.bwd[i], 0, 1);
    }
}
extern "C" __device__ __inline_hint__ void mok_act_fwd(const void* view, const MokActTask* task,
        const MokCounters* counters, uint32_t* phase) {
    using mok_act_owner::Owner;
    using mok_port::RoutedPrecision;
    if (pipelink_mok_precision() == MOK_PRECISION_MXFP8) {
        if (pipelink_mok_clamp()) Owner<RoutedPrecision::MXFP8, true>::fwd(view, *task, *counters, *phase);
        else Owner<RoutedPrecision::MXFP8, false>::fwd(view, *task, *counters, *phase);
    } else {
        if (pipelink_mok_clamp()) Owner<RoutedPrecision::BF16, true>::fwd(view, *task, *counters, *phase);
        else Owner<RoutedPrecision::BF16, false>::fwd(view, *task, *counters, *phase);
    }
}

#define MOK_ACT_PASS_EXPORT(PASS) \
extern "C" __device__ __inline_hint__ void mok_act_##PASS(const void* view, const MokActTask* task, \
        const MokCounters* counters, uint32_t* phase) { \
    using mok_act_owner::Owner; \
    using mok_port::RoutedPrecision; \
    if (pipelink_mok_precision() == MOK_PRECISION_MXFP8) { \
        if (pipelink_mok_clamp()) Owner<RoutedPrecision::MXFP8, true>::PASS(view, *task, *counters, *phase); \
        else Owner<RoutedPrecision::MXFP8, false>::PASS(view, *task, *counters, *phase); \
    } else { \
        if (pipelink_mok_clamp()) Owner<RoutedPrecision::BF16, true>::PASS(view, *task, *counters, *phase); \
        else Owner<RoutedPrecision::BF16, false>::PASS(view, *task, *counters, *phase); \
    } \
}
MOK_ACT_PASS_EXPORT(bwd)
MOK_ACT_PASS_EXPORT(rec)
#undef MOK_ACT_PASS_EXPORT

#endif
PL_STAGE("mok.act", "1.0.0");
PL_NO_PARAMS();
PL_TOPOLOGY(pl::topology{2, pl::cluster{{2,1,1}, pl::launch{2,{2,1,1}}}, true, "clc"});
PL_EXPORT(mok_act_init, mok_act_init);
PL_EXPORT(mok_act_fwd, mok_act_fwd);
PL_EXPORT(mok_act_bwd, mok_act_bwd);
PL_EXPORT(mok_act_rec, mok_act_rec);
PL_CALL(act_init, mok_act_init, pl::consume, "mok_act_init");
PL_CALL(act_fwd, mok_act_fwd, pl::consume, "mok_act_fwd");
PL_CALL(act_bwd, mok_act_bwd, pl::consume, "mok_act_bwd");
PL_CALL(act_rec, mok_act_rec, pl::consume, "mok_act_rec");
PL_VARIANT(0, {{"pipelink_mok_precision", 0}, {"pipelink_mok_clamp", 0}});
PL_VARIANT(1, {{"pipelink_mok_precision", 0}, {"pipelink_mok_clamp", 1}});
PL_VARIANT(2, {{"pipelink_mok_precision", 1}, {"pipelink_mok_clamp", 0}});
PL_VARIANT(3, {{"pipelink_mok_precision", 1}, {"pipelink_mok_clamp", 1}});
PL_CLAIM("pipelink_mok_act_arena", "smem", 229376, 1024, "phase:instr_act");
PL_CLAIM("pipelink_mok_act_sems", "smem", 40, 8, "kernel");
PL_ENDPOINT(gate_up_tile_ready, pl::endpoint("gate_up_tile_ready", pl::consume)
    .exports({"mok_act_fwd", "mok_act_rec"})
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1}).layout(pl::affine{{1},{1},32})
    .protocol(pl::counter{"pipelink_mok_cnt_gate_up_tile_ready", "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"}));
PL_ENDPOINT(hidden_row_block_ready, pl::endpoint("hidden_row_block_ready", pl::produce)
    .exports({"mok_act_fwd", "mok_act_rec"})
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1}).layout(pl::affine{{1},{1},32})
    .protocol(pl::counter{"pipelink_mok_cnt_hidden_row_block_ready"}));
PL_ENDPOINT(d_hidden_ready, pl::endpoint("d_hidden_ready", pl::consume)
    .exports({"mok_act_bwd"})
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1}).layout(pl::affine{{1},{1},32})
    .protocol(pl::counter{"pipelink_mok_cnt_d_hidden_ready", "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"}));
PL_ENDPOINT(d_gate_up_ready, pl::endpoint("d_gate_up_ready", pl::produce)
    .exports({"mok_act_bwd"})
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1}).layout(pl::affine{{1},{1},32})
    .protocol(pl::counter{"pipelink_mok_cnt_d_gate_up_ready"}));
PL_ENDPOINT(replayed_gate_up_ready, pl::endpoint("replayed_gate_up_ready", pl::consume)
    .exports({"mok_act_bwd"})
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1}).layout(pl::affine{{1},{1},32})
    .protocol(pl::counter{"pipelink_mok_cnt_replayed_gate_up_ready", "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"}));
PL_ENDPOINT(replayed_hidden_ready, pl::endpoint("replayed_hidden_ready", pl::produce)
    .exports({"mok_act_bwd"})
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1}).layout(pl::affine{{1},{1},32})
    .protocol(pl::counter{"pipelink_mok_cnt_replayed_hidden_ready"}));
PL_ENDPOINT(routed_buffers_done, pl::endpoint("routed_buffers_done", pl::produce)
    .exports({"mok_act_bwd"})
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1}).layout(pl::affine{{1},{1},32})
    .protocol(pl::counter{"pipelink_mok_cnt_routed_buffers_done"}));
PL_HOST_GROUP("mok.act", "mok.act.params", "mok_act", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "ep");
PL_HOST_ABI_EMPTY(100, MokActResidue);
PL_HOST_RESIDUE(2, "mok.act.residue", MokActResidue, 100, true, pl::host::roles({}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({}))
    .field(pl::host::scalar(1, "mok.num_local_tokens", "i64").caller("num_local_tokens"))
    .field(pl::host::scalar(2, "mok.hidden", "i64").caller("hidden"))
    .field(pl::host::scalar(3, "mok.intermediate", "i64").caller("intermediate"))
    .field(pl::host::scalar(5, "mok.schedule_capacity", "i64").caller("schedule_capacity"))
    .field(pl::host::scalar(6, "mok.macrobatch_size", "i64").caller("macrobatch_size"))
    .field(pl::host::scalar(10, "mok.swiglu_limit", "f64").caller("swiglu_limit"))
    .field(pl::host::scalar(11, "mok.pass", "i64").binding("pipelink_mok_pass"))
    .field(pl::host::scalar(12, "mok.precision", "i64").binding("pipelink_mok_precision"))
    .field(pl::host::scalar(13, "mok.world", "i64").binding("pipelink_mok_world"))
    .field(pl::host::scalar(14, "mok.clamp", "i64").binding("pipelink_mok_clamp"))
    .field(pl::host::scalar(25, "mok.gemm.MLP_Mb", "i64").fact("mok.gemm.MLP_Mb"))
    .field(pl::host::scalar(26, "mok.gemm.MLP_Nb", "i64").fact("mok.gemm.MLP_Nb")));
PL_HOST_FACT(pl::host::fact("mok.act.SWIGLU_Mb").i32(mok_act_owner::config::SWIGLU_Mb)
    .owner("mok.act").stage("mok.act").consumers({"mok.framework"}));
PL_HOST_FACT(pl::host::fact("mok.act.SWIGLU_Nb").i32(mok_act_owner::config::SWIGLU_Nb)
    .owner("mok.act").stage("mok.act").consumers({"mok.framework", "mok.comm"}));
PL_HOST_FACT(pl::host::fact("mok.act.SWIGLU_FWD_PIPE_DEPTH").i32(mok_act_owner::config::SWIGLU_FWD_PIPE_DEPTH)
    .owner("mok.act").stage("mok.act").consumers({"mok.framework"}));
PL_HOST_FACT(pl::host::fact("mok.act.SWIGLU_BWD_PIPE_DEPTH").i32(mok_act_owner::config::SWIGLU_BWD_PIPE_DEPTH)
    .owner("mok.act").stage("mok.act").consumers({"mok.framework"}));
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("mok.act.gate_shared", "bf16").rank(2).read().caller("gate_shared"));
PL_HOST_CHANNEL(2, uint16_t, pl::host::channel("mok.act.up_shared", "bf16").rank(2).read().caller("up_shared"));
PL_HOST_CHANNEL(3, uint16_t, pl::host::channel("mok.act.gate_routed", "bf16").rank(2).read().caller("gate_routed"));
PL_HOST_CHANNEL(4, uint16_t, pl::host::channel("mok.act.up_routed", "bf16").rank(2).read().caller("up_routed"));
PL_HOST_CHANNEL(5, uint16_t, pl::host::channel("mok.act.hidden_shared", "bf16").rank(2).write().caller("hidden_shared"));
PL_HOST_CHANNEL(6, uint8_t, pl::host::channel("mok.act.hidden_routed_fp8", "u8").rank(2).write().caller("hidden_routed_fp8"));
PL_HOST_CHANNEL(7, uint8_t, pl::host::channel("mok.act.hidden_routed_sc", "u8").rank(4).write().caller("hidden_routed_sc"));
PL_HOST_CHANNEL(8, uint8_t, pl::host::channel("mok.act.hidden_routed_t_fp8", "u8").rank(2).write().caller("hidden_routed_t_fp8"));
PL_HOST_CHANNEL(9, uint8_t, pl::host::channel("mok.act.hidden_routed_t_sc", "u8").rank(4).write().caller("hidden_routed_t_sc"));
PL_HOST_CHANNEL(10, uint16_t, pl::host::channel("mok.act.hidden_routed_bf16", "bf16").rank(2).write().caller("hidden_routed_bf16"));
PL_HOST_CHANNEL(11, uint8_t, pl::host::channel("mok.act.gate_routed_fp8", "u8").rank(2).read().caller("gate_routed_fp8"));
PL_HOST_CHANNEL(12, uint8_t, pl::host::channel("mok.act.gate_routed_sc", "u8").rank(4).read().caller("gate_routed_sc"));
PL_HOST_CHANNEL(13, uint8_t, pl::host::channel("mok.act.up_routed_fp8", "u8").rank(2).read().caller("up_routed_fp8"));
PL_HOST_CHANNEL(14, uint8_t, pl::host::channel("mok.act.up_routed_sc", "u8").rank(4).read().caller("up_routed_sc"));
PL_HOST_CHANNEL(15, uint16_t, pl::host::channel("mok.act.d_hidden_shared", "bf16").rank(2).read().caller("d_hidden_shared"));
PL_HOST_CHANNEL(16, uint16_t, pl::host::channel("mok.act.d_hidden_routed", "bf16").rank(2).read().caller("d_hidden_routed"));
PL_HOST_CHANNEL(17, uint16_t, pl::host::channel("mok.act.d_gate_shared", "bf16").rank(2).write().caller("d_gate_shared"));
PL_HOST_CHANNEL(18, uint16_t, pl::host::channel("mok.act.d_up_shared", "bf16").rank(2).write().caller("d_up_shared"));
PL_HOST_CHANNEL(19, uint8_t, pl::host::channel("mok.act.d_gate_routed_fp8", "u8").rank(2).write().caller("d_gate_routed_fp8"));
PL_HOST_CHANNEL(20, uint8_t, pl::host::channel("mok.act.d_gate_routed_sc", "u8").rank(4).write().caller("d_gate_routed_sc"));
PL_HOST_CHANNEL(21, uint8_t, pl::host::channel("mok.act.d_up_routed_fp8", "u8").rank(2).write().caller("d_up_routed_fp8"));
PL_HOST_CHANNEL(22, uint8_t, pl::host::channel("mok.act.d_up_routed_sc", "u8").rank(4).write().caller("d_up_routed_sc"));
PL_HOST_CHANNEL(23, uint8_t, pl::host::channel("mok.act.d_gate_routed_t_fp8", "u8").rank(2).write().caller("d_gate_routed_t_fp8"));
PL_HOST_CHANNEL(24, uint8_t, pl::host::channel("mok.act.d_gate_routed_t_sc", "u8").rank(4).write().caller("d_gate_routed_t_sc"));
PL_HOST_CHANNEL(25, uint8_t, pl::host::channel("mok.act.d_up_routed_t_fp8", "u8").rank(2).write().caller("d_up_routed_t_fp8"));
PL_HOST_CHANNEL(26, uint8_t, pl::host::channel("mok.act.d_up_routed_t_sc", "u8").rank(4).write().caller("d_up_routed_t_sc"));
PL_HOST_CHANNEL(27, uint16_t, pl::host::channel("mok.act.d_gate_routed_bf16", "bf16").rank(2).write().caller("d_gate_routed_bf16"));
PL_HOST_CHANNEL(28, uint16_t, pl::host::channel("mok.act.d_up_routed_bf16", "bf16").rank(2).write().caller("d_up_routed_bf16"));
PL_HOST_CHANNEL(29, float, pl::host::channel("mok.act.router_weights", "f32").rank(1).read().caller("router_weights"));
PL_HOST_CHANNEL(30, float, pl::host::channel("mok.act.d_router_weight_partials", "f32").rank(2).write().caller("d_router_weight_partials"));
PL_HOST_CHANNEL(31, int32_t, pl::host::channel("mok.act.schedule_peer_rank", "i32").rank(1).read().caller("schedule_peer_rank"));
PL_HOST_SCHEDULER(pl::host::scheduler("mok.framework"));
PL_HOST_TARGET();
#endif
