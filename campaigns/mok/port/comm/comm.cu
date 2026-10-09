// The provider includes this TU with MOK_COMM_PROVIDER for its private descriptor layout.
#include "comm_abi.h"
#include "mok_port.cuh"
#include "mxfp8.cuh"
#include <pipelink_host_abi_v2.h>
#include <array>
#include <type_traits>
#ifndef MOK_COMM_PROVIDER
#include <pipelink/stage.hpp>
extern "C" __device__ int pipelink_mok_precision();
extern "C" __device__ int pipelink_mok_world();
extern "C" __device__ int pipelink_mok_comm_arena();
extern "C" __device__ int pipelink_mok_comm_sems();
extern "C" __device__ int pipelink_mok_cnt_x_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_y_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_y_routed_done();
extern "C" __device__ int pipelink_mok_cnt_d_y_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_d_x_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_replayed_x_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_routed_buffers_done();
extern "C" __device__ int pipelink_mok_cnt_router_weights_ready();
#endif

namespace mok_comm_owner {
using namespace kittens;
struct config {
    static constexpr int DISPATCH_Mb = 128, DISPATCH_Nb = 512, DISPATCH_OUT_TILES = 2;
    static constexpr int COMBINE_Mb = 16, COMBINE_Nb = 1024, COMBINE_PIPE_DEPTH = 7;
    static constexpr int QUANT_Mb = mxfp8::config::Mb, QUANT_Nb = mxfp8::config::Nb;
    static constexpr int CLUSTER_SIZE = 2, NUM_CONSUMERS = 1, NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};
using quant_bf16_tile = mxfp8::globals::x_bf16_tile;
using quant_fp8_tile = mxfp8::globals::x_fp8_tile;
using quant_sc_tile = mxfp8::globals::x_sc_tile;
using mlp_fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, quant_fp8_tile>;
using epi_bf16_gl = gl<bf16, 1, 1, -1, -1>;
using routed_bf16_gl = gl<bf16, 1, 1, -1, -1>;
using sc_gl = gl<fp8e8m0, -1, -1, 32, 16, quant_sc_tile>;
using index_gl = mok_port::index_view;
using router_weight_gl = gl<float, 1, 1, 1, -1>;
using d_router_weight_partials_gl = gl<float, 1, 1, -1, -1>;
// Fixed capacity is the residue contract; only the selected world's entries are live.
using activation_bf16_pgl = std::array<bf16*, 8>;
using router_weight_pgl = std::array<float*, 8>;
struct unused_gl {};
struct Facts { int32_t MLP_Mb, MLP_Nb; };
struct Residue {
    mlp_fp8_gl x_routed, x_routed_t;
    sc_gl x_sc, x_sc_t;
    epi_bf16_gl y_routed;
    routed_bf16_gl x_bf16, x_bf16_t;
    unused_gl unused;
    activation_bf16_pgl x_buffer, combine_buffer;
    index_gl schedule_peer_rank, schedule_peer_token_idx;
    Facts facts;
    mlp_fp8_gl d_y_routed, d_y_routed_t;
    sc_gl d_y_sc, d_y_sc_t;
    routed_bf16_gl d_y_bf16;
    epi_bf16_gl d_x_routed;
    activation_bf16_pgl d_y_buffer, d_x_buffer;
    router_weight_pgl router_weight_buffer, d_router_weight_buffer;
    router_weight_gl router_weights;
    d_router_weight_partials_gl d_router_weight_partials;
};
struct Semaphores {
    semaphore dispatch_inputs_arrived;
    semaphore combine_inputs_arrived[config::COMBINE_PIPE_DEPTH];
};
static_assert(sizeof(Semaphores) == 64);
enum Channel { XBuffer, CombineBuffer, XRouted, XSc, XRoutedT, XScT, YRouted,
               SchedulePeerRank, SchedulePeerTokenIdx, XBF16, DYBuffer, DXBuffer,
               RouterWeightBuffer, DRouterWeightBuffer, RouterWeights, DRouterWeightPartials,
               DYRouted, DYSc, DYRoutedT, DYScT, DYBF16, DXRouted, ChannelCount };

#ifndef MOK_COMM_PROVIDER
__device__ __forceinline__ const Residue& residue(const void* view) {
    const auto& owner_view = *static_cast<const pipelink_host_view*>(view);
    return *static_cast<const Residue*>(owner_view.residue);
}
__device__ __forceinline__ Semaphores& semaphores() {
    return *reinterpret_cast<Semaphores*>(mok_port::smem_claim_address(pipelink_mok_comm_sems()));
}
template <mok_port::RoutedPrecision Precision, int NUM_DEVICES>
struct Owner {
    static_assert(NUM_DEVICES == 1 || NUM_DEVICES == 4);
    static constexpr bool USE_MXFP8 = Precision == mok_port::RoutedPrecision::MXFP8;
    using config = mok_comm_owner::config;
    using routed_activation_gl = std::conditional_t<USE_MXFP8, mlp_fp8_gl, routed_bf16_gl>;
    using routed_transposed_gl = routed_activation_gl;
    using routed_sc_gl = std::conditional_t<USE_MXFP8, sc_gl, unused_gl>;
    static __device__ __forceinline__ const Facts& facts(const void* view) { return residue(view).facts; }
#include "megakernel/utils.cuh"
#include "megakernel/dispatch_combine.cuh"
#include "megakernel/preload.cuh"
    static __device__ __forceinline__ void forward(const void* view, const MokCommTask& t,
                                                   const MokCounters& counters, uint32_t& phase) {
        const auto& g = residue(view);
        auto& s = semaphores();
        const index_gl x_ready{counters.base + pipelink_mok_cnt_x_routed_ready(), MOK_SLOTS_X_ROUTED_READY};
        const index_gl y_ready{counters.base + pipelink_mok_cnt_y_routed_ready(), MOK_SLOTS_Y_ROUTED_READY};
        const index_gl y_done{counters.base + pipelink_mok_cnt_y_routed_done(), MOK_SLOTS_Y_ROUTED_DONE};
        const uint64_t arena = mok_port::smem_claim_address(pipelink_mok_comm_arena());
        if (t.op == MOK_COMM_FWD_DISPATCH) {
            const auto& x = [&]() -> const routed_activation_gl& {
                if constexpr (USE_MXFP8) return g.x_routed; else return g.x_bf16;
            }();
            const auto& xt = [&]() -> const routed_transposed_gl& {
                if constexpr (USE_MXFP8) return g.x_routed_t; else return g.x_bf16_t;
            }();
            const auto& sc = [&]() -> const routed_sc_gl& {
                if constexpr (USE_MXFP8) return g.x_sc; else return g.unused;
            }();
            const auto& sct = [&]() -> const routed_sc_gl& {
                if constexpr (USE_MXFP8) return g.x_sc_t; else return g.unused;
            }();
            dispatch_kernel<false>(view, g.x_buffer, x, &sc, &xt, &sct, nullptr,
                g.schedule_peer_rank, g.schedule_peer_token_idx,
                t.flags & MOK_COMM_TRANSFER_READY ? &y_ready : nullptr, nullptr, x_ready,
                s.dispatch_inputs_arrived, phase, t.num_tokens, t.macrobatch_size,
                t.minibatch_size, t.macrobatch_idx, t.task_idx, t.row_divisor,
                t.previous_macrobatch_idx, t.buffer_ready_required_count, arena);
        } else if (t.op == MOK_COMM_FWD_COMBINE) {
            combine_kernel<true>(view, g.combine_buffer, g.y_routed, nullptr, nullptr,
                g.schedule_peer_rank, g.schedule_peer_token_idx, y_ready,
                t.flags & MOK_COMM_TRANSFER_DONE ? &y_done : nullptr,
                s.combine_inputs_arrived, phase, t.num_tokens, t.macrobatch_size,
                t.minibatch_size, t.macrobatch_idx, t.task_idx, arena);
        }
    }
    template <bool Gradient>
    static __device__ __forceinline__ void dispatch(
        const void* view, const MokCommTask& t, const index_gl& ready, uint32_t& phase) {
        const auto& g = residue(view);
        const auto& x = [&]() -> const routed_activation_gl& {
            if constexpr (USE_MXFP8) return Gradient ? g.d_y_routed : g.x_routed;
            else return Gradient ? g.d_y_bf16 : g.x_bf16;
        }();
        const auto& xt = [&]() -> const routed_transposed_gl& {
            if constexpr (USE_MXFP8) return Gradient ? g.d_y_routed_t : g.x_routed_t;
            else return x;
        }();
        const auto& sc = [&]() -> const routed_sc_gl& {
            if constexpr (USE_MXFP8) return Gradient ? g.d_y_sc : g.x_sc;
            else return g.unused;
        }();
        const auto& sct = [&]() -> const routed_sc_gl& {
            if constexpr (USE_MXFP8) return Gradient ? g.d_y_sc_t : g.x_sc_t;
            else return g.unused;
        }();
        dispatch_kernel<Gradient>(view, Gradient ? g.d_y_buffer : g.x_buffer, x, &sc, &xt, &sct,
            Gradient ? &g.router_weights : nullptr, g.schedule_peer_rank, g.schedule_peer_token_idx,
            nullptr, nullptr, ready, semaphores().dispatch_inputs_arrived, phase,
            t.num_tokens, t.macrobatch_size, t.minibatch_size, t.macrobatch_idx, t.task_idx,
            t.row_divisor, t.previous_macrobatch_idx, t.buffer_ready_required_count,
            mok_port::smem_claim_address(pipelink_mok_comm_arena()));
    }
    static __device__ __forceinline__ void backward(
        const void* view, const MokCommTask& t, const MokCounters& counters, uint32_t& phase) {
        const auto& g = residue(view);
        const index_gl dy_ready{counters.base + pipelink_mok_cnt_d_y_routed_ready(), MOK_SLOTS_D_Y_ROUTED_READY};
        const index_gl dx_ready{counters.base + pipelink_mok_cnt_d_x_routed_ready(), MOK_SLOTS_D_X_ROUTED_READY};
        const index_gl replay_ready{counters.base + pipelink_mok_cnt_replayed_x_routed_ready(), MOK_SLOTS_REPLAYED_X_ROUTED_READY};
        const index_gl buffers_done{counters.base + pipelink_mok_cnt_routed_buffers_done(), MOK_SLOTS_ROUTED_BUFFERS_DONE};
        const index_gl weights_ready{counters.base + pipelink_mok_cnt_router_weights_ready(), MOK_SLOTS_ROUTER_WEIGHTS_READY};
        switch (t.op) {
        case MOK_COMM_BWD_REVERSE_COMBINE:
            dispatch<true>(view, t, dy_ready, phase);
            break;
        case MOK_COMM_BWD_REVERSE_DISPATCH:
            combine_kernel<false>(view, g.d_x_buffer, g.d_x_routed, &g.d_router_weight_buffer,
                &g.d_router_weight_partials, g.schedule_peer_rank, g.schedule_peer_token_idx,
                dx_ready, t.flags & MOK_COMM_TRANSFER_DONE ? &buffers_done : nullptr,
                semaphores().combine_inputs_arrived, phase, t.num_tokens, t.macrobatch_size,
                t.minibatch_size, t.macrobatch_idx, t.task_idx,
                mok_port::smem_claim_address(pipelink_mok_comm_arena()));
            break;
        case MOK_COMM_BWD_REPLAY_DISPATCH:
            dispatch<false>(view, t, replay_ready, phase);
            break;
        case MOK_COMM_BWD_PRELOAD_ROUTER_WEIGHTS:
            preload_router_weights_kernel(g.router_weight_buffer, g.router_weights,
                g.schedule_peer_rank, g.schedule_peer_token_idx,
                t.flags & MOK_COMM_BUFFER_READY ? &buffers_done : nullptr, weights_ready,
                t.num_tokens, t.macrobatch_size, t.macrobatch_idx, t.task_idx, t.num_workers,
                t.previous_macrobatch_idx, t.buffer_ready_required_count);
            break;
        }
    }
    static __device__ __forceinline__ void recompute(
        const void* view, const MokCommTask& t, const MokCounters& counters, uint32_t& phase) {
        const index_gl ready{counters.base + pipelink_mok_cnt_x_routed_ready(), MOK_SLOTS_X_ROUTED_READY};
        dispatch<false>(view, t, ready, phase);
    }
};
#endif
} // namespace mok_comm_owner
using MokCommResidue = mok_comm_owner::Residue;

#ifndef MOK_COMM_PROVIDER
MOK_COUNTERS_PL_TYPE;
MOK_COMM_TASK_PL_TYPE;
#ifndef PIPELINK_EMIT
extern "C" __device__ __inline_hint__ void mok_comm_init(const void* view) {
    auto& s = mok_comm_owner::semaphores();
    if (threadIdx.x == 0) {
        kittens::init_semaphore(s.dispatch_inputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < mok_comm_owner::config::COMBINE_PIPE_DEPTH; ++i)
            kittens::init_semaphore(s.combine_inputs_arrived[i], 0, 1);
    }
}
extern "C" __device__ __inline_hint__ void mok_comm_fwd(
    const void* view, const MokCommTask* task, const MokCounters* counters, uint32_t* phase) {
    using mok_comm_owner::Owner;
    using mok_port::RoutedPrecision;
    if (pipelink_mok_precision() == MOK_PRECISION_MXFP8) {
        if (pipelink_mok_world() == 4) Owner<RoutedPrecision::MXFP8, 4>::forward(view, *task, *counters, *phase);
        else Owner<RoutedPrecision::MXFP8, 1>::forward(view, *task, *counters, *phase);
    } else {
        if (pipelink_mok_world() == 4) Owner<RoutedPrecision::BF16, 4>::forward(view, *task, *counters, *phase);
        else Owner<RoutedPrecision::BF16, 1>::forward(view, *task, *counters, *phase);
    }
}
extern "C" __device__ __inline_hint__ void mok_comm_bwd(
    const void* view, const MokCommTask* task, const MokCounters* counters, uint32_t* phase) {
    using mok_comm_owner::Owner;
    using mok_port::RoutedPrecision;
    if (pipelink_mok_precision() == MOK_PRECISION_MXFP8) {
        if (pipelink_mok_world() == 4) Owner<RoutedPrecision::MXFP8, 4>::backward(view, *task, *counters, *phase);
        else Owner<RoutedPrecision::MXFP8, 1>::backward(view, *task, *counters, *phase);
    } else {
        if (pipelink_mok_world() == 4) Owner<RoutedPrecision::BF16, 4>::backward(view, *task, *counters, *phase);
        else Owner<RoutedPrecision::BF16, 1>::backward(view, *task, *counters, *phase);
    }
}
extern "C" __device__ __inline_hint__ void mok_comm_rec(
    const void* view, const MokCommTask* task, const MokCounters* counters, uint32_t* phase) {
    using mok_comm_owner::Owner;
    using mok_port::RoutedPrecision;
    if (pipelink_mok_precision() == MOK_PRECISION_MXFP8) {
        if (pipelink_mok_world() == 4) Owner<RoutedPrecision::MXFP8, 4>::recompute(view, *task, *counters, *phase);
        else Owner<RoutedPrecision::MXFP8, 1>::recompute(view, *task, *counters, *phase);
    } else {
        if (pipelink_mok_world() == 4) Owner<RoutedPrecision::BF16, 4>::recompute(view, *task, *counters, *phase);
        else Owner<RoutedPrecision::BF16, 1>::recompute(view, *task, *counters, *phase);
    }
}
#endif
PL_STAGE("mok.comm", "1.0.0");
PL_NO_PARAMS();
PL_TOPOLOGY(pl::topology{2, pl::cluster{{2,1,1}, pl::launch{2,{2,1,1}}}, true, "clc"});
PL_VARIANT(0, {{"pipelink_mok_precision", 0}, {"pipelink_mok_world", 1}});
PL_VARIANT(1, {{"pipelink_mok_precision", 0}, {"pipelink_mok_world", 4}});
PL_VARIANT(2, {{"pipelink_mok_precision", 1}, {"pipelink_mok_world", 1}});
PL_VARIANT(3, {{"pipelink_mok_precision", 1}, {"pipelink_mok_world", 4}});
PL_EXPORT(mok_comm_init, mok_comm_init);
PL_EXPORT(mok_comm_fwd, mok_comm_fwd);
PL_CALL(comm_init, mok_comm_init, pl::consume, "mok_comm_init");
PL_CALL(comm_fwd, mok_comm_fwd, pl::consume, "mok_comm_fwd");
PL_EXPORT(mok_comm_bwd, mok_comm_bwd);
PL_EXPORT(mok_comm_rec, mok_comm_rec);
PL_CALL(comm_bwd, mok_comm_bwd, pl::consume, "mok_comm_bwd");
PL_CALL(comm_rec, mok_comm_rec, pl::consume, "mok_comm_rec");
PL_CLAIM("pipelink_mok_comm_arena", "smem", 229376, 1024, "phase:instr_comm");
PL_CLAIM("pipelink_mok_comm_sems", "smem", 64, 8, "kernel");
PL_ENDPOINT(x_routed_ready, pl::endpoint("x_routed_ready", pl::produce)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_x_routed_ready"})
    .exports({"mok_comm_fwd", "mok_comm_rec"}));
PL_ENDPOINT(y_routed_ready, pl::endpoint("y_routed_ready", pl::consume)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_y_routed_ready",
        "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"}).exports({"mok_comm_fwd"}));
PL_ENDPOINT(y_routed_done, pl::endpoint("y_routed_done", pl::produce)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_y_routed_done"})
    .exports({"mok_comm_fwd"}));
PL_ENDPOINT(d_y_routed_ready, pl::endpoint("d_y_routed_ready", pl::produce)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_d_y_routed_ready"})
    .exports({"mok_comm_bwd"}));
PL_ENDPOINT(d_x_routed_ready, pl::endpoint("d_x_routed_ready", pl::consume)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_d_x_routed_ready",
        "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"}).exports({"mok_comm_bwd"}));
PL_ENDPOINT(replayed_x_routed_ready, pl::endpoint("replayed_x_routed_ready", pl::produce)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_replayed_x_routed_ready"})
    .exports({"mok_comm_bwd"}));
PL_ENDPOINT(routed_buffers_done, pl::endpoint("routed_buffers_done", pl::produce)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_routed_buffers_done"})
    .exports({"mok_comm_bwd"}));
PL_ENDPOINT(routed_buffers_done, pl::endpoint("routed_buffers_done", pl::consume)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_routed_buffers_done",
        "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"}).exports({"mok_comm_bwd"}));
PL_ENDPOINT(router_weights_ready, pl::endpoint("router_weights_ready", pl::produce)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_router_weights_ready"})
    .exports({"mok_comm_bwd"}));
PL_ENDPOINT(router_weights_ready, pl::endpoint("router_weights_ready", pl::consume)
    .space("gmem").scope("grid").proxy("generic").tile("i32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_mok_cnt_router_weights_ready",
        "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"}).exports({"mok_comm_bwd"}));
PL_HOST_GROUP("mok.comm", "mok.comm.params", "mok_comm", plk_header_v2,
              PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "ep");
PL_HOST_ABI_EMPTY(100, MokCommResidue);
PL_HOST_RESIDUE(2, "mok.comm.residue", MokCommResidue, 100, true, pl::host::roles({}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({}))
    .field(pl::host::scalar(1, "mok.num_local_tokens", "i64").caller("num_local_tokens"))
    .field(pl::host::scalar(2, "mok.hidden", "i64").caller("hidden"))
    .field(pl::host::scalar(3, "mok.intermediate", "i64").caller("intermediate"))
    .field(pl::host::scalar(5, "mok.schedule_capacity", "i64").caller("schedule_capacity"))
    .field(pl::host::scalar(6, "mok.macrobatch_size", "i64").caller("macrobatch_size"))
    .field(pl::host::scalar(8, "mok.topk", "i64").caller("topk"))
    .field(pl::host::scalar(11, "mok.pass", "i64").binding("pipelink_mok_pass"))
    .field(pl::host::scalar(12, "mok.precision", "i64").binding("pipelink_mok_precision"))
    .field(pl::host::scalar(13, "mok.world", "i64").binding("pipelink_mok_world"))
    .field(pl::host::scalar(25, "mok.gemm.MLP_Mb", "i64").fact("mok.gemm.MLP_Mb"))
    .field(pl::host::scalar(26, "mok.gemm.MLP_Nb", "i64").fact("mok.gemm.MLP_Nb"))
    .field(pl::host::scalar(28, "mok.act.SWIGLU_Nb", "i64").fact("mok.act.SWIGLU_Nb")));
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("mok.comm.x_buffer", "bf16").rank(2).read().caller("x_buffer").peer("ep"));
PL_HOST_CHANNEL(2, uint16_t, pl::host::channel("mok.comm.combine_buffer", "bf16").rank(2).write().caller("combine_buffer").peer("ep"));
PL_HOST_CHANNEL(3, uint8_t, pl::host::channel("mok.comm.x_routed_fp8", "u8").rank(2).write().caller("x_routed_fp8"));
PL_HOST_CHANNEL(4, uint8_t, pl::host::channel("mok.comm.x_routed_sc", "u8").rank(4).write().caller("x_routed_sc"));
PL_HOST_CHANNEL(5, uint8_t, pl::host::channel("mok.comm.x_routed_t_fp8", "u8").rank(2).write().caller("x_routed_t_fp8"));
PL_HOST_CHANNEL(6, uint8_t, pl::host::channel("mok.comm.x_routed_t_sc", "u8").rank(4).write().caller("x_routed_t_sc"));
PL_HOST_CHANNEL(7, uint16_t, pl::host::channel("mok.comm.y_routed", "bf16").rank(2).read().caller("y_routed"));
PL_HOST_CHANNEL(8, int32_t, pl::host::channel("mok.comm.schedule_peer_rank", "i32").rank(1).read().caller("schedule_peer_rank"));
PL_HOST_CHANNEL(9, int32_t, pl::host::channel("mok.comm.schedule_peer_token_idx", "i32").rank(1).read().caller("schedule_peer_token_idx"));
PL_HOST_CHANNEL(10, uint16_t, pl::host::channel("mok.comm.x_routed_bf16", "bf16").rank(2).write().caller("x_routed_bf16"));
PL_HOST_CHANNEL(11, uint16_t, pl::host::channel("mok.comm.d_y_buffer", "bf16").rank(2).read().caller("d_y_buffer").peer("ep"));
PL_HOST_CHANNEL(12, uint16_t, pl::host::channel("mok.comm.d_x_routed_buffer", "bf16").rank(2).write().caller("d_x_routed_buffer").peer("ep"));
PL_HOST_CHANNEL(13, float, pl::host::channel("mok.comm.router_weight_buffer", "f32").rank(2).read().caller("router_weight_buffer").peer("ep"));
PL_HOST_CHANNEL(14, float, pl::host::channel("mok.comm.d_router_weight_buffer", "f32").rank(2).write().caller("d_router_weight_buffer").peer("ep"));
PL_HOST_CHANNEL(15, float, pl::host::channel("mok.comm.router_weights", "f32").rank(1).read_write().caller("router_weights"));
PL_HOST_CHANNEL(16, float, pl::host::channel("mok.comm.d_router_weight_partials", "f32").rank(2).read().caller("d_router_weight_partials"));
PL_HOST_CHANNEL(17, uint8_t, pl::host::channel("mok.comm.d_y_routed_fp8", "u8").rank(2).write().caller("d_y_routed_fp8"));
PL_HOST_CHANNEL(18, uint8_t, pl::host::channel("mok.comm.d_y_routed_sc", "u8").rank(4).write().caller("d_y_routed_sc"));
PL_HOST_CHANNEL(19, uint8_t, pl::host::channel("mok.comm.d_y_routed_t_fp8", "u8").rank(2).write().caller("d_y_routed_t_fp8"));
PL_HOST_CHANNEL(20, uint8_t, pl::host::channel("mok.comm.d_y_routed_t_sc", "u8").rank(4).write().caller("d_y_routed_t_sc"));
PL_HOST_CHANNEL(21, uint16_t, pl::host::channel("mok.comm.d_y_routed_bf16", "bf16").rank(2).write().caller("d_y_routed_bf16"));
PL_HOST_CHANNEL(22, uint16_t, pl::host::channel("mok.comm.d_x_routed", "bf16").rank(2).read().caller("d_x_routed"));
PL_HOST_FACT(pl::host::fact("mok.comm.DISPATCH_Mb").i32(mok_comm_owner::config::DISPATCH_Mb)
    .owner("mok.comm").stage("mok.comm").consumers({"mok.framework", "mok.gemm"}));
PL_HOST_FACT(pl::host::fact("mok.comm.DISPATCH_Nb").i32(mok_comm_owner::config::DISPATCH_Nb)
    .owner("mok.comm").stage("mok.comm").consumers({"mok.framework", "mok.gemm"}));
PL_HOST_FACT(pl::host::fact("mok.comm.COMBINE_Mb").i32(mok_comm_owner::config::COMBINE_Mb)
    .owner("mok.comm").stage("mok.comm").consumers({"mok.framework", "mok.gemm"}));
PL_HOST_FACT(pl::host::fact("mok.comm.COMBINE_Nb").i32(mok_comm_owner::config::COMBINE_Nb)
    .owner("mok.comm").stage("mok.comm").consumers({"mok.framework", "mok.gemm"}));
PL_HOST_FACT(pl::host::fact("mok.comm.COMBINE_PIPE_DEPTH").i32(mok_comm_owner::config::COMBINE_PIPE_DEPTH)
    .owner("mok.comm").stage("mok.comm").consumers({"mok.framework"}));
PL_HOST_SCHEDULER(pl::host::scheduler("mok.framework"));
PL_HOST_TARGET();
#endif
