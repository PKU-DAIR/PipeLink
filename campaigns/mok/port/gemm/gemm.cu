#include "gemm_abi.h"
#include "mok_port.cuh"
#include "mxfp8.cuh"
#include <pipelink/stage.hpp>
#include "compact_gl.cuh"

extern "C" __device__ int pipelink_mok_precision();
extern "C" __device__ int pipelink_mok_gemm_arena();
extern "C" __device__ int pipelink_mok_gemm_sems();
extern "C" __device__ int pipelink_mok_cnt_gate_up_tile_ready();
extern "C" __device__ int pipelink_mok_cnt_hidden_row_block_ready();
extern "C" __device__ int pipelink_mok_cnt_x_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_y_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_y_routed_done();
extern "C" __device__ int pipelink_mok_cnt_d_y_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_d_hidden_ready();
extern "C" __device__ int pipelink_mok_cnt_d_gate_up_ready();
extern "C" __device__ int pipelink_mok_cnt_d_x_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_replayed_x_routed_ready();
extern "C" __device__ int pipelink_mok_cnt_replayed_gate_up_ready();
extern "C" __device__ int pipelink_mok_cnt_replayed_hidden_ready();
extern "C" __device__ int pipelink_mok_cnt_routed_buffers_done();

namespace mok_gemm_owner {
using namespace kittens;
struct config {
  static constexpr int MLP_Mb = 256, MLP_Nb = 256;
  static constexpr int MLP_FP8_Kb = 128, MLP_BF16_Kb = 64;
  static constexpr int MLP_SUPERGROUP_SIZE = 8, MLP_LOAD_PIPE_DEPTH = 6;
  static constexpr int MLP_EPI_PIPE_DEPTH = 8, MLP_NUM_BF16_D_TILES = 3;
  static constexpr int MLP_NUM_FP8_D_TILES = 4;
  static constexpr int QUANT_Mb = mxfp8::config::Mb, QUANT_Nb = mxfp8::config::Nb;
  static constexpr int CLUSTER_SIZE = 2, NUM_CONSUMERS = 1, NUM_PRODUCERS = 1;
  static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS;
  static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};
struct Facts {
  int DISPATCH_Mb, DISPATCH_Nb, COMBINE_Mb, COMBINE_Nb, EXPERT_PADDING;
};
struct Semaphores {
  semaphore inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
  semaphore scales_arrived[config::MLP_LOAD_PIPE_DEPTH];
  semaphore inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
  semaphore scales_finished[config::MLP_LOAD_PIPE_DEPTH];
  semaphore outputs_arrived, outputs_finished;
  uint32_t tmem_address;
};
static_assert(sizeof(Semaphores) == 216 && alignof(Semaphores) == 8);

__device__ __forceinline__ Semaphores& semaphores() {
  return *reinterpret_cast<Semaphores*>(mok_port::smem_claim_address(pipelink_mok_gemm_sems()));
}

template<mok_port::RoutedPrecision PRECISION>
struct Owner {
  using config = mok_gemm_owner::config;
  static constexpr bool USE_MXFP8 = PRECISION == mok_port::RoutedPrecision::MXFP8;
  template<typename T, int B, int D, int R, int C, typename... Tiles>
  using gl = compact_gl<T, B, D, R, C, Tiles...>;
  using mlp_fp8_tile = st_fp8e4m3<config::MLP_Mb / 2, config::MLP_FP8_Kb>;
  using mlp_bf16_tile = st_bf<config::MLP_Mb / 2, config::MLP_BF16_Kb>;
  using mlp_bf16_t_tile = st_bf<config::MLP_BF16_Kb, config::MLP_Mb / 2>;
  using mlp_sc_tile = st_fp8e8m0<32, 16, false>;
  using mlp_bf16_d_tile = st_bf<config::MLP_Mb / 2, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH>;
  using mlp_fp8_d_tile = st_fp8e4m3<config::MLP_Mb / 2, 32>;
  using mlp_bf16_gl = gl<bf16, 1, 1, -1, -1, mlp_bf16_tile, mlp_bf16_t_tile>;
  using epi_bf16_gl = gl<bf16, 1, 1, -1, -1, mlp_bf16_d_tile>;
  using wgrad_bf16_gl = gl<bf16, 1, 1, -1, -1, mlp_bf16_t_tile>;
  using mlp_fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, mlp_fp8_tile>;
  using gate_up_fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, mlp_fp8_d_tile>;
  using weight_bf16_gl = gl<bf16, 1, -1, -1, -1, mlp_bf16_tile, mlp_bf16_t_tile>;
  using d_weight_bf16_gl = gl<bf16, 1, -1, -1, -1, mlp_bf16_d_tile>;
  using weight_fp8_gl = gl<fp8e4m3, 1, -1, -1, -1, mlp_fp8_tile>;
  using sc_gl = gl<fp8e8m0, -1, -1, 32, 16, mlp_sc_tile>;
  using index_gl = mok_port::index_view;
  using routed_bf16_gl = gl<bf16, 1, 1, -1, -1, mlp_bf16_tile, mlp_bf16_t_tile, mlp_bf16_d_tile>;
  using routed_activation_gl = std::conditional_t<USE_MXFP8, mlp_fp8_gl, routed_bf16_gl>;
  using routed_gate_up_gl = std::conditional_t<USE_MXFP8, gate_up_fp8_gl, routed_bf16_gl>;
  using routed_weight_gl = std::conditional_t<USE_MXFP8, weight_fp8_gl, weight_bf16_gl>;
  struct unused_gl {};
  using routed_sc_gl = std::conditional_t<USE_MXFP8, sc_gl, unused_gl>;
  struct Residue {
    mlp_bf16_gl x_shared, hidden_shared;
    weight_bf16_gl w_shared_gate, w_shared_up, w_shared_down;
    routed_activation_gl x_fp8_routed, hidden_fp8_routed;
    routed_weight_gl w_routed_gate, w_routed_up, w_routed_down;
    routed_sc_gl x_sc_routed, hidden_sc_routed;
    routed_sc_gl w_routed_gate_sc, w_routed_up_sc, w_routed_down_sc;
    epi_bf16_gl gate_shared, up_shared, gate_routed, up_routed, y_shared, y_routed;
    routed_gate_up_gl gate_fp8_routed, up_fp8_routed;
    routed_sc_gl gate_sc_routed, up_sc_routed;
    wgrad_bf16_gl x_shared_wgrad, hidden_shared_wgrad;
    mlp_bf16_gl d_y_shared, d_gate_shared, d_up_shared;
    epi_bf16_gl d_hidden_shared, d_hidden_routed, d_x_shared, d_x_routed;
    d_weight_bf16_gl d_w_shared_gate, d_w_shared_up, d_w_shared_down;
    d_weight_bf16_gl d_w_routed_gate, d_w_routed_up, d_w_routed_down;
    routed_activation_gl x_fp8_t_routed, hidden_fp8_t_routed;
    routed_activation_gl d_y_fp8_routed, d_y_fp8_t_routed;
    routed_activation_gl d_gate_fp8_routed, d_gate_fp8_t_routed;
    routed_activation_gl d_up_fp8_routed, d_up_fp8_t_routed;
    routed_weight_gl w_routed_gate_T, w_routed_up_T, w_routed_down_T;
    routed_sc_gl x_sc_t_routed, hidden_sc_t_routed;
    routed_sc_gl d_y_sc_routed, d_y_sc_t_routed;
    routed_sc_gl d_gate_sc_routed, d_gate_sc_t_routed;
    routed_sc_gl d_up_sc_routed, d_up_sc_t_routed;
    routed_sc_gl w_routed_gate_sc_T, w_routed_up_sc_T, w_routed_down_sc_T;
    index_gl tokens_per_expert;
    Facts facts;
  };
  static __device__ __forceinline__ const Residue& residue(const void* view) {
    const auto& owner_view = *static_cast<const pipelink_host_view*>(view);
    return *static_cast<const Residue*>(owner_view.residue);
  }
  static __device__ __forceinline__ const Facts& facts(const void* view) { return residue(view).facts; }

#include "megakernel/utils.cuh"
#include "megakernel/grouped_gemm.cuh"

  static __device__ __forceinline__ void fwd(const void* view, const MokGemmTask& t, const MokCounters& counters, uint32_t& phase) {
    const auto& g = residue(view);
    auto& s = semaphores();
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt(s.tmem_address);
    full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> a_sc_tt(s.tmem_address + 256);
    full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> b_sc_tt(s.tmem_address + 384);
    const index_gl gate_up{counters.base + pipelink_mok_cnt_gate_up_tile_ready(), MOK_SLOTS_GATE_UP_TILE_READY};
    const index_gl hidden{counters.base + pipelink_mok_cnt_hidden_row_block_ready(), MOK_SLOTS_HIDDEN_ROW_BLOCK_READY};
    const index_gl x_ready{counters.base + pipelink_mok_cnt_x_routed_ready(), MOK_SLOTS_X_ROUTED_READY};
    const index_gl y_ready{counters.base + pipelink_mok_cnt_y_routed_ready(), MOK_SLOTS_Y_ROUTED_READY};
    const index_gl y_done{counters.base + pipelink_mok_cnt_y_routed_done(), MOK_SLOTS_Y_ROUTED_DONE};
    const uint64_t arena = mok_port::smem_claim_address(pipelink_mok_gemm_arena());
#define MOK_GEMM_TAIL \
    d_tt, a_sc_tt, b_sc_tt, s.inputs_arrived, s.scales_arrived, s.inputs_finished, \
    s.scales_finished, s.outputs_arrived, s.outputs_finished, phase, \
    t.num_tokens, t.macrobatch_size, t.minibatch_size, t.macrobatch_idx, t.minibatch_idx, \
    t.task_idx, t.cta_rank, t.input_minibatch_ready_num_cols, t.input_row_block_ready_base_index, \
    t.input_row_block_ready_required_count, t.output_tile_ready_base_index, t.buffer_done_index, arena
    switch (t.op) {
      case MOK_GEMM_FWD_SHARED_GATE:
        expert_grouped_gemm_kernel<true>(view, g.x_shared, g.w_shared_gate, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            g.gate_shared, nullptr, nullptr, g.tokens_per_expert, nullptr, nullptr, nullptr, &gate_up, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_FWD_SHARED_UP:
        expert_grouped_gemm_kernel<true>(view, g.x_shared, g.w_shared_up, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            g.up_shared, nullptr, nullptr, g.tokens_per_expert, nullptr, nullptr, nullptr, &gate_up, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_FWD_SHARED_DOWN:
        expert_grouped_gemm_kernel<true>(view, g.hidden_shared, g.w_shared_down, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            g.y_shared, nullptr, nullptr, g.tokens_per_expert, nullptr, &hidden, nullptr, nullptr, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_FWD_ROUTED_GATE:
        expert_grouped_gemm_kernel<false>(view, g.x_fp8_routed, g.w_routed_gate, &g.x_sc_routed, &g.w_routed_gate_sc, nullptr, nullptr, nullptr, nullptr,
            g.gate_routed, &g.gate_fp8_routed, &g.gate_sc_routed, g.tokens_per_expert, &x_ready, nullptr, nullptr, &gate_up, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_FWD_ROUTED_UP:
        expert_grouped_gemm_kernel<false>(view, g.x_fp8_routed, g.w_routed_up, &g.x_sc_routed, &g.w_routed_up_sc, nullptr, nullptr, nullptr, nullptr,
            g.up_routed, &g.up_fp8_routed, &g.up_sc_routed, g.tokens_per_expert, &x_ready, nullptr, nullptr, &gate_up, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_FWD_ROUTED_DOWN:
        expert_grouped_gemm_kernel<false>(view, g.hidden_fp8_routed, g.w_routed_down, &g.hidden_sc_routed, &g.w_routed_down_sc, nullptr, nullptr, nullptr, nullptr,
            g.y_routed, nullptr, nullptr, g.tokens_per_expert, nullptr, &hidden,
            (t.flags & MOK_GEMM_OUTPUT_ROW_READY) ? &y_done : nullptr, nullptr, &y_ready, nullptr, MOK_GEMM_TAIL);
        break;
    }
#undef MOK_GEMM_TAIL
  }
  static __device__ __forceinline__ void bwd(const void* view, const MokGemmTask& t, const MokCounters& counters, uint32_t& phase) {
    const auto& g = residue(view);
    auto& s = semaphores();
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt(s.tmem_address);
    full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> a_sc_tt(s.tmem_address + 256);
    full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> b_sc_tt(s.tmem_address + 384);
    const uint64_t arena = mok_port::smem_claim_address(pipelink_mok_gemm_arena());
#define MOK_GEMM_VIEW(NAME, EXTENT) const index_gl NAME{counters.base + pipelink_mok_cnt_##NAME(), EXTENT}
    MOK_GEMM_VIEW(d_y_routed_ready, MOK_SLOTS_D_Y_ROUTED_READY);
    MOK_GEMM_VIEW(d_hidden_ready, MOK_SLOTS_D_HIDDEN_READY);
    MOK_GEMM_VIEW(d_gate_up_ready, MOK_SLOTS_D_GATE_UP_READY);
    MOK_GEMM_VIEW(d_x_routed_ready, MOK_SLOTS_D_X_ROUTED_READY);
    MOK_GEMM_VIEW(replayed_x_routed_ready, MOK_SLOTS_REPLAYED_X_ROUTED_READY);
    MOK_GEMM_VIEW(replayed_gate_up_ready, MOK_SLOTS_REPLAYED_GATE_UP_READY);
    MOK_GEMM_VIEW(replayed_hidden_ready, MOK_SLOTS_REPLAYED_HIDDEN_READY);
    MOK_GEMM_VIEW(routed_buffers_done, MOK_SLOTS_ROUTED_BUFFERS_DONE);
#undef MOK_GEMM_VIEW
    const index_gl* done = (t.flags & MOK_GEMM_BUFFER_DONE) ? &routed_buffers_done : nullptr;
    const bool replayed = t.flags & MOK_GEMM_REPLAYED;
#define MOK_GEMM_TAIL \
    d_tt, a_sc_tt, b_sc_tt, s.inputs_arrived, s.scales_arrived, s.inputs_finished, \
    s.scales_finished, s.outputs_arrived, s.outputs_finished, phase, \
    t.num_tokens, t.macrobatch_size, t.minibatch_size, t.macrobatch_idx, t.minibatch_idx, \
    t.task_idx, t.cta_rank, t.input_minibatch_ready_num_cols, t.input_row_block_ready_base_index, \
    t.input_row_block_ready_required_count, t.output_tile_ready_base_index, t.buffer_done_index, arena
    switch (t.op) {
      case MOK_GEMM_BWD_SHARED_DGRAD_DOWN:
        expert_grouped_gemm_kernel<true, false, true>(view, g.d_y_shared, g.w_shared_down, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            g.d_hidden_shared, nullptr, nullptr, g.tokens_per_expert, nullptr, nullptr, nullptr, &d_hidden_ready, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_SHARED_DGRAD_GATE_UP:
        expert_grouped_gemm_kernel<true, false, true>(view, g.d_gate_shared, g.w_shared_gate, nullptr, nullptr, &g.d_up_shared, &g.w_shared_up, nullptr, nullptr,
            g.d_x_shared, nullptr, nullptr, g.tokens_per_expert, nullptr, &d_gate_up_ready, nullptr, nullptr, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_SHARED_WGRAD_DOWN:
        expert_grouped_gemm_kernel<true, true>(view, g.d_y_shared, g.hidden_shared_wgrad, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            g.d_w_shared_down, nullptr, nullptr, g.tokens_per_expert, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_SHARED_WGRAD_GATE:
        expert_grouped_gemm_kernel<true, true>(view, g.d_gate_shared, g.x_shared_wgrad, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            g.d_w_shared_gate, nullptr, nullptr, g.tokens_per_expert, nullptr, &d_gate_up_ready, nullptr, nullptr, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_SHARED_WGRAD_UP:
        expert_grouped_gemm_kernel<true, true>(view, g.d_up_shared, g.x_shared_wgrad, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            g.d_w_shared_up, nullptr, nullptr, g.tokens_per_expert, nullptr, &d_gate_up_ready, nullptr, nullptr, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_REPLAY_GATE:
        expert_grouped_gemm_kernel<false>(view, g.x_fp8_routed, g.w_routed_gate, &g.x_sc_routed, &g.w_routed_gate_sc, nullptr, nullptr, nullptr, nullptr,
            g.gate_routed, &g.gate_fp8_routed, &g.gate_sc_routed, g.tokens_per_expert, &replayed_x_routed_ready, nullptr, nullptr, &replayed_gate_up_ready, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_REPLAY_UP:
        expert_grouped_gemm_kernel<false>(view, g.x_fp8_routed, g.w_routed_up, &g.x_sc_routed, &g.w_routed_up_sc, nullptr, nullptr, nullptr, nullptr,
            g.up_routed, &g.up_fp8_routed, &g.up_sc_routed, g.tokens_per_expert, &replayed_x_routed_ready, nullptr, nullptr, &replayed_gate_up_ready, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_ROUTED_DGRAD_DOWN:
        expert_grouped_gemm_kernel<false, false, !USE_MXFP8>(view, g.d_y_fp8_routed, g.w_routed_down_T, &g.d_y_sc_routed, &g.w_routed_down_sc_T, nullptr, nullptr, nullptr, nullptr,
            g.d_hidden_routed, nullptr, nullptr, g.tokens_per_expert, &d_y_routed_ready, nullptr, nullptr, &d_hidden_ready, nullptr, done, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_ROUTED_DGRAD_GATE_UP:
        expert_grouped_gemm_kernel<false, false, !USE_MXFP8>(view, g.d_gate_fp8_routed, g.w_routed_gate_T, &g.d_gate_sc_routed, &g.w_routed_gate_sc_T,
            &g.d_up_fp8_routed, &g.w_routed_up_T, &g.d_up_sc_routed, &g.w_routed_up_sc_T,
            g.d_x_routed, nullptr, nullptr, g.tokens_per_expert, nullptr, &d_gate_up_ready, nullptr, nullptr, &d_x_routed_ready, done, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_ROUTED_WGRAD_DOWN:
        expert_grouped_gemm_kernel<false, true>(view, g.d_y_fp8_t_routed, g.hidden_fp8_t_routed, &g.d_y_sc_t_routed, &g.hidden_sc_t_routed, nullptr, nullptr, nullptr, nullptr,
            g.d_w_routed_down, nullptr, nullptr, g.tokens_per_expert, &d_y_routed_ready, replayed ? &replayed_hidden_ready : nullptr, nullptr, nullptr, nullptr, done, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_ROUTED_WGRAD_GATE:
        expert_grouped_gemm_kernel<false, true>(view, g.d_gate_fp8_t_routed, g.x_fp8_t_routed, &g.d_gate_sc_t_routed, &g.x_sc_t_routed, nullptr, nullptr, nullptr, nullptr,
            g.d_w_routed_gate, nullptr, nullptr, g.tokens_per_expert, replayed ? &replayed_x_routed_ready : nullptr, &d_gate_up_ready, nullptr, nullptr, nullptr, done, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_BWD_ROUTED_WGRAD_UP:
        expert_grouped_gemm_kernel<false, true>(view, g.d_up_fp8_t_routed, g.x_fp8_t_routed, &g.d_up_sc_t_routed, &g.x_sc_t_routed, nullptr, nullptr, nullptr, nullptr,
            g.d_w_routed_up, nullptr, nullptr, g.tokens_per_expert, replayed ? &replayed_x_routed_ready : nullptr, &d_gate_up_ready, nullptr, nullptr, nullptr, done, MOK_GEMM_TAIL);
        break;
    }
#undef MOK_GEMM_TAIL
  }

  static __device__ __forceinline__ void rec(const void* view, const MokGemmTask& t, const MokCounters& counters, uint32_t& phase) {
    const auto& g = residue(view);
    auto& s = semaphores();
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt(s.tmem_address);
    full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> a_sc_tt(s.tmem_address + 256);
    full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> b_sc_tt(s.tmem_address + 384);
    const index_gl gate_up{counters.base + pipelink_mok_cnt_gate_up_tile_ready(), MOK_SLOTS_GATE_UP_TILE_READY};
    const index_gl x_ready{counters.base + pipelink_mok_cnt_x_routed_ready(), MOK_SLOTS_X_ROUTED_READY};
    const uint64_t arena = mok_port::smem_claim_address(pipelink_mok_gemm_arena());
#define MOK_GEMM_TAIL \
    d_tt, a_sc_tt, b_sc_tt, s.inputs_arrived, s.scales_arrived, s.inputs_finished, \
    s.scales_finished, s.outputs_arrived, s.outputs_finished, phase, \
    t.num_tokens, t.macrobatch_size, t.minibatch_size, t.macrobatch_idx, t.minibatch_idx, \
    t.task_idx, t.cta_rank, t.input_minibatch_ready_num_cols, t.input_row_block_ready_base_index, \
    t.input_row_block_ready_required_count, t.output_tile_ready_base_index, t.buffer_done_index, arena
    switch (t.op) {
      case MOK_GEMM_REC_SHARED_GATE:
        expert_grouped_gemm_kernel<true>(view, g.x_shared, g.w_shared_gate, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            g.gate_shared, nullptr, nullptr, g.tokens_per_expert, nullptr, nullptr, nullptr, &gate_up, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_REC_SHARED_UP:
        expert_grouped_gemm_kernel<true>(view, g.x_shared, g.w_shared_up, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            g.up_shared, nullptr, nullptr, g.tokens_per_expert, nullptr, nullptr, nullptr, &gate_up, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_REC_ROUTED_GATE:
        expert_grouped_gemm_kernel<false>(view, g.x_fp8_routed, g.w_routed_gate, &g.x_sc_routed, &g.w_routed_gate_sc, nullptr, nullptr, nullptr, nullptr,
            g.gate_routed, &g.gate_fp8_routed, &g.gate_sc_routed, g.tokens_per_expert, &x_ready, nullptr, nullptr, &gate_up, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
      case MOK_GEMM_REC_ROUTED_UP:
        expert_grouped_gemm_kernel<false>(view, g.x_fp8_routed, g.w_routed_up, &g.x_sc_routed, &g.w_routed_up_sc, nullptr, nullptr, nullptr, nullptr,
            g.up_routed, &g.up_fp8_routed, &g.up_sc_routed, g.tokens_per_expert, &x_ready, nullptr, nullptr, &gate_up, nullptr, nullptr, MOK_GEMM_TAIL);
        break;
    }
#undef MOK_GEMM_TAIL
  }
};
}  // namespace mok_gemm_owner

union MokGemmResidue {
  mok_gemm_owner::Owner<mok_port::RoutedPrecision::MXFP8>::Residue mxfp8;
  mok_gemm_owner::Owner<mok_port::RoutedPrecision::BF16>::Residue bf16;
};

#ifndef MOK_GEMM_TYPES_ONLY
MOK_COUNTERS_PL_TYPE;
MOK_GEMM_TASK_PL_TYPE;
#ifndef PIPELINK_EMIT
extern "C" __device__ __inline_hint__ void mok_gemm_init(const void* view) {
  using namespace mok_gemm_owner;
  auto& s = semaphores();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < config::MLP_LOAD_PIPE_DEPTH; ++i) {
      init_semaphore(s.inputs_arrived[i], 0, 1);
      init_semaphore(s.scales_arrived[i], 0, 1);
      init_semaphore(s.inputs_finished[i], 0, 1);
      init_semaphore(s.scales_finished[i], 0, 1);
    }
    init_semaphore(s.outputs_arrived, 0, 1);
    init_semaphore(s.outputs_finished, 0, config::CLUSTER_SIZE);
  }
  tensor_allocator<1, config::CLUSTER_SIZE, false> allocator;
  // tcgen05.alloc.cta_group::2 accesses the peer CTA's shared memory, and CUDA requires every CTA
  // of the cluster to have started before that (distributed shared memory rules). Upstream omits
  // this barrier; racecheck reports the cross-CTA access without it.
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
  if (warpid() == 0) allocator.provision(s.tmem_address);
  asm volatile("tcgen05.fence::before_thread_sync;\n");
  asm volatile("bar.sync 0;\n");
  asm volatile("tcgen05.fence::after_thread_sync;\n");
}
extern "C" __device__ __inline_hint__ void mok_gemm_fini(const void* view) {
  using namespace mok_gemm_owner;
  if (warpid() == 0) {
    tensor_allocator<1, config::CLUSTER_SIZE, false> allocator;
    allocator.set_addr(semaphores().tmem_address);
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");
    allocator.deprovision();
  }
}
extern "C" __device__ __inline_hint__ void mok_gemm_fwd(const void* view, const MokGemmTask* task, const MokCounters* counters, uint32_t* phase) {
  if (pipelink_mok_precision() == MOK_PRECISION_MXFP8)
    mok_gemm_owner::Owner<mok_port::RoutedPrecision::MXFP8>::fwd(view, *task, *counters, *phase);
  else
    mok_gemm_owner::Owner<mok_port::RoutedPrecision::BF16>::fwd(view, *task, *counters, *phase);
}
extern "C" __device__ __inline_hint__ void mok_gemm_bwd(const void* view, const MokGemmTask* task, const MokCounters* counters, uint32_t* phase) {
  if (pipelink_mok_precision() == MOK_PRECISION_MXFP8)
    mok_gemm_owner::Owner<mok_port::RoutedPrecision::MXFP8>::bwd(view, *task, *counters, *phase);
  else
    mok_gemm_owner::Owner<mok_port::RoutedPrecision::BF16>::bwd(view, *task, *counters, *phase);
}
extern "C" __device__ __inline_hint__ void mok_gemm_rec(const void* view, const MokGemmTask* task, const MokCounters* counters, uint32_t* phase) {
  if (pipelink_mok_precision() == MOK_PRECISION_MXFP8)
    mok_gemm_owner::Owner<mok_port::RoutedPrecision::MXFP8>::rec(view, *task, *counters, *phase);
  else
    mok_gemm_owner::Owner<mok_port::RoutedPrecision::BF16>::rec(view, *task, *counters, *phase);
}
#endif

PL_STAGE("mok.gemm", "f3a8bd2+pipelink.p1");
PL_NO_PARAMS();
PL_TOPOLOGY(pl::topology{2, pl::cluster{{2,1,1}, pl::launch{2,{2,1,1}}}, true, "clc"});
PL_VARIANT(0, {{"pipelink_mok_precision", MOK_PRECISION_BF16}});
PL_VARIANT(1, {{"pipelink_mok_precision", MOK_PRECISION_MXFP8}});
PL_EXPORT(mok_gemm_init, mok_gemm_init);
PL_EXPORT(mok_gemm_fini, mok_gemm_fini);
PL_EXPORT(mok_gemm_fwd, mok_gemm_fwd);
PL_EXPORT(mok_gemm_bwd, mok_gemm_bwd);
PL_EXPORT(mok_gemm_rec, mok_gemm_rec);
PL_CALL(gemm_init, mok_gemm_init, pl::consume, "mok_gemm_init");
PL_CALL(gemm_fini, mok_gemm_fini, pl::consume, "mok_gemm_fini");
PL_CALL(gemm_fwd, mok_gemm_fwd, pl::consume, "mok_gemm_fwd");
PL_CALL(gemm_bwd, mok_gemm_bwd, pl::consume, "mok_gemm_bwd");
PL_CALL(gemm_rec, mok_gemm_rec, pl::consume, "mok_gemm_rec");
PL_CLAIM("pipelink_mok_gemm_arena", "smem", 230400, 1024, "phase:instr_gemm");
PL_CLAIM("pipelink_mok_gemm_sems", "smem", sizeof(mok_gemm_owner::Semaphores), 8, "kernel");
PL_CLAIM("pipelink_mok_gemm_tmem", "tmem", 512, 32, "kernel");
#define MOK_GEMM_COUNTER(NAME, DIR, EXPORTS, ...) \
  PL_ENDPOINT(NAME, pl::endpoint(#NAME, DIR).exports(EXPORTS).space("gmem").scope("grid").proxy("generic").tile("i32", {1}) \
      .layout(pl::affine{{1},{1},32}).protocol(pl::counter{"pipelink_mok_cnt_" #NAME, ##__VA_ARGS__}));
#define MOK_GEMM_FR {"mok_gemm_fwd", "mok_gemm_rec"}
MOK_GEMM_COUNTER(gate_up_tile_ready, pl::produce, MOK_GEMM_FR);
MOK_GEMM_COUNTER(hidden_row_block_ready, pl::consume, {"mok_gemm_fwd"}, "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu");
MOK_GEMM_COUNTER(x_routed_ready, pl::consume, MOK_GEMM_FR, "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu");
MOK_GEMM_COUNTER(y_routed_ready, pl::produce, {"mok_gemm_fwd"});
MOK_GEMM_COUNTER(y_routed_done, pl::consume, {"mok_gemm_fwd"}, "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu");
MOK_GEMM_COUNTER(d_y_routed_ready, pl::consume, {"mok_gemm_bwd"}, "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu");
MOK_GEMM_COUNTER(d_hidden_ready, pl::produce, {"mok_gemm_bwd"});
MOK_GEMM_COUNTER(d_gate_up_ready, pl::consume, {"mok_gemm_bwd"}, "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu");
MOK_GEMM_COUNTER(d_x_routed_ready, pl::produce, {"mok_gemm_bwd"});
MOK_GEMM_COUNTER(replayed_x_routed_ready, pl::consume, {"mok_gemm_bwd"}, "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu");
MOK_GEMM_COUNTER(replayed_gate_up_ready, pl::produce, {"mok_gemm_bwd"});
MOK_GEMM_COUNTER(replayed_hidden_ready, pl::consume, {"mok_gemm_bwd"}, "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu");
MOK_GEMM_COUNTER(routed_buffers_done, pl::produce, {"mok_gemm_bwd"});
#undef MOK_GEMM_FR
#undef MOK_GEMM_COUNTER
PL_HOST_GROUP("mok.gemm", "mok.gemm.params", "mok_gemm", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "ep");
PL_HOST_ABI_EMPTY(301, MokGemmResidue);
PL_HOST_RESIDUE(2, "mok.gemm.residue", MokGemmResidue, 301, true, pl::host::roles({}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({}))
    .field(pl::host::scalar(1, "mok.num_local_tokens", "i64").caller("num_local_tokens"))
    .field(pl::host::scalar(2, "mok.hidden", "i64").caller("hidden"))
    .field(pl::host::scalar(3, "mok.intermediate", "i64").caller("intermediate"))
    .field(pl::host::scalar(4, "mok.num_local_experts", "i64").caller("num_local_experts"))
    .field(pl::host::scalar(5, "mok.schedule_capacity", "i64").caller("schedule_capacity"))
    .field(pl::host::scalar(6, "mok.macrobatch_size", "i64").caller("macrobatch_size"))
    .field(pl::host::scalar(11, "mok.pass", "i64").binding("pipelink_mok_pass"))
    .field(pl::host::scalar(12, "mok.precision", "i64").binding("pipelink_mok_precision"))
    .field(pl::host::scalar(13, "mok.world", "i64").binding("pipelink_mok_world"))
    .field(pl::host::scalar(20, "mok.comm.DISPATCH_Mb", "i64").fact("mok.comm.DISPATCH_Mb"))
    .field(pl::host::scalar(21, "mok.comm.DISPATCH_Nb", "i64").fact("mok.comm.DISPATCH_Nb"))
    .field(pl::host::scalar(22, "mok.comm.COMBINE_Mb", "i64").fact("mok.comm.COMBINE_Mb"))
    .field(pl::host::scalar(23, "mok.comm.COMBINE_Nb", "i64").fact("mok.comm.COMBINE_Nb"))
    .field(pl::host::scalar(31, "mok.framework.EXPERT_PADDING", "i64").fact("mok.framework.EXPERT_PADDING")));
PL_HOST_FACT(pl::host::fact("mok.gemm.MLP_Mb").i32(mok_gemm_owner::config::MLP_Mb)
    .owner("mok.gemm").stage("mok.gemm").consumers({"mok.framework", "mok.comm", "mok.act"}));
PL_HOST_FACT(pl::host::fact("mok.gemm.MLP_Nb").i32(mok_gemm_owner::config::MLP_Nb)
    .owner("mok.gemm").stage("mok.gemm").consumers({"mok.framework", "mok.comm", "mok.act"}));
#define MOK_GEMM_CHANNEL(ID, TYPE, KEY, DTYPE, RANK, ACCESS) \
  PL_HOST_CHANNEL(ID, TYPE, pl::host::channel("mok.gemm." #KEY, DTYPE).rank(RANK).ACCESS().caller(#KEY));
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("mok.gemm.x_buffer", "bf16").rank(2).read().caller("x_buffer").peer("ep"));
MOK_GEMM_CHANNEL(2, uint8_t, x_routed_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(3, uint8_t, x_routed_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(4, uint16_t, w_shared_gate, "bf16", 2, read);
MOK_GEMM_CHANNEL(5, uint16_t, w_shared_up, "bf16", 2, read);
MOK_GEMM_CHANNEL(6, uint16_t, w_shared_down, "bf16", 2, read);
MOK_GEMM_CHANNEL(7, uint8_t, w_routed_gate_fp8, "u8", 3, read);
MOK_GEMM_CHANNEL(8, uint8_t, w_routed_gate_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(9, uint8_t, w_routed_up_fp8, "u8", 3, read);
MOK_GEMM_CHANNEL(10, uint8_t, w_routed_up_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(11, uint8_t, w_routed_down_fp8, "u8", 3, read);
MOK_GEMM_CHANNEL(12, uint8_t, w_routed_down_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(13, uint16_t, gate_shared, "bf16", 2, write);
MOK_GEMM_CHANNEL(14, uint16_t, up_shared, "bf16", 2, write);
MOK_GEMM_CHANNEL(15, uint16_t, gate_routed, "bf16", 2, write);
MOK_GEMM_CHANNEL(16, uint16_t, up_routed, "bf16", 2, write);
MOK_GEMM_CHANNEL(17, uint8_t, gate_routed_fp8, "u8", 2, write);
MOK_GEMM_CHANNEL(18, uint8_t, gate_routed_sc, "u8", 4, write);
MOK_GEMM_CHANNEL(19, uint8_t, up_routed_fp8, "u8", 2, write);
MOK_GEMM_CHANNEL(20, uint8_t, up_routed_sc, "u8", 4, write);
MOK_GEMM_CHANNEL(21, uint16_t, hidden_shared, "bf16", 2, read);
MOK_GEMM_CHANNEL(22, uint8_t, hidden_routed_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(23, uint8_t, hidden_routed_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(24, uint16_t, y_shared, "bf16", 2, write);
MOK_GEMM_CHANNEL(25, uint16_t, y_routed, "bf16", 2, write);
MOK_GEMM_CHANNEL(26, int32_t, tokens_per_expert, "i32", 1, read);
PL_HOST_CHANNEL(27, uint16_t, pl::host::channel("mok.gemm.d_y_buffer", "bf16").rank(2).read().caller("d_y_buffer").peer("ep"));
MOK_GEMM_CHANNEL(28, uint16_t, d_hidden_shared, "bf16", 2, write);
MOK_GEMM_CHANNEL(29, uint16_t, d_hidden_routed, "bf16", 2, write);
MOK_GEMM_CHANNEL(30, uint16_t, d_gate_shared, "bf16", 2, read);
MOK_GEMM_CHANNEL(31, uint16_t, d_up_shared, "bf16", 2, read);
MOK_GEMM_CHANNEL(32, uint16_t, d_x_shared, "bf16", 2, write);
MOK_GEMM_CHANNEL(33, uint16_t, d_x_routed, "bf16", 2, write);
MOK_GEMM_CHANNEL(34, uint16_t, d_w_shared_gate, "bf16", 2, write);
MOK_GEMM_CHANNEL(35, uint16_t, d_w_shared_up, "bf16", 2, write);
MOK_GEMM_CHANNEL(36, uint16_t, d_w_shared_down, "bf16", 2, write);
MOK_GEMM_CHANNEL(37, uint16_t, d_w_routed_gate, "bf16", 3, read_write);
MOK_GEMM_CHANNEL(38, uint16_t, d_w_routed_up, "bf16", 3, read_write);
MOK_GEMM_CHANNEL(39, uint16_t, d_w_routed_down, "bf16", 3, read_write);
MOK_GEMM_CHANNEL(40, uint8_t, x_routed_t_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(41, uint8_t, x_routed_t_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(42, uint8_t, hidden_routed_t_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(43, uint8_t, hidden_routed_t_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(44, uint8_t, w_routed_gate_t_fp8, "u8", 3, read);
MOK_GEMM_CHANNEL(45, uint8_t, w_routed_gate_t_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(46, uint8_t, w_routed_up_t_fp8, "u8", 3, read);
MOK_GEMM_CHANNEL(47, uint8_t, w_routed_up_t_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(48, uint8_t, w_routed_down_t_fp8, "u8", 3, read);
MOK_GEMM_CHANNEL(49, uint8_t, w_routed_down_t_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(50, uint8_t, d_y_routed_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(51, uint8_t, d_y_routed_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(52, uint8_t, d_y_routed_t_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(53, uint8_t, d_y_routed_t_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(54, uint8_t, d_gate_routed_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(55, uint8_t, d_gate_routed_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(56, uint8_t, d_gate_routed_t_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(57, uint8_t, d_gate_routed_t_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(58, uint8_t, d_up_routed_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(59, uint8_t, d_up_routed_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(60, uint8_t, d_up_routed_t_fp8, "u8", 2, read);
MOK_GEMM_CHANNEL(61, uint8_t, d_up_routed_t_sc, "u8", 4, read);
MOK_GEMM_CHANNEL(62, uint16_t, x_routed_bf16, "bf16", 2, read);
MOK_GEMM_CHANNEL(63, uint16_t, hidden_routed_bf16, "bf16", 2, read);
MOK_GEMM_CHANNEL(64, uint16_t, w_routed_gate_bf16, "bf16", 3, read);
MOK_GEMM_CHANNEL(65, uint16_t, w_routed_up_bf16, "bf16", 3, read);
MOK_GEMM_CHANNEL(66, uint16_t, w_routed_down_bf16, "bf16", 3, read);
MOK_GEMM_CHANNEL(67, uint16_t, d_y_routed_bf16, "bf16", 2, read);
MOK_GEMM_CHANNEL(68, uint16_t, d_gate_routed_bf16, "bf16", 2, read);
MOK_GEMM_CHANNEL(69, uint16_t, d_up_routed_bf16, "bf16", 2, read);
#undef MOK_GEMM_CHANNEL
PL_HOST_SCHEDULER(pl::host::scheduler("mok.framework"));
PL_HOST_TARGET();
#endif
