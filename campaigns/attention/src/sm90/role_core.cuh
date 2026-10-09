/*
 * Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Reduced from FA3 hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp and epilogue_fwd.hpp.
// Copyright (c) 2024 Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
// Pradeep Ramani, Tri Dao. FA3's RS-PV/intra-WG overlap is retained.
#pragma once
#include "split_context.cuh"
#include "softmax.h"
#include "utils.h"

extern "C" __device__ bool pipelink_call_mask_mod(
    int32_t q_pos, int32_t kv_pos, pipelink_attention_sm90_work_desc const& work);
#if !defined(PIPELINK_SM90_NO_SCORE_SEAM)
extern "C" __device__ float pipelink_call_score_mod(
    float s, pipelink_attention_sm90_work_desc const& work);
#endif

namespace pipelink::attention::sm90 {
using namespace cute;

#if !defined(PIPELINK_SM90_NO_SCORE_SEAM)
template <class TensorS>
CUTLASS_DEVICE void transform_scores(TensorS& scores, WorkDesc const& work) {
  if (!(work.flags & WorkFlags::needs_score_mod)) return;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(scores); ++i) {
    scores(i) = pipelink_call_score_mod(scores(i), work);
  }
}
#endif

// The coordinate partition is the same one as flash::Mask. Metadata changes the
// source of the column coordinate, not its association with the MMA fragment.
template <class Config, bool FirstStep = false, class TensorS, class Meta>
CUTLASS_DEVICE void mask_scores(TensorS& scores, Meta const& meta,
                                WorkDesc const& work, int thread_idx) {
  uint32_t flags = 0;
  if ((thread_idx & 31) == 0) flags = meta.flags;
  flags = __shfl_sync(0xffffffff, flags, 0);
  if (flags == KvFlags::fully_valid) return;
  using Mma = typename Config::Mainloop::TiledMmaQK;
  auto identity = make_identity_tensor(make_shape(Int<128>{}, Int<Config::kBlockN>{}));
  auto coords = Mma{}.get_thread_slice(thread_idx).partition_C(identity);
  auto rc = make_tensor(coords.data(), flash::convert_layout_acc_rowcol(coords.layout()));
  auto sr = make_tensor(scores.data(), flash::convert_layout_acc_rowcol(scores.layout()));
  if (FirstStep &&
      (flags & (contiguous | KvFlags::needs_mask_mod)) == contiguous) {
    int const base = meta.base[0];
    int const valid_count = flags >> contiguous_valid_shift;
    auto coords0 = Mma{}.get_thread_slice(_0{}).partition_C(identity);
    auto rc0 = make_tensor(coords0.data(), flash::convert_layout_acc_rowcol(coords0.layout()));
    int const column_offset = get<1>(rc(_0{}, _0{}));
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(sr); ++m) {
      int limit = valid_count - column_offset;
      if (flags & KvFlags::needs_causal) {
        int const q_pos = work.q_causal_offset + get<0>(rc(m, _0{}));
        limit = min(limit, q_pos - base + 1 - column_offset);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < size<1>(sr); ++n) {
        if (get<1>(rc0(_0{}, n)) >= limit) sr(m, n) = -INFINITY;
      }
    }
    __syncwarp();
    return;
  }
  CUTLASS_PRAGMA_UNROLL
  for (int n = 0; n < size<1>(sr); ++n) {
    int const col = get<1>(rc(_0{}, n));
    int const kv_pos = kv_coord(meta, col);
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(sr); ++m) {
      int const q_pos = work.q_causal_offset + get<0>(rc(m, _0{}));
      // The callee is pure. Gate it on the step-uniform policy flag, not on
      // the per-element tail/causal predicate: short-circuit control flow
      // makes the unrolled N=176 window path retain excessive predicates.
      bool mod_keep = true;
      if (flags & KvFlags::needs_mask_mod) {
        mod_keep = pipelink_call_mask_mod(q_pos, kv_pos, work);
      }
      bool const keep = kv_pos >= 0 &&
          (!(flags & KvFlags::needs_causal) || kv_pos <= q_pos) && mod_keep;
      if (!keep) sr(m, n) = -INFINITY;
    }
  }
  // All generic metadata readers finish before the warp's elected release.
  __syncwarp();
}

// An unchanged online maximum has an exactly unit scale, so its O row needs
// no scalar multiplies while the next QK batch is in flight.
template <class TensorO, class TensorScale>
CUTLASS_DEVICE void rescale_changed_rows(TensorO& output, TensorScale const& scales) {
  auto rows = make_tensor(output.data(), flash::convert_layout_acc_rowcol(output.layout()));
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < size<0>(rows); ++m) {
    if (scales(m) != 1.f) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < size<1>(rows); ++n) rows(m, n) *= scales(m);
    }
  }
}

template <class Config>
CUTLASS_DEVICE void run_core(CoreParams const& params, SplitContext<Config>& context) {
  using Mainloop = typename Config::Mainloop;
  using Epilogue = typename Config::Epilogue;
  using Element = typename Config::Element;
  using TiledMmaQK = typename Mainloop::TiledMmaQK;
  using TiledMmaPV = typename Mainloop::TiledMmaPV;
  using TileShape = typename Mainloop::TileShape_MNK;
  using TileShapePV = typename Mainloop::TileShape_MNK_PV;
  using WorkRing = typename Config::WorkRing;
  using KvIndexRing = typename Config::KvIndexRing;
  using FinalStatsRing = typename Config::FinalStatsRing;
  using OutputRing = typename Config::OutputRing;
  context.initialize();

  int const thread_idx = int(threadIdx.x) - 32 * pipelink_attention_warp_core();
  int const wg = __shfl_sync(0xffffffff, thread_idx / 128, 0);
  auto scheduler_sync = [&] {
    cutlass::arch::NamedBarrier::sync(256, wg == 0
        ? pipelink_attention_warp_scheduler_wg1() : pipelink_attention_warp_scheduler_wg2());
  };
  auto scheduler_arrive = [&] {
    cutlass::arch::NamedBarrier::arrive(256, wg == 0
        ? pipelink_attention_warp_scheduler_wg2() : pipelink_attention_warp_scheduler_wg1());
  };
  // FA3 seeds WG1 with its own 128 arrivals; its first sync supplies the rest.
  if (wg == 0) cutlass::arch::NamedBarrier::arrive(256, pipelink_attention_warp_scheduler_wg1());

  auto work_state = WorkRing::consumer_start();
  static_assert(FinalStatsRing::depth == 1 && OutputRing::depth == 1);
  cutlass::PipelineState<1> q_state;
  static_assert(Mainloop::kStages == 2 && KvIndexRing::depth == 4);
  // Store each ring's index and phase together; consumers do not use count().
  uint32_t kv_position = 0;
  auto kv_state = [](uint32_t position) {
    return typename Mainloop::PipelineState{int(position & 1u), position >> 1, 0};
  };
  uint32_t meta_position = 0;
  auto meta_state = [](uint32_t position) {
    return typename KvIndexRing::State{int(position & 3u), position >> 2};
  };
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto wg_mma_qk = tiled_mma_qk.get_slice(wg * 128);
  auto wg_mma_pv = tiled_mma_pv.get_slice(wg * 128);
  auto sQ = make_tensor(make_smem_ptr(context.shared.tensors.smem_q.data()), typename Mainloop::SmemLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(context.shared.tensors.smem_k.data()), typename Mainloop::SmemLayoutK{});
  auto sV = make_tensor(make_smem_ptr(context.shared.tensors.smem_v.data()), typename Mainloop::SmemLayoutVtMma{});
  auto tSrQ = wg_mma_qk.partition_fragment_A(sQ);
  auto tSrK = wg_mma_qk.partition_fragment_B(sK);
  auto tOrV = wg_mma_pv.partition_fragment_B(sV);
  auto wait = [](auto& pipeline, auto const& state) {
    auto token = pipeline.consumer_try_wait(state);
    pipeline.consumer_wait(state, token);
  };

  CUTLASS_PRAGMA_NO_UNROLL
  for (;;) {
    WorkDesc work = context.work_core.consumer_wait(work_state);
    __syncwarp();
    if ((thread_idx & 31) == 0) context.work_core.consumer_release(work_state);
    work_state.advance();
    if (!(work.flags & WorkFlags::valid)) break;
    if (work.flags & WorkFlags::empty_output) continue;

    flash::Softmax<2> softmax(params.scale_softmax_log2);
    auto tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShapePV{}));
    wait(context.q, q_state);
    auto tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape{}));
    wait(context.k, kv_state(kv_position));
    flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, int(kv_position & 1u)), tSrS);
    warpgroup_wait<0>();
    context.k.consumer_release(kv_state(kv_position));
    auto const& first_meta = context.kv_index.consumer_wait(meta_state(meta_position));
#if !defined(PIPELINK_SM90_NO_SCORE_SEAM)
    transform_scores(tSrS, work);
#endif
    mask_scores<Config, true>(tSrS, first_meta, work, thread_idx);
    if ((thread_idx & 31) == 0) context.kv_index.consumer_release(meta_state(meta_position));
    meta_position = (meta_position + 1) & 7u;
    auto scores_scale = softmax.template max_get_scale<true, true>(tSrS);
    softmax.template online_softmax<true, true>(tSrS);
    auto tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
    auto tOrP = make_tensor_like<Element>(tOrP_acc);
    flash::convert_type_out(tOrP_acc, tOrP);
    clear(tOrO);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int step = 1; step < work.kv_steps; ++step) {
      uint32_t const kv_previous = kv_position;
      kv_position = (kv_position + 1) & 3u;
      auto next_scores = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape{}));
      if (wg == 0) wait(context.k, kv_state(kv_position));
      scheduler_sync();
      flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, int(kv_position & 1u)), next_scores);
      // Consume the previous softmax scale while QK is in flight. Keeping O
      // independent of the softmax lets PV completion overlap its exponentials.
      rescale_changed_rows(tOrO, scores_scale);
      if (wg == 0) wait(context.v, kv_state(kv_previous));
      flash::gemm<false, -1>(tiled_mma_pv, tOrP, tOrV(_, _, _, int(kv_previous & 1u)), tOrO);
      scheduler_arrive();
      warpgroup_wait<1>();
      context.k.consumer_release(kv_state(kv_position));
      auto const& meta = context.kv_index.consumer_wait(meta_state(meta_position));
#if !defined(PIPELINK_SM90_NO_SCORE_SEAM)
      transform_scores(next_scores, work);
#endif
      mask_scores<Config>(next_scores, meta, work, thread_idx);
      if ((thread_idx & 31) == 0) context.kv_index.consumer_release(meta_state(meta_position));
      meta_position = (meta_position + 1) & 7u;
      // O rescaling is deferred to the next QK issue window.
      cute::copy(softmax.template max_get_scale<false, true>(next_scores), scores_scale);
      softmax.template online_softmax<false, true>(next_scores);
      warpgroup_wait<0>();
      context.v.consumer_release(kv_state(kv_previous));
      flash::convert_type_out(make_tensor(next_scores.data(), tOrP.layout()), tOrP);
    }
    // Each WG has retired its last QK before releasing the single Q buffer.
    context.q.consumer_release(q_state);
    ++q_state;
    wait(context.v, kv_state(kv_position));
    // The final softmax has no following QK issue window to apply its scale.
    rescale_changed_rows(tOrO, scores_scale);
    flash::gemm<false, -1>(tiled_mma_pv, tOrP, tOrV(_, _, _, int(kv_position & 1u)), tOrO);

    // FA3 finalize reduces l then overwrites it with LSE. Keep raw (m,l) for
    // the declared final_stats channel; the epilogue owns log and global LSE.
    flash::SumOp<float> sum_op;
    flash::quad_allreduce_(softmax.row_sum, softmax.row_sum, sum_op);
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size(softmax.row_sum); ++m) {
      float const sum = softmax.row_sum(m);
      scores_scale(m) = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
    }
    warpgroup_wait<0>();
    context.v.consumer_release(kv_state(kv_position));
    kv_position = (kv_position + 1) & 3u;
    softmax.rescale_o(tOrO, scores_scale);
    auto output = make_tensor_like<Element>(tOrO);
    flash::convert_type_out(tOrO, output);

    // Q has just advanced. Its consumer phase equals both depth-one output
    // producer phases: each advances exactly once per nonempty work item.
    typename OutputRing::State const output_state{0, q_state.phase()};
    typename FinalStatsRing::State const stats_state{0, q_state.phase()};
    // Both WGs must retire V before either overwrites its aliased O storage.
    context.corrected_o.producer_acquire(output_state);
    cutlass::arch::NamedBarrier::sync(256, pipelink_attention_epilogue_barrier());
    auto sO = make_tensor(make_smem_ptr(context.shared.tensors.smem_v.data()), typename Epilogue::SmemLayoutO{});
    auto copy_o = make_tiled_copy_C(typename Epilogue::SmemCopyAtomO{}, tiled_mma_pv);
    auto thread_copy_o = copy_o.get_thread_slice(thread_idx);
    auto rO = thread_copy_o.retile_S(output);
    auto dO = thread_copy_o.partition_D(sO);
    cute::copy(copy_o, rO, dO);
    cutlass::arch::fence_view_async_shared();
    context.corrected_o.producer_commit(output_state);

    auto& stats = context.final_stats.producer_acquire(stats_state);
    auto oc = tiled_mma_pv.get_thread_slice(thread_idx).partition_C(make_identity_tensor(select<0, 1>(TileShapePV{})));
    auto oc_rc = make_tensor(oc.data(), flash::convert_layout_acc_rowcol(oc.layout()));
    if (get<1>(oc_rc(_0{}, _0{})) == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size(softmax.row_sum); ++m) {
        int const row = get<0>(oc_rc(m, _0{}));
        stats.m[row] = softmax.row_max(m);
        stats.l[row] = softmax.row_sum(m);
      }
    }
    context.final_stats.producer_commit(stats_state);
  }
}
} // namespace pipelink::attention::sm90
