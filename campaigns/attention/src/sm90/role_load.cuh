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

// Reduced from FlashAttention-3 hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
// and tile_scheduler.hpp. Upstream copyright (c) 2024 Jay Shah, Ganesh
// Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
#pragma once

#include "split_context.cuh"

namespace pipelink::attention::sm90 {

// Lane zero publishes both work descriptors; their full barriers count one.
template <class Config>
CUTLASS_DEVICE void publish_load_work(
    SplitContext<Config>& c, WorkDesc const& work,
    uint32_t& work_count) {
  static_assert(Config::WorkRing::depth == 2);
  typename Config::WorkRing::State const state{
      int(work_count & 1u), ((work_count >> 1) & 1u) ^ 1u};
  auto& core = c.work_core.producer_acquire(state);
  auto& epilogue = c.work_epilogue.producer_acquire(state);
  if (threadIdx.x % 32 == 0) {
    core = work;
    epilogue = work;
    c.work_core.producer_commit(state);
    c.work_epilogue.producer_commit(state);
  }
  ++work_count;
}

template <class Config>
CUTLASS_DEVICE void publish_load_kv(
    SplitContext<Config>& c, int seqlen_k, int q_offset, int window,
    int n_block, bool causal, uint32_t kv_count) {
  static_assert(Config::KvIndexRing::depth == 4);
  typename Config::KvIndexRing::State const state{
      int(kv_count & 3u), ((kv_count >> 2) & 1u) ^ 1u};
  // Only this lane writes the record and supplies its single full arrival.
  // Dense steps are consumed from flags alone; masked steps retain every
  // base/valid field of the public coordinate encoding.
  if (threadIdx.x % 32 != 0) return;
  constexpr int N = Config::kBlockN;
  int const first = n_block * N;
  uint32_t flags = first + N <= seqlen_k ? KvFlags::fully_valid : 0;
  if (causal && min(first + N, seqlen_k) - 1 > q_offset) {
    flags |= KvFlags::needs_causal;
  }
  // The last Q row requires k > q_offset + kBlockM - 1 - window.
  if (window > 0 && first < q_offset + Config::kBlockM - window) {
    flags |= KvFlags::needs_mask_mod;
  }
  auto& meta = c.kv_index.producer_acquire(state);
  if (flags != KvFlags::fully_valid) {
    CUTLASS_PRAGMA_UNROLL
    for (int group = 0; group < N / 16; ++group) {
      meta.base[group] = first + group * 16;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int group = 0; group < (N + 31) / 32; ++group) {
      int const valid = max(0, min(min(N - group * 32, 32),
                                 seqlen_k - first - group * 32));
      meta.valid[group] = valid == 32 ? 0xffffffffu
          : valid == 0 ? 0u : (uint32_t(1) << valid) - 1u;
    }
    // The construction above proves arithmetic bases and a prefix bitmap.
    // Publish their compact description only after building the full record.
    flags |= contiguous |
        (uint32_t(max(0, min(N, seqlen_k - first))) << contiguous_valid_shift);
  }
  meta.flags = flags;
  c.kv_index.producer_commit(state);
}

template <class Config>
CUTLASS_DEVICE void run_load(
    LoadParams<Config::causal> const& params, SplitContext<Config>& c) {
  using namespace cute;
  using Mainloop = typename Config::Mainloop;
  using Scheduler = typename Config::Scheduler;
  using PipelineQ = typename Config::PipelineQ;
  using PipelineKV = typename Config::PipelineKV;
  using TileShape = typename Config::TileShape_MNK;
  static_assert(size(typename Config::ClusterShape{}) == 1);
  static_assert(Config::kBlockM == 128 && Mainloop::kHeadDim == 128);
  static_assert(Mainloop::Use_TMA_Q && Mainloop::Use_TMA_KV);

  c.initialize();
  bool const elected = threadIdx.x % 32 == 0;
  // Only the writer tracks producer phases. Passive lanes must not acquire
  // work/metadata rings: a fast writer can publish, have the slot consumed,
  // and advance its phase before a passive lane finishes the old acquire.
  if (!elected) return;
  if (elected) {
    prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
  }

  auto q_write = cutlass::make_producer_start_state<PipelineQ>();
  static_assert(PipelineKV::Stages == 2);
  uint32_t kv_count = 0;
  auto kv_state = [](uint32_t count) {
    return typename PipelineKV::PipelineState{
        int(count & 1u), ((count >> 1) & 1u) ^ 1u, count};
  };
  uint32_t work_write = 0;

  Tensor sQ = make_tensor(make_smem_ptr(c.shared.tensors.smem_q.data()),
                          typename Mainloop::SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(c.shared.tensors.smem_k.data()),
                          typename Mainloop::SmemLayoutK{});
  Tensor sVt = make_tensor(make_smem_ptr(c.shared.tensors.smem_v.data()),
                           typename Mainloop::SmemLayoutVt{});
  auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
  auto block_tma_K = params.tma_load_K.get_slice(_0{});
  auto block_tma_V = params.tma_load_V.get_slice(_0{});
  Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
  Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK));
  Tensor tVsVt = group_modes<0, 3>(block_tma_V.partition_D(sVt));

  // Keep FA3's host scheduler Params and WorkTileInfo coordinate decoding,
  // including causal LPT/L2 swizzling. Only load schedules: the upstream
  // consumer SMEM broadcast and its two named barriers become work rings.
  cutlass::arch::wait_on_dependent_grids();
  typename Scheduler::WorkTileInfo tile{int(blockIdx.x)};
  while (tile.is_valid(params.scheduler)) {
    auto const coord = tile.get_block_coord(params.scheduler);
    int const q_block = get<0>(coord);
    int const q_head = get<1>(coord);
    int const batch = get<2>(coord);
    int const q_offset = q_block * Config::kBlockM + params.seqlen_k - params.seqlen_q;
    bool const causal = params.causal || params.window_size > 0;
    int const max_k = causal ? max(0, min(params.seqlen_k, q_offset + Config::kBlockM))
                             : params.seqlen_k;
    int const n_max = cute::ceil_div(max_k, Int<Config::kBlockN>{});
    int const n_min = params.window_size > 0
        ? max(0, q_offset - params.window_size + 1) / Config::kBlockN : 0;
    int const steps = max(0, n_max - n_min);
    // Unpacked FA3 schedules a Q head, not a packed-GQA KV head.
    // kv_head carries that logical Q head; physical KV addressing divides it
    // below. qhead_per_kvhead=1 preserves the WorkDesc row convention.
    publish_load_work(c, WorkDesc{
        q_block, batch, q_head, steps, params.seqlen_q, params.seqlen_k,
        q_offset, 0,
        WorkFlags::valid | WorkFlags::apply_sink |
            (steps == 0 ? WorkFlags::empty_output : 0u) |
            (params.needs_score_mod ? WorkFlags::needs_score_mod : 0u),
        1u, params.window_size, params.score_param, params.score_param_inv}, work_write);

    int next_tile = tile.tile_idx + int(gridDim.x);
    auto prefetch_next = [&] {
      if constexpr (Config::causal) {
        if (elected) {
          next_tile = atomicAdd(params.scheduler.tile_count_semaphore, 1) + int(gridDim.x);
        }
      }
    };

    if (steps != 0) {
      int const kv_head = params.qhead_per_khead.divide(q_head);
      auto const shape_Q = make_shape(params.seqlen_q, Int<128>{},
                                     get<2>(params.shape_Q), get<3>(params.shape_Q));
      auto const shape_K = make_shape(params.seqlen_k, Int<128>{},
                                     get<2>(params.shape_K), get<3>(params.shape_K));
      Tensor mQ = params.tma_load_Q.get_tma_tensor(shape_Q)(_, _, q_head, batch);
      Tensor mK = params.tma_load_K.get_tma_tensor(shape_K)(_, _, kv_head, batch);
      auto const shape_V = make_shape(Int<128>{}, get<0>(params.shape_K),
                                     get<2>(params.shape_K), get<3>(params.shape_K));
      Tensor mVt = params.tma_load_V.get_tma_tensor(shape_V)(_, _, kv_head, batch);
      Tensor gQ = local_tile(mQ, select<0, 2>(TileShape{}), make_coord(q_block, _0{}));
      Tensor gK = local_tile(mK, select<1, 2>(TileShape{}), make_coord(_, _0{}));
      Tensor gVt = local_tile(mVt, Shape<Int<128>, Int<Config::kBlockN>>{},
                              make_coord(_0{}, _));
      Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
      Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK));
      Tensor tVgVt = group_modes<0, 3>(block_tma_V.partition_S(gVt));
      if (elected) { prefetch(params.tma_load_Q, tQgQ); }

      auto load_k = [&](int n_block, auto const& state) {
        if (elected) {
          c.k.producer_acquire(state);
          copy(params.tma_load_K.with(*c.k.producer_get_barrier(state), 0,
                                     TMA::CacheHintSm90::EVICT_LAST),
               tKgK(_, n_block), tKsK(_, state.index()));
        }
      };
      auto load_v = [&](int n_block, auto const& state) {
        if (elected) {
          c.v.producer_acquire(state);
          copy(params.tma_load_V.with(*c.v.producer_get_barrier(state), 0,
                                     TMA::CacheHintSm90::EVICT_LAST),
               tVgVt(_, n_block), tVsVt(_, state.index()));
        }
      };

      int n_block = n_max - 1;
      publish_load_kv(c, params.seqlen_k, q_offset, params.window_size,
                      n_block, causal, kv_count);
      load_k(n_block, kv_state(kv_count));
      // Q depth one replaces FA3 QueryEmpty: two consumer-WG releases.
      if (elected) {
        c.q.producer_acquire(q_write);
        copy(params.tma_load_Q.with(*c.q.producer_get_barrier(q_write), 0,
                                   TMA::CacheHintSm90::EVICT_FIRST), tQgQ, tQsQ);
      }
      // Q and O advance once per nonempty work item, so their phases agree.
      // Initial wait(1) is free; only epilogue arrivals unlock later V loads.
      c.o_drained.wait(q_write.phase());
      ++q_write;

      int n_previous = n_block;
      #pragma unroll 2
      for (--n_block; n_block >= n_min; --n_block) {
        ++kv_count;
        publish_load_kv(c, params.seqlen_k, q_offset, params.window_size,
                        n_block, causal, kv_count);
        load_k(n_block, kv_state(kv_count));
        load_v(n_previous, kv_state(kv_count - 1));  // V trails K by one step.
        n_previous = n_block;
      }
      prefetch_next();
      load_v(n_previous, kv_state(kv_count));
      ++kv_count;
    } else {
      prefetch_next();
    }
    // Normalize the scalar scheduler result into the uniform register path
    // without waiting for the 31 retired lanes.
    tile.tile_idx = __shfl_sync(1u, next_tile, 0);
  }

  publish_load_work(c, WorkDesc{}, work_write);
  // The terminal descriptor is visible before draining: every outstanding
  // consumer can finish, including an epilogue with a pending TMA store.
  c.o_drained.wait(q_write.phase());
  if (elected) {
    c.q.producer_tail(q_write);
    c.k.producer_tail(kv_state(kv_count));
    c.v.producer_tail(kv_state(kv_count));
  }
}

}  // namespace pipelink::attention::sm90
