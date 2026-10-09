/*
 * Portions derived from FlashAttention-3 hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp.
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

// FA3 producer sequence retained from role_load.cuh; only work/KV selection changes.
#pragma once
#include "role_load.cuh"
#include "block_load_params.cuh"

namespace pipelink::attention::sm90 {

template <class Config>
CUTLASS_DEVICE void publish_block_kv(
    SplitContext<Config>& c, PhiBlockPair const& pair, uint32_t kv_count) {
  static_assert(Config::kBlockM == 128 && Config::kBlockN == 128);
  if (threadIdx.x % 32 != 0) return;
  typename Config::KvIndexRing::State const state{
      int(kv_count & 3u), ((kv_count >> 2) & 1u) ^ 1u};
  auto& meta = c.kv_index.producer_acquire(state);
  if (pair.flags != KvFlags::fully_valid) {
    CUTLASS_PRAGMA_UNROLL
    for (int group = 0; group < 4; ++group) {
      meta.base[group] = pair.lo * 64 + group * 16;
      meta.base[group + 4] = pair.hi * 64 + group * 16;
    }
    meta.valid[0] = meta.valid[1] = 0xffffffffu;
    meta.valid[2] = meta.valid[3] = pair.hi >= 0 ? 0xffffffffu : 0u;
  }
  meta.flags = pair.flags;
  c.kv_index.producer_commit(state);
}

template <class Config>
CUTLASS_DEVICE void run_block_load(
    BlockLoadParams const& block, SplitContext<Config>& c) {
  auto const& params = block.base;
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
    prefetch_tma_descriptor(&block.tma_half_K);
    prefetch_tma_descriptor(&block.tma_half_V);
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
  auto* const sK = c.shared.tensors.smem_k.data();
  auto* const sV = c.shared.tensors.smem_v.data();
  auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
  Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));

  // Load owns strict LPT scheduling: the host orders CSR rows by list length.
  // Dynamic ticket allocation and every producer/ring phase remain FA3's.
  cutlass::arch::wait_on_dependent_grids();
  int ticket = int(blockIdx.x);
  while (ticket < block.num_work) {
    int const row = block.work_order[ticket];
    int head_batch, q_block, batch, q_head;
    block.q_tiles_div(head_batch, q_block, row);
    block.heads_q_div(batch, q_head, head_batch);
    int const q_offset = q_block * Config::kBlockM;
    int const row_begin = block.row_ptr[row];
    int const row_end = block.row_ptr[row + 1];
    int const steps = row_end - row_begin;
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

    int next_tile = ticket + int(gridDim.x);
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
      Tensor mQ = params.tma_load_Q.get_tma_tensor(shape_Q)(_, _, q_head, batch);
      Tensor gQ = local_tile(mQ, select<0, 2>(TileShape{}), make_coord(q_block, _0{}));
      Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
      if (elected) { prefetch(params.tma_load_Q, tQgQ); }

      // SW128 stores the two 64-dimension planes 16 KiB apart. Each logical
      // 64-row half therefore uses two 8 KiB boxes, not one compact 16 KiB
      // box. Four boxes complete the unchanged 32 KiB K/V transaction.
      // Box offsets and the shared base are 1 KiB aligned: SW128 xor is zero.
      auto load_k = [&](PhiBlockPair const& pair, auto const& state) {
        c.k.producer_acquire(state);
        auto* barrier = c.k.producer_get_barrier(state);
        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < 2; ++d) {
          SM90_TMA_LOAD::copy(&block.tma_half_K, barrier,
              uint64_t(TMA::CacheHintSm90::EVICT_LAST), sK + state.index() * 16384 + d * 8192,
              d * 64, pair.lo * 64, kv_head, batch);
          SM90_TMA_LOAD::copy(&block.tma_half_K, barrier,
              uint64_t(TMA::CacheHintSm90::EVICT_LAST), sK + state.index() * 16384 + d * 8192 + 4096,
              d * 64, pair.hi * 64, kv_head, batch);
        }
      };
      auto load_v = [&](PhiBlockPair const& pair, auto const& state) {
        c.v.producer_acquire(state);
        auto* barrier = c.v.producer_get_barrier(state);
        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < 2; ++d) {
          SM90_TMA_LOAD::copy(&block.tma_half_V, barrier,
              uint64_t(TMA::CacheHintSm90::EVICT_LAST), sV + state.index() * 16384 + d * 8192,
              d * 64, pair.lo * 64, kv_head, batch);
          SM90_TMA_LOAD::copy(&block.tma_half_V, barrier,
              uint64_t(TMA::CacheHintSm90::EVICT_LAST), sV + state.index() * 16384 + d * 8192 + 4096,
              d * 64, pair.hi * 64, kv_head, batch);
        }
      };

      PhiBlockPair pair = block.block_pairs[row_end - 1];
      publish_block_kv(c, pair, kv_count);
      load_k(pair, kv_state(kv_count));
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

      PhiBlockPair previous = pair;
      #pragma unroll 2
      for (int entry = row_end - 2; entry >= row_begin; --entry) {
        pair = block.block_pairs[entry];
        ++kv_count;
        publish_block_kv(c, pair, kv_count);
        load_k(pair, kv_state(kv_count));
        load_v(previous, kv_state(kv_count - 1));  // V trails K by one step.
        previous = pair;
      }
      prefetch_next();
      load_v(previous, kv_state(kv_count));
      ++kv_count;
    } else {
      prefetch_next();
    }
    // Normalize the scalar scheduler result into the uniform register path
    // without waiting for the 31 retired lanes.
    ticket = __shfl_sync(1u, next_tile, 0);
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

} // namespace pipelink::attention::sm90
