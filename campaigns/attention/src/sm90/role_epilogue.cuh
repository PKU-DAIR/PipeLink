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

// Reduced from FlashAttention-3 hopper/epilogue_fwd.hpp. Upstream copyright
// (c) 2024 Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
// Pradeep Ramani, Tri Dao.
#pragma once

#include "split_context.cuh"

namespace pipelink::attention::sm90 {

template <class Config>
CUTLASS_DEVICE void store_empty_output(
    EpilogueParams<Config::causal> const& params, WorkDesc const& work) {
  using namespace cute;
  static_assert(Config::kBlockM == 128 && sizeof(typename Config::Element) == 2);
  int const lane = threadIdx.x % 32;
  int const first_row = work.q_block * Config::kBlockM;
  int const last_row = min(first_row + Config::kBlockM, work.seqlen_q);
  Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O,
                          params.stride_O)(_, _, work.kv_head, work.batch, _0{});
  // FA3's 128-bit zero stores, repartitioned over one warp rather than the
  // two MMA WGs. TMA-compatible dense rows and head offsets are 16B aligned.
  int const column = (lane % 16) * 8;
  for (int row = first_row + lane / 16; row < last_row; row += 2) {
    *reinterpret_cast<uint4*>(&mO(row, column)) = make_uint4(0, 0, 0, 0);
  }
  auto const lse_offset = work.kv_head * get<1>(params.stride_LSE)
      + work.batch * get<2>(params.stride_LSE);
  #pragma unroll
  for (int row_in_tile = lane; row_in_tile < Config::kBlockM; row_in_tile += 32) {
    int const row = first_row + row_in_tile;
    if (row < work.seqlen_q) {
      params.ptr_LSE[lse_offset + row] = -INFINITY;
    }
  }
}

template <class Config>
CUTLASS_DEVICE void run_epilogue(
    EpilogueParams<Config::causal> const& params, SplitContext<Config>& c) {
  using namespace cute;
  using Epilogue = typename Config::Epilogue;
  static_assert(size(typename Config::ClusterShape{}) == 1);
  static_assert(Epilogue::Use_TMA_O);
  c.initialize();
  bool const elected = cute::elect_one_sync();
  int const lane = threadIdx.x % 32;
  if (elected) { prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor()); }

  auto work_read = Config::WorkRing::consumer_start();
  uint32_t output_phase = 0;
  while (true) {
    // Keep the descriptor slot owned until its tile is drained. Read fields
    // at their use sites instead of retaining a copied record across TMA/LSE.
    WorkDesc const& work = c.work_epilogue.consumer_wait(work_read);
    if (!(work.flags & WorkFlags::valid)) {
      c.work_epilogue.consumer_release(work_read);
      break;
    }
    if (work.flags & WorkFlags::empty_output) {
      store_empty_output<Config>(params, work);
      // No corrected_o/final_stats payload exists for an empty tile, and no
      // V/O alias use occurred: do not change their states or o_drained.
      c.work_epilogue.consumer_release(work_read);
      work_read.advance();
      continue;
    }

    // Both payloads occur exactly once per nonempty item.
    typename Config::OutputRing::State const output_read{0, output_phase};
    typename Config::FinalStatsRing::State const stats_read{0, output_phase};
    c.corrected_o.consumer_wait(output_read);
    if (elected) {
      // Core has already fenced its generic SMEM writes before publishing.
      // The channel's first bytes are FA3's V-stage-0 / O-storage alias.
      Tensor sO = make_tensor(make_smem_ptr(c.shared.tensors.smem_v.data()),
                              typename Epilogue::SmemLayoutO{});
      auto const shape_O = make_shape(get<0>(params.shape_O), Int<128>{},
          get<2>(params.shape_O), get<3>(params.shape_O), _1{});
      Tensor mO = params.tma_store_O.get_tma_tensor(shape_O)(
          _, _, work.kv_head, work.batch, _0{});
      Tensor gO = local_tile(mO, Shape<Int<Config::kBlockM>, Int<128>>{},
                             make_coord(work.q_block, _0{}));
      auto block_tma_O = params.tma_store_O.get_slice(_0{});
      Tensor tOgO = block_tma_O.partition_D(gO);
      Tensor tOsO = block_tma_O.partition_S(sO);
      copy(params.tma_store_O, tOsO, tOgO);
      tma_store_arrive();
    }

    auto const& stats = c.final_stats.consumer_wait(stats_read);
    auto const lse_offset = work.kv_head * get<1>(params.stride_LSE)
        + work.batch * get<2>(params.stride_LSE);
    #pragma unroll
    for (int row_in_tile = lane; row_in_tile < Config::kBlockM; row_in_tile += 32) {
      int const row = work.q_block * Config::kBlockM + row_in_tile;
      if (row < work.seqlen_q) {
        params.ptr_LSE[lse_offset + row] =
            logf(stats.l[row_in_tile]) + params.scale_softmax * stats.m[row_in_tile];
      }
    }
    c.final_stats.consumer_release(stats_read);  // every lane finished its rows

    if (elected) {
      // Completion, not merely issue, permits both the next O write and the
      // next tile's V load to reuse the aliased bytes. Exactly one arrival on
      // each barrier; the other 31 lanes must not release corrected_o.
      tma_store_wait<0>();
      c.corrected_o.consumer_release(output_read);
      c.o_drained.arrive();
    }
    output_phase ^= 1;
    c.work_epilogue.consumer_release(work_read);
    work_read.advance();
  }
}

}  // namespace pipelink::attention::sm90
