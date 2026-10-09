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

#pragma once
#include "msa_params.cuh"
#include "role_load.cuh"

namespace pipelink::attention::sm90 {

template <class Config>
CUTLASS_DEVICE void run_msa_load(MsaLoadParams const& params, SplitContext<Config>& c) {
  using namespace cute;
  using Mainloop = typename Config::Mainloop;
  using PipelineQ = typename Config::PipelineQ;
  using PipelineKV = typename Config::PipelineKV;
  static_assert(Config::kBlockM == 128 && Config::kBlockN == 128);
  int const lane = threadIdx.x % 32;
  bool const elected = lane == 0;
  c.initialize();  // All roles join the common 384-thread rendezvous.
  if (!elected) return;  // TMA-only load: no cooperative row-map writers remain.
  if (elected) {
    prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    prefetch_tma_descriptor(&params.tma_load_Q_packed);
  }
  // Q/output phases advance per item; K/V/metadata advance per KV step.
  // Counts avoid retaining several PipelineState triples in the producer warp.
  auto q_state = [](uint32_t count) {
    return typename PipelineQ::PipelineState{0, (count & 1u) ^ 1u, count};
  };
  auto kv_state = [](uint32_t count) {
    return typename PipelineKV::PipelineState{
        int(count & 1u), ((count >> 1) & 1u) ^ 1u, count};
  };
  uint32_t count = 0, kv_count = 0;
  Tensor sQ = make_tensor(make_smem_ptr(c.shared.tensors.smem_q.data()),
                          typename Mainloop::SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(c.shared.tensors.smem_k.data()),
                          typename Mainloop::SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(c.shared.tensors.smem_v.data()),
                          typename Mainloop::SmemLayoutVt{});
  auto tma_Q = params.tma_load_Q.get_slice(_0{});
  auto tma_K = params.tma_load_K.get_slice(_0{});
  auto tma_V = params.tma_load_V.get_slice(_0{});
  Tensor tQsQ = group_modes<0, 3>(tma_Q.partition_D(sQ));
  Tensor tKsK = group_modes<0, 3>(tma_K.partition_D(sK));
  Tensor tVsV = group_modes<0, 3>(tma_V.partition_D(sV));
  cutlass::arch::wait_on_dependent_grids();
  for (int index = blockIdx.x; index < params.num_items; index += gridDim.x, ++count) {
    MsaItem const& item = params.items[index];
    bool const local = item.q_head >= 0;
    if (elected) {
      uint32_t work_count = count;
      publish_load_work(c, WorkDesc{
          local ? item.block : 0, 0, local ? item.q_head : item.kv_head,
          item.kv_steps, params.seqlen, params.seqlen, local ? item.block * 128 : 0,
          index, WorkFlags::valid | WorkFlags::gathered_q, 1u, 0, 0.f, 0.f}, work_count);
    }

    if (elected) {
      publish_load_kv(c, params.seqlen, item.block * 128, 0,
                      item.block, local, kv_count);
      Tensor mK = params.tma_load_K.get_tma_tensor(
          make_shape(params.seqlen, Int<128>{}, 4, 1))(_, _, item.kv_head, 0);
      Tensor gK = local_tile(mK, Shape<_128, _128>{}, make_coord(_, _0{}));
      Tensor tKgK = group_modes<0, 3>(tma_K.partition_S(gK));
      c.k.producer_acquire(kv_state(kv_count));
      copy(params.tma_load_K.with(*c.k.producer_get_barrier(kv_state(kv_count)), 0,
                                 TMA::CacheHintSm90::EVICT_LAST),
           tKgK(_, item.block), tKsK(_, int(kv_count & 1u)));
      c.q.producer_acquire(q_state(count));
    }
    if (local) {
      if (elected) {
        Tensor mQ = params.tma_load_Q.get_tma_tensor(
            make_shape(params.seqlen, Int<128>{}, 64, 1))(_, _, item.q_head, 0);
        Tensor gQ = local_tile(mQ, Shape<_128, _128>{}, make_coord(item.block, _0{}));
        Tensor tQgQ = group_modes<0, 3>(tma_Q.partition_S(gQ));
        copy(params.tma_load_Q.with(*c.q.producer_get_barrier(q_state(count)), 0,
                                   TMA::CacheHintSm90::EVICT_FIRST), tQgQ, tQsQ);
      }
    } else if (elected) {
      auto* barrier = c.q.producer_get_barrier(q_state(count));
      auto* destination = c.shared.tensors.smem_q.data();
      // Eight whole-token GQA16 groups; each SW128 plane needs a 2-KiB box.
      // Box bases are 2-KiB aligned, so their SW128 base xor is zero.
      #pragma unroll 1
      for (int token = 0; token < 8; ++token) {
        int const q = params.tokens[index * 8 + token];  // Tail tokens equal seqlen.
        CUTLASS_PRAGMA_UNROLL
        for (int plane = 0; plane < 2; ++plane)
          SM90_TMA_LOAD::copy(&params.tma_load_Q_packed, barrier,
              uint64_t(TMA::CacheHintSm90::EVICT_FIRST),
              destination + token * 1024 + plane * 8192, plane * 64, 0, item.kv_head, q);
      }
      // All sixteen boxes complete the existing 32-KiB Q transaction.
    }
    if (elected) {
      // V stage zero aliases output: only epilogue may grant its next lifetime.
      c.o_drained.wait((count & 1u) ^ 1u);
      Tensor mK = params.tma_load_K.get_tma_tensor(
          make_shape(params.seqlen, Int<128>{}, 4, 1))(_, _, item.kv_head, 0);
      Tensor gK = local_tile(mK, Shape<_128, _128>{}, make_coord(_, _0{}));
      Tensor tKgK = group_modes<0, 3>(tma_K.partition_S(gK));
      Tensor mV = params.tma_load_V.get_tma_tensor(
          make_shape(Int<128>{}, params.seqlen, 4, 1))(_, _, item.kv_head, 0);
      Tensor gV = local_tile(mV, Shape<_128, _128>{}, make_coord(_0{}, _));
      Tensor tVgV = group_modes<0, 3>(tma_V.partition_S(gV));
      auto load_v = [&](int block, uint32_t step) {
        c.v.producer_acquire(kv_state(step));
        copy(params.tma_load_V.with(*c.v.producer_get_barrier(kv_state(step)), 0,
                                   TMA::CacheHintSm90::EVICT_LAST),
             tVgV(_, block), tVsV(_, int(step & 1u)));
      };
      int previous = item.block;
      #pragma unroll 1
      for (int step = 1; step < item.kv_steps; ++step) {
        int const block = params.blocks[item.block_begin + step];
        ++kv_count;
        publish_load_kv(c, params.seqlen, 0, 0, block, false, kv_count);
        c.k.producer_acquire(kv_state(kv_count));
        copy(params.tma_load_K.with(*c.k.producer_get_barrier(kv_state(kv_count)), 0,
                                   TMA::CacheHintSm90::EVICT_LAST),
             tKgK(_, block), tKsK(_, int(kv_count & 1u)));
        load_v(previous, kv_count - 1);  // Dense ordering: V trails K by one step.
        previous = block;
      }
      load_v(previous, kv_count);
      ++kv_count;
    }
  }
  if (elected) {
    uint32_t work_count = count;
    publish_load_work(c, WorkDesc{}, work_count);
    c.o_drained.wait((count & 1u) ^ 1u);
    c.q.producer_tail(q_state(count));
    c.k.producer_tail(kv_state(kv_count));
    c.v.producer_tail(kv_state(kv_count));
  }
}

}  // namespace pipelink::attention::sm90
