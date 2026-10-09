/*
 * Parameter subsets adapted from FlashAttention-3 hopper mainloop and epilogue.
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

// FA3 type selection and host descriptor construction remain the stock adapter's.
#pragma once
#include "stock/fa3_hdim128_bf16.cuh"

namespace pipelink::attention::sm90 {

template <bool Causal, int ClusterM = 1>
struct FamilyTypes {
  using Stock = StockKernel<Causal, ClusterM>;
  using Mainloop = typename Stock::CollectiveMainloop;
  using Epilogue = typename Stock::CollectiveEpilogue;
  using Scheduler = typename Stock::Scheduler;
  using Element = typename Stock::Element;
  using ClusterShape = typename Stock::ClusterShape;
  static constexpr int kBlockM = Stock::kBlockM;
  static constexpr int kBlockN = Stock::kBlockN;
  static constexpr bool causal = Causal;
};

template <bool Causal, int ClusterM = 1>
struct LoadParams {
  using Types = FamilyTypes<Causal, ClusterM>;
  using Mainloop = typename Types::Mainloop;
  typename Mainloop::TMA_Q tma_load_Q;
  typename Mainloop::TMA_K tma_load_K;
  typename Mainloop::TMA_V tma_load_V;
  typename Mainloop::ShapeQKV shape_Q;
  typename Mainloop::ShapeQKV shape_K;
  typename Mainloop::StrideV stride_V;
  typename Types::Scheduler::Params scheduler;
  int seqlen_q, seqlen_k;
  cutlass::FastDivmod qhead_per_khead;
  int causal;
  int window_size;
  float score_param, score_param_inv;
  int needs_score_mod;
};

struct CoreParams {
  float scale_softmax;
  float scale_softmax_log2;
  int seqlen_q, seqlen_k, qhead_per_khead;
  int window_size;
};

template <bool Causal, int ClusterM = 1>
struct EpilogueParams {
  using Epilogue = typename FamilyTypes<Causal, ClusterM>::Epilogue;
  typename Epilogue::TMA_O tma_store_O;
  typename Epilogue::ShapeO shape_O;
  typename Epilogue::StrideO stride_O;
  typename Epilogue::StrideLSE stride_LSE;
  typename FamilyTypes<Causal, ClusterM>::Element* ptr_O;
  float* ptr_LSE;
  float scale_softmax;
};

struct EmptyParams { int reserved = 0; };

template <bool Causal, int ClusterM = 1>
struct FamilyParams {
  LoadParams<Causal, ClusterM> load;
  CoreParams core;
  EpilogueParams<Causal, ClusterM> epilogue;
  EmptyParams empty;
};

// window_size == 0 selects dense/causal; positive values select the window callee.
// The persistent launch has num_sm CTAs, 384 threads, cluster (1,1,1).
// The caller clears params.tile_count_semaphore before every causal launch.
template <bool Causal, int ClusterM = 1>
inline FamilyParams<Causal, ClusterM> make_family_params(
    Flash_fwd_params const& params, int device, int window_size = 0) {
  auto stock = StockKernel<Causal, ClusterM>::make_params(params, device);
  auto const& m = stock.mainloop;
  auto const& e = stock.epilogue;
  bool const needs_score_mod = params.softcap > 0.f;
  float const score_param = needs_score_mod ? params.softcap / params.scale_softmax : 0.f;
  float const score_param_inv = needs_score_mod ? params.scale_softmax / params.softcap : 0.f;
  return {
      {m.tma_load_Q, m.tma_load_K, m.tma_load_V, m.shape_Q, m.shape_K,
       m.stride_V, stock.scheduler, params.seqlen_q, params.seqlen_k,
       m.qhead_per_khead_divmod, int(params.is_causal), window_size,
       score_param, score_param_inv, int(needs_score_mod)},
      {params.scale_softmax, params.scale_softmax * 1.4426950408889634074f,
       params.seqlen_q, params.seqlen_k, params.h / params.h_k, window_size},
      {e.tma_store_O, e.shape_O, e.stride_O, e.stride_LSE, e.ptr_O,
       e.ptr_LSE, params.scale_softmax},
      {0}};
}

} // namespace pipelink::attention::sm90
