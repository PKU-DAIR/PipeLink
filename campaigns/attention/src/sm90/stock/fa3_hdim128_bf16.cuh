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

// FA3 type/argument selection derived from flash_fwd_launch_template.h.
// Upstream copyright (c) 2024 Jay Shah, Ganesh Bikshandi, Ying Zhang,
// Vijay Thakkar, Pradeep Ramani, Tri Dao.
#pragma once
#include "cute/tensor.hpp"
#include "cutlass/device_kernel.h"
#include "flash.h"
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel_sm90.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd.hpp"

namespace pipelink::attention::sm90 {
using namespace cute;

// Deliberately the upstream kernel type, not a wrapper kernel: its mangled
// device_kernel symbol and single by-value Params ABI are the stock image.
template <bool Causal, int ClusterM, bool Is_local = false>
struct StockKernel {
  static_assert(ClusterM == 1 || ClusterM == 2);
  static_assert(!(Causal && Is_local), "causal and local are mutually exclusive FA3 modes");
  static_assert((!Causal && !Is_local) || ClusterM == 1,
                "causal and local cluster-2 are not shipped FA3 configurations");
  using Element = cutlass::bfloat16_t;
  using ElementOut = Element;
  static constexpr bool V_colmajor = false, PackGQA = false;
  static constexpr auto tile = tile_size_fwd_sm90(128, 128, Causal, Is_local, sizeof(Element), false, false, false);
  static constexpr int kBlockM = std::get<0>(tile), kBlockN = std::get<1>(tile);
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<128>>;
  using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<128>, Int<kBlockN>>;
  using ClusterShape = Shape<Int<ClusterM>, _1, _1>;
  using CollectiveMainloop = flash::CollectiveMainloopFwdSm90<
      2, ClusterShape, TileShape_MNK, 128, Element, float, cutlass::arch::Sm90,
      Causal, Is_local, false, false, false, false, false,
      std::get<2>(tile), std::get<3>(tile), false, false, false>;
  using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
      TileShape_MNK_PV, ClusterShape, ElementOut, cutlass::arch::Sm90,
      CollectiveMainloop::NumMmaThreads, false, false, false, false>;
  using Scheduler = std::conditional_t<Causal || Is_local,
      flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads,
          CollectiveMainloop::NumProducerThreads, false, false, true>,
      flash::StaticPersistentTileScheduler<false>>;
  using AttnKernel = flash::enable_sm90<flash::FlashAttnFwdSm90<
      CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

  // Flash_fwd_params is a torch-free plain pointer/stride/shape record. The
  // caller zero-initializes it and supplies the bf16 fields and mask windows.
  static typename AttnKernel::Params make_params(Flash_fwd_params const& params, int device) {
    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    bool const is_varlen_k_new = params.cu_seqlens_knew;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;
    typename CollectiveMainloop::StrideV v_strides =
        cute::conditional_return<!V_colmajor>(
            make_stride(params.v_row_stride, _1{}, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0),
            make_stride(_1{}, params.v_dim_stride, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0));
    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},  // stride_Q
        static_cast<Element*>(params.k_ptr),
        {!params.page_table ? (!is_varlen_k ? params.seqlen_k : params.total_k) : params.page_size,
         params.d, params.h_k, !params.page_table ? batch_k : params.num_pages},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},  // stride_K
        static_cast<Element*>(params.v_ptr),
        params.dv,  // headdim_v
        v_strides,  // stride_V
        static_cast<Element const*>(params.knew_ptr),
        {!is_varlen_k_new ? params.seqlen_knew : params.total_knew, params.d, params.h_k, !is_varlen_k_new ? params.b : 1},  // shape_K_new
        {params.knew_row_stride, _1{}, params.knew_head_stride, !is_varlen_k_new ? params.knew_batch_stride : 0},  // stride_K_new
        static_cast<Element const*>(params.vnew_ptr),
        {params.vnew_row_stride, _1{}, params.vnew_head_stride, !is_varlen_k_new ? params.vnew_batch_stride : 0}, // stride_V_new
        static_cast<Element const*>(params.qv_ptr),
        {params.qv_row_stride, _1{}, params.qv_head_stride, !is_varlen_q ? params.qv_batch_stride : 0},  // stride_Qv
        static_cast<Element const*>(params.rotary_cos_ptr),
        {params.seqlen_k, params.rotary_dim / 2},  // shape_rotary, the seqlen shape doesn't matter
        {params.rotary_dim / 2, _1{}},  // stride_rotary_cos
        static_cast<Element const*>(params.rotary_sin_ptr),
        {params.rotary_dim / 2, _1{}},  // stride_rotary_sin
        params.is_rotary_interleaved,
        params.page_table,
        // if page_size is not set, avoid dividing by zero
        {params.kv_batch_idx ? params.b_k : params.b, !params.page_table ? 0 : params.seqlen_k / params.page_size}, // shape_page_table
        {params.page_table_batch_stride, _1{}},  // stride_page_table
        params.scale_softmax,
        params.q_descale_ptr, params.k_descale_ptr, params.v_descale_ptr,
        {params.q_descale_batch_stride, params.q_descale_head_stride},
        {params.k_descale_batch_stride, params.k_descale_head_stride},
        {params.v_descale_batch_stride, params.v_descale_head_stride},
        params.window_size_left, params.window_size_right, params.attention_chunk,
        params.softcap,
        params.num_splits,
        params.kv_batch_idx,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k,
        params.leftpad_k, params.seqlens_rotary
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q, params.num_splits},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen_q ? params.o_batch_stride : 0, 0}, // stride_O
        static_cast<float*>(params.oaccum_ptr),
        {params.oaccum_row_stride, _1{}, params.oaccum_head_stride, !is_varlen_q ? params.oaccum_batch_stride : 0, params.oaccum_split_stride}, // stride_O_partial
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, 0},  // stride_LSE
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {_1{}, params.lseaccum_head_stride, !is_varlen_q ? params.lseaccum_batch_stride : 0, params.lseaccum_split_stride},  // stride_LSE_partial
        params.h_k,
        params.cu_seqlens_q, params.seqused_q
    };

    int qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, params.num_splits,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k, params.d, params.dv, sizeof(Element), 
        params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q,
        params.num_splits_dynamic_ptr,
        params.num_m_blocks_ptr,
        params.varlen_batch_idx_ptr,
        params.num_nheads_in_l2_ptr
    };
    return AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args});
  }
};
} // namespace pipelink::attention::sm90
