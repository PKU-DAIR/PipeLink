#pragma once
#include "block_load_params.cuh"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace pipelink::attention::sm90 {

struct PhiPattern {
  int block, local_blocks, vert_stride, head_sliding_step;
  bool homo_head;
};

template <class Json>
inline PhiPattern phi_pattern(Json const& mask) {
  PhiPattern p{mask.at("block").integer(), mask.at("local_blocks").integer(),
               mask.at("vert_stride").integer(), mask.at("head_sliding_step").integer(),
               mask.at("homo_head").boolean()};
  if (p.block != 64 || p.local_blocks != 16 || p.vert_stride != 8 ||
      p.head_sliding_step != 1 || p.homo_head)
    throw std::runtime_error("Phi package requires block=64 local=16 stride=8 step=1 homo_head=false");
  return p;
}

// Independent token-level oracle, rather than interpreting load metadata/CSR.
__host__ __device__ inline bool phi_visible(int q, int k, int head) {
  return k >= 0 && k <= q &&
      (q / 64 - k / 64 < 16 || (k / 64 + head + 1) % 8 == 0);
}

struct PhiPlan {
  int q_tiles;
  std::vector<int32_t> row_ptr{0}, work_order;
  std::vector<PhiBlockPair> block_pairs;
  uint64_t useful_pairs = 0, fully_valid_steps = 0, half_valid_steps = 0;

  PhiPlan(int batch, int heads, int seqlen, PhiPattern const& p) : q_tiles(seqlen / 128) {
    if (seqlen % 128 != 0)
      throw std::runtime_error("Phi requires complete 128-row query tiles");
    int const rows = batch * heads * q_tiles;
    row_ptr.reserve(rows + 1);
    auto visible = [&](int qb, int kb, int h) {
      return kb <= qb && (qb - kb < p.local_blocks ||
          (kb + h * p.head_sliding_step + 1) % p.vert_stride == 0);
    };
    for (int b = 0; b < batch; ++b) {
      for (int h = 0; h < heads; ++h) {
        for (int t = 0; t < q_tiles; ++t) {
          int const qb = t * 128 / p.block;
          int pending = -1, edge = -1;
          for (int kb = 0; kb < qb; ++kb) {
            bool const lo = visible(qb, kb, h), hi = visible(qb + 1, kb, h);
            useful_pairs += uint64_t(lo + hi) * p.block * p.block;
            if (lo && hi) {
              if (pending < 0) pending = kb;
              else {
                block_pairs.push_back({pending, kb, KvFlags::fully_valid});
                ++fully_valid_steps;
                pending = -1;
              }
            } else if (lo || hi) {
              // Only the local-band lower edge can differ below the diagonal.
              edge = kb;
            }
          }
          // Keep the row-dependent edge out of otherwise fully-valid pairs.
          if (edge >= 0 || pending >= 0) {
            int const first = pending >= 0 ? pending : edge;
            int const second = pending >= 0 ? edge : -1;
            block_pairs.push_back({first, second, edge >= 0 ? KvFlags::needs_mask_mod : 0u});
            half_valid_steps += second < 0;
          }
          // Reverse traversal starts with the contiguous causal diagonal,
          // ensuring at least one live score for every row in the first step.
          block_pairs.push_back({qb, qb + 1, KvFlags::fully_valid |
              KvFlags::needs_causal | contiguous | (128u << contiguous_valid_shift)});
          useful_pairs += uint64_t(128) * 129 / 2;
          row_ptr.push_back(int32_t(block_pairs.size()));
        }
      }
    }
    work_order.resize(rows);
    std::iota(work_order.begin(), work_order.end(), 0);
    std::sort(work_order.begin(), work_order.end(), [&](int a, int b) {
      int const na = row_ptr[a + 1] - row_ptr[a], nb = row_ptr[b + 1] - row_ptr[b];
      return na != nb ? na > nb : a > b;
    });
  }
};

// Created before any warmup/timing, shared by linked/control, alive through launches.
class PhiWorkspace {
  int32_t *rows_ = nullptr, *order_ = nullptr;
  PhiBlockPair* pairs_ = nullptr;
  int q_tiles_ = 0;
  cute::TmaDescriptor half_k_{}, half_v_{};

  static cute::TmaDescriptor half_descriptor(
      void* ptr, int seqlen, int heads, int batch,
      int64_t row_stride, int64_t head_stride, int64_t batch_stride) {
    cute::TmaDescriptor desc{};
    cuuint64_t dims[] = {128, cuuint64_t(seqlen), cuuint64_t(heads), cuuint64_t(batch)};
    cuuint64_t strides[] = {cuuint64_t(row_stride) * 2, cuuint64_t(head_stride) * 2,
                            cuuint64_t(batch_stride) * 2};
    cuuint32_t box[] = {64, 64, 1, 1}, element_strides[] = {1, 1, 1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, ptr,
        dims, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return desc;
  }
 public:
  PhiWorkspace(bool enabled, Flash_fwd_params const& args, PhiPattern const& pattern) {
    if (!enabled) return;
    auto const begin = std::chrono::steady_clock::now();
    PhiPlan plan(args.b, args.h, args.seqlen_q, pattern);
    half_k_ = half_descriptor(args.k_ptr, args.seqlen_k, args.h_k, args.b,
                             args.k_row_stride, args.k_head_stride, args.k_batch_stride);
    half_v_ = half_descriptor(args.v_ptr, args.seqlen_k, args.h_k, args.b,
                             args.v_row_stride, args.v_head_stride, args.v_batch_stride);
    q_tiles_ = plan.q_tiles;
    auto const built = std::chrono::steady_clock::now();
    auto upload = [](auto*& dst, auto const& src) {
      size_t const bytes = src.size() * sizeof(src[0]);
      CUDA_CHECK(cudaMalloc(&dst, bytes));
      CUDA_CHECK(cudaMemcpy(dst, src.data(), bytes, cudaMemcpyHostToDevice));
    };
    upload(rows_, plan.row_ptr);
    upload(pairs_, plan.block_pairs);
    upload(order_, plan.work_order);
    auto const done = std::chrono::steady_clock::now();
    std::cerr << "phi_setup selected_tiles=" << plan.block_pairs.size()
              << " executed_positions=" << uint64_t(plan.block_pairs.size()) * 128 * 128
              << " fully_valid_steps=" << plan.fully_valid_steps
              << " fully_valid_fraction=" << double(plan.fully_valid_steps) / plan.block_pairs.size()
              << " half_valid_steps=" << plan.half_valid_steps
              << " useful_pairs=" << plan.useful_pairs
              << " build_ms=" << std::chrono::duration<double, std::milli>(built - begin).count()
              << " upload_ms=" << std::chrono::duration<double, std::milli>(done - built).count()
              << " scheduling=longest-list-first\n";
  }
  ~PhiWorkspace() { cudaFree(rows_); cudaFree(pairs_); cudaFree(order_); }
  PhiWorkspace(PhiWorkspace const&) = delete;
  PhiWorkspace& operator=(PhiWorkspace const&) = delete;
  BlockLoadParams params(LoadParams<true> const& base) const {
    return {base, half_k_, half_v_, rows_, pairs_, order_,
            q_tiles_ * cute::get<2>(base.shape_Q) * cute::get<3>(base.shape_Q),
            cutlass::FastDivmod(q_tiles_), cutlass::FastDivmod(cute::get<2>(base.shape_Q))};
  }
};

} // namespace pipelink::attention::sm90
