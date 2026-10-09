#pragma once
#include "fa3_family.cuh"

namespace pipelink::attention::sm90 {

// One 128-column step; a missing high half is zero-filled by TMA and masked.
struct PhiBlockPair {
  int32_t lo, hi;
  uint32_t flags;
};

// CSR rows use logical Q heads, independently of physical GQA KV addressing.
struct BlockLoadParams {
  LoadParams<true> base;
  // 64 rows x 64 dimensions per box: two dimension planes per half tile.
  cute::TmaDescriptor tma_half_K, tma_half_V;
  int32_t const* row_ptr;
  PhiBlockPair const* block_pairs;
  int32_t const* work_order;  // CSR row IDs, longest lists first.
  int32_t num_work;
  cutlass::FastDivmod q_tiles_div, heads_q_div;
};

} // namespace pipelink::attention::sm90
