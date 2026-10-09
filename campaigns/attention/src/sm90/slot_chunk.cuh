#pragma once
#include "family_params.cuh"
#include <stdexcept>

namespace pipelink::attention::sm90 {

inline void validate_chunk_cell(int batch, int seqlen, int chunk) {
  if (batch != 1 || chunk != 8192 || seqlen % chunk != 0)
    throw std::runtime_error("chunked cells require batch=1 and complete 8192-token chunks");
}

// Oracle uses original coordinates; family work uses chunk-local coordinates.
__host__ __device__ inline bool chunk_visible(int q, int k, int chunk) {
  return k <= q && k / chunk == q / chunk;
}

inline Flash_fwd_params chunk_virtual_arguments(Flash_fwd_params params, int chunk) {
  int const seqlen = params.seqlen_q;
  params.b = params.b_k = seqlen / chunk;
  params.seqlen_q = params.seqlen_k = chunk;
  params.seqlen_q_rounded = params.seqlen_k_rounded = chunk;
  params.total_q = params.total_k = seqlen;
  params.q_batch_stride = params.q_row_stride * chunk;
  params.o_batch_stride = params.o_row_stride * chunk;
  params.k_batch_stride = params.k_row_stride * chunk;
  params.v_batch_stride = params.v_row_stride * chunk;
  params.is_causal = true;
  params.window_size_left = chunk - 1;
  params.window_size_right = 0;
  return params;
}

template <bool Causal>
inline FamilyParams<Causal> make_chunk_family_params(
    Flash_fwd_params const& params, int device, int chunk) {
  static_assert(Causal, "chunk virtual batches reuse the causal composition");
  auto family = make_family_params<Causal>(chunk_virtual_arguments(params, chunk), device);
  // Preserve [1,Hq,S] rather than the virtual [chunks,Hq,chunk] LSE layout.
  cute::get<1>(family.epilogue.stride_LSE) = params.seqlen_q;
  cute::get<2>(family.epilogue.stride_LSE) = chunk;
  return family;
}

} // namespace pipelink::attention::sm90
