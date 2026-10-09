#pragma once
#include "family_params.cuh"

namespace pipelink::attention::sm90 {

// One local head tile, or gathered rows sharing an exact remote block list.
struct MsaItem {
  int kv_head, block, q_head;  // q_head >= 0 means local; -1 means gathered remote.
  int row_begin, row_count;
  int block_begin, kv_steps;  // Remote CSR list is highest-first; local uses block.
};

struct MsaLoadParams {
  using Mainloop = FamilyTypes<true>::Mainloop;
  typename Mainloop::TMA_Q tma_load_Q;
  typename Mainloop::TMA_K tma_load_K;
  typename Mainloop::TMA_V tma_load_V;
  cute::TmaDescriptor tma_load_Q_packed;  // [D, head-in-group, KV group, token].
  MsaItem const* items;
  int32_t const* tokens;  // Eight gathered query tokens per item; seqlen pads tails.
  int32_t const* blocks;
  int num_items, seqlen;
};

struct MsaPartialParams {
  cute::TmaDescriptor tma_store_O;
  float* partial_m;
  float* partial_l;
};

}  // namespace pipelink::attention::sm90
