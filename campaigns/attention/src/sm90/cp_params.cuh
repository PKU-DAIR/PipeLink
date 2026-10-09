// Context-parallel (CP) causal attention: parameter records shared by the
// `cp-load` variant, the `kv-gather` comm stage and the multi-device CP slot.
//
// Zigzag sharding: rank r owns chunks r and 2W-1-r of C=S/2 tokens,
// packed early then late for Q/O and globally indexed in gathered K/V.
// Early K/V goes to every peer; late K/V goes only to lower ranks.
// One system-scope counter covers one 128-token block of one batch,
// across K, V and all KV heads.
#pragma once

#include <cstdint>
#include "family_params.cuh"

extern "C" __device__ int pipelink_cp_kv_ready_slot();
extern "C" __device__ int pipelink_cp_gather_reads_done();
extern "C" __device__ int pipelink_cp_comm_barrier();
extern "C" __device__ int pipelink_cp_warp_gather();
extern "C" __device__ int pipelink_cp_acquired_blocks();

namespace pipelink::attention::sm90 {

inline constexpr int kCpMaxWorld = 8;
inline constexpr int kCpChunkTokens = 128;
// gmem_counter slots claimed by both endpoints: batch * W*S/128 <= 2048
// (256K gathered tokens at batch one).
inline constexpr int kCpMaxChunks = 2048;

template <bool Causal, int ClusterM = 1>
struct CpLoadParams {
  // FA3 descriptors over this rank's full gathered K/V.
  LoadParams<Causal, ClusterM> load;
  uint32_t const* ready;       // this rank's counter array (slot base added on device)
  uint32_t const* generation;  // this rank's device-resident launch generation
  int local_low_block, local_high_block;  // starts of the two owned chunks
  int chunk_blocks;                     // C/128 blocks per ownership chunk
  int blocks_per_batch;        // W*S/128 counters per batch
};

struct KvGatherParams {
  // Per-rank symmetric allocations, indexed by rank (UVA addresses).
  uint64_t k_gather[kCpMaxWorld];  // [batch, W*S, heads_kv, 128] bf16
  uint64_t v_gather[kCpMaxWorld];
  uint64_t ready[kCpMaxWorld];     // u32 chunk counters, slot base added on device
  uint64_t released[kCpMaxWorld];  // u32[W] cross-launch release flags; released[p][r] written by rank r
  uint32_t* generation;            // local device-resident launch generation (zeroed once)
  uint32_t* queue;                 // local [tickets, done], zeroed every launch
  int rank, world, batch;
  int shard_blocks;                // S/128
  int blocks_per_batch;            // W*S/128
  int row_bytes;                   // heads_kv*128*2: one token of K or V
  int64_t batch_bytes;             // W*S*row_bytes
};

}  // namespace pipelink::attention::sm90
