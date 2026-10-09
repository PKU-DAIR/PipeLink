// Comm owner, KV gather for the sm90 attention CTA contract (shared push publication, device-resident launch generation, cross-launch release wait, CTA read-drain).
// Replaces `empty` on aux warps 2-3: performs empty's initialization join, then
// both warps drain the rank-local ticket queue. One ticket = one 128-token chunk
// (K and V, all KV heads) of one owned chunk, pushed by one warp into a
// peer's gathered buffers, then released on that peer's block counter.
// The generic copy/queue/counter/generation/release code is push_gather.cuh.
#include "split_context.cuh"
#include "cp_params.cuh"
#include "push_gather.cuh"
#include <pipelink/stage.hpp>

#if !defined(PIPELINK_SM90_SPEC)
#error "select one sm90 attention specialization"
#endif

extern "C" __device__ void PIPELINK_SM90_SYMBOL(kv_gather_role_gather)(
    void const* raw_params, unsigned variant) {
  using namespace pipelink::attention::sm90;
  using namespace comm_overlap;
  int local_warp = cutlass::canonical_warp_idx_sync() - pipelink_cp_warp_gather();
  if (variant != 0 || local_warp < 0 || local_warp >= 2) asm volatile("trap;");
  auto const& p = *reinterpret_cast<KvGatherParams const*>(raw_params);
  // Device-resident launch generation: latched before this CTA can take its done ticket.
  uint32_t const generation = latch(p.generation);
  extern __shared__ char storage[];
  SplitContext<Config> context(storage, Role::Empty);
  context.initialize();

  // Descending global chunks: late chunk to lower ranks, then early chunk
  // to every peer. Within each chunk deliver blocks highest first.
  uint32_t const chunk_blocks = uint32_t(p.shard_blocks / 2);
  uint32_t const chunks = uint32_t(p.batch) * chunk_blocks;
  uint32_t const high_tasks = chunks * uint32_t(p.rank);
  uint32_t const tasks = high_tasks + chunks * uint32_t(p.world - 1);
  uint32_t const* released_local = reinterpret_cast<uint32_t const*>(p.released[p.rank]);
  int const ready_slot = pipelink_cp_kv_ready_slot();
  unsigned cleared = 0;
  for (uint32_t task = warp_ticket(p.queue); task < tasks; task = warp_ticket(p.queue)) {
    bool const high = task < high_tasks;
    uint32_t const part = high ? task : task - high_tasks;
    uint32_t const peers = uint32_t(high ? p.rank : p.world - 1);
    int const peer_index = int(part % peers);
    int const peer = high ? peer_index : peer_index + (peer_index >= p.rank);
    uint32_t const chunk = chunks - 1u - part / peers;
    int const batch = int(chunk / chunk_blocks);
    int const owner_chunk = high ? 2 * p.world - 1 - p.rank : p.rank;
    int const block = owner_chunk * int(chunk_blocks) + int(chunk % chunk_blocks);
    if (!(cleared & (1u << peer))) {
      // Cross-launch release wait: the peer finished reading its gathered buffers in every earlier launch.
      warp_wait_released(released_local, peer, generation);
      cleared |= 1u << peer;
    }
    size_t const offset = size_t(batch) * size_t(p.batch_bytes) +
                          size_t(block) * kCpChunkTokens * size_t(p.row_bytes);
    // A chunk is contiguous: 128 tokens x all KV heads (a multiple of 32*16*4 bytes).
    size_t const chunk_bytes = size_t(kCpChunkTokens) * size_t(p.row_bytes);
    warp_copy_contiguous<4>(reinterpret_cast<unsigned char*>(p.k_gather[peer]) + offset,
                            reinterpret_cast<unsigned char const*>(p.k_gather[p.rank]) + offset, chunk_bytes);
    warp_copy_contiguous<4>(reinterpret_cast<unsigned char*>(p.v_gather[peer]) + offset,
                            reinterpret_cast<unsigned char const*>(p.v_gather[p.rank]) + offset, chunk_bytes);
    warp_publish_sys(reinterpret_cast<uint32_t*>(p.ready[peer]) + ready_slot +
                     batch * p.blocks_per_batch + block);
  }
  // Both comm warps' pushes are fenced; only the comm warps execute this barrier.
  asm volatile("bar.sync %0, %1;" :: "r"(pipelink_cp_comm_barrier()), "n"(64) : "memory");
  if (local_warp != 0) return;
  // CTA read-drain: join cp-load, whose gathered-buffer TMA reads have all landed. It
  // signals only after its last issue, so poll with a sleep backoff instead of
  // occupying an issue slot shared with core warps for the whole launch.
  uint32_t const reads_done = cute::cast_smem_ptr_to_uint(storage + 16 * pipelink_cp_gather_reads_done());
  for (uint32_t done = 0;;) {
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(reads_done) : "memory");
    if (done) break;
    __nanosleep(1000);
  }
  warp_done(p.queue + 1, p.generation, generation, p.released, p.rank, p.world);
}

#if defined(PIPELINK_EMIT)
using namespace pipelink::attention;
using namespace pipelink::attention::sm90;
using Facts = sm90::AnnotationFacts<Config>;
using StageParams = KvGatherParams;
PL_STAGE("org.pipelink.attention." PIPELINK_SM90_SPEC ".kv-gather", "1.0.0+pipelink.attention");
PL_PARAMS(StageParams, k_gather, v_gather, ready, released, generation, queue, rank, world, batch,
          shard_blocks, blocks_per_batch, row_bytes, batch_bytes);
#define SM90_ROLE(name, fn, warps) PL_ROLE(name, fn, warps)
SM90_ROLE(gather, PIPELINK_SM90_SYMBOL(kv_gather_role_gather), 2);
#undef SM90_ROLE
PL_CLAIM_EX(pl::claim("pipelink_cp_warp_gather", "warp").extent(2).align(1).at(2).roles({"gather"}));
PL_CLAIM_REGS("pipelink_attention_regs_aux_wg", "aux_wg", {40}, "gather");
PL_CLAIM("pipelink_attention_smem_base", "smem", Facts::shared_bytes, Facts::shared_align, "kernel");
PL_CLAIM("pipelink_attention_mbarrier_base", "mbarrier", Facts::pipeline_slots, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_q_full", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_q_empty", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_k_full", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_k_empty", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_v_full", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_v_empty", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_work_core_full", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_work_core_empty", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_work_epilogue_full", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_work_epilogue_empty", "mbarrier", 2, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_kv_index_full", "mbarrier", 4, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_kv_index_empty", "mbarrier", 4, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_corrected_o_full", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_corrected_o_empty", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_final_stats_full", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_final_stats_empty", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_attention_slots_o_drained_full", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_cp_kv_ready_slot", "gmem_counter", kCpMaxChunks, 1, "kernel");
PL_CLAIM("pipelink_cp_gather_reads_done", "mbarrier", 1, 1, "kernel");
// Comm-only counted barrier (2 warps) before the CTA-done ticket (push publication, CTA read-drain).
PL_CLAIM("pipelink_cp_comm_barrier", "named_barrier", 1, 1, "kernel", "gather");
PL_TOPOLOGY(pl::topology{1, pl::cluster{{1, 1, 1}, pl::launch{1, {1, 1, 1}}}, true, "static", {}});

PL_ENDPOINT(kv_ready, pl::endpoint("kv_ready", pl::produce, "gather")
    .space("gmem").scope("system").proxy("generic").tile("u32", {1})
    .layout(pl::affine{{1}, {1}, 32})
    .protocol(pl::counter_sys{"pipelink_cp_kv_ready_slot"}));
PL_ENDPOINT(gather_reads_done, pl::signal_endpoint("gather_reads_done", pl::consume, "gather")
    .scope("cta")
    .protocol(pl::mbarrier{pl::side{"pipelink_cp_gather_reads_done", {"load", 1}, 0},
                           std::nullopt, {"load", 0, "dispatch"}, "never"})
    .bind("full", "pipelink_cp_gather_reads_done"));
#endif
