// Comm owner, AllGather stage for the GEMM CTA contract (push publication, device-resident launch generation, cross-launch release wait, CTA read-drain; one warp).
// Each rank pushes its own [rows_per_rank, K] shard of the symmetric gathered buffer
// into every peer's copy, one 128 x ChunkCols chunk per queue ticket, and releases
// the peer's per-row-block counter. Consumers wait for (gen + 1) * K / ChunkCols.
#include "comm.h"
#include "push_gather.cuh"
#include <pipelink/stage.hpp>

namespace comm_overlap::ag {
inline constexpr int ChunkRows = 128;
inline constexpr int ChunkCols = 256;       // bf16 columns: 512 bytes = one 16-byte vector per lane
inline constexpr int MaxRowBlocks = 512;    // counter slots claimed: M <= 65536
}

extern "C" __device__ int pipelink_comm_ready_slot();
extern "C" __device__ int pipelink_comm_gather_reads_done();

extern "C" __device__ void comm_gather_role(const void* opaque, unsigned) {
  using namespace comm_overlap;
  using namespace comm_overlap::ag;
  auto const& view = *static_cast<plk_role_view_comm_gather_comm const*>(opaque);
  auto const& p = *static_cast<CommAgResidue const*>(view.residue);
  extern __shared__ __align__(1024) unsigned char comm_smem[];
  const uint32_t generation = latch(p.generation);
  const int row_blocks = p.rows_per_rank / ChunkRows;
  const int k_chunks = p.K / ChunkCols;
  const uint32_t per_row_block = uint32_t((p.world - 1) * k_chunks);
  const uint32_t tasks = uint32_t(row_blocks) * per_row_block;
  const size_t row_bytes = size_t(p.K) * 2;
  const size_t shard_offset = size_t(p.rank) * p.rows_per_rank * row_bytes;
  const auto* local = reinterpret_cast<const unsigned char*>(p.gather[p.rank]) + shard_offset;
  const auto* released_local = reinterpret_cast<const uint32_t*>(p.released[p.rank]);
  const int slot_base = pipelink_comm_ready_slot() + p.rank * row_blocks;
  unsigned cleared = 0;  // peers whose previous launch has released their gathered buffer
  // Every resident CTA drains the queue regardless of its own consumer's waits (push-publication liveness).
  for (uint32_t task = warp_ticket(p.tickets); task < tasks; task = warp_ticket(p.tickets)) {
    // Row-block major: early row-blocks complete first on every peer.
    const int row_block = int(task / per_row_block);
    const int rest = int(task % per_row_block);
    const int peer = (p.rank + 1 + rest / k_chunks) % p.world;
    const int chunk = rest % k_chunks;
    if (!(cleared & (1u << peer))) {
      // Cross-launch release wait: the peer finished reading its copy in every earlier launch.
      warp_wait_released(released_local, peer, generation);
      cleared |= 1u << peer;
    }
    const size_t offset = size_t(row_block) * ChunkRows * row_bytes + size_t(chunk) * ChunkCols * 2;
    auto* remote = reinterpret_cast<unsigned char*>(p.gather[peer]) + shard_offset;
    warp_copy(remote + offset, local + offset, ChunkRows, ChunkCols * 2 / 16, row_bytes);
    warp_publish_sys(reinterpret_cast<uint32_t*>(p.ready[peer]) + slot_base + row_block);
  }
  // CTA read-drain: join this CTA's load stage, whose gathered-buffer TMA reads have landed.
  const unsigned reads_done = __cvta_generic_to_shared(comm_smem + 16 * pipelink_comm_gather_reads_done());
  asm volatile("{ .reg .pred done; WAIT: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], 0; "
               "@!done bra WAIT; }" :: "r"(reads_done) : "memory");
  warp_done(p.tickets + 1, p.generation, generation, p.released, p.rank, p.world);
}

PL_STAGE("comm.gather", "1.0.0");
PL_NO_PARAMS();
PL_ROLE(comm, comm_gather_role, 1);
PL_CLAIM_EX(pl::claim("pipelink_comm_warp", "warp").extent(1).align(1).at(10).roles({"comm"}));
PL_CLAIM("pipelink_comm_ready_slot", "gmem_counter", comm_overlap::ag::MaxRowBlocks, 1, "kernel");
PL_CLAIM("pipelink_comm_gather_reads_done", "mbarrier", 1, 1, "kernel");
PL_ENDPOINT(chunk_ready, pl::endpoint("chunk_ready", pl::produce, "comm")
    .space("gmem").scope("system").proxy("generic").tile("u32", {1})
    .layout(pl::affine{{1}, {1}, 32})
    .protocol(pl::counter_sys{"pipelink_comm_ready_slot"}));
PL_ENDPOINT(gather_reads_done, pl::signal_endpoint("gather_reads_done", pl::consume, "comm")
    .scope("cta")
    .protocol(pl::mbarrier{pl::side{"pipelink_comm_gather_reads_done", {"load", 1}, 0},
                           std::nullopt, {"load", 0, "dispatch"}, "never"})
    .bind("full", "pipelink_comm_gather_reads_done"));

PL_HOST_GROUP("comm.owner", "comm.owner.ag", "comm_ag", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "tp");
PL_HOST_ABI_TYPE(100, CommAgResidue, gather, ready, released, generation, tickets, rank, world,
                 rows_per_rank, K);
PL_HOST_ABI_TYPE(101, plk_role_view_comm_gather_comm, residue, scalars);
PL_HOST_RESIDUE(2, "comm.owner.ag.residue", CommAgResidue, 100, true, pl::host::roles({"comm.gather:comm"}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"comm.gather:comm"}))
    .field(pl::host::scalar(1, "gemm.M", "i64").caller("M"))
    .field(pl::host::scalar(3, "gemm.K", "i64").caller("K"))
    .field(pl::host::scalar(4, "comm.ag.chunk_rows", "i64").fact("comm.ag.chunk_rows"))
    .field(pl::host::scalar(5, "comm.ag.chunk_cols", "i64").fact("comm.ag.chunk_cols"))
    .field(pl::host::scalar(6, "comm.ag.ready_slot", "i64").binding("pipelink_comm_ready_slot")));
// One symmetric caller allocation per rank; the comm owner writes peers' copies.
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("comm.ag.gather", "bf16").rank(2).read_write()
    .caller("gather").peer("tp"));
PL_HOST_CHANNEL(2, uint32_t, pl::host::channel("comm.ag.ready", "u32").rank(1).read_write()
    .caller("ready").peer("tp"));
PL_HOST_CHANNEL(3, uint32_t, pl::host::channel("comm.ag.released", "u32").rank(1).read_write()
    .caller("released").peer("tp"));
PL_HOST_CHANNEL(4, uint32_t, pl::host::channel("comm.ag.gen", "u32").rank(1).read_write().caller("gen"));
PL_HOST_WORKSPACE(1, 4, pl::host::workspace("comm.ag.tickets").zero().reset_launch());
// Consumers of chunk_ready need the counter granularity and per-launch arrival count.
PL_HOST_FACT(pl::host::fact("comm.ag.chunk_rows").i32(comm_overlap::ag::ChunkRows)
    .owner("comm.owner").stage("comm.gather").consumers({"gemm.owner"}));
PL_HOST_FACT(pl::host::fact("comm.ag.chunk_cols").i32(comm_overlap::ag::ChunkCols)
    .owner("comm.owner").stage("comm.gather").consumers({"gemm.owner"}));
PL_HOST_SCHEDULER(pl::host::scheduler("gemm.owner"));
PL_HOST_TARGET();
