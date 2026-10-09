// Push-only RS owner. This linked GEMM epilogue NEVER waits on peer producers.
// Owner-local reduction is a separate owner kernel launched afterwards on the
// same stream; see reduce_rs.cu. The GEMM owner's load/mma packages stay unchanged.
#include "comm.h"
#include "push_gather.cuh"
#include <pipelink/stage.hpp>

#ifndef COMM_RS_WARPS
#define COMM_RS_WARPS 2
#endif
static_assert(COMM_RS_WARPS >= 1 && COMM_RS_WARPS <= 3);

extern "C" __device__ int pipelink_gemm_output();
extern "C" __device__ int pipelink_gemm_ready();
extern "C" __device__ int pipelink_gemm_drained();
extern "C" __device__ int pipelink_rs_barrier();

extern "C" __device__ void comm_epilogue_rs_role(const void* opaque, unsigned) {
  using namespace comm_overlap;
  constexpr int Threads = 32 * COMM_RS_WARPS;
  const unsigned tid = threadIdx.x - 9 * 32;
  auto const& view = *static_cast<plk_role_view_comm_epilogue_rs_epilogue const*>(opaque);
  auto const& p = *static_cast<CommRsResidue const*>(view.residue);
  extern __shared__ __align__(1024) unsigned char rs_smem[];
  const uint64_t generation = latch(p.generation);
  const int parity = int(generation & 1u);
  const int rows_per_owner = p.M / p.world;
  const int row_blocks = rows_per_owner / 128;
  const int col_blocks = p.N / 128;
  const int owner = int(blockIdx.x) / row_blocks;
  const int local_tile = int(blockIdx.x) % row_blocks * col_blocks + int(blockIdx.y);
  const int tiles = row_blocks * col_blocks;
  const size_t source_stride = size_t(rows_per_owner) * (p.N / 4);
  const size_t parity_stride = size_t(parity) * p.world * source_stride;
  auto* destination = reinterpret_cast<float4*>(p.slots[owner]) + parity_stride + size_t(p.rank) * source_stride;
  const size_t origin = size_t(local_tile / col_blocks) * 128 * (p.N / 4) + (local_tile % col_blocks) * 32;
  const unsigned ready = __cvta_generic_to_shared(rs_smem + 16 * pipelink_gemm_ready());
  asm volatile("{ .reg .pred done; WAIT: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], 0; "
               "@!done bra WAIT; }" :: "r"(ready) : "memory");
  auto const* tile = reinterpret_cast<float4 const*>(rs_smem + pipelink_gemm_output());
  for (int i = tid; i < 4096; i += Threads)
    destination[origin + size_t(i / 32) * (p.N / 4) + i % 32] = tile[i];
  // One system publication point AFTER all stores: every remote writer fences,
  // then the memory-ordering comm-only barrier hands its stores to the publisher.
  // Own-slot visibility instead comes from the same-stream kernel boundary.
  if (owner != p.rank) fence_acq_rel_sys();
  if constexpr (COMM_RS_WARPS == 1) __syncwarp();
  else asm volatile("bar.sync %0, %1;" :: "r"(pipelink_rs_barrier()), "n"(Threads) : "memory");
  if (tid == 0 && owner != p.rank) {
    auto* counter = reinterpret_cast<uint32_t*>(p.counters[owner]) + parity * tiles + local_tile;
    asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"(counter) : "memory");
  }
  // Keep the frozen accumulator protocol's exact 32 arrivals. All extra-warps'
  // shared reads ended at the comm-only barrier; only warp 9 arrives on drained.
  if (tid < 32) {
    const unsigned drained = __cvta_generic_to_shared(rs_smem + 16 * pipelink_gemm_drained());
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(drained) : "memory");
  }
  // No cross-launch release flag wait: p cannot begin g+2 until its g+1 owner reduction consumed
  // every o's g+1 push; o cannot push g+1 until its g reduction finished reading
  // the parity slots p would overwrite. W>=2, stream order and double buffering
  // therefore enforce WAR without a per-tile released-flag handoff.
}

PL_STAGE("comm.epilogue.rs", "2.0.0");
PL_NO_PARAMS();
PL_ROLE(epilogue, comm_epilogue_rs_role, COMM_RS_WARPS);
PL_CLAIM_EX(pl::claim("pipelink_gemm_warp_epilogue", "warp")
    .extent(COMM_RS_WARPS).align(1).at(9).roles({"epilogue"}));
PL_CLAIM("pipelink_rs_barrier", "named_barrier", 1, 1, "kernel", "epilogue");
PL_CLAIM("pipelink_gemm_output", "smem", 128 * 128 * 4, 128, "kernel");
PL_CLAIM("pipelink_gemm_ready", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_gemm_drained", "mbarrier", 1, 1, "kernel");
PL_ENDPOINT(accumulator, pl::endpoint("accumulator", pl::consume, "epilogue")
    .space("smem").scope("cta").proxy("generic").tile("f32", {128, 128})
    .layout(pl::affine{{128, 128}, {128, 1}, 32})
    .protocol(pl::mbarrier{
        pl::side{"pipelink_gemm_ready", {"mma", 256}, 0},
        pl::side{"pipelink_gemm_drained", {"epilogue", 32}, 0}, {"load", 0, "dispatch"}})
    .ring(pl::fixed(1), 128 * 128 * 4)
    .bind("base", "pipelink_gemm_output").bind("full", "pipelink_gemm_ready")
    .bind("empty", "pipelink_gemm_drained"));

PL_HOST_GROUP("comm.owner", "comm.owner.rs", "comm_rs", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "tp");
PL_HOST_ABI_TYPE(100, CommRsResidue, slots, counters, output, released, generation, done, rank, world, M, N);
PL_HOST_ABI_TYPE(101, plk_role_view_comm_epilogue_rs_epilogue, residue, scalars);
PL_HOST_RESIDUE(2, "comm.owner.rs.residue", CommRsResidue, 100, true,
                pl::host::roles({"comm.epilogue.rs:epilogue"}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"comm.epilogue.rs:epilogue"}))
    .field(pl::host::scalar(1, "gemm.M", "i64").caller("M"))
    .field(pl::host::scalar(2, "gemm.N", "i64").caller("N")));
PL_HOST_CHANNEL(1, float, pl::host::channel("comm.rs.slots", "f32").rank(1).read_write().caller("slots").peer("tp"));
PL_HOST_CHANNEL(2, uint32_t, pl::host::channel("comm.rs.counters", "u32").rank(1).read_write()
    .caller("counters").peer("tp"));
PL_HOST_CHANNEL(3, float, pl::host::channel("comm.rs.output", "f32").rank(2).write().caller("output"));
PL_HOST_CHANNEL(4, uint32_t, pl::host::channel("comm.rs.released", "u32").rank(1).read_write()
    .caller("released").peer("tp"));
PL_HOST_CHANNEL(5, uint64_t, pl::host::channel("comm.rs.gen", "u64").rank(1).read_write().caller("gen"));
PL_HOST_WORKSPACE(1, 4, pl::host::workspace("comm.rs.done").zero().reset_launch());
PL_HOST_SCHEDULER(pl::host::scheduler("gemm.owner"));
PL_HOST_TARGET();
