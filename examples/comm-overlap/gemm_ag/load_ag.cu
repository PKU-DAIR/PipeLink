// GEMM owner, AllGather load variant: hopper-gemm-split's load.cu plus
// (1) one system-scope readiness wait for the CTA's A row-block before its first TMA,
//     skipped when the row-block is this rank's own shard, then the generic->async
//     proxy fence (the payload was written by peers' generic stores);
// (2) after the last issue, a non-consuming wait on the full phase of every filled
//     slot, then the CTA-local `gather_reads_done` signal to the comm stage (CTA read-drain).
#include "channels.cuh"
#include "host_annotations.cuh"
#include <cute/arch/copy_sm90_tma.hpp>

extern "C" __device__ int pipelink_comm_ready_slot();
extern "C" __device__ int pipelink_comm_gather_reads_done();

extern "C" __device__ void gemm_load_ag_role(const void* raw_params, unsigned) {
  using namespace hopper_gemm;
  auto const& view = *static_cast<const plk_role_view_gemm_load_ag_load*>(raw_params);
  auto const& residue = *static_cast<const GemmAgResidue*>(view.residue);
  auto const& params = residue.gemm.load;
  auto const& ag = residue.ag;
  extern __shared__ __align__(1024) unsigned char gemm_smem[];
  if ((threadIdx.x & 31) != 0) return;
  // Device-resident launch generation: launches completed by this rank; peers' pushes for this launch reach (g+1)*expected.
  const uint32_t generation = *reinterpret_cast<const volatile uint32_t*>(ag.gen);
  const int row_block = int(blockIdx.x);
  if (row_block * TileM / ag.rows_per_rank != ag.rank) {
    const uint32_t* counter = ag.ready + pipelink_comm_ready_slot() + row_block;
    const uint32_t target = (generation + 1) * uint32_t(ag.expected);
    uint32_t value;
    do {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(counter) : "memory");
    } while (int32_t(value - target) < 0);
    asm volatile("fence.proxy.async.global;" ::: "memory");
  }
  const int depth = pipelink_gemm_depth();
  const int k_tiles = params.K / TileK;
  for (int k = 0; k < k_tiles; ++k) {
    const int slot = k % depth;
    auto* full = barrier(gemm_smem, pipelink_gemm_full(), slot);
    auto* empty = barrier(gemm_smem, pipelink_gemm_empty(), slot);
    cutlass::arch::ClusterBarrier::wait(empty, ((k / depth) & 1) ^ 1);
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(full, SlotBytes);
    auto* operands = gemm_smem + pipelink_gemm_operands() + slot * SlotBytes;
    cute::SM90_TMA_LOAD_2D::copy(&params.a, full, uint64_t(0), operands,
                               k * TileK, int(blockIdx.x) * TileM);
    cute::SM90_TMA_LOAD_2D::copy(&params.b, full, uint64_t(0), operands + OperandBytes,
                               k * TileK, int(blockIdx.y) * TileN);
  }
  // Each slot's last fill is the newest phase it can complete; earlier fills already
  // completed before that fill's empty wait, so the parity test cannot alias.
  for (int slot = 0; slot < depth && slot < k_tiles; ++slot) {
    const int last = slot + (k_tiles - 1 - slot) / depth * depth;
    cutlass::arch::ClusterBarrier::wait(barrier(gemm_smem, pipelink_gemm_full(), slot), (last / depth) & 1);
  }
  cutlass::arch::ClusterBarrier::arrive(barrier(gemm_smem, pipelink_comm_gather_reads_done()));
}

PL_STAGE("gemm.load.ag", "1.0.0");
PL_NO_PARAMS();
PL_ROLE(load, gemm_load_ag_role, 1);
PL_CLAIM_EX(pl::claim("pipelink_gemm_warp_load", "warp")
    .extent(1).align(1).at(8).roles({"load"}));
GEMM_OPERAND_CLAIMS();
GEMM_OPERANDS(pl::produce, "load");
// The comm owner's counter vocabulary: one slot per gathered 128-row block.
PL_CLAIM("pipelink_comm_ready_slot", "gmem_counter", 512, 1, "kernel");
PL_CLAIM("pipelink_comm_gather_reads_done", "mbarrier", 1, 1, "kernel");
PL_ENDPOINT(chunk_ready, pl::endpoint("chunk_ready", pl::consume, "load")
    .space("gmem").scope("system").proxy("generic").tile("u32", {1})
    .layout(pl::affine{{1}, {1}, 32})
    .protocol(pl::counter_sys{"pipelink_comm_ready_slot"}));
PL_ENDPOINT(gather_reads_done, pl::signal_endpoint("gather_reads_done", pl::produce, "load")
    .scope("cta")
    .protocol(pl::mbarrier{pl::side{"pipelink_comm_gather_reads_done", {"load", 1}, 0},
                           std::nullopt, {"load", 0, "dispatch"}, "never"})
    .bind("full", "pipelink_comm_gather_reads_done"));
