#include <pipelink/stage.hpp>
#include "../params.h"

extern "C" __device__ void ring_producer(const void* raw, unsigned) {
  if ((threadIdx.x & 31) != 0) return;
  const auto& params = *static_cast<const ProducerParams*>(raw);
  extern __shared__ __align__(1024) unsigned char ring_smem[];
  const int depth = pipelink_ring_depth();
  // Dispatch initializes the full/empty barriers before either role starts.
  for (int tile = 0; tile < params.columns / TileCols; ++tile) {
    const int slot = tile % depth;
    const unsigned full = __cvta_generic_to_shared(
        ring_smem + 16 * (pipelink_ring_full() + slot));
    const unsigned empty = __cvta_generic_to_shared(
        ring_smem + 16 * (pipelink_ring_empty() + slot));
    ring_wait(empty, ((tile / depth) & 1) ^ 1);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 : : "r"(full), "r"(TileBytes) : "memory");
    const unsigned destination = __cvta_generic_to_shared(
        ring_smem + pipelink_ring_base() + slot * TileBytes);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 : : "r"(destination), "l"(&params.input), "r"(tile * TileCols),
                     "r"(int(blockIdx.x) * TileRows), "r"(full) : "memory");
  }
}

PL_STAGE("org.ring.producer", "1.0.0");
PL_PARAMS(ProducerParams, input, columns);
PL_ROLE(producer, ring_producer, 1);
PL_CLAIM_EX(pl::claim("pipelink_ring_base", "smem")
    .extent(0, {{"pipelink_ring_depth", TileBytes}}).align(1024));
PL_CLAIM_EX(pl::claim("pipelink_ring_full", "mbarrier")
    .extent(0, {{"pipelink_ring_depth", 1}}));
PL_CLAIM_EX(pl::claim("pipelink_ring_empty", "mbarrier")
    .extent(0, {{"pipelink_ring_depth", 1}}));
PL_CLAIM_EX(pl::claim("pipelink_ring_warp_producer", "warp")
    .extent(1).at(0).roles({"producer"}));
PL_ENDPOINT(tile_ring, pl::endpoint("tile_ring", pl::produce, "producer")
    .space("smem").scope("cta").proxy("async").tile("bf16", {TileRows, TileCols})
    .layout(pl::affine{{TileRows, TileCols}, {TileCols, 1}, 16}.swizzle(3, 3, 3))
    .protocol(pl::mbarrier{
        pl::side{"pipelink_ring_full", pl::arrive{"producer", 1}, TileBytes},
        pl::side{"pipelink_ring_empty", pl::arrive{"consumer", "threads(consumer)"}, 0},
        pl::init{"producer", 0, "dispatch"}})
    .ring(pl::handle("pipelink_ring_depth"), TileBytes)
    .bind("base", "pipelink_ring_base")
    .bind("full", "pipelink_ring_full")
    .bind("empty", "pipelink_ring_empty"));
PL_TOPOLOGY(pl::topology{1, pl::cluster{{1, 1, 1}, pl::launch{1, {1, 1, 1}}},
    false, "static", {}});
