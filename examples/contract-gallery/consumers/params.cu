// Mistake: inlines a 32,640-byte lookup table instead of passing a device pointer.
#include <pipelink/stage.hpp>
#include "../../ring-pipeline/params.h"

extern "C" __device__ void ring_consumer(const void* raw, unsigned) {
  const auto& params = *static_cast<const ConsumerParams*>(raw);
  extern __shared__ __align__(1024) unsigned char ring_smem[];
  const int thread = int(threadIdx.x) - 32 * pipelink_ring_warp_consumer();
  const int depth = pipelink_ring_depth();
  for (int tile = 0; tile < params.columns / TileCols; ++tile) {
    const int slot = tile % depth;
    const unsigned full = __cvta_generic_to_shared(
        ring_smem + 16 * (pipelink_ring_full() + slot));
    ring_wait(full, (tile / depth) & 1);
    const auto* data = reinterpret_cast<const __nv_bfloat16*>(
        ring_smem + pipelink_ring_base() + slot * TileBytes);
    for (int index = thread; index < TileElements; index += 32 * ConsumerWarps) {
      // SW128 permutes element bits [3:5] with bits [6:8].
      const int physical = index ^ ((index & (7 << 6)) >> 3);
      const float value = __bfloat162float(data[physical]);
      const int row = int(blockIdx.x) * TileRows + index / TileCols;
      const int column = tile * TileCols + index % TileCols;
      params.output[row * params.columns + column] = fmaxf(value * params.scale + params.bias, 0.0f);
    }
    // Every consumer thread releases once; the producer cannot overwrite early.
    const unsigned empty = __cvta_generic_to_shared(
        ring_smem + 16 * (pipelink_ring_empty() + slot));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(empty) : "memory");
  }
}

PL_STAGE("org.ring.consumer", "1.0.0");
struct InlineLookupParams { ConsumerParams consumer; float lookup[8160]; };
PL_PARAMS(InlineLookupParams, consumer, lookup);
PL_ROLE(consumer, ring_consumer, ConsumerWarps);
PL_CLAIM_EX(pl::claim("pipelink_ring_base", "smem")
    .extent(0, {{"pipelink_ring_depth", TileBytes}}).align(1024));
PL_CLAIM_EX(pl::claim("pipelink_ring_full", "mbarrier")
    .extent(0, {{"pipelink_ring_depth", 1}}));
PL_CLAIM_EX(pl::claim("pipelink_ring_empty", "mbarrier")
    .extent(0, {{"pipelink_ring_depth", 1}}));
PL_CLAIM_EX(pl::claim("pipelink_ring_warp_consumer", "warp")
    .extent(ConsumerWarps).at(1).roles({"consumer"}));
PL_ENDPOINT(tile_ring, pl::endpoint("tile_ring", pl::consume, "consumer")
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
