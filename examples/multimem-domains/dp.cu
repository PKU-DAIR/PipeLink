#include "abi.h"
#include <pipelink/stage.hpp>

extern "C" __device__ int pipelink_dp_barrier();

extern "C" __device__ void dp_gather_role(const void* opaque, unsigned) {
  using namespace multimem_domains::device;
  const auto& view = *static_cast<const plk_role_view_dp_gather_dp*>(opaque);
  const auto& p = *static_cast<const MultimemDpResidue*>(view.residue);
  const int barrier = pipelink_dp_barrier();
  for (uint64_t slice = blockIdx.x; slice < p.slices; slice += gridDim.x) {
    const uint32_t epoch = next_epoch(p.epoch, slice);
    const uint32_t target = 2u * epoch;
    // Global wait order is tp then dp on every rank, including skewed launches.
    if (lane() == 0) {
      wait_counter(p.tp_done + slice, target);
      alias_fence();
    }
    rendezvous(barrier);
    system_fence();
    alias_fence();  // multicast red writes become unicast payload reads

    // A participant enters only after its preceding whole kernel has finished;
    // no rank may overwrite a peer's output while that launch still uses it.
    peer_publish(p.bar, slice, barrier);
    peer_wait(p.bar, p.dp_rank, slice, target, barrier);
    const uint64_t begin = slice * p.slice_elements;
    const uint64_t row = uint64_t(p.dp_rank) * p.elements;
    for (uint64_t offset = lane() * 4; offset < p.slice_elements; offset += 128) {
      const float4 value = *reinterpret_cast<const float4*>(p.red + begin + offset);
#pragma unroll
      for (unsigned peer = 0; peer < 2; ++peer) {
        auto* output = reinterpret_cast<float*>(p.output[peer]);
        *reinterpret_cast<float4*>(output + row + begin + offset) = value;
      }
    }
    // Every writing lane fences before the elected lane releases both peers.
    peer_publish(p.done, slice, barrier);
    peer_wait(p.done, p.dp_rank, slice, target, barrier);
    if (lane() == 0) p.epoch[slice] = epoch;
  }
}

PL_STAGE("dp.gather", "1.0.0");
PL_NO_PARAMS();
PL_ROLE(dp, dp_gather_role, 1);
PL_CLAIM_EX(pl::claim("pipelink_dp_warp", "warp").extent(1).align(1).at(1).roles({"dp"}));
PL_CLAIM_EX(pl::claim("pipelink_dp_barrier", "named_barrier").extent(1).align(1).roles({"dp"}));
PL_TOPOLOGY(pl::topology{1, pl::cluster{{1, 1, 1}, pl::launch{1, {1, 1, 1}}}, true, "static", {}});

PL_HOST_GROUP("dp.owner", "dp.owner", "dp", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "tp");
PL_HOST_DOMAIN(2, "dp");
PL_HOST_ABI_TYPE(100, MultimemDpResidue, red, tp_done, bar, done, output, epoch,
                 elements, slice_elements, slices, dp_rank);
PL_HOST_ABI_TYPE(101, plk_role_view_dp_gather_dp, residue, scalars);
PL_HOST_RESIDUE(2, "dp.residue", MultimemDpResidue, 100, true,
                pl::host::roles({"dp.gather:dp"}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"dp.gather:dp"}))
    .field(pl::host::scalar(1, "witness.elements", "i64").caller("elements"))
    .field(pl::host::scalar(2, "witness.slice_elements", "i64").caller("slice_elements")));
#ifdef MULTIMEM_DOMAINS_BAD_DP_RED
PL_HOST_CHANNEL(1, float, pl::host::channel("dp.red", "f32").rank(1).read()
    .caller("red").domain("dp"));
#else
PL_HOST_CHANNEL(1, float, pl::host::channel("dp.red", "f32").rank(1).read()
    .caller("red").domain("tp"));
#endif
PL_HOST_CHANNEL(2, uint32_t, pl::host::channel("dp.tp_done", "u32").rank(1).read()
    .caller("tp_done").domain("tp"));
PL_HOST_CHANNEL(3, uint32_t, pl::host::channel("dp.dp_bar", "u32").rank(1).read_write()
    .caller("dp_bar").peer("dp"));
PL_HOST_CHANNEL(4, uint32_t, pl::host::channel("dp.dp_done", "u32").rank(1).read_write()
    .caller("dp_done").peer("dp"));
PL_HOST_CHANNEL(5, uint32_t, pl::host::channel("dp.epoch", "u32").rank(1).read_write()
    .caller("dp_epoch"));
PL_HOST_CHANNEL(6, float, pl::host::channel("dp.out", "f32").rank(1).write()
    .caller("out").peer("dp"));
PL_HOST_SCHEDULER(pl::host::scheduler("mc.owner"));
PL_HOST_TARGET();
