#include "abi.h"
#include <pipelink/stage.hpp>

extern "C" __device__ int pipelink_mc_barrier();

extern "C" __device__ void mc_tp_allreduce_role(const void* opaque, unsigned) {
  using namespace multimem_domains::device;
  const auto& view = *static_cast<const plk_role_view_mc_tp_allreduce_mc*>(opaque);
  const auto& p = *static_cast<const MultimemMcResidue*>(view.residue);
  const int barrier = pipelink_mc_barrier();
  const uint32_t half = p.slice_elements / 2;
  for (uint64_t slice = blockIdx.x; slice < p.slices; slice += gridDim.x) {
    const uint32_t epoch = next_epoch(p.epoch, slice);
    const uint32_t target = 2u * epoch;
    // The prior *whole kernel*, including dp's red reads, must have finished on
    // this rank's stream before the next session launches. Each tp participant
    // therefore releases its old red reads before this entry rendezvous lets
    // any rank overwrite red. mc completion alone is NOT that release.
    multicast_barrier(p.bar_mc + slice, p.bar_local + slice, target, barrier);
    alias_fence();  // caller filled input through its local unicast mapping
    const uint64_t begin = slice * p.slice_elements + uint64_t(p.tp_rank) * half;
    for (uint32_t offset = lane() * 4; offset < half; offset += 128) {
      float x, y, z, w;
      asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                   : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                   : "l"(p.input_mc + begin + offset) : "memory");
      asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(p.red_mc + begin + offset), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
    }
    system_fence();
    rendezvous(barrier);
    if (lane() == 0)
      asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;"
                   :: "l"(p.done_mc + slice) : "memory");
    if (lane() == 0) p.epoch[slice] = epoch;
  }
}

PL_STAGE("mc.tp-allreduce", "1.0.0");
PL_NO_PARAMS();
PL_ROLE(mc, mc_tp_allreduce_role, 1);
PL_CLAIM_EX(pl::claim("pipelink_mc_warp", "warp").extent(1).align(1).at(0).roles({"mc"}));
PL_CLAIM_EX(pl::claim("pipelink_mc_barrier", "named_barrier").extent(1).align(1).roles({"mc"}));
PL_TOPOLOGY(pl::topology{1, pl::cluster{{1, 1, 1}, pl::launch{1, {1, 1, 1}}}, true, "static", {}});

PL_HOST_GROUP("mc.owner", "mc.owner", "mc", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_DOMAIN(1, "tp");
PL_HOST_ABI_TYPE(100, MultimemMcResidue, input_mc, red_mc, bar_mc, bar_local,
                 done_mc, epoch, slice_elements, slices, tp_rank);
PL_HOST_ABI_TYPE(101, plk_role_view_mc_tp_allreduce_mc, residue, scalars);
PL_HOST_RESIDUE(2, "mc.residue", MultimemMcResidue, 100, true,
                pl::host::roles({"mc.tp-allreduce:mc"}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"mc.tp-allreduce:mc"}))
    .field(pl::host::scalar(1, "witness.elements", "i64").caller("elements"))
    .field(pl::host::scalar(2, "witness.slice_elements", "i64").caller("slice_elements"))
    .field(pl::host::scalar(3, "mc.observed_sm_count", "i64").hardware("observed_sm_count")));
PL_HOST_CHANNEL(1, float, pl::host::channel("mc.in", "f32").rank(1).read()
    .caller("in").multicast("tp"));
PL_HOST_CHANNEL(2, float, pl::host::channel("mc.red", "f32").rank(1).write()
    .caller("red").multicast("tp"));
PL_HOST_CHANNEL(3, uint32_t, pl::host::channel("mc.tp_bar", "u32").rank(1).read_write()
    .caller("tp_bar").multicast("tp"));
PL_HOST_CHANNEL(4, uint32_t, pl::host::channel("mc.tp_done", "u32").rank(1).read_write()
    .caller("tp_done").multicast("tp"));
PL_HOST_CHANNEL(5, uint32_t, pl::host::channel("mc.epoch", "u32").rank(1).read_write()
    .caller("mc_epoch"));
PL_HOST_SCHEDULER(pl::host::scheduler("mc.owner", "multimem.persistent-slices", pl::host::schema()));
PL_HOST_TARGET();
