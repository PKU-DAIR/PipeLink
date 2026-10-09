#include "consumer.cuh"
#include <pipelink/stage.hpp>
#include <pipelink_host_abi_v2.h>


extern "C" __device__ void org_pipelink_profile_consumer_0_interpreter_role_sched(
    void const* opaque, unsigned variant) {
  using namespace pipelink_profile_consumer;

  auto const& view =
      *static_cast<plk_role_view_org_pipelink_profile_consumer_0_interpreter_sched const*>(opaque);
  auto const* params = static_cast<InterpreterParams const*>(view.residue);
  SchedulerShared* state = scheduler_shared();
  unsigned const lane = threadIdx.x & 31u;

  if (lane == 0) {
    state->descriptor = params->descriptors[0];
    state->input = params->input;
    state->intermediate = params->intermediate;
    state->output = params->output;
    state->counters = params->counters;
    state->errors = params->errors;

    unsigned const page1 = static_cast<unsigned>(pipelink_profile_consumer_page_stage1());
    unsigned const page2 = static_cast<unsigned>(pipelink_profile_consumer_page_stage2());
    state->page_offset_bytes = page1 * kPageBytes;
    state->status = 0;
    if (variant != 0 || page1 != page2 || state->descriptor.count != kElements ||
        state->descriptor.page_bytes != kPageBytes) {
      state->status = 1;
      atomicAdd(state->errors, 1u);
    }
  }

  scheduler_barrier();
}

namespace pipelink_profile_consumer {
PL_STAGE("org.pipelink.profile-consumer.0-interpreter", "1.0.0");
PL_NO_PARAMS();
PL_HOST_GROUP("org.pipelink.profile-consumer", "org.pipelink.profile-consumer.params",
              "interpreter", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_ABI_TYPE(100, InterpreterParams,
                 descriptors, input, intermediate, output, counters, errors);
PL_HOST_ABI_TYPE(101, plk_role_view_org_pipelink_profile_consumer_0_interpreter_sched,
                 residue, scalars);
PL_HOST_RESIDUE(1, "org.pipelink.profile-consumer.residue",
                InterpreterParams, 100, true,
                pl::host::roles({"org.pipelink.profile-consumer.0-interpreter:sched"}));
PL_HOST_CHANNEL(1, unsigned,
                pl::host::channel("org.pipelink.profile-consumer.descriptors", "u32").rank(1).read().caller("descriptors"));
PL_HOST_CHANNEL(2, unsigned,
                pl::host::channel("org.pipelink.profile-consumer.input", "u32").rank(1).read().caller("input"));
PL_HOST_CHANNEL(3, unsigned,
                pl::host::channel("org.pipelink.profile-consumer.intermediate", "u32").rank(1).read_write().caller("intermediate"));
PL_HOST_CHANNEL(4, unsigned,
                pl::host::channel("org.pipelink.profile-consumer.output", "u32").rank(1).write().caller("output"));
PL_HOST_CHANNEL(5, unsigned,
                pl::host::channel("org.pipelink.profile-consumer.counters", "u32").rank(1).read_write().caller("counters"));
PL_HOST_CHANNEL(6, unsigned,
                pl::host::channel("org.pipelink.profile-consumer.errors", "u32").rank(1).read_write().caller("errors"));
PL_HOST_SCHEDULER(pl::host::scheduler("org.pipelink.profile-consumer",
    "org.pipelink.profile-consumer.static-grid.v1", pl::host::schema()));
PL_HOST_TARGET();
PL_ROLE(sched, org_pipelink_profile_consumer_0_interpreter_role_sched, kThreads / 3 / 32);
PL_CLAIM_EX(pl::claim("pipelink_profile_consumer_scheduler_smem_base", "smem")
    .extent(sizeof(SchedulerShared)).align(alignof(SchedulerShared))
    .lifetime("kernel").roles({"sched"}));
PL_CLAIM("pipelink_profile_consumer_warp_sched", "warp", kThreads / 3 / 32, 1, "kernel", "sched");
PL_TOPOLOGY(pl::topology{1, pl::cluster{pl::dynamic, pl::launch{1, {kBlocks, 1, 1}}},
                         true, "static", {}});
}  // namespace pipelink_profile_consumer
