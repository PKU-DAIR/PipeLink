#include "consumer.cuh"
#include <pipelink/stage.hpp>

extern "C" __device__ void org_pipelink_profile_consumer_2_instruction_role_instr2(
    void const* opaque, unsigned variant) {
  using namespace pipelink_profile_consumer;

  auto const* params = static_cast<InstructionParams const*>(opaque);
  SchedulerShared* state = scheduler_shared();
  unsigned const lane = threadIdx.x & 31u;

  scheduler_barrier();

  if (lane == 0 &&
      (variant != 0 || params->expected_opcode != state->descriptor.stage2_opcode ||
       params->expected_opcode != kStage2Opcode)) {
    atomicAdd(state->errors, 1u);
  }

  page_barrier();
  if (blockIdx.x != 1) {
    return;
  }

  counter_acquire(state->counters);
  auto const* page = reinterpret_cast<unsigned const*>(
      dynamic_smem() + state->page_offset_bytes);
  for (unsigned index = lane; index < state->descriptor.count; index += 32) {
    if (page[index] != state->intermediate[index]) {
      atomicAdd(state->errors, 1u);
    }
    state->output[index] = stage2_value(page[index], state->descriptor);
  }
}

namespace pipelink_profile_consumer {
PL_STAGE("org.pipelink.profile-consumer.2-instruction", "1.0.0");
PL_PARAMS(pipelink_profile_consumer::InstructionParams, expected_opcode, reserved);
PL_ROLE(instr2, org_pipelink_profile_consumer_2_instruction_role_instr2, kThreads / 3 / 32);
PL_DESCRIPTOR(InstructionDescriptor);
PL_ENDPOINT(page_handoff, pl::endpoint("page_handoff", pl::consume, "instr2")
    .space("smem").scope("cta").proxy("generic").tile("u32", {kElements})
    .layout(pl::affine{{{kElements}}, {{1}}, sizeof(unsigned) * 8})
    .protocol(pl::named_barrier{"pipelink_profile_consumer_page_barrier", kThreads / 3 * 2})
    .bind("base", "pipelink_profile_consumer_page_stage2"));
PL_ENDPOINT(dependency_counter, pl::endpoint("dependency_counter", pl::consume, "instr2")
    .space("gmem").scope("grid").proxy("generic").tile("u32", {1})
    .layout(pl::affine{{{1}}, {{1}}, sizeof(unsigned) * 8})
    .protocol(pl::counter{"pipelink_profile_consumer_counter_slot"}));
PL_CLAIM("pipelink_profile_consumer_page_stage2", "smem_page",
         (kElements * sizeof(unsigned) + kPageBytes - 1) / kPageBytes, 1,
         "phase:instr_org.pipelink.profile-consumer.2-instruction", "instr2");
PL_CLAIM("pipelink_profile_consumer_warp_instr2", "warp", kThreads / 3 / 32, 1, "kernel", "instr2");
PL_TOPOLOGY(pl::topology{1, pl::cluster{pl::dynamic, pl::launch{1, {kBlocks, 1, 1}}},
                         true, "static", {}});
}  // namespace pipelink_profile_consumer
