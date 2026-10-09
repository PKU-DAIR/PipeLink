#include "consumer.cuh"
#include <pipelink/stage.hpp>

extern "C" __device__ void org_pipelink_profile_consumer_1_instruction_role_instr1(
    void const* opaque, unsigned variant) {
  using namespace pipelink_profile_consumer;

  auto const* params = static_cast<InstructionParams const*>(opaque);
  SchedulerShared* state = scheduler_shared();
  unsigned const lane = threadIdx.x & 31u;

  scheduler_barrier();

  if (lane == 0 &&
      (variant != 0 || params->expected_opcode != state->descriptor.stage1_opcode ||
       params->expected_opcode != kStage1Opcode)) {
    atomicAdd(state->errors, 1u);
  }

  auto* page = reinterpret_cast<unsigned*>(dynamic_smem() + state->page_offset_bytes);
  for (unsigned index = lane; index < state->descriptor.count; index += 32) {
    unsigned const value = stage1_value(state->input[index], state->descriptor);
    page[index] = value;
    if (blockIdx.x == 0) {
      state->intermediate[index] = value;
    }
  }

  __syncwarp();
  if (blockIdx.x == 0 && lane == 0) {
    counter_release(state->counters);
  }
  page_barrier();
}

namespace pipelink_profile_consumer {
PL_STAGE("org.pipelink.profile-consumer.1-instruction", "1.0.0");
PL_PARAMS(pipelink_profile_consumer::InstructionParams, expected_opcode, reserved);
PL_ROLE(instr1, org_pipelink_profile_consumer_1_instruction_role_instr1, kThreads / 3 / 32);
PL_DESCRIPTOR(InstructionDescriptor);
PL_ENDPOINT(page_handoff, pl::endpoint("page_handoff", pl::produce, "instr1")
    .space("smem").scope("cta").proxy("generic").tile("u32", {kElements})
    .layout(pl::affine{{{kElements}}, {{1}}, sizeof(unsigned) * 8})
    .protocol(pl::named_barrier{"pipelink_profile_consumer_page_barrier", kThreads / 3 * 2})
    .bind("base", "pipelink_profile_consumer_page_stage1"));
PL_ENDPOINT(dependency_counter, pl::endpoint("dependency_counter", pl::produce, "instr1")
    .space("gmem").scope("grid").proxy("generic").tile("u32", {1})
    .layout(pl::affine{{{1}}, {{1}}, sizeof(unsigned) * 8})
    .protocol(pl::counter{"pipelink_profile_consumer_counter_slot"}));
PL_CLAIM("pipelink_profile_consumer_page_stage1", "smem_page",
         (kElements * sizeof(unsigned) + kPageBytes - 1) / kPageBytes, 1,
         "phase:instr_org.pipelink.profile-consumer.1-instruction", "instr1");
PL_CLAIM("pipelink_profile_consumer_page_barrier", "named_barrier", 1, 1, "kernel", "instr1");
PL_CLAIM("pipelink_profile_consumer_counter_slot", "gmem_counter", 1, 1, "kernel", "instr1");
PL_CLAIM("pipelink_profile_consumer_warp_instr1", "warp", kThreads / 3 / 32, 1, "kernel", "instr1");
PL_TOPOLOGY(pl::topology{1, pl::cluster{pl::dynamic, pl::launch{1, {kBlocks, 1, 1}}},
                         true, "static", {}});
}  // namespace pipelink_profile_consumer
