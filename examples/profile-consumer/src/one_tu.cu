#include "consumer.cuh"

extern "C" __device__ int pipelink_profile_consumer_scheduler_smem_base() { return 0; }
extern "C" __device__ int pipelink_profile_consumer_page_stage1() { return 1; }
extern "C" __device__ int pipelink_profile_consumer_page_stage2() { return 1; }
extern "C" __device__ int pipelink_profile_consumer_page_barrier() { return 1; }
extern "C" __device__ int pipelink_profile_consumer_counter_slot() {
  return static_cast<int>(pipelink_profile_consumer::kCounterSlot);
}

#include "interpreter.cu"
#include "instruction1.cu"
#include "instruction2.cu"

namespace pipelink_profile_consumer {

struct alignas(64) OneTuParams {
  alignas(64) InterpreterParams interpreter;
  alignas(64) InstructionParams instruction1;
  alignas(64) InstructionParams instruction2;
};

static_assert(offsetof(OneTuParams, interpreter) == 0);
static_assert(offsetof(OneTuParams, instruction1) == 64);
static_assert(offsetof(OneTuParams, instruction2) == 128);
static_assert(sizeof(OneTuParams) == 192);

}  // namespace pipelink_profile_consumer

extern "C" __global__ __launch_bounds__(96, 1)
void org_pipelink_profile_consumer_one_tu_kernel(
    __grid_constant__ pipelink_profile_consumer::OneTuParams const params) {
  unsigned const warp = threadIdx.x / 32u;
  if (warp == 0) {
    plk_role_view_org_pipelink_profile_consumer_0_interpreter_sched const view{
        &params.interpreter, nullptr};
    org_pipelink_profile_consumer_0_interpreter_role_sched(&view, 0);
    return;
  }
  if (warp == 1) {
    org_pipelink_profile_consumer_1_instruction_role_instr1(&params.instruction1, 0);
    return;
  }
  if (warp == 2) {
    org_pipelink_profile_consumer_2_instruction_role_instr2(&params.instruction2, 0);
  }
}
