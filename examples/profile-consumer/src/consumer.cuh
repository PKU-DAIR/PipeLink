#pragma once

#include <cstddef>
#include <cstdint>

namespace pipelink_profile_consumer {

constexpr unsigned kElements = 64;
constexpr unsigned kThreads = 96;
constexpr unsigned kBlocks = 2;
constexpr unsigned kPageBytes = 256;
constexpr unsigned kCounterCapacity = 64;
constexpr unsigned kCounterSlot = 17;
constexpr unsigned kStage1Opcode = 0x13579bdfu;
constexpr unsigned kStage2Opcode = 0x2468ace0u;

struct alignas(16) InstructionDescriptor {
  unsigned count;
  unsigned multiply;
  unsigned add;
  unsigned xor_mask;
  unsigned final_add;
  unsigned page_bytes;
  unsigned stage1_opcode;
  unsigned stage2_opcode;
};

struct alignas(16) InterpreterParams {
  InstructionDescriptor const* descriptors;
  unsigned const* input;
  unsigned* intermediate;
  unsigned* output;
  unsigned* counters;
  unsigned* errors;
};

struct alignas(16) InstructionParams {
  unsigned expected_opcode;
  unsigned reserved[3];
};

struct alignas(16) SchedulerShared {
  InstructionDescriptor descriptor;
  unsigned const* input;
  unsigned* intermediate;
  unsigned* output;
  unsigned* counters;
  unsigned* errors;
  unsigned page_offset_bytes;
  unsigned status;
};

static_assert(sizeof(InstructionDescriptor) == 32);
static_assert(alignof(InstructionDescriptor) == 16);
static_assert(sizeof(InterpreterParams) == 48);
static_assert(alignof(InterpreterParams) == 16);
static_assert(sizeof(InstructionParams) == 16);
static_assert(alignof(InstructionParams) == 16);
static_assert(sizeof(SchedulerShared) <= kPageBytes);

__host__ __device__ constexpr unsigned input_value(unsigned index) {
  unsigned value = index ^ 0x7f4a7c15u;
  value ^= value >> 16;
  value *= 0x45d9f3bu;
  value ^= value >> 16;
  return value ^ 0xa5c31e27u;
}

__host__ __device__ constexpr unsigned stage1_value(
    unsigned input, InstructionDescriptor const& descriptor) {
  return input * descriptor.multiply + descriptor.add;
}

__host__ __device__ constexpr unsigned stage2_value(
    unsigned input, InstructionDescriptor const& descriptor) {
  return (input ^ descriptor.xor_mask) + descriptor.final_add;
}

__device__ __forceinline__ unsigned char* dynamic_smem() {
  extern __shared__ __align__(16) unsigned storage[];
  return reinterpret_cast<unsigned char*>(storage);
}

}  // namespace pipelink_profile_consumer

extern "C" __device__ int pipelink_profile_consumer_scheduler_smem_base();
extern "C" __device__ int pipelink_profile_consumer_page_stage1();
extern "C" __device__ int pipelink_profile_consumer_page_stage2();
extern "C" __device__ int pipelink_profile_consumer_page_barrier();
extern "C" __device__ int pipelink_profile_consumer_counter_slot();

namespace pipelink_profile_consumer {

__device__ __forceinline__ SchedulerShared* scheduler_shared() {
  return reinterpret_cast<SchedulerShared*>(
      dynamic_smem() + pipelink_profile_consumer_scheduler_smem_base());
}

__device__ __forceinline__ void scheduler_barrier() {
  // Warp roles reach different instruction sites; the aligned bar.sync form is invalid.
  asm volatile("barrier.sync 0, 96;" ::: "memory");
}

__device__ __forceinline__ void page_barrier() {
  unsigned const id = static_cast<unsigned>(pipelink_profile_consumer_page_barrier());
  asm volatile("barrier.sync %0, 64;" : : "r"(id) : "memory");
}

__device__ __forceinline__ void counter_release(unsigned* counters) {
  unsigned* slot = counters + pipelink_profile_consumer_counter_slot();
  unsigned const increment = 1;
  asm volatile(
      "red.release.gpu.global.add.u32 [%0], %1;"
      :
      : "l"(slot), "r"(increment)
      : "memory");
}

__device__ __forceinline__ void counter_acquire(unsigned* counters) {
  unsigned* slot = counters + pipelink_profile_consumer_counter_slot();
  unsigned observed;
  do {
    asm volatile(
        "ld.acquire.gpu.global.u32 %0, [%1];"
        : "=r"(observed)
        : "l"(slot)
        : "memory");
  } while (observed == 0);
}

}  // namespace pipelink_profile_consumer

using InterpreterParams = pipelink_profile_consumer::InterpreterParams;

struct plk_role_view_org_pipelink_profile_consumer_0_interpreter_sched {
  void const* residue;
  void const* scalars;
};
