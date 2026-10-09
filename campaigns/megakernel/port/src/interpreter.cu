#include "llama.cuh"
#include "../../host/mk_host_abi.h"
#include <pipelink/stage.hpp>

using mk_config = ::config;
using mk_globals = ::llama_1b_globals;
using mk_state = ::megakernel::state<mk_config>;

PL_STAGE("org.pipelink.megakernel.interpreter", "7309cec+pipelink.megakernel");
PL_NO_PARAMS();
PL_HOST_GROUP("org.pipelink.megakernel", "org.pipelink.megakernel.params",
    "interpreter", MkProviderArgs, 1, true);
PL_HOST_ABI_TYPE(1, MkProviderArgs, version, reserved0, bar, instructions, timings,
    qkv_weights, attn_norm_weights, o_weights, mlp_norm_weights, up_weights,
    gate_weights, down_weights, lm_head_norm_weights, lm_head_weights, k_cache,
    v_cache, rope_cos, rope_sin, hidden_states, q_post_rope, attn_out,
    attn_lse_intermediates, attn_out_intermediates, silu_out, logits, num_sms,
    num_instructions, max_seq_len, rope_rows, num_layers, bar_rows, vocab_size,
    pos_id, attn_scale, rms_norm_eps, skip_attn_reduction, reserved1);
// The owner encodes gl descriptors; only their opaque storage layout is public.
PL_HOST_ABI_EMPTY(2, mk_globals);
PL_HOST_RESIDUE(8, "org.pipelink.megakernel.residue", mk_globals, 2, true,
    pl::host::roles({"org.pipelink.megakernel.interpreter:consumer",
                    "org.pipelink.megakernel.interpreter:loader",
                    "org.pipelink.megakernel.interpreter:storer",
                    "org.pipelink.megakernel.interpreter:launcher",
                    "org.pipelink.megakernel.interpreter:controller"}));
PL_HOST_SCHEDULER(pl::host::scheduler("org.pipelink.megakernel",
    "org.pipelink.megakernel.sm-grid.v1", pl::host::schema()));
PL_HOST_TARGET();
namespace mk_host {
#define MK_REFLECT_ROLE_VIEW(id, T) PL_HOST_ABI_TYPE(id, T, residue, scalars);
MK_HOST_ROLE_TYPES(MK_REFLECT_ROLE_VIEW)
#undef MK_REFLECT_ROLE_VIEW
}  // namespace mk_host
PL_TOPOLOGY(pl::topology{mk_config::CLUSTER_BLOCKS,
    pl::cluster{{mk_config::CLUSTER_BLOCKS, 1, 1}}, true});

extern __shared__ __align__(1024) unsigned char
    org_pipelink_megakernel_interpreter_smem[];
PL_CLAIM("pipelink_megakernel_pages_base", "smem", sizeof(mk_state::page_array_t),
         __alignof__(org_pipelink_megakernel_interpreter_smem), "kernel");
PL_CLAIM_SMEM("pipelink_megakernel_ring_base", mk_state::instruction_state_array_t, "kernel");
PL_CLAIM("pipelink_megakernel_vmbar_base", "smem",
    sizeof(mk_state::page_semaphore_array_t) + 2 * sizeof(mk_state::instruction_semaphore_array_t) +
        2 * sizeof(::kittens::semaphore), alignof(::kittens::semaphore), "kernel");
PL_CLAIM_EX(pl::claim("pipelink_megakernel_tmem_base", "tmem")
    .extent(mk_state::tensor_allocator_t::cols).align(32).at(0));
PL_CLAIM_EX(pl::claim("pipelink_megakernel_bar_vm", "named_barrier").extent(1).at(15));
PL_HINT("named_barrier_0", "tensor_allocator ctor barrier.sync 0 (all threads, prologue only)");

#define MK_CONFIG_HINT(NAME) PL_HINT("megakernel_config." #NAME, mk_config::NAME);
MK_CONFIG_HINT(INSTRUCTION_PIPELINE_STAGES)
MK_CONFIG_HINT(INSTRUCTION_PIPELINE_STAGES_BITS)
MK_CONFIG_HINT(INSTRUCTION_WIDTH)
MK_CONFIG_HINT(TIMING_WIDTH)
MK_CONFIG_HINT(DYNAMIC_SEMAPHORES)
MK_CONFIG_HINT(NUM_CONSUMER_WARPS)
MK_CONFIG_HINT(NUM_WARPS)
MK_CONFIG_HINT(NUM_THREADS)
MK_CONFIG_HINT(NUM_PAGES)
MK_CONFIG_HINT(PAGE_SIZE)
MK_CONFIG_HINT(SCRATCH_BYTES)
MK_CONFIG_HINT(CONSUMER_REGISTERS)
MK_CONFIG_HINT(NON_CONSUMER_REGISTERS)
MK_CONFIG_HINT(DYNAMIC_SHARED_MEMORY)
MK_CONFIG_HINT(STATIC_SHARED_MEMORY)
#undef MK_CONFIG_HINT

// The uniform register budget is the same warp-weighted budget as the default
// split; it is the alternative setmaxnreg choice exercised by rebind-regs.
#if defined(PIPELINK_EMIT)
constexpr int mk_uniform_registers =
    (mk_config::NUM_CONSUMER_WARPS * mk_config::CONSUMER_REGISTERS +
     (mk_config::NUM_WARPS - mk_config::NUM_CONSUMER_WARPS) * mk_config::NON_CONSUMER_REGISTERS) /
    mk_config::NUM_WARPS;
#endif
PL_CLAIM_REGS("pipelink_regs_consumer_wg", "consumer_wg",
    {mk_config::CONSUMER_REGISTERS, mk_uniform_registers}, "consumer");
PL_CLAIM_REGS("pipelink_regs_producer_wg", "producer_wg",
    {mk_config::NON_CONSUMER_REGISTERS, mk_uniform_registers},
    "loader", "storer", "launcher", "controller");
PL_HINT("launch_registers", mk_uniform_registers);

extern "C" __device__ int pipelink_megakernel_pages_base();
extern "C" __device__ int pipelink_megakernel_ring_base();
extern "C" __device__ int pipelink_megakernel_vmbar_base();
extern "C" __device__ int pipelink_megakernel_tmem_base();
extern "C" __device__ int pipelink_megakernel_bar_vm();
extern "C" __device__ int pipelink_megakernel_opcode_slot0();
extern "C" __device__ int pipelink_megakernel_opcode_slot1();
extern "C" __device__ int pipelink_megakernel_opcode_slot2();
extern "C" __device__ int pipelink_megakernel_opcode_slot3();
extern "C" __device__ int pipelink_megakernel_opcode_slot4();
extern "C" __device__ int pipelink_megakernel_opcode_slot5();
extern "C" __device__ int pipelink_megakernel_opcode_slot6();
extern "C" __device__ int pipelink_megakernel_opcode_slot7();

#define PIPELINK_MK_DECLARE_SLOT(k)                                            \
  extern "C" __device__ int pipelink_call_megakernel_slot##k##_release_lid(          \
      const void* g, int* instruction, int query);                             \
  extern "C" __device__ int pipelink_call_megakernel_slot##k##_init_semaphores(       \
      const void* g, void* s);                                                  \
  extern "C" __device__ void pipelink_call_megakernel_slot##k##_loader(               \
      const void* g, void* s);                                                  \
  extern "C" __device__ void pipelink_call_megakernel_slot##k##_launcher(             \
      const void* g, void* s);                                                  \
  extern "C" __device__ void pipelink_call_megakernel_slot##k##_consumer(             \
      const void* g, void* s);                                                  \
  extern "C" __device__ void pipelink_call_megakernel_slot##k##_storer(               \
      const void* g, void* s);                                                  \
  PL_CALL(megakernel_slot##k##_release_lid, pipelink_call_megakernel_slot##k##_release_lid, \
          pl::produce, "controller", "release_lid");                           \
  PL_CALL(megakernel_slot##k##_init_semaphores, pipelink_call_megakernel_slot##k##_init_semaphores, \
          pl::produce, "controller", "init_semaphores");                       \
  PL_CALL(megakernel_slot##k##_loader, pipelink_call_megakernel_slot##k##_loader, \
          pl::produce, "loader", "loader");                                   \
  PL_CALL(megakernel_slot##k##_launcher, pipelink_call_megakernel_slot##k##_launcher, \
          pl::produce, "launcher", "launcher");                               \
  PL_CALL(megakernel_slot##k##_consumer, pipelink_call_megakernel_slot##k##_consumer, \
          pl::produce, "consumer", "consumer");                               \
  PL_CALL(megakernel_slot##k##_storer, pipelink_call_megakernel_slot##k##_storer, \
          pl::produce, "storer", "storer")

PIPELINK_MK_DECLARE_SLOT(0);
PIPELINK_MK_DECLARE_SLOT(1);
PIPELINK_MK_DECLARE_SLOT(2);
PIPELINK_MK_DECLARE_SLOT(3);
PIPELINK_MK_DECLARE_SLOT(4);
PIPELINK_MK_DECLARE_SLOT(5);
PIPELINK_MK_DECLARE_SLOT(6);
PIPELINK_MK_DECLARE_SLOT(7);

#undef PIPELINK_MK_DECLARE_SLOT

namespace {

__device__ __forceinline__ void vm_barrier() {
  const int barrier_id = pipelink_megakernel_bar_vm();
  // Roles reach this named CTA barrier through different control-flow paths.
  asm volatile("barrier.sync %0, 640;\n" : : "r"(barrier_id) : "memory");
}

#define PIPELINK_MK_DISPATCH_WORKER(role, g, mks)                              \
  do {                                                                          \
    const int opcode = (mks).instruction()[0];                                  \
    if (opcode == pipelink_megakernel_opcode_slot0())                                   \
      pipelink_call_megakernel_slot0_##role(&(g), &(mks));                              \
    else if (opcode == pipelink_megakernel_opcode_slot1())                              \
      pipelink_call_megakernel_slot1_##role(&(g), &(mks));                              \
    else if (opcode == pipelink_megakernel_opcode_slot2())                              \
      pipelink_call_megakernel_slot2_##role(&(g), &(mks));                              \
    else if (opcode == pipelink_megakernel_opcode_slot3())                              \
      pipelink_call_megakernel_slot3_##role(&(g), &(mks));                              \
    else if (opcode == pipelink_megakernel_opcode_slot4())                              \
      pipelink_call_megakernel_slot4_##role(&(g), &(mks));                              \
    else if (opcode == pipelink_megakernel_opcode_slot5())                              \
      pipelink_call_megakernel_slot5_##role(&(g), &(mks));                              \
    else if (opcode == pipelink_megakernel_opcode_slot6())                              \
      pipelink_call_megakernel_slot6_##role(&(g), &(mks));                              \
    else if (opcode == pipelink_megakernel_opcode_slot7())                              \
      pipelink_call_megakernel_slot7_##role(&(g), &(mks));                              \
    else                                                                         \
      asm volatile("trap;");                                                    \
  } while (false)

#define PIPELINK_MK_DEFINE_WORKER(role, start_event, is_consumer)               \
  __device__ __forceinline__ void role##_main_loop(const mk_globals& g,         \
                                                    mk_state& mks) {             \
    MK_DEBUG_PRINT_START(#role);                                                 \
    int num_iters = g.instructions.rows();                                      \
    for (mks.instruction_index = 0, mks.instruction_ring = 0;                   \
         mks.instruction_index < num_iters; mks.next_instruction()) {           \
      mks.await_instruction();                                                   \
      if (::kittens::laneid() == 0) {                                           \
        if (is_consumer)                                                         \
          mks.record(start_event + 2 * ::kittens::warpid());                    \
        else                                                                     \
          mks.record(start_event);                                               \
      }                                                                          \
      PIPELINK_MK_DISPATCH_WORKER(role, g, mks);                                \
      if (::kittens::laneid() == 0) {                                           \
        if (is_consumer)                                                         \
          mks.record(start_event + 2 * ::kittens::warpid() + 1);                \
        else                                                                     \
          mks.record(start_event + 1);                                           \
      }                                                                          \
    }                                                                            \
    __syncwarp();                                                                \
    MK_DEBUG_PRINT_END(#role);                                                   \
  }

PIPELINK_MK_DEFINE_WORKER(consumer, ::megakernel::TEVENT_CONSUMER_START, true)
PIPELINK_MK_DEFINE_WORKER(loader, ::megakernel::TEVENT_LOADER_START, false)
PIPELINK_MK_DEFINE_WORKER(storer, ::megakernel::TEVENT_STORER_START, false)
PIPELINK_MK_DEFINE_WORKER(launcher, ::megakernel::TEVENT_LAUNCHER_START, false)

#undef PIPELINK_MK_DEFINE_WORKER
#undef PIPELINK_MK_DISPATCH_WORKER

__device__ __forceinline__ int dispatch_release_lid(
    int opcode, const mk_globals& g, int* instruction, int query) {
  if (opcode == pipelink_megakernel_opcode_slot0())
    return pipelink_call_megakernel_slot0_release_lid(&g, instruction, query);
  if (opcode == pipelink_megakernel_opcode_slot1())
    return pipelink_call_megakernel_slot1_release_lid(&g, instruction, query);
  if (opcode == pipelink_megakernel_opcode_slot2())
    return pipelink_call_megakernel_slot2_release_lid(&g, instruction, query);
  if (opcode == pipelink_megakernel_opcode_slot3())
    return pipelink_call_megakernel_slot3_release_lid(&g, instruction, query);
  if (opcode == pipelink_megakernel_opcode_slot4())
    return pipelink_call_megakernel_slot4_release_lid(&g, instruction, query);
  if (opcode == pipelink_megakernel_opcode_slot5())
    return pipelink_call_megakernel_slot5_release_lid(&g, instruction, query);
  if (opcode == pipelink_megakernel_opcode_slot6())
    return pipelink_call_megakernel_slot6_release_lid(&g, instruction, query);
  if (opcode == pipelink_megakernel_opcode_slot7())
    return pipelink_call_megakernel_slot7_release_lid(&g, instruction, query);
  asm volatile("trap;");
  return 0;
}

__device__ __forceinline__ int dispatch_init_semaphores(
    int opcode, const mk_globals& g, mk_state& mks) {
  int result;
  if (opcode == pipelink_megakernel_opcode_slot0())
    result = pipelink_call_megakernel_slot0_init_semaphores(&g, &mks);
  else if (opcode == pipelink_megakernel_opcode_slot1())
    result = pipelink_call_megakernel_slot1_init_semaphores(&g, &mks);
  else if (opcode == pipelink_megakernel_opcode_slot2())
    result = pipelink_call_megakernel_slot2_init_semaphores(&g, &mks);
  else if (opcode == pipelink_megakernel_opcode_slot3())
    result = pipelink_call_megakernel_slot3_init_semaphores(&g, &mks);
  else if (opcode == pipelink_megakernel_opcode_slot4())
    result = pipelink_call_megakernel_slot4_init_semaphores(&g, &mks);
  else if (opcode == pipelink_megakernel_opcode_slot5())
    result = pipelink_call_megakernel_slot5_init_semaphores(&g, &mks);
  else if (opcode == pipelink_megakernel_opcode_slot6())
    result = pipelink_call_megakernel_slot6_init_semaphores(&g, &mks);
  else if (opcode == pipelink_megakernel_opcode_slot7())
    result = pipelink_call_megakernel_slot7_init_semaphores(&g, &mks);
  else {
    asm volatile("trap;");
    result = 0;
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  return result;
}

__device__ __forceinline__ void controller_main_loop(const mk_globals& g,
                                                      mk_state& kvms) {
  auto laneid = ::kittens::laneid();
  int num_iters = g.instructions.rows();
  int num_semaphores[mk_config::INSTRUCTION_PIPELINE_STAGES];

  static_assert(mk_config::DYNAMIC_SEMAPHORES <= 32);
  static_assert(mk_config::NUM_PAGES <= 32);

  for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
       kvms.instruction_index < num_iters;
       kvms.instruction_index++,
      kvms.instruction_ring =
          ::megakernel::ring_advance<mk_config::INSTRUCTION_PIPELINE_STAGES>(
              kvms.instruction_ring)) {
    if (kvms.instruction_index >= mk_config::INSTRUCTION_PIPELINE_STAGES) {
      auto last_slot_instruction_index =
          kvms.instruction_index - mk_config::INSTRUCTION_PIPELINE_STAGES;

      int phasebit =
          (last_slot_instruction_index /
           mk_config::INSTRUCTION_PIPELINE_STAGES) &
          1;
      ::kittens::wait(kvms.instruction_finished[kvms.instruction_ring],
                      phasebit);

      if (laneid < num_semaphores[kvms.instruction_ring]) {
        ::kittens::invalidate_semaphore(
            kvms.all_instructions[kvms.instruction_ring].semaphores[laneid]);
      }

      ::kittens::warp::sync();

      if (laneid == 0) {
        if constexpr (mk_config::TIMING_RECORD_ENABLED) {
          kvms.record(::megakernel::TEVENT_CONTROLLER_END);
          ::megakernel::controller::store_timings_and_reset<mk_config,
                                                             mk_globals>(
              &kvms.all_instructions[kvms.instruction_ring].timings[0],
              last_slot_instruction_index, g);
        }
      }
    }

    if (laneid == 0)
      kvms.record(::megakernel::TEVENT_CONTROLLER_START);

    ::megakernel::controller::load_instructions<mk_config, mk_globals>(
        &kvms.instruction()[0], kvms.instruction_index, g);

    if (laneid == 0)
      kvms.record(::megakernel::TEVENT_IFETCH_DONE);

    int last_instruction_ring =
        (kvms.instruction_ring + mk_config::INSTRUCTION_PIPELINE_STAGES - 1) %
        mk_config::INSTRUCTION_PIPELINE_STAGES;

    if (kvms.instruction_index == 0) {
      if (laneid < mk_config::NUM_PAGES)
        kvms.pid_order()[laneid] = laneid;
    } else {
      auto last_opcode =
          kvms.all_instructions[last_instruction_ring].instructions[0];

      if (laneid < mk_config::NUM_PAGES) {
        int lid = dispatch_release_lid(
            last_opcode, g,
            kvms.all_instructions[last_instruction_ring].instructions, laneid);
        kvms.pid_order()[laneid] =
            kvms.all_instructions[last_instruction_ring].pid_order[lid];
      }
    }

    if (laneid == 0)
      kvms.record(::megakernel::TEVENT_PAGE_ALLOC_DONE);

    int opcode = kvms.instruction()[0];
    if (opcode == pipelink_megakernel_opcode_slot0()) {
      num_semaphores[kvms.instruction_ring] = 0;
    } else {
      if (laneid == 0) {
        num_semaphores[kvms.instruction_ring] =
            dispatch_init_semaphores(opcode, g, kvms);
      }

      auto shfl_val = __shfl_sync(
          0xffffffff, num_semaphores[kvms.instruction_ring], 0);
      num_semaphores[kvms.instruction_ring] = shfl_val;
    }

    if (laneid == 0) {
      kvms.record(::megakernel::TEVENT_SEMS_SETUP);
      ::kittens::arrive(
          kvms.instruction_arrived[kvms.instruction_ring], 1);
    }
  }

  for (int i = 0; i < mk_config::INSTRUCTION_PIPELINE_STAGES; i++) {
    auto instruction_index =
        num_iters - mk_config::INSTRUCTION_PIPELINE_STAGES + i;
    if (instruction_index < 0)
      continue;

    auto instruction_ring =
        instruction_index % mk_config::INSTRUCTION_PIPELINE_STAGES;
    auto phasebit =
        (instruction_index / mk_config::INSTRUCTION_PIPELINE_STAGES) & 1;
    ::kittens::wait(kvms.instruction_finished[instruction_ring], phasebit);

    if (laneid < num_semaphores[instruction_ring]) {
      ::kittens::invalidate_semaphore(
          kvms.all_instructions[instruction_ring].semaphores[laneid]);
    }

    kvms.instruction_index = instruction_index;
    kvms.instruction_ring = instruction_ring;
    if (laneid == 0)
      kvms.record(::megakernel::TEVENT_CONTROLLER_END);

    ::megakernel::controller::store_timings_and_reset<mk_config, mk_globals>(
        &kvms.all_instructions[instruction_ring].timings[0], instruction_index,
        g);
  }
}

enum class InterpreterRole { consumer, loader, storer, launcher, controller };

template <InterpreterRole role>
__device__ __forceinline__ void run_role_with_prologue(const mk_globals& g) {
  uint64_t start_time = (uint64_t)clock64();

#ifdef MK_DEBUG
  if (threadIdx.x == 0)
    printf("Thread %d: Kernel launched\n", threadIdx.x);
  vm_barrier();
#endif

  unsigned char* const smem = org_pipelink_megakernel_interpreter_smem;
  auto& instruction_state =
      *reinterpret_cast<mk_state::instruction_state_array_t*>(
          smem + pipelink_megakernel_ring_base());

  ::kittens::semaphore* const vm_semaphores =
      reinterpret_cast<::kittens::semaphore*>(
          smem + pipelink_megakernel_vmbar_base());
  auto& page_finished =
      *reinterpret_cast<mk_state::page_semaphore_array_t*>(vm_semaphores);
  constexpr int page_semaphore_count =
      mk_config::NUM_PAGES * mk_config::INSTRUCTION_PIPELINE_STAGES_BITS;
  auto& instruction_arrived =
      *reinterpret_cast<mk_state::instruction_semaphore_array_t*>(
          vm_semaphores + page_semaphore_count);
  auto& instruction_finished =
      *reinterpret_cast<mk_state::instruction_semaphore_array_t*>(
          vm_semaphores + page_semaphore_count +
          mk_config::INSTRUCTION_PIPELINE_STAGES);
  ::kittens::semaphore& tensor_finished =
      vm_semaphores[page_semaphore_count +
                    2 * mk_config::INSTRUCTION_PIPELINE_STAGES];
  ::kittens::semaphore& semaphores_ready =
      vm_semaphores[page_semaphore_count +
                    2 * mk_config::INSTRUCTION_PIPELINE_STAGES + 1];

  auto& pages = *reinterpret_cast<mk_state::page_array_t*>(
      smem + pipelink_megakernel_pages_base());
  const int tmem_base = pipelink_megakernel_tmem_base();
  (void)tmem_base;
  mk_state::tensor_allocator_t tensor_alloc{};

#ifdef MK_DEBUG
  if (threadIdx.x == 0)
    printf("Thread %d: Pre-MKS creation\n", threadIdx.x);
  vm_barrier();
#endif

  mk_state mks{instruction_state,
               instruction_arrived,
               instruction_finished,
               0,
               0,
               {},
               pages,
               page_finished,
               tensor_finished,
               semaphores_ready,
               start_time,
               tensor_alloc};

#ifdef MK_DEBUG
  if (threadIdx.x == 0)
    printf("Thread %d: Created MKS\n", threadIdx.x);
  vm_barrier();
#endif

  if (threadIdx.x < mk_config::TIMING_WIDTH) {
#pragma unroll
    for (int i = 0; i < mk_config::INSTRUCTION_PIPELINE_STAGES; i++)
      instruction_state[i].timings[threadIdx.x] = 0;
  }

  if (threadIdx.x < mk_config::INSTRUCTION_PIPELINE_STAGES) {
    ::kittens::init_semaphore(instruction_arrived[threadIdx.x], 1);
    ::kittens::init_semaphore(instruction_finished[threadIdx.x],
                              mk_config::NUM_WARPS - 1);
  }
  if (threadIdx.x < mk_config::NUM_PAGES) {
    for (int i = 0; i < mk_config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
      auto count = mk_config::NUM_CONSUMER_WARPS * (1 << i);
      ::kittens::init_semaphore(page_finished[threadIdx.x][i], count);
      ::kittens::arrive(page_finished[threadIdx.x][i], count);
    }
  }
  if (threadIdx.x == 0) {
    ::kittens::init_semaphore(tensor_finished,
                              mk_config::NUM_CONSUMER_WARPS);
    ::kittens::arrive(tensor_finished, mk_config::NUM_CONSUMER_WARPS);
    ::kittens::init_semaphore(semaphores_ready, 1);
  }

  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  vm_barrier();

#ifdef MK_DEBUG
  if (blockIdx.x == 0 && threadIdx.x == 0)
    mks.print();
#endif

  if constexpr (role == InterpreterRole::consumer)
    consumer_main_loop(g, mks);
  else if constexpr (role == InterpreterRole::loader)
    loader_main_loop(g, mks);
  else if constexpr (role == InterpreterRole::storer)
    storer_main_loop(g, mks);
  else if constexpr (role == InterpreterRole::launcher)
    launcher_main_loop(g, mks);
  else
    controller_main_loop(g, mks);

#ifdef MK_DEBUG
  printf("Thread %d arriving at final barrier\n", threadIdx.x);
#endif

  __syncwarp();
  vm_barrier();

#ifdef MK_DEBUG
  uint64_t end_time = (uint64_t)clock64();
  if (threadIdx.x == 0)
    printf("Overall VM execution time: %lu\n", end_time - start_time);
#endif
}

}  // namespace

extern "C" __device__ void org_pipelink_megakernel_interpreter_role_consumer(
    const void* params, unsigned variant) {
  const auto& view = *static_cast<const mk_host::plk_role_view_org_pipelink_megakernel_interpreter_consumer*>(params);
  const auto& g = *static_cast<const llama_1b_globals*>(view.residue);
  (void)variant;
  run_role_with_prologue<InterpreterRole::consumer>(g);
}
PL_ROLE(consumer, org_pipelink_megakernel_interpreter_role_consumer, mk_config::NUM_CONSUMER_WARPS);
PL_CLAIM_EX(pl::claim("pipelink_megakernel_warp_consumer", "warp")
    .extent(mk_config::NUM_CONSUMER_WARPS).at(0).roles({"consumer"}));

extern "C" __device__ void org_pipelink_megakernel_interpreter_role_loader(
    const void* params, unsigned variant) {
  const auto& view = *static_cast<const mk_host::plk_role_view_org_pipelink_megakernel_interpreter_loader*>(params);
  const auto& g = *static_cast<const llama_1b_globals*>(view.residue);
  (void)variant;
  run_role_with_prologue<InterpreterRole::loader>(g);
}
PL_ROLE(loader, org_pipelink_megakernel_interpreter_role_loader, 1);
PL_CLAIM_EX(pl::claim("pipelink_megakernel_warp_loader", "warp")
    .extent(1).at(mk_config::NUM_CONSUMER_WARPS).roles({"loader"}));

extern "C" __device__ void org_pipelink_megakernel_interpreter_role_storer(
    const void* params, unsigned variant) {
  const auto& view = *static_cast<const mk_host::plk_role_view_org_pipelink_megakernel_interpreter_storer*>(params);
  const auto& g = *static_cast<const llama_1b_globals*>(view.residue);
  (void)variant;
  run_role_with_prologue<InterpreterRole::storer>(g);
}
PL_ROLE(storer, org_pipelink_megakernel_interpreter_role_storer, 1);
PL_CLAIM_EX(pl::claim("pipelink_megakernel_warp_storer", "warp")
    .extent(1).at(mk_config::NUM_CONSUMER_WARPS + 1).roles({"storer"}));

extern "C" __device__ void org_pipelink_megakernel_interpreter_role_launcher(
    const void* params, unsigned variant) {
  const auto& view = *static_cast<const mk_host::plk_role_view_org_pipelink_megakernel_interpreter_launcher*>(params);
  const auto& g = *static_cast<const llama_1b_globals*>(view.residue);
  (void)variant;
  run_role_with_prologue<InterpreterRole::launcher>(g);
}
PL_ROLE(launcher, org_pipelink_megakernel_interpreter_role_launcher, 1);
PL_CLAIM_EX(pl::claim("pipelink_megakernel_warp_launcher", "warp")
    .extent(1).at(mk_config::NUM_CONSUMER_WARPS + 2).roles({"launcher"}));

extern "C" __device__ void org_pipelink_megakernel_interpreter_role_controller(
    const void* params, unsigned variant) {
  const auto& view = *static_cast<const mk_host::plk_role_view_org_pipelink_megakernel_interpreter_controller*>(params);
  const auto& g = *static_cast<const llama_1b_globals*>(view.residue);
  (void)variant;
  run_role_with_prologue<InterpreterRole::controller>(g);
}
PL_ROLE(controller, org_pipelink_megakernel_interpreter_role_controller, 1);
PL_CLAIM_EX(pl::claim("pipelink_megakernel_warp_controller", "warp")
    .extent(1).at(mk_config::NUM_WARPS - 1).roles({"controller"}));
