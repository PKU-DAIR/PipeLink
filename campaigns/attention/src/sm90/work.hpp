#pragma once

// Work-identity transport for the attention family.
// The load role is the only scheduler; every other role consumes these
// records. Transport only: no arithmetic on the records in generated code.

#include <cstdint>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"

struct pipelink_attention_final_stats {
  float m[128];  // raw-score row max as exported by the core
  float l[128];  // row sum after the final normalisation-input adjustments (sink included in final mode)
};
static_assert(sizeof(pipelink_attention_final_stats) == 1024);

namespace pipelink::attention {

using FinalStats = pipelink_attention_final_stats;

struct WorkFlags {
  static constexpr uint32_t valid = 1u << 0;         // 0 = terminate
  static constexpr uint32_t apply_sink = 1u << 1;    // final-output mode
  static constexpr uint32_t empty_output = 1u << 2;  // kv_steps == 0; epilogue only
  static constexpr uint32_t gathered_q = 1u << 3;    // q_rows channel active
};

struct KvFlags {
  static constexpr uint32_t fully_valid = 1u << 0;
  static constexpr uint32_t needs_causal = 1u << 1;
  static constexpr uint32_t needs_mask_mod = 1u << 2;
};

// Single-producer ring in CTA shared memory with one full/empty ClusterBarrier
// pair per slot. Producer: the producing threads write the slot with ordinary
// stores and arrive on `full` (count = producer_arrivals, zero transaction
// bytes). Consumers: every consumer thread waits on `full`, copies the slot,
// then arrives on `empty` (count = consumer thread count). mbarrier
// arrive/wait carry release/acquire semantics at CTA scope, so generic-proxy
// stores need no extra fence; a cooperative single-lane producer must
// `__syncwarp()` before the elected arrive.
template <class T, int Depth>
struct SmemRing {
  static constexpr int depth = Depth;
  static constexpr int slot_bytes = int(sizeof(T));
  static constexpr int slot_align = int(alignof(T));
  static constexpr int storage_bytes = Depth * slot_bytes;
  static constexpr int barrier_slots = Depth;  // per side, 16-byte mbarrier slots

  struct State {
    int index = 0;
    uint32_t phase = 0;
    CUTLASS_DEVICE void advance() {
      if (++index == Depth) {
        index = 0;
        phase ^= 1;
      }
    }
  };
  CUTLASS_DEVICE static State producer_start() { return State{0, 1}; }
  CUTLASS_DEVICE static State consumer_start() { return State{0, 0}; }

  T* slots = nullptr;
  cutlass::arch::ClusterBarrier* full = nullptr;
  cutlass::arch::ClusterBarrier* empty = nullptr;

  SmemRing() = default;
  // `storage` is the kernel's dynamic shared memory base; the three handles are
  // the linker-bound claim values (smem in bytes, mbarrier in 16-byte slots).
  CUTLASS_DEVICE SmemRing(char* storage, int smem_bytes, int full_slot, int empty_slot)
      : slots(reinterpret_cast<T*>(storage + smem_bytes)),
        full(reinterpret_cast<cutlass::arch::ClusterBarrier*>(storage + 16 * full_slot)),
        empty(reinterpret_cast<cutlass::arch::ClusterBarrier*>(storage + 16 * empty_slot)) {}

  // Called once by the initializing (load) warp's elected lane before the
  // CTA-wide initialization join. `producer_arrivals` is 1 for single-lane
  // producers and the role's thread count when every producer thread writes
  // its own rows and arrives (final_stats).
  CUTLASS_DEVICE void init(uint32_t producer_arrivals, uint32_t consumer_arrivals) const {
    for (int i = 0; i < Depth; ++i) {
      full[i].init(producer_arrivals);
      empty[i].init(consumer_arrivals);
    }
  }

  CUTLASS_DEVICE T& producer_acquire(State const& s) const {
    empty[s.index].wait(s.phase);
    return slots[s.index];
  }
  CUTLASS_DEVICE void producer_commit(State const& s) const { full[s.index].arrive(); }

  CUTLASS_DEVICE T const& consumer_wait(State const& s) const {
    full[s.index].wait(s.phase);
    return slots[s.index];
  }
  CUTLASS_DEVICE void consumer_release(State const& s) const { empty[s.index].arrive(); }
};

}  // namespace pipelink::attention

// SM90 owns its score-transform descriptor.
// PL_TYPE restates this flat global type in generated callee bindings.
#ifndef PIPELINK_ATTENTION_SM90_WORK_DESC_DEFINED
#define PIPELINK_ATTENTION_SM90_WORK_DESC_DEFINED
struct pipelink_attention_sm90_work_desc {
  int32_t q_block;
  int32_t batch;
  int32_t kv_head;            // logical Q head for the unpacked SM90 scheduler
  int32_t kv_steps;
  int32_t seqlen_q;
  int32_t seqlen_k;
  int32_t q_causal_offset;
  int32_t split_idx;
  uint32_t flags;
  uint32_t qhead_per_kvhead;
  int32_t mask_param;
  float score_param;          // softcap / scale, in raw-score units; identity: 0
  float score_param_inv;      // scale / softcap, independently host-computed; identity: 0
};
#endif
static_assert(sizeof(pipelink_attention_sm90_work_desc) == 52);
static_assert(alignof(pipelink_attention_sm90_work_desc) == 4);

namespace pipelink::attention::sm90 {
using WorkDesc = pipelink_attention_sm90_work_desc;

struct WorkFlags : pipelink::attention::WorkFlags {
  static constexpr uint32_t needs_score_mod = 1u << 4;
};
} // namespace pipelink::attention::sm90
