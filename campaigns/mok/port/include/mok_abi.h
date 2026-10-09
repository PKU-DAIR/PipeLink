/* MoK port: framework-published call ABI shared by every owner (C, host and device).
 * No owner template bodies, no geometry of another owner: tile geometry crosses owners
 * only as host facts. */
#ifndef MOK_ABI_H
#define MOK_ABI_H

#include <stdint.h>

/* Counters workspace lent by the framework to every op call. Op code forms its counter views as
 * base + pipelink_mok_cnt_<channel>() for the counter channels it declares. The guard is the
 * one PL_TYPE's generated prelude uses, so a stub-first one-TU control does not redeclare it.
 * Every TU whose callables take `const MokCounters*` registers it with `MOK_COUNTERS_PL_TYPE;`. */
#ifndef MOKCOUNTERS_DEFINED
#define MOKCOUNTERS_DEFINED
typedef struct MokCounters {
  int32_t* base;
} MokCounters;
#endif
#define MOK_COUNTERS_PL_TYPE PL_TYPE(MokCounters, int32_t* base;)

/* Conservative power-of-two envelope; upstream family derivation: P2 contract §5.
 * Capacity is independent of local tokens/top-k: EP4 functional tests use factor 6. */
#define MOK_ENVELOPE_LOCAL_TOKENS 16384
#define MOK_ENVELOPE_CAPACITY 524288
#define MOK_ENVELOPE_INTERMEDIATE 4096
#define MOK_ENVELOPE_MIN_MINIBATCH 256
#define MOK_ENVELOPE_MIN_TILE 128

/* Power-of-two counter extents. Round Lmax+Pmax up to 2*Pmax for combined rows. */
#define MOK_SLOTS_X_ROUTED_READY (MOK_ENVELOPE_CAPACITY / MOK_ENVELOPE_MIN_MINIBATCH)
#define MOK_SLOTS_GATE_UP_TILE_READY \
  (((2 * MOK_ENVELOPE_CAPACITY) / MOK_ENVELOPE_MIN_TILE) * \
   (MOK_ENVELOPE_INTERMEDIATE / MOK_ENVELOPE_MIN_TILE))
#define MOK_SLOTS_HIDDEN_ROW_BLOCK_READY \
  ((2 * MOK_ENVELOPE_CAPACITY) / MOK_ENVELOPE_MIN_TILE)
#define MOK_SLOTS_Y_ROUTED_READY (MOK_ENVELOPE_CAPACITY / MOK_ENVELOPE_MIN_MINIBATCH)
#define MOK_SLOTS_Y_ROUTED_DONE (MOK_ENVELOPE_CAPACITY / (MOK_ENVELOPE_MIN_TILE / 2))

/* Backward extents and pass capacities: P2 contract §5. */
#define MOK_SLOTS_D_Y_ROUTED_READY MOK_SLOTS_X_ROUTED_READY
#define MOK_SLOTS_D_HIDDEN_READY MOK_SLOTS_GATE_UP_TILE_READY
#define MOK_SLOTS_D_GATE_UP_READY MOK_SLOTS_HIDDEN_ROW_BLOCK_READY
#define MOK_SLOTS_D_X_ROUTED_READY MOK_SLOTS_X_ROUTED_READY
#define MOK_SLOTS_REPLAYED_X_ROUTED_READY MOK_SLOTS_X_ROUTED_READY
#define MOK_SLOTS_REPLAYED_GATE_UP_READY \
  ((MOK_ENVELOPE_CAPACITY / MOK_ENVELOPE_MIN_TILE) * \
   (MOK_ENVELOPE_INTERMEDIATE / MOK_ENVELOPE_MIN_TILE))
#define MOK_SLOTS_REPLAYED_HIDDEN_READY (MOK_ENVELOPE_CAPACITY / MOK_ENVELOPE_MIN_TILE)
#define MOK_SLOTS_ROUTED_BUFFERS_DONE (MOK_ENVELOPE_CAPACITY / MOK_ENVELOPE_MIN_MINIBATCH)
#define MOK_SLOTS_ROUTER_WEIGHTS_READY (MOK_ENVELOPE_CAPACITY / MOK_ENVELOPE_MIN_MINIBATCH)
#define MOK_COUNTER_CAPACITY_FWD 282624
#define MOK_COUNTER_CAPACITY_BWD 415744
#define MOK_COUNTER_CAPACITY_REC 272384

/* Shared scalar wire IDs (one merged image; every reader declares the same id/name/type/source). */
enum {
  MOK_SCALAR_NUM_LOCAL_TOKENS = 1,   /* mok.num_local_tokens  i64 caller num_local_tokens */
  MOK_SCALAR_HIDDEN = 2,             /* mok.hidden            i64 caller hidden */
  MOK_SCALAR_INTERMEDIATE = 3,       /* mok.intermediate      i64 caller intermediate */
  MOK_SCALAR_NUM_LOCAL_EXPERTS = 4,  /* mok.num_local_experts i64 caller num_local_experts */
  MOK_SCALAR_SCHEDULE_CAPACITY = 5,  /* mok.schedule_capacity i64 caller schedule_capacity */
  MOK_SCALAR_MACROBATCH_SIZE = 6,    /* mok.macrobatch_size   i64 caller macrobatch_size */
  MOK_SCALAR_MINIBATCH_SIZE = 7,     /* mok.minibatch_size    i64 caller minibatch_size */
  MOK_SCALAR_TOPK = 8,               /* mok.topk              i64 caller topk */
  MOK_SCALAR_NUM_COMM_SMS = 9,       /* mok.num_comm_sms      i64 caller num_comm_sms */
  MOK_SCALAR_SWIGLU_LIMIT = 10,      /* mok.swiglu_limit      f64 caller swiglu_limit */
  MOK_SCALAR_PASS = 11,
  MOK_SCALAR_PRECISION = 12,
  MOK_SCALAR_WORLD = 13,
  MOK_SCALAR_CLAMP = 14,
  MOK_SCALAR_DISPATCH_MB = 20,       /* mok.comm.DISPATCH_Mb        i64 fact */
  MOK_SCALAR_DISPATCH_NB = 21,       /* mok.comm.DISPATCH_Nb        i64 fact */
  MOK_SCALAR_COMBINE_MB = 22,        /* mok.comm.COMBINE_Mb         i64 fact */
  MOK_SCALAR_COMBINE_NB = 23,        /* mok.comm.COMBINE_Nb         i64 fact */
  MOK_SCALAR_COMBINE_PIPE_DEPTH = 24,/* mok.comm.COMBINE_PIPE_DEPTH i64 fact */
  MOK_SCALAR_MLP_MB = 25,            /* mok.gemm.MLP_Mb             i64 fact */
  MOK_SCALAR_MLP_NB = 26,            /* mok.gemm.MLP_Nb             i64 fact */
  MOK_SCALAR_SWIGLU_MB = 27,         /* mok.act.SWIGLU_Mb           i64 fact */
  MOK_SCALAR_SWIGLU_NB = 28,         /* mok.act.SWIGLU_Nb           i64 fact */
  MOK_SCALAR_SWIGLU_FWD_PIPE_DEPTH = 29, /* mok.act.SWIGLU_FWD_PIPE_DEPTH i64 fact */
  MOK_SCALAR_SWIGLU_BWD_PIPE_DEPTH = 30, /* mok.act.SWIGLU_BWD_PIPE_DEPTH i64 fact */
  MOK_SCALAR_EXPERT_PADDING = 31,    /* mok.framework.EXPERT_PADDING i64 fact */
  MOK_SCALAR_SLOT_X_ROUTED_READY = 40,         /* mok.slot.x_routed_ready         i64 binding */
  MOK_SCALAR_SLOT_GATE_UP_TILE_READY = 41,     /* mok.slot.gate_up_tile_ready     i64 binding */
  MOK_SCALAR_SLOT_HIDDEN_ROW_BLOCK_READY = 42, /* mok.slot.hidden_row_block_ready i64 binding */
  MOK_SCALAR_SLOT_Y_ROUTED_READY = 43,         /* mok.slot.y_routed_ready         i64 binding */
  MOK_SCALAR_SLOT_Y_ROUTED_DONE = 44,          /* mok.slot.y_routed_done          i64 binding */
  MOK_SCALAR_SLOT_D_Y_ROUTED_READY = 45,
  MOK_SCALAR_SLOT_D_HIDDEN_READY = 46,
  MOK_SCALAR_SLOT_D_GATE_UP_READY = 47,
  MOK_SCALAR_SLOT_D_X_ROUTED_READY = 48,
  MOK_SCALAR_SLOT_REPLAYED_X_ROUTED_READY = 49,
  MOK_SCALAR_SLOT_REPLAYED_GATE_UP_READY = 50,
  MOK_SCALAR_SLOT_REPLAYED_HIDDEN_READY = 51,
  MOK_SCALAR_SLOT_ROUTED_BUFFERS_DONE = 52,
  MOK_SCALAR_SLOT_ROUTER_WEIGHTS_READY = 53
};

/* Variant handle values (pipelink_mok_precision / _clamp / _world). */
#define MOK_PRECISION_BF16 0
#define MOK_PRECISION_MXFP8 1
#define MOK_PASS_FWD 0
#define MOK_PASS_BWD 1
#define MOK_PASS_REC 2

#endif
