#ifndef PIPELINK_MULTIMEM_DOMAINS_ABI_H
#define PIPELINK_MULTIMEM_DOMAINS_ABI_H

#include <stdint.h>
#include <pipelink_host_abi_v2.h>

// Caller-owned counters and per-slice epochs are shared by every session in a
// launch sequence. Initialize them once, not at session open or graph replay.
typedef struct MultimemMcResidue {
  const float* input_mc;
  float* red_mc;
  uint32_t* bar_mc;
  const uint32_t* bar_local;
  uint32_t* done_mc;
  uint32_t* epoch;
  uint32_t slice_elements;
  uint32_t slices;
  uint32_t tp_rank;
} MultimemMcResidue;

typedef struct MultimemDpResidue {
  const float* red;
  const uint32_t* tp_done;
  uint64_t bar[2];
  uint64_t done[2];
  uint64_t output[2];
  uint32_t* epoch;
  uint64_t elements;
  uint32_t slice_elements;
  uint32_t slices;
  uint32_t dp_rank;
} MultimemDpResidue;

typedef struct plk_role_view_mc_tp_allreduce_mc {
  const void* residue;
  const void* scalars;
} plk_role_view_mc_tp_allreduce_mc;

typedef struct plk_role_view_dp_gather_dp {
  const void* residue;
  const void* scalars;
} plk_role_view_dp_gather_dp;

#ifdef __cplusplus
extern "C" {
#endif
int pipelink_params_abi_mc(plk_abi_report_v2*);
int pipelink_params_query_mc(const plk_common_v2*, const void*, plk_demands_v2*);
int pipelink_params_build_mc(const plk_common_v2*, const void*, const plk_resources_v2*,
                            void*, size_t, plk_transfer_report_v2*, uint32_t, plk_session**);
int pipelink_params_release_mc(plk_session*);
int pipelink_params_abi_dp(plk_abi_report_v2*);
int pipelink_params_query_dp(const plk_common_v2*, const void*, plk_demands_v2*);
int pipelink_params_build_dp(const plk_common_v2*, const void*, const plk_resources_v2*,
                            void*, size_t, plk_transfer_report_v2*, uint32_t, plk_session**);
int pipelink_params_release_dp(plk_session*);
#ifdef __cplusplus
}

#include <cstring>
namespace multimem_domains::host {
struct Inputs {
  int64_t elements = 0;
  int64_t slice_elements = 0;
  int64_t sm_count = 0;
  uint32_t slices = 0;
};

inline bool inputs_from(const plk_shared_scalars_v2* shared, Inputs& in, bool geometry_owner) {
  if (!shared) return false;
  const auto* records = reinterpret_cast<const plk_scalar_v2*>(
      reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
  for (uint32_t i = 0; i < shared->record_count; ++i) {
    const auto& r = records[i];
    int64_t* field = r.scalar_id == 1 ? &in.elements :
                     r.scalar_id == 2 ? &in.slice_elements :
                     r.scalar_id == 3 && geometry_owner ? &in.sm_count : nullptr;
    if (!field) continue;
    if (r.scalar_type != PLK_SCALAR_I64) return false;
    std::memcpy(field, &r.value_bits, sizeof(*field));
  }
  // Full slices give each tp rank a whole number of 128-float warp vectors.
  if (in.elements <= 0 || uint64_t(in.elements) > UINT64_MAX / 8 ||
      in.slice_elements <= 0 || uint64_t(in.slice_elements) > UINT32_MAX ||
      in.slice_elements % 256 || in.elements % in.slice_elements ||
      uint64_t(in.elements / in.slice_elements) > UINT32_MAX) return false;
  if (geometry_owner && (in.sm_count <= 0 || uint64_t(in.sm_count) > UINT32_MAX)) return false;
  in.slices = uint32_t(in.elements / in.slice_elements);
  return true;
}

inline bool header_is(const plk_header_v2& header, size_t bytes) {
  return header.abi_version == PLK_HOST_ABI_V2 && header.struct_bytes == bytes;
}

inline bool pair_domain(const plk_domain_v2* domain) {
  return domain && domain->world == 2 && domain->rank < 2;
}

inline void vector_demand(plk_channel_demand_v2& demand, const plk_tensor_meta_v2& source,
                          uint64_t elements, uint32_t access, uint64_t alignment) {
  demand.dtype_id = source.dtype_id;
  demand.access = access;
  demand.minimum_allocation_bytes = elements * 4;
  demand.alignment_bytes = alignment;
  demand.extents[0] = elements;
  demand.bit_strides[0] = 32;
}
}  // namespace multimem_domains::host
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
namespace multimem_domains::device {
__device__ __forceinline__ unsigned lane() { return threadIdx.x & 31u; }

__device__ __forceinline__ void rendezvous(int barrier) {
  asm volatile("bar.sync %0, 32;" :: "r"(barrier) : "memory");
}

__device__ __forceinline__ void system_fence() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void alias_fence() {
  asm volatile("fence.proxy.alias;" ::: "memory");
}

__device__ __forceinline__ void wait_counter(const uint32_t* counter, uint32_t target) {
  uint32_t observed;
  do {
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(observed) : "l"(counter) : "memory");
    if (int32_t(observed - target) >= 0) return;
    __nanosleep(32);
  } while (true);
}

__device__ __forceinline__ uint32_t next_epoch(const uint32_t* epochs, uint64_t slice) {
  uint32_t next = 0;
  if (lane() == 0) next = *reinterpret_cast<const volatile uint32_t*>(epochs + slice) + 1u;
  return __shfl_sync(0xffffffffu, next, 0);
}

// All issuing lanes rendezvous before the elected release. The alias fence
// connects multicast counter writes to the local acquire; the second named
// barrier and per-lane fence hand the acquire to the entire payload warp.
__device__ __forceinline__ void multicast_barrier(
    uint32_t* multicast, const uint32_t* local, uint32_t target, int barrier) {
  rendezvous(barrier);
  if (lane() == 0) {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;"
                 :: "l"(multicast) : "memory");
    alias_fence();
    wait_counter(local, target);
  }
  rendezvous(barrier);
  system_fence();
}

__device__ __forceinline__ void peer_publish(const uint64_t* peers, uint64_t slice, int barrier) {
  system_fence();
  rendezvous(barrier);
  if (lane() == 0) {
#pragma unroll
    for (unsigned peer = 0; peer < 2; ++peer) {
      auto* counter = reinterpret_cast<uint32_t*>(peers[peer]) + slice;
      asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"(counter) : "memory");
    }
  }
}

__device__ __forceinline__ void peer_wait(
    const uint64_t* peers, unsigned rank, uint64_t slice, uint32_t target, int barrier) {
  if (lane() == 0) wait_counter(reinterpret_cast<const uint32_t*>(peers[rank]) + slice, target);
  rendezvous(barrier);
  system_fence();
}
}  // namespace multimem_domains::device
#endif
#endif
