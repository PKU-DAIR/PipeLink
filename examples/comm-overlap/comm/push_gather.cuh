#pragma once
// Comm owner's shared device code: push copies, the rank-local ticket queue,
// system-scope counters, the device-resident launch generation, the
// cross-launch `released` flags and the CTA read-drain's last-arriver ticket.
// Stage TUs adapt it to one CTA contract each; nothing here names a GEMM or
// attention fact. Every function is executed by one whole warp unless noted.
#include <stdint.h>

namespace comm_overlap {

inline constexpr int MaxWorld = 8;

__device__ __forceinline__ unsigned lane_id() { return threadIdx.x & 31u; }

// u32 counters and generations wrap; compare by signed distance.
__device__ __forceinline__ bool reached(uint32_t value, uint32_t target) {
  return int32_t(value - target) >= 0;
}

__device__ __forceinline__ uint32_t load_acquire_sys(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void fence_acq_rel_sys() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}

// One lane spins; the warp then rendezvouses. The other lanes' later accesses are
// ordered after the acquire by the warp barrier and their own fence.
__device__ __forceinline__ void warp_wait_reached_sys(const uint32_t* address, uint32_t target) {
  if (lane_id() == 0)
    while (!reached(load_acquire_sys(address), target)) __nanosleep(32);
  __syncwarp();
  fence_acq_rel_sys();
}

// Plain read of a generation written by a previous launch (kernel boundary orders it).
// Generations are u32 or u64; u64 keeps per-parity epochs exact across the u32 wrap.
template <typename Generation>
__device__ __forceinline__ Generation latch(const Generation* address) {
  return *reinterpret_cast<const volatile Generation*>(address);
}

// Rank-local work queue ticket, broadcast to the warp.
__device__ __forceinline__ uint32_t warp_ticket(uint32_t* queue) {
  uint32_t ticket = 0;
  if (lane_id() == 0) ticket = atomicAdd(queue, 1u);
  return __shfl_sync(0xffffffffu, ticket, 0);
}

// Copy `rows` rows of `row_vectors` 16-byte vectors between strided buffers with
// the whole warp. Unroll loads ahead of the (possibly peer) stores.
template <int Unroll = 8>
__device__ __forceinline__ void warp_copy(unsigned char* destination, const unsigned char* source,
                                          int rows, int row_vectors, size_t stride_bytes) {
  const int total = rows * row_vectors;
  for (int base = 0; base < total; base += 32 * Unroll) {
    uint4 values[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      const int i = base + u * 32 + int(lane_id());
      if (i < total)
        values[u] = *reinterpret_cast<const uint4*>(
            source + size_t(i / row_vectors) * stride_bytes + size_t(i % row_vectors) * 16);
    }
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      const int i = base + u * 32 + int(lane_id());
      if (i < total)
        *reinterpret_cast<uint4*>(
            destination + size_t(i / row_vectors) * stride_bytes + size_t(i % row_vectors) * 16) = values[u];
    }
  }
}

// Contiguous fast path: `bytes` is a multiple of 32 * 16 * Unroll. No division, no
// bounds branches; each lane loads all Unroll vectors before storing any.
template <int Unroll = 4>
__device__ __forceinline__ void warp_copy_contiguous(unsigned char* destination, const unsigned char* source,
                                                     size_t bytes) {
  auto* out = reinterpret_cast<uint4*>(destination) + lane_id();
  auto const* in = reinterpret_cast<const uint4*>(source) + lane_id();
  const size_t steps = bytes / (32 * 16 * Unroll);
  for (size_t step = 0; step < steps; ++step, in += 32 * Unroll, out += 32 * Unroll) {
    uint4 values[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; ++u) values[u] = in[u * 32];
#pragma unroll
    for (int u = 0; u < Unroll; ++u) out[u * 32] = values[u];
  }
}

// Push publication: every lane's peer stores, then each lane's .sys fence, then
// the warp rendezvous; only afterwards one lane releases the peer counter.
__device__ __forceinline__ void warp_publish_sys(uint32_t* counter) {
  fence_acq_rel_sys();
  __syncwarp();
  if (lane_id() == 0)
    asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"(counter) : "memory");
  __syncwarp();
}

// Last-arriver reduction arrival: fence this warp's stores, then one acq_rel RMW. Returns the old value.
__device__ __forceinline__ uint32_t warp_arrive_sys(uint32_t* counter) {
  fence_acq_rel_sys();
  __syncwarp();
  uint32_t old = 0;
  if (lane_id() == 0)
    asm volatile("atom.acq_rel.sys.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(counter) : "memory");
  old = __shfl_sync(0xffffffffu, old, 0);
  // A shuffle carries no memory ordering: hand lane 0's acquire to the warp with the
  // warp barrier, then each lane's fence (as in warp_wait_reached_sys).
  __syncwarp();
  fence_acq_rel_sys();
  return old;
}

// Cross-launch release wait: wait until peer `rank` has published `released >= target` into our copy.
__device__ __forceinline__ void warp_wait_released(const uint32_t* released_local, int rank, uint32_t target) {
  warp_wait_reached_sys(released_local + rank, target);
}

// CTA read-drain: non-waiting CTA-done ticket. The caller has already completed (and
// .sys-fenced) every access this CTA's comm protocol covers. The CTA drawing the
// last ticket bumps the local generation and publishes released[rank] = next
// (narrowed to u32) into every rank's copy (including its own).
template <typename Generation>
__device__ __forceinline__ void warp_done(uint32_t* done, Generation* generation, Generation latched,
                                          const uint64_t* released_peers, int rank, int world) {
  const uint32_t ctas = gridDim.x * gridDim.y * gridDim.z;
  fence_acq_rel_sys();
  __syncwarp();
  uint32_t ticket = 0;
  if (lane_id() == 0)
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(ticket) : "l"(done) : "memory");
  ticket = __shfl_sync(0xffffffffu, ticket, 0);
  if (ticket != ctas - 1) return;
  // The acquiring lane publishes, so every release is ordered after its acquire.
  if (lane_id() == 0) {
    const Generation next = latched + 1;
    *reinterpret_cast<volatile Generation*>(generation) = next;
    for (int peer = 0; peer < world; ++peer) {
      auto* flag = reinterpret_cast<uint32_t*>(released_peers[peer]) + rank;
      asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(flag), "r"(uint32_t(next)) : "memory");
    }
  }
  __syncwarp();
}

}  // namespace comm_overlap
