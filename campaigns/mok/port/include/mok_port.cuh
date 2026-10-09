// Shared device helpers of the MoK port. Every owner package compiles this header
// from the one pinned revision, like mxfp8.cuh and ThunderKittens.
// It carries no owner geometry and no owner template bodies.
#pragma once

#include <cstdint>
#include "kittens.cuh"

namespace mok_port {

enum class RoutedPrecision { BF16, MXFP8 };

// Device-constructible stand-in for upstream `index_gl = gl<int, 1, 1, 1, -1>` (no TMA tile),
// with the subset of the gl interface the op bodies use: operator[]({i}) and the extents.
// Providers build it on the host from a caller pointer; op wrappers build counter views on
// the device from the framework counter workspace plus a linker-bound slot base.
template <typename T>
struct vector_view {
    T* raw_ptr;
    int cols_internal;
    __host__ __device__ static constexpr int batch() { return 1; }
    __host__ __device__ static constexpr int depth() { return 1; }
    __host__ __device__ static constexpr int rows() { return 1; }
    __host__ __device__ int cols() const { return cols_internal; }
    __device__ T& operator[](const kittens::coord<kittens::ducks::default_type>& idx) const {
        return raw_ptr[idx.c];
    }
};
using index_view = vector_view<int>;
static_assert(sizeof(index_view) == 16 && alignof(index_view) == 8, "index_view is a 16-byte residue field");

// Generic address of a linker-placed dynamic shared-memory claim.
__device__ __forceinline__ uint64_t smem_claim_address(int byte_offset) {
    extern __shared__ __align__(1024) unsigned char pipelink_smem[];
    return reinterpret_cast<uint64_t>(pipelink_smem + byte_offset);
}

}  // namespace mok_port
