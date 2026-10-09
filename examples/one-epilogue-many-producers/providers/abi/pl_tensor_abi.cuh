#ifndef PIPELINK_TENSOR_IMPORT_ABI_CUH
#define PIPELINK_TENSOR_IMPORT_ABI_CUH

#include <cstdint>
#include <pipelink/tensor_source.h>
#include "gemm_tile_args.h"

// Example interface; source placement is supplied by the producer manifest.
// One CTA, exactly 128 participating threads, warps 0 through 3.
constexpr uint32_t kPlTensorThreads = 128;
constexpr uint32_t kPlTensorRows = 128;
constexpr uint32_t kPlTensorColumns = 32;
constexpr uint32_t kPlTensorFragmentElements = 32;
constexpr uint32_t kPlTensorImportScratchBytes = 16384;
constexpr uint32_t kPlTensorSharedAlignment = 128;

#ifdef __CUDACC__
// args borrows the public GEMM call interface, never a private params segment.
// The consumer constructs it from its own params. Its pointers do not escape.
// Both entry points are called by all 128 threads. Only the complete first warp
// allocates/frees the 32-column TMEM allocation. Produce returns after all MMA
// completions; release is legal only after the caller's import_done rendezvous.
extern "C" __device__ pipelink_tensor_source plx_produce(const pl_gemm_tile_args* args, void* frame);
extern "C" __device__ void plx_release(const pl_gemm_tile_args* args);

// Exact normalized TensorImport.callable.declaration (including parameter names).
// source_base is an opaque producer-returned handle, not a consumer observation
// of a placement binding. The generated adapter unpacks it from the export facts.
// It stays live and immutable until the generated done event.
// fragment32 addresses 32 contiguous, 4-byte-aligned private float cells belonging
// only to the calling thread: position n receives logical value (threadIdx.x,n).
// It does not alias another thread's fragment, shared storage, or global buffers;
// it cannot escape or be accessed by peers. All 32 values are usable on return.
// import_scratch is a CTA-shared, 128-byte-aligned, caller-provided 16384-byte
// reservation. Its solver-relocatable base is uniform, nonnull, and disjoint from
// producer operand/barrier storage. Access is bounded
// by that reservation; no pointer or outstanding operation may outlive import_done.
// The import may use only the collectives/observations declared in CONTRACT.md.
extern "C" __device__ void pl_import(pipelink_tensor_source source_base, float* fragment32, void* import_scratch);
extern "C" __device__ void pl_import_ready();
extern "C" __device__ void pl_import_done();

// Ordinary call-channel imports. The linker supplies forwarding definitions;
// caller code must not call the peer's plx_produce/plx_release symbols directly.
extern "C" __device__ pipelink_tensor_source pipelink_call_tensor_produce(const pl_gemm_tile_args* args, void* frame);
extern "C" __device__ void pipelink_call_tensor_release(const pl_gemm_tile_args* args);

// Existing consumer-owned role and fixed helper envelope. The arithmetic helper
// is emitted/frozen owner-side by the restricted TileLang TIR route; it is
// not authored or recovered at link time. It neither allocates nor synchronizes.
extern "C" __device__ void plx_role_tensor_caller(const void* raw_params, unsigned variant);
extern "C" __device__ void plx_consume(const void* raw_params, pipelink_tensor_source source_base, void* import_scratch);
extern "C" __device__ void plx_relu_values(const float* fragment32, const float* bias, float* output, int logical_feature, int token_base, int output_stride);

// Live solver-binding functions: CTA dynamic-shared offsets in bytes, not host
// pointers. All declarations bind to the same composition-wide reservations.
extern "C" __device__ int pipelink_tensor_producer_smem();
extern "C" __device__ int pipelink_tensor_tmem_base_cell();
extern "C" __device__ int pipelink_tensor_import_smem();
#endif

#endif
