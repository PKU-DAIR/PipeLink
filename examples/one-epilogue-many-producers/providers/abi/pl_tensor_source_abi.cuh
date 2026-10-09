#ifndef PIPELINK_TENSOR_SOURCE_ABI_CUH
#define PIPELINK_TENSOR_SOURCE_ABI_CUH

#include "pl_tensor_abi.cuh"

#ifdef __CUDACC__
// Register exports write into the caller frame; shared/TMEM exports ignore it.
// The returned opaque handle stays live through the generated done event.
extern "C" __device__ pipelink_tensor_source plx_produce(const pl_gemm_tile_args* args, void* frame);

// Producer-owned solver bindings: byte offset of a disjoint shared tile, and a
// named barrier id for the 128 producer participants (never the caller's bar 0).
extern "C" __device__ int pipelink_tensor_export_tile();
extern "C" __device__ int pipelink_tensor_producer_barrier();
#endif

#endif
