#pragma once
#include <pipelink/tensor_source.h>
#include "../providers/abi/gemm_tile_args.h"
#include "../providers/rowscale/params.h"

#ifdef __CUDACC__
extern "C" __device__ pipelink_tensor_source pipelink_call_tensor_produce(const pl_gemm_tile_args* args, void* frame);
extern "C" __device__ void pipelink_call_tensor_release(const pl_gemm_tile_args* args);
extern "C" __device__ pipelink_tensor_source pipelink_call_scale_produce(const pl_rowscale_args* args, void* frame);
extern "C" __device__ void pipelink_call_scale_release(const pl_rowscale_args* args);
extern "C" __device__ void pl_import_acc(pipelink_tensor_source source, float* dest, void* scratch);
extern "C" __device__ void pl_import_acc_ready();
extern "C" __device__ void pl_import_acc_done();
extern "C" __device__ void pl_import_scale(pipelink_tensor_source source, float* dest, void* scratch);
extern "C" __device__ void pl_import_scale_ready();
extern "C" __device__ void pl_import_scale_done();
extern "C" __device__ int pipelink_epilogue_acc_smem();
extern "C" __device__ int pipelink_epilogue_scale_smem();
extern "C" __device__ void plx_role_epilogue(const void* params, unsigned variant);
#endif
