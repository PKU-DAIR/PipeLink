#pragma once
#include <pipelink/tensor_source.h>
#include "../abi/coda_tile_args.h"

extern "C" __device__ pipelink_tensor_source pipelink_call_tensor_produce(const void* view, coda_tile_args* args, void* frame);
extern "C" __device__ void pipelink_call_tensor_release(const void* view, coda_tile_args* args);
extern "C" __device__ void pl_import_acc(pipelink_tensor_source source, uint32_t chunk, float* dest, void* scratch);
extern "C" __device__ void pl_import_acc_ready();
extern "C" __device__ void pl_import_acc_done();
extern "C" __device__ int pipelink_coda_acc_smem();
extern "C" __device__ int pipelink_coda_output_smem();
extern "C" __device__ void plx_role_coda_epilogue(const void* params, unsigned variant);
