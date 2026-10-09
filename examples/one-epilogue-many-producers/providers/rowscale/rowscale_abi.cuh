#pragma once
#include <pipelink/tensor_source.h>
#include "params.h"

#ifdef __CUDACC__
extern "C" __device__ pipelink_tensor_source plx_scale_produce(const pl_rowscale_args* args, void* frame);
extern "C" __device__ void plx_scale_release(const pl_rowscale_args* args);
extern "C" __device__ int pipelink_rowscale_smem();
#endif
