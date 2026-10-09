// Compile probe: upstream Megakernels llama-1b decode kernel, no pybind.
#include "llama.cuh"

#include "rms_matvec_rope_append.cu"
#include "attention_partial.cu"
#include "attention_reduction.cu"
#include "matvec_adds.cu"
#include "upgate.cu"
#include "rms_lm_head.cu"

using namespace kittens;
using namespace megakernel;

using rms_qkv_rope_append_op = rms_qkv_rope_append<default_config, llama_1b_globals>;
using attention_partial_op = attention_partial<default_config, llama_1b_globals>;
using attention_reduction_op = attention_reduction<default_config, llama_1b_globals>;
using o_proj_op = o_proj<default_config, llama_1b_globals>;
using rms_upgate_silu_op = rms_upgate_silu<default_config, llama_1b_globals>;
using downproj_op = downproj<default_config, llama_1b_globals>;
using rms_lm_head_op = rms_lm_head<default_config, llama_1b_globals>;

template __global__ void megakernel::mk<default_config, llama_1b_globals, attention_partial_op,
                            attention_reduction_op, rms_qkv_rope_append_op, downproj_op,
                            o_proj_op, rms_upgate_silu_op, rms_lm_head_op>(
    const __grid_constant__ llama_1b_globals g);
