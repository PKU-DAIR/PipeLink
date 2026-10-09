#include <pipelink/stage.hpp>

struct HelloCallerParams {
    const float* input;
    float* output;
    unsigned count;
};

extern "C" __device__ float pipelink_call_scale(float value);
extern "C" __device__ void hello_call(const void* raw, unsigned /*variant*/) {
    const auto* params = static_cast<const HelloCallerParams*>(raw);
    // Flatten all grid dimensions; the linker supplies the four-warp block.
    const size_t block = blockIdx.x + size_t(gridDim.x) *
        (blockIdx.y + size_t(gridDim.y) * blockIdx.z);
    const size_t stride = size_t(blockDim.x) * gridDim.x * gridDim.y * gridDim.z;
    for (size_t i = block * blockDim.x + threadIdx.x; i < params->count; i += stride) {
        params->output[i] = pipelink_call_scale(params->input[i]);
    }
}

PL_STAGE("org.hello.caller", "1.0.0");
PL_PARAMS(HelloCallerParams, input, output, count);
PL_ROLE(call, hello_call, 4);
PL_CALL(scale, pipelink_call_scale, pl::produce, "call");
