// Minimal sm90-legal caller: imports the provider's `scale` export through
// the generated pipelink_call_scale trampoline. Default topology
// (cta_group 1 / static scheduler).
#include <pipelink/stage.hpp>

struct Params { float* output; unsigned count; };

extern "C" __device__ float pipelink_call_scale(float value);
extern "C" __device__ void sm90_caller_work(const void* raw, unsigned variant) {
    const auto* params = static_cast<const Params*>(raw);
    if (params->count > variant) params->output[variant] = pipelink_call_scale(3.0f);
}

PL_STAGE("org.sm90.caller", "1.0.0");
PL_PARAMS(Params, output, count);
PL_ROLE(call, sm90_caller_work, 4);
PL_CALL(scale, pipelink_call_scale, pl::produce, "call");
