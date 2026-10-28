// Nested-call caller: role `work` (4 warps) calls export `accumulate` of org.nested.middle
// through the generated pipelink_call_accumulate trampoline, as one 128-thread cohort.
#include <pipelink/stage.hpp>

struct Params { float* data; unsigned count; };

extern "C" __device__ float pipelink_call_accumulate(float* data, unsigned count);
extern "C" __device__ void nested_outer_work(const void* raw, unsigned) {
    const auto* params = static_cast<const Params*>(raw);
    params->data[threadIdx.x] = pipelink_call_accumulate(params->data, params->count);
}

PL_STAGE("org.nested.outer", "1.0.0");
PL_PARAMS(Params, data, count);
PL_ROLE(work, nested_outer_work, 4);
PL_CALL(accumulate, pipelink_call_accumulate, pl::produce, "work");
PL_CALL_SITE(accumulate, pl::site{pl::cohort{128}, {pl::param{0, "gmem", "rw", "f32", 256}}});
PL_ENTRY_RENDEZVOUS(0, 128);
