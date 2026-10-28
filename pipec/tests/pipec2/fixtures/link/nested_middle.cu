// Role-less, topology-free library stage: export `accumulate` (128-thread cohort) makes a
// nested call into org.nested.inner's `reduce` from each of its warps (32-thread sub-cohort).
#include <pipelink/stage.hpp>

extern "C" __device__ float pipelink_call_reduce(const float* data);
extern "C" __device__ float nested_middle_accumulate(float* data, unsigned count) {
    const unsigned warp = threadIdx.x / 32;
    const float partial = pipelink_call_reduce(data + 32 * warp);
    if (threadIdx.x < count) data[threadIdx.x] += partial;
    return partial;
}

PL_STAGE("org.nested.middle", "1.0.0");
PL_NO_PARAMS();
PL_CALL(accumulate, nested_middle_accumulate, pl::consume, "accumulate");
PL_INTERFACE(accumulate, pl::interface{pl::cohort{128}, pl::collective::none,
                                       {pl::param{0, "gmem", "rw", "f32", 128}}});
PL_CALL(reduce, pipelink_call_reduce, pl::produce, pl::in_exports{"accumulate"}, "reduce");
PL_CALL_SITE(reduce, pl::site{pl::cohort{32}, {pl::param{0, "gmem", "r", "f32", 32}}});
