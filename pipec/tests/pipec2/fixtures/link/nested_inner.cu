// Role-less, topology-free leaf: export `reduce` sums 32 values over one warp.
#include <pipelink/stage.hpp>

extern "C" __device__ float nested_inner_reduce(const float* data) {
    float value = data[threadIdx.x % 32];
    for (int offset = 16; offset; offset /= 2) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

PL_STAGE("org.nested.inner", "1.0.0");
PL_NO_PARAMS();
PL_CALL(reduce, nested_inner_reduce, pl::consume, "reduce");
PL_INTERFACE(reduce, pl::interface{pl::cohort{32}, pl::collective::warp, {pl::param{0, "gmem", "r", "f32", 32}}});
