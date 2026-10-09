#include <pipelink/stage.hpp>

// Multiplication by two is exact for the bounded float32 input in run.py.
extern "C" __device__ float hello_scale(float value) { return value * 2.0f; }

PL_STAGE("org.hello.provider", "1.0.0");
PL_NO_PARAMS();
PL_EXPORT(scale, hello_scale);
PL_CALL(scale, hello_scale, pl::consume, "scale");
