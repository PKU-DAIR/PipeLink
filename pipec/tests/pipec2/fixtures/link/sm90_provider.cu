// Minimal sm90-legal provider: a role-less library stage exporting one
// callable that the caller stage reaches through the generated
// pipelink_call_scale trampoline (the pattern of gemm-evt's register-compute
// stage). Only sm90+ baseline vocabulary: no TMEM, CLC or 2-SM cta_group;
// the default topology is cta_group 1 / static.
#include <pipelink/stage.hpp>

extern "C" __device__ float sm90_provider_scale(float value) { return value * 2.0f; }

PL_STAGE("org.sm90.provider", "1.0.0");
PL_NO_PARAMS();
PL_EXPORT(scale, sm90_provider_scale);
PL_CALL(scale, sm90_provider_scale, pl::consume, "scale");
