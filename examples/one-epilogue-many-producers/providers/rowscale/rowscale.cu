#include "rowscale_abi.cuh"
#include "params.h"
#include "row_scale.cuh"
#include <pipelink/stage.hpp>

PL_STAGE("row-scale", "1.0.0");
PL_NO_PARAMS();
PL_EXPORT(produce, plx_scale_produce);
PL_CALL(scale_produce, plx_scale_produce, pl::consume, "produce");
PL_EXPORT(release, plx_scale_release);
PL_CALL(scale_release, plx_scale_release, pl::consume, "release");
PL_CLAIM("pipelink_rowscale_smem", "smem", 128 * sizeof(float), 128, "kernel");
PL_EXPORT_TENSOR(row_scale,
    pl::identity{"row_scale", "tensor_work", "completed", "f32", "rms_row_scale", {{"m", 0, 128}}},
    pl::placement{"smem", "byte", {"address"},
                  [](int m) { return std::array<int, 1>{m * 4}; }, pl::coord{"m", 0, 128}},
    pl::strings{"smem_load_scalar"}, pl::readiness{"scale_ready", "scale_done"}, pl::fixed_bindings{},
    pl::lends{pl::smem("scale_storage", 128 * sizeof(float), 128, "pipelink_rowscale_smem",
                       "row_scale", "tensor_work", "scale_done")},
    pl::calls{"scale_produce", "scale_release"});

extern "C" __device__ pipelink_tensor_source plx_scale_produce(const pl_rowscale_args* args, void* frame) {
  (void)frame;
  const auto& params = *args;
  extern __shared__ __align__(1024) unsigned char plx_rowscale_smem[];
  auto* scale = reinterpret_cast<float*>(plx_rowscale_smem + pipelink_rowscale_smem());
  pl_row_scale(reinterpret_cast<const __half*>(params.a), params.K,
               int(blockIdx.x) * 128, params.eps, scale);
  return pl::source_of(scale);
}

extern "C" __device__ void plx_scale_release(const pl_rowscale_args* args) {
  (void)args;
}
