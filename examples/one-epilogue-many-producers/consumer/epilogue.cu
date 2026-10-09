// One owner schedule and arithmetic body for every accumulator placement.
#include "epilogue_abi.cuh"
#include "params.h"
#include <pipelink/stage.hpp>
PL_TYPE(pl_gemm_tile_args, const void* a; const void* b; int32_t M; int32_t N; int32_t K;);

PL_STAGE("rowscale-relu-epilogue", "1.0.0");
PL_PARAMS(EpilogueParams, a, b, bias, output, M, N, K, eps);
PL_ROLE(epilogue, plx_role_epilogue, 4);
PL_CALL(tensor_produce, pipelink_call_tensor_produce, pl::produce, "epilogue", "produce");
PL_CALL(tensor_release, pipelink_call_tensor_release, pl::produce, "epilogue", "release");
PL_CALL(scale_produce, pipelink_call_scale_produce, pl::produce, "epilogue", "produce");
PL_CALL(scale_release, pipelink_call_scale_release, pl::produce, "epilogue", "release");
PL_SITE(tensor_import_site, "epilogue", pl::all_threads, "tensor_work",
    pl::guarantees{"enter_once", "single_completion", "converged", "full_warps", "no_early_return"},
    pl::strings{"tensor_ready", "scale_ready"});
PL_IMPORT_TENSOR(acc, pl_import_acc,
    pl::identity{"accumulator", "tensor_work", "completed_k", "f32", "fp16_dot_fp32_completed",
                 {{"m", 0, 128}, {"n", 0, 128}}},
    pl::args{{"source", "source_base"}, {"dest", "dest_fragment"}, {"scratch", "scratch"}},
    pl::demand_positions{pl::coord{"t", 0, 128}, pl::coord{"p", 0, 128},
                        [](int t, int p) { return std::array<int, 4>{t, p, t, p}; }},
    pl::strings{"tensor_import_site"}, "import_done",
    pl::lends{
        pl::smem("acc_scratch", 65536, 128, "pipelink_epilogue_acc_smem", "epilogue", "tensor_ready", "import_done"),
        pl::regs("acc_dest", 128, "acc_dest", "epilogue", "tensor_ready", "import_done"),
        pl::metadata("acc_metadata", 2, "acc_metadata", "epilogue", "tensor_ready", "import_done"),
        pl::frame("acc_frame", 128, "acc_frame", "epilogue", "tensor_work", "import_done"),
        pl::cta_barrier("acc_barrier", 0, "epilogue", "tensor_work", "import_done", {2, 1})},
    {pl::rendezvous{"tensor_ready", "acc_barrier", pl::all_threads},
     pl::rendezvous{"import_done", "acc_barrier", pl::all_threads}});
PL_IMPORT_TENSOR(scale, pl_import_scale,
    pl::identity{"row_scale", "tensor_work", "completed", "f32", "rms_row_scale", {{"m", 0, 128}}},
    pl::args{{"source", "source_base"}, {"dest", "dest_fragment"}, {"scratch", "scratch"}},
    pl::demand_positions{pl::coord{"t", 0, 128},
                        [](int t) { return std::array<int, 3>{t, t, 0}; }},
    pl::strings{"tensor_import_site"}, "scale_done",
    pl::lends{
        pl::smem("scale_scratch", 1024, 128, "pipelink_epilogue_scale_smem", "epilogue", "scale_ready", "scale_done"),
        pl::regs("scale_dest", 1, "scale_dest", "epilogue", "scale_ready", "scale_done"),
        pl::cta_barrier("scale_barrier", 0, "epilogue", "tensor_work", "scale_done", {2, 1})},
    {pl::rendezvous{"scale_ready", "scale_barrier", pl::all_threads},
     pl::rendezvous{"scale_done", "scale_barrier", pl::all_threads}});

extern "C" __device__ void plx_role_epilogue(const void* raw_params, unsigned variant) {
  (void)variant;
  const auto& params = *static_cast<const EpilogueParams*>(raw_params);
  pl_gemm_tile_args acc_args{params.a, params.b, params.M, params.N, params.K};
  pl_rowscale_args scale_args{params.a, params.M, params.K, params.eps};
  extern __shared__ __align__(1024) unsigned char plx_epilogue_smem[];
  float frame[128];
  const auto acc_source = pipelink_call_tensor_produce(&acc_args, frame);
  pl_import_acc_ready();
  float acc[128];
  pl_import_acc(acc_source, acc, plx_epilogue_smem + pipelink_epilogue_acc_smem());
  const auto scale_source = pipelink_call_scale_produce(&scale_args, nullptr);
  pl_import_scale_ready();
  float scale;
  pl_import_scale(scale_source, &scale, plx_epilogue_smem + pipelink_epilogue_scale_smem());
  const int row = int(blockIdx.x) * 128 + int(threadIdx.x);
  const int column = int(blockIdx.y) * 128;
  const float bias = params.bias[row];  // Loaded once: output stores may alias bias.
  #pragma unroll
  for (int n = 0; n < 128; n += 4) {
    float4 values;
    float* lanes = reinterpret_cast<float*>(&values);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float value = acc[n + i] * scale + bias;
      lanes[i] = value > 0.0f ? value : 0.0f;
    }
    *reinterpret_cast<float4*>(params.output + row * params.N + column + n) = values;
  }
  pl_import_scale_done();
  pipelink_call_scale_release(&scale_args);
  pl_import_acc_done();
  pipelink_call_tensor_release(&acc_args);
}
