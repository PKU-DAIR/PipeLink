#include "epilogue_abi.cuh"
#include "params.h"
#include "output.cuh"
#include <pipelink/stage.hpp>

PL_TYPE(coda_tile_args, a, b, M, N, K, m0, n0);
PL_STAGE("coda.swiglu", "1.0.0");
PL_ROLE(epilogue, plx_role_coda_epilogue, CODA_THREADS / 32);
PL_TOPOLOGY(pl::topology{CODA_CTA_GROUP,
    pl::cluster{{CODA_CLUSTER_X, 1, 1}, pl::launch{CODA_CLUSTER_X, {CODA_CLUSTER_X, 1, 1}}},
    false, "static", {}});
PL_CALL(tensor_produce, pipelink_call_tensor_produce, pl::produce, "epilogue", "produce");
PL_CALL(tensor_release, pipelink_call_tensor_release, pl::produce, "epilogue", "release");
PL_BINDING("chunk", "bits[32]", pl::coord{"chunk", 0, 8}, "epilogue",
           "chunk", "tensor_import_site", "tensor_ready_to_import_done");
PL_SITE(tensor_import_site, "epilogue",
    pl::participants{pl::coord{"t", CODA_FIRST_DATA_THREAD, CODA_FIRST_DATA_THREAD + 256}, [](int t) { return t; }},
    "tile_work", pl::guarantees{"enter_once", "converged", "full_warps", "no_early_return"},
    pl::strings{"tensor_ready", "chunk"});
PL_CLAIM_SMEM("pipelink_coda_output_smem", coda_epilogue::OutputTile,
              pl::during{"acc", "tensor_ready"}, "epilogue");
PL_IMPORT_TENSOR(acc, pl_import_acc,
    pl::identity{"accumulator", "tile_work", "completed_k", "f32", "bf16_dot_fp32_completed",
                 {{"m", 0, 128}, {"n", 0, 256}}},
    pl::args{{"source", "source_base"}, {"chunk", "binding"}, {"dest", "dest_fragment"}, {"scratch", "scratch"}},
    pl::demand_positions{pl::coord{"t", 0, 256}, pl::coord{"p", 0, 16}, pl::bind("chunk", "bits[32]"),
        [](int t, int p, pl::bound chunk) { return std::array<int, 4>{
            coda_epilogue::row(t), 32 * chunk + 16 * coda_epilogue::half_row(t) + p,
            t + CODA_FIRST_DATA_THREAD, p}; }},
    pl::strings{"tensor_import_site"}, "import_done",
    pl::lends{
        pl::frame("acc_frame", 128, "frame", "epilogue", "tile_work", "import_done"),
#if CODA_IMPORT_SCRATCH_BYTES != 0
        pl::smem("acc_scratch", CODA_IMPORT_SCRATCH_BYTES, 128, "pipelink_coda_acc_smem", "epilogue", "tensor_ready", "import_done"),
#endif
        pl::regs("acc_dest", 16, "dest", "epilogue", "tensor_ready", "import_done"),
        pl::metadata("acc_metadata", 2, "acc_metadata", "epilogue", "tensor_ready", "import_done"),
#if CODA_PROFILE == 90
        pl::cta_barrier("acc_inner", 1, "epilogue", "tensor_ready", "import_done", {2, 1}),
#endif
        pl::cta_barrier("acc_outer", 0, "epilogue", "tile_work", "import_done", {2, 1})},
    {pl::rendezvous{"tensor_ready", "acc_outer", pl::all_threads},
     pl::rendezvous{"import_done", "acc_outer", pl::all_threads}});

#include "host_annotations.cuh"

extern "C" __device__ void plx_role_coda_epilogue(const void* raw_params, unsigned variant) {
  (void)variant;
  const auto& view = *static_cast<const plk_role_view_coda_swiglu_epilogue*>(raw_params);
  const auto& params = *static_cast<const CodaEpilogueParams*>(view.residue);
  coda_tile_args args{params.a, params.b, params.M, params.N, params.K, 0, 0};
  extern __shared__ __align__(1024) unsigned char coda_epilogue_smem[];
  float frame[128];
  const auto source = pipelink_call_tensor_produce(raw_params, &args, frame);
  pl_import_acc_ready();
  auto& tile = *reinterpret_cast<coda_epilogue::OutputTile*>(coda_epilogue_smem + pipelink_coda_output_smem());
  const int t = int(threadIdx.x) - CODA_FIRST_DATA_THREAD;
#if CODA_IMPORT_SCRATCH_BYTES != 0
  void* scratch = coda_epilogue_smem + pipelink_coda_acc_smem();
#else
  void* scratch = nullptr;
#endif
  #pragma unroll
  for (uint32_t chunk = 0; chunk < 8; ++chunk) {
    if (t >= 0) {
      float acc[16];
      pl_import_acc(source, chunk, acc, scratch);
      coda_epilogue::stage(tile, t, chunk, acc);
    }
#if CODA_IMPORT_SCRATCH_BYTES != 0
    pl_import_acc_done();
#endif
  }
#if CODA_IMPORT_SCRATCH_BYTES == 0
  // No shared transport storage is lent, so chunks need no reuse rendezvous.
  pl_import_acc_done();
#endif
  if (t >= 0)
    coda_epilogue::store(tile, params.output, args.m0, args.n0, params.N, t);
  // All stores are synchronous; every CTA thread participates in owner release.
  pipelink_call_tensor_release(raw_params, &args);
}
