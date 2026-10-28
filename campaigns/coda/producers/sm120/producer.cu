#include "mainloop.cuh"
#include <pipelink/stage.hpp>
#include "abi.h"

extern "C" __device__ int pipelink_coda_sm120_smem();
extern "C" __device__ pipelink_tensor_source coda_produce(const void* view, coda_tile_args* args, void* frame) {
  using namespace coda_sm120;
  extern __shared__ __align__(128) unsigned char coda_sm120_shared[];
  auto& shared = *reinterpret_cast<Config::SharedStorage*>(coda_sm120_shared + pipelink_coda_sm120_smem());
  Config::origin(int(blockIdx.x), args->M, args->N, args->m0, args->n0);
  auto const& params = *static_cast<CodaSm120Residue const*>(
      static_cast<pipelink_host_view const*>(view)->residue);
  args->a = params.a;
  args->b = params.b;
  Config::Fragment acc;
  Config::run(*args, shared, acc);
  CUTLASS_PRAGMA_UNROLL
  for (int p = 0; p < Config::Fragment::kElements; ++p) static_cast<float*>(frame)[p] = acc[p];
  return pl::source_of(frame);
}
extern "C" __device__ void coda_release(const void*, coda_tile_args*) {
  // Caller-owned register storage needs no allocation reclamation.
}

#ifdef PIPELINK_EMIT
PL_TYPE(coda_tile_args, a, b, M, N, K, m0, n0);
namespace coda_sm120 {
PL_STAGE("coda.producer.sm120", "1.0.0");
PL_CALL(tensor_produce, coda_produce, pl::consume, "produce");
PL_CALL(tensor_release, coda_release, pl::consume, "release");
PL_CLAIM_SMEM("pipelink_coda_sm120_smem", Config::SharedStorage,
              pl::during{"accumulator", "tile_work", "tensor_ready"});
PL_EXPORT_TENSOR(accumulator,
    pl::identity{"accumulator", "tile_work", "completed_k", "f32", "bf16_dot_fp32_completed",
                 {{"m", 0, 128}, {"n", 0, 256}}},
    pl::placement{"reg", "scalar", {"participant", "position"}, [](int m, int n) {
      // Capture the same CUTLASS accumulator coordinate helper used by the native stores.
      static auto addresses = [] {
        std::array<int, 128 * 256> result{};
        for (int t = 0; t < 256; ++t) {
          for (int p = 0; p < Config::Fragment::kElements; ++p) {
            auto rc = Config::coordinate(t, p);
            result[rc.row() * 256 + rc.column()] = t * 128 + p;
          }
        }
        return result;
      }();
      int address = addresses[m * 256 + n];
      return std::array<int, 2>{address / 128, address % 128};
    }, pl::coord{"m", 0, 128}, pl::coord{"n", 0, 256}},
    pl::strings{"reg_copy", "reg_copy_round", "shfl_xor", "smem_store_scalar", "smem_store_round"},
    pl::readiness{"tensor_ready", "import_done"}, pl::fixed_bindings{}, pl::lends{},
    pl::calls{"tensor_produce", "tensor_release"});

PL_HOST_GROUP("coda.sm120.owner", "coda.sm120.params", "coda_sm120", plk_header_v2,
              PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_ABI_TYPE(120, CodaSm120Residue, a, b);
PL_HOST_RESIDUE(2, "coda.sm120.residue", CodaSm120Residue, 120, true, pl::host::roles({}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({}))
    .field(pl::host::scalar(1, "coda.M", "i64").caller("M"))
    .field(pl::host::scalar(2, "coda.N", "i64").caller("N"))
    .field(pl::host::scalar(3, "coda.K", "i64").caller("K")));
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("coda.x", "bf16").rank(2).read().caller("x"));
PL_HOST_CHANNEL(2, uint16_t, pl::host::channel("coda.w", "bf16").rank(2).read().caller("w"));
PL_HOST_SCHEDULER(pl::host::scheduler("coda.sm120.owner", "coda.sm120.static-grid", pl::host::schema()));
}
#endif