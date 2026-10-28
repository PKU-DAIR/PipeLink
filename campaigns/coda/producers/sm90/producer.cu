#include "mainloop.cuh"
#include <pipelink/stage.hpp>
#include <pipelink/host_view.h>
#include <pipelink_host_abi_v2.h>
extern "C" __device__ int pipelink_coda_sm90_storage();
extern "C" __device__ __inline_hint__ pipelink_tensor_source coda_produce(
    const void* view, coda_tile_args* args, void* frame) {
  extern __shared__ __align__(1024) unsigned char coda_sm90_smem[];
  auto& storage = *reinterpret_cast<coda_sm90::Storage*>(coda_sm90_smem + pipelink_coda_sm90_storage());
  const auto& params = *static_cast<const CodaSm90Params*>(static_cast<const pipelink_host_view*>(view)->residue);
  coda_sm90::Accumulator accumulator;
  coda_sm90::mainloop(params, *args, storage, accumulator,
                     [&](const coda_sm90::Accumulator& accumulator, const coda_tile_args&) {
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      #pragma unroll
      for (int p = 0; p < 4; ++p) {
        static_cast<float*>(frame)[j * 8 + p * 2] = accumulator.tiles[0][j].data[p].x;
        static_cast<float*>(frame)[j * 8 + p * 2 + 1] = accumulator.tiles[0][j].data[p].y;
      }
    }
  });
  return pl::source_of(frame);
}
extern "C" __device__ void coda_release(const void*, coda_tile_args*) {}
PL_TYPE(coda_tile_args, a, b, M, N, K, m0, n0);
PL_STAGE("coda.producer.sm90", "1.0.0");
PL_CALL(tensor_produce, coda_produce, pl::consume, "produce");
PL_CALL(tensor_release, coda_release, pl::consume, "release");
PL_CLAIM_SMEM("pipelink_coda_sm90_storage", coda_sm90::Storage,
              pl::during{"accumulator", "tile_work", "tensor_ready"});
PL_EXPORT_TENSOR(accumulator,
  pl::identity{"accumulator", "tile_work", "completed_k", "f32", "bf16_dot_fp32_completed",
               {{"m", 0, 128}, {"n", 0, 256}}},
  pl::placement{"reg", "scalar", {"participant", "position"}, [](int m, int n) {
    static const auto coordinates = [] {
      std::array<int, 128 * 256> map{};
      for (int t = 128; t < 384; ++t)
        for (int p = 0; p < int(sizeof(coda_sm90::Accumulator) / sizeof(float)); ++p)
          map[coda_sm90::row(t, p) * 256 + coda_sm90::col(t, p)] = t * 128 + p;
      return map;
    }();
    const int address = coordinates[m * 256 + n];
    return std::array<int, 2>{address / 128, address % 128};
  }, pl::coord{"m", 0, 128}, pl::coord{"n", 0, 256}},
  pl::strings{"reg_copy", "reg_copy_round", "shfl_xor", "smem_store_scalar", "smem_store_round"},
  pl::readiness{"tensor_ready", "import_done"}, pl::fixed_bindings{}, pl::lends{},
  pl::calls{"tensor_produce", "tensor_release"});
PL_HOST_GROUP("coda.sm90", "coda.sm90.params", "coda_sm90", plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_RESIDUE(2, "coda.sm90.residue", CodaSm90Params, 301, true, pl::host::roles({}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({}))
  .field(pl::host::scalar(1, "coda.M", "i64").caller("M"))
  .field(pl::host::scalar(2, "coda.N", "i64").caller("N"))
  .field(pl::host::scalar(3, "coda.K", "i64").caller("K")));
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("coda.x", "bf16").rank(2).read().caller("x"));
PL_HOST_CHANNEL(2, uint16_t, pl::host::channel("coda.w", "bf16").rank(2).read().caller("w"));
PL_HOST_SCHEDULER(pl::host::scheduler("coda.sm90", "coda.sm90.grid", pl::host::schema()));
