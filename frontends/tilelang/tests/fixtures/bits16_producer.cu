#include <pipelink/stage.hpp>
#include <cstdint>

#ifdef EXPORT_F16
#define EXPORT_DTYPE "f16"
#define VALUE_BITS 0x3c00
#else
#define EXPORT_DTYPE "bf16"
#define VALUE_BITS 0x3f80
#endif

struct bits16_args { int32_t tag; };
struct alignas(128) Bits16Tile { uint16_t values[128 * 2]; };
extern "C" __device__ pipelink_tensor_source bits16_produce(const bits16_args*, void*);
extern "C" __device__ void bits16_release(const bits16_args*);
#ifdef EXPORT_SMEM
extern "C" __device__ int pipelink_bits16_storage();
#endif

#if defined(__CUDACC__)
extern "C" __device__ pipelink_tensor_source bits16_produce(const bits16_args* args, void* frame) {
#ifdef EXPORT_SMEM
  extern __shared__ __align__(128) unsigned char bits16_shared[];
  auto* destination = reinterpret_cast<uint16_t*>(bits16_shared + pipelink_bits16_storage());
#else
  auto* destination = static_cast<uint16_t*>(frame);
#endif
  for (int p = 0; p < 2; ++p) {
    uint16_t value = uint16_t(VALUE_BITS + ((int(threadIdx.x) * 2 + p + args->tag) & 127));
#ifdef EXPORT_SMEM
    destination[int(threadIdx.x) * 2 + p] = value;
#else
    destination[p] = value;
#endif
  }
  return pl::source_of(destination);
}
extern "C" __device__ void bits16_release(const bits16_args*) {
  __syncthreads();
}
#endif

PL_STAGE("test.bits16-producer", "1.0.0");
PL_NO_PARAMS();
PL_TYPE(bits16_args, tag);
PL_CALL(produce, bits16_produce, pl::consume, "produce");
PL_CALL(release, bits16_release, pl::consume, "release");
#ifdef EXPORT_SMEM
PL_CLAIM_SMEM("pipelink_bits16_storage", Bits16Tile, "kernel");
#endif
PL_EXPORT_TENSOR(tile,
  pl::identity{"values", "tile_work", "completed", EXPORT_DTYPE, "exact_float_payload",
               {{"m", 0, 128}, {"n", 0, 2}}},
#ifdef EXPORT_SMEM
  pl::placement{"smem", "byte", {"address"}, [](int m, int n) {
    return std::array<int, 1>{(m * 2 + n) * 2};
  }, pl::coord{"m", 0, 128}, pl::coord{"n", 0, 2}},
  pl::strings{"smem_load_scalar", "smem_load_round"},
#else
  pl::placement{"reg", "scalar", {"participant", "position"}, [](int m, int n) {
    return std::array<int, 2>{m, n};
  }, pl::coord{"m", 0, 128}, pl::coord{"n", 0, 2}},
  pl::strings{"reg_copy", "reg_copy_round"},
#endif
  pl::readiness{"tensor_ready", "import_done"}, pl::fixed_bindings{},
  pl::lends{
#ifdef EXPORT_SMEM
    {"source_storage", "smem", sizeof(Bits16Tile), alignof(Bits16Tile), "cta", "relocatable",
     "pipelink_bits16_storage", std::nullopt, "producer", "tile_work", "tensor_reclaimed", pl::sync{2, 1}}
#endif
  }, pl::calls{"produce", "release"});
