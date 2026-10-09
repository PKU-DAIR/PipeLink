// Consumer Blackwell TMA/mma.sync provider: the exported register coordinates are
// exactly the shared mainloop's mma.sync fragment coordinates, not a copied canonical tile.
#include "../abi/pl_tensor_source_abi.cuh"
#include "gemm_mainloop_sm120.cuh"
#include <pipelink/stage.hpp>
PL_TYPE(pl_gemm_tile_args, const void* a; const void* b; int32_t M; int32_t N; int32_t K;);

#if defined(PL_TILE_N) && PL_TILE_N != 128
#error "The sm120 teaching provider uses a 128x128 tile"
#endif
#if defined(PL_EXPORT_TMEM) || defined(PL_ELEM_BF16)
#error "The sm120 provider exports f32 register or shared storage only"
#endif
#ifdef PL_EXPORT_SMEM
#define PL_SOURCE_NAMESPACE pl_tensor_owner_sm120_smem128n128
#define PL_PARAMS_ACCESSOR pipelink_params_tensor_producer_sm120_smem128n128
#else
#define PL_SOURCE_NAMESPACE pl_tensor_owner_sm120_reg128n128
#define PL_PARAMS_ACCESSOR pipelink_params_tensor_producer_sm120_reg128n128
#endif
extern "C" __device__ const void* PL_PARAMS_ACCESSOR();

namespace PL_SOURCE_NAMESPACE {
using namespace pl_gemm_sm120;
}

#if defined(__CUDACC__)
extern "C" __device__ pipelink_tensor_source plx_produce(const pl_gemm_tile_args* args, void* frame) {
  using namespace pl_gemm_sm120;
  extern __shared__ __align__(1024) unsigned char plx_tensor_owner_smem[];
  auto& shared = *reinterpret_cast<Configuration::SharedStorage*>(
      plx_tensor_owner_smem + pipelink_tensor_producer_smem());
  const auto& params = *static_cast<const pl_gemm_tma_params*>(PL_PARAMS_ACCESSOR());
  auto coordinates = Configuration::partitioned_coordinates(int(threadIdx.x));
  auto accumulator = Configuration::TiledMma{}.get_slice(threadIdx.x).make_fragment_C(coordinates);
  mainloop(&params.a, &params.b, args->K, shared, pipelink_tensor_producer_barrier(), accumulator);
#ifdef PL_EXPORT_SMEM
  (void)frame;
  // Every issued TMA tile was awaited and consumed; mainloop's collective
  // drain orders every ldmatrix reader before we reuse the dead operand ring.
  // Only this smem export keeps producer_storage lent through reclamation.
  static_assert(sizeof(shared.a) + sizeof(shared.b) >= 65536);
  auto* source = reinterpret_cast<float*>(&shared);
  CUTE_UNROLL
  for (int i = 0; i < columns; i += 2) {
    auto logical = coordinates(i);
    const int offset = export_offset(int(cute::get<0>(logical)), int(cute::get<1>(logical)));
    // Fragment values i, i+1 are adjacent columns; the XOR may reverse the pair but never separates it.
    *reinterpret_cast<float2*>(source + (offset & ~1)) = (offset & 1)
        ? make_float2(accumulator(i + 1), accumulator(i))
        : make_float2(accumulator(i), accumulator(i + 1));
  }
  __threadfence_block();
  return pl::source_of(source);
#else
  CUTE_UNROLL
  for (int i = 0; i < columns; ++i) static_cast<float*>(frame)[i] = accumulator(i);
  return pl::source_of(frame);
#endif
}

extern "C" __device__ void plx_release(const pl_gemm_tile_args*) {
  pl_gemm_sm120::rendezvous(pipelink_tensor_producer_barrier());
}
#endif

#ifdef PIPELINK_EMIT
namespace PL_SOURCE_NAMESPACE {
#ifdef PL_EXPORT_SMEM
PL_STAGE("tensor.producer.sm120-smem128n128", "1.0.0");
#else
PL_STAGE("tensor.producer.sm120-reg128n128", "1.0.0");
#endif
PL_PARAMS(pl_gemm_tma_params, a, b);
PL_EXPORT(produce, plx_produce);
PL_CALL(tensor_produce, plx_produce, pl::consume, "produce");
PL_EXPORT(release, plx_release);
PL_CALL(tensor_release, plx_release, pl::consume, "release");
#ifdef PL_EXPORT_SMEM
PL_CLAIM_SMEM("pipelink_tensor_producer_smem", Configuration::SharedStorage, "kernel");
#else
PL_CLAIM_SMEM("pipelink_tensor_producer_smem", Configuration::SharedStorage,
              pl::during{"accumulator", "tensor_work", "tensor_ready"});
#endif
PL_CLAIM("pipelink_tensor_producer_barrier", "named_barrier", 1, 1, "kernel");
PL_EXPORT_TENSOR(accumulator,
    pl::identity{"accumulator", "tensor_work", "completed_k", "f32", "fp16_dot_fp32_completed",
                 {{"m", 0, rows}, {"n", 0, columns}}},
#ifdef PL_EXPORT_SMEM
    pl::placement{"smem", "byte", {"address"}, [](int m, int n) {
        return std::array<int, 1>{export_offset(m, n) * 4};
      }, pl::coord{"m", 0, rows}, pl::coord{"n", 0, columns}},
    pl::strings{"smem_load_scalar", "smem_load_round"},
#else
    pl::placement{"reg", "scalar", {"participant", "position"}, [](int m, int n) {
        static const auto addresses = [] {
          std::array<int, rows * columns> out{};
          for (int thread = 0; thread < threads; ++thread) {
            auto coordinates = Configuration::partitioned_coordinates(thread);
            for (int i = 0; i < int(cute::size(coordinates)); ++i) {
              auto logical = coordinates(i);
              out[int(cute::get<0>(logical)) * columns + int(cute::get<1>(logical))] = thread * columns + i;
            }
          }
          return out;
        }();
        const int address = addresses[m * columns + n];
        return std::array<int, 2>{address / columns, address % columns};
      }, pl::coord{"m", 0, rows}, pl::coord{"n", 0, columns}},
    pl::strings{"reg_copy", "reg_copy_round", "shfl_xor", "smem_store_scalar", "smem_store_round"},
#endif
    pl::readiness{"tensor_ready", "import_done"}, pl::fixed_bindings{},
    pl::lends{
#ifdef PL_EXPORT_SMEM
              {"producer_storage", "smem", sizeof(Configuration::SharedStorage),
               alignof(Configuration::SharedStorage), "cta", "relocatable",
               "pipelink_tensor_producer_smem", std::nullopt, "tensor_producer",
               "tensor_work", "tensor_reclaimed", pl::sync{2, 1}},
#endif
              {"producer_rendezvous", "barrier", 1, 1, "cta", "relocatable",
               "pipelink_tensor_producer_barrier", std::nullopt, "tensor_producer",
               "tensor_work", "tensor_reclaimed", pl::sync{2, 1}}
    }, pl::calls{"tensor_produce", "tensor_release"});
}  // namespace PL_SOURCE_NAMESPACE
#endif
