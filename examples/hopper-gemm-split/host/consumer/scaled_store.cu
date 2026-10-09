#include "epilogue.h"
#include <pipelink/stage.hpp>

extern "C" __device__ int pipelink_gemm_output();
extern "C" __device__ int pipelink_gemm_ready();
extern "C" __device__ int pipelink_gemm_drained();

extern "C" __device__ void gemm_epilogue_scaled_store_role(const void* opaque, unsigned) {
  auto const& view = *static_cast<plk_role_view_gemm_epilogue_scaled_store_epilogue const*>(opaque);
  auto const& params = *static_cast<ScaledStoreResidue const*>(view.residue);
  extern __shared__ __align__(1024) unsigned char gemm_smem[];
  const unsigned ready = __cvta_generic_to_shared(gemm_smem + 16 * pipelink_gemm_ready());
  // The consumer owns this declared protocol, not the producer's CuTe fragments.
  asm volatile("{ .reg .pred done; WAIT: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], 0; "
               "@!done bra WAIT; }" : : "r"(ready) : "memory");
  auto const* tile = reinterpret_cast<float const*>(gemm_smem + pipelink_gemm_output());
  for (int i = int(threadIdx.x) & 31; i < 128 * 128; i += 32) {
    const int row = int(blockIdx.x) * 128 + i / 128;
    const int col = int(blockIdx.y) * 128 + i % 128;
    params.output[size_t(row) * params.N + col] = params.alpha * tile[i];
  }
  const unsigned drained = __cvta_generic_to_shared(gemm_smem + 16 * pipelink_gemm_drained());
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(drained) : "memory");
}

PL_STAGE("gemm.epilogue.scaled-store", "1.0.0");
PL_NO_PARAMS();
PL_ROLE(epilogue, gemm_epilogue_scaled_store_role, 1);
PL_CLAIM_EX(pl::claim("pipelink_gemm_warp_epilogue", "warp")
    .extent(1).align(1).at(9).roles({"epilogue"}));
PL_CLAIM("pipelink_gemm_output", "smem", 128 * 128 * 4, 128, "kernel");
PL_CLAIM("pipelink_gemm_ready", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_gemm_drained", "mbarrier", 1, 1, "kernel");
PL_ENDPOINT(accumulator, pl::endpoint("accumulator", pl::consume, "epilogue")
    .space("smem").scope("cta").proxy("generic").tile("f32", {128, 128})
    .layout(pl::affine{{128, 128}, {128, 1}, 32})
    .protocol(pl::mbarrier{
        pl::side{"pipelink_gemm_ready", {"mma", 256}, 0},
        pl::side{"pipelink_gemm_drained", {"epilogue", 32}, 0}, {"load", 0, "dispatch"}})
    .ring(pl::fixed(1), 128 * 128 * 4)
    .bind("base", "pipelink_gemm_output").bind("full", "pipelink_gemm_ready")
    .bind("empty", "pipelink_gemm_drained"));

// Only local records are annotated; pipec supplies the neutral C ABI inventory.
PL_HOST_GROUP("gemm.user", "gemm.user.params", "user",
              plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_ABI_TYPE(100, ScaledStoreResidue, output, M, N, alpha);
PL_HOST_ABI_TYPE(101, plk_role_view_gemm_epilogue_scaled_store_epilogue, residue, scalars);
PL_HOST_RESIDUE(2, "gemm.user.residue", ScaledStoreResidue, 100, true,
                pl::host::roles({"gemm.epilogue.scaled-store:epilogue"}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"gemm.epilogue.scaled-store:epilogue"}))
    .field(pl::host::scalar(1, "gemm.M", "i64").caller("M"))
    .field(pl::host::scalar(2, "gemm.N", "i64").caller("N"))
    .field(pl::host::scalar(4, "gemm.alpha", "f64").caller("alpha")));
PL_HOST_CHANNEL(1, float, pl::host::channel("gemm.output", "f32")
    .rank(2).write().caller("output"));
PL_HOST_SCHEDULER(pl::host::scheduler("gemm.owner"));
PL_HOST_TARGET();
