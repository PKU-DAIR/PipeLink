#pragma once
#include "params.h"
#include <cutlass/arch/barrier.h>
#include <pipelink/stage.hpp>

namespace hopper_gemm {
inline constexpr int TileM = 128;
inline constexpr int TileN = 128;
inline constexpr int TileK = 64;
inline constexpr int OperandBytes = TileM * TileK * 2;
inline constexpr int SlotBytes = OperandBytes * 2;
inline constexpr int OutputBytes = TileM * TileN * 4;
}

extern "C" __device__ int pipelink_gemm_depth();
extern "C" __device__ int pipelink_gemm_operands();
extern "C" __device__ int pipelink_gemm_full();
extern "C" __device__ int pipelink_gemm_empty();
extern "C" __device__ int pipelink_gemm_output();
extern "C" __device__ int pipelink_gemm_ready();
extern "C" __device__ int pipelink_gemm_drained();
extern "C" __device__ int pipelink_gemm_warp_mma();

namespace hopper_gemm {
// PipeLink's allocated mbarrier slots occupy 16 bytes, not sizeof(uint64_t).
__device__ __forceinline__ uint64_t* barrier(unsigned char* smem, int base, int slot = 0) {
  return reinterpret_cast<uint64_t*>(smem + 16 * (base + slot));
}
}

// Both owners state the same physical facts. These are shared vocabulary, not
// generated transport or arithmetic. Extents shrink when depth is rebound.
#define GEMM_OPERAND_CLAIMS() \
  PL_CLAIM_EX(pl::claim("pipelink_gemm_operands", "smem") \
      .extent(0, {{"pipelink_gemm_depth", hopper_gemm::SlotBytes}}).align(1024)); \
  PL_CLAIM_EX(pl::claim("pipelink_gemm_full", "mbarrier") \
      .extent(0, {{"pipelink_gemm_depth", 1}}).align(1)); \
  PL_CLAIM_EX(pl::claim("pipelink_gemm_empty", "mbarrier") \
      .extent(0, {{"pipelink_gemm_depth", 1}}).align(1))

#define GEMM_OPERANDS(direction, role) \
  PL_ENDPOINT(operands, pl::endpoint("operands", direction, role) \
      .space("smem").scope("cta").proxy("async") \
      .tile("bf16", {2, 128, 64}) \
      .layout(pl::affine{{2, 128, 64}, {8192, 64, 1}, 16}.swizzle(3, 3, 3)) \
      .protocol(pl::mbarrier{ \
          pl::side{"pipelink_gemm_full", {"load", 1}, hopper_gemm::SlotBytes}, \
          pl::side{"pipelink_gemm_empty", {"mma", 256}, 0}, \
          {"load", 0, "dispatch"}}) \
      .ring(pl::handle("pipelink_gemm_depth"), hopper_gemm::SlotBytes) \
      .bind("base", "pipelink_gemm_operands") \
      .bind("full", "pipelink_gemm_full").bind("empty", "pipelink_gemm_empty"))

#define GEMM_OUTPUT_CLAIMS() \
  PL_CLAIM("pipelink_gemm_output", "smem", hopper_gemm::OutputBytes, 128, "kernel"); \
  PL_CLAIM("pipelink_gemm_ready", "mbarrier", 1, 1, "kernel"); \
  PL_CLAIM("pipelink_gemm_drained", "mbarrier", 1, 1, "kernel")

// One CTA uses this depth-one channel once, with the ordinary full/empty protocol.
// All 256 MMA threads publish their own stores before their ready arrival.
#define GEMM_OUTPUT(direction, role) \
  PL_ENDPOINT(accumulator, pl::endpoint("accumulator", direction, role) \
      .space("smem").scope("cta").proxy("generic").tile("f32", {128, 128}) \
      .layout(pl::affine{{128, 128}, {128, 1}, 32}) \
      .protocol(pl::mbarrier{ \
          pl::side{"pipelink_gemm_ready", {"mma", 256}, 0}, \
          pl::side{"pipelink_gemm_drained", {"epilogue", 32}, 0}, \
          {"load", 0, "dispatch"}}) \
      .ring(pl::fixed(1), hopper_gemm::OutputBytes) \
      .bind("base", "pipelink_gemm_output").bind("full", "pipelink_gemm_ready") \
      .bind("empty", "pipelink_gemm_drained"))
