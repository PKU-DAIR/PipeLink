#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>
#include "cutlass/numeric_types.h"

namespace pipelink::attention::slot {
__host__ __device__ __forceinline__ uint32_t mul_hi(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
  return __umulhi(a, b);
#else
  return uint32_t((uint64_t(a) * b) >> 32);
#endif
}

__device__ __forceinline__ uint64_t round_shift_even(uint64_t value, int shift) {
  uint64_t quotient = value >> shift;
  uint64_t mask = (uint64_t(1) << shift) - 1;
  uint64_t remainder = value & mask;
  uint64_t halfway = uint64_t(1) << (shift - 1);
  return quotient + (remainder > halfway || (remainder == halfway && (quotient & 1)));
}

__device__ __forceinline__ __half double_to_half_rn(double value) {
  uint64_t bits = __double_as_longlong(value);
  uint16_t sign = uint16_t(bits >> 48) & 0x8000u;
  int exponent = int((bits >> 52) & 0x7ffu) - 1023;
  uint64_t significand = (uint64_t(1) << 52) | (bits & ((uint64_t(1) << 52) - 1));
  uint16_t out;
  if (exponent > 15) {
    out = sign | 0x7c00u;
  } else if (exponent >= -14) {
    uint64_t rounded = round_shift_even(significand, 42);
    if (rounded == 2048) {
      rounded = 1024;
      ++exponent;
    }
    out = exponent > 15 ? uint16_t(sign | 0x7c00u)
                        : uint16_t(sign | uint16_t((exponent + 15) << 10) | uint16_t(rounded & 0x3ffu));
  } else {
    int shift = 28 - exponent;
    uint64_t rounded = shift < 64 ? round_shift_even(significand, shift) : 0;
    out = rounded >= 1024 ? uint16_t(sign | 0x0400u)
                          : uint16_t(sign | uint16_t(rounded));
  }
  __half_raw raw;
  raw.x = out;
  return __half(raw);
}

__host__ __device__ uint32_t philox_word(uint64_t j, uint32_t numeric_case, uint32_t role) {
  uint64_t block = j >> 2;
  uint32_t v0 = uint32_t(block);
  uint32_t v1 = uint32_t(block >> 32);
  uint32_t v2 = numeric_case;
  uint32_t v3 = role;
  uint32_t k0 = 0x50495045u;
  uint32_t k1 = 0x00000002u;
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    uint32_t lo0 = 0xD2511F53u * v0;
    uint32_t hi0 = mul_hi(0xD2511F53u, v0);
    uint32_t lo1 = 0xCD9E8D57u * v2;
    uint32_t hi1 = mul_hi(0xCD9E8D57u, v2);
    uint32_t n0 = hi1 ^ v1 ^ k0;
    uint32_t n1 = lo1;
    uint32_t n2 = hi0 ^ v3 ^ k1;
    uint32_t n3 = lo0;
    v0 = n0; v1 = n1; v2 = n2; v3 = n3;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return (j & 3) == 0 ? v0 : (j & 3) == 1 ? v1 : (j & 3) == 2 ? v2 : v3;
}

// Fixture values are defined in f16 (Philox U[-2,2) rounded to nearest f16).
// fp8 configurations re-quantize those values: bf16 Q with RN and e4m3 K=V
// with RN satfinite at per-tensor scale 1.
template <class T>
__device__ __forceinline__ T from_f16(__half value) {
  if constexpr (std::is_same_v<T, cutlass::half_t>) return cutlass::half_t(value);
  else return T(__half2float(value));
}

template <class T>
__global__ void fill_philox(T* dst, uint64_t count, uint32_t numeric_case, uint32_t role) {
  uint64_t j = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= count) return;
  double u = (double(philox_word(j, numeric_case, role)) + 0.5) * 0x1p-32;
  dst[j] = from_f16<T>(double_to_half_rn(-2.0 + 4.0 * u));
}

// The sm90 bf16 family rounds Philox values directly, not via the f16 fixtures
// retained by the other families.
__global__ void fill_philox_bf16(cutlass::bfloat16_t* dst, uint64_t count,
                               uint32_t numeric_case, uint32_t role) {
  uint64_t j = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= count) return;
  double u = (double(philox_word(j, numeric_case, role)) + 0.5) * 0x1p-32;
  dst[j] = cutlass::bfloat16_t(__double2bfloat16(-2.0 + 4.0 * u));
}

// Family-2 Gate-B tolerances: evaluator/harness/parity/constants.py
// ATTENTION_ATOL and ATTENTION_RTOL. LSE uses the same finite relative predicate.
__device__ bool reference_matches(float actual, float reference) {
  if (reference == -INFINITY) return actual == -INFINITY;
  return isfinite(reference) && isfinite(actual) &&
         fabsf(actual - reference) <= 0.02f + 0.02f * fabsf(reference);
}

template <class T>
__global__ void compare_reference(T const* actual, float const* reference, uint64_t n, int* bad) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n && !reference_matches(float(actual[i]), reference[i])) atomicExch(bad, 1);
}

} // namespace pipelink::attention::slot
