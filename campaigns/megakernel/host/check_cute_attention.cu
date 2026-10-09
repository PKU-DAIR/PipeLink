// Isolated mathematical oracle for the CuTe owner, including empty partitions.
#include "../port/src/ops/cute_attention.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace ca = pipelink::cute_attention;
using E = ca::Element;

static void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
    std::exit(1);
  }
}

__global__ void attention(const E* q, const E* k, const E* v,
                          float* out, float* lse, int length, int partitions) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto* work = reinterpret_cast<ca::Workspace*>(storage);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int head = blockIdx.x / partitions, partition = blockIdx.x % partitions;
  const int blocks = (length + 15) / 16;
  const int width = ((blocks + partitions - 1) / partitions) * 16;
  const int begin = partition * width, end = min(length, begin + width);
  ca::run_warp(work[warp], q + head * 4 * 64, k + head * 64, v + head * 64,
               8 * 64, begin, end, 0.125f, warp, lane);
  __syncthreads();
  if (warp == 0) {
    const ca::Workspace* parts[ca::warps];
#pragma unroll
    for (int w = 0; w < ca::warps; ++w) parts[w] = &work[w];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      float weights[ca::warps];
      const float logsum = ca::merge_weights(parts, h, weights);
#pragma unroll
      for (int d = lane; d < 64; d += 32)
        out[(blockIdx.x * 4 + h) * 64 + d] = ca::merge_output(parts, weights, h * 64 + d);
      if (lane == 0) lse[blockIdx.x * 4 + h] = logsum;
    }
  }
}

int main() {
  constexpr int max_length = 4098, max_partitions = 24;
  std::mt19937 rng(20261007);
  std::normal_distribution<float> normal(0, 0.7f);
  std::vector<E> q(32 * 64), k(max_length * 8 * 64), v(k.size());
  for (E& x : q) x = E(normal(rng));
  for (E& x : k) x = E(normal(rng));
  for (E& x : v) x = E(normal(rng));
  E *dq, *dk, *dv;
  float *dout, *dlse;
  check(cudaMalloc(&dq, q.size() * sizeof(E)));
  check(cudaMalloc(&dk, k.size() * sizeof(E)));
  check(cudaMalloc(&dv, v.size() * sizeof(E)));
  check(cudaMalloc(&dout, 8 * max_partitions * 4 * 64 * sizeof(float)));
  check(cudaMalloc(&dlse, 8 * max_partitions * 4 * sizeof(float)));
  check(cudaMemcpy(dq, q.data(), q.size() * sizeof(E), cudaMemcpyHostToDevice));
  check(cudaMemcpy(dk, k.data(), k.size() * sizeof(E), cudaMemcpyHostToDevice));
  check(cudaMemcpy(dv, v.data(), v.size() * sizeof(E), cudaMemcpyHostToDevice));
  check(cudaFuncSetAttribute(attention, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            ca::warps * sizeof(ca::Workspace)));
  int cases = 0, failures = 0;
  double worst = 0, worst_lse = 0;
  for (int length : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129, 1026, 4098}) {
    for (int partitions : {1, 2, 3, 8, 24}) {
      const int count = 8 * partitions * 4;
      std::vector<float> output(count * 64), logsum(count);
      check(cudaMemset(dout, 0xff, output.size() * sizeof(float)));
      check(cudaMemset(dlse, 0xff, logsum.size() * sizeof(float)));
      attention<<<8 * partitions, 128, ca::warps * sizeof(ca::Workspace)>>>(
          dq, dk, dv, dout, dlse, length, partitions);
      check(cudaGetLastError());
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(output.data(), dout, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
      check(cudaMemcpy(logsum.data(), dlse, logsum.size() * sizeof(float), cudaMemcpyDeviceToHost));
      const int width = ((((length + 15) / 16) + partitions - 1) / partitions) * 16;
      int case_failures = 0;
      for (int head = 0; head < 8; ++head) {
        for (int part = 0; part < partitions; ++part) {
          const int begin = part * width, end = std::min(length, begin + width);
          for (int h = 0; h < 4; ++h) {
            const int index = (head * partitions + part) * 4 + h;
            if (begin >= end) {
              case_failures += logsum[index] != -INFINITY;
              for (int d = 0; d < 64; ++d) case_failures += output[index * 64 + d] != 0;
              continue;
            }
            std::vector<double> scores(end - begin);
            double maximum = -INFINITY, denom = 0;
            for (int t = begin; t < end; ++t) {
              double dot = 0;
              for (int d = 0; d < 64; ++d)
                dot += double(float(q[(head * 4 + h) * 64 + d])) *
                       double(float(k[(t * 8 + head) * 64 + d]));
              maximum = std::max(maximum, scores[t - begin] = dot * 0.125);
            }
            for (double& score : scores) denom += score = std::exp(score - maximum);
            double expected[64]{}, rms = 0;
            for (int d = 0; d < 64; ++d) {
              for (int t = begin; t < end; ++t)
                expected[d] += scores[t - begin] * double(float(v[(t * 8 + head) * 64 + d]));
              expected[d] /= denom;
              rms += expected[d] * expected[d];
            }
            rms = std::sqrt(rms / 64);
            for (int d = 0; d < 64; ++d) {
              const double error = std::abs(output[index * 64 + d] - expected[d]);
              worst = std::max(worst, error);
              case_failures += !std::isfinite(output[index * 64 + d]) ||
                               error > 0.02 * (rms + std::abs(expected[d]));
            }
            const double expected_lse = (maximum + std::log(denom)) / std::log(2.0);
            const double lse_error = std::abs(logsum[index] - expected_lse);
            worst_lse = std::max(worst_lse, lse_error);
            case_failures += !std::isfinite(logsum[index]) || lse_error > 2e-4;
          }
        }
      }
      ++cases;
      failures += case_failures;
      std::printf("length=%d partitions=%d failures=%d\n", length, partitions, case_failures);
    }
  }
  std::printf("cases=%d failures=%d max_abs=%.9g max_lse_abs=%.9g\n", cases, failures, worst, worst_lse);
  check(cudaFree(dq)); check(cudaFree(dk)); check(cudaFree(dv));
  check(cudaFree(dout)); check(cudaFree(dlse));
  return failures != 0;
}
