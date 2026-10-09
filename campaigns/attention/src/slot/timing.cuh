#pragma once
#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <ostream>
#include "request.cuh"

// CUDA_CHECK is supplied by the slot, as in the original timing loop.
namespace pipelink::attention::slot {
struct Timing {
  double configured_batch_quantum_ms;
  double kBatchTargetFraction;
  int kInnerBatchCap;
  double single_launch_calibration_ms;
  int inner_batch_launches;
  uint64_t slot_start;
  uint64_t nominal_slot_end;
  bool last_batch_start_before_nominal_end;
  double maximum_batch_ms;
  uint64_t batch_count;
  double maximum_discard_batch_ms;
  uint64_t sample_start;
  uint64_t iterations;
  uint64_t sample_end;
  float elapsed_ms;
  uint64_t slot_end;
};

template <class Launch>
Timing time_slot(Request const& r, Launch&& launch) {
  double configured_batch_quantum_ms = r.batch_quantum_ms;
  constexpr double kBatchTargetFraction = 0.5;
  constexpr int kInnerBatchCap = 8;
  using SteadyClock = std::chrono::steady_clock;
  auto run_batch = [&](int count) {
    auto batch_start = SteadyClock::now();
    for (int i = 0; i < count; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    return std::chrono::duration<double, std::milli>(
        SteadyClock::now() - batch_start).count();
  };
  for (int i = 0; i < 3; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  double single_launch_calibration_ms = 0.0;
  for (int i = 0; i < 9; ++i) {
    single_launch_calibration_ms = std::max(
        single_launch_calibration_ms, run_batch(1));
  }
  if (!(single_launch_calibration_ms > 0.0)) {
    throw std::runtime_error("non-positive batch calibration");
  }
  double target_batch_ms =
      kBatchTargetFraction * configured_batch_quantum_ms;
  int inner_batch_launches = std::clamp(
      int(std::floor(target_batch_ms / single_launch_calibration_ms)),
      1, kInnerBatchCap);
  auto shrink_batch = [&](double observed_ms) {
    if (observed_ms <= target_batch_ms || inner_batch_launches <= 1) return;
    int scaled = int(std::floor(
        double(inner_batch_launches) * target_batch_ms / observed_ms));
    inner_batch_launches = std::max(
        1, std::min(inner_batch_launches - 1, scaled));
  };

  uint64_t slot_start = epoch_ns();
  uint64_t nominal_slot_end = slot_start + uint64_t(std::llround(r.slot_seconds * 1.0e9));
  uint64_t discard_end = slot_start + uint64_t(std::llround(r.discard_seconds * 1.0e9));
  bool last_batch_start_before_nominal_end = true;
  double maximum_batch_ms = 0.0;
  uint64_t batch_count = 0;
  while (true) {
    uint64_t batch_start = epoch_ns();
    if (batch_start >= discard_end) break;
    last_batch_start_before_nominal_end &= batch_start < nominal_slot_end;
    double batch_ms = run_batch(inner_batch_launches);
    maximum_batch_ms = std::max(maximum_batch_ms, batch_ms);
    shrink_batch(batch_ms);
  }
  double maximum_discard_batch_ms = maximum_batch_ms;
  maximum_batch_ms = 0.0;
  uint64_t sample_start = epoch_ns();
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(begin));
  uint64_t iterations = 0;
  while (true) {
    uint64_t batch_start = epoch_ns();
    if (batch_start >= nominal_slot_end) break;
    last_batch_start_before_nominal_end &= batch_start < nominal_slot_end;
    double batch_ms = run_batch(inner_batch_launches);
    iterations += uint64_t(inner_batch_launches);
    maximum_batch_ms = std::max(maximum_batch_ms, batch_ms);
    if (batch_ms > configured_batch_quantum_ms) {
      throw std::runtime_error(
          "observed launch batch exceeds configured shared quantum "
          "(observed_batch_ms=" + std::to_string(batch_ms) +
          ", quantum_ms=" + std::to_string(configured_batch_quantum_ms) +
          ", single_launch_calibration_ms=" +
          std::to_string(single_launch_calibration_ms) +
          ", resulting_batch_iterations=" +
          std::to_string(inner_batch_launches) + ")");
    }
    ++batch_count;
  }
  CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  uint64_t sample_end = epoch_ns();
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
  cudaEventDestroy(begin); cudaEventDestroy(end);
  uint64_t slot_end = epoch_ns();
  if (iterations == 0) throw std::runtime_error("slot ended before any scored launch batch");

  return {configured_batch_quantum_ms, kBatchTargetFraction, kInnerBatchCap, single_launch_calibration_ms, inner_batch_launches, slot_start, nominal_slot_end, last_batch_start_before_nominal_end, maximum_batch_ms, batch_count, maximum_discard_batch_ms, sample_start, iterations, sample_end, elapsed_ms, slot_end};
}

inline std::ostream& write_timing(std::ostream& out, Request const& r, Timing const& timing) {
  auto const& [configured_batch_quantum_ms, kBatchTargetFraction, kInnerBatchCap, single_launch_calibration_ms, inner_batch_launches, slot_start, nominal_slot_end, last_batch_start_before_nominal_end, maximum_batch_ms, batch_count, maximum_discard_batch_ms, sample_start, iterations, sample_end, elapsed_ms, slot_end] = timing;
  return out << "\"steady_cuda_event_milliseconds\":" << elapsed_ms << ","
            << "\"slot_start_ns\":" << slot_start << ",\"sampling_window_start_ns\":" << sample_start << ","
            << "\"sampling_window_end_ns\":" << sample_end
            << ",\"cuda_scoring_window_start_ns\":" << sample_start
            << ",\"cuda_scoring_window_end_ns\":" << sample_end
            << ",\"slot_end_ns\":" << slot_end << ","
            << "\"slot_duration_seconds\":" << r.slot_seconds << ","
            << "\"leading_discard_seconds\":" << r.discard_seconds << ","
            << "\"steady_duration_seconds\":" << r.steady_seconds << ","
            << "\"configured_batch_quantum_ms\":" << configured_batch_quantum_ms << ","
            << "\"last_batch_start_before_nominal_end\":"
            << (last_batch_start_before_nominal_end ? "true" : "false") << ","
            << "\"declared_slot_seconds\":" << r.slot_seconds << ","
            << "\"observed_slot_seconds\":" << double(slot_end - slot_start) / 1.0e9 << ","
            << "\"declared_scored_seconds\":" << r.steady_seconds << ","
            << "\"observed_scored_seconds\":" << double(sample_end - sample_start) / 1.0e9 << ","
            << "\"protocol_version\":" << json_quote(r.protocol_version) << ","
            << "\"nominal_slot_end_ns\":" << nominal_slot_end << ","
            << "\"inner_batch_launches\":" << inner_batch_launches
            << ",\"single_launch_calibration_ms\":" << single_launch_calibration_ms
            << ",\"batch_sizing_target_fraction\":" << kBatchTargetFraction
            << ",\"inner_batch_cap\":" << kInnerBatchCap
            << ",\"batch_count\":" << batch_count << ","
            << "\"maximum_discard_batch_ms\":" << maximum_discard_batch_ms << ","
            << "\"maximum_observed_batch_ms\":" << maximum_batch_ms << ","
            << "\"timing_method\":\"cuda-events-repeated-batches\","
            << "\"steady_timing_excludes\":[\"compilation\",\"fa4-jit\",\"resolver-solve-validation\","
               "\"stub-compile\",\"nvjitlink\",\"cubin-retrieval-write\",\"module-load\","
               "\"allocation\",\"initialization\",\"one-time-prepack\"],";
}

struct AbsoluteTiming {
  uint64_t n;
  std::vector<float> per_rep_ms;
  double median_ms;
};

class AbsoluteInterval {
  cudaEvent_t begin_, end_;
 public:
  AbsoluteInterval() {
    CUDA_CHECK(cudaEventCreate(&begin_));
    CUDA_CHECK(cudaEventCreate(&end_));
  }
  ~AbsoluteInterval() {
    cudaEventDestroy(begin_);
    cudaEventDestroy(end_);
  }
  AbsoluteInterval(AbsoluteInterval const&) = delete;
  AbsoluteInterval& operator=(AbsoluteInterval const&) = delete;

  template <class Launch>
  float run(uint64_t n, Launch&& launch) {
    CUDA_CHECK(cudaEventRecord(begin_));
    for (uint64_t i = 0; i < n; ++i) launch();
    CUDA_CHECK(cudaEventRecord(end_));
    CUDA_CHECK(cudaEventSynchronize(end_));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin_, end_));
    if (!(ms > 0.0f)) throw std::runtime_error("non-positive absolute timing interval");
    return ms;
  }
};

inline AbsoluteTiming absolute_summary(uint64_t n, std::vector<float> samples) {
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  double median = (double(sorted[4]) + sorted[5]) / (2.0 * n);
  return {n, std::move(samples), median};
}

template <class Launch>
AbsoluteTiming time_absolute(Launch&& launch) {
  for (int i = 0; i < 10; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  AbsoluteInterval interval;
  uint64_t n = 1;
  std::vector<float> samples;
  samples.reserve(10);
  // Size to 25 ms for margin. If any measured repetition falls below 20 ms,
  // resize and restart the set: every retained interval meets the contract.
  while (samples.size() < 10) {
    float ms = interval.run(n, launch);
    if (ms < 20.0f) {
      n = std::max(n + 1, uint64_t(std::ceil(double(n) * 25.0 / ms)));
      samples.clear();
    } else {
      samples.push_back(ms);
    }
  }
  return absolute_summary(n, std::move(samples));
}

// Each retained pair has two >=20 ms intervals. A resized interval restarts
// both sample sets, so candidate-first/reference-first order stays balanced.
template <class Candidate, class Reference>
std::array<AbsoluteTiming, 2> time_absolute_pair(Candidate&& candidate, Reference&& reference) {
  for (int i = 0; i < 10; ++i) {
    candidate();
    reference();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  AbsoluteInterval interval;
  std::array<uint64_t, 2> n{1, 1};
  std::array<std::vector<float>, 2> samples;
  for (auto& sample : samples) sample.reserve(10);
  while (samples[0].size() < 10) {
    std::array<float, 2> ms;
    if (samples[0].size() % 2 == 0) {
      ms[0] = interval.run(n[0], candidate);
      ms[1] = interval.run(n[1], reference);
    } else {
      ms[1] = interval.run(n[1], reference);
      ms[0] = interval.run(n[0], candidate);
    }
    if (ms[0] < 20.0f || ms[1] < 20.0f) {
      for (int i = 0; i < 2; ++i) {
        if (ms[i] < 20.0f)
          n[i] = std::max(n[i] + 1, uint64_t(std::ceil(double(n[i]) * 25.0 / ms[i])));
        samples[i].clear();
      }
    } else {
      for (int i = 0; i < 2; ++i) samples[i].push_back(ms[i]);
    }
  }
  return {absolute_summary(n[0], std::move(samples[0])),
          absolute_summary(n[1], std::move(samples[1]))};
}

inline std::ostream& write_absolute_timing(std::ostream& out, AbsoluteTiming const& timing) {
  out << "\"absolute_median_ms\":" << timing.median_ms
      << ",\"reps\":" << timing.per_rep_ms.size() << ",\"n\":" << timing.n
      << ",\"per_rep_ms\":[";
  for (size_t i = 0; i < timing.per_rep_ms.size(); ++i) {
    if (i) out << ",";
    out << timing.per_rep_ms[i];
  }
  return out << "],\"timing_method\":\"cuda-events-back-to-back\","
             << "\"iteration_definition\":\"full per-call launch sequence including resets\",";
}
} // namespace pipelink::attention::slot
