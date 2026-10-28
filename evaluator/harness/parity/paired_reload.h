#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace pipelink::parity {

inline constexpr std::array<const char*, 8> orders{
    "ABBA", "BAAB", "BAAB", "ABBA", "BAAB", "ABBA", "ABBA", "BAAB"};
inline constexpr int repetitions = 10;
inline constexpr int warmup_repetitions = 8;

struct PairBlock {
    const char* order = nullptr;
    std::array<double, 4> milliseconds{};
    std::array<std::array<std::int64_t, 2>, 4> wall_intervals_ns{};
    double ratio_onetu_over_linked = 0;
};

struct PairResult {
    std::array<PairBlock, 8> blocks{};
    double ratio_onetu_over_linked = 0;
};

inline std::int64_t epoch_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

inline double median_four(double a, double b, double c, double d) {
    if (a > b) std::swap(a, b);
    if (c > d) std::swap(c, d);
    return (std::max(a, c) + std::min(b, d)) * 0.5;
}

// time_batch(candidate, n) times n complete calls of the candidate (true) or the
// reference (false) and returns total CUDA-event milliseconds, including the
// campaign's usual resets. The caller creates candidate then reference and owns
// the fixtures. reload() drains work and reloads reference then candidate,
// keeping the logical arms. There is no minimum batch duration or per-arm
// telemetry check.
template<class TimeBatch, class Reload>
PairResult time_pair(TimeBatch&& time_batch, Reload&& reload) {
    PairResult result;
    for (std::size_t block_index = 0; block_index < orders.size(); ++block_index) {
        if (block_index == 4) reload();
        if (block_index == 0 || block_index == 4) {
            time_batch(true, warmup_repetitions);
            time_batch(false, warmup_repetitions);
        }
        auto& block = result.blocks[block_index];
        block.order = orders[block_index];
        double candidate_ms = 0, reference_ms = 0;
        for (std::size_t batch = 0; batch < 4; ++batch) {
            const bool candidate = block.order[batch] == 'A';
            block.wall_intervals_ns[batch][0] = epoch_ns();
            const double elapsed = time_batch(candidate, repetitions);
            block.wall_intervals_ns[batch][1] = epoch_ns();
            if (!std::isfinite(elapsed) || elapsed <= 0)
                throw std::runtime_error("fixed-count timing requires positive finite elapsed time");
            block.milliseconds[batch] = elapsed;
            (candidate ? candidate_ms : reference_ms) += elapsed;
        }
        block.ratio_onetu_over_linked = reference_ms / candidate_ms;
    }
    const auto& b = result.blocks;
    const double first = median_four(b[0].ratio_onetu_over_linked, b[1].ratio_onetu_over_linked,
                                     b[2].ratio_onetu_over_linked, b[3].ratio_onetu_over_linked);
    const double second = median_four(b[4].ratio_onetu_over_linked, b[5].ratio_onetu_over_linked,
                                      b[6].ratio_onetu_over_linked, b[7].ratio_onetu_over_linked);
    result.ratio_onetu_over_linked = std::sqrt(first * second);
    return result;
}

inline void write_json(std::ostream& out, const PairResult& result) {
    const auto previous_precision = out.precision();
    out << std::setprecision(17)
        << "{\"protocol\":\"paired-reload-v2\",\"repetitions\":10,"
           "\"statistic\":\"geometric mean of per-order median speedups\","
           "\"status\":\"PASS\",\"clock_mutation\":false,\"blocks\":[";
    for (std::size_t i = 0; i < result.blocks.size(); ++i) {
        if (i) out << ',';
        const auto& block = result.blocks[i];
        out << "{\"order\":\"" << block.order << "\",\"milliseconds\":[";
        for (std::size_t j = 0; j < 4; ++j) {
            if (j) out << ',';
            out << block.milliseconds[j];
        }
        out << "],\"wall_intervals_ns\":[";
        for (std::size_t j = 0; j < 4; ++j) {
            if (j) out << ',';
            out << '[' << block.wall_intervals_ns[j][0] << ',' << block.wall_intervals_ns[j][1] << ']';
        }
        out << "],\"ratio_onetu_over_linked\":" << block.ratio_onetu_over_linked << '}';
    }
    out << "],\"ratio_onetu_over_linked\":" << result.ratio_onetu_over_linked << '}';
    out.precision(previous_precision);
}

}  // namespace pipelink::parity
