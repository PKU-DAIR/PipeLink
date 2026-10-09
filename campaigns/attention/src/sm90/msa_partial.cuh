#pragma once
#include "msa_params.cuh"
#include "split_context.cuh"

namespace pipelink::attention::sm90 {

template <class Config>
CUTLASS_DEVICE void run_msa_partial(MsaPartialParams const& params, SplitContext<Config>& c) {
  using namespace cute;
  c.initialize();
  int const lane = threadIdx.x % 32;
  auto work_read = Config::WorkRing::consumer_start();
  uint32_t phase = 0;
  auto* source = c.shared.tensors.smem_v.data();
  while (true) {
    auto const& work = c.work_epilogue.consumer_wait(work_read);
    if (!(work.flags & WorkFlags::valid)) {
      c.work_epilogue.consumer_release(work_read);
      break;
    }
    typename Config::OutputRing::State const output_read{0, phase};
    typename Config::FinalStatsRing::State const stats_read{0, phase};
    c.corrected_o.consumer_wait(output_read);
    int const base = work.split_idx * 128;
    if (lane == 0) {
      // The frozen O layout has two SW128 64-column planes. Both stores
      // target one contiguous item-major tile; no query/rank scatter remains.
      SM90_TMA_STORE::copy(&params.tma_store_O, source, 0, base);
      SM90_TMA_STORE::copy(&params.tma_store_O, source + 8192, 64, base);
      tma_store_arrive();
    }
    auto const& stats = c.final_stats.consumer_wait(stats_read);
    for (int row = lane; row < 128; row += 32) {
      params.partial_m[base + row] = stats.m[row];
      params.partial_l[base + row] = stats.l[row];
    }
    __syncwarp();
    c.final_stats.consumer_release(stats_read);
    if (lane == 0) {
      // Completion protects V/O alias reuse, not just TMA issue.
      tma_store_wait<0>();
      c.corrected_o.consumer_release(output_read);
      c.o_drained.arrive();
    }
    c.work_epilogue.consumer_release(work_read);
    work_read.advance();
    phase ^= 1;
  }
}

}  // namespace pipelink::attention::sm90
