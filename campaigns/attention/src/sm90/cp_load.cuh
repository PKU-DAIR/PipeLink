// Zigzag context-parallel load (attention owner): FA3 tile scheduling and
// publications, with diagonal-first local-before-remote K/V order, plus
//  (1) before the first K/V TMA of each remote 128-token block, a system-scope
//      acquire of that chunk's counter (launch-generation target gen+1) and the generic->async
//      proxy fence (peers wrote the payload with generic stores; push publication);
//  (2) after the last issue, a non-consuming wait on the newest full phase of
//      every K/V slot it filled, then the CTA-local `gather_reads_done` arrival
//      that lets kv-gather take its done ticket (CTA read-drain).
// Adapted from role_load.cuh (FlashAttention-3 hopper mainloop/tile scheduler;
// BSD-3-Clause, see role_load.cuh for the upstream notice).
#pragma once

#include "role_load.cuh"
#include "cp_params.cuh"

namespace pipelink::attention::sm90 {

template <class Config>
CUTLASS_DEVICE void run_cp_load(
    CpLoadParams<Config::causal> const& cp, SplitContext<Config>& c) {
  using namespace cute;
  using Mainloop = typename Config::Mainloop;
  using Scheduler = typename Config::Scheduler;
  using PipelineQ = typename Config::PipelineQ;
  using PipelineKV = typename Config::PipelineKV;
  using TileShape = typename Config::TileShape_MNK;
  static_assert(size(typename Config::ClusterShape{}) == 1);
  static_assert(Config::kBlockM == 128 && Mainloop::kHeadDim == 128);
  static_assert(Config::kBlockN == kCpChunkTokens, "one counter per K/V block");
  static_assert(Mainloop::Use_TMA_Q && Mainloop::Use_TMA_KV);
  auto const& params = cp.load;

  c.initialize();
  bool const elected = threadIdx.x % 32 == 0;
  if (!elected) return;
  // Device-resident launch generation: launches this rank completed; this launch's pushes reach gen+1.
  uint32_t const target = *reinterpret_cast<uint32_t const volatile*>(cp.generation) + 1u;
  uint32_t const* ready = cp.ready + pipelink_cp_kv_ready_slot();
  prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
  prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
  prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());

  auto q_write = cutlass::make_producer_start_state<PipelineQ>();
  static_assert(PipelineKV::Stages == 2);
  uint32_t kv_count = 0;
  auto kv_state = [](uint32_t count) {
    return typename PipelineKV::PipelineState{
        int(count & 1u), ((count >> 1) & 1u) ^ 1u, count};
  };
  uint32_t work_write = 0;
  // Only this lane accesses the CTA's exact acquired-block set.
  extern __shared__ char storage[];
  auto* acquired = reinterpret_cast<uint32_t*>(storage + pipelink_cp_acquired_blocks());
  int ready_batch = -1;

  Tensor sQ = make_tensor(make_smem_ptr(c.shared.tensors.smem_q.data()),
                          typename Mainloop::SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(c.shared.tensors.smem_k.data()),
                          typename Mainloop::SmemLayoutK{});
  Tensor sVt = make_tensor(make_smem_ptr(c.shared.tensors.smem_v.data()),
                           typename Mainloop::SmemLayoutVt{});
  auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
  auto block_tma_K = params.tma_load_K.get_slice(_0{});
  auto block_tma_V = params.tma_load_V.get_slice(_0{});
  Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
  Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK));
  Tensor tVsVt = group_modes<0, 3>(block_tma_V.partition_D(sVt));

  cutlass::arch::wait_on_dependent_grids();
  typename Scheduler::WorkTileInfo tile{int(blockIdx.x)};
  while (tile.is_valid(params.scheduler)) {
    auto const coord = tile.get_block_coord(params.scheduler);
    int const q_block = get<0>(coord);
    int const q_head = get<1>(coord);
    int const batch = get<2>(coord);
    int const local_q = q_block * Config::kBlockM;
    int const chunk_tokens = cp.chunk_blocks * Config::kBlockN;
    bool const late = local_q >= chunk_tokens;
    int const chunk_first = late ? cp.local_high_block : cp.local_low_block;
    int const q_offset = chunk_first * Config::kBlockN + local_q % chunk_tokens;
    int const seqlen_k = (chunk_first + cp.chunk_blocks) * Config::kBlockN;
    bool const causal = true;
    int const n_max = q_offset / Config::kBlockN + 1;
    int const steps = n_max;
    // Diagonal and the rest of this chunk, then the early local chunk,
    // then remote blocks descending. Every causal block occurs exactly once.
    int const own_steps = n_max - chunk_first;
    auto scheduled_block = [&](int step) {
      if (step < own_steps) return n_max - 1 - step;
      step -= own_steps;
      if (late) {
        if (step < cp.chunk_blocks)
          return cp.local_low_block + cp.chunk_blocks - 1 - step;
        step -= cp.chunk_blocks;
      }
      int block = chunk_first - 1 - step;
      if (late && block < cp.local_low_block + cp.chunk_blocks)
        block -= cp.chunk_blocks;
      return block;
    };
    publish_load_work(c, WorkDesc{
        q_block, batch, q_head, steps, params.seqlen_q, seqlen_k,
        q_offset, 0,
        WorkFlags::valid | WorkFlags::apply_sink |
            (steps == 0 ? WorkFlags::empty_output : 0u) |
            (params.needs_score_mod ? WorkFlags::needs_score_mod : 0u),
        1u, params.window_size, params.score_param, params.score_param_inv}, work_write);

    int next_tile = tile.tile_idx + int(gridDim.x);
    auto prefetch_next = [&] {
      if constexpr (Config::causal) {
        next_tile = atomicAdd(params.scheduler.tile_count_semaphore, 1) + int(gridDim.x);
      }
    };

    if (steps != 0) {
      if (batch != ready_batch) {
        ready_batch = batch;
        for (int word = 0; word < (cp.blocks_per_batch + 31) / 32; ++word)
          acquired[word] = 0;
      }
      uint32_t const* batch_ready = ready + batch * cp.blocks_per_batch;
      // K(n) is issued before V(n), so one wait covers the whole block.
      auto await_block = [&](int n_block) {
        if ((n_block >= cp.local_low_block && n_block < cp.local_low_block + cp.chunk_blocks) ||
            (n_block >= cp.local_high_block && n_block < cp.local_high_block + cp.chunk_blocks))
          return;
        uint32_t const bit = 1u << (n_block & 31);
        if (acquired[n_block >> 5] & bit) return;
        uint32_t value;
        do {
          asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                       : "=r"(value) : "l"(batch_ready + n_block) : "memory");
        } while (int32_t(value - target) < 0);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        acquired[n_block >> 5] |= bit;
      };
      int const kv_head = params.qhead_per_khead.divide(q_head);
      auto const shape_Q = make_shape(params.seqlen_q, Int<128>{},
                                     get<2>(params.shape_Q), get<3>(params.shape_Q));
      auto const shape_K = make_shape(get<0>(params.shape_K), Int<128>{},
                                     get<2>(params.shape_K), get<3>(params.shape_K));
      Tensor mQ = params.tma_load_Q.get_tma_tensor(shape_Q)(_, _, q_head, batch);
      Tensor mK = params.tma_load_K.get_tma_tensor(shape_K)(_, _, kv_head, batch);
      auto const shape_V = make_shape(Int<128>{}, get<0>(params.shape_K),
                                     get<2>(params.shape_K), get<3>(params.shape_K));
      Tensor mVt = params.tma_load_V.get_tma_tensor(shape_V)(_, _, kv_head, batch);
      Tensor gQ = local_tile(mQ, select<0, 2>(TileShape{}), make_coord(q_block, _0{}));
      Tensor gK = local_tile(mK, select<1, 2>(TileShape{}), make_coord(_, _0{}));
      Tensor gVt = local_tile(mVt, Shape<Int<128>, Int<Config::kBlockN>>{},
                              make_coord(_0{}, _));
      Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
      Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK));
      Tensor tVgVt = group_modes<0, 3>(block_tma_V.partition_S(gVt));
      prefetch(params.tma_load_Q, tQgQ);

      auto load_k = [&](int n_block, auto const& state) {
        await_block(n_block);
        c.k.producer_acquire(state);
        copy(params.tma_load_K.with(*c.k.producer_get_barrier(state), 0,
                                   TMA::CacheHintSm90::EVICT_LAST),
             tKgK(_, n_block), tKsK(_, state.index()));
      };
      auto load_v = [&](int n_block, auto const& state) {
        c.v.producer_acquire(state);
        copy(params.tma_load_V.with(*c.v.producer_get_barrier(state), 0,
                                   TMA::CacheHintSm90::EVICT_LAST),
             tVgVt(_, n_block), tVsVt(_, state.index()));
      };

      int n_block = n_max - 1;
      publish_load_kv(c, seqlen_k, q_offset, params.window_size,
                      n_block, causal, kv_count);
      load_k(n_block, kv_state(kv_count));
      c.q.producer_acquire(q_write);
      copy(params.tma_load_Q.with(*c.q.producer_get_barrier(q_write), 0,
                                 TMA::CacheHintSm90::EVICT_FIRST), tQgQ, tQsQ);
      c.o_drained.wait(q_write.phase());
      ++q_write;

      int n_previous = n_block;
      #pragma unroll 2
      for (int step = 1; step < steps; ++step) {
        n_block = scheduled_block(step);
        ++kv_count;
        publish_load_kv(c, seqlen_k, q_offset, params.window_size,
                        n_block, causal, kv_count);
        load_k(n_block, kv_state(kv_count));
        load_v(n_previous, kv_state(kv_count - 1));  // V trails K by one step.
        n_previous = n_block;
      }
      prefetch_next();
      load_v(n_previous, kv_state(kv_count));
      ++kv_count;
    } else {
      prefetch_next();
    }
    tile.tile_idx = __shfl_sync(1u, next_tile, 0);
  }

  // CTA read-drain: every gathered-buffer TMA read of this CTA has landed once the newest
  // fill of each K/V slot completed; older fills completed before that fill's
  // empty wait, so the parity test cannot alias. Non-consuming: core still waits.
  for (uint32_t stage = 0; stage < 2u && stage < kv_count; ++stage) {
    uint32_t const last = stage + (kv_count - 1u - stage) / 2u * 2u;
    uint32_t const parity = (last >> 1) & 1u;
    c.pipelines.k.full_barrier_[stage].wait(parity);
    c.pipelines.v.full_barrier_[stage].wait(parity);
  }
  cutlass::arch::ClusterBarrier::arrive(reinterpret_cast<uint64_t const*>(
      storage + 16 * pipelink_cp_gather_reads_done()));

  publish_load_work(c, WorkDesc{}, work_write);
  c.o_drained.wait(q_write.phase());
  c.q.producer_tail(q_write);
  c.k.producer_tail(kv_state(kv_count));
  c.v.producer_tail(kv_state(kv_count));
}

}  // namespace pipelink::attention::sm90
