#include "mk_abi.cuh"
#include "cute_attention.cuh"

// ThunderKittens is used only for the existing VM state/page/semaphore protocol.
// QK, online softmax, PV, and the internal split merge belong to the CuTe owner.
struct mk_op_attn_partial_cute {
  static constexpr int opcode = OPCODE_PartialAttention;
  static constexpr int pages = pipelink::cute_attention::warps;
  using Math = pipelink::cute_attention::Workspace;
  using Element = pipelink::cute_attention::Element;
  static_assert(sizeof(Math) <= mk_config::PAGE_SIZE);

  struct instruction {
    int layer, kv_head, partitions, partition;
    __device__ explicit instruction(mk_state& s)
        : layer(s.instruction()[1]), kv_head(s.instruction()[2]),
          partitions(s.instruction()[3]), partition(s.instruction()[4]) {}
  };

  __device__ static Math& workspace(mk_state& s, int warp) {
    return *reinterpret_cast<Math*>(s.pages[s.pid(warp)].ptr());
  }

  struct controller {
    __device__ static int release_lid(const mk_globals&, mk_config::instruction_t&, int& query) {
      return query < mk_config::NUM_PAGES - pages ? query + pages
                                                 : query - (mk_config::NUM_PAGES - pages);
    }
    __device__ static int init_semaphores(const mk_globals&, mk_state& s) {
#pragma unroll
      for (int warp = 0; warp < pages; ++warp)
        kittens::init_semaphore(s.semaphores()[warp], 32);
      return pages;
    }
  };

  struct loader {
    __device__ static void run(const mk_globals&, mk_state& s) {
      const int lane = threadIdx.x % 32;
      if (lane >= pages && lane < mk_config::NUM_PAGES) {
        const int page = s.pid(lane);
        s.wait_page_ready(page);
        s.finish_page(page, mk_config::NUM_CONSUMER_WARPS);
      }
    }
  };

  struct launcher {
    __device__ static void run(const mk_globals&, mk_state& s) {
      if (threadIdx.x % 32 == 0) {
        s.wait_tensor_ready();
        kittens::arrive(s.tensor_finished, mk_config::NUM_CONSUMER_WARPS);
      }
    }
  };

  struct consumer {
    __device__ static void run(const mk_globals& g, mk_state& s) {
      const int warp = threadIdx.x / 32;
      const int lane = threadIdx.x % 32;
      if (warp >= pages) return;
      const instruction inst(s);
      s.wait_page_ready(s.pid(warp));
      if (lane == 0) {
        if (warp == 0) s.record(megakernel::TEVENT_AT_GMEM_WAIT);
#pragma unroll
        for (int head = 0; head < 4; ++head) {
          while (*reinterpret_cast<volatile int*>(&g.Bar[{
                     inst.layer, OPCODE_RMS_QKV_MatVecRopeAppend - 1,
                     4 * inst.kv_head + head}]) < 4)
            __nanosleep(mk_config::GMEM_SPIN_LOOP_SLEEP_NANOS);
        }
#pragma unroll
        for (int kind = 0; kind < 2; ++kind) {
          while (*reinterpret_cast<volatile int*>(&g.Bar[{
                     inst.layer, OPCODE_RMS_QKV_MatVecRopeAppend - 1,
                     mk_globals::num_attention_heads + kind * mk_globals::num_kv_heads + inst.kv_head}]) < 4)
            __nanosleep(mk_config::GMEM_SPIN_LOOP_SLEEP_NANOS);
        }
        if (warp == 0) s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
      }
      __syncwarp();
      const int length = g.pos_id + 1;
      const int blocks = (length + mk_globals::kv_block_size - 1) / mk_globals::kv_block_size;
      const int blocks_per_partition = (blocks + inst.partitions - 1) / inst.partitions;
      const int begin = inst.partition * blocks_per_partition * mk_globals::kv_block_size;
      const int end = min(length, begin + blocks_per_partition * mk_globals::kv_block_size);
      const auto* q = reinterpret_cast<const Element*>(g.q_post_rope.raw_ptr) + inst.kv_head * 4 * 64;
      const size_t kv_base = (static_cast<size_t>(inst.layer) * g.k_cache.depth() *
                              mk_globals::num_kv_heads + inst.kv_head) * 64;
      const auto* k = reinterpret_cast<const Element*>(g.k_cache.raw_ptr) + kv_base;
      const auto* v = reinterpret_cast<const Element*>(g.v_cache.raw_ptr) + kv_base;
      pipelink::cute_attention::run_warp(workspace(s, warp), q, k, v,
          mk_globals::num_kv_heads * 64, begin, end, g.attn_scale, warp, lane);
      // Every producer lane publishes its own shared stores. Every storer lane
      // acquires all four completions before reading any workspace.
      kittens::arrive(s.semaphores()[warp]);
    }
  };

  struct storer {
    __device__ static void run(const mk_globals& g, mk_state& s) {
      const int lane = threadIdx.x % 32;
      const instruction inst(s);
      const Math* parts[pages];
#pragma unroll
      for (int warp = 0; warp < pages; ++warp) {
        kittens::wait(s.semaphores()[warp], 0);
        parts[warp] = &workspace(s, warp);
      }
      if (lane == 0) s.record(megakernel::TEVENT_OUTPUT_READY);
#pragma unroll
      for (int head = 0; head < 4; ++head) {
        float weights[pages];
        const float lse = pipelink::cute_attention::merge_weights(parts, head, weights);
        const int q_head = inst.kv_head * 4 + head;
#pragma unroll
        for (int column = lane; column < 64; column += 32) {
          const float value = pipelink::cute_attention::merge_output(parts, weights, head * 64 + column);
          if (g.skip_attn_reduction)
            reinterpret_cast<Element*>(g.attn_out.raw_ptr)[q_head * 64 + column] = Element(value);
          else
            g.attn_out_intermediates.raw_ptr[(q_head * g.attn_out_intermediates.rows() +
                                              inst.partition) * 64 + column] = value;
        }
        if (lane == 0 && !g.skip_attn_reduction)
          g.attn_lse_intermediates.raw_ptr[q_head * g.attn_lse_intermediates.cols() + inst.partition] = lse;
      }
      __threadfence();
      __syncwarp();
      if (lane < 4) {
        if (g.skip_attn_reduction)
          atomicAdd(&g.Bar[{inst.layer, OPCODE_AttentionReduction - 1, 0}], 1);
        else
          atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.kv_head * 4 + lane}], 1);
      }
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int warp = 0; warp < pages; ++warp)
          s.finish_page(s.pid(warp), mk_config::NUM_CONSUMER_WARPS);
      }
    }
  };
};

MK_DEFINE_OP_EXPORTS(attn_partial, 2, mk_op_attn_partial_cute,
    mk_op_attn_partial_cute::pages, mk_op_attn_partial_cute::pages, 0, 0, false)
