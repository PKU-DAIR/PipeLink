// Multi-device context-parallel (CP) slot for pipeline d128-sm90-c/cp.
//
// One process drives W ranks on visible devices 0..W-1 with lockstep launches
// (the launch-generation/release-wait caller obligation: every rank runs the same launch sequence).
// Rank r holds chunks r and 2W-1-r of C = T/(2W) tokens in local [early,late]
// Q/O/LSE order. Gathered K/V remains in full-sequence global order; kv-gather
// pushes exactly the remote chunks each rank's causal queries require.
//
// Actions:
//   gate      : >= 5 changing payloads, each launched as linked then control on
//               every rank (with skewed-rank starts), then graph replays; every
//               launch's O/LSE is checked against the fp32 boundary-row oracle on
//               the full sequence and linked/control must be bitwise equal.
//               Remote K/V regions and outputs are poisoned before each launch,
//               so stale or early reads cannot pass. Then, per mode (direct,
//               graph replay), >= 5 linked and >= 5 control launches are queued
//               with no host/global sync between them: a stream-ordered payload
//               change per launch, the last rank delayed 2 ms each launch, and
//               per-launch O/LSE copies; all are checked after one final sync.
//   sanitize  : two launches of --mode's image (run under compute-sanitizer).
//   parity    : linked/control all-rank timing in alternating balanced blocks.
//   placement : CP linked vs sequential peer-copy gather + `selected` (same
//               shards) vs single-GPU full-sequence `selected` on device 0.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "slot/request.cuh"
#include "slot/fixtures.cuh"
#include "sm90/family_params.cuh"
#include "sm90/cp_params.cuh"

#define CUDA_CHECK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e_)); } while (0)
#define CU_CHECK(expr) do { CUresult e_ = (expr); if (e_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorString(e_, &s_); throw std::runtime_error(std::string(#expr) + ": " + (s_ ? s_ : "driver error")); } } while (0)
#include "slot/timing.cuh"
#include "slot/family.cuh"

// Named, not anonymous: `using namespace sm90` brings cute's anonymous namespace
// in, which makes nvcc's host-stub `_NV_ANON_NAMESPACE` references ambiguous.
namespace cp_slot {
using namespace pipelink::attention::slot;
using namespace pipelink::attention::sm90;
using Element = cutlass::bfloat16_t;
static_assert(std::is_same_v<Element, typename StockKernel<true, 1>::Element>);

struct Cell {
  std::string id;
  int total, batch, heads_q, heads_kv;
  uint32_t numeric_case;
  float scale;
};

Cell resolve_cell(std::string const& path, std::string const& id) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open cells JSON: " + path);
  std::string raw{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  Json document = JsonParser(raw).parse();
  if (document.at("schema").string() != "pipelink-sm90-cells-v1")
    throw std::runtime_error("unsupported cells JSON schema");
  for (auto const& row : document.at("cells").array()) {
    if (row.at("id").string() != id) continue;
    Cell c{id, row.at("seqlen_q").integer(), row.at("batch").integer(),
           row.at("heads_q").integer(), row.at("heads_kv").integer(),
           uint32_t(row.at("numeric_case").integer()), float(row.at("scale").number())};
    if (row.at("mask").at("kind").string() != "causal" ||
        row.at("softcap").kind != Json::Kind::null || row.at("dtype").string() != "bf16" ||
        row.at("head_dim").integer() != 128 || row.at("seqlen_k").integer() != c.total ||
        c.batch <= 0 || c.heads_kv <= 0 || c.heads_q % c.heads_kv != 0 || !(c.scale > 0))
      throw std::runtime_error(id + ": CP requires a bf16 d128 causal GQA cell without softcap");
    return c;
  }
  throw std::runtime_error("unknown sm90 cell: " + id);
}

struct Options {
  std::string action, mode = "linked", link_json, linked_cubin, control_cubin;
  std::string selected_link_json, selected_cubin, cells_json, cell;
  int world = 0, payloads = 6, replays = 3, async_launches = 6;
};

Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&](char const* key) -> std::string const* {
      static std::string v;
      std::string prefix = std::string("--") + key + "=";
      if (a.rfind(prefix, 0) != 0) return nullptr;
      v = a.substr(prefix.size());
      return &v;
    };
    if (auto v = value("action")) o.action = *v;
    else if (auto v = value("mode")) o.mode = *v;
    else if (auto v = value("link-json")) o.link_json = *v;
    else if (auto v = value("linked-cubin")) o.linked_cubin = *v;
    else if (auto v = value("control-cubin")) o.control_cubin = *v;
    else if (auto v = value("selected-link-json")) o.selected_link_json = *v;
    else if (auto v = value("selected-cubin")) o.selected_cubin = *v;
    else if (auto v = value("cells-json")) o.cells_json = *v;
    else if (auto v = value("cell")) o.cell = *v;
    else if (auto v = value("world")) o.world = std::stoi(*v);
    else if (auto v = value("payloads")) o.payloads = std::stoi(*v);
    else if (auto v = value("replays")) o.replays = std::stoi(*v);
    else if (auto v = value("async-launches")) o.async_launches = std::stoi(*v);
    else throw std::runtime_error("unknown option: " + a);
  }
  if (o.action != "gate" && o.action != "sanitize" && o.action != "parity" && o.action != "placement")
    throw std::runtime_error("--action must be gate, sanitize, parity or placement");
  if (o.mode != "linked" && o.mode != "control") throw std::runtime_error("--mode must be linked or control");
  if (o.world < 2 || o.world > kCpMaxWorld) throw std::runtime_error("--world must be in [2, 8]");
  if (o.payloads < 5 || o.replays < 1 || o.async_launches < 5) throw std::runtime_error("gate needs >= 5 payloads, >= 1 replay and >= 5 async launches");
  if (o.link_json.empty() || o.linked_cubin.empty() || o.control_cubin.empty() ||
      o.cells_json.empty() || o.cell.empty())
    throw std::runtime_error("--link-json, --linked-cubin, --control-cubin, --cells-json, --cell are required");
  if (o.action == "placement" && (o.selected_link_json.empty() || o.selected_cubin.empty()))
    throw std::runtime_error("placement requires --selected-link-json and --selected-cubin");
  return o;
}

// Rows [dst_row0, dst_row0+rows) of dst [batch, dst_rows, row_elems] take the
// fixture values of logical rows [src_row0, ...) of a [batch, src_rows, row_elems]
// tensor: shards and gathered buffers hold exactly the full tensor's values.
__global__ void fill_rows(cutlass::bfloat16_t* dst, int batch, int rows, int row_elems,
                          uint64_t dst_rows, uint64_t dst_row0, uint64_t src_rows, uint64_t src_row0,
                          uint32_t numeric_case, uint32_t role) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t per_batch = uint64_t(rows) * row_elems;
  if (i >= per_batch * batch) return;
  uint64_t b = i / per_batch, rest = i % per_batch;
  uint64_t r = rest / row_elems, e = rest % row_elems;
  uint64_t j = (b * src_rows + src_row0 + r) * row_elems + e;
  double u = (double(philox_word(j, numeric_case, role)) + 0.5) * 0x1p-32;
  dst[(b * dst_rows + dst_row0 + r) * row_elems + e] = cutlass::bfloat16_t(__double2bfloat16(-2.0 + 4.0 * u));
}

// bf16 0xffff / f32 0xffffffff are NaN: a stale or early read cannot pass the oracle.
void poison_rows(Element* dst, int batch, uint64_t batch_rows, uint64_t rows, int row_elems, cudaStream_t s) {
  if (rows == 0) return;
  CUDA_CHECK(cudaMemset2DAsync(dst, batch_rows * row_elems * sizeof(Element), 0xff,
                               rows * row_elems * sizeof(Element), batch, s));
}

__global__ void spin_ns(uint64_t ns) {
  uint64_t start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); } while (now - start < ns);
}

uint32_t payload_case(Cell const& c, int payload) { return c.numeric_case + 7919u * uint32_t(payload); }

std::vector<int> boundary_rows(Cell const& c, int world) {
  int const chunk = c.total / (2 * world);
  std::vector<int> rows{0, 1, 127, 128, c.total / 2, c.total - 1};
  for (int r = 1; r < 2 * world; ++r)
    rows.insert(rows.end(), {r * chunk - 1, r * chunk, r * chunk + 1, r * chunk + 127, r * chunk + 128});
  rows.erase(std::remove_if(rows.begin(), rows.end(), [&](int row) { return row >= c.total; }), rows.end());
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

// fp32 boundary-row oracle for global `rows` over full-sequence reference tensors.
struct Oracle {
  Cell c;
  std::vector<int> rows;
  int sample_count;
  Buffer<int> samples, output_samples, bad;
  Buffer<float> ref_o, ref_lse, scratch;
  Oracle(Cell const& c, std::vector<int> rows_, int world = 0, int rank = 0)
      : c(c), rows(std::move(rows_)), sample_count(int(rows.size()) * c.batch * c.heads_q),
        samples(rows.size()), output_samples(rows.size()), bad(1), ref_o(uint64_t(sample_count) * 128),
        ref_lse(sample_count), scratch(uint64_t(sample_count) * c.total) {
    CUDA_CHECK(cudaMemcpy(samples.data, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<int> local_rows = rows;
    if (world) {
      int const chunk = c.total / (2 * world);
      for (int& row : local_rows)
        row = row / chunk == rank ? row - rank * chunk : chunk + row - (2 * world - 1 - rank) * chunk;
    }
    CUDA_CHECK(cudaMemcpy(output_samples.data, local_rows.data(), local_rows.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  void compute(Element const* q, Element const* k, Element const* v, cudaStream_t s) {
    spot_reference<<<sample_count, 256, 0, s>>>(q, k, v, samples.data, int(rows.size()), c.total,
        c.heads_q, c.heads_kv, 0, 0, false, nullptr, c.scale, 0.f, scratch.data, ref_o.data, ref_lse.data);
    CUDA_CHECK(cudaGetLastError());
  }
  // Returns bit 1 for O and bit 2 for LSE mismatches.
  int check(Element const* o, float const* lse, int row_offset, int out_rows, cudaStream_t s) {
    CUDA_CHECK(cudaMemsetAsync(bad.data, 0, sizeof(int), s));
    compare_spots<<<sample_count, 128, 0, s>>>(o, lse, ref_o.data, ref_lse.data, output_samples.data,
        int(rows.size()), row_offset, out_rows, c.heads_q, bad.data);
    CUDA_CHECK(cudaGetLastError());
    int failed = 0;
    CUDA_CHECK(cudaMemcpyAsync(&failed, bad.data, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return failed;
  }
};

enum class Image { linked, control, selected };

struct RankData {
  Buffer<Element> q, kg, vg, o, o_ctl;
  Buffer<float> lse, lse_ctl;
  Buffer<uint32_t> ready, released, gen;
  Buffer<int> resets, bad;  // resets: [scheduler semaphore, gather tickets, gather done]
  Buffer<Element> q_ref, k_ref, v_ref;
  Oracle oracle;
  std::unique_ptr<FamilyKernel> linked, control, selected;
  std::unique_ptr<PackedParams> linked_params, control_params, selected_params[2];
  std::unique_ptr<Buffer<float>> selected_lse[2];
  // Per-launch O/LSE captures of the asynchronous sequences: [linked, control].
  std::unique_ptr<Buffer<Element>> seq_o[2];
  std::unique_ptr<Buffer<float>> seq_lse[2];
  RankData(Cell const& c, int world, int rank, uint64_t ready_count)
      : q(uint64_t(c.batch) * (c.total / world) * c.heads_q * 128),
        kg(uint64_t(c.batch) * c.total * c.heads_kv * 128), vg(kg_count(c)),
        o(q_count(c, world)), o_ctl(q_count(c, world)),
        lse(uint64_t(c.batch) * c.heads_q * (c.total / world)), lse_ctl(uint64_t(c.batch) * c.heads_q * (c.total / world)),
        ready(ready_count), released(kCpMaxWorld), gen(1), resets(3), bad(1),
        q_ref(uint64_t(c.batch) * c.total * c.heads_q * 128), k_ref(kg_count(c)), v_ref(kg_count(c)),
        oracle(c, shard_rows(c, world, rank), world, rank) {}
  static uint64_t kg_count(Cell const& c) { return uint64_t(c.batch) * c.total * c.heads_kv * 128; }
  static uint64_t q_count(Cell const& c, int world) { return uint64_t(c.batch) * (c.total / world) * c.heads_q * 128; }
  static std::vector<int> shard_rows(Cell const& c, int world, int rank) {
    int const chunk = c.total / (2 * world);
    std::vector<int> rows;
    for (int row : boundary_rows(c, world))
      if (row / chunk == rank || row / chunk == 2 * world - 1 - rank) rows.push_back(row);
    return rows;
  }
};

struct Rank {
  int device, rank;
  cudaStream_t stream = nullptr;
  std::unique_ptr<RankData> d;
  cudaGraphExec_t graphs[2] = {nullptr, nullptr};
  ~Rank() {
    cudaSetDevice(device);
    for (auto g : graphs) if (g) cudaGraphExecDestroy(g);
    d.reset();
    if (stream) cudaStreamDestroy(stream);
  }
};

class Cp {
 public:
  Options const& opts;
  Cell c;
  int world, shard, chunk, grid;
  LinkMetadata meta, selected_meta;
  std::vector<std::unique_ptr<Rank>> ranks;

  Cp(Options const& o, Cell const& cell)
      : opts(o), c(cell), world(o.world), shard(cell.total / o.world), chunk(shard / 2) {
    if (c.total <= 0 || c.total % (2 * world * 128) != 0)
      throw std::runtime_error("CP requires T divisible by 2*world*128");
    if (uint64_t(c.batch) * (c.total / 128) > uint64_t(kCpMaxChunks))
      throw std::runtime_error("CP chunk counters exceed the claimed gmem_counter extent");
    meta = load_link_metadata(opts.link_json,
        {"cp-load", "core", "epilogue", "kv-gather", "mask-default", "score-identity"});
    if (opts.action == "placement")
      selected_meta = load_link_metadata(opts.selected_link_json,
          {"tma-load", "core", "epilogue", "empty", "mask-default", "score-identity"});
    int const ready_slot = meta.bindings.at("pipelink_cp_kv_ready_slot").array().at(0).integer();
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < world) throw std::runtime_error("CP needs one visible device per rank");
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      cudaDeviceProp prop{};
      CUDA_CHECK(cudaGetDeviceProperties(&prop, r));
      if (prop.major != 9 || prop.minor != 0) throw std::runtime_error("CP slot requires sm90 GPUs");
      if (r == 0) grid = std::min(c.batch * (shard / 128) * c.heads_q, prop.multiProcessorCount);
      for (int p = 0; p < world; ++p) {
        if (p == r) continue;
        int can = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can, r, p));
        if (!can) throw std::runtime_error("CP requires peer access between all ranks");
        cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else CUDA_CHECK(e);
      }
      auto rank = std::make_unique<Rank>();
      rank->device = r; rank->rank = r;
      CUDA_CHECK(cudaStreamCreateWithFlags(&rank->stream, cudaStreamNonBlocking));
      rank->d = std::make_unique<RankData>(c, world, r, uint64_t(ready_slot) + kCpMaxChunks);
      // Launch-generation/release-wait state is zeroed once; counters and generations are never reset.
      CUDA_CHECK(cudaMemset(rank->d->ready.data, 0, (uint64_t(ready_slot) + kCpMaxChunks) * 4));
      CUDA_CHECK(cudaMemset(rank->d->released.data, 0, kCpMaxWorld * 4));
      CUDA_CHECK(cudaMemset(rank->d->gen.data, 0, 4));
      ranks.push_back(std::move(rank));
    }
    for (auto& rank : ranks) build_images(*rank);
    sync_all();
  }

  Flash_fwd_params arguments(Rank& rank, Element* o, float* lse, int seqlen_q, int seqlen_k,
                             Element const* q, Element* k, Element* v, int kv_rows) {
    Flash_fwd_params p{};
    p.q_ptr = const_cast<Element*>(q); p.k_ptr = k; p.v_ptr = v;
    p.o_ptr = o; p.softmax_lse_ptr = lse;
    p.b = c.batch; p.b_k = c.batch; p.h = c.heads_q; p.h_k = c.heads_kv;
    p.seqlen_q = seqlen_q; p.seqlen_k = seqlen_k;
    p.d = p.dv = 128;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128;
    p.q_row_stride = p.o_row_stride = c.heads_q * 128;
    p.k_row_stride = p.v_row_stride = c.heads_kv * 128;
    p.q_batch_stride = p.o_batch_stride = int64_t(p.q_row_stride) * seqlen_q;
    p.k_batch_stride = p.v_batch_stride = int64_t(p.k_row_stride) * kv_rows;
    p.v_dim_stride = 1;
    p.scale_softmax = c.scale;
    p.num_splits = 1; p.is_causal = true; p.is_bf16 = true;
    p.window_size_left = seqlen_k - 1;
    p.window_size_right = 0;
    p.tile_count_semaphore = rank.d->resets.data;
    p.arch = 90;
    p.num_sm = std::min(c.batch * cute::ceil_div(seqlen_q, 128) * c.heads_q, grid_limit(rank));
    return p;
  }
  int grid_limit(Rank& rank) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, rank.device));
    return prop.multiProcessorCount;
  }

  KvGatherParams gather_params(Rank& rank) {
    KvGatherParams g{};
    for (int p = 0; p < world; ++p) {
      g.k_gather[p] = reinterpret_cast<uint64_t>(ranks[p]->d->kg.data);
      g.v_gather[p] = reinterpret_cast<uint64_t>(ranks[p]->d->vg.data);
      g.ready[p] = reinterpret_cast<uint64_t>(ranks[p]->d->ready.data);
      g.released[p] = reinterpret_cast<uint64_t>(ranks[p]->d->released.data);
    }
    g.generation = rank.d->gen.data;
    g.queue = reinterpret_cast<uint32_t*>(rank.d->resets.data + 1);
    g.rank = rank.rank; g.world = world; g.batch = c.batch;
    g.shard_blocks = shard / 128; g.blocks_per_batch = c.total / 128;
    g.row_bytes = c.heads_kv * 128 * int(sizeof(Element));
    g.batch_bytes = int64_t(c.total) * g.row_bytes;
    return g;
  }

  void build_images(Rank& rank) {
    CUDA_CHECK(cudaSetDevice(rank.device));
    auto& d = *rank.d;
    int const r = rank.rank;
    auto pack_cp = [&](Element* o, float* lse) {
      auto args = arguments(rank, o, lse, shard, c.total, d.q.data, d.kg.data, d.vg.data, c.total);
      auto fp = make_family_params<true>(args, rank.device, 0);
      auto packed = std::make_unique<PackedParams>(meta);
      packed->put(meta.params[0], CpLoadParams<true>{fp.load, d.ready.data, d.gen.data,
          r * chunk / 128, (2 * world - 1 - r) * chunk / 128, chunk / 128, c.total / 128});
      packed->put(meta.params[1], fp.core);
      packed->put(meta.params[2], fp.epilogue);
      packed->put(meta.params[3], gather_params(rank));
      for (int i = 4; i < 6; ++i)
        if (meta.params[i].size != 0) throw std::runtime_error("callees must have no parameter blob");
      return packed;
    };
    d.linked_params = pack_cp(d.o.data, d.lse.data);
    d.control_params = pack_cp(d.o_ctl.data, d.lse_ctl.data);
    d.linked = std::make_unique<FamilyKernel>(opts.linked_cubin, meta, grid, d.resets.data, 3 * sizeof(int));
    d.control = std::make_unique<FamilyKernel>(opts.control_cubin, meta, grid, d.resets.data, 3 * sizeof(int));
    if (opts.action == "placement") {
      // Two unchanged selected launches, one per Q segment, over exactly-needed
      // gathered K/V. Q/O keep shard batch strides; LSE needs segment storage.
      for (int segment = 0; segment < 2; ++segment) {
        int const global_chunk = segment == 0 ? r : 2 * world - 1 - r;
        uint64_t const offset = uint64_t(segment) * chunk * c.heads_q * 128;
        d.selected_lse[segment] = std::make_unique<Buffer<float>>(uint64_t(c.batch) * c.heads_q * chunk);
        auto args = arguments(rank, d.o.data + offset, d.selected_lse[segment]->data,
            chunk, (global_chunk + 1) * chunk, d.q.data + offset, d.kg.data, d.vg.data, c.total);
        args.q_batch_stride = args.o_batch_stride = int64_t(shard) * c.heads_q * 128;
        auto fp = make_family_params<true>(args, rank.device, 0);
        d.selected_params[segment] = std::make_unique<PackedParams>(selected_meta);
        d.selected_params[segment]->put(selected_meta.params[0], fp.load);
        d.selected_params[segment]->put(selected_meta.params[1], fp.core);
        d.selected_params[segment]->put(selected_meta.params[2], fp.epilogue);
        d.selected_params[segment]->put(selected_meta.params[3], fp.empty);
      }
      int const selected_grid = std::min(c.batch * (chunk / 128) * c.heads_q, grid_limit(rank));
      d.selected = std::make_unique<FamilyKernel>(opts.selected_cubin, selected_meta, selected_grid, d.resets.data, sizeof(int));
    }
  }

  void sync_all() {
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      CUDA_CHECK(cudaStreamSynchronize(rank->stream));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  // Stream-ordered fill of this rank's Q shard and own K/V shard with `payload`.
  void fill_shards(Rank& rank, int payload) {
    uint32_t const nc = payload_case(c, payload);
    int const kv_elems = c.heads_kv * 128, q_elems = c.heads_q * 128;
    auto blocks = [](uint64_t n) { return int((n + 255) / 256); };
    auto& d = *rank.d;
    auto s = rank.stream;
    for (int segment = 0; segment < 2; ++segment) {
      uint64_t const global_row = uint64_t(segment == 0 ? rank.rank : 2 * world - 1 - rank.rank) * chunk;
      fill_rows<<<blocks(uint64_t(c.batch) * chunk * q_elems), 256, 0, s>>>(
          d.q.data, c.batch, chunk, q_elems, shard, uint64_t(segment) * chunk, c.total, global_row, nc, 1);
      fill_rows<<<blocks(uint64_t(c.batch) * chunk * kv_elems), 256, 0, s>>>(
          d.kg.data, c.batch, chunk, kv_elems, c.total, global_row, c.total, global_row, nc, 2);
      fill_rows<<<blocks(uint64_t(c.batch) * chunk * kv_elems), 256, 0, s>>>(
          d.vg.data, c.batch, chunk, kv_elems, c.total, global_row, c.total, global_row, nc, 3);
    }
    CUDA_CHECK(cudaGetLastError());
  }

  // Full-sequence reference tensors for `payload` and this rank's oracle rows.
  void references(Rank& rank, int payload) {
    uint32_t const nc = payload_case(c, payload);
    int const kv_elems = c.heads_kv * 128, q_elems = c.heads_q * 128;
    auto blocks = [](uint64_t n) { return int((n + 255) / 256); };
    auto& d = *rank.d;
    auto s = rank.stream;
    fill_rows<<<blocks(uint64_t(c.batch) * c.total * q_elems), 256, 0, s>>>(
        d.q_ref.data, c.batch, c.total, q_elems, c.total, 0, c.total, 0, nc, 1);
    fill_rows<<<blocks(uint64_t(c.batch) * c.total * kv_elems), 256, 0, s>>>(
        d.k_ref.data, c.batch, c.total, kv_elems, c.total, 0, c.total, 0, nc, 2);
    fill_rows<<<blocks(uint64_t(c.batch) * c.total * kv_elems), 256, 0, s>>>(
        d.v_ref.data, c.batch, c.total, kv_elems, c.total, 0, c.total, 0, nc, 3);
    CUDA_CHECK(cudaGetLastError());
    d.oracle.compute(d.q_ref.data, d.k_ref.data, d.v_ref.data, s);
  }

  // Payload fill (Q shard, own K/V shard) and full-sequence references; optionally
  // poison every remote K/V row this rank reads and every output.
  void prepare(int payload, bool with_references, bool poison) {
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      fill_shards(*rank, payload);
      if (poison) poison_remote(*rank, Image::linked);
      if (with_references) references(*rank, payload);
    }
    sync_all();
  }

  // Cross-launch release-wait and CTA read-drain coverage without any host or global synchronization between launches.
  // Per launch and per rank, on that rank's stream only: a fresh payload is written
  // into the local chunks, the last rank is delayed 2 ms, the image
  // runs (directly or as a graph replay), and O/LSE are copied into per-launch
  // buffers. Faster ranks run ahead into launch e+1 and push into a peer whose
  // launch e is still pending or reading; only the released/done protocol keeps
  // those pushes from overwriting data launch e reads. The linked sequence and the
  // control sequence are queued back to back; after the single final
  // synchronization every launch of both is checked against its own payload's
  // oracle, and same-payload launches must be bitwise equal.
  int async_sequences(bool graphs, int launches, int first_payload) {
    uint64_t const q_count = RankData::q_count(c, world);
    uint64_t const lse_count = uint64_t(c.batch) * c.heads_q * shard;
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      auto& d = *rank->d;
      for (int i = 0; i < 2; ++i) {
        d.seq_o[i] = std::make_unique<Buffer<Element>>(q_count * launches);
        d.seq_lse[i] = std::make_unique<Buffer<float>>(lse_count * launches);
      }
    }
    sync_all();
    for (int i = 0; i < 2; ++i) {
      bool const control = i == 1;
      for (int e = 0; e < launches; ++e) {
        for (auto& rank : ranks) {
          CUDA_CHECK(cudaSetDevice(rank->device));
          auto& d = *rank->d;
          fill_shards(*rank, first_payload + e);
          if (rank->rank == world - 1) spin_ns<<<1, 1, 0, rank->stream>>>(2'000'000);
          if (graphs) CUDA_CHECK(cudaGraphLaunch(rank->graphs[i], rank->stream));
          else launch(*rank, control ? Image::control : Image::linked);
          CUDA_CHECK(cudaMemcpyAsync(d.seq_o[i]->data + e * q_count, control ? d.o_ctl.data : d.o.data,
                                     q_count * sizeof(Element), cudaMemcpyDeviceToDevice, rank->stream));
          CUDA_CHECK(cudaMemcpyAsync(d.seq_lse[i]->data + e * lse_count, control ? d.lse_ctl.data : d.lse.data,
                                     lse_count * sizeof(float), cudaMemcpyDeviceToDevice, rank->stream));
        }
      }
    }
    sync_all();
    std::string const mode = graphs ? "graph-replay" : "direct";
    for (int e = 0; e < launches; ++e) {
      for (auto& rank : ranks) {
        CUDA_CHECK(cudaSetDevice(rank->device));
        references(*rank, first_payload + e);
      }
      for (auto& rank : ranks) {
        CUDA_CHECK(cudaSetDevice(rank->device));
        auto& d = *rank->d;
        std::string const where = " " + mode + " async launch " + std::to_string(e) + " rank " +
                                  std::to_string(rank->rank);
        for (int i = 0; i < 2; ++i) {
          int failed = d.oracle.check(d.seq_o[i]->data + e * q_count, d.seq_lse[i]->data + e * lse_count,
                                      0, shard, rank->stream);
          if (failed) throw std::runtime_error(std::string(failed & 1 ? "Gate B: " : "Gate C: ") +
              (i ? "control" : "linked") + where + (failed & 1 ? " O" : " LSE") + " differs from fp32 oracle");
        }
        CUDA_CHECK(cudaMemsetAsync(d.bad.data, 0, sizeof(int), rank->stream));
        compare_bitwise<<<int((q_count + 255) / 256), 256, 0, rank->stream>>>(
            d.seq_o[0]->data + e * q_count, d.seq_o[1]->data + e * q_count, q_count, d.bad.data);
        compare_bitwise<<<int((lse_count + 255) / 256), 256, 0, rank->stream>>>(
            d.seq_lse[0]->data + e * lse_count, d.seq_lse[1]->data + e * lse_count, lse_count, d.bad.data);
        CUDA_CHECK(cudaGetLastError());
        int failed = 0;
        CUDA_CHECK(cudaMemcpyAsync(&failed, d.bad.data, sizeof(int), cudaMemcpyDeviceToHost, rank->stream));
        CUDA_CHECK(cudaStreamSynchronize(rank->stream));
        if (failed) throw std::runtime_error("Gate A:" + where + " linked/control O/LSE are not bitwise equal");
      }
    }
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      for (int i = 0; i < 2; ++i) { rank->d->seq_o[i].reset(); rank->d->seq_lse[i].reset(); }
    }
    return 2 * launches;
  }

  // Poison every remote K/V row this rank reads and the outputs `next` writes.
  void poison_remote(Rank& rank, Image next) {
    auto& d = *rank.d;
    int const kv_elems = c.heads_kv * 128;
    int const low = rank.rank, high = 2 * world - 1 - rank.rank;
    for (int s = 0; s <= high; ++s) {
      if (s == low || s == high) continue;
      uint64_t const offset = uint64_t(s) * chunk * kv_elems;
      poison_rows(d.kg.data + offset, c.batch, c.total, chunk, kv_elems, rank.stream);
      poison_rows(d.vg.data + offset, c.batch, c.total, chunk, kv_elems, rank.stream);
    }
    bool const control = next == Image::control;
    uint64_t const lse_bytes = uint64_t(c.batch) * c.heads_q * shard * sizeof(float);
    CUDA_CHECK(cudaMemsetAsync(control ? d.o_ctl.data : d.o.data, 0xff,
                               RankData::q_count(c, world) * sizeof(Element), rank.stream));
    CUDA_CHECK(cudaMemsetAsync(control ? d.lse_ctl.data : d.lse.data, 0xff, lse_bytes, rank.stream));
    if (next == Image::selected)
      for (auto& lse : d.selected_lse)
        CUDA_CHECK(cudaMemsetAsync(lse->data, 0xff, lse_bytes / 2, rank.stream));
  }

  void poison_all(Image next) {
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      poison_remote(*rank, next);
    }
    sync_all();
  }

  void launch(Rank& rank, Image image) {
    auto& d = *rank.d;
    if (image == Image::linked) d.linked->launch(*d.linked_params, rank.stream);
    else if (image == Image::control) d.control->launch(*d.control_params, rank.stream);
    else {
      // Pull each needed remote chunk once, then run both Q segments. Each
      // FamilyKernel launch resets its scheduler inside this timed boundary.
      uint64_t const row_bytes = uint64_t(c.heads_kv) * 128 * sizeof(Element);
      for (int s = 0; s <= 2 * world - 1 - rank.rank; ++s) {
        int const p = s < world ? s : 2 * world - 1 - s;
        if (p == rank.rank) continue;
        auto& src = *ranks[p]->d;
        for (int b = 0; b < c.batch; ++b) {
          uint64_t const offset = (uint64_t(b) * c.total + uint64_t(s) * chunk) * row_bytes;
          CUDA_CHECK(cudaMemcpyPeerAsync(reinterpret_cast<char*>(d.kg.data) + offset, rank.device,
              reinterpret_cast<char const*>(src.kg.data) + offset, p, chunk * row_bytes, rank.stream));
          CUDA_CHECK(cudaMemcpyPeerAsync(reinterpret_cast<char*>(d.vg.data) + offset, rank.device,
              reinterpret_cast<char const*>(src.vg.data) + offset, p, chunk * row_bytes, rank.stream));
        }
      }
      for (auto& params : d.selected_params) d.selected->launch(*params, rank.stream);
    }
  }

  // Lockstep: issue launch e to every rank before any rank's launch e+1.
  void launch_all(Image image, int skew_rank = -1) {
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      if (rank->rank == skew_rank) spin_ns<<<1, 1, 0, rank->stream>>>(2'000'000);
      launch(*rank, image);
    }
  }

  void capture_graphs() {
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      for (int i = 0; i < 2; ++i) {
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(rank->stream, cudaStreamCaptureModeThreadLocal));
        launch(*rank, i == 0 ? Image::linked : Image::control);
        CUDA_CHECK(cudaStreamEndCapture(rank->stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&rank->graphs[i], graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
      }
    }
  }

  void replay_all(Image image) {
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      CUDA_CHECK(cudaGraphLaunch(rank->graphs[image == Image::linked ? 0 : 1], rank->stream));
    }
  }

  // Oracle on every rank for one image; throws with the failing launch label.
  void check(Image image, std::string const& label) {
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      auto& d = *rank->d;
      bool const control = image == Image::control;
      // Only correctness assembles the separate segment LSEs into CP layout.
      if (image == Image::selected)
        for (int segment = 0; segment < 2; ++segment)
          CUDA_CHECK(cudaMemcpy2DAsync(d.lse.data + segment * chunk, shard * sizeof(float),
              d.selected_lse[segment]->data, chunk * sizeof(float), chunk * sizeof(float),
              c.batch * c.heads_q, cudaMemcpyDeviceToDevice, rank->stream));
      int failed = d.oracle.check(control ? d.o_ctl.data : d.o.data, control ? d.lse_ctl.data : d.lse.data,
                                  0, shard, rank->stream);
      if (failed & 1) throw std::runtime_error("Gate B: " + label + " rank " + std::to_string(rank->rank) + " O differs from fp32 oracle");
      if (failed & 2) throw std::runtime_error("Gate C: " + label + " rank " + std::to_string(rank->rank) + " LSE differs from fp32 oracle");
    }
  }

  void check_bitwise(std::string const& label) {
    for (auto& rank : ranks) {
      CUDA_CHECK(cudaSetDevice(rank->device));
      auto& d = *rank->d;
      uint64_t const n = RankData::q_count(c, world), rows = uint64_t(c.batch) * c.heads_q * shard;
      CUDA_CHECK(cudaMemsetAsync(d.bad.data, 0, sizeof(int), rank->stream));
      compare_bitwise<<<int((n + 255) / 256), 256, 0, rank->stream>>>(d.o.data, d.o_ctl.data, n, d.bad.data);
      compare_bitwise<<<int((rows + 255) / 256), 256, 0, rank->stream>>>(d.lse.data, d.lse_ctl.data, rows, d.bad.data);
      CUDA_CHECK(cudaGetLastError());
      int failed = 0;
      CUDA_CHECK(cudaMemcpyAsync(&failed, d.bad.data, sizeof(int), cudaMemcpyDeviceToHost, rank->stream));
      CUDA_CHECK(cudaStreamSynchronize(rank->stream));
      if (failed) throw std::runtime_error("Gate A: " + label + " rank " + std::to_string(rank->rank) +
                                           " linked/control O/LSE are not bitwise equal");
    }
  }

  // One payload: linked then control on every rank, each preceded by poisoning.
  void gate_payload(int payload, bool graphs, int skew_linked, int skew_control) {
    std::string const tag = (graphs ? "graph-replay payload " : "payload ") + std::to_string(payload);
    prepare(payload, true, true);
    if (graphs) replay_all(Image::linked); else launch_all(Image::linked, skew_linked);
    sync_all();
    check(Image::linked, "linked " + tag);
    poison_all(Image::control);
    if (graphs) replay_all(Image::control); else launch_all(Image::control, skew_control);
    sync_all();
    check(Image::control, "control " + tag);
    check_bitwise(tag);
  }

  uint32_t generation() {
    uint32_t g = 0;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(&g, ranks[0]->d->gen.data, 4, cudaMemcpyDeviceToHost));
    return g;
  }

  // ---- all-rank timing boundary ----
  struct Block { double step_ms, wall_ms; std::vector<double> rank_ms; };

  Block run_block(Image image, uint64_t n) {
    std::vector<cudaEvent_t> begin(world), end(world);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventCreate(&begin[r]));
      CUDA_CHECK(cudaEventCreate(&end[r]));
    }
    auto host_begin = std::chrono::steady_clock::now();
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventRecord(begin[r], ranks[r]->stream));
    }
    for (uint64_t i = 0; i < n; ++i) launch_all(image);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventRecord(end[r], ranks[r]->stream));
    }
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventSynchronize(end[r]));
    }
    double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - host_begin).count();
    Block block{0, wall / double(n), {}};
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, begin[r], end[r]));
      block.rank_ms.push_back(ms / double(n));
      block.step_ms = std::max(block.step_ms, ms / double(n));
      cudaEventDestroy(begin[r]); cudaEventDestroy(end[r]);
    }
    return block;
  }

  // Per-rank duration of one CP launch or sequential gather plus two launches.
  std::vector<double> rank_launch_ms(Image image, int reps) {
    std::vector<std::vector<double>> samples(world);
    for (int k = 0; k < reps; ++k) {
      std::vector<cudaEvent_t> a(world), b(world);
      for (auto& rank : ranks) {
        CUDA_CHECK(cudaSetDevice(rank->device));
        CUDA_CHECK(cudaEventCreate(&a[rank->rank])); CUDA_CHECK(cudaEventCreate(&b[rank->rank]));
        CUDA_CHECK(cudaEventRecord(a[rank->rank], rank->stream));
        launch(*rank, image);
        CUDA_CHECK(cudaEventRecord(b[rank->rank], rank->stream));
      }
      sync_all();
      for (int r = 0; r < world; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a[r], b[r]));
        samples[r].push_back(ms);
        cudaEventDestroy(a[r]); cudaEventDestroy(b[r]);
      }
    }
    std::vector<double> medians;
    for (auto& s : samples) medians.push_back(median(s));
    return medians;
  }

  static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }

  // Size n so every block spans >= 20 ms on the slowest rank; resize restarts.
  uint64_t size_block(Image image) {
    uint64_t n = 1;
    while (true) {
      double ms = run_block(image, n).step_ms * double(n);
      if (ms >= 20.0) return n;
      n = std::max(n + 1, uint64_t(std::ceil(double(n) * 25.0 / ms)));
    }
  }
};

void write_block_series(std::ostream& out, char const* key, std::vector<Cp::Block> const& blocks, uint64_t n) {
  out << '"' << key << "\":{\"n\":" << n << ",\"step_ms\":[";
  for (size_t i = 0; i < blocks.size(); ++i) out << (i ? "," : "") << blocks[i].step_ms;
  out << "],\"wall_step_ms\":[";
  for (size_t i = 0; i < blocks.size(); ++i) out << (i ? "," : "") << blocks[i].wall_ms;
  out << "],\"rank_step_ms\":[";
  for (size_t i = 0; i < blocks.size(); ++i) {
    out << (i ? "," : "") << '[';
    for (size_t r = 0; r < blocks[i].rank_ms.size(); ++r) out << (r ? "," : "") << blocks[i].rank_ms[r];
    out << ']';
  }
  std::vector<double> steps, walls;
  for (auto const& b : blocks) { steps.push_back(b.step_ms); walls.push_back(b.wall_ms); }
  out << "],\"median_step_ms\":" << Cp::median(steps) << ",\"median_wall_step_ms\":" << Cp::median(walls) << '}';
}

void write_vector(std::ostream& out, std::vector<double> const& v) {
  out << '[';
  for (size_t i = 0; i < v.size(); ++i) out << (i ? "," : "") << v[i];
  out << ']';
}

void prefix(std::ostream& out, Options const& o, Cell const& c, Cp const& cp) {
  out << std::setprecision(17) << "{\"schema\":\"pipelink-attention-cp-slot-v1\",\"action\":" << json_quote(o.action)
      << ",\"cell\":" << json_quote(c.id) << ",\"world\":" << o.world << ",\"total_tokens\":" << c.total
      << ",\"sharding\":\"zigzag\",\"chunk_tokens\":" << cp.chunk
      << ",\"shard_tokens\":" << cp.shard << ",\"batch\":" << c.batch << ",\"heads_q\":" << c.heads_q
      << ",\"heads_kv\":" << c.heads_kv << ",\"grid\":" << cp.grid << ',';
}

int run(Options const& o) {
  Cell c = resolve_cell(o.cells_json, o.cell);
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
  CU_CHECK(cuInit(0));
  Cp cp(o, c);
  auto& out = std::cout;
  if (o.action == "sanitize") {
    Image image = o.mode == "linked" ? Image::linked : Image::control;
    cp.prepare(0, false, false);
    cp.launch_all(image);
    cp.launch_all(image);
    cp.sync_all();
    prefix(out, o, c, cp);
    out << "\"images_executed\":[" << json_quote(o.mode) << "],\"launches\":2,\"generation\":" << cp.generation() << "}\n";
    return 0;
  }
  if (o.action == "gate") {
    // Lockstep launches with changing payloads. Skews: one rank starts 2 ms late
    // (rank 0 and rank W-1 exercise opposite ownership extremes) on alternate payloads.
    int launches = 0;
    for (int payload = 0; payload < o.payloads; ++payload) {
      int const skew = payload % 4 == 1 ? 0 : payload % 4 == 3 ? o.world - 1 : -1;
      cp.gate_payload(payload, false, skew, payload == 2 ? o.world - 1 : -1);
      launches += 2;
    }
    cp.capture_graphs();
    for (int replay = 0; replay < o.replays; ++replay) {
      cp.gate_payload(o.payloads + replay, true, -1, -1);
      launches += 2;
    }
    int const synced_launches = launches;
    // Asynchronous sequences (no host/global sync between launches), direct then graph.
    int const async_payload = o.payloads + o.replays;
    launches += cp.async_sequences(false, o.async_launches, async_payload);
    launches += cp.async_sequences(true, o.async_launches, async_payload + o.async_launches);
    uint32_t const gen = cp.generation();
    if (gen != uint32_t(launches)) throw std::runtime_error("launch generation does not count every launch");
    prefix(out, o, c, cp);
    out << "\"correctness\":{\"all_true\":true,\"gate_a_bitwise\":true,\"gate_b_linked\":true,"
        << "\"gate_b_control\":true,\"gate_c_lse\":true},\"payloads\":" << o.payloads
        << ",\"graph_replay_payloads\":" << o.replays << ",\"synchronized_launches_per_rank\":" << synced_launches
        << ",\"async_sequences\":{\"modes\":[\"direct\",\"graph-replay\"],\"launches_per_image\":"
        << o.async_launches << ",\"images\":[\"linked\",\"control\"],\"delayed_rank\":" << o.world - 1
        << ",\"delay_ms_per_launch\":2,\"host_sync_between_launches\":false,\"every_launch_checked\":true}"
        << ",\"launches_per_rank\":" << launches
        << ",\"generation\":" << gen << ",\"skewed_ranks\":[0," << o.world - 1 << "],\"skew_ms\":2,"
        << "\"poisoned_remote_kv_and_outputs\":\"synchronized launches\",\"oracle\":\"fp32-boundary-rows-all-heads-full-sequence\","
        << "\"oracle_rows\":";
    std::vector<double> rows;
    for (int row : boundary_rows(c, o.world)) rows.push_back(row);
    write_vector(out, rows);
    out << "}\n";
    return 0;
  }
  // Timing actions: one correct, poisoned check first, then warm L2 and payload 0.
  cp.gate_payload(0, false, -1, -1);
  cp.prepare(0, false, false);
  if (o.action == "parity") {
    for (int i = 0; i < 10; ++i) { cp.launch_all(Image::linked); cp.launch_all(Image::control); }
    cp.sync_all();
    uint64_t n = std::max(cp.size_block(Image::linked), cp.size_block(Image::control));
    std::vector<Cp::Block> linked, control;
    std::vector<double> ratios;
    for (int pair = 0; pair < 10; ++pair) {
      Cp::Block a, b;
      if (pair % 2 == 0) { a = cp.run_block(Image::linked, n); b = cp.run_block(Image::control, n); }
      else { b = cp.run_block(Image::control, n); a = cp.run_block(Image::linked, n); }
      linked.push_back(a); control.push_back(b);
      ratios.push_back(b.step_ms / a.step_ms);  // linked throughput / control throughput
    }
    auto rank_linked = cp.rank_launch_ms(Image::linked, 10);
    auto rank_control = cp.rank_launch_ms(Image::control, 10);
    double const ratio = Cp::median(ratios);
    prefix(out, o, c, cp);
    out << "\"correctness\":{\"checked_before_timing\":true,\"gate_a_bitwise\":true},"
        << "\"timing_method\":\"all-rank-boundary\",\"step\":\"max_r(elapsed_r)/n\","
        << "\"order\":\"alternating-linked-first-control-first\",\"pairs\":10,";
    write_block_series(out, "linked", linked, n);
    out << ',';
    write_block_series(out, "control", control, n);
    out << ",\"pair_ratios\":";
    write_vector(out, ratios);
    out << ",\"ratio\":" << ratio << ",\"ratio_definition\":\"median(control_step/linked_step) = linked/control throughput\","
        << "\"ratio_floor\":0.97,\"verdict\":" << (ratio >= 0.97 ? "\"PASS\"" : "\"FAIL\"")
        << ",\"rank_launch_ms\":{\"linked\":";
    write_vector(out, rank_linked);
    out << ",\"control\":";
    write_vector(out, rank_control);
    out << "}}\n";
    return 0;
  }
  // placement (descriptive)
  cp.poison_all(Image::selected);
  cp.launch_all(Image::selected);
  cp.sync_all();
  cp.check(Image::selected, "sequential gather + selected");
  cp.prepare(0, false, false);
  auto absolute = [&](Image image) {
    for (int i = 0; i < 10; ++i) cp.launch_all(image);
    cp.sync_all();
    uint64_t n = cp.size_block(image);
    std::vector<Cp::Block> blocks;
    for (int i = 0; i < 10; ++i) blocks.push_back(cp.run_block(image, n));
    return std::make_pair(n, blocks);
  };
  auto [cp_n, cp_blocks] = absolute(Image::linked);
  auto [seq_n, seq_blocks] = absolute(Image::selected);
  auto cp_rank = cp.rank_launch_ms(Image::linked, 10);
  auto seq_rank = cp.rank_launch_ms(Image::selected, 10);
  // Single-GPU full-sequence `selected` on device 0 over its reference tensors.
  CUDA_CHECK(cudaSetDevice(0));
  Rank& r0 = *cp.ranks[0];
  auto& d0 = *r0.d;
  uint64_t const full_q = uint64_t(c.batch) * c.total * c.heads_q * 128;
  Buffer<Element> full_o(full_q);
  Buffer<float> full_lse(uint64_t(c.batch) * c.heads_q * c.total);
  cp.prepare(0, true, false);
  CUDA_CHECK(cudaSetDevice(0));
  auto full_args = cp.arguments(r0, full_o.data, full_lse.data, c.total, c.total, d0.q_ref.data,
                                d0.k_ref.data, d0.v_ref.data, c.total);
  auto full_fp = make_family_params<true>(full_args, 0, 0);
  PackedParams full_params(cp.selected_meta);
  full_params.put(cp.selected_meta.params[0], full_fp.load);
  full_params.put(cp.selected_meta.params[1], full_fp.core);
  full_params.put(cp.selected_meta.params[2], full_fp.epilogue);
  full_params.put(cp.selected_meta.params[3], full_fp.empty);
  int const full_grid = std::min(c.batch * (c.total / 128) * c.heads_q, cp.grid_limit(r0));
  FamilyKernel full(o.selected_cubin, cp.selected_meta, full_grid, d0.resets.data, sizeof(int));
  {
    Oracle oracle(c, boundary_rows(c, o.world));
    oracle.compute(d0.q_ref.data, d0.k_ref.data, d0.v_ref.data, nullptr);
    full.launch(full_params);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (oracle.check(full_o.data, full_lse.data, 0, c.total, nullptr))
      throw std::runtime_error("single-GPU selected differs from fp32 oracle");
  }
  auto single = time_absolute([&] { full.launch(full_params); });
  prefix(out, o, c, cp);
  out << "\"descriptive\":true,\"correctness\":{\"cp_linked\":true,\"sequential_selected\":true,\"single_gpu_selected\":true},"
      << "\"timing_method\":\"all-rank-boundary\",\"reps\":10,";
  write_block_series(out, "cp_linked", cp_blocks, cp_n);
  out << ',';
  write_block_series(out, "sequential_gather_selected", seq_blocks, seq_n);
  out << ",\"rank_launch_ms\":{\"cp_linked\":";
  write_vector(out, cp_rank);
  out << ",\"sequential_gather_selected\":";
  write_vector(out, seq_rank);
  out << "},\"single_gpu_selected\":{";
  write_absolute_timing(out, single);
  out << "\"device\":0,\"grid\":" << full_grid << "}}\n";
  return 0;
}
}  // namespace cp_slot
int main(int argc, char** argv) {
  try {
    return cp_slot::run(cp_slot::parse_options(argc, argv));
  } catch (std::exception const& e) {
    std::cerr << "attention-slot-cp-sm90: " << e.what() << '\n';
    return 2;
  }
}
