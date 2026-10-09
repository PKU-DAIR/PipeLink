#pragma once
// Host pieces shared by the single-device family slot (slot_sm90.cu) and the
// multi-device context-parallel slot (slot_cp_sm90.cu): link.json launch and
// Params layout, packed Params, kernel modules, device buffers, linked/control
// bitwise comparison and the independent fp32 boundary-row oracle.
// CUDA_CHECK / CU_CHECK are supplied by the including slot.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "json.cuh"
#include "fixtures.cuh"
#include "../sm90/slot_chunk.cuh"
#include "../sm90/slot_phi.cuh"
#include "../sm90/slot_msa.cuh"

namespace pipelink::attention::slot {
using OracleElement = cutlass::bfloat16_t;

struct ParamEntry {
  std::string id;
  int offset, size, alignment;
};

struct LinkMetadata {
  std::string symbol;
  int threads, dyn_smem_bytes, params_bytes, params_align;
  std::array<int, 3> cluster;
  std::array<ParamEntry, 6> params;
  Json bindings;
};

// `stages` are package suffixes of org.pipelink.attention.d128-sm90-c.*, in the
// order the slot packs them.
inline LinkMetadata load_link_metadata(std::string const& path, std::array<std::string, 6> const& stages) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open link.json: " + path);
  std::string raw{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  Json meta = JsonParser(raw).parse();
  LinkMetadata out{};
  out.symbol = meta.at("kernel_symbol").string();
  out.threads = meta.at("threads").integer();
  out.dyn_smem_bytes = meta.at("dyn_smem_bytes").integer();
  auto const& cluster = meta.at("cluster").array();
  if (cluster.size() != 3) throw std::runtime_error("link.json cluster must be a fixed 3D shape");
  for (int i = 0; i < 3; ++i) out.cluster[i] = cluster[i].integer();
  auto const& params = meta.at("params");
  out.params_bytes = params.at("total_bytes").integer();
  out.params_align = params.at("align").integer();
  if (out.symbol.empty() || out.threads != 384 || out.dyn_smem_bytes <= 0 ||
      out.cluster != std::array<int, 3>{1, 1, 1} || out.params_bytes <= 0 ||
      out.params_align < 64 || (out.params_align & (out.params_align - 1)) != 0)
    throw std::runtime_error("link.json violates sm90 family launch contract (384 threads, cluster-1)");
  auto const& entries = params.at("entries").array();
  std::string prefix = "org.pipelink.attention.d128-sm90-c.";
  for (int i = 0; i < int(stages.size()); ++i) {
    std::string id = prefix + stages[i];
    auto entry = std::find_if(entries.begin(), entries.end(), [&](Json const& item) {
      auto const& fields = item.array();
      return !fields.empty() && fields[0].string() == id;
    });
    if (entry == entries.end()) throw std::runtime_error("link.json has no params entry for " + id);
    auto const& fields = entry->array();
    if (fields.size() < 4) throw std::runtime_error("malformed params entry for " + id);
    auto& p = out.params[i];
    p = {id, fields[1].integer(), fields[2].integer(), fields[3].integer()};
    if (p.offset < 0 || p.size < 0 || p.size > out.params_bytes ||
        p.alignment <= 0 || out.params_align % p.alignment != 0 ||
        p.offset % p.alignment != 0 || p.offset > out.params_bytes - p.size)
      throw std::runtime_error("malformed params layout for " + id);
  }
  out.bindings = meta.at("bindings");
  return out;
}

class PackedParams {
 public:
  void* data = nullptr;
  explicit PackedParams(LinkMetadata const& meta) {
    if (posix_memalign(&data, meta.params_align, meta.params_bytes) != 0)
      throw std::runtime_error("cannot allocate aligned family Params");
    std::memset(data, 0, meta.params_bytes);
  }
  ~PackedParams() { std::free(data); }
  PackedParams(PackedParams const&) = delete;
  PackedParams& operator=(PackedParams const&) = delete;

  template <class T>
  void put(ParamEntry const& entry, T const& value) {
    if (entry.size != sizeof(T) || entry.alignment != alignof(T))
      throw std::runtime_error("link.json Params size/alignment differs from compiled params for " + entry.id);
    std::memcpy(static_cast<unsigned char*>(data) + entry.offset, &value, sizeof(T));
  }
};

// One linked or control image in the current device's context. Every launch
// first zeroes `reset_bytes` at `reset` (scheduler semaphore and any per-launch
// workspace) on the same stream; both are captured together by stream capture.
class FamilyKernel {
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
  void* reset_;
  size_t reset_bytes_;
  CUlaunchConfig launch_{};
 public:
  FamilyKernel(std::string const& path, LinkMetadata const& meta, int grid, void* reset, size_t reset_bytes)
      : reset_(reset), reset_bytes_(reset_bytes) {
    CU_CHECK(cuModuleLoad(&module_, path.c_str()));
    CU_CHECK(cuModuleGetFunction(&function_, module_, meta.symbol.c_str()));
    CU_CHECK(cuFuncSetAttribute(function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, meta.dyn_smem_bytes));
    launch_.gridDimX = grid; launch_.gridDimY = launch_.gridDimZ = 1;
    launch_.blockDimX = meta.threads; launch_.blockDimY = launch_.blockDimZ = 1;
    launch_.sharedMemBytes = meta.dyn_smem_bytes;
    // Only cluster (1,1,1) is supported; no launch attribute is needed.
  }
  ~FamilyKernel() { if (module_) cuModuleUnload(module_); }
  FamilyKernel(FamilyKernel const&) = delete;
  FamilyKernel& operator=(FamilyKernel const&) = delete;

  void launch(PackedParams& params, cudaStream_t stream = nullptr) {
    if (reset_bytes_) CUDA_CHECK(cudaMemsetAsync(reset_, 0, reset_bytes_, stream));
    void* args[] = {params.data};
    launch_.hStream = reinterpret_cast<CUstream>(stream);
    CU_CHECK(cuLaunchKernelEx(&launch_, function_, args, nullptr));
  }
};

template <class T>
__global__ void compare_bitwise(T const* a, T const* b, uint64_t n, int* bad) {
  using Bits = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
  static_assert(sizeof(Bits) == sizeof(T));
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n && reinterpret_cast<Bits const*>(a)[i] != reinterpret_cast<Bits const*>(b)[i])
    atomicExch(bad, 1);
}

template <class T>
class Buffer {
 public:
  T* data = nullptr;
  explicit Buffer(uint64_t count) { CUDA_CHECK(cudaMalloc(&data, count * sizeof(T))); }
  ~Buffer() { cudaFree(data); }
  Buffer(Buffer const&) = delete;
  Buffer& operator=(Buffer const&) = delete;
};

// Independent fp32 spot oracle: all heads at boundary rows, full visible keys.
// Scores live in scratch rather than S-sized shared memory, including at 128K.
__global__ void spot_reference(OracleElement const* q, OracleElement const* k, OracleElement const* v,
    int const* samples, int sample_count, int seqlen, int heads_q, int heads_kv,
    int window, int chunk, bool phi, int32_t const* selection,
    float scale, float softcap, float* scratch, float* out, float* lse) {
  __shared__ float reduce[256];
  int sample = blockIdx.x % sample_count;
  int hb = blockIdx.x / sample_count;
  int head = hb % heads_q, batch = hb / heads_q;
  int kv_head = head / (heads_q / heads_kv);
  int row = samples[sample], begin = window > 0 ? max(0, row - window + 1) : 0;
  if (chunk > 0) begin = max(begin, row / chunk * chunk);
  int tid = threadIdx.x;
  uint64_t qi = (uint64_t(batch) * seqlen * heads_q + row * heads_q + head) * 128;
  float* scores = scratch + uint64_t(blockIdx.x) * seqlen;
  float maximum = -INFINITY;
  for (int col = begin + tid; col <= row; col += blockDim.x) {
    uint64_t ki = (uint64_t(batch) * seqlen * heads_kv + col * heads_kv + kv_head) * 128;
    float dot = 0;
    for (int d = 0; d < 128; ++d) dot += float(q[qi + d]) * float(k[ki + d]);
    float logit = dot * scale;
    if (softcap > 0) logit = softcap * tanhf(logit / softcap);
    bool visible = (chunk == 0 || pipelink::attention::sm90::chunk_visible(row, col, chunk)) &&
        (!phi || pipelink::attention::sm90::phi_visible(row, col, head)) &&
        (!selection || pipelink::attention::sm90::msa_visible(selection, seqlen, head, row, col));
    scores[col] = visible ? logit : -INFINITY;
    maximum = fmaxf(maximum, scores[col]);
  }
  reduce[tid] = maximum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    __syncthreads();
  }
  maximum = reduce[0];
  __syncthreads();
  float sum = 0;
  for (int col = begin + tid; col <= row; col += blockDim.x) {
    scores[col] = expf(scores[col] - maximum);
    sum += scores[col];
  }
  reduce[tid] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }
  sum = reduce[0];
  if (tid < 128) {
    float acc = 0;
    for (int col = begin; col <= row; ++col) {
      uint64_t vi = (uint64_t(batch) * seqlen * heads_kv + col * heads_kv + kv_head) * 128;
      acc += scores[col] * float(v[vi + tid]);
    }
    out[uint64_t(blockIdx.x) * 128 + tid] = acc / sum;
  }
  if (tid == 0) lse[blockIdx.x] = maximum + logf(sum);
}

// `output`/`output_lse` hold rows [row_offset, row_offset + out_seqlen) of the
// sequence the samples index (a context-parallel shard, or the whole sequence).
__global__ void compare_spots(OracleElement const* output, float const* output_lse,
    float const* ref_o, float const* ref_lse, int const* samples,
    int sample_count, int row_offset, int out_seqlen, int heads_q, int* bad) {
  int hb = blockIdx.x / sample_count;
  int row = samples[blockIdx.x % sample_count] - row_offset;
  uint64_t oi = (uint64_t(hb / heads_q) * out_seqlen * heads_q +
                 row * heads_q + hb % heads_q) * 128;
  if (!reference_matches(float(output[oi + threadIdx.x]),
                         ref_o[uint64_t(blockIdx.x) * 128 + threadIdx.x]))
    atomicOr(bad, 1);
  if (threadIdx.x == 0 &&
      !reference_matches(output_lse[uint64_t(hb) * out_seqlen + row], ref_lse[blockIdx.x]))
    atomicOr(bad, 2);
}

}  // namespace pipelink::attention::slot
