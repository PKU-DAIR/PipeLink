#pragma once
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "msa_combine.cuh"

namespace pipelink::attention::sm90 {

__device__ inline bool msa_visible(int32_t const* selection, int seqlen,
                                  int head, int query, int key) {
  if (key > query) return false;
  int const block = key / 128;
  auto const* selected = selection + (uint64_t(head / 16) * seqlen + query) * 16;
  for (int rank = 0; rank < 16; ++rank) {
    if (selected[rank] == block) return true;
    if (selected[rank] < 0 || selected[rank] > block) break;
  }
  return false;
}

// Immutable host CSR. A row contributes one local partial and either one
// exact-set remote partial or one partial per selected remote block.
struct MsaWorklist {
  std::vector<int32_t> selection, rows, blocks, tokens, inverse;
  std::vector<uint8_t> partial_counts;
  std::vector<MsaItem> items;
  int local_items = 0, grouped_items = 0, per_block_items = 0;
  int partial_shift = 0;

  MsaWorklist(std::string const& path, int seqlen) {
    if (seqlen <= 0 || seqlen % 128 != 0 || seqlen > 131072)
      throw std::runtime_error("MSA requires complete 128-row blocks, S <= 128K");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    size_t const entries = size_t(4) * seqlen * 16;
    if (!input || input.tellg() != std::streamoff(entries * sizeof(int32_t)))
      throw std::runtime_error("MSA selection must be raw int32 [4,S,16]: " + path);
    selection.resize(entries);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(selection.data()), entries * sizeof(int32_t)))
      throw std::runtime_error("cannot read MSA selection: " + path);
    int const num_blocks = seqlen / 128;
    for (int g = 0; g < 4; ++g) {
      for (int q = 0; q < seqlen; ++q) {
        int const local = q / 128, count = std::min(16, local + 1);
        auto const* selected = selection.data() + (size_t(g) * seqlen + q) * 16;
        int previous = -1;
        for (int rank = 0; rank < 16; ++rank) {
          int const block = selected[rank];
          if (rank >= count) {
            if (block != -1) throw std::runtime_error("MSA selection padding must be -1");
          } else {
            if (block <= previous || block > local || block < 0 ||
                (rank == count - 1 && block != local))
              throw std::runtime_error("MSA selection must be sorted, distinct, causal and include local");
            previous = block;
          }
        }
      }
    }
    // Eight GQA16 tokens fill a tile. Requiring G=128 shared rows avoids
    // turning the mostly unique uniform sets into sparsely occupied items.
    constexpr int group_rows = 128;
    struct Group { int head, begin, tokens; };
    std::vector<Group> groups;
    std::vector<int> sorted_tokens(4 * seqlen);
    std::vector<bool> grouped(4 * seqlen, false);
    for (int g = 0; g < 4; ++g) {
      auto begin = sorted_tokens.begin() + g * seqlen + 128;
      auto end = sorted_tokens.begin() + (g + 1) * seqlen;
      std::iota(begin, end, 128);
      auto selected = [&](int q) { return selection.data() + (size_t(g) * seqlen + q) * 16; };
      std::sort(begin, end, [&](int a, int b) {
        int const na = std::min(15, a / 128), nb = std::min(15, b / 128);
        auto const* sa = selected(a);
        auto const* sb = selected(b);
        int const n = std::min(na, nb);
        for (int rank = 0; rank < n; ++rank)
          if (sa[rank] != sb[rank]) return sa[rank] < sb[rank];
        return na != nb ? na < nb : a < b;
      });
      for (int first = g * seqlen + 128; first < (g + 1) * seqlen;) {
        int const q = sorted_tokens[first], n = std::min(15, q / 128);
        int last = first + 1;
        while (last < (g + 1) * seqlen) {
          int const other = sorted_tokens[last];
          if (std::min(15, other / 128) != n ||
              !std::equal(selected(q), selected(q) + n, selected(other))) break;
          ++last;
        }
        if ((last - first) * 16 >= group_rows) {
          groups.push_back({g, first, last - first});
          for (int pos = first; pos < last; ++pos)
            grouped[g * seqlen + sorted_tokens[pos]] = true;
        }
        first = last;
      }
    }

    // A power-of-two stride makes row-map entries directly usable by scatter,
    // without a dependent global offset lookup for every output sector.
    partial_counts.resize(seqlen * 64);
    int total_partials = 0, max_partials = 1;
    for (int q = 0; q < seqlen; ++q) {
      for (int h = 0; h < 64; ++h) {
        int const count = grouped[(h / 16) * seqlen + q] ? 2 : std::min(16, q / 128 + 1);
        int const row = q * 64 + h;
        partial_counts[row] = uint8_t(count);
        total_partials += count;
        max_partials = std::max(max_partials, count);
      }
    }
    while ((1 << partial_shift) < max_partials) ++partial_shift;
    std::vector<int> counts(4 * num_blocks, 0), offsets(4 * num_blocks);
    for (int g = 0; g < 4; ++g) {
      for (int q = 128; q < seqlen; ++q) {
        if (grouped[g * seqlen + q]) continue;
        auto const* selected = selection.data() + (size_t(g) * seqlen + q) * 16;
        for (int rank = 0; rank < std::min(15, q / 128); ++rank)
          counts[g * num_blocks + selected[rank]] += 16;
      }
    }
    int group_begin = seqlen * 64;
    for (int bucket = 0; bucket < 4 * num_blocks; ++bucket) {
      offsets[bucket] = group_begin;
      group_begin += counts[bucket];
    }
    rows.resize(total_partials);
    items.reserve((rows.size() + 127) / 128 + 4 * num_blocks + groups.size());
    for (int g = 0; g < 4; ++g) {
      for (int block = 0; block < num_blocks; ++block) {
        for (int h = g * 16; h < (g + 1) * 16; ++h) {
          int const begin = ((g * num_blocks + block) * 16 + h % 16) * 128;
          items.push_back({g, block, h, begin, 128, -1, 1});
          ++local_items;
          for (int r = 0; r < 128; ++r) {
            int const row = (block * 128 + r) * 64 + h;
            rows[begin + r] = (row << partial_shift) + partial_counts[row] - 1;
          }
        }
        int const bucket = g * num_blocks + block;
        if (counts[bucket] == 0) continue;
        int const block_begin = int(blocks.size());
        blocks.push_back(block);
        for (int offset = 0; offset < counts[bucket]; offset += 128) {
          items.push_back({g, block, -1, offsets[bucket] + offset,
                           std::min(128, counts[bucket] - offset), block_begin, 1});
          ++per_block_items;
        }
      }
      for (int q = 128; q < seqlen; ++q) {
        if (grouped[g * seqlen + q]) continue;
        auto const* selected = selection.data() + (size_t(g) * seqlen + q) * 16;
        for (int rank = 0; rank < std::min(15, q / 128); ++rank) {
          int& dest = offsets[g * num_blocks + selected[rank]];
          for (int h = g * 16; h < (g + 1) * 16; ++h)
            rows[dest++] = ((q * 64 + h) << partial_shift) + rank;
        }
      }
    }
    for (auto const& group : groups) {
      int const q = sorted_tokens[group.begin], steps = std::min(15, q / 128);
      auto const* selected = selection.data() + (size_t(group.head) * seqlen + q) * 16;
      int const block_begin = int(blocks.size());
      for (int rank = steps - 1; rank >= 0; --rank) blocks.push_back(selected[rank]);
      for (int offset = 0; offset < group.tokens * 16; offset += 128) {
        items.push_back({group.head, selected[steps - 1], -1, group_begin + offset,
                         std::min(128, group.tokens * 16 - offset), block_begin, steps});
        ++grouped_items;
      }
      for (int pos = group.begin; pos < group.begin + group.tokens; ++pos)
        for (int h = group.head * 16; h < (group.head + 1) * 16; ++h)
          rows[group_begin++] = (sorted_tokens[pos] * 64 + h) << partial_shift;
    }
    // Remote GQA16 heads share one inverse entry; local heads occupy adjacent
    // whole tiles, so combine adds head*128 instead of head for the last rank.
    tokens.assign(items.size() * 8, seqlen);
    inverse.assign(size_t(4 * seqlen) << partial_shift, -1);
    for (int index = 0; index < int(items.size()); ++index) {
      auto const& item = items[index];
      for (int r = 0; r < item.row_count; ++r) {
        int const encoded = rows[item.row_begin + r], logical = encoded >> partial_shift;
        int const q = logical / 64, head = logical % 64;
        if (item.q_head < 0 && r % 16 == 0) tokens[index * 8 + r / 16] = q;
        if (head % 16 != 0) continue;
        int const rank = encoded & ((1 << partial_shift) - 1);
        inverse[((head / 16 * seqlen + q) << partial_shift) + rank] = index * 128 + r;
      }
    }
  }
};

template <class T>
class MsaBuffer {
 public:
  T* data = nullptr;
  MsaBuffer() = default;
  MsaBuffer(MsaBuffer const&) = delete;
  MsaBuffer& operator=(MsaBuffer const&) = delete;
  ~MsaBuffer() { cudaFree(data); }
  void allocate(uint64_t count) { CUDA_CHECK(cudaMalloc(&data, count * sizeof(T))); }
  void upload(std::vector<T> const& values) {
    allocate(values.size());
    CUDA_CHECK(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
};

class MsaWorkspace {
  MsaBuffer<MsaItem> items_;
  MsaBuffer<int32_t> tokens_, blocks_, inverse_, selection_;
  MsaBuffer<uint8_t> counts_;
  MsaBuffer<FamilyTypes<true>::Element> partial_o_;
  MsaBuffer<float> partial_m_, partial_l_;
  int seqlen_, num_items_, partial_shift_;
  cute::TmaDescriptor store_o_{};
 public:
  explicit MsaWorkspace(std::string const& path, int seqlen) : seqlen_(seqlen) {
    auto const start = std::chrono::steady_clock::now();
    MsaWorklist host(path, seqlen);
    auto const built = std::chrono::steady_clock::now();
    num_items_ = int(host.items.size());
    items_.upload(host.items);
    tokens_.upload(host.tokens);
    inverse_.upload(host.inverse);
    blocks_.upload(host.blocks);
    counts_.upload(host.partial_counts);
    partial_shift_ = host.partial_shift;
    selection_.upload(host.selection);
    uint64_t const capacity = uint64_t(num_items_) * 128;
    partial_o_.allocate(capacity * 128);
    partial_m_.allocate(capacity);
    partial_l_.allocate(capacity);
    cuuint64_t dims[] = {128, capacity}, strides[] = {128 * 2};
    cuuint32_t box[] = {64, 128}, element_strides[] = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(&store_o_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        partial_o_.data, dims, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    auto const uploaded = std::chrono::steady_clock::now();
    std::cerr << "MSA untimed preparation: selection+CSR_ms="
              << std::chrono::duration<double, std::milli>(built - start).count()
              << " upload+workspace_ms="
              << std::chrono::duration<double, std::milli>(uploaded - built).count()
              << " items=" << num_items_ << " local_items=" << host.local_items
              << " grouped_items=" << host.grouped_items << " per_block_items=" << host.per_block_items
              << " partial_rows=" << host.rows.size()
              << " partial_capacity_bytes=" << capacity * (256 + 8) << '\n';
  }
  int32_t const* selection() const { return selection_.data; }
  MsaLoadParams load(LoadParams<true> const& base, FamilyTypes<true>::Element const* q) const {
    cute::TmaDescriptor packed{};
    cuuint64_t dims[] = {128, 16, 4, cuuint64_t(seqlen_)};
    cuuint64_t strides[] = {128 * 2, 16 * 128 * 2, 64 * 128 * 2};
    cuuint32_t box[] = {64, 16, 1, 1}, element_strides[] = {1, 1, 1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(&packed, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        const_cast<FamilyTypes<true>::Element*>(q), dims, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return {base.tma_load_Q, base.tma_load_K, base.tma_load_V, packed, items_.data,
            tokens_.data, blocks_.data, num_items_, seqlen_};
  }
  MsaPartialParams partial() const {
    return {store_o_, partial_m_.data, partial_l_.data};
  }
  void combine(FamilyTypes<true>::Element* output, float* lse, float scale) const {
    if (partial_shift_ <= 1)
      msa_combine<true><<<(seqlen_ * 64 + 7) / 8, 128>>>(
          partial_o_.data, partial_m_.data, partial_l_.data, inverse_.data, counts_.data,
          partial_shift_, seqlen_, scale, output, lse);
    else
      msa_combine<false><<<(seqlen_ * 64 + 7) / 8, 128>>>(
          partial_o_.data, partial_m_.data, partial_l_.data, inverse_.data, counts_.data,
          partial_shift_, seqlen_, scale, output, lse);
    CUDA_CHECK(cudaGetLastError());
  }
};

}  // namespace pipelink::attention::sm90
