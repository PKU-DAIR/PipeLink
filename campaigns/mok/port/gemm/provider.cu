// Owner-private descriptors remain outside the callable ABI.
#ifndef MOK_GEMM_TYPES_ONLY
#define MOK_GEMM_TYPES_ONLY
#endif
#include "gemm.cu"
#include <pipelink_host_abi_v2.h>
#include <climits>
#include <cstring>

namespace {
using Config = mok_gemm_owner::config;
constexpr uint32_t channel_count = 69;
struct Inputs {
  int64_t value[32]{};
  int L() const { return int(value[MOK_SCALAR_NUM_LOCAL_TOKENS]); }
  int H() const { return int(value[MOK_SCALAR_HIDDEN]); }
  int I() const { return int(value[MOK_SCALAR_INTERMEDIATE]); }
  int E() const { return int(value[MOK_SCALAR_NUM_LOCAL_EXPERTS]); }
  int A() const { return int(value[MOK_SCALAR_MACROBATCH_SIZE]); }
  int pass() const { return int(value[MOK_SCALAR_PASS]); }
  bool fp8() const { return value[MOK_SCALAR_PRECISION] == MOK_PRECISION_MXFP8; }
};
bool inputs(const plk_shared_scalars_v2* shared, Inputs& in) {
  if (!shared) return false;
  const auto* records = reinterpret_cast<const plk_scalar_v2*>(
      reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
  uint32_t seen = 0;
  for (uint32_t i = 0; i < shared->record_count; ++i) {
    const auto& r = records[i];
    const auto id = r.scalar_id;
    if (!((id >= 1 && id <= 6) || (id >= 11 && id <= 13) || (id >= 20 && id <= 23) || id == 31)) continue;
    if (r.scalar_type != PLK_SCALAR_I64) return false;
    std::memcpy(&in.value[id], &r.value_bits, sizeof(int64_t));
    seen |= uint32_t(1) << id;
    if (in.value[id] < 0 || in.value[id] > INT_MAX) return false;
  }
  for (int id : {1, 2, 3, 4, 5, 6, 11, 12, 13, 20, 21, 22, 23, 31})
    if (!(seen & (uint32_t(1) << id))) return false;
  for (int id : {1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 31})
    if (in.value[id] <= 0) return false;
  return in.pass() <= MOK_PASS_REC && in.value[12] <= MOK_PRECISION_MXFP8 &&
      (in.value[13] == 1 || in.value[13] == 4) &&
      in.L() >= 512 && in.L() % Config::MLP_Mb == 0 &&
      in.H() % Config::MLP_Nb == 0 && in.I() % Config::MLP_Nb == 0 &&
      in.A() % Config::MLP_Mb == 0 && in.value[5] % Config::MLP_Mb == 0 &&
      in.value[31] % Config::MLP_Mb == 0 &&
      int64_t(in.E()) * (in.H() / Config::QUANT_Mb) <= INT_MAX &&
      int64_t(in.E()) * (in.I() / Config::QUANT_Mb) <= INT_MAX;
}
bool topology(const plk_common_v2& common, const plk_domain_v2& ep) {
  return common.block.x == Config::NUM_THREADS && common.block.y == 1 && common.block.z == 1 &&
      common.cluster.x == Config::CLUSTER_SIZE && common.cluster.y == 1 && common.cluster.z == 1 &&
      ep.rank < ep.world;
}
struct Shape {
  uint32_t rank, bits, access;
  uint64_t extents[4];
};
bool active(uint32_t id, const Inputs& in) {
  const bool bwd = in.pass() == MOK_PASS_BWD, fwd = in.pass() == MOK_PASS_FWD;
  if (id == 6) return in.pass() != MOK_PASS_REC;
  if (id == 24 || id == 25) return fwd;
  if (id >= 27 && id <= 39) return bwd;
  if (id >= 62) return !in.fp8() && (id < 66 || (id == 66 ? in.pass() != MOK_PASS_REC : bwd));
  if (id >= 40) return in.fp8() && (id <= 43 || bwd);
  if (id == 2 || id == 3 || (id >= 7 && id <= 12) || (id >= 17 && id <= 20) || id == 22 || id == 23)
    return in.fp8() && ((id == 11 || id == 12) ? fwd : true);
  return true;
}
Shape shape(uint32_t id, const Inputs& in) {
  const uint64_t L = in.L(), H = in.H(), I = in.I(), E = in.E(), A = in.A();
  Shape s{};
  switch (id) {
    case 1: case 27: s = {2,16,PLK_ACCESS_READ,{L,H}}; break;
    case 2: case 50: s = {2,8,PLK_ACCESS_READ,{A,H}}; break;
    case 3: case 51: s = {4,8,PLK_ACCESS_READ,{A/128,H/128,32,16}}; break;
    case 4: case 5: s = {2,16,PLK_ACCESS_READ,{I,H}}; break;
    case 6: s = {2,16,PLK_ACCESS_READ,{H,I}}; break;
    case 7: case 9: case 48: s = {3,8,PLK_ACCESS_READ,{E,I,H}}; break;
    case 8: case 10: case 49: s = {4,8,PLK_ACCESS_READ,{E*I/128,H/128,32,16}}; break;
    case 11: case 44: case 46: s = {3,8,PLK_ACCESS_READ,{E,H,I}}; break;
    case 12: case 45: case 47: s = {4,8,PLK_ACCESS_READ,{E*H/128,I/128,32,16}}; break;
    case 13: case 14: case 28: s = {2,16,PLK_ACCESS_WRITE,{L,I}}; break;
    case 15: case 16: case 29: s = {2,16,PLK_ACCESS_WRITE,{A,I}}; break;
    case 17: case 19: s = {2,8,PLK_ACCESS_WRITE,{A,I}}; break;
    case 18: case 20: s = {4,8,PLK_ACCESS_WRITE,{A/128,I/128,32,16}}; break;
    case 21: case 30: case 31: s = {2,16,PLK_ACCESS_READ,{L,I}}; break;
    case 22: case 54: case 58: s = {2,8,PLK_ACCESS_READ,{A,I}}; break;
    case 23: case 55: case 59: s = {4,8,PLK_ACCESS_READ,{A/128,I/128,32,16}}; break;
    case 24: case 32: s = {2,16,PLK_ACCESS_WRITE,{L,H}}; break;
    case 25: case 33: s = {2,16,PLK_ACCESS_WRITE,{A,H}}; break;
    case 26: s = {1,32,PLK_ACCESS_READ,{E}}; break;
    case 34: case 35: s = {2,16,PLK_ACCESS_WRITE,{I,H}}; break;
    case 36: s = {2,16,PLK_ACCESS_WRITE,{H,I}}; break;
    case 37: case 38: s = {3,16,PLK_ACCESS_READ | PLK_ACCESS_WRITE,{E,I,H}}; break;
    case 39: s = {3,16,PLK_ACCESS_READ | PLK_ACCESS_WRITE,{E,H,I}}; break;
    case 40: case 52: s = {2,8,PLK_ACCESS_READ,{H,A}}; break;
    case 41: case 53: s = {4,8,PLK_ACCESS_READ,{H/128,A/128,32,16}}; break;
    case 42: case 56: case 60: s = {2,8,PLK_ACCESS_READ,{I,A}}; break;
    case 43: case 57: case 61: s = {4,8,PLK_ACCESS_READ,{I/128,A/128,32,16}}; break;
    case 62: case 67: s = {2,16,PLK_ACCESS_READ,{A,H}}; break;
    case 63: case 68: case 69: s = {2,16,PLK_ACCESS_READ,{A,I}}; break;
    case 64: case 65: s = {3,16,PLK_ACCESS_READ,{E,I,H}}; break;
    case 66: s = {3,16,PLK_ACCESS_READ,{E,H,I}}; break;
  }
  if (!active(id, in)) for (auto& e : s.extents) e = 0;
  return s;
}
bool byte_size(const Shape& s, uint64_t& bytes) {
  bytes = s.bits / 8;
  for (uint32_t d = 0; d < s.rank; ++d) {
    if (!s.extents[d]) { bytes = 0; return true; }
    if (bytes > uint64_t(INT64_MAX / 8) / s.extents[d]) return false;
    bytes *= s.extents[d];
  }
  return true;
}
template<class GL>
void descriptor(void* out, size_t offset, const uint64_t* addresses, uint32_t id, int b, int d, int r, int c) {
  if (!addresses[id - 1]) return;
  const auto gl = kittens::make_gl<GL>(addresses[id - 1], b, d, r, c);
  std::memcpy(static_cast<unsigned char*>(out) + offset, &gl, sizeof(gl));
}
template<mok_port::RoutedPrecision Precision>
void build_residue(void* out, const uint64_t* a, const Inputs& in) {
  using G = mok_gemm_owner::Owner<Precision>;
  using R = typename G::Residue;
  const int L = in.L(), H = in.H(), I = in.I(), E = in.E(), A = in.A();
#define DESC(MEMBER, TYPE, ID, B, D, ROWS, COLS) descriptor<typename G::TYPE>(out, offsetof(R, MEMBER), a, ID, B, D, ROWS, COLS)
  DESC(x_shared, mlp_bf16_gl, 1, 1, 1, L, H);
  DESC(hidden_shared, mlp_bf16_gl, 21, 1, 1, L, I);
  DESC(w_shared_gate, weight_bf16_gl, 4, 1, 1, I, H);
  DESC(w_shared_up, weight_bf16_gl, 5, 1, 1, I, H);
  DESC(w_shared_down, weight_bf16_gl, 6, 1, 1, H, I);
  DESC(gate_shared, epi_bf16_gl, 13, 1, 1, L, I);
  DESC(up_shared, epi_bf16_gl, 14, 1, 1, L, I);
  DESC(gate_routed, epi_bf16_gl, 15, 1, 1, A, I);
  DESC(up_routed, epi_bf16_gl, 16, 1, 1, A, I);
  DESC(y_shared, epi_bf16_gl, 24, 1, 1, L, H);
  DESC(y_routed, epi_bf16_gl, 25, 1, 1, A, H);
  DESC(x_shared_wgrad, wgrad_bf16_gl, 1, 1, 1, L, H);
  DESC(hidden_shared_wgrad, wgrad_bf16_gl, 21, 1, 1, L, I);
  DESC(d_y_shared, mlp_bf16_gl, 27, 1, 1, L, H);
  DESC(d_hidden_shared, epi_bf16_gl, 28, 1, 1, L, I);
  DESC(d_hidden_routed, epi_bf16_gl, 29, 1, 1, A, I);
  DESC(d_gate_shared, mlp_bf16_gl, 30, 1, 1, L, I);
  DESC(d_up_shared, mlp_bf16_gl, 31, 1, 1, L, I);
  DESC(d_x_shared, epi_bf16_gl, 32, 1, 1, L, H);
  DESC(d_x_routed, epi_bf16_gl, 33, 1, 1, A, H);
  DESC(d_w_shared_gate, d_weight_bf16_gl, 34, 1, 1, I, H);
  DESC(d_w_shared_up, d_weight_bf16_gl, 35, 1, 1, I, H);
  DESC(d_w_shared_down, d_weight_bf16_gl, 36, 1, 1, H, I);
  DESC(d_w_routed_gate, d_weight_bf16_gl, 37, 1, E, I, H);
  DESC(d_w_routed_up, d_weight_bf16_gl, 38, 1, E, I, H);
  DESC(d_w_routed_down, d_weight_bf16_gl, 39, 1, E, H, I);
  if constexpr (G::USE_MXFP8) {
#define ACTIVATION(MEMBER, SC, ID, R, C) \
    DESC(MEMBER, routed_activation_gl, ID, 1, 1, R, C); \
    DESC(SC, routed_sc_gl, ID+1, R/128, C/128, 32, 16)
#define WEIGHT(MEMBER, SC, ID, R, C) \
    DESC(MEMBER, routed_weight_gl, ID, 1, E, R, C); \
    DESC(SC, routed_sc_gl, ID+1, int(int64_t(E)*R/128), C/128, 32, 16)
    ACTIVATION(x_fp8_routed, x_sc_routed, 2, A, H);
    ACTIVATION(hidden_fp8_routed, hidden_sc_routed, 22, A, I);
    ACTIVATION(x_fp8_t_routed, x_sc_t_routed, 40, H, A);
    ACTIVATION(hidden_fp8_t_routed, hidden_sc_t_routed, 42, I, A);
    ACTIVATION(d_y_fp8_routed, d_y_sc_routed, 50, A, H);
    ACTIVATION(d_y_fp8_t_routed, d_y_sc_t_routed, 52, H, A);
    ACTIVATION(d_gate_fp8_routed, d_gate_sc_routed, 54, A, I);
    ACTIVATION(d_gate_fp8_t_routed, d_gate_sc_t_routed, 56, I, A);
    ACTIVATION(d_up_fp8_routed, d_up_sc_routed, 58, A, I);
    ACTIVATION(d_up_fp8_t_routed, d_up_sc_t_routed, 60, I, A);
    WEIGHT(w_routed_gate, w_routed_gate_sc, 7, I, H);
    WEIGHT(w_routed_up, w_routed_up_sc, 9, I, H);
    WEIGHT(w_routed_down, w_routed_down_sc, 11, H, I);
    WEIGHT(w_routed_gate_T, w_routed_gate_sc_T, 44, H, I);
    WEIGHT(w_routed_up_T, w_routed_up_sc_T, 46, H, I);
    WEIGHT(w_routed_down_T, w_routed_down_sc_T, 48, I, H);
    DESC(gate_fp8_routed, routed_gate_up_gl, 17, 1, 1, A, I);
    DESC(up_fp8_routed, routed_gate_up_gl, 19, 1, 1, A, I);
    DESC(gate_sc_routed, routed_sc_gl, 18, A/128, I/128, 32, 16);
    DESC(up_sc_routed, routed_sc_gl, 20, A/128, I/128, 32, 16);
#undef ACTIVATION
#undef WEIGHT
  } else {
    DESC(x_fp8_routed, routed_activation_gl, 62, 1, 1, A, H);
    DESC(hidden_fp8_routed, routed_activation_gl, 63, 1, 1, A, I);
    DESC(w_routed_gate, routed_weight_gl, 64, 1, E, I, H);
    DESC(w_routed_up, routed_weight_gl, 65, 1, E, I, H);
    DESC(w_routed_down, routed_weight_gl, 66, 1, E, H, I);
    DESC(gate_fp8_routed, routed_gate_up_gl, 15, 1, 1, A, I);
    DESC(up_fp8_routed, routed_gate_up_gl, 16, 1, 1, A, I);
    DESC(d_y_fp8_routed, routed_activation_gl, 67, 1, 1, A, H);
    DESC(d_gate_fp8_routed, routed_activation_gl, 68, 1, 1, A, I);
    DESC(d_up_fp8_routed, routed_activation_gl, 69, 1, 1, A, I);
#define ALIAS(DEST, SOURCE, TYPE) \
    std::memcpy(static_cast<unsigned char*>(out) + offsetof(R, DEST), \
        static_cast<unsigned char*>(out) + offsetof(R, SOURCE), sizeof(typename G::TYPE))
    ALIAS(x_fp8_t_routed, x_fp8_routed, routed_activation_gl);
    ALIAS(hidden_fp8_t_routed, hidden_fp8_routed, routed_activation_gl);
    ALIAS(d_y_fp8_t_routed, d_y_fp8_routed, routed_activation_gl);
    ALIAS(d_gate_fp8_t_routed, d_gate_fp8_routed, routed_activation_gl);
    ALIAS(d_up_fp8_t_routed, d_up_fp8_routed, routed_activation_gl);
    ALIAS(w_routed_gate_T, w_routed_gate, routed_weight_gl);
    ALIAS(w_routed_up_T, w_routed_up, routed_weight_gl);
    ALIAS(w_routed_down_T, w_routed_down, routed_weight_gl);
#undef ALIAS
  }
#undef DESC
  const typename G::index_gl tokens{reinterpret_cast<int*>(a[25]), E};
  const mok_gemm_owner::Facts facts{int(in.value[20]), int(in.value[21]), int(in.value[22]), int(in.value[23]), int(in.value[31])};
  std::memcpy(static_cast<unsigned char*>(out) + offsetof(R, tokens_per_expert), &tokens, sizeof(tokens));
  std::memcpy(static_cast<unsigned char*>(out) + offsetof(R, facts), &facts, sizeof(facts));
}
} // namespace

extern "C" int pipelink_params_query_mok_gemm(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
  if (!common || args || !out) return PLK_E_ARGUMENT;
  if (common->header.abi_version != PLK_HOST_ABI_V2 || out->header.abi_version != PLK_HOST_ABI_V2)
    return PLK_E_VERSION;
  const auto* ep = plk_find_domain(common, 1);
  if (!ep || !topology(*common, *ep)) return PLK_E_CONTRACT;
  if (common->channel_count != channel_count || !common->channels ||
      out->channel_capacity != channel_count || !out->channels || out->workspace_capacity)
    return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs(common->shared, in) || ep->world != in.value[MOK_SCALAR_WORLD]) return PLK_E_ARGUMENT;
  for (uint32_t i = 0; i < channel_count; ++i) {
    const auto& source = common->channels[i];
    auto& demand = out->channels[i];
    const auto s = shape(i + 1, in);
    if (source.channel_id != i + 1 || demand.channel_id != i + 1 ||
        source.rank != s.rank || demand.rank != s.rank || source.access != s.access ||
        !source.extents || !source.bit_strides || !demand.extents || !demand.bit_strides)
      return PLK_E_CONTRACT;
    uint64_t bytes;
    if (!byte_size(s, bytes)) return PLK_E_ARGUMENT;
    demand.dtype_id = source.dtype_id;
    demand.access = s.access;
    demand.minimum_allocation_bytes = bytes;
    demand.alignment_bytes = i == 25 ? alignof(int32_t) : 16;
    int64_t stride = bytes ? s.bits : 0;
    for (int d = int(s.rank) - 1; d >= 0; --d) {
      if (source.extents[d] != s.extents[d] || source.bit_strides[d] != stride)
        return PLK_E_ARGUMENT;
      demand.extents[d] = s.extents[d];
      demand.bit_strides[d] = stride;
      stride *= int64_t(s.extents[d]);
    }
  }
  out->channel_count = channel_count;
  out->workspace_count = 0;
  out->geometry = {};
  return PLK_OK;
}
extern "C" int pipelink_params_build_mok_gemm(
    const plk_common_v2* common, const void* args, const plk_resources_v2* resources,
    void* out, size_t cap, plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
  if (!session) return PLK_E_ARGUMENT;
  *session = nullptr;
  if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
  if (cap < sizeof(MokGemmResidue) || transfer_capacity) return PLK_E_CAPACITY;
  if (resources->channel_count != channel_count || !resources->channels || resources->workspace_count)
    return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs(resources->shared, in)) return PLK_E_ARGUMENT;
  const auto* ep = plk_find_domain(common, 1);
  if (!ep) return PLK_E_CONTRACT;
  uint64_t addresses[channel_count]{};
  for (uint32_t i = 0; i < channel_count; ++i) {
    const auto& r = resources->channels[i];
    uint64_t bytes;
    if (!byte_size(shape(i + 1, in), bytes)) return PLK_E_ARGUMENT;
    if (r.channel_id != i + 1 || r.origin_bit_offset || r.origin_byte_offset > r.allocation_bytes ||
        bytes > r.allocation_bytes - r.origin_byte_offset) return PLK_E_RESOURCE;
    const bool peer = i == 0 || i == 26;
    if (!bytes) {
      if (r.allocation_address || r.allocation_bytes || r.origin_byte_offset) return PLK_E_RESOURCE;
      if (peer) {
        if (!r.peer_addresses) return PLK_E_RESOURCE;
        for (uint32_t rank = 0; rank < ep->world; ++rank)
          if (r.peer_addresses[rank]) return PLK_E_RESOURCE;
      }
      continue;
    }
    const uint64_t base = peer ? (r.peer_addresses ? r.peer_addresses[ep->rank] : 0) : r.allocation_address;
    if (!base || base > UINT64_MAX - r.origin_byte_offset) return PLK_E_RESOURCE;
    addresses[i] = base + r.origin_byte_offset;
    if (addresses[i] % (i == 25 ? alignof(int32_t) : 16)) return PLK_E_RESOURCE;
  }
  try {
    std::memset(out, 0, sizeof(MokGemmResidue));
    if (in.fp8()) build_residue<mok_port::RoutedPrecision::MXFP8>(out, addresses, in);
    else build_residue<mok_port::RoutedPrecision::BF16>(out, addresses, in);
  } catch (...) {
    return PLK_E_BUILD;
  }
  return PLK_OK;
}
extern "C" int pipelink_params_release_mok_gemm(plk_session* session) {
  return session ? PLK_E_RELEASE : PLK_OK;
}
