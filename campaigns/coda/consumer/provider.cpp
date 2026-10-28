#include "params.h"
#include <cstring>
#include <limits>

namespace coda_epilogue_host {
int dimensions(const plk_common_v2* common, CodaEpilogueParams& params) {
  if (common->header.abi_version != PLK_HOST_ABI_V2) return PLK_E_VERSION;
  const auto* shared = common->shared;
  const auto* fields = reinterpret_cast<const plk_scalar_v2*>(
      reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
  unsigned found = 0;
  for (unsigned i = 0; i < shared->record_count; ++i) {
    const auto& field = fields[i];
    if (field.scalar_id < 1 || field.scalar_id > 3) continue;
    if (field.scalar_type != PLK_SCALAR_I64) return PLK_E_ARGUMENT;
    int64_t value;
    std::memcpy(&value, &field.value_bits, sizeof(value));
    const int alignment = field.scalar_id == 1 ? 128 : field.scalar_id == 2 ? 256 : 64;
    if (value <= 0 || value > std::numeric_limits<int32_t>::max() || value % alignment)
      return PLK_E_ARGUMENT;
    if (field.scalar_id == 1) params.M = int32_t(value);
    if (field.scalar_id == 2) params.N = int32_t(value);
    if (field.scalar_id == 3) params.K = int32_t(value);
    found |= 1u << (field.scalar_id - 1);
  }
  return found == 7 ? PLK_OK : PLK_E_CONTRACT;
}
}

extern "C" int pipelink_params_query_coda_epilogue(
    const plk_common_v2* common, const void*, plk_demands_v2* out) {
  CodaEpilogueParams params{};
  const int status = coda_epilogue_host::dimensions(common, params);
  if (status != PLK_OK) return status;
  if (out->channel_capacity != 3 || common->channel_count != 3) return PLK_E_CAPACITY;
  for (unsigned i = 0; i < 3; ++i) {
    auto& demand = out->channels[i];
    const auto& channel = common->channels[i];
    const bool a = channel.channel_id == 1, b = channel.channel_id == 2;
    const int rows = b ? params.N : params.M;
    const int cols = a || b ? params.K : params.N / 2;
    demand.dtype_id = channel.dtype_id;
    demand.access = channel.access;
    demand.minimum_allocation_bytes = uint64_t(rows) * cols * 2;
    demand.alignment_bytes = 16;
    demand.extents[0] = rows;
    demand.extents[1] = cols;
    demand.bit_strides[0] = int64_t(cols) * 16;
    demand.bit_strides[1] = 16;
  }
  out->channel_count = 3;
  out->workspace_count = 0;
  out->geometry = {};
  return PLK_OK;
}

extern "C" int pipelink_params_build_coda_epilogue(
    const plk_common_v2* common, const void*, const plk_resources_v2* resources,
    void* residue, size_t capacity, plk_transfer_report_v2*, unsigned transfer_capacity,
    plk_session** session) {
  *session = nullptr;
  if (capacity != sizeof(CodaEpilogueParams) || transfer_capacity || resources->channel_count != 3)
    return PLK_E_CAPACITY;
  CodaEpilogueParams params{};
  const int status = coda_epilogue_host::dimensions(common, params);
  if (status != PLK_OK) return status;
  for (unsigned i = 0; i < 3; ++i) {
    const auto& channel = resources->channels[i];
    void* pointer = reinterpret_cast<void*>(channel.allocation_address + channel.origin_byte_offset);
    if (channel.channel_id == 1) params.a = pointer;
    if (channel.channel_id == 2) params.b = pointer;
    if (channel.channel_id == 3) params.output = pointer;
  }
  std::memcpy(residue, &params, sizeof(params));
  return PLK_OK;
}

extern "C" int pipelink_params_release_coda_epilogue(plk_session*) {
  // Resources are borrowed; the provider allocates no workspace or session state.
  return PLK_OK;
}
