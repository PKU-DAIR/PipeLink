#include "epilogue.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace scaled_store {
int inputs(plk_common_v2 const* common, ScaledStoreResidue& params) {
  if (common->header.abi_version != PLK_HOST_ABI_V2) return PLK_E_VERSION;
  auto const* shared = common->shared;
  auto const* fields = reinterpret_cast<plk_scalar_v2 const*>(
      reinterpret_cast<unsigned char const*>(shared) + shared->record_offset);
  unsigned found = 0;
  for (unsigned i = 0; i < shared->record_count; ++i) {
    auto const& field = fields[i];
    if (field.scalar_id == 1 || field.scalar_id == 2) {
      if (field.scalar_type != PLK_SCALAR_I64) return PLK_E_ARGUMENT;
      int64_t value;
      std::memcpy(&value, &field.value_bits, sizeof(value));
      if (value <= 0 || value > std::numeric_limits<int32_t>::max() || value % 128)
        return PLK_E_ARGUMENT;
      (field.scalar_id == 1 ? params.M : params.N) = int32_t(value);
      found |= 1u << (field.scalar_id - 1);
    } else if (field.scalar_id == 4) {
      if (field.scalar_type != PLK_SCALAR_F64) return PLK_E_ARGUMENT;
      double value;
      std::memcpy(&value, &field.value_bits, sizeof(value));
      params.alpha = float(value);
      if (!std::isfinite(params.alpha)) return PLK_E_ARGUMENT;
      found |= 4;
    }
  }
  return found == 7 ? PLK_OK : PLK_E_CONTRACT;
}
}

extern "C" int pipelink_params_query_user(
    plk_common_v2 const* common, void const*, plk_demands_v2* out) {
  ScaledStoreResidue params{};
  const int status = scaled_store::inputs(common, params);
  if (status != PLK_OK) return status;
  if (out->channel_capacity != 1 || common->channel_count != 1) return PLK_E_CAPACITY;
  auto& demand = out->channels[0];
  demand.dtype_id = common->channels[0].dtype_id;
  demand.access = common->channels[0].access;
  demand.minimum_allocation_bytes = uint64_t(params.M) * params.N * sizeof(float);
  demand.alignment_bytes = alignof(float);
  demand.extents[0] = params.M;
  demand.extents[1] = params.N;
  demand.bit_strides[0] = int64_t(params.N) * 32;
  demand.bit_strides[1] = 32;
  out->channel_count = 1;
  out->workspace_count = 0;
  out->geometry = {};  // The GEMM owner, not its epilogue consumer, chooses the grid.
  return PLK_OK;
}

extern "C" int pipelink_params_build_user(
    plk_common_v2 const* common, void const*, plk_resources_v2 const* resources,
    void* residue, size_t capacity, plk_transfer_report_v2*, unsigned transfer_capacity,
    plk_session** session) {
  *session = nullptr;
  if (capacity != sizeof(ScaledStoreResidue) || transfer_capacity || resources->channel_count != 1)
    return PLK_E_CAPACITY;
  ScaledStoreResidue params{};
  const int status = scaled_store::inputs(common, params);
  if (status != PLK_OK) return status;
  auto const& output = resources->channels[0];
  params.output = reinterpret_cast<float*>(output.allocation_address + output.origin_byte_offset);
  std::memcpy(residue, &params, sizeof(params));
  return PLK_OK;
}

extern "C" int pipelink_params_release_user(plk_session*) {
  return PLK_OK;  // Caller-owned resources; no provider allocation or persistent state.
}
