#include "abi.h"
#include "provider.h"
#include <climits>
#include <cstring>
#include <new>

namespace {
bool shape_from(plk_shared_scalars_v2 const* shared, int64_t (&shape)[3]) {
  if (!shared) return false;
  auto records = reinterpret_cast<plk_scalar_v2 const*>(
      reinterpret_cast<unsigned char const*>(shared) + shared->record_offset);
  for (uint32_t i = 0; i < shared->record_count; ++i) {
    auto const& r = records[i];
    if (r.scalar_id < 1 || r.scalar_id > 3) continue;
    if (r.scalar_type != PLK_SCALAR_I64) return false;
    std::memcpy(&shape[r.scalar_id - 1], &r.value_bits, sizeof(int64_t));
  }
  return shape[0] > 0 && shape[0] <= INT_MAX && shape[0] % 128 == 0 &&
      shape[1] > 0 && shape[1] <= INT_MAX && shape[1] % 256 == 0 &&
      shape[2] > 0 && shape[2] <= INT_MAX && shape[2] % 32 == 0;
}
bool header_is(plk_header_v2 const& h, size_t bytes) {
  return h.abi_version == PLK_HOST_ABI_V2 && h.struct_bytes == bytes;
}
}
extern "C" int pipelink_params_query_coda_sm120(
    plk_common_v2 const* common, void const* args, plk_demands_v2* out) {
  if (!common || args || !out) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) || !header_is(out->header, sizeof(*out)))
    return PLK_E_VERSION;
  if (common->channel_count != 2 || !common->channels || out->channel_capacity != 2 ||
      !out->channels || out->workspace_capacity) return PLK_E_CONTRACT;
  int64_t shape[3]{};
  if (!shape_from(common->shared, shape)) return PLK_E_ARGUMENT;
  for (uint32_t i = 0; i < 2; ++i) {
    auto const& r = common->channels[i];
    auto& d = out->channels[i];
    uint64_t rows = shape[i], k = shape[2];
    if (r.channel_id != i + 1 || d.channel_id != i + 1 || r.rank != 2 || d.rank != 2 ||
        r.access != PLK_ACCESS_READ || !r.extents || !r.bit_strides || !d.extents || !d.bit_strides)
      return PLK_E_CONTRACT;
    if (r.origin_bit_offset || r.extents[0] != rows || r.extents[1] != k ||
        r.bit_strides[0] != int64_t(k * 16) || r.bit_strides[1] != 16) return PLK_E_ARGUMENT;
    d.dtype_id = r.dtype_id;
    d.access = PLK_ACCESS_READ;
    d.minimum_allocation_bytes = rows * k * 2;
    d.alignment_bytes = 16;
    d.extents[0] = rows; d.extents[1] = k;
    d.bit_strides[0] = k * 16; d.bit_strides[1] = 16;
  }
  out->channel_count = 2;
  out->workspace_count = 0;
  out->geometry = {1, 1, {uint32_t((shape[0] / 128) * (shape[1] / 256)), 1, 1},
                   common->block, {1, 1, 1}, 0, common->dynamic_smem_bytes};
  return PLK_OK;
}
extern "C" int pipelink_params_build_coda_sm120(
    plk_common_v2 const* common, void const* args, plk_resources_v2 const* resources,
    void* out, size_t capacity, plk_transfer_report_v2*, uint32_t transfers, plk_session** session) {
  if (!session) return PLK_E_ARGUMENT;
  *session = nullptr;
  if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) ||
      !header_is(resources->header, sizeof(*resources))) return PLK_E_VERSION;
  if (capacity < sizeof(CodaSm120Residue) || transfers) return PLK_E_CAPACITY;
  if (resources->channel_count != 2 || !resources->channels || resources->workspace_count)
    return PLK_E_CONTRACT;
  int64_t shape[3]{};
  if (!shape_from(resources->shared, shape)) return PLK_E_ARGUMENT;
  uint64_t addresses[2];
  for (uint32_t i = 0; i < 2; ++i) {
    auto const& r = resources->channels[i];
    uint64_t bytes = uint64_t(shape[i]) * shape[2] * 2;
    if (r.channel_id != i + 1 || r.access != PLK_ACCESS_READ) return PLK_E_CONTRACT;
    if (!r.allocation_address || r.origin_bit_offset || r.origin_byte_offset > r.allocation_bytes ||
        bytes > r.allocation_bytes - r.origin_byte_offset ||
        r.allocation_address > UINT64_MAX - r.origin_byte_offset) return PLK_E_RESOURCE;
    addresses[i] = r.allocation_address + r.origin_byte_offset;
    if (addresses[i] % 16) return PLK_E_RESOURCE;
  }
  new (out) CodaSm120Residue{reinterpret_cast<void const*>(addresses[0]),
                            reinterpret_cast<void const*>(addresses[1])};
  return PLK_OK;
}
extern "C" int pipelink_params_release_coda_sm120(plk_session* session) {
  return session ? PLK_E_RELEASE : PLK_OK;
}
