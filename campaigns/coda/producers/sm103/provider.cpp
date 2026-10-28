#include "params.h"
#include "provider.h"
#include <climits>
#include <cstring>
#include <new>

namespace {
struct Shape {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t stagger = 22500;
};

bool shape_from(const plk_shared_scalars_v2* shared, Shape& shape) {
  if (!shared) return false;
  auto* records = reinterpret_cast<const plk_scalar_v2*>(
      reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
  for (uint32_t i = 0; i < shared->record_count; ++i) {
    const auto& record = records[i];
    if (record.scalar_id < 1 || record.scalar_id > 4) continue;
    if (record.scalar_type != PLK_SCALAR_I64) return false;
    int64_t value;
    std::memcpy(&value, &record.value_bits, sizeof(value));
    if (record.scalar_id == 1) shape.m = value;
    if (record.scalar_id == 2) shape.n = value;
    if (record.scalar_id == 3) shape.k = value;
    if (record.scalar_id == 4) shape.stagger = value;
  }
  // Device loop counters and tile coordinates are signed 32-bit values.
  return shape.m > 0 && shape.m <= INT_MAX && shape.m % 256 == 0 &&
         shape.n > 0 && shape.n <= INT_MAX && shape.n % 256 == 0 &&
         shape.k > 0 && shape.k <= INT_MAX && shape.k % 64 == 0 && shape.stagger >= 0 && shape.stagger <= UINT32_MAX;
}

bool header_is(const plk_header_v2& header, size_t bytes) {
  return header.abi_version == PLK_HOST_ABI_V2 && header.struct_bytes == bytes;
}

CUresult encode(CUtensorMap& descriptor, uint64_t address, uint64_t rows, uint64_t k) {
  const cuuint64_t dimensions[2] = {k, rows};
  const cuuint64_t strides[1] = {k * sizeof(uint16_t)};
  const cuuint32_t box[2] = {64, 128};
  const cuuint32_t element_strides[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      reinterpret_cast<void*>(address), dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace

extern "C" int pipelink_params_query_coda_sm103(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
  if (!common || !out || args) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) ||
      !header_is(out->header, sizeof(*out))) return PLK_E_VERSION;
  if (common->channel_count != 2 || !common->channels ||
      out->channel_capacity != 2 || !out->channels || out->workspace_capacity)
    return PLK_E_CONTRACT;
  Shape shape;
  if (!shape_from(common->shared, shape)) return PLK_E_ARGUMENT;
  for (uint32_t i = 0; i < 2; ++i) {
    const auto& source = common->channels[i];
    auto& demand = out->channels[i];
    if (source.channel_id != i + 1 || demand.channel_id != source.channel_id ||
        source.rank != 2 || demand.rank != 2 || source.access != PLK_ACCESS_READ ||
        !source.extents || !source.bit_strides || !demand.extents || !demand.bit_strides)
      return PLK_E_CONTRACT;
    const uint64_t rows = static_cast<uint64_t>(i == 0 ? shape.m : shape.n);
    if (source.origin_bit_offset || source.extents[0] != rows ||
        source.extents[1] != static_cast<uint64_t>(shape.k) ||
        source.bit_strides[0] != shape.k * 16 || source.bit_strides[1] != 16)
      return PLK_E_ARGUMENT;
    demand.dtype_id = source.dtype_id;
    demand.access = PLK_ACCESS_READ;
    demand.minimum_allocation_bytes = rows * static_cast<uint64_t>(shape.k) * sizeof(uint16_t);
    demand.alignment_bytes = 16;
    demand.extents[0] = rows;
    demand.extents[1] = static_cast<uint64_t>(shape.k);
    demand.bit_strides[0] = shape.k * 16;
    demand.bit_strides[1] = 16;
  }
  out->channel_count = 2;
  out->workspace_count = 0;
  out->geometry = {1, 1,
      {static_cast<uint32_t>((shape.m / 128) * (shape.n / 256)), 1, 1},
      common->block, {2, 1, 1}, 0, common->dynamic_smem_bytes};
  return PLK_OK;
}

extern "C" int pipelink_params_build_coda_sm103(
    const plk_common_v2* common, const void* args, const plk_resources_v2* resources,
    void* out, size_t cap, plk_transfer_report_v2*, uint32_t transfer_capacity,
    plk_session** session) {
  if (!session) return PLK_E_ARGUMENT;
  *session = nullptr;
  if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) ||
      !header_is(resources->header, sizeof(*resources))) return PLK_E_VERSION;
  if (cap < sizeof(CodaSm103Params) || transfer_capacity) return PLK_E_CAPACITY;
  if (resources->channel_count != 2 || !resources->channels || resources->workspace_count)
    return PLK_E_CONTRACT;
  Shape shape;
  if (!shape_from(resources->shared, shape)) return PLK_E_ARGUMENT;
  uint64_t addresses[2];
  for (uint32_t i = 0; i < 2; ++i) {
    const auto& resource = resources->channels[i];
    if (resource.channel_id != i + 1 || resource.access != PLK_ACCESS_READ)
      return PLK_E_CONTRACT;
    const uint64_t rows = static_cast<uint64_t>(i == 0 ? shape.m : shape.n);
    const uint64_t bytes = rows * static_cast<uint64_t>(shape.k) * sizeof(uint16_t);
    if (!resource.allocation_address || resource.origin_bit_offset ||
        resource.origin_byte_offset > resource.allocation_bytes ||
        bytes > resource.allocation_bytes - resource.origin_byte_offset ||
        resource.allocation_address > UINT64_MAX - resource.origin_byte_offset)
      return PLK_E_RESOURCE;
    addresses[i] = resource.allocation_address + resource.origin_byte_offset;
    if (addresses[i] % 16) return PLK_E_RESOURCE;
  }
  // The broker makes the caller's CUDA context current before entering build.
  // Tensor maps are stored inline, so the provider allocates no GPU memory and
  // keeps no session state across invocations with different dimensions or resources.
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) return PLK_E_RESOURCE;
  auto& residue = *new (out) CodaSm103Params{};
  if (encode(residue.a, addresses[0], shape.m, shape.k) != CUDA_SUCCESS ||
      encode(residue.b, addresses[1], shape.n, shape.k) != CUDA_SUCCESS)
    return PLK_E_BUILD;
  CUdevice device;
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&residue.num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) != CUDA_SUCCESS) return PLK_E_BUILD;
  residue.stagger_ns = static_cast<uint32_t>(shape.stagger);
  return PLK_OK;
}

extern "C" int pipelink_params_release_coda_sm103(plk_session* session) {
  return session ? PLK_E_RELEASE : PLK_OK;
}
