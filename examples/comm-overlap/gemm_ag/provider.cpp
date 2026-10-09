// GEMM owner provider for the AllGather host group: the original owner's descriptor
// encoding, with A over this rank's gathered buffer [M, K] and load_ag's readiness inputs.
#include "abi.h"
#include <climits>
#include <cstring>
#include <new>

extern "C" {
int pipelink_params_query_gemm_ag(const plk_common_v2*, const void*, plk_demands_v2*);
int pipelink_params_build_gemm_ag(const plk_common_v2*, const void*, const plk_resources_v2*, void*, size_t,
                                  plk_transfer_report_v2*, uint32_t, plk_session**);
int pipelink_params_release_gemm_ag(plk_session*);
}

namespace {
enum Channel { A = 0, B = 1, Ready = 2, Gen = 3, Count = 4 };

struct Inputs {
  int64_t m = 0, n = 0, k = 0, chunk_rows = 0, chunk_cols = 0, slot = -1;
};

bool inputs_from(const plk_shared_scalars_v2* shared, uint32_t world, Inputs& in) {
  if (!shared || world == 0) return false;
  auto* records = reinterpret_cast<const plk_scalar_v2*>(
      reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
  int64_t* fields[] = {&in.m, &in.n, &in.k, &in.chunk_rows, &in.chunk_cols, &in.slot};
  for (uint32_t i = 0; i < shared->record_count; ++i) {
    const auto& record = records[i];
    if (record.scalar_id < 1 || record.scalar_id > 6) continue;
    if (record.scalar_type != PLK_SCALAR_I64) return false;
    std::memcpy(fields[record.scalar_id - 1], &record.value_bits, sizeof(int64_t));
  }
  // Counter granularity must be this GEMM's 128-row A tile; K splits into whole chunks.
  return in.m > 0 && in.m <= INT_MAX && in.m % (128 * int64_t(world)) == 0 &&
         in.n > 0 && in.n <= INT_MAX && in.n % 128 == 0 &&
         in.k > 0 && in.k <= INT_MAX && in.k % 64 == 0 &&
         in.chunk_rows == 128 && in.chunk_cols > 0 && in.k % in.chunk_cols == 0 && in.slot >= 0;
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

void matrix(plk_channel_demand_v2& demand, uint64_t rows, uint64_t k) {
  demand.minimum_allocation_bytes = rows * k * sizeof(uint16_t);
  demand.alignment_bytes = 16;
  demand.extents[0] = rows;
  demand.extents[1] = k;
  demand.bit_strides[0] = int64_t(k) * 16;
  demand.bit_strides[1] = 16;
}

void vector(plk_channel_demand_v2& demand, uint64_t elements) {
  demand.minimum_allocation_bytes = elements * sizeof(uint32_t);
  demand.alignment_bytes = 4;
  demand.extents[0] = elements;
  demand.bit_strides[0] = 32;
}
}  // namespace

extern "C" int pipelink_params_query_gemm_ag(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
  if (!common || !out || args) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) || !header_is(out->header, sizeof(*out)))
    return PLK_E_VERSION;
  if (common->channel_count != Count || out->channel_capacity != Count || out->workspace_capacity)
    return PLK_E_CONTRACT;
  const auto* domain = plk_find_domain(common, 1);
  if (!domain) return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs_from(common->shared, domain->world, in)) return PLK_E_ARGUMENT;
  for (uint32_t i = 0; i < Count; ++i) {
    const auto& source = common->channels[i];
    auto& demand = out->channels[i];
    demand.dtype_id = source.dtype_id;
    demand.access = PLK_ACCESS_READ;
    if (i == A || i == B) {
      matrix(demand, uint64_t(i == A ? in.m : in.n), uint64_t(in.k));
    } else if (i == Ready) {
      // Caller sizes the slot array; it must cover this link's slot base + M/128 row-blocks.
      if (source.extents[0] < uint64_t(in.slot + in.m / 128)) return PLK_E_ARGUMENT;
      vector(demand, source.extents[0]);
    } else {
      vector(demand, 1);
    }
  }
  out->channel_count = Count;
  out->workspace_count = 0;
  out->geometry = {1, 1, {uint32_t(in.m / 128), uint32_t(in.n / 128), 1},
                   common->block, {1, 1, 1}, 0, common->dynamic_smem_bytes};
  return PLK_OK;
}

extern "C" int pipelink_params_build_gemm_ag(
    const plk_common_v2* common, const void* args, const plk_resources_v2* resources,
    void* out, size_t cap, plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
  if (!session) return PLK_E_ARGUMENT;
  *session = nullptr;
  if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) || !header_is(resources->header, sizeof(*resources)))
    return PLK_E_VERSION;
  if (cap < sizeof(GemmAgResidue) || transfer_capacity) return PLK_E_CAPACITY;
  if (resources->channel_count != Count || resources->workspace_count) return PLK_E_CONTRACT;
  const auto* domain = plk_find_domain(common, 1);
  if (!domain) return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs_from(resources->shared, domain->world, in)) return PLK_E_ARGUMENT;
  uint64_t address[Count];
  for (uint32_t i = 0; i < Count; ++i) {
    const auto& resource = resources->channels[i];
    if (!resource.allocation_address || resource.origin_bit_offset) return PLK_E_RESOURCE;
    address[i] = resource.allocation_address + resource.origin_byte_offset;
  }
  if (address[A] % 16 || address[B] % 16) return PLK_E_RESOURCE;
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) return PLK_E_RESOURCE;
  auto& residue = *new (out) GemmAgResidue{};
  if (encode(residue.gemm.load.a, address[A], uint64_t(in.m), uint64_t(in.k)) != CUDA_SUCCESS ||
      encode(residue.gemm.load.b, address[B], uint64_t(in.n), uint64_t(in.k)) != CUDA_SUCCESS)
    return PLK_E_BUILD;
  residue.gemm.load.K = int32_t(in.k);
  residue.gemm.mma.K = int32_t(in.k);
  residue.ag.ready = reinterpret_cast<const uint32_t*>(address[Ready]);
  residue.ag.gen = reinterpret_cast<const uint32_t*>(address[Gen]);
  residue.ag.rank = int32_t(domain->rank);
  residue.ag.rows_per_rank = int32_t(in.m / domain->world);
  residue.ag.expected = int32_t(in.k / in.chunk_cols);
  return PLK_OK;
}

extern "C" int pipelink_params_release_gemm_ag(plk_session* session) {
  return session ? PLK_E_RELEASE : PLK_OK;
}
