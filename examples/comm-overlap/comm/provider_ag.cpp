// Comm owner provider, AllGather group: encodes the peer address tables into the residue.
#include "comm.h"
#include <climits>
#include <cstring>
#include <new>

namespace {
enum Channel { Gather = 0, Ready = 1, Released = 2, Gen = 3, Count = 4 };

struct Inputs {
  int64_t m = 0, k = 0, chunk_rows = 0, chunk_cols = 0, slot = -1;
};

bool inputs_from(const plk_shared_scalars_v2* shared, uint32_t world, Inputs& in) {
  if (!shared || world == 0 || world > COMM_MAX_WORLD) return false;
  auto* records = reinterpret_cast<const plk_scalar_v2*>(
      reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
  for (uint32_t i = 0; i < shared->record_count; ++i) {
    const auto& record = records[i];
    int64_t* field = record.scalar_id == 1 ? &in.m : record.scalar_id == 3 ? &in.k :
                     record.scalar_id == 4 ? &in.chunk_rows : record.scalar_id == 5 ? &in.chunk_cols :
                     record.scalar_id == 6 ? &in.slot : nullptr;
    if (!field) continue;
    if (record.scalar_type != PLK_SCALAR_I64) return false;
    std::memcpy(field, &record.value_bits, sizeof(int64_t));
  }
  // Rows split evenly into whole chunks per rank; K into whole 16-byte-per-lane chunks.
  return in.m > 0 && in.m <= INT_MAX && in.chunk_rows > 0 && in.m % (in.chunk_rows * world) == 0 &&
         in.k > 0 && in.k <= INT_MAX && in.chunk_cols > 0 && in.k % in.chunk_cols == 0 && in.slot >= 0;
}

void vector(plk_channel_demand_v2& demand, uint64_t elements) {
  demand.minimum_allocation_bytes = elements * sizeof(uint32_t);
  demand.alignment_bytes = 16;
  demand.extents[0] = elements;
  demand.bit_strides[0] = 32;
}
}  // namespace

extern "C" int pipelink_params_query_comm_ag(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
  if (!common || !out || args) return PLK_E_ARGUMENT;
  if (common->header.abi_version != PLK_HOST_ABI_V2) return PLK_E_VERSION;
  if (common->channel_count != Count || out->channel_capacity != Count || out->workspace_capacity != 1)
    return PLK_E_CONTRACT;
  const auto* domain = plk_find_domain(common, 1);
  if (!domain) return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs_from(common->shared, domain->world, in)) return PLK_E_ARGUMENT;
  for (uint32_t i = 0; i < Count; ++i) {
    const auto& source = common->channels[i];
    auto& demand = out->channels[i];
    demand.dtype_id = source.dtype_id;
    demand.access = PLK_ACCESS_READ_WRITE;
    if (i == Gather) {
      demand.minimum_allocation_bytes = uint64_t(in.m) * uint64_t(in.k) * 2;
      demand.alignment_bytes = 16;  // 16-byte vector copies
      demand.extents[0] = uint64_t(in.m);
      demand.extents[1] = uint64_t(in.k);
      demand.bit_strides[0] = in.k * 16;
      demand.bit_strides[1] = 16;
    } else if (i == Ready) {
      if (source.extents[0] < uint64_t(in.slot + in.m / in.chunk_rows)) return PLK_E_ARGUMENT;
      vector(demand, source.extents[0]);
    } else {
      vector(demand, i == Released ? domain->world : 1);
    }
  }
  out->workspaces[0].bytes = 2 * sizeof(uint32_t);
  out->workspaces[0].alignment_bytes = 4;
  out->channel_count = Count;
  out->workspace_count = 1;
  out->geometry = {};  // The GEMM owner chooses the grid.
  return PLK_OK;
}

extern "C" int pipelink_params_build_comm_ag(
    const plk_common_v2* common, const void* args, const plk_resources_v2* resources,
    void* out, size_t cap, plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
  if (!session) return PLK_E_ARGUMENT;
  *session = nullptr;
  if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
  if (cap < sizeof(CommAgResidue) || transfer_capacity) return PLK_E_CAPACITY;
  if (resources->channel_count != Count || resources->workspace_count != 1) return PLK_E_CONTRACT;
  const auto* domain = plk_find_domain(common, 1);
  if (!domain) return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs_from(resources->shared, domain->world, in)) return PLK_E_ARGUMENT;
  auto& residue = *new (out) CommAgResidue{};
  uint64_t* tables[] = {residue.gather, residue.ready, residue.released};
  for (uint32_t i = 0; i < Count; ++i) {
    const auto& resource = resources->channels[i];
    if (resource.origin_bit_offset) return PLK_E_RESOURCE;
    if (i == Gen) {
      residue.generation = reinterpret_cast<uint32_t*>(resource.allocation_address + resource.origin_byte_offset);
      continue;
    }
    if (!resource.peer_addresses) return PLK_E_RESOURCE;
    for (uint32_t r = 0; r < domain->world; ++r)
      tables[i][r] = resource.peer_addresses[r] + resource.origin_byte_offset;
  }
  residue.tickets = reinterpret_cast<uint32_t*>(resources->workspaces[0].device_address);
  residue.rank = int32_t(domain->rank);
  residue.world = int32_t(domain->world);
  residue.rows_per_rank = int32_t(in.m / domain->world);
  residue.K = int32_t(in.k);
  return PLK_OK;
}

extern "C" int pipelink_params_release_comm_ag(plk_session* session) {
  return session ? PLK_E_RELEASE : PLK_OK;
}
