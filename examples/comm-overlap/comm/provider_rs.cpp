// Comm owner provider: peer receive slots/counters/released, local output/generation,
// and a launch-reset follow-on completion ticket. Transport is push-only.
#include "comm.h"
#include <climits>
#include <cstring>
#include <new>

namespace {
enum Channel { Slots = 0, Counters = 1, Output = 2, Released = 3, Gen = 4, Count = 5 };

bool shape_from(const plk_shared_scalars_v2* shared, uint32_t world, int64_t& m, int64_t& n) {
  if (!shared || world < 2 || world > COMM_MAX_WORLD) return false;
  auto* records = reinterpret_cast<const plk_scalar_v2*>(
      reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
  m = n = 0;
  for (uint32_t i = 0; i < shared->record_count; ++i) {
    const auto& record = records[i];
    if (record.scalar_id != 1 && record.scalar_id != 2) continue;
    if (record.scalar_type != PLK_SCALAR_I64) return false;
    std::memcpy(record.scalar_id == 1 ? &m : &n, &record.value_bits, sizeof(int64_t));
  }
  // Each owner holds whole 128-row tiles of the scattered output.
  return m > 0 && m <= INT_MAX && m % (128 * int64_t(world)) == 0 && n > 0 && n <= INT_MAX && n % 128 == 0;
}

void vector(plk_channel_demand_v2& demand, uint64_t elements, uint32_t bits) {
  demand.minimum_allocation_bytes = elements * bits / 8;
  demand.alignment_bytes = 16;
  demand.extents[0] = elements;
  demand.bit_strides[0] = bits;
}
}  // namespace

extern "C" int pipelink_params_query_comm_rs(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
  if (!common || !out || args) return PLK_E_ARGUMENT;
  if (common->header.abi_version != PLK_HOST_ABI_V2) return PLK_E_VERSION;
  if (common->channel_count != Count || out->channel_capacity != Count || out->workspace_capacity != 1)
    return PLK_E_CONTRACT;
  const auto* domain = plk_find_domain(common, 1);
  if (!domain) return PLK_E_CONTRACT;
  int64_t m, n;
  if (!shape_from(common->shared, domain->world, m, n)) return PLK_E_ARGUMENT;
  const uint64_t rows = uint64_t(m) / domain->world;
  for (uint32_t i = 0; i < Count; ++i) {
    auto& demand = out->channels[i];
    demand.dtype_id = common->channels[i].dtype_id;
    demand.access = i == Output ? PLK_ACCESS_WRITE : PLK_ACCESS_READ_WRITE;
    if (i == Slots) vector(demand, 2 * uint64_t(m) * uint64_t(n), 32);  // [2][W][M/W][N]
    if (i == Counters) vector(demand, 2 * (rows / 128) * (uint64_t(n) / 128), 32);
    if (i == Released) vector(demand, domain->world, 32);
    if (i == Gen) vector(demand, 1, 64);
    if (i == Output) {
      demand.minimum_allocation_bytes = rows * uint64_t(n) * 4;
      demand.alignment_bytes = 16;
      demand.extents[0] = rows;
      demand.extents[1] = uint64_t(n);
      demand.bit_strides[0] = n * 32;
      demand.bit_strides[1] = 32;
    }
  }
  out->workspaces[0].bytes = sizeof(uint32_t);
  out->workspaces[0].alignment_bytes = 4;
  out->channel_count = Count;
  out->workspace_count = 1;
  out->geometry = {};  // The GEMM owner chooses the grid.
  return PLK_OK;
}

extern "C" int pipelink_params_build_comm_rs(
    const plk_common_v2* common, const void* args, const plk_resources_v2* resources,
    void* out, size_t cap, plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
  if (!session) return PLK_E_ARGUMENT;
  *session = nullptr;
  if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
  if (cap < sizeof(CommRsResidue) || transfer_capacity) return PLK_E_CAPACITY;
  if (resources->channel_count != Count || resources->workspace_count != 1) return PLK_E_CONTRACT;
  const auto* domain = plk_find_domain(common, 1);
  if (!domain) return PLK_E_CONTRACT;
  int64_t m, n;
  if (!shape_from(resources->shared, domain->world, m, n)) return PLK_E_ARGUMENT;
  auto& residue = *new (out) CommRsResidue{};
  uint64_t* tables[] = {residue.slots, residue.counters, nullptr, residue.released};
  for (uint32_t i = 0; i < Count; ++i) {
    const auto& resource = resources->channels[i];
    if (resource.origin_bit_offset) return PLK_E_RESOURCE;
    if (i == Gen) {
      residue.generation = reinterpret_cast<uint64_t*>(resource.allocation_address + resource.origin_byte_offset);
      continue;
    }
    if (i == Output) {
      residue.output = reinterpret_cast<float*>(resource.allocation_address + resource.origin_byte_offset);
      continue;
    }
    if (!resource.peer_addresses) return PLK_E_RESOURCE;
    for (uint32_t r = 0; r < domain->world; ++r)
      tables[i][r] = resource.peer_addresses[r] + resource.origin_byte_offset;
  }
  residue.done = reinterpret_cast<uint32_t*>(resources->workspaces[0].device_address);
  residue.rank = int32_t(domain->rank);
  residue.world = int32_t(domain->world);
  residue.M = int32_t(m);
  residue.N = int32_t(n);
  return PLK_OK;
}

extern "C" int pipelink_params_release_comm_rs(plk_session* session) {
  return session ? PLK_E_RELEASE : PLK_OK;
}
