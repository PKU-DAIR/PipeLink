#include "abi.h"
#include <new>

namespace multimem_domains::mc_provider {
enum Channel { Input, Red, Bar, Done, Epoch, Count };
}

extern "C" int pipelink_params_query_mc(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
  using namespace multimem_domains::host;
  using namespace multimem_domains::mc_provider;
  if (!common || !out || args) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) || !header_is(out->header, sizeof(*out)))
    return PLK_E_VERSION;
  if (common->channel_count != Count || out->channel_capacity != Count || out->workspace_capacity)
    return PLK_E_CONTRACT;
  if (!pair_domain(plk_find_domain(common, 1))) return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs_from(common->shared, in, true)) return PLK_E_ARGUMENT;
  for (uint32_t i = 0; i < Count; ++i) {
    const bool payload = i == Input || i == Red;
    const uint32_t access = i == Input ? PLK_ACCESS_READ :
                            i == Red ? PLK_ACCESS_WRITE : PLK_ACCESS_READ_WRITE;
    vector_demand(out->channels[i], common->channels[i],
                  payload ? uint64_t(in.elements) : in.slices, access, payload ? 16 : 4);
  }
  const uint32_t grid = in.slices < uint64_t(in.sm_count) ? in.slices : uint32_t(in.sm_count);
  out->channel_count = Count;
  out->workspace_count = 0;
  // Hardware is observed by the caller's backend, never guessed from a model.
  // The runner checks full-grid residency independently for both kernel images.
  out->geometry = {1, 1, {grid, 1, 1}, common->block, {1, 1, 1}, 0, common->dynamic_smem_bytes};
  return PLK_OK;
}

extern "C" int pipelink_params_build_mc(
    const plk_common_v2* common, const void* args, const plk_resources_v2* resources,
    void* out, size_t cap, plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
  using namespace multimem_domains::host;
  using namespace multimem_domains::mc_provider;
  if (!session) return PLK_E_ARGUMENT;
  *session = nullptr;
  if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) || !header_is(resources->header, sizeof(*resources)))
    return PLK_E_VERSION;
  if (cap < sizeof(MultimemMcResidue) || transfer_capacity) return PLK_E_CAPACITY;
  if (resources->channel_count != Count || resources->workspace_count) return PLK_E_CONTRACT;
  const auto* tp = plk_find_domain(common, 1);
  if (!pair_domain(tp)) return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs_from(resources->shared, in, true)) return PLK_E_ARGUMENT;
  uint64_t local[Count];
  uint64_t multicast[Epoch];
  for (uint32_t i = 0; i < Count; ++i) {
    const auto& resource = resources->channels[i];
    if (!resource.allocation_address || resource.origin_bit_offset) return PLK_E_RESOURCE;
    local[i] = resource.allocation_address + resource.origin_byte_offset;
    if (i != Epoch) {
      if (resource.domain_id != 1 || !resource.multicast_address) return PLK_E_RESOURCE;
      multicast[i] = resource.multicast_address + resource.origin_byte_offset;
    } else if (resource.domain_id) {
      return PLK_E_RESOURCE;
    }
  }
  auto& p = *new (out) MultimemMcResidue{};
  p.input_mc = reinterpret_cast<const float*>(multicast[Input]);
  p.red_mc = reinterpret_cast<float*>(multicast[Red]);
  p.bar_mc = reinterpret_cast<uint32_t*>(multicast[Bar]);
  p.bar_local = reinterpret_cast<const uint32_t*>(local[Bar]);
  p.done_mc = reinterpret_cast<uint32_t*>(multicast[Done]);
  p.epoch = reinterpret_cast<uint32_t*>(local[Epoch]);
  p.slice_elements = uint32_t(in.slice_elements);
  p.slices = in.slices;
  p.tp_rank = tp->rank;
  return PLK_OK;
}

extern "C" int pipelink_params_release_mc(plk_session* session) {
  return session ? PLK_E_RELEASE : PLK_OK;
}
