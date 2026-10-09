#include "abi.h"
#include <new>

namespace multimem_domains::dp_provider {
enum Channel { Red, TpDone, Bar, Done, Epoch, Output, Count };
}

extern "C" int pipelink_params_query_dp(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
  using namespace multimem_domains::host;
  using namespace multimem_domains::dp_provider;
  if (!common || !out || args) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) || !header_is(out->header, sizeof(*out)))
    return PLK_E_VERSION;
  if (common->channel_count != Count || out->channel_capacity != Count || out->workspace_capacity)
    return PLK_E_CONTRACT;
  if (!pair_domain(plk_find_domain(common, 1)) || !pair_domain(plk_find_domain(common, 2)))
    return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs_from(common->shared, in, false)) return PLK_E_ARGUMENT;
  for (uint32_t i = 0; i < Count; ++i) {
    const bool payload = i == Red || i == Output;
    const uint64_t elements = i == Red ? uint64_t(in.elements) :
                              i == Output ? 2 * uint64_t(in.elements) : in.slices;
    const uint32_t access = i == Red || i == TpDone ? PLK_ACCESS_READ :
                            i == Output ? PLK_ACCESS_WRITE : PLK_ACCESS_READ_WRITE;
    vector_demand(out->channels[i], common->channels[i], elements, access, payload ? 16 : 4);
  }
  out->channel_count = Count;
  out->workspace_count = 0;
  out->geometry = {};  // mc is the sole geometry authority.
  return PLK_OK;
}

extern "C" int pipelink_params_build_dp(
    const plk_common_v2* common, const void* args, const plk_resources_v2* resources,
    void* out, size_t cap, plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
  using namespace multimem_domains::host;
  using namespace multimem_domains::dp_provider;
  if (!session) return PLK_E_ARGUMENT;
  *session = nullptr;
  if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
  if (!header_is(common->header, sizeof(*common)) || !header_is(resources->header, sizeof(*resources)))
    return PLK_E_VERSION;
  if (cap < sizeof(MultimemDpResidue) || transfer_capacity) return PLK_E_CAPACITY;
  if (resources->channel_count != Count || resources->workspace_count) return PLK_E_CONTRACT;
  const auto* dp = plk_find_domain(common, 2);
  if (!pair_domain(plk_find_domain(common, 1)) || !pair_domain(dp)) return PLK_E_CONTRACT;
  Inputs in;
  if (!inputs_from(resources->shared, in, false)) return PLK_E_ARGUMENT;
  uint64_t local[Count];
  for (uint32_t i = 0; i < Count; ++i) {
    const auto& resource = resources->channels[i];
    const uint32_t domain = i == Red || i == TpDone ? 1 : i == Epoch ? 0 : 2;
    if (!resource.allocation_address || resource.origin_bit_offset || resource.domain_id != domain)
      return PLK_E_RESOURCE;
    if (domain == 2 && !resource.peer_addresses) return PLK_E_RESOURCE;
    local[i] = resource.allocation_address + resource.origin_byte_offset;
  }
  auto& p = *new (out) MultimemDpResidue{};
  p.red = reinterpret_cast<const float*>(local[Red]);
  p.tp_done = reinterpret_cast<const uint32_t*>(local[TpDone]);
  p.epoch = reinterpret_cast<uint32_t*>(local[Epoch]);
  uint64_t* tables[] = {p.bar, p.done, p.output};
  const uint32_t channels[] = {Bar, Done, Output};
  for (uint32_t i = 0; i < 3; ++i) {
    const auto& resource = resources->channels[channels[i]];
    for (uint32_t rank = 0; rank < 2; ++rank)
      tables[i][rank] = resource.peer_addresses[rank] + resource.origin_byte_offset;
  }
  p.elements = uint64_t(in.elements);
  p.slice_elements = uint32_t(in.slice_elements);
  p.slices = in.slices;
  p.dp_rank = dp->rank;
  return PLK_OK;
}

extern "C" int pipelink_params_release_dp(plk_session* session) {
  return session ? PLK_E_RELEASE : PLK_OK;
}
