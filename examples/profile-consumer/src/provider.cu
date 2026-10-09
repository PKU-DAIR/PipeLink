#include "consumer.cuh"
#include <pipelink_host_abi_v2.h>

#include <cstring>

extern "C" int pipelink_params_query_interpreter(
    plk_common_v2 const* common, void const*, plk_demands_v2* out) {
  using namespace pipelink_profile_consumer;
  if (common->header.abi_version != PLK_HOST_ABI_V2 ||
      out->header.abi_version != PLK_HOST_ABI_V2) {
    return PLK_E_VERSION;
  }
  if (out->channel_capacity < 6) {
    return PLK_E_CAPACITY;
  }
  unsigned const extents[]{sizeof(InstructionDescriptor) / sizeof(unsigned),
                           kElements, kElements, kElements, kCounterCapacity, 1};
  for (unsigned index = 0; index < 6; ++index) {
    auto& demand = out->channels[index];
    demand.dtype_id = common->channels[index].dtype_id;
    demand.access = common->channels[index].access;
    demand.minimum_allocation_bytes = extents[index] * sizeof(unsigned);
    demand.alignment_bytes = index == 0 ? alignof(InstructionDescriptor) : alignof(unsigned);
    demand.extents[0] = extents[index];
    demand.bit_strides[0] = sizeof(unsigned) * 8;
  }
  out->channel_count = 6;
  out->workspace_count = 0;
  out->geometry = {1, 1, {kBlocks, 1, 1}, common->block, {1, 1, 1}, 0,
                   common->dynamic_smem_bytes};
  return PLK_OK;
}

extern "C" int pipelink_params_build_interpreter(
    plk_common_v2 const*, void const*, plk_resources_v2 const* resources,
    void* residue, size_t residue_cap, plk_transfer_report_v2*,
    uint32_t transfer_capacity, plk_session** session) {
  using namespace pipelink_profile_consumer;
  *session = nullptr;
  if (residue_cap != sizeof(InterpreterParams) || transfer_capacity != 0) {
    return PLK_E_CAPACITY;
  }
  auto address = [&](unsigned index) {
    auto const& channel = resources->channels[index];
    return channel.allocation_address + channel.origin_byte_offset;
  };
  InterpreterParams const params{
      reinterpret_cast<InstructionDescriptor const*>(address(0)),
      reinterpret_cast<unsigned const*>(address(1)),
      reinterpret_cast<unsigned*>(address(2)),
      reinterpret_cast<unsigned*>(address(3)),
      reinterpret_cast<unsigned*>(address(4)),
      reinterpret_cast<unsigned*>(address(5))};
  std::memcpy(residue, &params, sizeof(params));
  return PLK_OK;
}

extern "C" int pipelink_params_release_interpreter(plk_session*) {
  // The caller owns every allocation; build creates no provider-owned state.
  return PLK_OK;
}
