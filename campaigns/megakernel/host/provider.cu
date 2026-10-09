#include "mk_host_abi.h"

#ifndef MK_PROVIDER_SELFTEST

#include "llama.cuh"

#include <cstdio>
#include <cstring>
#include <exception>
#include <new>

struct plk_session_v2 {
  unsigned char marker = 0;
};

static_assert(sizeof(llama_1b_globals) == MK_GLOBALS_BYTES);
static_assert(alignof(llama_1b_globals) == MK_GLOBALS_ALIGN);

namespace {

bool valid_args(const MkProviderArgs& a) {
  if (a.version != MK_PROVIDER_ARGS_VERSION) {
    return false;
  }

  if (a.bar == 0 || a.instructions == 0 || a.timings == 0 ||
      a.qkv_weights == 0 || a.attn_norm_weights == 0 || a.o_weights == 0 ||
      a.mlp_norm_weights == 0 || a.up_weights == 0 || a.gate_weights == 0 ||
      a.down_weights == 0 || a.lm_head_norm_weights == 0 ||
      a.lm_head_weights == 0 || a.k_cache == 0 || a.v_cache == 0 ||
      a.rope_cos == 0 || a.rope_sin == 0 || a.hidden_states == 0 ||
      a.q_post_rope == 0 || a.attn_out == 0 ||
      a.attn_lse_intermediates == 0 || a.attn_out_intermediates == 0 ||
      a.silu_out == 0 || a.logits == 0) {
    return false;
  }

  return a.num_sms > 0 && a.num_instructions > 0 && a.max_seq_len > 0 &&
         a.rope_rows > 0 && a.num_layers > 0 && a.bar_rows > 0 &&
         a.vocab_size > 0;
}

}  // namespace

extern "C" int pipelink_params_query_interpreter(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
  if (!common || !args || !out) return PLK_E_ARGUMENT;
  if (common->header.abi_version != PLK_HOST_ABI_V2 ||
      common->header.struct_bytes != sizeof(*common) ||
      out->header.abi_version != PLK_HOST_ABI_V2 ||
      out->header.struct_bytes != sizeof(*out)) return PLK_E_VERSION;
  if (common->channel_count || common->shared ||
      out->channel_capacity || out->workspace_capacity) return PLK_E_CONTRACT;
  const auto& a = *static_cast<const MkProviderArgs*>(args);
  if (!valid_args(a)) return PLK_E_ARGUMENT;
  out->channel_count = 0;
  out->workspace_count = 0;
  out->geometry = {1, 1, {static_cast<uint32_t>(a.num_sms), 1, 1},
                   common->block, common->cluster, 0, common->dynamic_smem_bytes};
  return PLK_OK;
}

extern "C" int pipelink_params_build_interpreter(
    const plk_common_v2* common, const void* args,
    const plk_resources_v2* resources, void* out, size_t cap,
    plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
  if (!session) return PLK_E_ARGUMENT;
  *session = nullptr;
  if (!common || !args || !resources || !out) return PLK_E_ARGUMENT;
  if (cap != sizeof(llama_1b_globals) || transfer_capacity != 0)
    return PLK_E_CAPACITY;
  if (resources->header.abi_version != PLK_HOST_ABI_V2 ||
      resources->header.struct_bytes != sizeof(*resources)) return PLK_E_VERSION;
  if (resources->channel_count || resources->workspace_count || resources->shared)
    return PLK_E_CONTRACT;
  *session = new (std::nothrow) plk_session;
  if (!*session) return PLK_E_BUILD;
  const auto& a = *static_cast<const MkProviderArgs*>(args);

  try {
    llama_1b_globals g{
        kittens::make_gl<decltype(llama_1b_globals::Bar)>(
            a.bar, 1, a.num_layers, a.bar_rows, 48),
        kittens::make_gl<decltype(llama_1b_globals::instructions)>(
            a.instructions, 1, a.num_sms, a.num_instructions, 32),
        kittens::make_gl<decltype(llama_1b_globals::timings)>(
            a.timings, 1, a.num_sms, a.num_instructions, 128),

        kittens::make_gl<decltype(llama_1b_globals::qkv_weights)>(
            a.qkv_weights, 1, a.num_layers, 3072, 2048),
        kittens::make_gl<decltype(llama_1b_globals::attn_norm_weights)>(
            a.attn_norm_weights, 1, 1, a.num_layers, 2048),
        kittens::make_gl<decltype(llama_1b_globals::o_weights)>(
            a.o_weights, 1, a.num_layers, 2048, 2048),
        kittens::make_gl<decltype(llama_1b_globals::mlp_norm_weights)>(
            a.mlp_norm_weights, 1, 1, a.num_layers, 2048),
        kittens::make_gl<decltype(llama_1b_globals::up_weights)>(
            a.up_weights, 1, a.num_layers, 8192, 2048),
        kittens::make_gl<decltype(llama_1b_globals::gate_weights)>(
            a.gate_weights, 1, a.num_layers, 8192, 2048),
        kittens::make_gl<decltype(llama_1b_globals::down_weights)>(
            a.down_weights, 1, a.num_layers, 2048, 8192),
        kittens::make_gl<decltype(llama_1b_globals::lm_head_norm_weights)>(
            a.lm_head_norm_weights, 1, 1, 1, 2048),
        kittens::make_gl<decltype(llama_1b_globals::lm_head_weights)>(
            a.lm_head_weights, 1, 1, a.vocab_size, 2048),
        kittens::make_gl<decltype(llama_1b_globals::k_cache)>(
            a.k_cache, a.num_layers, a.max_seq_len, 8, 64),
        kittens::make_gl<decltype(llama_1b_globals::v_cache)>(
            a.v_cache, a.num_layers, a.max_seq_len, 8, 64),

        kittens::make_gl<decltype(llama_1b_globals::rope_cos)>(
            a.rope_cos, 1, 1, a.rope_rows, 64),
        kittens::make_gl<decltype(llama_1b_globals::rope_sin)>(
            a.rope_sin, 1, 1, a.rope_rows, 64),

        kittens::make_gl<decltype(llama_1b_globals::hidden_states)>(
            a.hidden_states, 1, 1, 1, 2048),
        kittens::make_gl<decltype(llama_1b_globals::q_post_rope)>(
            a.q_post_rope, 1, 1, 1, 2048),
        kittens::make_gl<decltype(llama_1b_globals::attn_out)>(
            a.attn_out, 1, 1, 1, 2048),
        kittens::make_gl<decltype(llama_1b_globals::attn_lse_intermediates)>(
            a.attn_lse_intermediates, 1, 1, 32, a.num_sms),
        kittens::make_gl<decltype(llama_1b_globals::attn_out_intermediates)>(
            a.attn_out_intermediates, 1, 32, a.num_sms, 64),
        kittens::make_gl<decltype(llama_1b_globals::silu_out)>(
            a.silu_out, 1, 1, 1, 8192),
        kittens::make_gl<decltype(llama_1b_globals::logits)>(
            a.logits, 1, 1, 1, a.vocab_size),

        a.pos_id,
        a.attn_scale,
        a.rms_norm_eps,
        static_cast<bool>(a.skip_attn_reduction),
    };
    std::memcpy(out, &g, sizeof(g));
    return PLK_OK;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "MK params provider: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "MK params provider: unknown exception\n");
  }
  return PLK_E_BUILD;
}

extern "C" int pipelink_params_release_interpreter(plk_session* session) {
  delete session;
  return PLK_OK;
}

#else

#include <cuda.h>
#include <dlfcn.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

bool check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "provider_selftest: %s failed: %s (%s)\n", operation,
               name != nullptr ? name : "unknown",
               message != nullptr ? message : "unknown");
  return false;
}

class DeviceAllocations {
 public:
  bool allocate(uint64_t bytes, uint64_t& destination) {
    CUdeviceptr pointer = 0;
    if (!check(cuMemAlloc(&pointer, static_cast<size_t>(bytes)), "cuMemAlloc")) {
      return false;
    }
    pointers_.push_back(pointer);
    destination = static_cast<uint64_t>(pointer);
    return true;
  }

  ~DeviceAllocations() {
    for (CUdeviceptr pointer : pointers_) {
      cuMemFree(pointer);
    }
  }

 private:
  std::vector<CUdeviceptr> pointers_;
};

constexpr uint64_t product(std::initializer_list<uint64_t> dimensions) {
  uint64_t value = 1;
  for (uint64_t dimension : dimensions) {
    value *= dimension;
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s PROVIDER_SO\n", argv[0]);
    return 2;
  }

  if (!check(cuInit(0), "cuInit")) {
    return 1;
  }
  CUdevice device = 0;
  if (!check(cuDeviceGet(&device, 0), "cuDeviceGet")) {
    return 1;
  }
  CUctxCreateParams context_params{};
  CUcontext context = nullptr;
  if (!check(cuCtxCreate(&context, &context_params, 0, device), "cuCtxCreate")) {
    return 1;
  }

  int result = 1;
  {
    constexpr uint64_t kNumSms = 148;
    constexpr uint64_t kNumInstructions = 2;
    constexpr uint64_t kMaxSeqLen = 64;
    constexpr uint64_t kRopeRows = 64;
    constexpr uint64_t kNumLayers = 16;
    constexpr uint64_t kBarRows = 10;
    constexpr uint64_t kVocabSize = 128256;
    constexpr uint64_t kBf16Bytes = 2;
    constexpr uint64_t kF32Bytes = 4;
    constexpr uint64_t kI32Bytes = 4;

    MkProviderArgs a{};
    a.version = MK_PROVIDER_ARGS_VERSION;
    a.num_sms = static_cast<int32_t>(kNumSms);
    a.num_instructions = static_cast<int32_t>(kNumInstructions);
    a.max_seq_len = static_cast<int32_t>(kMaxSeqLen);
    a.rope_rows = static_cast<int32_t>(kRopeRows);
    a.num_layers = static_cast<int32_t>(kNumLayers);
    a.bar_rows = static_cast<int32_t>(kBarRows);
    a.vocab_size = static_cast<int32_t>(kVocabSize);
    a.pos_id = 37;
    a.attn_scale = 0.125f;
    a.rms_norm_eps = 1.0e-5f;
    a.skip_attn_reduction = 1;

    DeviceAllocations allocations;
    const bool allocated =
        allocations.allocate(product({1, kNumLayers, kBarRows, 48, kI32Bytes}), a.bar) &&
        allocations.allocate(product({1, kNumSms, kNumInstructions, 32, kI32Bytes}), a.instructions) &&
        allocations.allocate(product({1, kNumSms, kNumInstructions, 128, kI32Bytes}), a.timings) &&
        allocations.allocate(product({1, kNumLayers, 3072, 2048, kBf16Bytes}), a.qkv_weights) &&
        allocations.allocate(product({1, 1, kNumLayers, 2048, kBf16Bytes}), a.attn_norm_weights) &&
        allocations.allocate(product({1, kNumLayers, 2048, 2048, kBf16Bytes}), a.o_weights) &&
        allocations.allocate(product({1, 1, kNumLayers, 2048, kBf16Bytes}), a.mlp_norm_weights) &&
        allocations.allocate(product({1, kNumLayers, 8192, 2048, kBf16Bytes}), a.up_weights) &&
        allocations.allocate(product({1, kNumLayers, 8192, 2048, kBf16Bytes}), a.gate_weights) &&
        allocations.allocate(product({1, kNumLayers, 2048, 8192, kBf16Bytes}), a.down_weights) &&
        allocations.allocate(product({1, 1, 1, 2048, kBf16Bytes}), a.lm_head_norm_weights) &&
        allocations.allocate(product({1, 1, kVocabSize, 2048, kBf16Bytes}), a.lm_head_weights) &&
        allocations.allocate(product({kNumLayers, kMaxSeqLen, 8, 64, kBf16Bytes}), a.k_cache) &&
        allocations.allocate(product({kNumLayers, kMaxSeqLen, 8, 64, kBf16Bytes}), a.v_cache) &&
        allocations.allocate(product({1, 1, kRopeRows, 64, kF32Bytes}), a.rope_cos) &&
        allocations.allocate(product({1, 1, kRopeRows, 64, kF32Bytes}), a.rope_sin) &&
        allocations.allocate(product({1, 1, 1, 2048, kBf16Bytes}), a.hidden_states) &&
        allocations.allocate(product({1, 1, 1, 2048, kBf16Bytes}), a.q_post_rope) &&
        allocations.allocate(product({1, 1, 1, 2048, kBf16Bytes}), a.attn_out) &&
        allocations.allocate(product({1, 1, 32, kNumSms, kF32Bytes}), a.attn_lse_intermediates) &&
        allocations.allocate(product({1, 32, kNumSms, 64, kF32Bytes}), a.attn_out_intermediates) &&
        allocations.allocate(product({1, 1, 1, 8192, kBf16Bytes}), a.silu_out) &&
        allocations.allocate(product({1, 1, 1, kVocabSize, kBf16Bytes}), a.logits);

    if (allocated) {
      void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
      if (library == nullptr) {
        std::fprintf(stderr, "provider_selftest: dlopen failed: %s\n", dlerror());
      } else {
        const auto abi = reinterpret_cast<plk_abi_fn_v2>(
            dlsym(library, "pipelink_params_abi_interpreter"));
        const auto query = reinterpret_cast<plk_query_fn_v2>(
            dlsym(library, "pipelink_params_query_interpreter"));
        const auto build = reinterpret_cast<plk_build_fn_v2>(
            dlsym(library, "pipelink_params_build_interpreter"));
        const auto release = reinterpret_cast<plk_release_fn_v2>(
            dlsym(library, "pipelink_params_release_interpreter"));
        if (!abi || !query || !build || !release) {
          std::fprintf(stderr, "provider_selftest: missing host ABI export\n");
        } else {
#define MK_ABI_ENTRY(id, T, ...) {id, 0, sizeof(T), alignof(T), {}},
#define MK_ROLE_ENTRY(id, T) {id, 0, sizeof(mk_host::T), alignof(mk_host::T), {}},
          plk_layout_entry_v2 entries[] = {
              {1, 0, sizeof(MkProviderArgs), alignof(MkProviderArgs), {}},
              {2, 0, MK_GLOBALS_BYTES, MK_GLOBALS_ALIGN, {}},
              MK_HOST_ROLE_TYPES(MK_ROLE_ENTRY)
              PLK_HOST_ABI_TYPES(MK_ABI_ENTRY)};
#undef MK_ABI_ENTRY
#undef MK_ROLE_ENTRY
          plk_abi_report_v2 report{};
          report.header = {PLK_HOST_ABI_V2, sizeof(report)};
          report.entry_capacity = sizeof(entries) / sizeof(entries[0]);
          report.entries = entries;
          int status = abi(&report);
          plk_common_v2 common{};
          common.header = {PLK_HOST_ABI_V2, sizeof(common)};
          common.owner_id = 1;
          common.block = {640, 1, 1};
          common.cluster = {1, 1, 1};
          common.profile = report.profile;
          common.owner_contract = report.owner_contract;
          plk_demands_v2 demands{};
          demands.header = {PLK_HOST_ABI_V2, sizeof(demands)};
          if (status == PLK_OK) status = query(&common, &a, &demands);
          plk_resources_v2 resources{};
          resources.header = {PLK_HOST_ABI_V2, sizeof(resources)};
          resources.geometry = demands.geometry;
          alignas(MK_GLOBALS_ALIGN) std::array<std::byte, MK_GLOBALS_BYTES> blob{};
          plk_session* session = nullptr;
          if (status == PLK_OK) {
            status = build(&common, &a, &resources, blob.data(), blob.size(),
                           nullptr, 0, &session);
            const int release_status = release(session);
            if (status == PLK_OK) status = release_status;
          }
          uint32_t pos_id = 0;
          float attn_scale = 0.0f;
          std::memcpy(&pos_id, blob.data() + 9472, sizeof(pos_id));
          std::memcpy(&attn_scale, blob.data() + 9476, sizeof(attn_scale));
          if (status != PLK_OK || report.entry_count != report.entry_capacity ||
              !demands.geometry.geometry_present || !demands.geometry.launch_required ||
              demands.geometry.grid.x != kNumSms) {
            std::fprintf(stderr, "provider_selftest: host ABI failed with %d\n", status);
          } else if (pos_id != a.pos_id || attn_scale != a.attn_scale) {
            std::fprintf(stderr,
                         "provider_selftest: scalar mismatch: pos_id=%u attn_scale=%.9g\n",
                         pos_id, static_cast<double>(attn_scale));
          } else {
            std::printf("provider_selftest: PASS pos_id=%u attn_scale=%.9g\n",
                        pos_id, static_cast<double>(attn_scale));
            result = 0;
          }
        }
        dlclose(library);
      }
    }
  }

  if (!check(cuCtxDestroy(context), "cuCtxDestroy")) {
    return 1;
  }
  return result;
}

#endif
