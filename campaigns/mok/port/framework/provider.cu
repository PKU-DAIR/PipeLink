#include "framework_abi.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <initializer_list>
#include <new>

#ifndef MOK_FRAMEWORK_PASS
#error "Build the framework provider with MOK_FRAMEWORK_PASS=0, 1, or 2"
#endif
#if MOK_FRAMEWORK_PASS < MOK_PASS_FWD || MOK_FRAMEWORK_PASS > MOK_PASS_REC
#error "Invalid MOK_FRAMEWORK_PASS"
#endif

namespace mok_framework_provider {
#if MOK_FRAMEWORK_PASS == MOK_PASS_BWD
constexpr int first_slot = MOK_SCALAR_SLOT_D_Y_ROUTED_READY;
constexpr int64_t extents[] = {MOK_SLOTS_D_Y_ROUTED_READY, MOK_SLOTS_D_HIDDEN_READY,
    MOK_SLOTS_D_GATE_UP_READY, MOK_SLOTS_D_X_ROUTED_READY, MOK_SLOTS_REPLAYED_X_ROUTED_READY,
    MOK_SLOTS_REPLAYED_GATE_UP_READY, MOK_SLOTS_REPLAYED_HIDDEN_READY,
    MOK_SLOTS_ROUTED_BUFFERS_DONE, MOK_SLOTS_ROUTER_WEIGHTS_READY};
#else
constexpr int first_slot = MOK_SCALAR_SLOT_X_ROUTED_READY;
constexpr int64_t extents[] = {MOK_SLOTS_X_ROUTED_READY, MOK_SLOTS_GATE_UP_TILE_READY,
    MOK_SLOTS_HIDDEN_ROW_BLOCK_READY
#if MOK_FRAMEWORK_PASS == MOK_PASS_FWD
    , MOK_SLOTS_Y_ROUTED_READY, MOK_SLOTS_Y_ROUTED_DONE
#endif
};
#endif
constexpr int slot_count = sizeof(extents) / sizeof(extents[0]);
struct Inputs {
    MokFrameworkResidue g{};
    uint64_t workspace_bytes = 0;
    uint32_t grid = 0;
};
bool fits_i32(std::initializer_list<int64_t> values) {
    for (int64_t value : values) if (value < 0 || value > INT_MAX) return false;
    return true;
}
bool fact_is_live(uint32_t id) {
    if (id == MOK_SCALAR_DISPATCH_MB || id == MOK_SCALAR_DISPATCH_NB ||
        (id >= MOK_SCALAR_MLP_MB && id <= MOK_SCALAR_SWIGLU_FWD_PIPE_DEPTH)) return true;
#if MOK_FRAMEWORK_PASS != MOK_PASS_REC
    if (id >= MOK_SCALAR_COMBINE_MB && id <= MOK_SCALAR_COMBINE_PIPE_DEPTH) return true;
#endif
#if MOK_FRAMEWORK_PASS == MOK_PASS_BWD
    if (id == MOK_SCALAR_SWIGLU_BWD_PIPE_DEPTH) return true;
#endif
    return false;
}
bool inputs(const plk_shared_scalars_v2* shared, uint32_t world, Inputs& out) {
    if (!shared) return false;
    int64_t values[54]{};
    bool seen[54]{};
    const auto* records = reinterpret_cast<const plk_scalar_v2*>(
        reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
    for (uint32_t i = 0; i < shared->record_count; ++i) {
        const auto& r = records[i];
        if (!((r.scalar_id >= MOK_SCALAR_NUM_LOCAL_TOKENS && r.scalar_id <= MOK_SCALAR_NUM_COMM_SMS) ||
              (r.scalar_id >= MOK_SCALAR_PASS && r.scalar_id <= MOK_SCALAR_CLAMP) ||
              fact_is_live(r.scalar_id) ||
              (r.scalar_id >= first_slot && r.scalar_id < first_slot + slot_count))) continue;
        if (r.scalar_type != PLK_SCALAR_I64 || seen[r.scalar_id]) return false;
        std::memcpy(&values[r.scalar_id], &r.value_bits, sizeof(int64_t));
        seen[r.scalar_id] = true;
        if (!fits_i32({values[r.scalar_id]})) return false;
    }
    for (int id = MOK_SCALAR_PASS; id <= MOK_SCALAR_CLAMP; ++id)
        if (!seen[id]) return false;
    if (values[MOK_SCALAR_PASS] != MOK_FRAMEWORK_PASS ||
        (values[MOK_SCALAR_PRECISION] != MOK_PRECISION_BF16 &&
         values[MOK_SCALAR_PRECISION] != MOK_PRECISION_MXFP8) ||
        (values[MOK_SCALAR_WORLD] != 1 && values[MOK_SCALAR_WORLD] != 4) ||
        values[MOK_SCALAR_WORLD] != world || values[MOK_SCALAR_CLAMP] > 1) return false;
    auto& g = out.g;
    int32_t* dimensions[] = {&g.num_local_tokens, &g.hidden, &g.intermediate, &g.num_local_experts,
        &g.schedule_capacity, &g.macrobatch_size, &g.minibatch_size, &g.topk, &g.num_comm_sms};
    int32_t* facts[] = {&g.DISPATCH_Mb, &g.DISPATCH_Nb, &g.COMBINE_Mb, &g.COMBINE_Nb,
        &g.COMBINE_PIPE_DEPTH, &g.MLP_Mb, &g.MLP_Nb, &g.SWIGLU_Mb, &g.SWIGLU_Nb,
        &g.SWIGLU_FWD_PIPE_DEPTH, &g.SWIGLU_BWD_PIPE_DEPTH};
    for (int i = 0; i < 9; ++i) {
        if (!seen[i + 1] || values[i + 1] == 0) return false;
        *dimensions[i] = static_cast<int32_t>(values[i + 1]);
    }
    for (int i = 0; i < 11; ++i) {
        if (!fact_is_live(i + 20)) continue;
        if (!seen[i + 20] || values[i + 20] == 0) return false;
        *facts[i] = static_cast<int32_t>(values[i + 20]);
    }
    if (g.MLP_Mb % 2 || g.num_comm_sms % 2 || g.macrobatch_size % g.minibatch_size ||
        g.num_local_tokens % g.MLP_Mb || g.schedule_capacity % g.MLP_Mb ||
        g.minibatch_size % g.MLP_Mb || g.minibatch_size % g.DISPATCH_Mb ||
        g.num_local_tokens % g.SWIGLU_Mb || g.minibatch_size % g.SWIGLU_Mb ||
        g.intermediate % g.MLP_Nb || g.hidden % g.MLP_Nb ||
        g.intermediate % g.SWIGLU_Nb || g.MLP_Mb % g.SWIGLU_Mb) return false;
#if MOK_FRAMEWORK_PASS != MOK_PASS_REC
    if (g.minibatch_size % g.COMBINE_Mb) return false;
#endif
    const int64_t L = g.num_local_tokens, P = g.schedule_capacity, A = g.macrobatch_size;
    const int64_t B = g.minibatch_size, I = g.intermediate, H = g.hidden;
    const int64_t M = g.MLP_Mb, N = g.MLP_Nb, C = 2;
    const int64_t Pr = std::min(P, A);
#if MOK_FRAMEWORK_PASS == MOK_PASS_REC
    const int64_t routed_capacity = Pr;
#else
    const int64_t routed_capacity = P;
#endif
    // These are also the i32 ceiling numerators evaluated by the unchanged interpreter.
    if (!fits_i32({routed_capacity + B - 1, H + g.DISPATCH_Nb - 1})) return false;
#if MOK_FRAMEWORK_PASS != MOK_PASS_REC
    if (!fits_i32({P + A - 1})) return false;
#endif
    const int64_t minibatches = (routed_capacity + B - 1) / B;
#if MOK_FRAMEWORK_PASS == MOK_PASS_BWD
    const int64_t macros = (P + A - 1) / A;
#endif
    const int64_t sr = L / M, rr = routed_capacity / M, br = B / M;
    const int64_t gi = I / N, gh = H / N;
#if MOK_FRAMEWORK_PASS == MOK_PASS_BWD
    const int64_t counts[] = {minibatches, (sr + rr) * gi, sr + rr, minibatches,
        minibatches, rr * gi, rr, macros, macros};
#else
    const int64_t counts[] = {minibatches, (sr + rr) * gi, sr + rr
#if MOK_FRAMEWORK_PASS == MOK_PASS_FWD
        , minibatches, P / (M / C)
#endif
    };
#endif
    for (int i = 0; i < slot_count; ++i) {
        if (counts[i] > extents[i] || !seen[first_slot + i]) return false;
        const int64_t end = values[first_slot + i] + extents[i];
        if (!fits_i32({end})) return false;
        out.workspace_bytes = std::max(out.workspace_bytes, uint64_t(4 * end));
    }
    const int64_t shared_tiles = (L / g.SWIGLU_Mb) * (I / g.SWIGLU_Nb);
    const int64_t minibatch_tiles = (B / g.SWIGLU_Mb) * (I / g.SWIGLU_Nb);
    const int64_t fdepth = C * g.SWIGLU_FWD_PIPE_DEPTH;
    if (!fits_i32({fdepth, minibatch_tiles + fdepth - 1,
        sr * gi, br * gi, (M / g.SWIGLU_Mb) * (I / g.SWIGLU_Nb)})) return false;
#if MOK_FRAMEWORK_PASS != MOK_PASS_BWD
    if (!fits_i32({shared_tiles + fdepth - 1})) return false;
    const int64_t sf = (shared_tiles + fdepth - 1) / fdepth;
#endif
    const int64_t bf = (minibatch_tiles + fdepth - 1) / fdepth;
    const int64_t dispatch_cols = (H + g.DISPATCH_Nb - 1) / g.DISPATCH_Nb;
    if (!fits_i32({(Pr / g.DISPATCH_Mb) * dispatch_cols})) return false;
#if MOK_FRAMEWORK_PASS != MOK_PASS_REC
    if (!fits_i32({sr * gh, br * gh, H + g.COMBINE_Nb - 1})) return false;
    const int64_t combine_cols = (H + g.COMBINE_Nb - 1) / g.COMBINE_Nb;
    const int64_t combine_tiles = (Pr / g.COMBINE_Mb) * combine_cols;
    if (!fits_i32({combine_tiles + g.COMBINE_PIPE_DEPTH - 1})) return false;
#endif
    // Capacity-based grids from upstream types.cuh:138-152,255-273,326-339.
#if MOK_FRAMEWORK_PASS == MOK_PASS_BWD
    const int64_t bdepth = C * g.SWIGLU_BWD_PIPE_DEPTH;
    if (!fits_i32({bdepth, shared_tiles + bdepth - 1, minibatch_tiles + bdepth - 1,
        gi * gh})) return false;
    const int64_t sb = (shared_tiles + bdepth - 1) / bdepth;
    const int64_t bb = (minibatch_tiles + bdepth - 1) / bdepth;
    const int64_t shared_wgrad = 3 * gi * gh;
    if (!fits_i32({shared_wgrad})) return false;
    const int64_t wgrad = g.num_local_experts * shared_wgrad;
    const int64_t shared_tasks = sr * gi + sb + sr * gh + shared_wgrad;
    const int64_t minibatch_tasks = br * gi + bb + br * gh;
    const int64_t replay_tasks = 2 * br * gi + bf;
    // Keep upstream's full replay macros, even for a partial last capacity macro.
    const int64_t replay_minibatches = (macros - 1) * (A / B);
    if (!fits_i32({wgrad, shared_tasks, minibatch_tasks, replay_tasks, replay_minibatches})) return false;
    const int64_t routed_tasks = minibatches * minibatch_tasks;
    const int64_t all_replay_tasks = replay_minibatches * replay_tasks;
    const int64_t all_wgrad_tasks = macros * wgrad;
    if (!fits_i32({routed_tasks, all_replay_tasks, all_wgrad_tasks})) return false;
    const int64_t compute_tasks = shared_tasks + routed_tasks + all_replay_tasks + all_wgrad_tasks;
    const int64_t combine_tasks = (combine_tiles + g.COMBINE_PIPE_DEPTH - 1) / g.COMBINE_PIPE_DEPTH;
    const int64_t macro_bwd_tasks = ((Pr + B - 1) / B) * minibatch_tasks;
    if (!fits_i32({macro_bwd_tasks, C * (macro_bwd_tasks + wgrad) + combine_tasks})) return false;
#else
    const int64_t shared_tasks = 2 * sr * gi + sf
#if MOK_FRAMEWORK_PASS == MOK_PASS_FWD
        + sr * gh
#endif
        ;
    const int64_t minibatch_tasks = 2 * br * gi + bf
#if MOK_FRAMEWORK_PASS == MOK_PASS_FWD
        + br * gh
#endif
        ;
    if (!fits_i32({shared_tasks, minibatch_tasks})) return false;
    const int64_t routed_tasks = minibatches * minibatch_tasks;
    if (!fits_i32({routed_tasks})) return false;
    const int64_t compute_tasks = shared_tasks + routed_tasks;
#endif
    const int64_t grid = C * compute_tasks + g.num_comm_sms;
    if (!fits_i32({grid}) || grid == 0) return false;
    out.grid = static_cast<uint32_t>(grid);
    return true;
}
bool header_is(const plk_header_v2& header, size_t bytes) {
    return header.abi_version == PLK_HOST_ABI_V2 && header.struct_bytes == bytes;
}
bool topology_is(const plk_common_v2& common) {
    return common.block.x == 256 && common.block.y == 1 && common.block.z == 1 &&
        common.cluster.x == 2 && common.cluster.y == 1 && common.cluster.z == 1;
}
}

extern "C" int pipelink_params_query_mok_framework(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
    if (!common || args || !out) return PLK_E_ARGUMENT;
    if (!mok_framework_provider::header_is(common->header, sizeof(*common)) ||
        !mok_framework_provider::header_is(out->header, sizeof(*out))) return PLK_E_VERSION;
    if (!mok_framework_provider::topology_is(*common) || common->channel_count != 1 ||
        !common->channels || out->channel_capacity != 1 || !out->channels ||
        out->workspace_capacity != 1 || !out->workspaces) return PLK_E_CONTRACT;
    mok_framework_provider::Inputs in;
    const auto* ep = plk_find_domain(common, 1);
    if (!ep) return PLK_E_CONTRACT;
    if (!mok_framework_provider::inputs(common->shared, ep->world, in)) return PLK_E_ARGUMENT;
    auto& demand = out->channels[0];
    const auto& source = common->channels[0];
    if (source.channel_id != 1 || demand.channel_id != 1 || source.rank != 1 ||
        demand.rank != 1 || source.access != PLK_ACCESS_READ ||
        !source.extents || !source.bit_strides || !demand.extents || !demand.bit_strides ||
        out->workspaces[0].workspace_id != 1) return PLK_E_CONTRACT;
    if (source.origin_bit_offset || source.extents[0] != 1 || source.bit_strides[0] != 32)
        return PLK_E_ARGUMENT;
    demand.dtype_id = source.dtype_id;
    demand.access = PLK_ACCESS_READ;
    demand.minimum_allocation_bytes = 4;
    demand.alignment_bytes = 4;
    demand.extents[0] = 1;
    demand.bit_strides[0] = 32;
    out->workspaces[0].bytes = in.workspace_bytes;
    out->workspaces[0].alignment_bytes = 256;
    out->channel_count = out->workspace_count = 1;
    out->geometry = {1, 1, {in.grid, 1, 1}, {256, 1, 1}, {2, 1, 1}, 0, common->dynamic_smem_bytes};
    return PLK_OK;
}
extern "C" int pipelink_params_build_mok_framework(
    const plk_common_v2* common, const void* args, const plk_resources_v2* resources,
    void* out, size_t cap, plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
    if (!session) return PLK_E_ARGUMENT;
    *session = nullptr;
    if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
    if (!mok_framework_provider::header_is(common->header, sizeof(*common)) ||
        !mok_framework_provider::header_is(resources->header, sizeof(*resources))) return PLK_E_VERSION;
    if (cap < sizeof(MokFrameworkResidue) || transfer_capacity) return PLK_E_CAPACITY;
    if (!mok_framework_provider::topology_is(*common) || resources->channel_count != 1 ||
        !resources->channels || resources->workspace_count != 1 || !resources->workspaces)
        return PLK_E_CONTRACT;
    mok_framework_provider::Inputs in;
    const auto* ep = plk_find_domain(common, 1);
    if (!ep) return PLK_E_CONTRACT;
    if (!mok_framework_provider::inputs(resources->shared, ep->world, in)) return PLK_E_ARGUMENT;
    const auto& resource = resources->channels[0];
    const auto& workspace = resources->workspaces[0];
    if (resource.channel_id != 1 || resource.access != PLK_ACCESS_READ || workspace.workspace_id != 1)
        return PLK_E_CONTRACT;
    if (!resource.allocation_address || resource.origin_bit_offset ||
        resource.origin_byte_offset > resource.allocation_bytes ||
        4 > resource.allocation_bytes - resource.origin_byte_offset ||
        resource.allocation_address > UINT64_MAX - resource.origin_byte_offset ||
        (resource.allocation_address + resource.origin_byte_offset) % 4 ||
        !workspace.device_address || workspace.device_address % 256 ||
        workspace.bytes < in.workspace_bytes) return PLK_E_RESOURCE;
    in.g.num_tokens = reinterpret_cast<int32_t*>(resource.allocation_address + resource.origin_byte_offset);
    in.g.counters = reinterpret_cast<int32_t*>(workspace.device_address);
    *new (out) MokFrameworkResidue(in.g);
    return PLK_OK;
}
extern "C" int pipelink_params_release_mok_framework(plk_session* session) {
    return session ? PLK_E_RELEASE : PLK_OK;
}
