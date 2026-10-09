// Owner-private layout and TK host descriptor constructors, shared with the stage.
#ifndef MOK_COMM_PROVIDER
#define MOK_COMM_PROVIDER
#endif
#include "comm.cu"
#include <climits>
#include <cstring>

namespace {
using namespace mok_comm_owner;
struct Inputs {
    int64_t local_tokens = 0, hidden = 0, capacity = 0, macrobatch = 0, topk = 0;
    int64_t intermediate = 0, mlp_mb = 0, mlp_nb = 0, swiglu_nb = 0;
    int64_t pass = -1, precision = -1, world = 0;
};
bool inputs_from(const plk_shared_scalars_v2* shared, Inputs& in) {
    if (!shared) return false;
    const auto* records = reinterpret_cast<const plk_scalar_v2*>(
        reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
    for (uint32_t i = 0; i < shared->record_count; ++i) {
        const auto& r = records[i];
        int64_t* field = r.scalar_id == MOK_SCALAR_NUM_LOCAL_TOKENS ? &in.local_tokens :
            r.scalar_id == MOK_SCALAR_HIDDEN ? &in.hidden :
            r.scalar_id == MOK_SCALAR_INTERMEDIATE ? &in.intermediate :
            r.scalar_id == MOK_SCALAR_PASS ? &in.pass :
            r.scalar_id == MOK_SCALAR_PRECISION ? &in.precision :
            r.scalar_id == MOK_SCALAR_WORLD ? &in.world :
            r.scalar_id == MOK_SCALAR_SWIGLU_NB ? &in.swiglu_nb :
            r.scalar_id == MOK_SCALAR_SCHEDULE_CAPACITY ? &in.capacity :
            r.scalar_id == MOK_SCALAR_MACROBATCH_SIZE ? &in.macrobatch :
            r.scalar_id == MOK_SCALAR_TOPK ? &in.topk :
            r.scalar_id == MOK_SCALAR_MLP_MB ? &in.mlp_mb :
            r.scalar_id == MOK_SCALAR_MLP_NB ? &in.mlp_nb : nullptr;
        if (!field) continue;
        if (r.scalar_type != PLK_SCALAR_I64) return false;
        std::memcpy(field, &r.value_bits, sizeof(*field));
    }
    return in.pass >= MOK_PASS_FWD && in.pass <= MOK_PASS_REC &&
        (in.precision == MOK_PRECISION_BF16 || in.precision == MOK_PRECISION_MXFP8) &&
        in.intermediate > 0 && in.intermediate <= INT_MAX &&
        in.swiglu_nb > 0 && in.swiglu_nb <= INT_MAX && in.intermediate % in.swiglu_nb == 0 &&
        in.local_tokens >= 512 && in.local_tokens <= INT_MAX && in.local_tokens % 256 == 0 &&
        in.hidden > 0 && in.hidden <= INT_MAX && in.hidden % 256 == 0 &&
        in.capacity > 0 && in.capacity <= INT_MAX && in.capacity % 256 == 0 &&
        in.macrobatch > 0 && in.macrobatch <= INT_MAX && in.macrobatch % 256 == 0 &&
        in.topk > 0 && in.topk <= 255 && in.local_tokens <= INT_MAX / in.topk &&
        in.mlp_mb > 0 && in.mlp_mb <= INT_MAX && in.mlp_mb % config::CLUSTER_SIZE == 0 &&
        (in.mlp_mb / config::CLUSTER_SIZE) % config::COMBINE_Mb == 0 &&
        in.mlp_nb > 0 && in.mlp_nb <= INT_MAX && in.hidden % in.mlp_nb == 0;
}
bool topology(const plk_common_v2& common, const plk_domain_v2& ep) {
    return common.block.x == config::NUM_THREADS && common.block.y == 1 && common.block.z == 1 &&
        common.cluster.x == config::CLUSTER_SIZE && common.cluster.y == 1 && common.cluster.z == 1 &&
        (ep.world == 1 || ep.world == 4) && ep.rank < ep.world;
}
void demand(plk_channel_demand_v2& out, uint32_t dtype, uint32_t access,
            uint32_t bits, std::initializer_list<uint64_t> dimensions, uint64_t alignment) {
    out.dtype_id = dtype;
    out.access = access;
    out.alignment_bytes = alignment;
    uint64_t stride = bits;
    unsigned axis = dimensions.size();
    for (auto it = dimensions.end(); it != dimensions.begin();) {
        --it;
        --axis;
        out.extents[axis] = *it;
        out.bit_strides[axis] = stride;
        stride *= *it;
    }
    out.minimum_allocation_bytes = stride / 8;
}
bool active(unsigned channel, const Inputs& in) {
    const bool fp8 = in.precision == MOK_PRECISION_MXFP8;
    switch (channel) {
    case CombineBuffer: case YRouted: return in.pass == MOK_PASS_FWD;
    case XRouted: case XRoutedT: case XSc: case XScT: return fp8;
    case XBF16: return !fp8;
    case DYRouted: case DYRoutedT: case DYSc: case DYScT:
        return in.pass == MOK_PASS_BWD && fp8;
    case DYBF16: return in.pass == MOK_PASS_BWD && !fp8;
    case DYBuffer: case DXBuffer: case RouterWeightBuffer: case DRouterWeightBuffer:
    case RouterWeights: case DRouterWeightPartials: case DXRouted:
        return in.pass == MOK_PASS_BWD;
    default: return true;
    }
}
uint32_t access(unsigned channel) {
    switch (channel) {
    case XBuffer: case YRouted: case SchedulePeerRank: case SchedulePeerTokenIdx:
    case DYBuffer: case RouterWeightBuffer: case DRouterWeightPartials: case DXRouted:
        return PLK_ACCESS_READ;
    case RouterWeights: return PLK_ACCESS_READ | PLK_ACCESS_WRITE;
    default: return PLK_ACCESS_WRITE;
    }
}
} // namespace

extern "C" int pipelink_params_query_mok_comm(
    const plk_common_v2* common, const void* args, plk_demands_v2* out) {
    if (!common || !out || args) return PLK_E_ARGUMENT;
    if (common->header.abi_version != PLK_HOST_ABI_V2) return PLK_E_VERSION;
    if (common->channel_count != ChannelCount || out->channel_capacity != ChannelCount ||
        out->workspace_capacity != 0) return PLK_E_CONTRACT;
    Inputs in;
    const auto* ep = plk_find_domain(common, 1);
    if (!ep) return PLK_E_CONTRACT;
    if (!topology(*common, *ep) || !inputs_from(common->shared, in) || ep->world != in.world)
        return PLK_E_ARGUMENT;
    const uint64_t L = in.local_tokens, H = in.hidden, A = in.macrobatch, P = in.capacity;
    for (unsigned i = 0; i < ChannelCount; ++i) {
        const auto dtype = common->channels[i].dtype_id;
        auto& d = out->channels[i];
        d.minimum_allocation_bytes = 0;
        for (unsigned axis = 0; axis < d.rank; ++axis) {
            d.extents[axis] = 0;
            d.bit_strides[axis] = 0;
        }
        d.dtype_id = dtype;
        d.access = access(i);
        d.alignment_bytes = i == SchedulePeerRank || i == SchedulePeerTokenIdx ? 4 : 16;
        if (!active(i, in)) continue;
        switch (i) {
        case XBuffer: case DYBuffer:
            demand(d, dtype, d.access, 16, {L, H}, 16); break;
        case CombineBuffer: case DXBuffer:
            demand(d, dtype, d.access, 16, {L * in.topk, H}, 16); break;
        case XRouted: case DYRouted:
            demand(d, dtype, d.access, 8, {A, H}, 16); break;
        case XSc: case DYSc:
            demand(d, dtype, d.access, 8, {A / 128, H / 128, 32, 16}, 16); break;
        case XRoutedT: case DYRoutedT:
            demand(d, dtype, d.access, 8, {H, A}, 16); break;
        case XScT: case DYScT:
            demand(d, dtype, d.access, 8, {H / 128, A / 128, 32, 16}, 16); break;
        case YRouted: case XBF16: case DYBF16: case DXRouted:
            demand(d, dtype, d.access, 16, {A, H}, 16); break;
        case RouterWeightBuffer: case DRouterWeightBuffer:
            demand(d, dtype, d.access, 32, {L, uint64_t(in.topk)}, 16); break;
        case RouterWeights:
            demand(d, dtype, d.access, 32, {A}, 16); break;
        case DRouterWeightPartials:
            demand(d, dtype, d.access, 32, {A, uint64_t(in.intermediate / in.swiglu_nb)}, 16); break;
        default: demand(d, dtype, d.access, 32, {P}, 4); break;
        }
    }
    out->channel_count = ChannelCount;
    out->workspace_count = 0;
    out->geometry = {};
    return PLK_OK;
}

extern "C" int pipelink_params_build_mok_comm(
    const plk_common_v2* common, const void* args, const plk_resources_v2* resources,
    void* out, size_t cap, plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
    if (!session) return PLK_E_ARGUMENT;
    *session = nullptr;
    if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
    if (common->header.abi_version != PLK_HOST_ABI_V2) return PLK_E_VERSION;
    if (cap < sizeof(Residue) || transfer_capacity) return PLK_E_CAPACITY;
    if (resources->channel_count != ChannelCount || resources->workspace_count != 0)
        return PLK_E_CONTRACT;
    Inputs in;
    const auto* ep = plk_find_domain(common, 1);
    if (!ep) return PLK_E_CONTRACT;
    if (!topology(*common, *ep) || !inputs_from(resources->shared, in) || ep->world != in.world)
        return PLK_E_ARGUMENT;
    uint64_t addresses[ChannelCount];
    for (unsigned i = 0; i < ChannelCount; ++i) {
        const auto& r = resources->channels[i];
        if (r.origin_bit_offset) return PLK_E_RESOURCE;
        addresses[i] = r.allocation_address + r.origin_byte_offset;
    }
    // TK deletes gl's default constructor. Populate the zeroed host ABI
    // envelope by offset without constructing inactive TMA descriptors.
    std::memset(out, 0, sizeof(Residue));
    auto store = [&](size_t offset, const auto& value) {
        std::memcpy(static_cast<unsigned char*>(out) + offset, &value, sizeof(value));
    };
    auto peers = [&]<typename T>(size_t offset, unsigned channel, T*) {
        std::array<T*, 8> dst{};
        const auto& r = resources->channels[channel];
        if (!r.peer_addresses) return false;
        if (active(channel, in)) {
            for (unsigned rank = 0; rank < ep->world; ++rank)
                dst[rank] = reinterpret_cast<typename std::remove_reference_t<decltype(dst)>::value_type>(
                    r.peer_addresses[rank] + r.origin_byte_offset);
        }
        store(offset, dst);
        return true;
    };
    if (!peers(offsetof(Residue, x_buffer), XBuffer, static_cast<bf16*>(nullptr)) ||
        !peers(offsetof(Residue, combine_buffer), CombineBuffer, static_cast<bf16*>(nullptr)) ||
        !peers(offsetof(Residue, d_y_buffer), DYBuffer, static_cast<bf16*>(nullptr)) ||
        !peers(offsetof(Residue, d_x_buffer), DXBuffer, static_cast<bf16*>(nullptr)) ||
        !peers(offsetof(Residue, router_weight_buffer), RouterWeightBuffer, static_cast<float*>(nullptr)) ||
        !peers(offsetof(Residue, d_router_weight_buffer), DRouterWeightBuffer, static_cast<float*>(nullptr)))
        return PLK_E_RESOURCE;
    try {
        // Encode only live descriptors; inactive channel loans are zero-byte views.
        if (in.precision == MOK_PRECISION_MXFP8) {
            store(offsetof(Residue, x_routed), make_gl<mlp_fp8_gl>(addresses[XRouted], 1, 1, in.macrobatch, in.hidden));
            store(offsetof(Residue, x_routed_t), make_gl<mlp_fp8_gl>(addresses[XRoutedT], 1, 1, in.hidden, in.macrobatch));
            store(offsetof(Residue, x_sc), make_gl<sc_gl>(addresses[XSc], in.macrobatch / 128, in.hidden / 128, 32, 16));
            store(offsetof(Residue, x_sc_t), make_gl<sc_gl>(addresses[XScT], in.hidden / 128, in.macrobatch / 128, 32, 16));
            if (in.pass == MOK_PASS_BWD) {
                store(offsetof(Residue, d_y_routed), make_gl<mlp_fp8_gl>(addresses[DYRouted], 1, 1, in.macrobatch, in.hidden));
                store(offsetof(Residue, d_y_routed_t), make_gl<mlp_fp8_gl>(addresses[DYRoutedT], 1, 1, in.hidden, in.macrobatch));
                store(offsetof(Residue, d_y_sc), make_gl<sc_gl>(addresses[DYSc], in.macrobatch / 128, in.hidden / 128, 32, 16));
                store(offsetof(Residue, d_y_sc_t), make_gl<sc_gl>(addresses[DYScT], in.hidden / 128, in.macrobatch / 128, 32, 16));
            }
        } else {
            const auto x = make_gl<routed_bf16_gl>(addresses[XBF16], 1, 1, in.macrobatch, in.hidden);
            store(offsetof(Residue, x_bf16), x);
            store(offsetof(Residue, x_bf16_t), x);
            if (in.pass == MOK_PASS_BWD)
                store(offsetof(Residue, d_y_bf16), make_gl<routed_bf16_gl>(addresses[DYBF16], 1, 1, in.macrobatch, in.hidden));
        }
        if (in.pass == MOK_PASS_FWD)
            store(offsetof(Residue, y_routed), make_gl<epi_bf16_gl>(addresses[YRouted], 1, 1, in.macrobatch, in.hidden));
        if (in.pass == MOK_PASS_BWD) {
            store(offsetof(Residue, d_x_routed), make_gl<epi_bf16_gl>(addresses[DXRouted], 1, 1, in.macrobatch, in.hidden));
            store(offsetof(Residue, router_weights), make_gl<router_weight_gl>(addresses[RouterWeights], 1, 1, 1, in.macrobatch));
            store(offsetof(Residue, d_router_weight_partials), make_gl<d_router_weight_partials_gl>(
                addresses[DRouterWeightPartials], 1, 1, in.macrobatch, in.intermediate / in.swiglu_nb));
        }
        store(offsetof(Residue, schedule_peer_rank), index_gl{reinterpret_cast<int*>(addresses[SchedulePeerRank]), int(in.capacity)});
        store(offsetof(Residue, schedule_peer_token_idx), index_gl{reinterpret_cast<int*>(addresses[SchedulePeerTokenIdx]), int(in.capacity)});
        store(offsetof(Residue, facts), Facts{int32_t(in.mlp_mb), int32_t(in.mlp_nb)});
    } catch (...) {
        return PLK_E_BUILD;
    }
    return PLK_OK;
}
extern "C" int pipelink_params_release_mok_comm(plk_session* session) {
    return session ? PLK_E_RELEASE : PLK_OK;
}
