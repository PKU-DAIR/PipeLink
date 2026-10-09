#ifndef MOK_ACT_PROVIDER
#define MOK_ACT_PROVIDER
#endif
#include "act.cu"
#include <climits>
#include <cstring>
#include <exception>

namespace {
constexpr uint32_t ChannelCount = 31;
struct Inputs {
    int64_t local = 0, hidden = 0, intermediate = 0, capacity = 0, macro = 0;
    int64_t mlp_mb = 0, mlp_nb = 0;
    int64_t pass = -1, precision = -1, world = -1, clamp = -1;
    double limit = 0;
};

bool inputs_from(const plk_shared_scalars_v2* shared, Inputs& in) {
    if (!shared) return false;
    const auto* records = reinterpret_cast<const plk_scalar_v2*>(
        reinterpret_cast<const unsigned char*>(shared) + shared->record_offset);
    bool has_limit = false;
    for (uint32_t i = 0; i < shared->record_count; ++i) {
        const auto& record = records[i];
        if (record.scalar_id == MOK_SCALAR_SWIGLU_LIMIT) {
            if (record.scalar_type != PLK_SCALAR_F64) return false;
            std::memcpy(&in.limit, &record.value_bits, sizeof(in.limit));
            has_limit = true;
            continue;
        }
        int64_t* field = nullptr;
        switch (record.scalar_id) {
            case MOK_SCALAR_NUM_LOCAL_TOKENS: field = &in.local; break;
            case MOK_SCALAR_HIDDEN: field = &in.hidden; break;
            case MOK_SCALAR_INTERMEDIATE: field = &in.intermediate; break;
            case MOK_SCALAR_SCHEDULE_CAPACITY: field = &in.capacity; break;
            case MOK_SCALAR_MACROBATCH_SIZE: field = &in.macro; break;
            case MOK_SCALAR_MLP_MB: field = &in.mlp_mb; break;
            case MOK_SCALAR_MLP_NB: field = &in.mlp_nb; break;
            case MOK_SCALAR_PASS: field = &in.pass; break;
            case MOK_SCALAR_PRECISION: field = &in.precision; break;
            case MOK_SCALAR_WORLD: field = &in.world; break;
            case MOK_SCALAR_CLAMP: field = &in.clamp; break;
            default: continue;
        }
        if (record.scalar_type != PLK_SCALAR_I64) return false;
        std::memcpy(field, &record.value_bits, sizeof(*field));
    }
    using C = mok_act_owner::config;
    // GEMM parent slots must contain whole activation tiles. Minibatch rules are
    // framework-owned (scalar 7); act validates only its declared shape inputs.
    return has_limit && !(in.limit < 0) &&
        in.pass >= MOK_PASS_FWD && in.pass <= MOK_PASS_REC &&
        (in.precision == MOK_PRECISION_BF16 || in.precision == MOK_PRECISION_MXFP8) &&
        (in.world == 1 || in.world == 4) && (in.clamp == 0 || in.clamp == 1) &&
        in.mlp_mb > 0 && in.mlp_mb <= INT_MAX && in.mlp_mb % C::SWIGLU_Mb == 0 &&
        in.mlp_nb > 0 && in.mlp_nb <= INT_MAX && in.mlp_nb % C::SWIGLU_Nb == 0 &&
        in.local >= 512 && in.local <= INT_MAX && in.local % in.mlp_mb == 0 &&
        in.hidden > 0 && in.hidden <= INT_MAX && in.hidden % 256 == 0 &&
        in.intermediate > 0 && in.intermediate <= INT_MAX && in.intermediate % in.mlp_nb == 0 &&
        in.capacity > 0 && in.capacity <= INT_MAX && in.capacity % in.mlp_mb == 0 &&
        in.macro > 0 && in.macro <= INT_MAX && in.macro % in.mlp_mb == 0;
}

bool header_is(const plk_header_v2& header, size_t bytes) {
    return header.abi_version == PLK_HOST_ABI_V2 && header.struct_bytes == bytes;
}
bool topology_is(const plk_common_v2& common) {
    return common.block.x == 256 && common.block.y == 1 && common.block.z == 1 &&
        common.cluster.x == 2 && common.cluster.y == 1 && common.cluster.z == 1;
}

struct Shape {
    uint32_t rank, bits, access, alignment;
    uint64_t extents[4];
    int64_t strides[4];
    uint64_t bytes;
};
Shape channel_shape(uint32_t index, const Inputs& in) {
    const uint32_t id = index + 1;
    const bool scale = id == 7 || id == 9 || id == 12 || id == 14 ||
        id == 20 || id == 22 || id == 24 || id == 26;
    const bool transposed = id == 8 || id == 9 || (id >= 23 && id <= 26);
    const bool fp8 = (id >= 6 && id <= 9) || (id >= 11 && id <= 14) ||
        (id >= 19 && id <= 26);
    const bool bf16_only = id == 10 || id == 27 || id == 28;
    const bool backward_only = id >= 15 && id <= 30;
    const bool active = (!backward_only || in.pass == MOK_PASS_BWD) &&
        (!fp8 || in.precision == MOK_PRECISION_MXFP8) &&
        (!bf16_only || in.precision == MOK_PRECISION_BF16);
    Shape shape{};
    shape.bits = id >= 29 ? 32 : (fp8 ? 8 : 16);
    shape.access = (id <= 4 || (id >= 11 && id <= 16) || id == 29 || id == 31)
        ? PLK_ACCESS_READ : PLK_ACCESS_WRITE;
    shape.rank = scale ? 4 : ((id == 29 || id == 31) ? 1 : 2);
    shape.alignment = id == 31 ? 4 : 16;
    if (!active) return shape;
    uint64_t rows = (id <= 2 || id == 5 || id == 15 || id == 17 || id == 18)
        ? in.local : in.macro;
    uint64_t cols = in.intermediate;
    if (transposed) { const auto tmp = rows; rows = cols; cols = tmp; }
    if (scale) {
        shape.extents[0] = rows / 128;
        shape.extents[1] = cols / 128;
        shape.extents[2] = 32;
        shape.extents[3] = 16;
    } else if (id == 31) {
        shape.extents[0] = in.capacity;
    } else {
        shape.extents[0] = rows;
        if (shape.rank == 2)
            shape.extents[1] = id == 30 ? cols / mok_act_owner::config::SWIGLU_Nb : cols;
    }
    uint64_t stride = shape.bits;
    for (int d = int(shape.rank) - 1; d >= 0; --d) {
        shape.strides[d] = stride;
        stride *= shape.extents[d];
    }
    shape.bytes = stride / 8;
    return shape;
}

template <mok_port::RoutedPrecision Precision>
void build_residue(const Inputs& in, const uint64_t* a, void* out) {
    using O = mok_act_owner::Owner<Precision, false>;
    using R = typename O::Residue;
    // TK deletes gl's default constructor. Serialize active descriptors into the
    // zeroed host residue envelope without constructing inactive TMA descriptors.
    std::memset(out, 0, sizeof(MokActResidue));
    auto store = [&](size_t offset, const auto& value) {
        std::memcpy(static_cast<unsigned char*>(out) + offset, &value, sizeof(value));
    };
    auto matrix = [&]<class Descriptor>(size_t offset, unsigned id, int rows, int cols) {
        if (channel_shape(id - 1, in).bytes)
            store(offset, kittens::make_gl<Descriptor>(a[id - 1], 1, 1, rows, cols));
    };
    auto alias = [&](size_t dst, size_t src, size_t bytes) {
        auto* storage = static_cast<unsigned char*>(out);
        std::memcpy(storage + dst, storage + src, bytes);
    };
#define MOK_ACT_MATRIX(member, id, rows, cols) \
    matrix.template operator()<decltype(R::member)>(offsetof(R, member), id, rows, cols)
    MOK_ACT_MATRIX(gate_shared, 1, in.local, in.intermediate);
    MOK_ACT_MATRIX(up_shared, 2, in.local, in.intermediate);
    MOK_ACT_MATRIX(gate_routed, 3, in.macro, in.intermediate);
    MOK_ACT_MATRIX(up_routed, 4, in.macro, in.intermediate);
    MOK_ACT_MATRIX(hidden_shared, 5, in.local, in.intermediate);
    MOK_ACT_MATRIX(d_hidden_shared, 15, in.local, in.intermediate);
    MOK_ACT_MATRIX(d_hidden_routed, 16, in.macro, in.intermediate);
    MOK_ACT_MATRIX(d_gate_shared, 17, in.local, in.intermediate);
    MOK_ACT_MATRIX(d_up_shared, 18, in.local, in.intermediate);
    if constexpr (O::USE_MXFP8) {
        auto scale = [&](size_t offset, unsigned id, int rows, int cols) {
            if (channel_shape(id - 1, in).bytes)
                store(offset, kittens::make_gl<typename O::sc_gl>(
                    a[id - 1], rows / 128, cols / 128, 32, 16));
        };
        MOK_ACT_MATRIX(hidden_routed, 6, in.macro, in.intermediate);
        scale(offsetof(R, hidden_sc), 7, in.macro, in.intermediate);
        MOK_ACT_MATRIX(hidden_t, 8, in.intermediate, in.macro);
        scale(offsetof(R, hidden_sc_t), 9, in.intermediate, in.macro);
        MOK_ACT_MATRIX(saved_gate, 11, in.macro, in.intermediate);
        scale(offsetof(R, gate_sc), 12, in.macro, in.intermediate);
        MOK_ACT_MATRIX(saved_up, 13, in.macro, in.intermediate);
        scale(offsetof(R, up_sc), 14, in.macro, in.intermediate);
        MOK_ACT_MATRIX(d_gate, 19, in.macro, in.intermediate);
        scale(offsetof(R, d_gate_sc), 20, in.macro, in.intermediate);
        MOK_ACT_MATRIX(d_up, 21, in.macro, in.intermediate);
        scale(offsetof(R, d_up_sc), 22, in.macro, in.intermediate);
        MOK_ACT_MATRIX(d_gate_t, 23, in.intermediate, in.macro);
        scale(offsetof(R, d_gate_sc_t), 24, in.intermediate, in.macro);
        MOK_ACT_MATRIX(d_up_t, 25, in.intermediate, in.macro);
        scale(offsetof(R, d_up_sc_t), 26, in.intermediate, in.macro);
    } else {
        MOK_ACT_MATRIX(hidden_routed, 10, in.macro, in.intermediate);
        alias(offsetof(R, hidden_t), offsetof(R, hidden_routed), sizeof(decltype(R::hidden_t)));
        MOK_ACT_MATRIX(saved_gate, 3, in.macro, in.intermediate);
        MOK_ACT_MATRIX(saved_up, 4, in.macro, in.intermediate);
        MOK_ACT_MATRIX(d_gate, 27, in.macro, in.intermediate);
        MOK_ACT_MATRIX(d_up, 28, in.macro, in.intermediate);
        alias(offsetof(R, d_gate_t), offsetof(R, d_gate), sizeof(decltype(R::d_gate_t)));
        alias(offsetof(R, d_up_t), offsetof(R, d_up), sizeof(decltype(R::d_up_t)));
    }
    if (in.pass == MOK_PASS_BWD) {
        store(offsetof(R, router_weights),
            kittens::make_gl<typename O::router_weight_gl>(a[28], 1, 1, 1, in.macro));
        MOK_ACT_MATRIX(d_router_weight_partials, 30, in.macro,
            in.intermediate / mok_act_owner::config::SWIGLU_Nb);
    }
#undef MOK_ACT_MATRIX
    store(offsetof(R, schedule_peer_rank),
        typename O::index_gl{reinterpret_cast<int32_t*>(a[30]), static_cast<int>(in.capacity)});
    store(offsetof(R, facts),
        mok_act_owner::Facts{static_cast<int32_t>(in.mlp_mb), static_cast<int32_t>(in.mlp_nb)});
    store(offsetof(R, swiglu_limit), static_cast<float>(in.limit));
}
} // namespace

extern "C" int pipelink_params_query_mok_act(
        const plk_common_v2* common, const void* args, plk_demands_v2* out) {
    if (!common || !out || args) return PLK_E_ARGUMENT;
    if (!header_is(common->header, sizeof(*common)) || !header_is(out->header, sizeof(*out)))
        return PLK_E_VERSION;
    if (!topology_is(*common)) return PLK_E_CONTRACT;
    if (common->channel_count != ChannelCount || !common->channels ||
        out->channel_capacity != ChannelCount || !out->channels || out->workspace_capacity)
        return PLK_E_CONTRACT;
    Inputs in;
    if (!inputs_from(common->shared, in)) return PLK_E_ARGUMENT;
    const auto* ep = plk_find_domain(common, 1);
    if (!ep || ep->world != in.world) return PLK_E_CONTRACT;
    for (uint32_t i = 0; i < ChannelCount; ++i) {
        const auto& source = common->channels[i];
        auto& demand = out->channels[i];
        const Shape shape = channel_shape(i, in);
        if (source.channel_id != i + 1 || demand.channel_id != source.channel_id ||
            source.rank != shape.rank || demand.rank != shape.rank || source.access != shape.access ||
            !source.extents || !source.bit_strides || !demand.extents || !demand.bit_strides)
            return PLK_E_CONTRACT;
        if (source.origin_bit_offset) return PLK_E_ARGUMENT;
        for (uint32_t d = 0; d < shape.rank; ++d) {
            if (source.extents[d] != shape.extents[d] || source.bit_strides[d] != shape.strides[d])
                return PLK_E_ARGUMENT;
            demand.extents[d] = shape.extents[d];
            demand.bit_strides[d] = shape.strides[d];
        }
        demand.dtype_id = source.dtype_id;
        demand.access = shape.access;
        demand.minimum_allocation_bytes = shape.bytes;
        demand.alignment_bytes = shape.alignment;
    }
    out->channel_count = ChannelCount;
    out->workspace_count = 0;
    out->geometry = {}; // The framework owns launch geometry.
    return PLK_OK;
}

extern "C" int pipelink_params_build_mok_act(const plk_common_v2* common,
        const void* args, const plk_resources_v2* resources, void* out, size_t cap,
        plk_transfer_report_v2*, uint32_t transfer_capacity, plk_session** session) {
    if (!session) return PLK_E_ARGUMENT;
    *session = nullptr;
    if (!common || args || !resources || !out) return PLK_E_ARGUMENT;
    if (!header_is(common->header, sizeof(*common)) ||
        !header_is(resources->header, sizeof(*resources))) return PLK_E_VERSION;
    if (cap < sizeof(MokActResidue) || transfer_capacity) return PLK_E_CAPACITY;
    if (!topology_is(*common) || resources->channel_count != ChannelCount ||
        !resources->channels || resources->workspace_count) return PLK_E_CONTRACT;
    Inputs in;
    if (!inputs_from(resources->shared, in)) return PLK_E_ARGUMENT;
    const auto* ep = plk_find_domain(common, 1);
    if (!ep || ep->world != in.world) return PLK_E_CONTRACT;
    uint64_t addresses[ChannelCount];
    for (uint32_t i = 0; i < ChannelCount; ++i) {
        const auto& resource = resources->channels[i];
        const Shape shape = channel_shape(i, in);
        if (resource.channel_id != i + 1 || resource.access != shape.access) return PLK_E_CONTRACT;
        if (!shape.bytes) {
            if (resource.allocation_address || resource.allocation_bytes ||
                resource.origin_byte_offset || resource.origin_bit_offset) return PLK_E_RESOURCE;
            addresses[i] = 0;
            continue;
        }
        if (!resource.allocation_address || resource.origin_bit_offset ||
            resource.origin_byte_offset > resource.allocation_bytes ||
            shape.bytes > resource.allocation_bytes - resource.origin_byte_offset ||
            resource.allocation_address > UINT64_MAX - resource.origin_byte_offset)
            return PLK_E_RESOURCE;
        addresses[i] = resource.allocation_address + resource.origin_byte_offset;
        if (addresses[i] % shape.alignment) return PLK_E_RESOURCE;
    }
    try {
        if (in.precision == MOK_PRECISION_MXFP8)
            build_residue<mok_port::RoutedPrecision::MXFP8>(in, addresses, out);
        else
            build_residue<mok_port::RoutedPrecision::BF16>(in, addresses, out);
    } catch (const std::exception&) {
        return PLK_E_BUILD;
    }
    return PLK_OK;
}

extern "C" int pipelink_params_release_mok_act(plk_session* session) {
    return session ? PLK_E_RELEASE : PLK_OK;
}
