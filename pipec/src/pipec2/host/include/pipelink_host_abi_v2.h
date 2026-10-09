#ifndef PIPELINK_HOST_ABI_V2_H
#define PIPELINK_HOST_ABI_V2_H

/* Installed neutral PipeLink host ABI v2; manifest schema 3.0.
 * Owner IDs are group-local; neutral ABI type IDs are pipec-owned below.
 * See pipec/README.md for host ABI semantics; 0 is reserved/absent.
 *
 * Owners build their own params bytes (descriptors included) from the
 * caller-supplied resources handed to `build`. The broker never encodes a
 * descriptor; it checks each owner's transfer report against the device
 * endpoint the manifest binds (L-HOSTCHAN@open).
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PLK_HOST_ABI_V2 2u
#define PLK_OK 0
#define PLK_E_VERSION 1
#define PLK_E_CAPACITY 2
#define PLK_E_ARGUMENT 3
#define PLK_E_CONTRACT 4
#define PLK_E_RESOURCE 5
#define PLK_E_BUILD 6
#define PLK_E_RELEASE 7

#define PLK_SCALAR_U64 1u
#define PLK_SCALAR_I64 2u
#define PLK_SCALAR_F64 3u
#define PLK_SCALAR_BOOL 4u
#define PLK_ACCESS_READ 1u
#define PLK_ACCESS_WRITE 2u
#define PLK_ACCESS_READ_WRITE 3u

/* Transfer representation kinds reported by `build`. */
#define PLK_TRANSFER_TMA_TILED 1u   /* cuTensorMapEncodeTiled (also gather4) */
#define PLK_TRANSFER_TMA_IM2COL 2u  /* cuTensorMapEncodeIm2col */
#define PLK_TRANSFER_STRIDED 3u     /* plain strided view, no encoder */

/* No C enum, bitfield, packed attribute or language-native tuple crosses ABI. */
typedef struct plk_header_v2 {
    uint32_t abi_version;
    uint32_t struct_bytes;
} plk_header_v2;

typedef struct plk_digest_v2 {
    uint8_t sha256[32];
} plk_digest_v2;

/* The one scoped profile contract an owner binds to: the dtype table. */
typedef struct plk_profile_digest_v2 {
    plk_digest_v2 dtypes;
} plk_profile_digest_v2;

typedef struct plk_dim3_v2 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} plk_dim3_v2;

typedef struct plk_scalar_v2 {
    uint32_t scalar_id;
    uint32_t scalar_type;
    uint64_t value_bits;
} plk_scalar_v2;

/* Device-safe image: record_offset is relative to this header, not a pointer. */
typedef struct plk_shared_scalars_v2 {
    plk_header_v2 header;
    uint64_t image_bytes;
    uint32_t record_count;
    uint32_t record_offset;
} plk_shared_scalars_v2;

/* Host-only caller metadata; allocation addresses intentionally absent at query. */
typedef struct plk_tensor_meta_v2 {
    uint32_t channel_id;
    uint32_t dtype_id;
    uint32_t rank;
    uint32_t access;
    uint64_t allocation_bytes;
    uint64_t alignment_bytes;
    uint64_t origin_byte_offset;
    uint32_t origin_bit_offset;
    uint32_t reserved;
    const uint64_t *extents;
    const int64_t *bit_strides;
} plk_tensor_meta_v2;

typedef struct plk_geometry_v2 {
    uint32_t geometry_present;
    uint32_t launch_required;
    plk_dim3_v2 grid;
    plk_dim3_v2 block;
    plk_dim3_v2 cluster;
    uint32_t reserved;
    uint64_t dynamic_smem_bytes;
} plk_geometry_v2;

/* One caller-named communication domain (e.g. "tp", "ep") the owner declared with
 * PL_HOST_DOMAIN(id, name). rank/world are the caller's facts for this session; domain
 * identity is the name, never inferred from equal (rank, world). */
typedef struct plk_domain_v2 {
    uint32_t domain_id;            /* owner-local wire ID */
    uint32_t rank;
    uint32_t world;
    uint32_t reserved;
} plk_domain_v2;

typedef struct plk_common_v2 {
    plk_header_v2 header;
    uint32_t owner_id;
    uint32_t channel_count;
    uint32_t domain_count;
    uint32_t reserved;
    const plk_shared_scalars_v2 *shared;
    const plk_tensor_meta_v2 *channels;
    const plk_domain_v2 *domains;  /* the owner's declared domains, ascending domain_id */
    plk_dim3_v2 block;
    plk_dim3_v2 cluster;
    uint64_t dynamic_smem_bytes;
    plk_profile_digest_v2 profile;
    plk_digest_v2 owner_contract;
} plk_common_v2;

/* IDs, ranks, pointers and backing-array capacities are broker initialized;
 * the owner states the minimum allocation, alignment, extents and strides it
 * needs from the caller's resource. */
typedef struct plk_channel_demand_v2 {
    uint32_t channel_id;
    uint32_t dtype_id;
    uint32_t rank;
    uint32_t access;
    uint64_t minimum_allocation_bytes;
    uint64_t alignment_bytes;
    uint64_t *extents;
    int64_t *bit_strides;
} plk_channel_demand_v2;

typedef struct plk_workspace_demand_v2 {
    uint32_t workspace_id;
    uint32_t reserved;
    uint64_t bytes;
    uint64_t alignment_bytes;
} plk_workspace_demand_v2;

typedef struct plk_demands_v2 {
    plk_header_v2 header;
    uint32_t channel_count;
    uint32_t channel_capacity;
    plk_channel_demand_v2 *channels;
    uint32_t workspace_count;
    uint32_t workspace_capacity;
    plk_workspace_demand_v2 *workspaces;
    plk_geometry_v2 geometry;
} plk_demands_v2;

/* Host records containing bounded device allocations, never host dereferences.
 * A channel with a domain is one symmetric allocation per domain rank (identical
 * allocation_bytes and origin on every rank); its views are what the channel declared:
 * - peer: peer_addresses[r] is domain rank r's allocation base as addressable from THIS
 *   rank's CUDA context; peer_addresses[domain rank] == allocation_address. Else NULL.
 * - multicast: multicast_address is this allocation's base in the domain's multicast VA
 *   space (multimem.* only); offsets agree with allocation_address and every peer. Else 0.
 * Bytes written through one view and read through another are virtual aliases: the
 * owners need fence.proxy.alias on the synchronization path. */
typedef struct plk_device_resource_v2 {
    uint32_t channel_id;
    uint32_t access;
    uint64_t allocation_address;
    uint64_t allocation_bytes;
    uint64_t origin_byte_offset;
    uint32_t origin_bit_offset;
    uint32_t domain_id;               /* the owner's declared domain ID, 0 without a domain */
    const uint64_t *peer_addresses;   /* [domain world] or NULL */
    uint64_t multicast_address;       /* 0 without a multicast view */
} plk_device_resource_v2;

typedef struct plk_workspace_resource_v2 {
    uint32_t workspace_id;
    uint32_t reserved;
    uint64_t device_address;
    uint64_t bytes;
    uint64_t alignment_bytes;
} plk_workspace_resource_v2;

typedef struct plk_resources_v2 {
    plk_header_v2 header;
    uint32_t channel_count;
    uint32_t workspace_count;
    const plk_device_resource_v2 *channels;
    const plk_workspace_resource_v2 *workspaces;
    const plk_shared_scalars_v2 *shared;
    uint64_t cuda_stream_token;
    plk_geometry_v2 geometry;
} plk_resources_v2;

/* The only vendor detail the broker reads (vendor_kind == PLK_TRANSFER_TMA_TILED). */
typedef struct plk_tma_tiled_detail_v2 {
    uint32_t swizzle;               /* CUtensorMapSwizzle numeric value */
    uint32_t rank;
    uint64_t box_elements[5];
    uint32_t element_strides[5];
    uint32_t reserved;
} plk_tma_tiled_detail_v2;

/* One representation-independent self-description per declared TRANSFER,
 * filled by `build`. The broker pre-fills transfer_id (declared order) and
 * zeroes the rest; the owner states what it encoded. Footprint and credit are
 * distinct numbers from the owner's own issue/multicast configuration. */
typedef struct plk_transfer_report_v2 {
    uint32_t transfer_id;              /* declared TRANSFER id (broker initialized) */
    uint32_t access;                   /* PLK_ACCESS_* actually encoded */
    uint64_t residue_offset;           /* representation bytes inside this owner's residue */
    uint64_t representation_bytes;     /* 128 for a CUtensorMap; the view size for STRIDED */
    uint64_t representation_align;
    uint32_t storage_dtype_id;         /* profile dtype of the encoded/stored element */
    uint32_t ring_depth;               /* ring geometry the owner's addressing assumes; 1 if none */
    uint64_t ring_stage_stride_bytes;
    uint64_t footprint_bytes_per_stage; /* bytes landing in THIS CTA's endpoint stage buffer */
    uint64_t credit_bytes_per_stage;    /* bytes credited to THIS CTA's completion barrier per
                                           stage, including peer-multicast arrivals */
    uint64_t source_origin_bits;       /* resource-relative origin of the addressed span */
    uint64_t source_span_bits;         /* bounded span the transfer may touch in the resource */
    uint32_t vendor_kind;              /* PLK_TRANSFER_* or 0 */
    uint32_t reserved;
    const void *vendor_detail;         /* plk_tma_tiled_detail_v2 for TMA_TILED; else diagnostic */
} plk_transfer_report_v2;

typedef struct plk_layout_entry_v2 {
    uint32_t abi_type_id;
    uint32_t reserved;
    uint64_t bytes;
    uint64_t alignment_bytes;
    plk_digest_v2 compiler_layout;
} plk_layout_entry_v2;

typedef struct plk_abi_report_v2 {
    plk_header_v2 header;
    uint32_t entry_count;
    uint32_t entry_capacity;
    plk_layout_entry_v2 *entries;
    plk_profile_digest_v2 profile;
    plk_digest_v2 owner_contract;
} plk_abi_report_v2;

/* The neutral inventory is owned here, never restated by a stage author.
 * These IDs leave owner-local inventories untouched and fit C11 enum values.
 * Layouts are measured by each compiler; no layout numbers live in this list. */
#define PLK_ABI_TYPE_ID_MIN 0x70000000u
#define PLK_ABI_TYPE_ID_MAX 0x700000ffu
#define PLK_HOST_ABI_TYPES(X) \
  X(0x70000001, plk_abi_report_v2, entries, entry_capacity, entry_count, header, owner_contract, profile) \
  X(0x70000002, plk_channel_demand_v2, access, alignment_bytes, bit_strides, channel_id, dtype_id, extents, minimum_allocation_bytes, rank) \
  X(0x70000003, plk_common_v2, block, channel_count, channels, cluster, domain_count, domains, dynamic_smem_bytes, header, owner_contract, owner_id, profile, reserved, shared) \
  X(0x70000004, plk_demands_v2, channel_capacity, channel_count, channels, geometry, header, workspace_capacity, workspace_count, workspaces) \
  X(0x70000005, plk_tma_tiled_detail_v2, box_elements, element_strides, rank, reserved, swizzle) \
  X(0x70000006, plk_device_resource_v2, access, allocation_address, allocation_bytes, channel_id, domain_id, multicast_address, origin_bit_offset, origin_byte_offset, peer_addresses) \
  X(0x70000007, plk_digest_v2, sha256) \
  X(0x70000008, plk_dim3_v2, x, y, z) \
  X(0x70000009, plk_geometry_v2, block, cluster, dynamic_smem_bytes, geometry_present, grid, launch_required, reserved) \
  X(0x7000000a, plk_header_v2, abi_version, struct_bytes) \
  X(0x7000000b, plk_layout_entry_v2, abi_type_id, alignment_bytes, bytes, compiler_layout, reserved) \
  X(0x7000000c, plk_profile_digest_v2, dtypes) \
  X(0x7000000d, plk_resources_v2, channel_count, channels, cuda_stream_token, geometry, header, shared, workspace_count, workspaces) \
  X(0x7000000e, plk_scalar_v2, scalar_id, scalar_type, value_bits) \
  X(0x7000000f, plk_shared_scalars_v2, header, image_bytes, record_count, record_offset) \
  X(0x70000010, plk_transfer_report_v2, access, credit_bytes_per_stage, footprint_bytes_per_stage, representation_align, representation_bytes, reserved, residue_offset, ring_depth, ring_stage_stride_bytes, source_origin_bits, source_span_bits, storage_dtype_id, transfer_id, vendor_detail, vendor_kind) \
  X(0x70000011, plk_tensor_meta_v2, access, alignment_bytes, allocation_bytes, bit_strides, channel_id, dtype_id, extents, origin_bit_offset, origin_byte_offset, rank, reserved) \
  X(0x70000012, plk_workspace_demand_v2, alignment_bytes, bytes, reserved, workspace_id) \
  X(0x70000013, plk_workspace_resource_v2, alignment_bytes, bytes, device_address, reserved, workspace_id) \
  X(0x70000014, plk_domain_v2, domain_id, rank, reserved, world)

#define PLK_DETAIL_ABI_ID(id, T, ...) PLK_ABI_TYPE_ID_##T = id,
enum { PLK_HOST_ABI_TYPES(PLK_DETAIL_ABI_ID) };
#undef PLK_DETAIL_ABI_ID
#define PLK_ABI_TYPE_ID(T) PLK_ABI_TYPE_ID_##T

/* The owner's declared domain `domain_id`, or NULL if it did not declare it. */
static inline const plk_domain_v2 *plk_find_domain(const plk_common_v2 *common, uint32_t domain_id) {
    for (uint32_t i = 0; i < common->domain_count; ++i)
        if (common->domains[i].domain_id == domain_id) return &common->domains[i];
    return NULL;
}

/* Incomplete by design: provider-owned host bookkeeping, no shared layout.
 * One session per build; released exactly once by `release`. */
typedef struct plk_session_v2 plk_session;

typedef int (*plk_abi_fn_v2)(plk_abi_report_v2 *out);
typedef int (*plk_query_fn_v2)(const plk_common_v2 *common, const void *args,
                               plk_demands_v2 *out);
typedef int (*plk_build_fn_v2)(const plk_common_v2 *common, const void *args,
                               const plk_resources_v2 *resources, void *residue,
                               size_t residue_cap, plk_transfer_report_v2 *transfers,
                               uint32_t transfer_capacity, plk_session **session);
typedef int (*plk_release_fn_v2)(plk_session *session);

#ifdef __cplusplus
}
#endif
#endif
