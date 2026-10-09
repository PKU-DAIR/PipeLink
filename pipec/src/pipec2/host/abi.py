"""The installed C v2 protocol and compiler-derived layout inventories."""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

PLK_HOST_ABI_V2 = 2
NEUTRAL_TYPE_ID_MIN = 0x70000000
NEUTRAL_TYPE_ID_MAX = 0x700000FF
PLK_OK = 0
PLK_E_VERSION = 1
PLK_E_CAPACITY = 2
PLK_E_ARGUMENT = 3
PLK_E_CONTRACT = 4
PLK_E_RESOURCE = 5
PLK_E_BUILD = 6
PLK_E_RELEASE = 7
PLK_SCALAR_U64 = 1
PLK_SCALAR_I64 = 2
PLK_SCALAR_F64 = 3
PLK_SCALAR_BOOL = 4
PLK_ACCESS_READ = 1
PLK_ACCESS_WRITE = 2
PLK_ACCESS_READ_WRITE = 3
PLK_TRANSFER_TMA_TILED = 1
PLK_TRANSFER_TMA_IM2COL = 2
PLK_TRANSFER_STRIDED = 3


class plk_header_v2(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("struct_bytes", ct.c_uint32),
    ]


class plk_digest_v2(ct.Structure):
    _fields_ = [
        ("sha256", ct.c_uint8 * 32),
    ]


class plk_profile_digest_v2(ct.Structure):
    _fields_ = [
        ("dtypes", plk_digest_v2),
    ]


class plk_dim3_v2(ct.Structure):
    _fields_ = [
        ("x", ct.c_uint32),
        ("y", ct.c_uint32),
        ("z", ct.c_uint32),
    ]


class plk_scalar_v2(ct.Structure):
    _fields_ = [
        ("scalar_id", ct.c_uint32),
        ("scalar_type", ct.c_uint32),
        ("value_bits", ct.c_uint64),
    ]


class plk_shared_scalars_v2(ct.Structure):
    _fields_ = [
        ("header", plk_header_v2),
        ("image_bytes", ct.c_uint64),
        ("record_count", ct.c_uint32),
        ("record_offset", ct.c_uint32),
    ]


class plk_tensor_meta_v2(ct.Structure):
    _fields_ = [
        ("channel_id", ct.c_uint32),
        ("dtype_id", ct.c_uint32),
        ("rank", ct.c_uint32),
        ("access", ct.c_uint32),
        ("allocation_bytes", ct.c_uint64),
        ("alignment_bytes", ct.c_uint64),
        ("origin_byte_offset", ct.c_uint64),
        ("origin_bit_offset", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("extents", ct.POINTER(ct.c_uint64)),
        ("bit_strides", ct.POINTER(ct.c_int64)),
    ]


class plk_geometry_v2(ct.Structure):
    _fields_ = [
        ("geometry_present", ct.c_uint32),
        ("launch_required", ct.c_uint32),
        ("grid", plk_dim3_v2),
        ("block", plk_dim3_v2),
        ("cluster", plk_dim3_v2),
        ("reserved", ct.c_uint32),
        ("dynamic_smem_bytes", ct.c_uint64),
    ]


class plk_domain_v2(ct.Structure):
    _fields_ = [
        ("domain_id", ct.c_uint32),
        ("rank", ct.c_uint32),
        ("world", ct.c_uint32),
        ("reserved", ct.c_uint32),
    ]


class plk_common_v2(ct.Structure):
    _fields_ = [
        ("header", plk_header_v2),
        ("owner_id", ct.c_uint32),
        ("channel_count", ct.c_uint32),
        ("domain_count", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("shared", ct.POINTER(plk_shared_scalars_v2)),
        ("channels", ct.POINTER(plk_tensor_meta_v2)),
        ("domains", ct.POINTER(plk_domain_v2)),
        ("block", plk_dim3_v2),
        ("cluster", plk_dim3_v2),
        ("dynamic_smem_bytes", ct.c_uint64),
        ("profile", plk_profile_digest_v2),
        ("owner_contract", plk_digest_v2),
    ]


class plk_channel_demand_v2(ct.Structure):
    _fields_ = [
        ("channel_id", ct.c_uint32),
        ("dtype_id", ct.c_uint32),
        ("rank", ct.c_uint32),
        ("access", ct.c_uint32),
        ("minimum_allocation_bytes", ct.c_uint64),
        ("alignment_bytes", ct.c_uint64),
        ("extents", ct.POINTER(ct.c_uint64)),
        ("bit_strides", ct.POINTER(ct.c_int64)),
    ]


class plk_workspace_demand_v2(ct.Structure):
    _fields_ = [
        ("workspace_id", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("bytes", ct.c_uint64),
        ("alignment_bytes", ct.c_uint64),
    ]


class plk_demands_v2(ct.Structure):
    _fields_ = [
        ("header", plk_header_v2),
        ("channel_count", ct.c_uint32),
        ("channel_capacity", ct.c_uint32),
        ("channels", ct.POINTER(plk_channel_demand_v2)),
        ("workspace_count", ct.c_uint32),
        ("workspace_capacity", ct.c_uint32),
        ("workspaces", ct.POINTER(plk_workspace_demand_v2)),
        ("geometry", plk_geometry_v2),
    ]


class plk_device_resource_v2(ct.Structure):
    _fields_ = [
        ("channel_id", ct.c_uint32),
        ("access", ct.c_uint32),
        ("allocation_address", ct.c_uint64),
        ("allocation_bytes", ct.c_uint64),
        ("origin_byte_offset", ct.c_uint64),
        ("origin_bit_offset", ct.c_uint32),
        ("domain_id", ct.c_uint32),
        ("peer_addresses", ct.POINTER(ct.c_uint64)),
        ("multicast_address", ct.c_uint64),
    ]


class plk_workspace_resource_v2(ct.Structure):
    _fields_ = [
        ("workspace_id", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("device_address", ct.c_uint64),
        ("bytes", ct.c_uint64),
        ("alignment_bytes", ct.c_uint64),
    ]


class plk_resources_v2(ct.Structure):
    _fields_ = [
        ("header", plk_header_v2),
        ("channel_count", ct.c_uint32),
        ("workspace_count", ct.c_uint32),
        ("channels", ct.POINTER(plk_device_resource_v2)),
        ("workspaces", ct.POINTER(plk_workspace_resource_v2)),
        ("shared", ct.POINTER(plk_shared_scalars_v2)),
        ("cuda_stream_token", ct.c_uint64),
        ("geometry", plk_geometry_v2),
    ]


class plk_tma_tiled_detail_v2(ct.Structure):
    _fields_ = [
        ("swizzle", ct.c_uint32),
        ("rank", ct.c_uint32),
        ("box_elements", ct.c_uint64 * 5),
        ("element_strides", ct.c_uint32 * 5),
        ("reserved", ct.c_uint32),
    ]


class plk_transfer_report_v2(ct.Structure):
    _fields_ = [
        ("transfer_id", ct.c_uint32),
        ("access", ct.c_uint32),
        ("residue_offset", ct.c_uint64),
        ("representation_bytes", ct.c_uint64),
        ("representation_align", ct.c_uint64),
        ("storage_dtype_id", ct.c_uint32),
        ("ring_depth", ct.c_uint32),
        ("ring_stage_stride_bytes", ct.c_uint64),
        ("footprint_bytes_per_stage", ct.c_uint64),
        ("credit_bytes_per_stage", ct.c_uint64),
        ("source_origin_bits", ct.c_uint64),
        ("source_span_bits", ct.c_uint64),
        ("vendor_kind", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("vendor_detail", ct.c_void_p),
    ]


class plk_layout_entry_v2(ct.Structure):
    _fields_ = [
        ("abi_type_id", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("bytes", ct.c_uint64),
        ("alignment_bytes", ct.c_uint64),
        ("compiler_layout", plk_digest_v2),
    ]


class plk_abi_report_v2(ct.Structure):
    _fields_ = [
        ("header", plk_header_v2),
        ("entry_count", ct.c_uint32),
        ("entry_capacity", ct.c_uint32),
        ("entries", ct.POINTER(plk_layout_entry_v2)),
        ("profile", plk_profile_digest_v2),
        ("owner_contract", plk_digest_v2),
    ]


class plk_session_v2(ct.Structure):
    pass


plk_session = plk_session_v2
plk_abi_fn_v2 = ct.CFUNCTYPE(ct.c_int, ct.POINTER(plk_abi_report_v2))
plk_query_fn_v2 = ct.CFUNCTYPE(
    ct.c_int, ct.POINTER(plk_common_v2), ct.c_void_p, ct.POINTER(plk_demands_v2)
)
plk_build_fn_v2 = ct.CFUNCTYPE(
    ct.c_int, ct.POINTER(plk_common_v2), ct.c_void_p,
    ct.POINTER(plk_resources_v2), ct.c_void_p, ct.c_size_t,
    ct.POINTER(plk_transfer_report_v2), ct.c_uint32,
    ct.POINTER(ct.POINTER(plk_session)),
)
plk_release_fn_v2 = ct.CFUNCTYPE(ct.c_int, ct.POINTER(plk_session))

ABI_TYPES = {
    name: value for name, value in tuple(globals().items())
    if name.startswith("plk_") and isinstance(value, type)
    and issubclass(value, ct.Structure) and hasattr(value, "_fields_")
}


def header_path() -> Path:
    return Path(__file__).with_name("include") / "pipelink_host_abi_v2.h"


class aligned_buffer:
    """Own zeroed host bytes at a C ABI alignment; a zero-sized buffer is null."""

    def __init__(self, size: int, align: int):
        if type(size) is not int or size < 0:
            raise ValueError("buffer size must be a nonnegative integer")
        if type(align) is not int or align < 1 or align & (align - 1):
            raise ValueError("buffer alignment must be a positive power of two")
        self.size = size
        self.align = align
        self._storage = ct.create_string_buffer(size + align - 1) if size else None
        self.address = (ct.addressof(self._storage) + align - 1) & -align if size else 0

    @property
    def bytes(self) -> bytes:
        return ct.string_at(self.address, self.size) if self.size else b""

    def write(self, data: bytes, offset: int = 0) -> None:
        if type(offset) is not int or offset < 0 or offset + len(data) > self.size:
            raise ValueError("write exceeds aligned buffer")
        if data:
            ct.memmove(self.address + offset, data, len(data))


def _inventory(extra_types):
    inventory = {name: [field for field, _ in typ._fields_] for name, typ in ABI_TYPES.items()}
    for name, fields in (extra_types or {}).items():
        fields = list(fields)
        if name in inventory and set(fields) != set(inventory[name]):
            raise ValueError(f"neutral ABI field inventory cannot be overridden: {name}")
        inventory[name] = fields
    for name, fields in inventory.items():
        if not re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", name):
            raise ValueError(f"invalid ABI C type name: {name!r}")
        if len(set(fields)) != len(fields):
            raise ValueError(f"duplicate ABI fields: {name}")
        for field in fields:
            if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", field):
                raise ValueError(f"invalid ABI field name: {field!r}")
    return inventory


def layout_probe_source(extra_headers=(), extra_types=None) -> str:
    """Generate a C11/C++17 TU; every reported number is a compiler expression.

    Own nested records must be listed in extra_types along with args/residue/role
    views. The complete neutral pointer-reachable inventory is always included.
    Field declaration order is deliberately not the report order.
    """
    inventory = _inventory(extra_types)
    lines = ["#include <stdio.h>", "#include <stddef.h>"]
    for header in (header_path(), *extra_headers):
        name = str(Path(header).resolve())
        if any(ch in name for ch in '\n\r"'):
            raise ValueError("invalid probe header path")
        lines.append(f'#include "{name}"')
    lines += ["#ifdef __cplusplus", "#define PLK_ALIGNOF(T) alignof(T)", "#else",
              "#define PLK_ALIGNOF(T) _Alignof(T)", "#endif",
              '#define PLK_REPORT_TYPE_ID(id, T, ...) printf("I\\t" #T "\\t%u\\t" #__VA_ARGS__ "\\n", (unsigned)(id));',
              "int main(void) {", "  PLK_HOST_ABI_TYPES(PLK_REPORT_TYPE_ID)"]
    for name, fields in sorted(inventory.items()):
        lines.append(f'  printf("T\\t{name}\\t%zu\\t%zu\\n", sizeof({name}), PLK_ALIGNOF({name}));')
        for field in sorted(fields):
            lines.append(f'  printf("F\\t{name}\\t{field}\\t%zu\\t%zu\\n", offsetof({name}, {field}), sizeof((({name}*)0)->{field}));')
    lines += ["  return 0;", "}", ""]
    return "\n".join(lines)


def layout_digest(name: str, size: int, align: int, members: dict) -> str:
    """Hash the compiler layout payload shared by probes and annotations."""
    canonical = json.dumps({"name": name, "bytes": size, "align": align, "members": members},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def probe_layouts(extra_headers=(), extra_types=None, compiler="g++", *, compile_flags=()) -> dict:
    """Compile and execute one isolated layout probe, returning per-type digests.

    Canonical digest payload is UTF-8 compact sorted JSON with type name, bytes,
    align and name-keyed members containing offset/bytes. No pointer value or
    padding byte participates. Results are intentionally not cached across builds.
    """
    source = layout_probe_source(extra_headers, extra_types)
    executable_name = Path(compiler).name
    cxx = "++" in executable_name or executable_name in ("c++", "nvcc")
    with tempfile.TemporaryDirectory(prefix="plk-host-layout-") as directory:
        directory = Path(directory)
        tu = directory / ("probe.cpp" if cxx else "probe.c")
        tu.write_text(source, encoding="utf-8")
        executable = directory / "probe"
        command = [str(compiler), "-std=c++17" if cxx else "-std=c11", *compile_flags, str(tu), "-o", str(executable)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        report = subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout
    layouts = {}
    neutral = {}
    for row in report.splitlines():
        parts = row.split("\t")
        if parts[0] == "I" and len(parts) == 4:
            _, name, type_id, fields = parts
            neutral[name] = (int(type_id), [field.strip() for field in fields.split(",")])
        elif parts[0] == "T" and len(parts) == 4:
            _, name, size, align = parts
            layouts[name] = {"bytes": int(size), "align": int(align), "members": {}}
        elif parts[0] == "F" and len(parts) == 5:
            _, name, field, offset, size = parts
            layouts[name]["members"][field] = {"offset": int(offset), "bytes": int(size)}
        else:
            raise ValueError(f"invalid compiler layout report row: {row!r}")
    expected = _inventory(extra_types)
    if set(neutral) != set(ABI_TYPES):
        raise ValueError("installed neutral ABI type inventory differs from ctypes")
    for name, (type_id, fields) in neutral.items():
        if set(fields) != set(expected[name]):
            raise ValueError(f"installed neutral ABI field inventory differs from ctypes: {name}")
        layouts[name]["id"] = type_id
    if set(layouts) != set(expected):
        raise ValueError("compiler layout type inventory mismatch")
    for name, layout in layouts.items():
        if set(layout["members"]) != set(expected[name]):
            raise ValueError(f"compiler layout field inventory mismatch: {name}")
        layout["layout_digest"] = layout_digest(name, layout["bytes"], layout["align"], layout["members"])
        layout["fields"] = sorted(layout["members"])
    return layouts
