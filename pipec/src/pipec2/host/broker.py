"""Native owner-provider broker for host ABI v3: open / launch* / close.

``open_host`` loads every provider (RTLD_LOCAL), proves its ABI, resolves shared
scalars, queries demands, selects geometry, allocates one workspace, assembles
the params image and runs every ``build`` once; the returned ``HostSession`` is
pinned to those addresses, scalars, cluster and geometry. ``HostSession.launch``
issues only stream work on the caller's stream (per-launch zero fills, kernel
launch) and is CUDA-graph capture safe. ``close`` drains the context, releases
every attempted build in reverse, frees the workspace and unloads the providers.
"""
from __future__ import annotations

import ctypes as C
import math
import struct
from _ctypes import dlclose
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from functools import lru_cache
from ..model import ClauseFailure, LinkError, ParamsEntry, ProfileError
from . import abi as A
from .profile import HOST_ABI, TOOLCHAIN_PROVENANCE, dtypes_digest, validate_host_abi
from .fixture import Domain, Resource

_ACCESS = {"read": 1, "write": 2, "read_write": 3}
_SCALARS = {"u64": 1, "i64": 2, "f64": 3, "bool": 4}


def _fail(reason: str, path: str, message: str, *, abi: bool = False, **details: Any) -> LinkError:
    return LinkError([ClauseFailure("L-HOSTABI" if abi else "L-HOSTCHAN", path, message,
                                   {"subreason": reason, **details})])


def _write_stage_blobs(image: A.aligned_buffer, entries: Sequence[ParamsEntry],
                       blobs: Mapping[str, bytes], fail: Callable[..., LinkError]) -> None:
    """Copy opaque stage blobs into the common by-value kernel params image."""
    for name, offset, size, _align in entries:
        if name in blobs:
            data = blobs[name]
            if not isinstance(data, bytes) or len(data) != size:
                raise fail(name, "blob size differs from published params entry",
                           expected=size, actual=len(data) if isinstance(data, bytes) else type(data).__name__)
            image.write(data, offset)


def _digest(text: str) -> Any:
    value = A.plk_digest_v2()
    value.sha256[:] = bytes.fromhex(text)
    return value


def _hex(value: Any) -> str:
    return bytes(value.sha256).hex()


def _header(record: Any) -> None:
    record.header.abi_version = 2
    record.header.struct_bytes = C.sizeof(type(record))


def _pointer(value: Any) -> int:
    return C.cast(value, C.c_void_p).value or 0


def _dims(value: Any) -> tuple[int, int, int]:
    return value.x, value.y, value.z


def _geometry(value: Any) -> tuple[Any, ...]:
    return (value.geometry_present, value.launch_required, _dims(value.grid),
            _dims(value.block), _dims(value.cluster), value.dynamic_smem_bytes, value.reserved)


def _integer(value: Any, bits: int, *, signed: bool = False, path: str = "runtime") -> int:
    lower = -(1 << (bits - 1)) if signed else 0
    upper = (1 << (bits - (1 if signed else 0))) - 1
    if type(value) is not int or not lower <= value <= upper:
        raise _fail("resource_range", path, f"expected {'i' if signed else 'u'}{bits}", actual=value)
    return value


def _scalar_bits(value: Any, kind: str, path: str) -> int:
    if kind == "bool":
        if type(value) is not bool:
            raise _fail("scalar", path, "boolean scalar requires a boolean value", actual=value)
        return int(value)
    if kind == "f64":
        if isinstance(value, dict) and set(value) == {"bits"}:
            return _integer(value["bits"], 64, path=path)
        if type(value) not in (int, float):
            raise _fail("scalar", path, "f64 scalar requires numeric value or typed IEEE bits", actual=value)
        try:
            return struct.unpack("<Q", struct.pack("<d", value))[0]
        except (OverflowError, struct.error) as exc:
            raise _fail("scalar", path, "f64 scalar is outside binary64 range", actual=value) from exc
    return _integer(value, 64, signed=kind == "i64", path=path) & ((1 << 64) - 1)


def _scalar_image(fields: list[dict[str, Any]], values: Mapping[str, Any]) -> Any:
    size = C.sizeof(A.plk_shared_scalars_v2) + len(fields) * C.sizeof(A.plk_scalar_v2)
    image = A.aligned_buffer(size, 8)
    header = A.plk_shared_scalars_v2.from_address(image.address)
    _header(header)
    header.image_bytes = size
    header.record_count = len(fields)
    header.record_offset = C.sizeof(A.plk_shared_scalars_v2)
    records = (A.plk_scalar_v2 * len(fields)).from_address(image.address + header.record_offset)
    for output, item in zip(records, sorted(fields, key=lambda v: v["id"]), strict=True):
        output.scalar_id = item["id"]
        output.scalar_type = _SCALARS[item["type"]]
        output.value_bits = _scalar_bits(values[item["name"]], item["type"], item["name"])
    return image


@dataclass
class _Owner:
    group: dict[str, Any]
    library: Any
    functions: dict[str, Any]
    args: Any
    common: Any = None
    demands: Any = None
    resources: Any = None
    reports: Any = None
    session: Any = field(default_factory=C.c_void_p)
    keep: list[Any] = field(default_factory=list)
    channels: dict[str, Resource] = field(default_factory=dict)
    workspace_demands: dict[str, tuple[int, int]] = field(default_factory=dict)
    channel_demands: dict[str, Any] = field(default_factory=dict)
    domain_ids: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.group["owner"]


def _load_owner(group: dict[str, Any], path: Path, args: bytes) -> _Owner:
    provider = group["provider"]
    contract = provider["args"]
    if len(args) != contract["bytes"]:
        raise _fail("inventory", group["owner"], "owner args size differs", abi=True,
                    expected=contract["bytes"], actual=len(args))
    buffer = A.aligned_buffer(len(args), contract["align"])
    if args:
        C.memmove(buffer.address, args, len(args))
    try:
        library = C.CDLL(str(path.resolve()), mode=C.RTLD_LOCAL)
    except OSError as exc:
        raise _fail("inventory", group["owner"], f"cannot load provider: {exc}", abi=True) from exc
    signatures = {
        "abi": [C.POINTER(A.plk_abi_report_v2)],
        "query": [C.POINTER(A.plk_common_v2), C.c_void_p, C.POINTER(A.plk_demands_v2)],
        "build": [C.POINTER(A.plk_common_v2), C.c_void_p, C.POINTER(A.plk_resources_v2),
                  C.c_void_p, C.c_size_t, C.POINTER(A.plk_transfer_report_v2),
                  C.c_uint32, C.POINTER(C.c_void_p)],
        "release": [C.c_void_p],
    }
    functions = {}
    for name, types in signatures.items():
        symbol = provider[name]
        try:
            function = getattr(library, symbol)
        except AttributeError as exc:
            dlclose(library._handle)
            raise _fail("inventory", group["owner"], "missing provider export", abi=True,
                        expected=symbol) from exc
        function.argtypes = types
        function.restype = C.c_int
        functions[name] = function
    return _Owner(group, library, functions, buffer)


def _provider_status(owner: _Owner, phase: str, status: int) -> None:
    if status:
        raise RuntimeError(f"provider {owner.name} {phase} failed with status {status}")


def _report(owner: _Owner, target: dict[str, str]) -> None:
    inventory = sorted(owner.group["provider"]["abi_types"], key=lambda item: item["id"])
    entries = (A.plk_layout_entry_v2 * len(inventory))()
    for entry, declared in zip(entries, inventory, strict=True):
        entry.abi_type_id = declared["id"]
    report = A.plk_abi_report_v2()
    _header(report)
    report.entry_capacity = len(entries)
    report.entries = entries
    snapshot = (report.header.abi_version, report.header.struct_bytes,
                report.entry_capacity, _pointer(report.entries))
    _provider_status(owner, "abi", owner.functions["abi"](C.byref(report)))
    actual = (report.header.abi_version, report.header.struct_bytes,
              report.entry_capacity, _pointer(report.entries))
    if actual != snapshot or report.entry_count != len(entries):
        raise _fail("inventory", owner.name, "ABI report altered envelope or type count", abi=True,
                    expected=snapshot, actual=actual, count=report.entry_count)
    if _hex(report.profile.dtypes) != target["dtypes"]:
        raise _fail("layout", owner.name, "ABI public contract digest differs", abi=True,
                    contract="dtypes", expected=target["dtypes"], actual=_hex(report.profile.dtypes))
    if _hex(report.owner_contract) != owner.group["provider"]["owner_contract_digest"]:
        raise _fail("layout", owner.name, "ABI owner contract digest differs", abi=True,
                    expected=owner.group["provider"]["owner_contract_digest"],
                    actual=_hex(report.owner_contract))
    for entry, declared in zip(entries, inventory, strict=True):
        expected = (declared["id"], declared["bytes"], declared["align"], declared["layout_digest"], 0)
        actual = (entry.abi_type_id, entry.bytes, entry.alignment_bytes, _hex(entry.compiler_layout), entry.reserved)
        if actual != expected:
            raise _fail("layout", f"{owner.name}:{declared['name']}", "compiled ABI layout differs", abi=True,
                        expected=expected, actual=actual)


@lru_cache(maxsize=1)
def _neutral_layouts() -> dict[str, Any]:
    return A.probe_layouts()


def _validate_publication(result: Any) -> None:
    host = result.host
    try:
        validate_host_abi()
        target = {"dtypes": dtypes_digest(host["profile"])}
    except (ValueError, KeyError, TypeError, ProfileError) as exc:
        raise _fail("layout", "link.host", str(exc), abi=True) from exc
    if host["host_abi"] != HOST_ABI or host["target"] != target:
        raise _fail("layout", "link.host.target", "unsupported host ABI/contracts", abi=True)
    total, align = host["params"]["bytes"], host["params"]["align"]
    if (type(total) is not int or not 0 <= total <= 32764 or type(align) is not int or
            align <= 0 or align & (align - 1) or total % align or
            total != result.params.total_bytes or align != result.params.align):
        raise _fail("resource_range", "link.host.params", "invalid aggregate parameter extent/alignment")
    occupied = []
    shared = None
    for group in host["groups"]:
        if group["target"] != target:
            raise _fail("layout", group["owner"], "owner dtype contract differs", abi=True)
        regions = [] if group["residue"] is None else [group["residue"]]
        scalar = group["scalars"]
        if scalar is not None:
            image = (scalar["offset"], scalar["fields"])
            if shared is not None and shared != image:
                raise _fail("inventory", group["owner"], "owners disagree on the single shared scalar image")
            if shared is None:
                shared = image
                size = C.sizeof(A.plk_shared_scalars_v2) + len(scalar["fields"]) * C.sizeof(A.plk_scalar_v2)
                regions.append({"name": "host.scalars", "offset": scalar["offset"], "bytes": size, "align": 8})
        for region in regions:
            offset, size, alignment = region["offset"], region["bytes"], region["align"]
            if (any(type(v) is not int for v in (offset, size, alignment)) or
                    offset < 0 or size < 0 or alignment <= 0 or alignment & (alignment - 1) or
                    alignment > align or offset % alignment or offset + size > total):
                raise _fail("resource_range", region["name"], "published region exceeds or misaligns aggregate")
            if size:
                occupied.append((offset, offset + size, region["name"]))
        declared = {row["name"]: row for row in group["provider"]["abi_types"]}
        for name, native in _neutral_layouts().items():
            row = declared.get(name)
            if row is None or any(row.get(key) != native[key] for key in ("id", "bytes", "align", "layout_digest", "fields")):
                raise _fail("layout", f"{group['owner']}:{name}", "declared neutral ABI differs from launcher compiler probe",
                            abi=True)
    entries = {name: (offset, size, alignment) for name, offset, size, alignment in result.params.entries}
    for blob in host["params"].get("blobs", []):
        name = blob["stage_id"]
        expected = (blob["offset"], blob["bytes"], blob["align"])
        if entries.get(name) != expected:
            raise _fail("inventory", name, "blob publication differs from params entry",
                        expected=expected, actual=entries.get(name))
        offset, size, alignment = expected
        if (offset < 0 or size < 0 or alignment <= 0 or alignment & (alignment - 1) or
                alignment > align or offset % alignment or offset + size > total):
            raise _fail("resource_range", name, "blob region exceeds or misaligns aggregate")
        if size:
            occupied.append((offset, offset + size, name))
    occupied.sort()
    for prior, current in zip(occupied, occupied[1:]):
        if prior[1] > current[0]:
            raise _fail("resource_range", current[2], "published host regions overlap", other=prior[2])


def _value(source: dict[str, Any], scalars: Mapping[str, Any], bindings: Mapping[str, Any],
           facts: Mapping[str, Any], hardware: Mapping[str, Any]) -> Any:
    if source["kind"] == "constant":
        return source["value"]
    if source["kind"] == "tuple":
        return [_value(item, scalars, bindings, facts, hardware) for item in source["items"]]
    maps = {"caller": scalars, "binding": bindings, "fact": facts, "hardware": hardware}
    try:
        value = maps[source["kind"]][source["name"]]
    except KeyError as exc:
        raise _fail("scalar", source.get("name", "source"), "missing declared scalar source") from exc
    if source["kind"] == "binding" and isinstance(value, (list, tuple)):
        value = value[0]
    if source["kind"] == "fact" and isinstance(value, dict) and "value" in value:
        value = value["value"]
    return value


def _evaluate(expr: Any, values: dict[str, Any], host: dict[str, Any]) -> int:
    if type(expr) is int:
        return expr
    for key, source in (("scalar", values), ("fact", host.get("facts", {})), ("binding", host.get("bindings", {}))):
        if key in expr:
            try:
                value = source[expr[key]]
            except KeyError as exc:
                raise _fail("alias", expr[key], "missing alias expression input") from exc
            if key == "binding" and isinstance(value, (tuple, list)):
                value = value[0]
            return value
    left, right = (_evaluate(arg, values, host) for arg in expr["args"])
    return left + right if expr["op"] == "add" else left * right


def _channel_resource(channel: dict[str, Any], supplied: Mapping[str, Resource],
                      domains: Mapping[str, tuple[int, int]]) -> Resource:
    if channel["source"]["kind"] == "workspace":
        rank = channel["rank"]
        return Resource(0, 0, (0,) * rank, (0,) * rank, channel["dtype_id"], 1)
    name = channel["source"]["name"]
    if name not in supplied:
        raise _fail("inventory", channel["name"], "missing caller resource", expected=name)
    resource = supplied[name]
    if resource.dtype_id != channel["dtype_id"]:
        raise _fail("dtype", channel["name"], "caller dtype differs",
                    expected=channel["dtype_id"], actual=resource.dtype_id)
    _check_resource(channel, resource, domains)
    return resource


def _check_resource(channel: dict[str, Any], resource: Resource, domains: Mapping[str, tuple[int, int]]) -> None:
    from .profile import DTYPES
    path = channel["name"]
    for key, value in (("address", resource.address), ("allocation_bytes", resource.allocation_bytes),
                       ("origin_byte_offset", resource.origin_byte_offset)):
        _integer(value, 64, path=f"{path}.{key}")
    if resource.alignment <= 0 or resource.alignment & (resource.alignment - 1):
        raise _fail("resource_range", path, "resource alignment is not a power of two")
    if not 0 <= resource.origin_bit_offset < 8:
        raise _fail("resource_range", path, "origin bit offset is outside 0..7")
    if len(resource.extents) != channel["rank"] or len(resource.bit_strides) != channel["rank"]:
        raise _fail("extent", path, "resource rank differs")
    if resource.allocation_bytes and not resource.address:
        raise _fail("resource_range", path, "nonempty resource has null address")
    if resource.address % resource.alignment:
        raise _fail("resource_range", path, "resource address violates alignment")
    low = high = resource.origin_byte_offset * 8 + resource.origin_bit_offset
    empty = False
    for extent, stride in zip(resource.extents, resource.bit_strides, strict=True):
        _integer(extent, 64, path=path)
        _integer(stride, 64, signed=True, path=path)
        empty |= extent == 0
        delta = max(0, extent - 1) * stride
        low += min(0, delta)
        high += max(0, delta)
    dtype = DTYPES[channel["storage_dtype"]]
    width = dtype["bits"]
    if not empty and (low < 0 or high + width > resource.allocation_bytes * 8):
        raise _fail("resource_range", path, "addressed bit interval exceeds allocation",
                    interval=(low, high + width), allocation_bits=resource.allocation_bytes * 8)
    if resource.address + resource.allocation_bytes > (1 << 64):
        raise _fail("resource_range", path, "allocation address overflows u64")
    if channel["domain"] is None:
        if resource.domain is not None or resource.peers or resource.multicast:
            raise _fail("domain", path, "domain-less channel received a symmetric resource", actual=resource.domain)
        return
    if resource.domain != channel["domain"]:
        raise _fail("domain", path, "resource is symmetric over a different domain",
                    expected=channel["domain"], actual=resource.domain)
    rank, world = domains[channel["domain"]]
    if resource.peers or "peer" in channel["views"]:
        if len(resource.peers) != world:
            raise _fail("domain", path, "peer table length differs from the domain world",
                        expected=world, actual=len(resource.peers))
        if resource.peers[rank] != resource.address:
            raise _fail("domain", path, "peer table entry of this rank differs from the local address",
                        expected=resource.address, actual=resource.peers[rank])
    if "multicast" in channel["views"] and resource.allocation_bytes and not resource.multicast:
        raise _fail("domain", path, "multicast view requires a multicast address")
    # Symmetric allocations: the bit-interval check above bounds every peer and the multicast view.
    for key, base in (*((f"peers[{i}]", peer) for i, peer in enumerate(resource.peers)),
                      *((("multicast", resource.multicast),) if resource.multicast else ())):
        _integer(base, 64, path=f"{path}.{key}")
        if resource.allocation_bytes and not base:
            raise _fail("resource_range", f"{path}.{key}", "nonempty symmetric allocation has null address")
        if base % resource.alignment:
            raise _fail("resource_range", f"{path}.{key}", "symmetric address violates alignment")
        if base + resource.allocation_bytes > (1 << 64):
            raise _fail("resource_range", f"{path}.{key}", "symmetric allocation address overflows u64")


def _bases(resource: Resource) -> tuple[int, ...]:
    """Every address of one resource's allocation base: local, then peers, then multicast."""
    return (resource.address, *resource.peers, *((resource.multicast,) if resource.multicast else ()))


def _check_domains(groups: list[dict[str, Any]], domains: Mapping[str, tuple[int, int]]) -> None:
    needed = {domain["name"] for group in groups for domain in group["domains"]}
    if needed - domains.keys():
        raise _fail("inventory", "domains", "caller omitted declared communication domains",
                    expected=sorted(needed), actual=sorted(domains))
    for name in sorted(needed):
        rank, world = domains[name]
        if type(rank) is not int or type(world) is not int or not 0 <= rank < world < 2**32:
            raise _fail("domain", f"domains.{name}", "domain requires 0 <= rank < world < 2^32",
                        rank=rank, world=world)


def _prepare_query(owner: _Owner, shared: Any, resources: Mapping[str, Resource],
                   domains: Mapping[str, tuple[int, int]], block: tuple[int, int, int],
                   cluster: tuple[int, int, int], smem: int) -> None:
    common = A.plk_common_v2()
    _header(common)
    common.owner_id = owner.group["owner_id"]
    common.shared = C.cast(shared.address, C.POINTER(A.plk_shared_scalars_v2))
    common.block = A.plk_dim3_v2(*block)
    common.cluster = A.plk_dim3_v2(*cluster)
    common.dynamic_smem_bytes = smem
    common.profile.dtypes = _digest(owner.group["target"]["dtypes"])
    common.owner_contract = _digest(owner.group["provider"]["owner_contract_digest"])
    declared = sorted(owner.group["domains"], key=lambda d: d["id"])
    table = (A.plk_domain_v2 * len(declared))(*(A.plk_domain_v2(d["id"], *domains[d["name"]]) for d in declared))
    common.domain_count = len(table)
    common.domains = table
    owner.domain_ids = {d["name"]: d["id"] for d in declared}
    owner.keep.append(table)
    channels = sorted(owner.group["channels"], key=lambda c: c["id"])
    metadata = (A.plk_tensor_meta_v2 * len(channels))()
    demands = (A.plk_channel_demand_v2 * len(channels))()
    for meta, demand, declaration in zip(metadata, demands, channels, strict=True):
        resource = _channel_resource(declaration, resources, domains)
        owner.channels[declaration["name"]] = resource
        rank = declaration["rank"]
        extents = (C.c_uint64 * rank)(*resource.extents)
        strides = (C.c_int64 * rank)(*resource.bit_strides)
        output_extents = (C.c_uint64 * rank)()
        output_strides = (C.c_int64 * rank)()
        owner.keep.extend((extents, strides, output_extents, output_strides))
        meta.channel_id = demand.channel_id = declaration["id"]
        meta.rank = demand.rank = rank
        meta.dtype_id = resource.dtype_id
        meta.access = _ACCESS[declaration["access"]]
        meta.allocation_bytes = resource.allocation_bytes
        meta.alignment_bytes = resource.alignment
        meta.origin_byte_offset = resource.origin_byte_offset
        meta.origin_bit_offset = resource.origin_bit_offset
        meta.extents = extents
        meta.bit_strides = strides
        demand.extents = output_extents
        demand.bit_strides = output_strides
    common.channel_count = len(metadata)
    common.channels = metadata
    workspaces = sorted(owner.group["workspaces"], key=lambda w: w["id"])
    workspace_demands = (A.plk_workspace_demand_v2 * len(workspaces))()
    for demand, declaration in zip(workspace_demands, workspaces, strict=True):
        demand.workspace_id = declaration["id"]
    out = A.plk_demands_v2()
    _header(out)
    out.channel_capacity = len(demands)
    out.channels = demands
    out.workspace_capacity = len(workspace_demands)
    out.workspaces = workspace_demands
    owner.common, owner.demands = common, out
    owner.keep.extend((metadata, demands, workspace_demands))


def _query(owner: _Owner) -> None:
    out = owner.demands
    immutable = (out.header.abi_version, out.header.struct_bytes, out.channel_capacity,
                 _pointer(out.channels), out.workspace_capacity, _pointer(out.workspaces))
    channels = [(c.channel_id, c.rank, _pointer(c.extents), _pointer(c.bit_strides))
                for c in out.channels[:out.channel_capacity]]
    workspace_ids = [w.workspace_id for w in out.workspaces[:out.workspace_capacity]]
    common_bytes = C.string_at(C.addressof(owner.common), C.sizeof(owner.common))
    _provider_status(owner, "query", owner.functions["query"](
        C.byref(owner.common), owner.args.address if owner.group["provider"]["args"]["bytes"] else None,
        C.byref(out)))
    actual = (out.header.abi_version, out.header.struct_bytes, out.channel_capacity,
              _pointer(out.channels), out.workspace_capacity, _pointer(out.workspaces))
    if actual != immutable or out.channel_count != out.channel_capacity or out.workspace_count != out.workspace_capacity:
        raise _fail("inventory", owner.name, "query changed envelope or exact inventory", expected=immutable,
                    actual=actual, counts=(out.channel_count, out.workspace_count))
    if common_bytes != C.string_at(C.addressof(owner.common), C.sizeof(owner.common)):
        raise _fail("inventory", owner.name, "query changed immutable common")
    for before, demand, declaration in zip(channels, out.channels[:out.channel_count],
                                          sorted(owner.group["channels"], key=lambda c: c["id"]), strict=True):
        after = (demand.channel_id, demand.rank, _pointer(demand.extents), _pointer(demand.bit_strides))
        if before != after:
            raise _fail("inventory", declaration["name"], "query changed channel ID/rank/pointers",
                        expected=before, actual=after)
        path = declaration["name"]
        if demand.dtype_id != declaration["dtype_id"] or demand.access != _ACCESS[declaration["access"]]:
            raise _fail("dtype", path, "query changed declared dtype/access",
                        expected=(declaration["dtype_id"], _ACCESS[declaration["access"]]),
                        actual=(demand.dtype_id, demand.access))
        alignment = demand.alignment_bytes
        if alignment <= 0 or alignment & (alignment - 1):
            raise _fail("resource_range", path, "query alignment is not a power of two", actual=alignment)
        owner.channel_demands[path] = demand
        if declaration["source"]["kind"] == "caller":
            _check_demand(declaration, owner.channels[path], demand)
    for before, demand, declaration in zip(workspace_ids, out.workspaces[:out.workspace_count],
                                           sorted(owner.group["workspaces"], key=lambda w: w["id"]), strict=True):
        if demand.workspace_id != before or demand.reserved:
            raise _fail("inventory", declaration["name"], "query changed workspace inventory")
        if demand.alignment_bytes != declaration["align"]:
            raise _fail("resource_range", declaration["name"], "query workspace alignment differs",
                        expected=declaration["align"], actual=demand.alignment_bytes)
        owner.workspace_demands[declaration["name"]] = (demand.bytes, demand.alignment_bytes)


def _check_demand(channel: dict[str, Any], resource: Resource, demand: Any) -> None:
    expected = (tuple(demand.extents[:demand.rank]), tuple(demand.bit_strides[:demand.rank]))
    actual = (resource.extents, resource.bit_strides)
    if actual != expected:
        raise _fail("extent", channel["name"], "caller metadata differs from query demand",
                    expected=expected, actual=actual)
    if (resource.allocation_bytes < demand.minimum_allocation_bytes or
            resource.alignment < demand.alignment_bytes or
            any((base + resource.origin_byte_offset) % demand.alignment_bytes for base in _bases(resource))):
        raise _fail("resource_range", channel["name"], "resource does not satisfy query demand",
                    expected={"minimum_bytes": demand.minimum_allocation_bytes, "alignment": demand.alignment_bytes},
                    actual={"bytes": resource.allocation_bytes, "alignment": resource.alignment,
                            "origin_address": resource.address + resource.origin_byte_offset})


def _select_geometry(owners: list[_Owner], host: dict[str, Any], block: tuple[int, int, int],
                     cluster: tuple[int, int, int], smem: int, backend: Any) -> Any:
    contract = host["scheduler_contract"]
    authority = [o for o in owners if o.name == contract["owner"]]
    if len(authority) != 1:
        raise _fail("geometry", "host.scheduler_contract.owner", "one integration query authority is required")
    geometry = authority[0].demands.geometry
    if geometry.geometry_present != 1 or geometry.launch_required not in (0, 1) or geometry.reserved:
        raise _fail("geometry", authority[0].name, "integration query omitted/invalid geometry")
    if (_dims(geometry.block), _dims(geometry.cluster), geometry.dynamic_smem_bytes) != (block, cluster, smem):
        raise _fail("geometry", authority[0].name, "query violates solved static geometry",
                    expected=(block, cluster, smem), actual=_geometry(geometry))
    if not geometry.launch_required:
        if not contract["schema"]["allow_no_launch"]:
            raise _fail("geometry", authority[0].name, "strategy does not permit no-launch outcome")
        if _dims(geometry.grid) != (0, 0, 0):
            raise _fail("geometry", authority[0].name, "no-launch geometry must have explicit zero grid")
    else:
        limits = backend.observations
        for i, (grid, cl) in enumerate(zip(_dims(geometry.grid), cluster, strict=True)):
            if grid <= 0 or grid % cl or grid > limits.get("max_grid", (2147483647, 65535, 65535))[i]:
                raise _fail("geometry", authority[0].name, "grid violates CUDA/cluster limits", actual=_dims(geometry.grid))
        if math.prod(block) > limits.get("max_threads", 1024) or any(
                size > limit for size, limit in zip(block, limits.get("max_block", (1024, 1024, 64)), strict=True)):
            raise _fail("geometry", authority[0].name, "block violates device limits")
    for owner in owners:
        other = owner.demands.geometry
        if other.geometry_present not in (0, 1) or (other.geometry_present and _geometry(other) != _geometry(geometry)):
            raise _fail("geometry", owner.name, "owner geometry disagrees with integration query",
                        expected=_geometry(geometry), actual=_geometry(other))
    return A.plk_geometry_v2.from_buffer_copy(geometry)


def _aliases(owners: list[_Owner], values: dict[str, Any], host: dict[str, Any],
             all_resources: dict[str, Resource]) -> None:
    channels = {c["name"]: c for o in owners for c in o.group["channels"]}
    names = sorted(channels)
    for name in names:
        declaration = channels[name]
        alias = declaration["alias"]
        if alias is not None:
            target = alias["channel"]
            offset = (_evaluate(alias["byte_offset"], values, host) * 8 +
                      _evaluate(alias["bit_offset"], values, host))
            source, destination = all_resources[name], all_resources[target]
            # Aliases hold on every view of the allocation: local, each peer and multicast.
            source_bases, target_bases = _bases(source), _bases(destination)
            if len(source_bases) != len(target_bases):
                raise _fail("alias", name, "alias and target carry different symmetric views", other=target)
            for source_base, target_base in zip(source_bases, target_bases, strict=True):
                source_origin = (source_base + source.origin_byte_offset) * 8 + source.origin_bit_offset
                target_origin = (target_base + destination.origin_byte_offset) * 8 + destination.origin_bit_offset
                if source_origin != target_origin + offset:
                    raise _fail("alias", name, "declared alias origin differs", expected=target_origin + offset,
                                actual=source_origin)
    for i, name in enumerate(names):
        a = all_resources[name]
        for other in names[i + 1:]:
            b = all_resources[other]
            overlap = max(a.address, b.address) < min(a.address + a.allocation_bytes, b.address + b.allocation_bytes)
            declared = ((channels[name]["alias"] or {}).get("channel") == other or
                        (channels[other]["alias"] or {}).get("channel") == name)
            same_source = channels[name]["source"] == channels[other]["source"]
            if overlap and not declared and not same_source:
                raise _fail("alias", name, "undeclared overlapping caller/workspace allocations", other=other)


def _build_resources(owner: _Owner, geometry: Any, workspace_resources: dict[str, Resource], backend: Any) -> None:
    channels = sorted(owner.group["channels"], key=lambda c: c["id"])
    device = (A.plk_device_resource_v2 * len(channels))()
    for output, declaration in zip(device, channels, strict=True):
        resource = owner.channels[declaration["name"]]
        output.channel_id = declaration["id"]
        output.access = _ACCESS[declaration["access"]]
        output.allocation_address = resource.address
        output.allocation_bytes = resource.allocation_bytes
        output.origin_byte_offset = resource.origin_byte_offset
        output.origin_bit_offset = resource.origin_bit_offset
        if declaration["domain"] is not None:
            output.domain_id = owner.domain_ids[declaration["domain"]]
        if "peer" in declaration["views"]:
            table = (C.c_uint64 * len(resource.peers))(*resource.peers)
            output.peer_addresses = table
            owner.keep.append(table)
        if "multicast" in declaration["views"]:
            output.multicast_address = resource.multicast
    workspaces = sorted(owner.group["workspaces"], key=lambda w: w["id"])
    workspace = (A.plk_workspace_resource_v2 * len(workspaces))()
    for output, declaration in zip(workspace, workspaces, strict=True):
        resource = workspace_resources[declaration["name"]]
        output.workspace_id = declaration["id"]
        output.device_address = resource.address
        output.bytes = resource.allocation_bytes
        output.alignment_bytes = resource.alignment
    resources = A.plk_resources_v2()
    _header(resources)
    resources.channel_count = len(device)
    resources.workspace_count = len(workspace)
    resources.channels, resources.workspaces = device, workspace
    resources.shared = owner.common.shared
    resources.cuda_stream_token = backend.stream_token
    resources.geometry = geometry
    owner.resources = resources
    owner.reports = (A.plk_transfer_report_v2 * len(owner.group["transfers"]))()
    for report, transfer in zip(owner.reports, owner.group["transfers"], strict=True):
        report.transfer_id = transfer["id"]
    owner.keep.extend((device, workspace))


def _swizzle_mode(endpoint: dict[str, Any]) -> int | None:
    swizzle = endpoint["swizzle"]
    if swizzle is None or swizzle["B"] == 0:
        return 0
    unit = swizzle.get("unit") or "element"
    bits = {"byte": 8, "bit": 1, "element": endpoint["elem_bits"]}[unit]
    if bits <= 0 or bits & (bits - 1):
        return None
    byte_m = swizzle["M"] + bits.bit_length() - 1 - 3
    if byte_m != 4 or swizzle["S"] != 3 or swizzle["B"] not in (1, 2, 3):
        return None
    return swizzle["B"]


def _check_transfers(owner: _Owner, params: Any) -> list[dict[str, Any]]:
    channels = {channel["name"]: channel for channel in owner.group["channels"]}
    totals: dict[tuple[str, str], list[Any]] = {}
    records = []
    residue = owner.group["residue"]
    for report, transfer in zip(owner.reports, owner.group["transfers"], strict=True):
        name, endpoint = transfer["name"], transfer["endpoint"]
        channel = channels[transfer["channel"]]
        resource = owner.channels[channel["name"]]
        def equal(check: str, expected: Any, actual: Any) -> None:
            if expected != actual:
                raise _fail("transfer", name, "transfer report differs from bound endpoint",
                            check=check, expected=expected, actual=actual)
        equal("transfer_id", transfer["id"], report.transfer_id)
        equal("storage_dtype_id", channel["dtype_id"], report.storage_dtype_id)
        equal("access", _ACCESS[transfer["access"]], report.access)
        equal("endpoint_access", {"produce": A.PLK_ACCESS_READ, "consume": A.PLK_ACCESS_WRITE}[endpoint["access"]],
              report.access)
        equal("channel_access", report.access, report.access & _ACCESS[channel["access"]])
        ring = endpoint["ring"]
        equal("ring_depth", ring["depth"] if ring else 1, report.ring_depth)
        equal("ring_stage_stride_bytes", ring["stage_stride_bytes"] if ring else 0,
              report.ring_stage_stride_bytes)
        end = report.residue_offset + report.representation_bytes
        if end > residue["bytes"]:
            raise _fail("transfer", name, "representation exceeds owner residue", check="residue_bounds",
                        expected={"maximum_end": residue["bytes"]},
                        actual={"offset": report.residue_offset, "end": end})
        alignment = report.representation_align
        address = params.address + residue["offset"] + report.residue_offset
        if (alignment == 0 or alignment & (alignment - 1) or
                alignment > residue["align"] or address % alignment):
            raise _fail("transfer", name, "representation alignment is not honoured", check="representation_align",
                        expected={"maximum_align": residue["align"], "address_remainder": 0},
                        actual={"align": alignment, "address": address})
        span_end = report.source_origin_bits + report.source_span_bits
        if span_end > resource.allocation_bytes * 8:
            raise _fail("transfer", name, "transfer span exceeds channel allocation", check="source_span",
                        expected={"maximum_end_bits": resource.allocation_bytes * 8},
                        actual={"origin_bits": report.source_origin_bits, "end_bits": span_end})
        record = {key: getattr(report, key) for key, _ in A.plk_transfer_report_v2._fields_
                  if key != "vendor_detail"}
        record.update(owner=owner.name, name=name, channel=channel["name"],
                      device_endpoint=dict(transfer["device_endpoint"]))
        if report.vendor_kind == A.PLK_TRANSFER_TMA_TILED:
            if not report.vendor_detail:
                raise _fail("transfer", name, "tiled transfer omitted its vendor detail", check="vendor_detail",
                            expected="plk_tma_tiled_detail_v2", actual=None)
            detail = C.cast(report.vendor_detail, C.POINTER(A.plk_tma_tiled_detail_v2)).contents
            equal("swizzle", _swizzle_mode(endpoint), detail.swizzle)
            record["vendor_detail"] = {"swizzle": detail.swizzle, "rank": detail.rank,
                                       "box_elements": list(detail.box_elements),
                                       "element_strides": list(detail.element_strides),
                                       "reserved": detail.reserved}
        else:
            record["vendor_detail"] = report.vendor_detail
        key = (transfer["device_endpoint"]["stage"], transfer["device_endpoint"]["endpoint"])
        total = totals.setdefault(key, [transfer, 0, 0])
        total[1] += report.footprint_bytes_per_stage
        total[2] += report.credit_bytes_per_stage
        records.append(record)
    for transfer, footprint, credit in totals.values():
        endpoint = transfer["endpoint"]
        for check, expected, actual in (
                ("footprint_bytes_per_stage", endpoint["tile_bytes"], footprint),
                ("credit_bytes_per_stage", transfer["expect_tx_bytes"], credit)):
            if expected != actual:
                raise _fail("transfer", transfer["name"], "endpoint transfer totals differ", check=check,
                            expected=expected, actual=actual)
        if endpoint["ring"] and footprint > endpoint["ring"]["stage_stride_bytes"]:
            raise _fail("transfer", transfer["name"], "endpoint footprint exceeds ring stage stride",
                        check="ring_footprint", expected=endpoint["ring"]["stage_stride_bytes"], actual=footprint)
    for record in records:
        record["status"] = "PASS"
    return records


@dataclass
class HostSession:
    """An immutable invocation: geometry, provenance, reports, params, launch and close."""
    backend: Any
    geometry: dict[str, Any] = field(default_factory=dict)
    toolchain_provenance: dict[str, Any] = field(default_factory=dict)
    transfer_reports: list[dict[str, Any]] = field(default_factory=list)
    params: Any = None
    owners: list[_Owner] = field(default_factory=list)
    attempted: list[_Owner] = field(default_factory=list)
    allocations: list[int] = field(default_factory=list)
    resets: list[tuple[int, int]] = field(default_factory=list)
    keep: list[Any] = field(default_factory=list)
    closed: bool = False
    _geometry: Any = None

    def launch(self, stream: int, image: Any = None) -> None:
        """Enqueue launch resets and the kernel on `stream`.

        `image` is an optional second backend on the same context holding a same-source control
        image of this link (same symbol, threads, shared memory and params layout); it launches
        with this session's params, geometry and resets.
        """
        for address, size in self.resets:
            self.backend.zero(address, size, stream)
        if self._geometry.launch_required:
            (image or self.backend).launch(self.params, self._geometry, stream)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        errors = _teardown(self)
        if errors:
            raise ExceptionGroup("host release/cleanup failed", errors)

    def __enter__(self) -> HostSession:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _teardown(session: HostSession) -> list[Exception]:
    """The only release path: drain (or abandon), reverse release, free, dlclose."""
    backend = session.backend
    errors: list[Exception] = []
    if session.attempted or session.allocations:
        try:
            backend.synchronize()
        except Exception as exc:
            errors.append(exc)
            backend.abandon()
    if session.attempted:
        try:
            with backend.current():
                for owner in reversed(session.attempted):
                    try:
                        _provider_status(owner, "release", owner.functions["release"](owner.session))
                    except Exception as exc:
                        errors.append(exc)
        except Exception as exc:
            errors.append(exc)
    if not backend.lost:
        for address in reversed(session.allocations):
            try:
                backend.free(address)
            except Exception as exc:
                errors.append(exc)
    # Owners, their CDLL handles, metadata, args, scalar image and the
    # aggregate remain reachable until all release calls above have returned.
    for owner in reversed(session.owners):
        try:
            dlclose(owner.library._handle)
        except Exception as exc:
            errors.append(exc)
    return errors


def open_host(result: Any, provider_paths: Mapping[str, Path], provider_args: Mapping[str, bytes],
              resources: Mapping[str, Resource], scalars: Mapping[str, Any], backend: Any,
              cluster: tuple[int, int, int] | None = None, *,
              blobs: Mapping[str, bytes] | None = None,
              domains: Mapping[str, Domain] | None = None) -> HostSession:
    """Report/query/build once, check transfer reports, and return the pinned session.

    Providers and args are keyed by params_group; blobs by host-less stage ID.
    ``domains`` maps every communication domain an owner declared to this rank's
    ``Domain(rank, world)``. Caller resources must be ready on the stream later passed
    to launch(); symmetric resources on every rank of their domain. Providers run with
    ``backend.current()`` pushed.
    The session owns intermediate buffers and workspaces through reverse release.
    """
    host = result.host
    if not host or host["protocol"] != 2:
        raise _fail("inventory", "link.host", "a host ABI v3 link is required")
    _validate_publication(result)
    groups = host["groups"]
    expected = {group["params_group"] for group in groups}
    if set(provider_paths) != expected or not set(provider_args) <= expected:
        raise _fail("inventory", "providers", "provider maps must exactly resolve params groups", abi=True,
                    expected=sorted(expected), actual=sorted(provider_paths), args=sorted(provider_args))
    blobs = {} if blobs is None else blobs
    blob_names = {item["stage_id"] for item in host["params"].get("blobs", []) if item["bytes"]}
    if set(blobs) != blob_names:
        raise _fail("inventory", "blobs", "blobs must exactly cover host-less nonzero params stages",
                    expected=sorted(blob_names), actual=sorted(blobs))
    domains = {} if domains is None else dict(domains)
    _check_domains(groups, domains)
    block = (result.threads, 1, 1)
    from ..launcher import _select_cluster
    cluster = _select_cluster(result.cluster, cluster)
    session = HostSession(backend)
    owners = session.owners
    values: dict[str, Any] = {}
    all_resources: dict[str, Resource] = {}
    workspaces: dict[str, Resource] = {}
    try:
        for group in groups:
            name = group["params_group"]
            owners.append(_load_owner(group, Path(provider_paths[name]), provider_args.get(name, b"")))
        # Providers (report/query/build) run in the backend's context, current on this thread.
        with backend.current():
            for owner in owners:
                _report(owner, host["target"])
            from .contracts import validate_scheduler_value
            contract = host["scheduler_contract"]
            schema = contract["schema"]
            leaves = {**schema["required"], **schema["optional"]}
            enum_sources = []
            for item in contract.get("inputs", []):
                value = _value(item["source"], scalars, host.get("bindings", {}),
                               host.get("facts", {}), backend.observations)
                leaf = leaves[item["name"]]
                validate_scheduler_value(value, leaf, "host.scheduler_contract.inputs." + item["name"])
                if leaf["kind"] == "enum":
                    enum_sources.append((item["source"], leaf["values"]))
            scalar = next((group["scalars"] for group in groups if group["scalars"] is not None), None)
            fields = [] if scalar is None else scalar["fields"]
            for item in fields:
                value = _value(item["source"], scalars, host.get("bindings", {}),
                               host.get("facts", {}), backend.observations)
                for source, names in enum_sources:
                    if item["source"] == source and isinstance(value, str):
                        value = names.index(value)
                values[item["name"]] = value
            shared = _scalar_image(fields, values)
            session.keep.append(shared)
            for owner in owners:
                _prepare_query(owner, shared, resources, domains, block, cluster, result.dyn_smem_bytes)
            for owner in owners:
                _query(owner)
            geometry = _select_geometry(owners, host, block, cluster, result.dyn_smem_bytes, backend)
            # Allocate once and slice by owner demand, including legal zero slices.
            cursor, workspace_align = 0, 1
            slices: dict[str, tuple[int, int, int]] = {}
            for owner in owners:
                for name, (size, align) in sorted(owner.workspace_demands.items()):
                    if name in slices:
                        raise _fail("inventory", name, "multiple workspace allocating owners")
                    cursor = (cursor + align - 1) & -align
                    slices[name] = (cursor, size, align)
                    cursor += size
                    _integer(cursor, 64, path="workspace.total_bytes")
                    workspace_align = max(workspace_align, align)
            base = backend.allocate(cursor, workspace_align)
            session.allocations.append(base)
            for name, (offset, size, align) in slices.items():
                workspaces[name] = Resource(base + offset if size else 0, size, (size,), (8,), 1, align)
            for owner in owners:
                for channel in owner.group["channels"]:
                    name = channel["name"]
                    if channel["source"]["kind"] == "workspace":
                        allocation = workspaces[channel["source"]["name"]]
                        demand = owner.channel_demands[name]
                        resource = Resource(allocation.address, allocation.allocation_bytes,
                                            tuple(demand.extents[:demand.rank]), tuple(demand.bit_strides[:demand.rank]),
                                            demand.dtype_id, allocation.alignment)
                        _check_resource(channel, resource, domains)
                        _check_demand(channel, resource, demand)
                        owner.channels[name] = resource
                    resource = owner.channels[name]
                    if name in all_resources and all_resources[name] != resource:
                        raise _fail("resource_range", name, "owners demand different shared resources")
                    all_resources[name] = resource
            _aliases(owners, values, host, all_resources)
            params = session.params = A.aligned_buffer(host["params"]["bytes"], host["params"]["align"])
            if scalar is not None:
                C.memmove(params.address + scalar["offset"], shared.address, shared.size)
            _write_stage_blobs(params, result.params.entries, blobs,
                               lambda name, message, **detail: _fail("resource_range", name, message, **detail))
            # Open-time zero fills precede build. Per-launch resets stay on the launch stream.
            for owner in owners:
                for workspace in sorted(owner.group["workspaces"], key=lambda w: w["id"]):
                    resource = workspaces[workspace["name"]]
                    if workspace["initializer"]["kind"] == "zero":
                        backend.zero(resource.address, resource.allocation_bytes, backend.stream_token)
                    if workspace["reset"] == "launch":
                        session.resets.append((resource.address, resource.allocation_bytes))
            for owner in owners:
                _build_resources(owner, geometry, workspaces, backend)
                residue = owner.group["residue"]
                residue_bytes = 0 if residue is None else residue["bytes"]
                session.attempted.append(owner)
                before_geometry = _geometry(owner.resources.geometry)
                status = owner.functions["build"](
                    C.byref(owner.common), owner.args.address if owner.group["provider"]["args"]["bytes"] else None,
                    C.byref(owner.resources), params.address + residue["offset"] if residue_bytes else None,
                    residue_bytes, owner.reports, len(owner.reports), C.byref(owner.session))
                _provider_status(owner, "build", status)
                if _geometry(owner.resources.geometry) != before_geometry:
                    raise _fail("geometry", owner.name, "build attempted to override query geometry")
                session.transfer_reports.extend(_check_transfers(owner, params))
            session._geometry = geometry
            session.geometry = {"grid": _dims(geometry.grid), "block": _dims(geometry.block),
                                "cluster": _dims(geometry.cluster), "dynamic_smem_bytes": geometry.dynamic_smem_bytes,
                                "launch_required": bool(geometry.launch_required)}
            session.toolchain_provenance = {**TOOLCHAIN_PROVENANCE, "observations": dict(backend.observations)}
            backend.synchronize()
    except BaseException as primary:
        errors = _teardown(session)
        if errors:
            raise BaseExceptionGroup("host invocation and cleanup failed", [primary, *errors]) from None
        raise
    return session
