"""Emit explicit TileLang roles whose borrowed resources come from packed contracts."""
from collections import Counter
from dataclasses import dataclass, replace
import json
from math import prod
from pathlib import Path

from tilelang import tvm

import pipelink_tilelang

from .annotate import AnnotationError, FactsHeader, Ledger, symbol_prefix
from .capture import CaptureError, fragment_rows
from .contract import Channel, TensorImportChannel, _import_capture
from .external import ExternalBarrier, ExternalStorage, ExternalThreadBase
from .declarations import ParamField, ParamsSpec
from .externalize import _local_barrier, _variables, _uses, externalize_role


@dataclass(frozen=True)
class RoleSpec:
    warp_range: tuple[int, int]
    entry_params: tuple[str, ...]


@dataclass(frozen=True)
class Export:
    name: str
    symbol: str
    signature: str
    cpp_body: str


@dataclass(frozen=True)
class HostParams:
    """Owner-authored host annotations and their public residue type."""
    header: str
    residue_type: str
    defines: dict[str, str]


@dataclass(frozen=True)
class ContractStage:
    stage_id: str
    version: str
    launch_register_cap: int
    params: ParamsSpec
    roles: dict[str, RoleSpec]
    channels: list[Channel | TensorImportChannel]
    exports: tuple[Export, ...] = ()
    program: object = None
    host: HostParams | None = None

    def __post_init__(self):
        # Keep provider endpoint names, qualifying only collisions. This runs
        # before program() builds TIR, so its externs and PL_CALL agree.
        endpoints = [endpoint.name for channel in self.channels
                     for endpoint in ((channel.produce_endpoint, channel.release_endpoint)
                                      if isinstance(channel, TensorImportChannel) else (channel.endpoint,))
                     if endpoint is not None]
        counts = Counter(endpoints)
        for channel in self.channels:
            if isinstance(channel, TensorImportChannel):
                for operation in ("produce", "release"):
                    endpoint = getattr(channel, operation + "_endpoint")
                    if endpoint is not None:
                        name = endpoint.name
                        if counts[name] > 1:
                            name += "_" + channel.symbol
                        setattr(channel, operation + "_name", name)

DEFAULT_SEED = 10390
# TMA descriptors belong to the owner's host ABI; imports get no implicit descriptor param.
DEFAULT_PASS_CONFIGS = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}


def contract_stage(stage_id, version, program, *, launch_register_cap=255, host=None):
    """Derive a single whole-kernel role and its params from an import program."""
    channels = []
    token = _import_capture.set(channels)
    try:
        original = program()
    finally:
        _import_capture.reset(token)
    if not channels:
        raise AnnotationError("tensor.imports", "contract_stage needs a tensor import")
    role = str(original.attrs["global_symbol"])
    threads = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and node.thread_binding is not None:
            if node.thread_binding.thread_tag == "threadIdx.x":
                threads.append(int(node.extent))
            elif node.thread_binding.thread_tag in ("threadIdx.y", "threadIdx.z") and int(node.extent) != 1:
                raise AnnotationError("roles.full_warps", "only one-dimensional CTA thread cohorts are supported")

    tvm.tirx.stmt_functor.post_order_visit(original.body, visit)
    if len(threads) != 1 or threads[0] % 32:
        raise AnnotationError("roles.full_warps", "one kernel with a whole-warp thread extent is required")
    scalar_types = {"float32": "float", "float64": "double", "float16": "uint16_t",
                    "bfloat16": "uint16_t", "bool": "bool",
                    **{f"{kind}{bits}": f"{kind}{bits}_t" for kind in ("int", "uint") for bits in (8, 16, 32, 64)}}
    member_types = {}
    for channel in channels:
        channel.role = role
        for member in channel.argument_decl.members:
            if member.name in channel.outputs:
                continue
            if member.name not in channel.arguments:
                raise AnnotationError(f"tensor.imports.{channel.name}.arguments", f"missing member {member.name}")
            name = channel.arguments[member.name]
            previous = member_types.setdefault(name, member.ctype)
            if previous != member.ctype:
                raise AnnotationError(f"tensor.imports.{channel.name}.arguments", f"inconsistent C types for {name}")
    fields = []
    names = {}
    for param in original.params:
        buffer = original.buffer_map.get(param)
        name = buffer.name if buffer is not None else param.name
        names[param] = name
        if buffer is not None:
            # PrimFunc buffer types carry no C++ const qualifier; read-only
            # inference is tracked apart from the compiler's final attrs.
            ctype = scalar_types[str(buffer.dtype)] + "*"
        elif str(param.dtype) == "handle":
            if name not in member_types:
                raise AnnotationError(f"params.{name}", "a raw handle needs a published argument type or a match_buffer")
            ctype = member_types[name]
        else:
            ctype = scalar_types[str(param.dtype)]
        if name in member_types and ctype != member_types[name]:
            raise AnnotationError(f"params.{name}", "produce-argument binding requires the same C type")
        fields.append(ParamField(name, ctype))
    for name, ctype in member_types.items():
        if name not in {field.name for field in fields}:
            fields.append(ParamField(name, ctype))
    used = _variables(original.body)
    entry_params = tuple(names[param] for param in original.params
                         if _uses(param, used) or (param in original.buffer_map
                            and _uses(original.buffer_map[param].data, used)))
    entry_params += tuple(channel.args_param for channel in channels)
    entry_params += tuple(channel._view_var.name for channel in channels if channel.view_call)

    def lower_program():
        # Rebuild after endpoint names are qualified, so the compiler buffers are new.
        function = program()
        used = _variables(function.body)
        params = [param for param in function.params if _uses(param, used) or
                  (param in function.buffer_map and _uses(function.buffer_map[param].data, used))]
        buffer_map = {param: buffer for param, buffer in function.buffer_map.items() if param in params}
        params.extend(channel._args_var for channel in channels)
        params.extend(channel._view_var for channel in channels if channel.view_call)
        lowered = tvm.tirx.PrimFunc(params, function.body, ret_type=function.ret_type,
                                  buffer_map=buffer_map, attrs=function.attrs)
        lowered = lowered.with_attr("pipelink.full_cta_import_role", True)
        remaps = {channel.symbol: list(channel._chunk_offsets) for channel in channels if channel.chunk}
        return (lowered.with_attr("pipelink.chunk_thread_offsets", remaps)
                if any(any(offsets) for offsets in remaps.values()) else lowered)

    return ContractStage(stage_id, version, launch_register_cap,
                         ParamsSpec("Params", tuple(fields), alignas=8),
                         {role: RoleSpec((0, threads[0] // 32), entry_params)}, channels,
                         program=lower_program, host=host)

def _tree(value):
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(_tree(item) for item in value) + "}"
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "nullptr"
    return str(value).lower() if isinstance(value, bool) else str(value)


def _pairs(value):
    return "{" + ", ".join("{" + _tree(key) + ", " + _tree(item) + "}" for key, item in value.items()) + "}"


def _layout(layout):
    if "lanes" in layout:
        return f'pl::tmem{{{layout["lanes"][0]}, {layout["lanes"][1]}, {layout["cols"]}, {layout["elem_bits"]}}}'
    result = f'pl::affine{{{_tree(layout["shape"])}, {_tree(layout["stride"])}, {layout["elem_bits"]}}}'
    if "atom" in layout:
        result += f'.atom({_tree(layout["atom"])})'
    if "swizzle" in layout:
        swizzle = layout["swizzle"]
        result += f'.swizzle({swizzle["B"]}, {swizzle["M"]}, {swizzle["S"]}, {_tree(swizzle.get("unit"))})'
    for field in ("origin_bits", "pointer_flag_bits"):
        if field in layout:
            result += f'.{field}({layout[field]})'
    return result


def _side(side):
    tx = side["tx_bytes"]
    tx = f'pl::sum{{{_tree(tx["sum"])}}}' if isinstance(tx, dict) else "pl::tile" if tx == "tile" else str(tx)
    if "member_of" in side:
        return f'pl::member{{{_tree(side["member_of"])}, {tx}}}'
    arrive = side["arrive"]
    result = f'pl::side{{{_tree(side["slots"])}, pl::arrive{{{_tree(arrive["by"])}, {_tree(arrive["count"])} }}, {tx}}}'
    if "native" in side:
        native = side["native"]
        result += f'.native(pl::barrier_storage{{{_tree(native["base"])}, {native["offset_bytes"]}, {native["stride_bytes"]}}})'
    return result


def _protocol(protocol):
    kind = protocol["kind"]
    if kind == "mbarrier":
        init = protocol["init"]
        initializer = f'pl::init{{{_tree(init.get("owner"))}, {_tree(init["phase"])}, {_tree(init["materializer"])}'
        if "warp" in init:
            initializer += f', {init["warp"]}'
        initializer += "}"
        empty = _side(protocol["empty"]) if "empty" in protocol else "std::nullopt"
        return f'pl::mbarrier{{{_side(protocol["full"])}, {empty}, {initializer}, {_tree(protocol["release"])}}}'
    if kind == "named_barrier":
        named = protocol
        return f'pl::named_barrier{{{_tree(named["id"])}, {named["count"]}}}'
    if kind == "counter":
        counter = protocol
        return f'pl::counter{{{_tree(counter["slot"])}, {_tree(counter["release"])}, {_tree(counter["acquire"])}}}'
    raise AnnotationError("protocol", f"unsupported contract protocol {kind}")


def _claim(claim):
    result = f'pl::claim({_tree(claim.handle)}, {_tree(claim.domain)})'
    if claim.extent.table:
        result += f'.table({_pairs(claim.extent.table)})'
    else:
        result += f'.extent({claim.extent.const}, {_pairs(claim.extent.per)})'
    result += f'.align({claim.align}).lifetime({_tree(claim.lifetime)})'
    for field, method in (("absolute_base", "at"), ("offset", "offset"), ("alias", "alias")):
        value = getattr(claim, field)
        if value is not None:
            result += f'.{method}({_tree(value)})'
    return f'PL_CLAIM_EX({result});'

def _warp_handle(composition, role):
    candidates = [name for name, row in composition.get("handles", {}).items()
                  if row.get("domain") == "warp" and
                  (row.get("owner") == f"claim:warp-{role}" or name.endswith("_warp_" + role))]
    if len(candidates) > 1:
        raise AnnotationError("roles", f"ambiguous warp claim for {role}")
    return candidates[0] if candidates else f"pipelink_tilelang_warp_{role}"


def _contract_leaves(ledger, field, value, source):
    if isinstance(value, dict):
        for name, item in value.items():
            escaped = name.replace("~", "~0").replace("/", "~1")
            _contract_leaves(ledger, f"{field}.{name}", item, f"{source}/{escaped}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _contract_leaves(ledger, f"{field}.{index}", item, f"{source}/{index}")
    else:
        ledger.record(field, source, kind="contract")


def _partners(stage):
    return list({channel.partner.sha256: channel.partner for channel in stage.channels}.values())


def _source(partner, pointer):
    return f"{partner.path}#sha256={partner.sha256}:{pointer}"


def entry_rendezvous(stage):
    """The partners' agreed entry rendezvous, or ``None`` (always for role-less stages)."""
    declared = {(partner.manifest.entry_rendezvous.barrier, partner.manifest.entry_rendezvous.threads): partner
                for partner in _partners(stage) if partner.manifest.entry_rendezvous is not None}
    if len(declared) > 1:
        raise AnnotationError("stage.entry_rendezvous", "partners declare different entry rendezvous")
    if not declared or not stage.roles:
        return None
    (barrier, threads), partner = next(iter(declared.items()))
    return barrier, threads, partner


def _entry_rendezvous(stage, ledger):
    join = entry_rendezvous(stage)
    if join is None:
        return []
    barrier, threads, partner = join
    for key in ("barrier", "threads"):
        ledger.record(f"stage.entry_rendezvous.{key}", _source(partner, f"/stage/entry_rendezvous/{key}"), kind="contract")
    return [f"PL_ENTRY_RENDEZVOUS({barrier}, {threads});"]


def _reg_groups(stage, composition, ledger):
    """Join partner-named register groups: ``COMPOSITION["reg_groups"] = {group: (role, ...)}``.

    The handle and values come from the partner's regs claim of that group.
    """
    lines = []
    for group, roles in composition.get("reg_groups", {}).items():
        unknown = set(roles) - set(stage.roles)
        if not roles or unknown:
            raise AnnotationError(f"reg_groups.{group}", f"maps to unknown roles {sorted(unknown)}")
        found = {}
        for partner in _partners(stage):
            for index, claim in enumerate(partner.manifest.claims):
                if claim.domain == "regs" and claim.reg_group == group:
                    found.setdefault((claim.handle, tuple(claim.values)), (partner, index))
        if len(found) != 1:
            raise AnnotationError(f"reg_groups.{group}", "expected one agreed partner regs claim for the group")
        (handle, values), (partner, index) = next(iter(found.items()))
        lines.append(f'PL_CLAIM_REGS({_tree(handle)}, {_tree(group)}, {_tree(values)}, {", ".join(_tree(role) for role in roles)});')
        for key in ("handle", "reg_group", "values"):
            ledger.record(f"claims.{handle}.{key}", _source(partner, f"/claims/{index}/{key}"), kind="contract")
        ledger.record(f"claims.{handle}.roles", f"COMPOSITION.reg_groups.{group}", kind="declared")
    return lines


def _staging_bytes(buffer, generated):
    """Capture physical allocations after layout lowering, before arena merging."""
    sizes = []

    def visit(node):
        if isinstance(node, tvm.tirx.AllocBuffer):
            physical = node.buffer
            if physical.scope().startswith("shared") and (
                    physical.name == buffer.name or physical.data.name == buffer.data.name):
                dtype = tvm.DataType(physical.dtype)
                sizes.append(prod(int(extent) for extent in physical.shape) * dtype.bits * dtype.lanes // 8)

    for generation in generated.values():
        for function in generation["modules"]["physical_allocations"].functions.values():
            tvm.tirx.stmt_functor.post_order_visit(function.body, visit)
    if not sizes:
        raise AnnotationError(f"staging.{buffer.name}.extent", "no lowered physical storage view")
    return max(sizes)


def _tensor_annotations(channel, sites, types, ledger, demands, generated):
    identity = channel.export["identity"]
    demand = channel.demand
    source = f"{channel.partner.path}#sha256={channel.partner.sha256}:/tensor/exports/{channel.export_index}"
    for field in ("identity", "calls", "readiness"):
        _contract_leaves(ledger, f"tensor.imports.{channel.name}.{field}", channel.export[field], source + "/" + field)
    ledger.record(f"tensor.imports.{channel.name}.demand", f"captures.layouts.{demand.fragment}", kind="captured")
    for field in ("demand.site", "lends", "rendezvous", "completion"):
        ledger.record(f"tensor.imports.{channel.name}.{field}", "frontend generated import", kind="captured")
    ledger.record(f"tensor.imports.{channel.name}.arguments",
                  "import_tensor.arguments" if channel._arguments_explicit else "published member name/type binding",
                  kind="declared" if channel._arguments_explicit else "convention")
    if channel.outputs:
        ledger.record(f"tensor.imports.{channel.name}.outputs", "import_tensor.outputs", kind="declared")
    if channel.chunk:
        ledger.record(f"tensor.imports.{channel.name}.chunk", "fragment.chunk axis/width", kind="declared")
    ledger.record(f"tensor.imports.{channel.name}.specialization", source + "/placements", kind="contract")
    for guarantee in demand.guarantees:
        ledger.record(f"tensor.sites.{demand.site}.guarantees.{guarantee}",
                      "captures.pre_codegen import schedule", kind="captured")
    elem = {"f32": "float", "f64": "double", "f16": "uint16_t", "bf16": "uint16_t",
            "i8": "int8_t", "u8": "uint8_t", "i16": "int16_t", "u16": "uint16_t",
            "i32": "int32_t", "u32": "uint32_t", "i64": "int64_t", "u64": "uint64_t",
            "s8": "int8_t", "s16": "int16_t", "s32": "int32_t", "s64": "int64_t"}[identity["dtype"]]
    lines = [
        channel.call_declaration(channel.produce_endpoint, channel.produce_symbol),
        f'extern "C" __device__ void {channel.symbol}(pipelink_tensor_source source, '
        + ('uint32_t chunk, ' if channel.chunk else '') + f'{elem}* fragment, void* scratch);',
        f'extern "C" __device__ void {channel.symbol}_ready();',
        f'extern "C" __device__ void {channel.symbol}_done();',
    ]
    if channel.argument_type is not None:
        previous = types.get(channel.argument_type)
        if previous is not None and previous != channel.argument_decl:
            raise AnnotationError(f"tensor.arguments.{channel.argument_type}", "partners disagree on the published argument type")
        if previous is None:
            # The call declaration's prelude defines the type; PL_TYPE names its members.
            lines.append(f'PL_TYPE({channel.argument_type}, {", ".join(channel.argument_members)});')
            types[channel.argument_type] = channel.argument_decl
    lines.append(f'PL_CALL({channel.produce_name}, {channel.produce_symbol}, pl::produce, {_tree(channel.role)}, {_tree(channel.produce_endpoint.protocol.call.export)});')
    if channel.release_endpoint is not None:
        lines.extend([
            channel.call_declaration(channel.release_endpoint, channel.release_symbol),
            f'PL_CALL({channel.release_name}, {channel.release_symbol}, pl::produce, {_tree(channel.role)}, {_tree(channel.release_endpoint.protocol.call.export)});',
        ])
    site = (channel.role, demand.participants, identity["work"], demand.guarantees)
    if demand.site in sites and sites[demand.site][0] != site:
        raise AnnotationError(f"tensor.sites.{demand.site}", "imports at a site must agree on its contract")
    sites.setdefault(demand.site, (site, []))[1].append(channel.ready)
    if channel.chunk:
        sites[demand.site][1].append("chunk")
        lines.append(f'PL_BINDING("chunk", "bits[32]", pl::coord{{"chunk", 0, {channel.chunk[2]}}}, '
                     f'{_tree(channel.role)}, "chunk", {_tree(demand.site)}, {_tree(channel.ready + "_to_" + channel.done)});')
    coords = ", ".join("{" + f'{_tree(item["name"])}, {item["lo"]}, {item["hi"]}' + "}"
                       for item in identity["coords"])
    identity_cpp = "pl::identity{" + ", ".join(_tree(identity[key]) for key in
        ("result", "work", "version", "dtype", "interpretation")) + ", {" + coords + "}}"
    record, rows = demands[channel.name]
    shape = record["logical_shape"]
    expected_shape = [item["hi"] - item["lo"] for item in identity["coords"]]
    if channel.chunk:
        expected_shape[channel.chunk[0]] = channel.chunk[1]
    if shape != expected_shape:
        raise AnnotationError(f"tensor.imports.{channel.name}.demand", "fragment shape differs from the export's coordinates")
    # Inputs index the fragment (plus its replica); outputs are identity coordinates, participant, position.
    inputs = [f"i{index}" for index in range(len(shape))] + (["replica"] if record["replication"] > 1 else [])
    extents = [*shape] + ([record["replication"]] if record["replication"] > 1 else [])
    domains = ", ".join(f'pl::coord{{{_tree(name)}, 0, {extent}}}' for name, extent in zip(inputs, extents))
    linear = "0"
    for name, extent in zip([inputs[-1], *inputs[:-1]] if record["replication"] > 1 else inputs,
                            [extents[-1], *extents[:-1]] if record["replication"] > 1 else extents):
        linear = f"({linear} * {extent} + {name})"
    table = ", ".join(f"{{{participant}, {position}}}" for _, participant, position in rows)
    outputs = ", ".join(
        (f"{channel.chunk[1]} * chunk + " if channel.chunk and channel.chunk[0] == index else "")
        + (f"{item['lo']} + i{index}" if item["lo"] else f"i{index}")
        for index, item in enumerate(identity["coords"]))
    binding_domain = ', pl::bind("chunk", "bits[32]")' if channel.chunk else ''
    parameters = ', '.join('int ' + name for name in inputs) + (', pl::bound chunk' if channel.chunk else '')
    offset_table = (f"static constexpr int offsets[] = {{{', '.join(map(str, channel._chunk_offsets))}}}; "
                    if channel.chunk and any(channel._chunk_offsets) else "")
    participant = "holder[0] + offsets[chunk]" if offset_table else "holder[0]"
    demand_cpp = (f"pl::demand_positions{{{domains}{binding_domain}, []({parameters}) {{ "
                  f"static constexpr std::array<int, 2> holders[] = {{{table}}}; "
                  f"const auto& holder = holders[{linear}]; {offset_table}"
                  f"return std::array<int, {len(shape) + 2}>{{{outputs}, {participant}, holder[1]}}; }}}}")
    lends = []
    for lend in channel.lends:
        kind = lend["kind"]
        args = [_tree(f"{channel.name}.{lend['name']}")]
        if kind == "cta_barrier":
            args.append(str(lend["base"]))
        else:
            extent = lend.get("extent")
            if extent is None and kind == "regs":
                extent = prod(record["physical_shape"])
            args.append(str(extent))
            if kind == "smem":
                args.append(str(lend["align"]))
            args.append(_tree(lend["handle"]))
        args.extend(_tree(lend.get(key, default)) for key, default in
                    (("owner", channel.role), ("from", channel.ready), ("to", channel.done)))
        if kind == "cta_barrier":
            args.append(_tree(lend["sync"]))
        lends.append(f'pl::{kind}({", ".join(args)})')
        if kind == "smem":
            lines.append(f'extern "C" __device__ int {lend["handle"]}();')
    rendezvous = ", ".join(f"pl::rendezvous{{{_tree(item.event)}, {_tree(channel.name + '.' + item.barrier)}, {item.cohort}}}"
                           for item in channel.rendezvous)
    lines.append(f'PL_IMPORT_TENSOR({channel.name}, {channel.symbol}, {identity_cpp}, '
                 'pl::args{{"source", "source_base"}'
                 + (', {"chunk", "binding"}' if channel.chunk else '')
                 + ', {"fragment", "dest_fragment"}, {"scratch", "scratch"}}, '
                 f'{demand_cpp}, pl::strings{{{_tree(demand.site)}}}, {_tree(channel.done)}, '
                 f'pl::lends{{{", ".join(lends)}}}, {{{rendezvous}}});')
    for buffer, handle in channel._staging:
        size = _staging_bytes(buffer, generated)
        until = "" if channel.chunk else f", {_tree(channel.done)}"
        lines.append(f'PL_CLAIM("{handle}", "smem", {size}, 128, '
                     f'pl::during{{{_tree(channel.name)}, {_tree(channel.ready)}{until}}});')
        lines.append(f'extern "C" __device__ int {handle}();')
        ledger.record(f"claims.{handle}.extent", f"captures.physical_allocations.{buffer.name}", kind="captured")
        ledger.record(f"claims.{handle}.lifetime", source + "/readiness", kind="contract")
    return lines



def _demands(stage, composition, generated):
    """Each import's captured destination layout; every lowered depth must agree.

    Captured participants are physical CTA thread indices, including whole-warp
    subsets, so the enclosing callable role must start at warp zero; a relocated
    role is rejected.
    """
    from .import_checks import validate_imports
    validate_imports(stage, generated)
    demands = {}
    for channel in stage.channels:
        if not isinstance(channel, TensorImportChannel):
            continue
        name = channel.demand.fragment
        field = f"tensor.imports.{channel.name}.demand"
        spec = stage.roles[channel.role]
        base = composition.get("forced_bindings", {}).get(_warp_handle(composition, channel.role), spec.warp_range[0])
        width = 32 * (spec.warp_range[1] - spec.warp_range[0])
        if spec.warp_range[0] != 0 or base != 0:
            raise AnnotationError(field, "captured demands require their role at warp 0")
        captured = []
        for generation in generated.values():
            if name not in generation["layout_objects"]:
                raise AnnotationError(field, f"no captured fragment layout named {name!r}")
            record = generation["facts"]["layouts"][name]
            threads = generation["facts"]["captures"]["pre_codegen"]["attrs"]["thread_extent"]["threadIdx.x"]
            first, extent = record["thread_min"], record["thread_extent"]
            if threads != width or first < 0 or first + extent > width:
                raise AnnotationError(field, f"fragment {name!r} exceeds the role's {width} threads")
            try:
                rows = fragment_rows(generation["layout_objects"][name], record, field)
            except CaptureError as error:
                raise AnnotationError(field, str(error)) from error
            offsets = channel._chunk_offsets if channel.chunk else (0,)
            participants = sorted({thread + offset for _, thread, _ in rows for offset in offsets})
            if participants[0] < 0 or participants[-1] >= width:
                raise AnnotationError(field, "chunk owners exceed the physical CTA")
            if (not participants or participants[0] % 32 or len(participants) % 32
                    or participants != list(range(participants[0], participants[-1] + 1))):
                raise AnnotationError(field, "fragment owners must be a contiguous complete-warp subset")
            if participants != list(range(width)):
                cohort = (f'pl::participants{{pl::coord{{"t", {participants[0]}, {participants[-1] + 1}}}, '
                          '[](int t) { return t; }}')
                channel.demand = replace(channel.demand, participants=cohort)
            captured.append((record, rows))
        if any(rows != captured[0][1] for _, rows in captured[1:]):
            raise AnnotationError(field, "depth variants lower different destination layouts")
        demands[channel.name] = captured[0]
    return demands


def _annotations(stage, composition, ledger, generated):
    prefix = symbol_prefix(stage)
    demands = _demands(stage, composition, generated)
    lines = [f'PL_STAGE({_tree(stage.stage_id)}, {_tree(stage.version)});']
    if stage.host is None:
        lines.append(f'PL_PARAMS({stage.params.name}, {", ".join(field.name for field in stage.params.fields)});')
    else:
        # The included host group implies empty outer params.
        lines.extend(f"#define {name} {value}" for name, value in stage.host.defines.items())
        lines.append(f'#include "{stage.host.header}"')
        lines.extend(f"#undef {name}" for name in stage.host.defines)
        residue = stage.host.residue_type
        lines.append(f"static_assert(sizeof({stage.params.name}) == sizeof({residue}));")
        lines.append(f"static_assert(alignof({stage.params.name}) == alignof({residue}));")
        lines.extend(f"static_assert(offsetof({stage.params.name}, {field.name}) == offsetof({residue}, {field.name}));"
                     for field in stage.params.fields)
        ledger.record("host", "STAGE.host owner annotations", kind="declared")
    for field in ("stage_id", "version", "params", "launch_register_cap", "roles"):
        derived = stage.program is not None and field in ("params", "roles")
        ledger.record("declarations." + field, "PrimFunc signature/kernel" if derived else "STAGE." + field,
                      kind="captured" if derived else "declared")
    for field in ("bytes", "align"):
        ledger.record("params." + field, f'{"sizeof" if field == "bytes" else "alignof"}({stage.params.name})', kind="reflected")
    for field in stage.params.fields:
        ledger.record(f"params.fields.{field.name}.offset", f"offsetof({stage.params.name}, {field.name})", kind="reflected")
        ledger.record(f"params.fields.{field.name}.bytes", f"sizeof((({stage.params.name}*)nullptr)->{field.name})", kind="reflected")
        ledger.record(f"params.fields.{field.name}.align", f"__alignof__((({stage.params.name}*)nullptr)->{field.name})", kind="reflected")
    handles = composition.get("handles", {})
    for role, spec in stage.roles.items():
        warps = spec.warp_range[1] - spec.warp_range[0]
        lines.append(f'PL_ROLE({role}, {prefix}role_{role}, {warps});')
        handle = _warp_handle(composition, role)
        base = composition.get("forced_bindings", {}).get(handle, spec.warp_range[0])
        alignment = 4 if warps % 4 == 0 and base % 4 == 0 else 1
        lines.append(f'PL_CLAIM_EX(pl::claim({_tree(handle)}, "warp").extent({warps}).align({alignment}).at({base}).roles({{{_tree(role)}}}));')
        ledger.record(f"claims.{handle}", f"STAGE.roles.{role}.warp_range + COMPOSITION.forced_bindings", kind="declared")
        for name, row in handles.items():
            if row.get("domain") == "regs" and row.get("owner") == f"reg_group:{role}":
                lines.append(f'PL_CLAIM_REGS({_tree(name)}, {_tree(role)}, {_tree(row["values"])}, {_tree(role)});')
                ledger.record(f"claims.{name}", f"COMPOSITION.handles.{name}", kind="declared")
    lines.extend(_reg_groups(stage, composition, ledger))
    lines.extend(_entry_rendezvous(stage, ledger))
    endpoints, claims, topologies, variants = {}, {}, [], None
    slot_handles: set[str] = set()
    tensor_sites, tensor_types = {}, {}
    for channel in stage.channels:
        topology = channel.partner.raw.get("topology")
        if topology is not None and topology not in topologies:
            topologies.append(topology)
        if isinstance(channel, TensorImportChannel):
            lines.extend(_tensor_annotations(channel, tensor_sites, tensor_types, ledger, demands, generated))
            continue
        declaration = channel.declaration(stage.stage_id, stage.version, stage.params, stage.launch_register_cap)
        if declaration.variants is not None:
            if variants is not None and variants != declaration.variants:
                raise AnnotationError("variants", "independent depth handles require a joint variant recipe")
            variants = declaration.variants
        for field, source, kind in channel.ledger_entries():
            if field not in ledger.sources:
                ledger.record(field, source, kind=kind)
        names = {endpoint.name for endpoint in declaration.endpoints}
        for endpoint in channel.partner.raw["endpoints"]:
            if endpoint["name"] in names:
                key = (channel.partner.sha256, endpoint["name"])
                endpoints.setdefault(key, (endpoint, channel.role, channel.direction))
        needed = set(channel.binds.values())
        native = {side.native.base for side in (channel.full, channel.empty) if getattr(side, "native", None)}
        if channel.protocol.named_barrier:
            needed.add(channel.protocol.named_barrier.id_handle)
        partner_claims = {claim.handle: (index, claim) for index, claim in enumerate(channel.partner.manifest.claims)}
        # The ring implies barrier slot allocations; the partner or the linker owns them.
        slots = {side.slots for side in (channel.full, channel.empty) if getattr(side, "slots", None)}
        slot_handles.update(slots)
        # Native barrier storage must be a local claim, even when it is also a slot handle.
        needed = (needed - slots) | native
        while needed:
            handle = min(needed)
            needed.remove(handle)
            key = (channel.partner.sha256, handle)
            if key in claims:
                continue
            index, claim = partner_claims[handle]
            previous = next((other for other in claims.values() if other.handle == handle), None)
            if previous is not None:
                if previous != claim:
                    raise AnnotationError(f"claims.{handle}", "partner claims sharing a handle must agree")
                continue
            claims[key] = claim
            ledger.record(f"claims.{handle}", f"{channel.partner.path}#sha256={channel.partner.sha256}:/claims/{index}", kind="contract")
            _contract_leaves(ledger, f"claims.{handle}", channel.partner.raw["claims"][index],
                             f"{channel.partner.path}#sha256={channel.partner.sha256}:/claims/{index}")
    # Role-less partners publish no topology; with none declared, pack applies the default.
    if len(topologies) > 1:
        raise AnnotationError("topology", "channels must agree on partner topology")
    for name, ((role, participants, work, guarantees), observes) in tensor_sites.items():
        lines.append(f'PL_SITE({name}, {_tree(role)}, {participants}, {_tree(work)}, '
                     f'pl::guarantees{_tree(guarantees)}, pl::strings{_tree(tuple(dict.fromkeys(observes)))});')
    lines.extend(_claim(claim) for claim in claims.values())
    for handle in sorted({claim.handle for claim in claims.values()} | slot_handles):
        lines.append(f'extern "C" __device__ int {handle}();')
    if variants:
        for index, depth in enumerate(variants.values):
            lines.append(f'PL_VARIANT({index}, {_pairs({variants.depth_handle: depth})});')
    exports = {export.name: export for export in stage.exports}
    provided = {endpoint["protocol"]["export"] for endpoint, _, direction in endpoints.values()
                if endpoint["protocol"]["kind"] == "call" and direction == "consume"}
    # A provided PL_CALL implies its export; only call-less exports are declared.
    for export in stage.exports:
        if export.name not in provided:
            lines.append(f'PL_EXPORT({export.name}, {export.symbol});')
    endpoint_names = set()
    for endpoint, role, direction in endpoints.values():
        name = endpoint["name"]
        if name in endpoint_names:
            raise AnnotationError(f"endpoints.{name}", "channels from different partners need distinct local endpoint names")
        endpoint_names.add(name)
        protocol = endpoint["protocol"]
        if protocol["kind"] == "call":
            export = exports[protocol["export"]]
            owner = f", {_tree(role)}" if direction == "produce" else ""
            lines.append(f'PL_CALL({name}, {export.symbol}, pl::{direction}{owner}, {_tree(export.name)});')
            continue
        if endpoint["space"] == "none":
            result = (f'pl::signal_endpoint({_tree(name)}, {_tree(direction)}, {_tree(role)})'
                      f'.scope({_tree(endpoint["scope"])}).protocol({_protocol(protocol)})')
        else:
            result = f'pl::endpoint({_tree(name)}, {_tree(direction)}, {_tree(role)})'
            for field in ("space", "scope", "proxy"):
                result += f'.{field}({_tree(endpoint[field])})'
            result += f'.tile({_tree(endpoint["tile"]["dtype"])}, {_tree(endpoint["tile"]["shape"])})'
            result += f'.layout({_layout(endpoint["layout"])}).protocol({_protocol(protocol)})'
        if "ring" in endpoint:
            ring = endpoint["ring"]
            kind, value = next(iter(ring["depth"].items()))
            result += f'.ring(pl::{kind}({_tree(value)}), {ring["stage_stride_bytes"]})'
        for key, handle in endpoint["binds"].items():
            result += f'.bind({_tree(key)}, {_tree(handle)})'
        lines.append(f'PL_ENDPOINT({name}, {result});')
    for topology in topologies:
        cluster = topology["cluster"]
        limits = cluster["launch"]
        compiled = "pl::dynamic" if cluster["compiled"] == "dynamic" else _tree(cluster["compiled"])
        lines.append(f'PL_TOPOLOGY(pl::topology{{{topology["cta_group"]["fixed"]}, pl::cluster{{{compiled}, pl::launch{{{limits["x_multiple_of"]}, {_tree(limits["max"])}}}}}, {_tree(topology["persistent"])}, {_tree(topology["scheduler"])}, {_tree(topology["multicast"])}}});')
        if "scheduler_strategy" in topology:
            strategy = topology["scheduler_strategy"]
            lines.append(f'PL_SCHEDULER_STRATEGY({_tree(strategy["id"])}, {_tree(json.dumps(strategy["inputs"]))});')
    return "\n".join(lines) + "\n", variants


def _entry_prologue(records, first_thread, join):
    """Role-entry code: materialize this role's kept barriers, fence, then join the CTA.

    The join is ``barrier.sync id, threads`` in the entry wrapper, never in TIR;
    the externalized body then drops its own initialization.
    """
    lines, seen = [], set()
    for record in records:
        key = (record.handle, record.native, record.side)
        if key in seen:
            continue
        seen.add(key)
        if not isinstance(record.count, int):
            raise AnnotationError("split.barrier.count", f"{record.buffer}: unresolved count {record.count!r}")
        native = record.native
        base = (f"{native.base}() * {native.unit_bytes} + {native.offset_bytes}" if native is not None
                else f"{record.handle}() * 16")
        stride = native.stride_bytes if native is not None else 16
        thread = f"{record.init_warp * 32}" if record.init_warp is not None else first_thread
        lines.append(f"    if (threadIdx.x == {thread}) for (int slot = 0; slot < {record.depth}; ++slot)\n"
                     f'      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(static_cast<unsigned>('
                     f"__cvta_generic_to_shared(pipelink_entry_smem + {base} + slot * {stride}))), \"r\"({record.count}) : \"memory\");\n")
    if lines:
        lines.append('    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n')
    barrier, threads, _ = join
    lines.append(f'    asm volatile("barrier.sync {barrier}, {threads};" ::: "memory");\n')
    return "".join(lines)


def _internal_storage(stage, generated, spec):
    """Share compiler-owned planes and barrier arrays across extracted roles."""
    arenas, barriers, extents = {}, {}, {}
    for depth, generation in generated.items():
        allocations, initializers = [], []

        def visit(node):
            if isinstance(node, tvm.tirx.AllocBuffer):
                allocations.append(node.buffer)
            elif (isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op)
                  and node.op.name == "tirx.ptx_init_barrier_thread_count"):
                initializers.append(node)

        for func in generation["modules"]["pre_codegen"].functions.values():
            tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
        dynamic = [buffer for buffer in allocations if buffer.scope() == "shared.dyn"]
        if len(dynamic) != 1:
            raise AnnotationError("internal_storage", "expected one merged shared arena")
        arena = arenas[depth] = dynamic[0]
        offset = prod(int(value) for value in arena.shape) * tvm.DataType(arena.dtype).bits // 8
        borrowed = {record.buffer for records in generation["external"].values() for record in records}
        local = [buffer for buffer in allocations if buffer.scope() in ("shared.barrier", "shared.cluster_barrier")
                 and buffer.name not in borrowed and buffer.data.name not in borrowed]
        if local and spec.get("initializer") not in stage.roles:
            raise AnnotationError("internal_storage.initializer", "local barriers require an initializing role")
        barriers[depth] = []
        for buffer in local:
            offset = (offset + 7) // 8 * 8
            barrier = _local_barrier(buffer, initializers, spec["handle"], offset)
            barriers[depth].append(barrier)
            offset += barrier.depth * 8
        extents[depth] = offset
    return arenas, barriers, max(extents.values())


def emit(out, module, generated):
    stage = module.STAGE
    prefix = symbol_prefix(stage)
    target = tvm.target.Target({"kind": "cuda", "arch": pipelink_tilelang.context.profile})
    ledger = Ledger()
    composition = getattr(module, "COMPOSITION", {})
    annotations, variants = _annotations(stage, composition, ledger, generated)
    depths = tuple(variants.values) if variants else (None,)
    if set(generated) != set(depths):
        raise AnnotationError("variants", "every contract depth must be lowered")
    facts = {"variants": {str(depth): generation["facts"] for depth, generation in generated.items()}}
    facts["tensor_imports"] = {
        channel.name: {"specialization": channel.specialization, "chunk": channel.chunk,
                       "outputs": channel.outputs, "completion": channel.completion}
        for channel in stage.channels if isinstance(channel, TensorImportChannel) and channel.generated}
    for depth in depths:
        ledger.record(f"captures.{depth}", f"facts.json#/variants/{depth}", kind="captured")
    bodies, entries, splits, preambles = [], [], {}, []
    fields = {field.name: field for field in stage.params.fields}
    internal = composition.get("internal_storage")
    arenas, local_barriers = {}, {}
    if internal:
        arenas, local_barriers, size = _internal_storage(stage, generated, internal)
        annotations += f'PL_CLAIM_EX(pl::claim({_tree(internal["handle"])}, "smem").extent({size}).align({internal["align"]}));\n'
        ledger.record(f'claims.{internal["handle"]}.extent', "captures.pre_codegen.allocations (shared.dyn + shared.barrier)", kind="captured")
        ledger.record(f'claims.{internal["handle"]}.align', "COMPOSITION.internal_storage.align", kind="declared")
    for barrier in composition.get("internal_named_barriers", []):
        annotations += f'PL_CLAIM_EX(pl::claim({_tree(barrier["handle"])}, "named_barrier").extent(1).align(1).roles({_tree(barrier["roles"])}));\n'
        annotations += f'extern "C" __device__ int {barrier["handle"]}();\n'
        ledger.record(f'claims.{barrier["handle"]}', "COMPOSITION.internal_named_barriers", kind="declared")
    join = entry_rendezvous(stage)
    for role, spec in stage.roles.items():
        handle = _warp_handle(composition, role)
        base = composition.get("forced_bindings", {}).get(handle, spec.warp_range[0])
        width = spec.warp_range[1] - spec.warp_range[0]
        array = next((name for name in spec.entry_params if name in fields and fields[name].elements is not None), None)
        if array is not None:
            raise AnnotationError(f"roles.{role}.entry_params", f"array params field {array!r} cannot be a role argument")
        owned_initializers = set()
        for channel in stage.channels:
            if isinstance(channel, TensorImportChannel):
                continue
            init = channel.protocol.init
            if (init is not None and init.materializer == "stage" and init.warp is not None
                    and base <= init.warp < base + width):
                owned_initializers.update(side.slots for side in (channel.full, channel.empty)
                                          if hasattr(side, "slots"))
        pointer = ("static_cast<const pipelink_host_view*>(raw_params)->residue" if stage.host is not None else "raw_params")
        entry = (f'extern "C" __device__ void {prefix}role_{role}(const void* raw_params, unsigned variant) {{\n'
                 f'  auto const& params = *static_cast<const {stage.params.name}*>({pointer});\n')
        tensor_arguments = {}
        for channel in stage.channels:
            if isinstance(channel, TensorImportChannel) and channel.role == role and channel.arguments is not None:
                entry += f"  {channel.argument_type} {channel.args_param}{{}};\n"
                for member, field in channel.arguments.items():
                    entry += f"  {channel.args_param}.{member} = params.{field};\n"
                tensor_arguments[channel.args_param] = "&" + channel.args_param
                if channel.view_call:
                    tensor_arguments[channel._view_var.name] = "const_cast<void*>(raw_params)"
        cases = []
        splits[role] = {}
        for index, depth in enumerate(depths):
            generation = generated[depth]
            records = generation.get("external")
            if records is None:
                records = {owner: [record for channel in stage.channels if channel.role == owner
                                   for record in channel.external] for owner in stage.roles}
            # Allocations are hoisted before role selection, including borrowed
            # resources only used by another arm. Rewrite them before pruning.
            external = [replace(record, init="drop")
                        if isinstance(record, ExternalBarrier) and record.init == "keep" and owner != role
                        else record
                        for owner, borrowed in records.items() for record in borrowed]
            external = [replace(record, init="keep")
                        if isinstance(record, ExternalBarrier) and record.handle in owned_initializers
                        else record for record in external]
            if internal:
                external.append(ExternalStorage(arenas[depth].name, internal["handle"], "smem"))
                external.extend(replace(record, init="keep" if internal["initializer"] == role else "drop")
                                for record in local_barriers[depth])
            if composition.get("forced_bindings", {}).get(handle, spec.warp_range[0]) != spec.warp_range[0]:
                external.append(ExternalThreadBase(handle, spec.warp_range[1] - spec.warp_range[0]))
            prologue = ""
            if join is not None:
                kept = [record for record in external if isinstance(record, ExternalBarrier) and record.init == "keep"]
                first_thread = (f"{handle}() * 32" if any(isinstance(record, ExternalThreadBase) for record in external)
                                else str(spec.warp_range[0] * 32))
                prologue = _entry_prologue(kept, first_thread, join)
                external = [replace(record, init="drop") if record in kept else record for record in external]
            split_prefix = prefix + role + f"_v{index}_"
            split = externalize_role(generation["modules"]["pre_codegen"], role=role, warp_range=spec.warp_range,
                                     branch=None, external=external, prefix=split_prefix,
                                     declared_params=spec.entry_params, target=target)
            if split.preamble not in preambles:
                preambles.append(split.preamble)
            bodies.append(split.body)
            splits[role][str(depth)] = split.record
            function = split.functions[0]
            primfunc = next(iter(generation["modules"]["pre_codegen"].functions.values()))
            parameter_types = {param.name: param.type_annotation for param in primfunc.params}
            arguments = []
            for name in function.params:
                if name in tensor_arguments:
                    arguments.append(tensor_arguments[name])
                    continue
                argument = "params." + name
                annotation = parameter_types[name]
                if fields[name].cpp_type == "CUtensorMap":
                    argument = "&" + argument
                elif isinstance(annotation, tvm.ir.PointerType):
                    # TIR pointer types do not carry C++ pointee constness.
                    cpp_type = fields[name].cpp_type
                    if cpp_type.startswith("const ") and cpp_type.endswith("*"):
                        argument = f"const_cast<{cpp_type.removeprefix('const ')}>({argument})"
                    # TileLang's CUDA half aliases are CUTLASS storage types,
                    # whereas the public host ABI may name CUDA's builtin types.
                    dtype = str(annotation.element_type.dtype)
                    storage_type = {"bfloat16": "bfloat16_t", "float16": "half_t"}.get(dtype)
                    if storage_type:
                        argument = f"reinterpret_cast<{storage_type}*>({argument})"
                arguments.append(argument)
            cases.append((index, prologue, f'{function.name}({", ".join(arguments)}); return;\n'))
            # Externalization emits this helper even when no local synchronization survives.
            # Bind it to a declared claim only when the generated body actually calls it.
            if split.body.count(split_prefix + "named_barrier_id(") > 1:
                annotations += f'PL_CLAIM_EX(pl::claim("pipelink_{split_prefix}named_barrier", "named_barrier").extent(1).align(1).roles({{{_tree(role)}}}));\n'
        # A join without role-owned initialization runs at entry, before variant dispatch.
        joined_at_entry = len({prologue for _, prologue, _ in cases}) == 1 and "mbarrier.init" not in cases[0][1]
        if any(prologue for _, prologue, _ in cases):
            entry += "  extern __shared__ __align__(1024) unsigned char pipelink_entry_smem[];\n" * (not joined_at_entry)
            entry += cases[0][1].replace("    ", "  ", 1) if joined_at_entry else ""
        entry += "  switch (variant) {\n"
        for index, prologue, call in cases:
            entry += f"    case {index}: {{\n{'' if joined_at_entry else prologue}      {call}    }}\n"
        entry += '    default: asm volatile("trap;");\n  }\n}\n'
        entries.append(entry)
    declarations = "".join(f'extern "C" __device__ void {prefix}role_{role}(const void*, unsigned);\n' for role in stage.roles)
    export_code = "".join(f'extern "C" __device__ {export.signature} {{\n{export.cpp_body}\n}}\n' for export in stage.exports)
    params = f'struct alignas({stage.params.alignas}) {stage.params.name} {{\n' + "".join(f"  {field.declarator};\n" for field in stage.params.fields) + "};\n"
    path = Path(out) / "generation" / stage.stage_id.rsplit(".", 1)[-1]
    path.mkdir(parents=True, exist_ok=True)
    source = "".join(preambles) + '#include <climits>\n#include <pipelink/stage.hpp>\n#include "facts.cuh"\n#include "params.h"\n'
    source += f'inline constexpr int {prefix}LaunchRegisterCap = {stage.launch_register_cap};\n'
    if any(isinstance(channel, TensorImportChannel) for channel in stage.channels):
        source += (
            '__device__ __forceinline__ unsigned long long pipelink_tilelang_source_word(pipelink_tensor_source source) { return source.word; }\n'
            '__device__ __forceinline__ pipelink_tensor_source pipelink_tilelang_source_from_word(unsigned long long word) { return {word}; }\n'
        )
    if any(isinstance(channel, TensorImportChannel) and channel.export["identity"]["dtype"] in ("f16", "bf16")
           for channel in stage.channels):
        source += ('__device__ __forceinline__ uint16_t* pipelink_tilelang_bits16(void* p) '
                   '{ return static_cast<uint16_t*>(p); }\n')
    source += declarations + export_code + annotations
    for channel in stage.channels:
        if isinstance(channel, TensorImportChannel) and channel.argument_type is not None:
            source += (f'__device__ __forceinline__ {channel.argument_type}* {channel.symbol}_args_cast(const void* p) '
                       f'{{ return const_cast<{channel.argument_type}*>(static_cast<const {channel.argument_type}*>(p)); }}\n')
            for name in sorted(channel._output_reads):
                member = next(m for m in channel.argument_decl.members if m.name == name)
                source += (f'__device__ __forceinline__ {member.ctype} {channel.symbol}_args_{name}(const void* p) '
                           f'{{ return static_cast<const {channel.argument_type}*>(p)->{name}; }}\n')
    source += "".join(bodies) + "".join(entries)
    projection = {"stage_id": stage.stage_id, "launch_register_cap": stage.launch_register_cap,
                  "ledger": ledger.sources, "split": splits, "symbol_prefix": prefix,
                  "params_type": stage.params.name, "params_header": str(path / "params.h"),
                  "source": str(path / "roles.cu"), "variants": list(variants.values) if variants else []}
    (path / "roles.cu").write_text(source)
    (path / "params.h").write_text('#pragma once\n#include <cstdint>\n#include <cuda.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n' + params)
    (path / "facts.cuh").write_text(FactsHeader.render(facts, prefix + "facts"))
    (path / "facts.json").write_text(json.dumps({**facts, "passes": {str(depth): item["passes"] for depth, item in generated.items()}}, indent=2) + "\n")
    (path / "projection.json").write_text(json.dumps(projection, indent=2) + "\n")
    for depth, generation in generated.items():
        for name, snapshot in generation["snapshots"].items():
            (path / f"depth{depth}.{name}.tir").write_text(snapshot)
    return {stage.stage_id: projection}
