"""Emit explicit TileLang roles whose borrowed resources come from packed contracts."""
from collections import Counter
from dataclasses import dataclass, replace
import json
from math import prod
from pathlib import Path

from tilelang import tvm

import pipelink_tilelang

from .annotate import AnnotationError, FactsHeader, Ledger, symbol_prefix
from .contract import Channel, TensorImportChannel
from .external import ExternalBarrier, ExternalStorage, ExternalThreadBase, NativeStorage
from .declarations import ParamsSpec
from .externalize import externalize_role


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
class ContractStage:
    stage_id: str
    version: str
    launch_register_cap: int
    params: ParamsSpec
    roles: dict[str, RoleSpec]
    channels: list[Channel | TensorImportChannel]
    exports: tuple[Export, ...] = ()

    def __post_init__(self):
        # Keep provider channel names when possible. Qualify genuine collisions
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


def _tensor_annotations(channel, sites, types, ledger):
    identity = channel.export["identity"]
    demand = channel.demand
    source = f"{channel.partner.path}#sha256={channel.partner.sha256}:/tensor/exports/{channel.export_index}"
    for field in ("identity", "calls", "readiness"):
        _contract_leaves(ledger, f"tensor.imports.{channel.name}.{field}", channel.export[field], source + "/" + field)
    ledger.record(f"tensor.imports.{channel.name}.demand", "STAGE.channels.demand", kind="declared")
    ledger.record(f"tensor.imports.{channel.name}.lends", "STAGE.channels.lends", kind="declared")
    ledger.record(f"tensor.imports.{channel.name}.rendezvous", "STAGE.channels.rendezvous", kind="declared")
    ledger.record(f"tensor.imports.{channel.name}.arguments", "STAGE.channels.arguments", kind="declared")
    elem = {"f32": "float", "f64": "double", "f16": "__half", "bf16": "__nv_bfloat16",
            "i8": "int8_t", "u8": "uint8_t", "i16": "int16_t", "u16": "uint16_t",
            "i32": "int32_t", "u32": "uint32_t", "i64": "int64_t", "u64": "uint64_t"}[identity["dtype"]]
    lines = [
        channel.call_declaration(channel.produce_endpoint, channel.produce_symbol),
        f'extern "C" __device__ void {channel.symbol}(pipelink_tensor_source source, {elem}* fragment, void* scratch);',
        f'extern "C" __device__ void {channel.symbol}_ready();',
        f'extern "C" __device__ void {channel.symbol}_done();',
    ]
    if channel.argument_type is not None:
        previous = types.get(channel.argument_type)
        if previous is not None and previous != channel.type_members:
            raise AnnotationError(f"tensor.arguments.{channel.argument_type}", "partners disagree on the published argument type")
        if previous is None:
            lines.append(f'PL_TYPE({channel.argument_type}, {channel.type_members});')
            types[channel.argument_type] = channel.type_members
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
    coords = ", ".join("{" + f'{_tree(item["name"])}, {item["lo"]}, {item["hi"]}' + "}"
                       for item in identity["coords"])
    identity_cpp = "pl::identity{" + ", ".join(_tree(identity[key]) for key in
        ("result", "work", "version", "dtype", "interpretation")) + ", {" + coords + "}}"
    domains = ", ".join(f'pl::coord{{{_tree(name)}, {lo}, {hi}}}' for name, lo, hi in demand.domain)
    params = ", ".join(f"int {name}" for name, _, _ in demand.domain)
    mapping = ", ".join(map(str, demand.mapping))
    demand_cpp = f"pl::demand_positions{{{domains}, []({params}) {{ return std::array<int, {len(demand.mapping)}>{{{mapping}}}; }}}}"
    lends = []
    for lend in channel.lends:
        kind = lend["kind"]
        args = [_tree(f"{channel.name}.{lend['name']}")]
        if kind == "cta_barrier":
            args.append(str(lend["base"]))
        else:
            args.append(str(lend["extent"]))
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
                 'pl::args{{"source", "source_base"}, {"fragment", "dest_fragment"}, {"scratch", "scratch"}}, '
                 f'{demand_cpp}, pl::strings{{{_tree(demand.site)}}}, {_tree(channel.done)}, '
                 f'pl::lends{{{", ".join(lends)}}}, {{{rendezvous}}});')
    return lines



def _annotations(stage, composition, ledger):
    prefix = symbol_prefix(stage)
    lines = [f'PL_STAGE({_tree(stage.stage_id)}, {_tree(stage.version)});',
             f'PL_PARAMS({stage.params.name}, {", ".join(field.name for field in stage.params.fields)});']
    for field in ("stage_id", "version", "params", "launch_register_cap", "roles"):
        ledger.record("declarations." + field, "STAGE." + field, kind="declared")
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
    endpoints, claims, topologies, variants = {}, {}, [], None
    tensor_sites, tensor_types = {}, {}
    for channel in stage.channels:
        topology = channel.partner.raw["topology"]
        if topology not in topologies:
            topologies.append(topology)
        if isinstance(channel, TensorImportChannel):
            lines.extend(_tensor_annotations(channel, tensor_sites, tensor_types, ledger))
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
        for side in (channel.full, channel.empty):
            if getattr(side, "native", None):
                needed.add(side.native.base)
        if channel.protocol.named_barrier:
            needed.add(channel.protocol.named_barrier.id_handle)
        partner_claims = {claim.handle: (index, claim) for index, claim in enumerate(channel.partner.manifest.claims)}
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
    if len(topologies) != 1:
        raise AnnotationError("topology", "channels must agree on partner topology")
    for name, ((role, participants, work, guarantees), observes) in tensor_sites.items():
        lines.append(f'PL_SITE({name}, {_tree(role)}, {participants}, {_tree(work)}, '
                     f'pl::guarantees{_tree(guarantees)}, pl::strings{_tree(tuple(dict.fromkeys(observes)))});')
    lines.extend(_claim(claim) for claim in claims.values())
    for claim in claims.values():
        lines.append(f'extern "C" __device__ int {claim.handle}();')
    if variants:
        for index, depth in enumerate(variants.values):
            lines.append(f'PL_VARIANT({index}, {_pairs({variants.depth_handle: depth})});')
    exports = {export.name: export for export in stage.exports}
    for export in stage.exports:
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
    topology = topologies[0]
    cluster = topology["cluster"]
    limits = cluster["launch"]
    compiled = "pl::dynamic" if cluster["compiled"] == "dynamic" else _tree(cluster["compiled"])
    lines.append(f'PL_TOPOLOGY(pl::topology{{{topology["cta_group"]["fixed"]}, pl::cluster{{{compiled}, pl::launch{{{limits["x_multiple_of"]}, {_tree(limits["max"])}}}}}, {_tree(topology["persistent"])}, {_tree(topology["scheduler"])}, {_tree(topology["multicast"])}}});')
    if "scheduler_strategy" in topology:
        strategy = topology["scheduler_strategy"]
        lines.append(f'PL_SCHEDULER_STRATEGY({_tree(strategy["id"])}, {_tree(json.dumps(strategy["inputs"]))});')
    return "\n".join(lines) + "\n", variants


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
            count = {int(call.args[1]) for call in initializers
                     if isinstance(call.args[0], tvm.tirx.BufferLoad)
                     and call.args[0].buffer.data.same_as(buffer.data)}
            if len(count) != 1:
                raise AnnotationError("internal_storage.barrier", f"{buffer.name}: expected a uniform captured arrive count")
            slots = prod(int(value) for value in buffer.shape)
            barriers[depth].append(ExternalBarrier(
                buffer.name, "full", spec["handle"], NativeStorage(spec["handle"], 1, offset, 8),
                slots, count.pop(), "keep", 0))
            offset += slots * 8
        extents[depth] = offset
    return arenas, barriers, max(extents.values())


def emit(out, module, generated):
    stage = module.STAGE
    prefix = symbol_prefix(stage)
    target = tvm.target.Target({"kind": "cuda", "arch": pipelink_tilelang.context.profile})
    ledger = Ledger()
    composition = getattr(module, "COMPOSITION", {})
    annotations, variants = _annotations(stage, composition, ledger)
    depths = tuple(variants.values) if variants else (None,)
    if set(generated) != set(depths):
        raise AnnotationError("variants", "every contract depth must be lowered")
    facts = {"variants": {str(depth): generation["facts"] for depth, generation in generated.items()}}
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
    for role, spec in stage.roles.items():
        handle = _warp_handle(composition, role)
        base = composition.get("forced_bindings", {}).get(handle, spec.warp_range[0])
        width = spec.warp_range[1] - spec.warp_range[0]
        owned_initializers = set()
        for channel in stage.channels:
            if isinstance(channel, TensorImportChannel):
                continue
            init = channel.protocol.init
            if (init is not None and init.materializer == "stage" and init.warp is not None
                    and base <= init.warp < base + width):
                owned_initializers.update(side.slots for side in (channel.full, channel.empty)
                                          if hasattr(side, "slots"))
        entry = f'extern "C" __device__ void {prefix}role_{role}(const void* raw_params, unsigned variant) {{\n  auto const& params = *static_cast<const {stage.params.name}*>(raw_params);\n'
        tensor_arguments = {}
        for channel in stage.channels:
            if isinstance(channel, TensorImportChannel) and channel.role == role and channel.arguments is not None:
                entry += f"  {channel.argument_type} {channel.args_param}{{}};\n"
                for member, field in channel.arguments.items():
                    entry += f"  {channel.args_param}.{member} = params.{field};\n"
                tensor_arguments[channel.args_param] = "&" + channel.args_param
        entry += "  switch (variant) {\n"
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
            entry += f'    case {index}: {function.name}({", ".join(arguments)}); return;\n'
            # Externalization emits this helper even when no local synchronization survives.
            # Bind it to a declared claim only when the generated body actually calls it.
            if split.body.count(split_prefix + "named_barrier_id(") > 1:
                annotations += f'PL_CLAIM_EX(pl::claim("pipelink_{split_prefix}named_barrier", "named_barrier").extent(1).align(1).roles({{{_tree(role)}}}));\n'
        entry += '    default: asm volatile("trap;");\n  }\n}\n'
        entries.append(entry)
    declarations = "".join(f'extern "C" __device__ void {prefix}role_{role}(const void*, unsigned);\n' for role in stage.roles)
    export_code = "".join(f'extern "C" __device__ {export.signature} {{\n{export.cpp_body}\n}}\n' for export in stage.exports)
    params = f'struct alignas({stage.params.alignas}) {stage.params.name} {{\n' + "".join(f"  {field.cpp_type} {field.name};\n" for field in stage.params.fields) + "};\n"
    path = Path(out) / "generation" / stage.stage_id.rsplit(".", 1)[-1]
    path.mkdir(parents=True, exist_ok=True)
    source = "".join(preambles) + '#include <climits>\n#include <pipelink/stage.hpp>\n#include "facts.cuh"\n#include "params.h"\n'
    source += f'inline constexpr int {prefix}LaunchRegisterCap = {stage.launch_register_cap};\n'
    if any(isinstance(channel, TensorImportChannel) for channel in stage.channels):
        source += (
            '__device__ __forceinline__ unsigned long long pipelink_tilelang_source_word(pipelink_tensor_source source) { return source.word; }\n'
            '__device__ __forceinline__ pipelink_tensor_source pipelink_tilelang_source_from_word(unsigned long long word) { return {word}; }\n'
        )
    source += declarations + export_code + annotations
    for channel in stage.channels:
        if isinstance(channel, TensorImportChannel) and channel.argument_type is not None:
            source += (f'__device__ __forceinline__ const {channel.argument_type}* {channel.symbol}_args_cast(const void* p) '
                       f'{{ return static_cast<const {channel.argument_type}*>(p); }}\n')
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
