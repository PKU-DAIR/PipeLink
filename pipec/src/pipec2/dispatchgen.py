from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from collections.abc import Iterable

from .host.contracts import host_link
from .model import (
    ClauseFailure,
    LinkError,
    ParamsLayout,
    Pipeline,
    ProtocolExpansion,
    RegGroup,
    Solution,
    StageManifest,
    Variant,
)
from .profile import Profile
from .protocol import check_init_warps
from .stubgen import reg_config_symbol
from .symbols import PARAMS_BASE_HANDLE, params_accessor_stages, params_accessor_symbol


@dataclass(frozen=True)
class _ParamsField:
    stage_id: str
    name: str
    offset: int
    byte_count: int
    align: int


@dataclass(frozen=True)
class _RoleDispatch:
    stage_id: str
    role_name: str
    symbol: str
    warp_lo: int
    warp_hi: int
    params_field: str
    variant_index: int | None
    variant_entry: bool
    variant_constant: str | None


@dataclass(frozen=True)
class _InitSpec:
    channel: str
    full_slots: str
    empty_slots: str | None
    arrive_full: int
    arrive_empty: int | None
    owner_role: str
    depth: int
    scope: str
    warp: int | None


def _plain_link_error(message: str) -> LinkError:
    return LinkError([ClauseFailure("L-LINK", "dispatch", message, {})])


def _params_error(path: str, message: str, **detail: object) -> LinkError:
    return LinkError([ClauseFailure("L-PARAMS", path, message, dict(detail))])




def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def params_layout(stages: Iterable[StageManifest], profile: Profile) -> ParamsLayout:
    """Lay out stage blobs; host links require a selected host projection."""

    base_align = profile.caps.param_align

    ordered = sorted(stages, key=lambda stage: stage.id)
    if host_link(ordered):
        raise _params_error("host", "host params require resolve_host followed by host_params_layout")
    if len({stage.id for stage in ordered}) != len(ordered):
        raise _params_error("stages", "stage IDs must be unique in the parameter layout")

    aggregate_align = base_align
    cursor = 0
    entries: list[tuple[str, int, int, int]] = []
    for stage in ordered:
        declared = stage.params
        effective_align = max(base_align, declared.align)
        offset = _align_up(cursor, effective_align)
        entries.append(
            (stage.id, offset, declared.bytes, declared.align)
        )
        cursor = offset + declared.bytes
        aggregate_align = max(aggregate_align, declared.align)

    total_bytes = _align_up(cursor, aggregate_align) if ordered else 0
    limit = profile.caps.param_limit_bytes
    if total_bytes > limit:
        raise _params_error(
            "params.total_bytes",
            "aggregate parameter layout exceeds the target parameter limit",
            actual=total_bytes,
            limit=limit,
        )
    return ParamsLayout(total_bytes, aggregate_align, entries)


def _fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9_]", "_", value).upper()
    if not fragment or fragment[0].isdigit():
        fragment = "_" + fragment
    return fragment


def _reg_group_name(group: RegGroup, pipeline: Pipeline) -> str:
    handle = pipeline.handles.get(group.handle)
    if handle is None or not handle.owner.startswith("reg_group:"):
        raise _plain_link_error(
            f"register handle {group.handle!r} does not identify its reg_group owner"
        )
    name = handle.owner.removeprefix("reg_group:")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise _plain_link_error(f"reg_group name must be a C identifier, got {name!r}")
    return name


def _ordered_groups(solution: Solution, pipeline: Pipeline) -> list[tuple[str, RegGroup]]:
    groups = [(_reg_group_name(group, pipeline), group) for group in solution.reg_groups]
    groups.sort(
        key=lambda item: (
            item[1].warp_lo,
            item[1].warp_hi,
            item[0],
            item[1].handle,
            tuple(sorted(item[1].roles)),
        )
    )
    names = [name for name, _ in groups]
    if len(names) != len(set(names)):
        raise _plain_link_error("solution contains duplicate reg_group names")
    return groups


def _selected_variant(stage: StageManifest, solution: Solution) -> Variant:
    selected = solution.variants.get(stage.id, {})
    candidates = [variant for variant in stage.variants if variant.key == selected]
    if len(candidates) != 1:
        raise _plain_link_error(
            f"stage {stage.id!r} has no unique variant for selected handle tuple"
        )
    return candidates[0]


def _params_fields(
    params: ParamsLayout, stages: list[StageManifest]
) -> tuple[list[_ParamsField], dict[str, _ParamsField]]:
    ordered_entries = sorted(params.entries, key=lambda item: item[0])
    expected_ids = [stage.id for stage in sorted(stages, key=lambda item: item.id)]
    actual_ids = [entry[0] for entry in ordered_entries]
    if actual_ids != expected_ids:
        raise _plain_link_error(
            "ParamsLayout entries must cover linked stages exactly in canonical stage order"
        )

    fields: list[_ParamsField] = []
    cursor = 0
    for index, (stage_id, offset, byte_count, align) in enumerate(ordered_entries):
        if offset < cursor or offset % align:
            raise _plain_link_error(f"ParamsLayout entry for {stage_id!r} overlaps or is misaligned")
        if byte_count == 0:
            cursor = offset
            continue
        name = f"stage_{index:04d}_{_fragment(stage_id).lower()}"
        field = _ParamsField(stage_id, name, offset, byte_count, align)
        fields.append(field)
        cursor = offset + byte_count
    if params.total_bytes < cursor:
        raise _plain_link_error("ParamsLayout total_bytes is smaller than its entries")
    if params.total_bytes and params.total_bytes % params.align:
        raise _plain_link_error("ParamsLayout total_bytes does not satisfy aggregate alignment")
    return fields, {field.stage_id: field for field in fields}


def _role_dispatches(
    solution: Solution,
    stages: list[StageManifest],
    fields: dict[str, _ParamsField],
) -> list[_RoleDispatch]:
    dispatches: list[_RoleDispatch] = []
    for stage_index, stage in enumerate(sorted(stages, key=lambda item: item.id)):
        variant = _selected_variant(stage, solution)
        if variant.entry is not None and len(stage.roles) != 1:
            raise _plain_link_error("Variant.entry requires a single-role stage")
        variant_constant = (
            None
            if variant.entry is not None
            else f"VARIANT_STAGE_{stage_index:04d}_{_fragment(stage.id)}"
        )
        for role_name in sorted(stage.roles):
            if role_name not in solution.warp_map:
                raise _plain_link_error(f"role {role_name!r} is absent from Solution.warp_map")
            warp_lo, warp_hi = solution.warp_map[role_name]
            dispatches.append(
                _RoleDispatch(
                    stage.id,
                    role_name,
                    variant.entry or stage.roles[role_name].entry,
                    warp_lo,
                    warp_hi,
                    fields[stage.id].name,
                    None if variant.entry is not None else variant.index,
                    variant.entry is not None,
                    variant_constant,
                )
            )
    dispatches.sort(key=lambda item: (item.warp_lo, item.warp_hi, item.role_name, item.stage_id))
    return dispatches


def _channel_endpoints(
    stages: list[StageManifest], channel: str
) -> list[tuple[str, object]]:
    endpoints = [
        (stage.id, endpoint)
        for stage in stages
        for endpoint in stage.endpoints
        if endpoint.name == channel
    ]
    endpoints.sort(key=lambda item: (item[0], item[1].direction, item[1].owner_role or ""))
    return endpoints


def _init_specs(
    solution: Solution,
    stages: list[StageManifest],
    expansions: list[ProtocolExpansion],
) -> list[_InitSpec]:
    specs: list[_InitSpec] = []
    materialized_roots: set[str] = set()
    for expansion in expansions:
        if expansion.init_materializer != "dispatch" or expansion.init_slots is None:
            continue
        root = expansion.shared_group[0] if expansion.shared_group is not None else expansion.channel
        if root in materialized_roots:
            continue
        materialized_roots.add(root)
        if (
            expansion.init_owner is None
            or expansion.arrive_full is None
            or (expansion.init_slots[1] is not None and expansion.arrive_empty is None)
        ):
            raise _plain_link_error(
                f"dispatch-materialized channel {expansion.channel!r} lacks init literals"
            )
        endpoints = _channel_endpoints(stages, expansion.channel)
        ring_endpoints = [endpoint for _stage_id, endpoint in endpoints if endpoint.ring is not None]
        if not ring_endpoints:
            raise _plain_link_error(
                f"dispatch-materialized channel {expansion.channel!r} has no ring declaration"
            )
        scopes = {endpoint.scope for endpoint in ring_endpoints}
        if len(scopes) != 1:
            raise _plain_link_error(
                f"dispatch-materialized channel {expansion.channel!r} has inconsistent scopes"
            )
        ring = ring_endpoints[0].ring
        assert ring is not None
        depth = (
            ring.depth.fixed
            if ring.depth.fixed is not None
            else solution.depths.get(ring.depth.handle or "")
        )
        if depth is None:
            raise _plain_link_error(
                f"dispatch-materialized channel {expansion.channel!r} has no selected depth"
            )
        full_slots, empty_slots = expansion.init_slots
        specs.append(
            _InitSpec(
                expansion.channel,
                full_slots,
                empty_slots,
                expansion.arrive_full,
                expansion.arrive_empty,
                expansion.init_owner,
                depth,
                next(iter(scopes)),
                next(
                    (endpoint.protocol.init.warp for _stage_id, endpoint in endpoints
                     if endpoint.protocol.init is not None and endpoint.protocol.init.warp is not None),
                    None,
                ),
            )
        )
    specs.sort(key=lambda item: item.channel)
    return specs


def _protocol_comments(expansions: list[ProtocolExpansion]) -> list[str]:
    lines: list[str] = []
    for expansion in expansions:
        if expansion.callable is not None:
            continue
        projection = {
            "arrive_empty": expansion.arrive_empty,
            "arrive_full": expansion.arrive_full,
            "channel": expansion.channel,
            "counter": expansion.counter,
            "init_materializer": expansion.init_materializer,
            "init_owner": expansion.init_owner,
            "init_slots": expansion.init_slots,
            "kind": expansion.kind,
            "named_barrier": expansion.named_barrier,
            "shared_group": expansion.shared_group,
            "tx_bytes_consumer": expansion.tx_bytes_consumer,
            "tx_bytes_producer": expansion.tx_bytes_producer,
        }
        lines.append(
            "// protocol "
            + json.dumps(projection, sort_keys=True, separators=(",", ":"))
        )
    return lines


def _struct_lines(params: ParamsLayout, fields: list[_ParamsField]) -> list[str]:
    lines = [f"struct alignas({params.align}) PipelinkParams {{"]
    cursor = 0
    padding_index = 0
    for field in fields:
        if field.offset > cursor:
            lines.append(
                f"  unsigned char padding_{padding_index:04d}[{field.offset - cursor}];"
            )
            padding_index += 1
        lines.append(
            f"  alignas({field.align}) unsigned char {field.name}[{field.byte_count}];"
        )
        cursor = field.offset + field.byte_count
    if params.total_bytes > cursor:
        lines.append(
            f"  unsigned char padding_{padding_index:04d}[{params.total_bytes - cursor}];"
        )
    lines.append("};")
    lines.append(
        f'static_assert(sizeof(PipelinkParams) == {params.total_bytes}, "PipelinkParams size");'
    )
    for field in fields:
        lines.append(
            f'static_assert(__builtin_offsetof(PipelinkParams, {field.name}) == {field.offset}, "{field.stage_id} params offset");'
        )
    return lines


def _extern_declarations(
    roles: list[_RoleDispatch],
    groups: list[tuple[str, RegGroup]],
    inits: list[_InitSpec],
) -> list[str]:
    declarations: dict[str, str] = {}

    def add(symbol: str, declaration: str) -> None:
        previous = declarations.get(symbol)
        if previous is not None and previous != declaration:
            raise _plain_link_error(f"generated symbol {symbol!r} requires conflicting declarations")
        declarations[symbol] = declaration

    for role in roles:
        if role.variant_entry:
            declaration = f'extern "C" __device__ void {role.symbol}(const void* params);'
        else:
            declaration = (
                f'extern "C" __device__ void {role.symbol}(const void* params, unsigned variant);'
            )
        add(role.symbol, declaration)
    for name, _group in groups:
        symbol = reg_config_symbol(name)
        add(symbol, f'extern "C" __device__ void {symbol}();')
    for spec in inits:
        for symbol in (spec.full_slots, spec.empty_slots):
            if symbol is not None:
                add(symbol, f'extern "C" __device__ int {symbol}();')
    return [declarations[symbol] for symbol in sorted(declarations)]


def generate_dispatch(
    solution: Solution,
    pipeline: Pipeline,
    stages: Iterable[StageManifest],
    expansions: Iterable[ProtocolExpansion],
    params: ParamsLayout,
    host: dict | None = None,
) -> str:
    """Generate the all-v2 kernel dispatch CUDA translation unit."""

    ordered_stages = sorted(stages, key=lambda stage: stage.id)

    if host is None:
        fields, fields_by_stage = _params_fields(params, ordered_stages)
    else:
        fields = [
            _ParamsField(name, f"segment_{index:04d}", offset, size, align)
            for index, (name, offset, size, align) in enumerate(
                sorted(params.entries, key=lambda entry: (entry[1], entry[0]))
            )
            if size
        ]
        fields_by_stage = {
            stage.id: next((field for field in fields if field.stage_id == stage.id),
                           _ParamsField(stage.id, "", 0, 0, 1))
            for stage in ordered_stages
        }
    roles = _role_dispatches(solution, ordered_stages, fields_by_stage)
    ordered_expansions = sorted(expansions, key=lambda item: item.channel)
    if any(item.init_slots is not None and item.init_owner is None for item in ordered_expansions):
        init_owners = check_init_warps(ordered_stages, solution.warp_map)
        ordered_expansions = [
            replace(item, init_owner=init_owners[item.channel])
            if item.init_slots is not None else item
            for item in ordered_expansions
        ]
    inits = _init_specs(solution, ordered_stages, ordered_expansions)
    groups = _ordered_groups(solution, pipeline)
    accessor_stages = params_accessor_stages(ordered_stages)
    params_cell = solution.native_bases[PARAMS_BASE_HANDLE] if accessor_stages else None
    kernel_params = host is None or bool(params.total_bytes)

    lines = ["// Generated by pipec2.dispatchgen; deterministic ABI v2 source.", ""]
    if kernel_params:
        lines.extend(_struct_lines(params, fields))
    if host is not None:
        lines.append(host["role_view_header"])
        for view in host["role_views"].values():
            layout = view["layout"]
            name = view["type"]
            lines.append(f'static_assert(sizeof({name}) == {layout["bytes"]}, "host role view size");')
            lines.append(f'static_assert(alignof({name}) == {layout["align"]}, "host role view alignment");')
            for field_name, member in layout["members"].items():
                lines.append(f'static_assert(__builtin_offsetof({name}, {field_name}) == {member["offset"]}, "host role view field offset");')


    for stage in accessor_stages:
        field = fields_by_stage[stage.id]
        lines.extend([
            "", f'extern "C" __device__ const void* {params_accessor_symbol(stage.id)}() {{',
            "  extern __shared__ unsigned char pipelink_smem[];",
            f"  const void* base = *reinterpret_cast<const void* const*>(pipelink_smem + {params_cell}u);",
            f"  return static_cast<const unsigned char*>(base) + {field.offset}u;",
            "}",
        ])
    declarations = _extern_declarations(roles, groups, inits)
    if declarations:
        lines.append("")
        lines.extend(declarations)

    comments = _protocol_comments(ordered_expansions)
    if comments:
        lines.append("")
        lines.extend(comments)

    if inits:
        lines.extend(
            [
                "",
                "__device__ __forceinline__ unsigned pipelink_shared_address(const void* ptr) {",
                "  unsigned address;",
                '  asm volatile("{ .reg .u64 p; cvta.to.shared.u64 p, %1; cvt.u32.u64 %0, p; }"',
                '               : "=r"(address) : "l"(ptr));',
                "  return address;",
                "}",
                "",
                "__device__ __forceinline__ void pipelink_init_mbarrier(const void* ptr, unsigned count) {",
                "  const unsigned address = pipelink_shared_address(ptr);",
                '  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"',
                '               : : "r"(address), "r"(count) : "memory");',
                "}",
            ]
        )

    if roles or groups or inits:
        lines.append("")
    for role in roles:
        role_fragment = _fragment(role.role_name)
        lines.append(f"constexpr unsigned ROLE_{role_fragment}_LO = {role.warp_lo}u;")
        lines.append(f"constexpr unsigned ROLE_{role_fragment}_HI = {role.warp_hi}u;")
    seen_variant_constants: set[str] = set()
    for role in roles:
        if role.variant_constant is None or role.variant_constant in seen_variant_constants:
            continue
        seen_variant_constants.add(role.variant_constant)
        assert role.variant_index is not None
        lines.append(
            f"constexpr unsigned {role.variant_constant} = {role.variant_index}u;"
        )
    for name, group in groups:
        fragment = _fragment(name)
        lines.append(
            f"constexpr unsigned REG_GROUP_{fragment}_LO = {group.warp_lo // 4}u;"
        )
        lines.append(
            f"constexpr unsigned REG_GROUP_{fragment}_HI = {group.warp_hi // 4}u;"
        )
    for spec in inits:
        lines.append(
            f"constexpr unsigned INIT_DEPTH_{_fragment(spec.channel)} = {spec.depth}u;"
        )

    lines.extend(["", 'extern "C" __global__'])
    # nvcc 13.3 rejects combining __launch_bounds__ and __maxnreg__; the
    # selected launch-register binding is the normative occupancy control.
    if "pipelink_launch_regs" in pipeline.handles:
        lines.append(f"__maxnreg__({solution.launch.regs_cap})")
    else:
        lines.append(f"__launch_bounds__({solution.launch.threads}, 1)")
    if isinstance(solution.launch.cluster, tuple):
        x, y, z = solution.launch.cluster
        lines.append(f"__cluster_dims__({x}, {y}, {z})")
    lines.extend(
        [
            "void pipelink_kernel() {" if not kernel_params else "void pipelink_kernel(__grid_constant__ const PipelinkParams p) {",
            "  const unsigned warp = threadIdx.x / 32u;",
            "  const unsigned warpgroup = warp / 4u;",
        ]
    )

    if inits or accessor_stages:
        lines.append("  extern __shared__ unsigned char pipelink_smem[];")
    if accessor_stages:
        lines.append("  if (threadIdx.x == 0u) {")
        if kernel_params:
            lines.append(f"    *reinterpret_cast<const void**>(pipelink_smem + {params_cell}u) = static_cast<const void*>(&p);")
        lines.append("  }")
    if inits:
        role_constants = {role.role_name: _fragment(role.role_name) for role in roles}
        for spec in inits:
            owner_fragment = role_constants.get(spec.owner_role)
            if owner_fragment is None:
                raise _plain_link_error(
                    f"init owner role {spec.owner_role!r} is not dispatched"
                )
            depth_constant = f"INIT_DEPTH_{_fragment(spec.channel)}"
            initializing_warp = f"{spec.warp}u" if spec.warp is not None else f"ROLE_{owner_fragment}_LO"
            lines.extend(
                [
                    f"  // dispatch-materialized init for {spec.channel}",
                    # Select the initializing thread by index, not `warp == X && elect`: a
                    # lane-divergent guard ahead of role dispatch makes nvcc compute `warp` before
                    # reconvergence and wrap dispatch in BSSY, which moves owner stages' uniform
                    # arithmetic off the uniform datapath.
                    f"  if (threadIdx.x == 32u * {initializing_warp}) {{",
                    f"    for (unsigned slot = 0; slot < {depth_constant}; ++slot) {{",
                    "      pipelink_init_mbarrier(",
                    f"          pipelink_smem + 16u * (static_cast<unsigned>({spec.full_slots}()) + slot),",
                    f"          {spec.arrive_full}u);",
                ]
            )
            if spec.empty_slots is not None:
                lines.extend([
                    "      pipelink_init_mbarrier(",
                    f"          pipelink_smem + 16u * (static_cast<unsigned>({spec.empty_slots}()) + slot),",
                    f"          {spec.arrive_empty}u);",
                ])
            lines.extend(
                [
                    "    }",
                    '    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");',
                    "  }",
                ]
            )
        lines.append("  __syncthreads();")
        if any(spec.scope == "cluster" for spec in inits):
            lines.extend(
                [
                    '  asm volatile("barrier.cluster.arrive.relaxed.aligned;" : : : "memory");',
                    '  asm volatile("barrier.cluster.wait.aligned;" : : : "memory");',
                ]
            )
    elif accessor_stages:
        lines.append("  __syncthreads();")

    def role_lines(role: _RoleDispatch, indent: str) -> list[str]:
        fragment = _fragment(role.role_name)
        view_lines = []
        if host is None or f"{role.stage_id}:{role.role_name}" not in host["role_views"]:
            params_expr = f"static_cast<const void*>(p.{role.params_field})" if role.params_field else "nullptr"
        else:
            view = host["role_views"][f"{role.stage_id}:{role.role_name}"]
            values = [
                "nullptr" if field["offset"] is None or not field["bytes"]
                else f"static_cast<const void*>(reinterpret_cast<const unsigned char*>(&p) + {field['offset']})"
                for field in view["fields"]
            ] or ["0"]
            view_lines.append(f"{indent}  const {view['type']} role_params = {{{', '.join(values)}}};")
            params_expr = "static_cast<const void*>(&role_params)"
        if role.variant_entry:
            call = f"{role.symbol}({params_expr});"
        else:
            assert role.variant_constant is not None
            call = f"{role.symbol}({params_expr}, {role.variant_constant});"
        return [
            f"{indent}if (warp >= ROLE_{fragment}_LO && warp < ROLE_{fragment}_HI) {{",
            *view_lines,
            f"{indent}  {call}",
            f"{indent}  return;",
            f"{indent}}}",
        ]

    # Register groups and their roles form ONE nested region each: the group's
    # setmaxnreg immediate is followed by that group's role branches without a
    # control-flow join. ptxas derives per-region register budgets from
    # setmaxnreg; a join after mixed .inc/.dec branches makes it compile every
    # later instruction to the smallest budget.
    grouped_roles: set[str] = set()
    for index, (name, group) in enumerate(groups):
        fragment = _fragment(name)
        prefix = "if" if index == 0 else "else if"
        lines.extend(
            [
                f"  {prefix} (warpgroup >= REG_GROUP_{fragment}_LO &&",
                f"           warpgroup < REG_GROUP_{fragment}_HI) {{",
                f"    {reg_config_symbol(name)}();",
            ]
        )
        for role in roles:
            if group.warp_lo <= role.warp_lo and role.warp_hi <= group.warp_hi:
                grouped_roles.add(role.role_name)
                lines.extend(role_lines(role, "    "))
        lines.append("  }")

    for role in roles:
        if role.role_name in grouped_roles:
            continue
        lines.extend(role_lines(role, "  "))

    lines.append("}")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["generate_dispatch", "params_layout"]
