"""Optional destination layouts derived from an export's transport semantics.

The layout is only a specialization hint; the planner and checker still decide
whether a connection holds. Unsupported maps return None.
"""
from __future__ import annotations

from itertools import product
from math import prod
from typing import Any, Sequence

from . import expr
from .contracts import Binding, TensorExport
from .machine import _option, tmem_column_count, tmem_lane_count


def _contains(node: expr.Expr, op: str) -> bool:
    return node.op == op or any(_contains(child, op) for child in node.args if isinstance(child, expr.Expr))


def _column_terms(node: expr.Expr) -> list[expr.Expr]:
    if node.op == "add":
        return _column_terms(node.args[0]) + _column_terms(node.args[1])
    return [node]


def delivery_layout(export: TensorExport, bindings: Sequence[Binding], profile: Any,
                    participants: Sequence[int]) -> tuple[expr.Expr, expr.Expr] | None:
    """Return (participant, private position) over zero-based identity coordinates.

    ``bindings`` are validated manifest binding declarations; ``participants``
    are the calling CTA thread IDs. Register placements are returned unchanged
    when binding-free. TMEM supports a single additive binding base and complete
    aligned receipts in every admitted world. Warps sharing a DP slice split
    whole column packets round-robin. The map is independent of bindings and targets.
    """
    coords = export.identity.coords
    if (len(export.placements) != 1 or not coords
            or any(c.values is not None or c.lo != 0 or c.hi <= 0 for c in coords)):
        return None
    placement, = export.placements
    if placement.guard is not None:
        return None
    components = placement.components
    if placement.space == "reg" and placement.units == "scalar":
        result = components["participant"], components["position"]
        return None if any(_contains(node, "bind") for node in result) else result
    if (placement.space != "tmem" or placement.units != "dp/column"
            or "tmem_packet_receipt" not in export.read_capabilities
            or export.identity.dtype not in {"fp32", "float32", "f32", "int32", "uint32", "i32", "u32"}):
        return None
    lanes = tmem_lane_count(profile)
    warp = _option(profile, "warp_size", 32)
    columns = _option(profile, "packet_columns", 16)
    if lanes is None or lanes < warp or lanes % warp:
        return None
    members = tuple(sorted(participants))
    # Start on a complete TMEM-lane cycle; additional whole warps share
    # lane quadrants and split their columns.
    if (not members or members[0] % lanes or len(members) % warp
            or members != tuple(range(members[0], members[0] + len(members)))):
        return None
    dp = components["dp"]
    if _contains(dp, "bind"):
        return None
    terms = _column_terms(components["column"])
    bases = [term for term in terms if _contains(term, "bind")]
    if len(bases) > 1 or any(_contains(term, "coord") for term in bases):
        return None
    local = [term for term in terms if not _contains(term, "bind")]
    position = expr.Expr("const", (0,), "int")
    if local:
        position = local[0]
        for term in local[1:]:
            position = expr.Expr("add", (position, term), "int")
    base = bases[0] if bases else expr.Expr("const", (0,), "int")
    used = [binding for binding in bindings if _contains_binding(base, binding.name)]
    domains = [(export.fixed_bindings[b.name],) if b.name in export.fixed_bindings else
               b.domain.values if b.domain.values is not None else range(b.domain.lo, b.domain.hi)
               for b in used]
    if prod(c.hi for c in coords) * prod(map(len, domains)) > _option(profile, "grounding_cap", 1_000_000):
        return None
    # Whole aligned rows give each packet a world-independent private offset.
    # Full rectangles rule out aliasing, partial DP slices and gaps, while
    # allowing coordinate permutations and whole-packet column shifts.
    rows: dict[int, set[int]] = {}
    for point in product(*(range(c.hi) for c in coords)):
        env = dict(zip((c.name for c in coords), point))
        row, slot = expr.evaluate(dp, env), expr.evaluate(position, env)
        if not 0 <= row < min(lanes, len(members)) or slot < 0 or slot in rows.setdefault(row, set()):
            return None
        rows[row].add(slot)
    for row, slots in rows.items():
        first = min(slots)
        if first % columns or len(slots) % columns or slots != set(range(first, first + len(slots))):
            return None
        if any(rows.get(other) != slots for other in range(row // warp * warp, (row // warp + 1) * warp)):
            return None
    extent = max(max(slots) + 1 for slots in rows.values())
    for world in product(*domains):
        offset = expr.evaluate(base, {**export.fixed_bindings, **dict(zip((b.name for b in used), world))})
        if offset < 0 or offset % columns or offset + extent > tmem_column_count(profile):
            return None
    participant = dp if members[0] == 0 else expr.Expr("add", (dp, expr.Expr("const", (members[0],), "int")), "int")
    if len(members) > lanes:
        def partition(copies):
            packet = expr.Expr("divc", (position, columns), "int")
            group = expr.Expr("modc", (packet, copies), "int")
            owner = expr.Expr("add", (participant, expr.Expr("mulc", (lanes, group), "int")), "int")
            private = expr.Expr("add", (
                expr.Expr("mulc", (columns, expr.Expr("divc", (position, columns * copies), "int")), "int"),
                expr.Expr("modc", (position, columns), "int")), "int")
            return owner, private

        cycles, remainder = divmod(len(members), lanes)
        owner, private = partition(cycles)
        if remainder:
            extra_owner, extra_private = partition(cycles + 1)
            extra = expr.Expr("lt", (dp, expr.Expr("const", (remainder,), "int")), "pred")
            owner = expr.Expr("select", (extra, extra_owner, owner), "int")
            private = expr.Expr("select", (extra, extra_private, private), "int")
        participant, position = owner, private
    return participant, position


def _contains_binding(node: expr.Expr, name: str) -> bool:
    return (node.op == "bind" and node.args[0] == name
            or any(_contains_binding(child, name) for child in node.args if isinstance(child, expr.Expr)))
