from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, NoReturn, TypeAlias

from .model import CanonEndpoint, Claim, ClauseFailure, Endpoint, LayoutError, SchemaError
from .profile import Profile


Tree: TypeAlias = int | tuple["Tree", ...]


class _Absent(Enum):
    VALUE = "absent"


_ABSENT = _Absent.VALUE


def _authored_fields(value: Any) -> dict[str, Any]:
    return {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != "space" and getattr(value, item.name) is not _ABSENT
    }


def _layout_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_layout_json(item) for item in value]
    if isinstance(value, Swizzle):
        return value.to_json_obj()
    return value


@dataclass(frozen=True)
class Swizzle:
    B: int
    M: int
    S: int
    unit: str | _Absent = _ABSENT

    def to_json_obj(self) -> dict[str, Any]:
        return _authored_fields(self)


@dataclass(frozen=True, kw_only=True)
class Layout:
    """Authored layout, not an expanded or canonical physical identity.

    The endpoint space selects AffineLayout or TmemLayout. Absence is distinct
    from null, and atom fields remain authored overrides until canonicalization.
    """

    space: str
    encoding: str | _Absent = _ABSENT
    atom: Any = _ABSENT
    elem_bits: int | _Absent = _ABSENT
    origin_bits: int | _Absent = _ABSENT

    def to_json_obj(self) -> dict[str, Any]:
        return {key: _layout_json(value) for key, value in _authored_fields(self).items()}


@dataclass(frozen=True, kw_only=True)
class AffineLayout(Layout):
    shape: Tree | _Absent = _ABSENT
    stride: Tree | _Absent = _ABSENT
    swizzle: Swizzle | None | _Absent = _ABSENT
    # Pointer-width validity depends on space and expanded atom fields.
    pointer_flag_bits: Any = _ABSENT


@dataclass(frozen=True, kw_only=True)
class TmemLayout(Layout):
    lanes: tuple[int, int] | _Absent = _ABSENT
    cols: int | _Absent = _ABSENT


def _syntax_failure(path: str, message: str, **detail: Any) -> NoReturn:
    raise SchemaError([ClauseFailure("L8", path, message, detail)])


def _layout_integer(value: Any, path: str, *, positive: bool) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        kind = "positive" if positive else "non-negative"
        _syntax_failure(path, f"must be a {kind} integer", actual=value)
    return value


def _tree(value: Any, path: str, *, positive: bool) -> Tree:
    if isinstance(value, (list, tuple)):
        if not value:
            _syntax_failure(path, "layout tree tuples must not be empty")
        return tuple(
            _tree(item, f"{path}[{index}]", positive=positive)
            for index, item in enumerate(value)
        )
    return _layout_integer(value, path, positive=positive)


def _load_swizzle(value: Any, path: str) -> Swizzle | None:
    if value is None:
        return None
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _syntax_failure(path, "layout.swizzle must be an object")
    unknown = set(value) - {"B", "M", "S", "unit"}
    if unknown:
        _syntax_failure(path, "layout.swizzle has unknown fields", fields=sorted(unknown))
    if not {"B", "M", "S"}.issubset(value):
        _syntax_failure(path, "layout.swizzle requires B, M, and S")
    b = _layout_integer(value["B"], f"{path}.B", positive=False)
    m = _layout_integer(value["M"], f"{path}.M", positive=False)
    s = value["S"]
    if type(s) is not int:
        _syntax_failure(f"{path}.S", "layout.swizzle.S must be an integer", actual=s)
    unit = value.get("unit", _ABSENT)
    if unit is not _ABSENT and unit not in ("element", "byte", "bit"):
        _syntax_failure(f"{path}.unit", "layout.swizzle.unit must be 'element', 'byte', or 'bit'", actual=unit)
    return Swizzle(b, m, s, unit)


def _complete_layout(layout: Layout, path: str) -> None:
    if layout.encoding != "pl-layout-v2":
        _syntax_failure(f"{path}.encoding", "layout.encoding must be 'pl-layout-v2'")
    required = ("elem_bits", "lanes", "cols") if isinstance(layout, TmemLayout) else ("elem_bits", "shape", "stride")
    for name in required:
        if getattr(layout, name) is _ABSENT:
            _syntax_failure(f"{path}.{name}", f"layout requires {name}")


def _matching_trees(layout: AffineLayout, path: str) -> None:
    if layout.shape is not _ABSENT and layout.stride is not _ABSENT:
        if _tree_profile(layout.shape) != _tree_profile(layout.stride):
            _syntax_failure(path, "layout shape and stride tree profiles differ")


def _load_layout(obj: Any, space: str, path: str, *, partial: bool) -> Layout:
    if not isinstance(obj, dict) or any(not isinstance(key, str) for key in obj):
        _syntax_failure(path, "layout must be an object with string keys")
    kind = TmemLayout if space == "tmem" else AffineLayout
    allowed = {item.name for item in fields(kind)} - {"space"}
    if partial:
        # Profile expansions are inline fields, not recursively named atoms.
        allowed.remove("atom")
    unknown = sorted(set(obj) - allowed)
    if unknown:
        _syntax_failure(path, "layout has unknown fields", fields=unknown)
    values = dict(obj)
    for name, positive in (("elem_bits", True), ("origin_bits", False)):
        if name in values:
            values[name] = _layout_integer(values[name], f"{path}.{name}", positive=positive)
    if kind is TmemLayout:
        if "lanes" in values:
            lanes = values["lanes"]
            if not isinstance(lanes, (list, tuple)) or len(lanes) != 2 or any(type(item) is not int for item in lanes):
                _syntax_failure(f"{path}.lanes", "TMEM layout.lanes must be a [start, stop] half-open range")
            start, stop = lanes
            if start < 0 or stop <= start:
                _syntax_failure(f"{path}.lanes", "TMEM lane range must be non-empty and non-negative")
            values["lanes"] = (start, stop)
        if "cols" in values:
            values["cols"] = _layout_integer(values["cols"], f"{path}.cols", positive=True)
    else:
        for name, positive in (("shape", True), ("stride", False)):
            if name in values:
                values[name] = _tree(values[name], f"{path}.{name}", positive=positive)
        if "swizzle" in values:
            values["swizzle"] = _load_swizzle(values["swizzle"], f"{path}.swizzle")
    layout = kind(space=space, **values)
    if not partial and layout.atom is _ABSENT:
        _complete_layout(layout, path)
    if isinstance(layout, AffineLayout):
        _matching_trees(layout, path)
    return layout


def load_layout(obj: Any, space: str, path: str) -> Layout:
    """Load only authored syntax; profile, tile, ring and claims remain contextual."""
    return _load_layout(obj, space, path, partial=False)


def _failure(ep: Endpoint, message: str, detail: dict[str, Any] | None = None) -> LayoutError:
    return LayoutError(
        [
            ClauseFailure(
                "L8",
                f"endpoints.{ep.name}.layout",
                message,
                {} if detail is None else detail,
            )
        ]
    )


def _positive_int(value: Any, name: str, ep: Endpoint) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _failure(ep, f"{name} must be a positive integer", {"actual": value})
    return value

def _default(value: Any, default: Any) -> Any:
    return default if value is _ABSENT else value


def _tree_profile(tree: Tree) -> object:
    if isinstance(tree, tuple):
        return tuple(_tree_profile(item) for item in tree)
    return None


def _flatten(tree: Tree) -> list[int]:
    if isinstance(tree, tuple):
        return [leaf for item in tree for leaf in _flatten(item)]
    return [tree]


def _tile_mode_extent(mode: Tree) -> int | None:
    """Return the extent represented by a scalar or singleton-tuple mode."""

    while isinstance(mode, tuple) and len(mode) == 1:
        mode = mode[0]
    return mode if isinstance(mode, int) else None


def _tile_congruent(shape: Tree, tile_shape: tuple[int, ...]) -> bool:
    if len(tile_shape) == 1:
        return _tile_mode_extent(shape) == tile_shape[0]
    if not isinstance(shape, tuple) or len(shape) != len(tile_shape):
        return False
    return all(
        _tile_mode_extent(mode) == extent
        for mode, extent in zip(shape, tile_shape, strict=True)
    )


def _coalesce(shape: Tree, stride: Tree) -> tuple[Tree, Tree]:
    modes: list[list[int]] = []
    for extent, step in zip(_flatten(shape), _flatten(stride), strict=True):
        if extent == 1:
            continue
        if modes and (
            modes[-1][1] * modes[-1][0] == step
            or (modes[-1][1] == 0 and step == 0)
        ):
            modes[-1][0] *= extent
        else:
            modes.append([extent, step])
    if not modes:
        return 1, 0
    if len(modes) == 1:
        return modes[0][0], modes[0][1]
    return tuple(mode[0] for mode in modes), tuple(mode[1] for mode in modes)


def _scale_tree(tree: Tree, scale: int) -> Tree:
    if isinstance(tree, tuple):
        return tuple(_scale_tree(item, scale) for item in tree)
    return tree * scale


def _cosize_bits(shape: Tree, stride_bits: Tree, elem_bits: int) -> int:
    maximum_start = sum(
        (extent - 1) * step
        for extent, step in zip(_flatten(shape), _flatten(stride_bits), strict=True)
    )
    return maximum_start + elem_bits


def _expand_atom(layout: Layout, profile: Profile, ep: Endpoint) -> Layout:
    atom = layout.atom
    if atom is _ABSENT:
        return layout
    if atom is None:
        raise _failure(ep, "layout has unknown fields", {"fields": ["atom"]})
    if not isinstance(atom, str) or not atom:
        raise _failure(ep, "layout.atom must be a non-empty string", {"actual": atom})
    try:
        expansion = profile.atoms[atom]
    except KeyError as exc:
        raise _failure(ep, f"unknown layout atom {atom!r}", {"atom": atom}) from exc
    if not isinstance(expansion, dict):
        raise _failure(ep, f"layout atom {atom!r} has a malformed expansion", {"atom": atom})
    if expansion.get("expansion") == "NotImplemented":
        raise _failure(ep, f"layout atom {atom!r} is NotImplemented", {"atom": atom})

    for key, value in layout.to_json_obj().items():
        if key != "atom" and key in expansion and expansion[key] != value:
            raise _failure(
                ep,
                f"layout field {key!r} conflicts with atom {atom!r}",
                {"atom": atom, "field": key, "atom_value": expansion[key], "actual": value},
            )
    try:
        # Only the newly encountered profile fields need syntax loading. Authored
        # overrides are already typed; merging them must not validate them again.
        expanded = _load_layout(expansion, ep.space, f"endpoints.{ep.name}.layout", partial=True)
        overrides = _authored_fields(layout)
        del overrides["atom"]
        result = replace(expanded, **overrides)
        _complete_layout(result, f"endpoints.{ep.name}.layout")
        if isinstance(result, AffineLayout) and (
            expanded.shape is _ABSENT or expanded.stride is _ABSENT
        ) and (layout.shape is _ABSENT or layout.stride is _ABSENT):
            _matching_trees(result, f"endpoints.{ep.name}.layout")
    except SchemaError as exc:
        failure = exc.failures[0]
        raise _failure(ep, failure.message, failure.detail) from exc
    return result


def _swizzle_bits(
    value: Swizzle | None,
    pointer_flag_bits: int,
    ep: Endpoint,
) -> tuple[int, int, int] | None:
    """Convert swizzle index units to bit positions, keeping B and S unchanged.

    Element indices use pointer_flag_bits (defaulting to layout.elem_bits);
    byte indices use eight bits, irrespective of the element width. CuTe's
    smem_ptr_flag_bits<B> does not make its outer Swizzle element-based: that
    swizzle acts on the byte pointer and must be imported with unit="byte" or
    recast to element units first.
    """
    if value is None:
        return None
    b, m, s = value.B, value.M, value.S
    unit = _default(value.unit, "element")
    if unit == "bit":
        shift = 0
    elif unit == "byte":
        shift = 3
    else:
        if pointer_flag_bits & (pointer_flag_bits - 1):
            raise _failure(
                ep,
                "element-unit swizzle requires a power-of-two pointer element width",
                {"pointer_flag_bits": pointer_flag_bits},
            )
        shift = pointer_flag_bits.bit_length() - 1
    return b, m + shift, s


def _ring(ep: Endpoint) -> tuple[tuple[str, int] | None, int | None]:
    if ep.ring is None:
        return None, None
    stride_bytes = _positive_int(
        ep.ring.stage_stride_bytes,
        "ring.stage_stride_bytes",
        ep,
    )
    depth = ep.ring.depth
    if depth.handle is not None:
        identity = depth.handle
    else:
        identity = f"fixed:{_positive_int(depth.fixed, 'ring.depth.fixed', ep)}"
    return (identity, stride_bytes), stride_bytes * 8


def _alignment_bits(
    ep: Endpoint,
    base_claim: Claim | None,
    profile: Profile,
    elem_bits: int,
) -> int | None:
    if base_claim is None:
        return None
    if base_claim.align is None:
        raise _failure(
            ep,
            "the base claim must carry an alignment",
            {"claim": base_claim.handle},
        )
    alignment = _positive_int(base_claim.align, "base claim alignment", ep)
    domain = profile.domains.get(base_claim.domain)
    if domain is not None:
        if domain.unit_scale is None:
            raise _failure(
                ep,
                "the base claim domain has no fixed physical unit scale",
                {"claim": base_claim.handle, "domain": base_claim.domain},
            )
        alignment *= domain.unit_scale
        physical_domain = domain.physical_domain
    else:
        physical_domain = base_claim.domain

    # SMEM and GMEM physical coordinates are bytes.  TMEM native coordinates
    # are columns, and the canonical address algebra measures those in bits.
    return alignment * elem_bits if physical_domain == "tmem" else alignment * 8


def _affine_layout(
    ep: Endpoint,
    layout: AffineLayout,
    elem_bits: int,
    ring_stride_bits: int | None,
) -> tuple[tuple[Any, ...], tuple[int, ...] | None, int]:
    shape, stride = layout.shape, layout.stride
    if not _tile_congruent(shape, ep.tile.shape):
        raise _failure(
            ep,
            "layout shape-tree profile is not congruent with tile.shape",
            {"layout_shape": _layout_json(shape), "tile_shape": list(ep.tile.shape)},
        )

    coalesced_shape, coalesced_stride = _coalesce(shape, stride)
    bit_stride = _scale_tree(coalesced_stride, elem_bits)
    origin_bits = _default(layout.origin_bits, 0)

    if ep.space == "smem":
        pointer_flag_bits = _positive_int(
            _default(layout.pointer_flag_bits, elem_bits),
            "layout.pointer_flag_bits",
            ep,
        )
        swizzle = _swizzle_bits(
            _default(layout.swizzle, None),
            pointer_flag_bits,
            ep,
        )
    else:
        if layout.swizzle is not _ABSENT or layout.pointer_flag_bits is not _ABSENT:
            raise _failure(ep, f"{ep.space} layouts do not support SMEM swizzle fields")
        pointer_flag_bits = None
        swizzle = None

    kind = "reg-tv" if ep.space == "reg" else ep.space
    address = (
        "pl-bit-address-v2",
        kind,
        ("shape", coalesced_shape),
        ("stride_bits", bit_stride),
        ("origin_bits", origin_bits),
        ("swizzle_xor_bits", swizzle),
        ("pointer_flag_bits", pointer_flag_bits),
        ("ring_stage_stride_bits", ring_stride_bits),
    )
    return address, swizzle, _cosize_bits(coalesced_shape, bit_stride, elem_bits)


def _tmem_layout(
    ep: Endpoint,
    layout: TmemLayout,
    elem_bits: int,
    ring_stride_bits: int | None,
) -> tuple[tuple[Any, ...], int]:
    lane_start, lane_stop = layout.lanes
    cols = layout.cols
    lane_count = lane_stop - lane_start
    if ep.tile.shape != (lane_count, cols):
        raise _failure(
            ep,
            "TMEM lane/column shape is not congruent with tile.shape",
            {"layout_shape": [lane_count, cols], "tile_shape": list(ep.tile.shape)},
        )

    lane_stride_bits = (1 << 16) * elem_bits
    col_stride_bits = elem_bits
    origin_bits = _default(layout.origin_bits, 0)
    origin_bits += lane_start * lane_stride_bits
    shape: Tree = (lane_count, cols)
    stride_bits: Tree = (lane_stride_bits, col_stride_bits)
    address = (
        "pl-bit-address-v2",
        "tmem",
        ("shape", shape),
        ("stride_bits", stride_bits),
        ("origin_bits", origin_bits),
        ("swizzle_xor_bits", None),
        ("pointer_flag_bits", None),
        ("ring_stage_stride_bits", ring_stride_bits),
    )
    return address, _cosize_bits(shape, stride_bits, elem_bits)


def canonicalize(
    ep: Endpoint,
    base_claim: Claim | None,
    profile: Profile,
) -> CanonEndpoint:
    """Canonicalize an endpoint layout into a symbolic bit-address function.

    ``shape``/``stride`` trees are checked against the endpoint tile before
    CuTe-style coalescing; strides always scale by ``elem_bits``. Manifest
    swizzles default to pointer-element units (``pointer_flag_bits``, or
    ``elem_bits`` when absent). Explicit ``unit="byte"`` and ``unit="bit"``
    parameters instead index bytes and bits without an element-width scale.
    The unit marker disappears after normalization to bit positions.
    """

    if ep.space == "none":
        depth = ep.ring.depth if ep.ring is not None else None
        identity = depth.handle or f"fixed:{depth.fixed}" if depth is not None else "fixed:1"
        return CanonEndpoint(
            addr_fn=("pl-signal-v2",), elem_bits=0, swizzle_bits=None,
            align_bits=None, footprint_bits=0, ring_geom=(identity, 0),
            space=ep.space, scope=ep.scope, proxy=ep.proxy,
            dtype=ep.tile.dtype, tile_shape=ep.tile.shape, tv_role=None,
        )

    layout = _expand_atom(ep.layout, profile, ep)
    elem_bits = layout.elem_bits
    if ep.space not in {"smem", "tmem", "reg", "gmem"}:
        raise _failure(ep, "unsupported endpoint address space", {"space": ep.space})

    ring_geom, ring_stride_bits = _ring(ep)
    if ep.space == "tmem":
        addr_fn, footprint_bits = _tmem_layout(ep, layout, elem_bits, ring_stride_bits)
        swizzle_bits = None
    else:
        addr_fn, swizzle_bits, footprint_bits = _affine_layout(
            ep,
            layout,
            elem_bits,
            ring_stride_bits,
        )

    align_bits = _alignment_bits(ep, base_claim, profile, elem_bits)
    return CanonEndpoint(
        addr_fn=addr_fn,
        elem_bits=elem_bits,
        swizzle_bits=swizzle_bits,
        align_bits=align_bits,
        footprint_bits=footprint_bits,
        ring_geom=ring_geom,
        space=ep.space,
        scope=ep.scope,
        proxy=ep.proxy,
        dtype=ep.tile.dtype,
        tile_shape=ep.tile.shape,
        tv_role=ep.owner_role if ep.space == "reg" and ep.protocol.kind != "call" else None,
    )


def unify(a: CanonEndpoint, b: CanonEndpoint) -> list[ClauseFailure]:
    """Return L-PAIR compatibility failures, L8 identity failures, or an empty list."""

    compatibility_fields = ("space", "scope", "proxy", "dtype", "tile_shape", "tv_role")
    compatibility_failures = [
        ClauseFailure(
            "L-PAIR",
            f"endpoint.{field}",
            f"endpoint compatibility field {field} differs",
            {"lhs": getattr(a, field), "rhs": getattr(b, field)},
        )
        for field in compatibility_fields
        if getattr(a, field) != getattr(b, field)
    ]
    if a.space == "none" and a.ring_geom != b.ring_geom:
        return [ClauseFailure("L-PAIR", "endpoint.ring", "completion-only ring depths differ", {})]
    if compatibility_failures:
        return compatibility_failures

    physical_fields = ("addr_fn", "elem_bits", "align_bits", "ring_geom")
    differing = [field for field in physical_fields if getattr(a, field) != getattr(b, field)]
    if not differing:
        return []
    return [
        ClauseFailure(
            "L8",
            "endpoint.physical_identity",
            "endpoint physical identities differ",
            {
                "differing_components": differing,
                "lhs": {field: getattr(a, field) for field in differing},
                "rhs": {field: getattr(b, field) for field in differing},
            },
        )
    ]


__all__ = ["AffineLayout", "Layout", "Swizzle", "TmemLayout", "canonicalize", "load_layout", "unify"]
