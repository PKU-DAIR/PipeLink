"""Exact pl-layout-v2 conversion for TileLang's structurally recognized layouts.

Bank constructors tile the contiguous axis into 32/64/128-byte atoms; wider
planes therefore have a split shape/stride mode, not a row-major affine map.
The pinned tcgen05 constructor ignores ``k_major`` and cannot represent a
column-major contract. Such contracts fail closed rather than silently transpose.
"""
from math import prod

from pipec2.layout import Layout as ContractLayout
from pipec2.profile import get_profile
from tilelang import layout as tl_layout, tvm


class LayoutUnsupported(ValueError):
    """The contract has no exact, lowering-recognized TileLang representation."""


_BANKS = {
    1: tl_layout.make_quarter_bank_swizzled_layout,
    2: tl_layout.make_half_bank_swizzled_layout,
    3: tl_layout.make_full_bank_swizzled_layout,
}


def _require(condition, reason):
    if not condition:
        raise LayoutUnsupported(reason)


def _expanded(layout, profile):
    _require(isinstance(layout, (ContractLayout, dict)), "expected a layout model or manifest object")
    value = layout.to_json_obj() if isinstance(layout, ContractLayout) else dict(layout)
    atom = value.pop("atom", None)
    if atom is not None:
        _require(isinstance(atom, str), "profile atom must be a name")
        expansion = get_profile(profile).atoms.get(atom)
        _require(expansion is not None, f"unknown {profile} atom {atom!r}")
        _require(expansion.get("expansion") != "NotImplemented", f"profile atom {atom!r} is NotImplemented")
        for key in value.keys() & expansion.keys():
            _require(value[key] == expansion[key], f"{key} conflicts with profile atom {atom!r}")
        value = {**expansion, **value}
    _require(value.get("encoding") == "pl-layout-v2", "expected pl-layout-v2 encoding")
    _require(value.get("origin_bits", 0) == 0, "nonzero origin_bits is unsupported")
    _require(value.get("pointer_flag_bits", 0) == 0, "nonzero pointer_flag_bits is unsupported")
    bits = value.get("elem_bits")
    _require(type(bits) is int and bits > 0 and bits & (bits - 1) == 0,
             "elem_bits must be a positive power of two")
    return value


def _mode(shape, stride):
    """Normalize a logical mode's least-significant-first factorization."""
    if isinstance(shape, (list, tuple)):
        _require(isinstance(stride, (list, tuple)) and len(shape) == len(stride) and bool(shape),
                 "shape and stride tree profiles differ")
        leaves = [leaf for s, d in zip(shape, stride) for leaf in _mode(s, d)]
    else:
        _require(type(shape) is int and shape > 0 and type(stride) is int and stride >= 0,
                 "shape and stride leaves must be positive extents and nonnegative strides")
        leaves = [(shape, stride)]
    merged = []
    for extent, step in leaves:
        if extent == 1:
            continue
        if merged and step == merged[-1][0] * merged[-1][1]:
            merged[-1] = (merged[-1][0] * extent, merged[-1][1])
        else:
            merged.append((extent, step))
    return merged


def _modes(value):
    shape, stride = value.get("shape"), value.get("stride")
    if type(shape) is int:
        shape, stride = [shape], [stride]
    _require(isinstance(shape, (list, tuple)) and isinstance(stride, (list, tuple))
             and len(shape) == len(stride) and len(shape) in (1, 2),
             "only one- or two-dimensional logical planes are supported")
    return [_mode(s, d) for s, d in zip(shape, stride)]


def _shape(modes):
    return tuple(prod(extent for extent, _ in mode) for mode in modes)


def _swizzle(value):
    swizzle = value.get("swizzle")
    if swizzle is None:
        return 0
    _require(isinstance(swizzle, dict) and not set(swizzle) - {"B", "M", "S", "unit"},
             "unsupported swizzle fields")
    b, m, s = (swizzle.get(key) for key in ("B", "M", "S"))
    _require(all(type(number) is int for number in (b, m, s)), "swizzle requires integer B, M, S")
    unit = swizzle.get("unit", "element")
    _require(unit in ("element", "byte", "bit"), f"unsupported swizzle unit {unit!r}")
    bits = value["elem_bits"]
    shift = {"element": bits.bit_length() - 1, "byte": 3, "bit": 0}[unit]
    m += shift - (bits.bit_length() - 1)
    _require(m >= 0, "swizzle is not aligned to an element boundary")
    _require(b in _BANKS and m == 7 - (bits.bit_length() - 1) and s == 3,
             f"unsupported element-unit swizzle ({b}, {m}, {s}) for {bits}-bit elements")
    return b


def _plane(shape, bits, bank):
    value = {"encoding": "pl-layout-v2", "elem_bits": bits, "shape": list(shape)}
    if bank:
        rows, cols = shape
        atom = (128 // bits) << bank
        _require((rows == 4 or rows % 8 == 0) and cols % atom == 0,
                 f"{16 << bank}B bank swizzle requires 4 or a multiple of 8 rows and an atom-aligned contiguous extent")
        _require(rows != 4 or cols == atom,
                 "a truncated four-row swizzle with multiple atom columns has no matching affine XOR map")
        if cols == atom:
            value["stride"] = [atom, 1]
        else:
            value["shape"] = [rows, [atom, cols // atom]]
            value["stride"] = [atom, [1, rows * atom]]
        value["swizzle"] = {"B": bank, "M": 7 - (bits.bit_length() - 1), "S": 3}
    else:
        value["stride"] = [prod(shape[index + 1:]) for index in range(len(shape))]
    return value


def _affine(value):
    _require(not {"lanes", "cols"} & value.keys(), "TMEM layouts use tmem_shape, not to_tilelang")
    _require(not set(value) - {"encoding", "elem_bits", "shape", "stride", "swizzle", "origin_bits",
                               "pointer_flag_bits", "ring_stride_elements"}, "unsupported affine layout fields")
    modes = _modes(value)
    shape = _shape(modes)
    bank = _swizzle(value)
    bits = value["elem_bits"]
    _require(not bank or bits in (8, 16, 32, 64), "bank swizzles support scalar 8/16/32/64-bit elements")
    _require(not bank or len(shape) == 2, "bank swizzles require a two-dimensional plane")
    if len(shape) == 2 and modes == _modes({"shape": list(shape), "stride": [1, shape[0]]}):
        _require(modes == _modes({"shape": list(shape), "stride": [shape[1], 1]}),
                 "MN-major/column-major layouts are unsupported: TileLang's pinned constructor ignores k_major")
    expected = _plane(shape, bits, bank)
    _require(modes == _modes(expected), "affine coordinates do not match TileLang's bank-tiled or linear map")
    return shape, bank


def _buffer(buffer):
    dtype = tvm.DataType(buffer.dtype)
    _require(dtype.lanes == 1, "vector-element buffers are unsupported")
    _require(all(isinstance(extent, tvm.tirx.IntImm) and int(extent) > 0 for extent in buffer.shape),
             "buffer extents must be positive compile-time constants")
    _require(not buffer.strides, "explicit buffer strides are unsupported")
    _require(isinstance(buffer.elem_offset, tvm.tirx.IntImm) and int(buffer.elem_offset) == 0,
             "nonzero buffer element offset is unsupported")
    return tuple(int(extent) for extent in buffer.shape), dtype.bits


def to_tilelang(layout, buffer, *, profile):
    """Use TileLang's own constructor, or reject the contract without approximation."""
    value = _expanded(layout, profile)
    shape, bank = _affine(value)
    buffer_shape, bits = _buffer(buffer)
    _require(bits == value["elem_bits"], "buffer dtype differs from layout.elem_bits")
    _require(buffer_shape == shape or len(buffer_shape) == len(shape) + 1 and buffer_shape[1:] == shape,
             "buffer shape must be the logical plane with an optional leading ring dimension")
    if "ring_stride_elements" in value:
        _require(len(buffer_shape) == len(shape) + 1 and value["ring_stride_elements"] == prod(shape),
                 "ring stride differs from TileLang's compact plane footprint")
    return _BANKS[bank](buffer) if bank else tl_layout.make_linear_layout(buffer)


def from_captured(layout_object, *, elem_bits: int, buffer) -> dict:
    """Recover only structurally recognized maps; rank three denotes a plane ring.

    ``ring_stride_elements`` is metadata for the endpoint ring, not a layout-schema
    field. Callers emitting manifests put it in the ring declaration separately.
    """
    shape, bits = _buffer(buffer)
    _require(bits == elem_bits, "captured buffer dtype differs from elem_bits")
    _require(len(shape) in (1, 2, 3), "captured buffer must be a plane or a leading-ring plane")
    plane_shape = shape[1:] if len(shape) == 3 else shape
    candidates = [0]
    if len(plane_shape) == 2 and bits in (8, 16, 32, 64):
        rows, cols = plane_shape
        if rows == 4 or rows % 8 == 0:
            candidates += [bank for bank in _BANKS if cols % ((128 // bits) << bank) == 0]
    for bank in candidates:
        candidate = _BANKS[bank](buffer) if bank else tl_layout.make_linear_layout(buffer)
        if tvm.ir.structural_equal(layout_object, candidate):
            value = _plane(plane_shape, bits, bank)
            if len(shape) == 3:
                value["ring_stride_elements"] = prod(int(extent) for extent in candidate.get_output_shape()[1:])
            return value
    raise LayoutUnsupported("captured layout is not StructuralEqual to a recognized TileLang constructor")


def tmem_shape(layout, *, profile) -> tuple[int, int]:
    """Return the zero-origin TMEM lane/column allocation shape."""
    value = _expanded(layout, profile)
    _require(not set(value) - {"encoding", "elem_bits", "origin_bits", "lanes", "cols"},
             "unsupported TMEM layout fields")
    lanes, cols = value.get("lanes"), value.get("cols")
    _require(isinstance(lanes, (list, tuple)) and len(lanes) == 2
             and all(type(lane) is int for lane in lanes) and lanes[0] == 0 and 0 < lanes[1] <= 128,
             "TMEM lanes must be a zero-origin nonempty range within 128 lanes")
    _require(type(cols) is int and 0 < cols <= 512, "TMEM columns must be within 1..512")
    return lanes[1], cols


def swizzle_atom(layout, *, profile) -> tuple[str, bool] | None:
    """Classify a supported affine layout; TMEM has no shared-memory swizzle."""
    value = _expanded(layout, profile)
    if "lanes" in value:
        tmem_shape(value, profile=profile)
        return None
    _, bank = _affine(value)
    return (f"{16 << bank}B" if bank else "NONE", True)
