"""CPU-only proofs of exact contract/captured-layout conversion."""
import importlib.util
from pathlib import Path

import pytest
from pipec2.layout import load_layout
from pipec2.profile import get_profile
from tilelang import layout as tl_layout, tvm
from tilelang.layout.cute import ComposedLayout

from pipelink_tilelang.capture import capture
from pipelink_tilelang.layout import (
    LayoutUnsupported, from_captured, swizzle_atom, tmem_shape, to_tilelang,
)


PROFILES = ("sm_103a", "sm_90a", "sm_120a")
# The implemented SW128 K-major atom under each profile's name.
SW128_ATOMS = {"sm_103a": "sw128_k_bf16", "sm_90a": "sw128_k_bf16", "sm_120a": "sw128_k_bf16"}
# Layouts without a profile atom expand identically under every profile.
_ANY_PROFILE = PROFILES[0]


@pytest.fixture(params=PROFILES)
def profile(request):
    return request.param


def _buffer(shape, dtype="bfloat16"):
    return tvm.tirx.decl_buffer(shape, dtype, scope="shared")


def _affine(shape=(128, 64), stride=(64, 1), bits=16, **fields):
    return {"encoding": "pl-layout-v2", "shape": list(shape), "stride": list(stride),
            "elem_bits": bits, **fields}


@pytest.fixture(scope="module")
def d1_captures():
    spec = importlib.util.spec_from_file_location("layout_test_recipe", Path(__file__).resolve().parent / "fixtures/ws_gemm.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    captures = {}
    for name in PROFILES:
        target = tvm.target.Target({"kind": "cuda", "arch": name})
        result = capture(recipe.program(3), target=target, pass_configs=recipe.PASS_CONFIGS)
        buffers = {}

        def visit(node):
            if isinstance(node, tvm.tirx.SBlock):
                buffers.update((buffer.name, buffer) for buffer in node.annotations.get("layout_map", {}))

        for func in result.modules["post_layout"].functions.values():
            tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
        captures[name] = result, buffers
    return captures


@pytest.fixture
def d1(profile, d1_captures):
    return d1_captures[profile]


def test_implemented_profile_atom_round_trip(profile):
    value = load_layout({"atom": SW128_ATOMS[profile]}, "smem", "layout")
    buffer = _buffer((8, 64))
    converted = to_tilelang(value, buffer, profile=profile)
    assert tvm.ir.structural_equal(converted, tl_layout.make_full_bank_swizzled_layout(buffer))
    captured = from_captured(converted, elem_bits=16, buffer=buffer)
    assert swizzle_atom(value, profile=profile) == swizzle_atom(captured, profile=profile) == ("128B", True)
    assert captured == _affine((8, 64), (64, 1), swizzle={"B": 3, "M": 3, "S": 3})


@pytest.mark.parametrize("profile,atom", [(name, atom) for name in PROFILES
                                          for atom in get_profile(name).atoms if atom != SW128_ATOMS[name]])
def test_unrepresentable_profile_atoms_fail_closed(profile, atom):
    reason = "MN-major" if atom.endswith("sw64_mn_bf16") else "NotImplemented"
    with pytest.raises(LayoutUnsupported, match=reason):
        to_tilelang({"atom": atom}, _buffer((32, 8)), profile=profile)


@pytest.mark.parametrize("bank,constructor", [
    (1, tl_layout.make_quarter_bank_swizzled_layout),
    (2, tl_layout.make_half_bank_swizzled_layout),
    (3, tl_layout.make_full_bank_swizzled_layout),
])
@pytest.mark.parametrize("bits,dtype", [(8, "int8"), (16, "bfloat16"), (32, "float32"), (64, "float64")])
def test_bank_layout_coordinates_round_trip(bank, constructor, bits, dtype):
    atom = (128 // bits) << bank
    # Two atom columns expose the block-major map that a row-major shortcut loses.
    buffer = _buffer((3, 16, atom * 2), dtype)
    original = constructor(buffer)
    value = from_captured(original, elem_bits=bits, buffer=buffer)
    assert value["shape"] == [16, [atom, 2]]
    assert value["stride"] == [atom, [1, 16 * atom]]
    assert value["ring_stride_elements"] == 16 * atom * 2
    assert swizzle_atom(value, profile=_ANY_PROFILE) == (f"{16 << bank}B", True)
    assert tvm.ir.structural_equal(to_tilelang(value, buffer, profile=_ANY_PROFILE), original)
    for slot in (0, 2):
        for row in (0, 7, 8, 15):
            for col in (0, atom - 1, atom, 2 * atom - 1):
                base = row * atom + col % atom + col // atom * 16 * atom
                mask = ((1 << bank) - 1) << value["swizzle"]["M"]
                expected = slot * value["ring_stride_elements"] + (base ^ ((base >> 3) & mask))
                actual = 0
                for extent, index in zip(original.get_output_shape(), original.map_forward_index([slot, row, col])):
                    actual = actual * int(extent) + int(index)
                assert actual == expected


def test_real_d1_capture_and_manifest_layout_select_128b(profile, d1):
    result, buffers = d1
    for name in ("A_shared", "B_shared"):
        buffer = buffers[name]
        value = from_captured(result.layout_objects[name], elem_bits=16, buffer=buffer)
        assert value == _affine(swizzle={"B": 3, "M": 3, "S": 3}, ring_stride_elements=8192)
        assert swizzle_atom(value, profile=profile) == ("128B", True)
        # Exactly the schema fields emitted by ws.py (ring stride belongs to ring).
        manifest_layout = load_layout({key: item for key, item in value.items() if key != "ring_stride_elements"},
                                      "smem", "endpoints.a_ring.layout")
        converted = to_tilelang(manifest_layout, buffer, profile=profile)
        assert tvm.ir.structural_equal(converted, tl_layout.make_full_bank_swizzled_layout(buffer))
        # This invokes the real tl.cute.swizzle_to_swizzle_mode FFI on the
        # converted live layout, not on our input swizzle tuple.
        composed = ComposedLayout.from_tilelang(converted, buffer)
        assert composed.swizzle.to_swizzle_mode().is_swizzle_128b()


@pytest.mark.parametrize("name,mma", [("sm_90a", "wgmma"), ("sm_120a", "mma_sync")])
def test_d1_capture_keeps_the_ring_contract_and_switches_the_mma_body(d1_captures, name, mma):
    """The captured channel contract is target-independent; only the opaque consumer body changes."""
    (sm103, sm103_buffers), (other, other_buffers) = d1_captures["sm_103a"], d1_captures[name]
    assert "tcgen05" not in other.kernel_source and "copy_sm100.h" not in other.kernel_source
    assert mma in other.kernel_source and "wgmma" not in sm103.kernel_source
    assert (name == "sm_90a") == ("wgmma" in other.kernel_source)
    for buffer in ("A_shared", "B_shared"):
        assert str(other.layout_objects[buffer]) == str(sm103.layout_objects[buffer])
        assert tvm.ir.structural_equal(other.layout_objects[buffer], sm103.layout_objects[buffer])
        assert (from_captured(other.layout_objects[buffer], elem_bits=16, buffer=other_buffers[buffer])
                == from_captured(sm103.layout_objects[buffer], elem_bits=16, buffer=sm103_buffers[buffer]))
    assert other.facts.keys() == sm103.facts.keys()


def test_numerically_equal_but_structurally_different_map_is_rejected():
    buffer = _buffer((128, 64))
    named = tl_layout.make_full_bank_swizzled_layout(buffer)
    custom = tl_layout.Layout((128, 64), lambda i, j: i * 64 + (j // 8 ^ i % 8) * 8 + j % 8)
    assert not tvm.ir.structural_equal(custom, named)
    for i in (0, 1, 7, 8, 127):
        for j in (0, 7, 8, 31, 32, 63):
            offsets = []
            for layout in (named, custom):
                offset = 0
                for extent, index in zip(layout.get_output_shape(), layout.map_forward_index([i, j])):
                    offset = offset * int(extent) + int(index)
                offsets.append(offset)
            assert offsets[0] == offsets[1]
    with pytest.raises(LayoutUnsupported, match="StructuralEqual"):
        from_captured(custom, elem_bits=16, buffer=buffer)


@pytest.mark.parametrize("unit,m", [("element", 3), ("byte", 4), ("bit", 7)])
def test_swizzle_unit_conversion(unit, m):
    buffer = _buffer((128, 64))
    converted = to_tilelang(_affine(swizzle={"B": 3, "M": m, "S": 3, "unit": unit}), buffer, profile=_ANY_PROFILE)
    assert tvm.ir.structural_equal(converted, tl_layout.make_full_bank_swizzled_layout(buffer))


@pytest.mark.parametrize("fields,reason", [
    ({"origin_bits": 8}, "origin_bits"),
    ({"pointer_flag_bits": 16}, "pointer_flag_bits"),
    ({"swizzle": {"B": 3, "M": 0, "S": 3, "unit": "byte"}}, "element boundary"),
    ({"swizzle": {"B": 2, "M": 3, "S": 2}}, "unsupported element-unit"),
    ({"elem_bits": 32}, "buffer dtype"),
    ({"ring_stride_elements": 16384}, "ring stride"),
])
def test_unsupported_layout_semantics_are_not_approximated(fields, reason):
    with pytest.raises(LayoutUnsupported, match=reason):
        to_tilelang(_affine(**fields), _buffer((3, 128, 64)), profile=_ANY_PROFILE)


def test_wide_swizzle_cannot_masquerade_as_row_major():
    with pytest.raises(LayoutUnsupported, match="affine coordinates"):
        to_tilelang(_affine((128, 128), (128, 1), swizzle={"B": 3, "M": 3, "S": 3}), _buffer((128, 128)), profile=_ANY_PROFILE)


def test_truncated_four_row_period_cannot_cross_atom_columns():
    buffer = _buffer((4, 128))
    original = tl_layout.make_full_bank_swizzled_layout(buffer)
    with pytest.raises(LayoutUnsupported, match="truncated four-row"):
        from_captured(original, elem_bits=16, buffer=buffer)


def test_linear_and_tmem_have_separate_representations():
    value = _affine((4,), (1,), bits=32)
    buffer = _buffer((4,), "uint32")
    assert swizzle_atom(value, profile=_ANY_PROFILE) == ("NONE", True)
    assert from_captured(to_tilelang(value, buffer, profile=_ANY_PROFILE), elem_bits=32, buffer=buffer) == value
    tmem = {"encoding": "pl-layout-v2", "lanes": [0, 128], "cols": 512, "elem_bits": 32}
    assert tmem_shape(load_layout(tmem, "tmem", "layout"), profile="sm_103a") == (128, 512)
    assert swizzle_atom(tmem, profile="sm_103a") is None
    with pytest.raises(LayoutUnsupported, match="zero-origin"):
        tmem_shape({**tmem, "lanes": [32, 128]}, profile="sm_103a")
