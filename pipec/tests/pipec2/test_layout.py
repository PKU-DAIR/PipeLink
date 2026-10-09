from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pipec2 import container
from pipec2.layout import canonicalize, load_layout, unify
from pipec2.model import (
    Claim,
    Endpoint,
    Extent,
    LayoutError,
    SchemaError,
    Protocol,
    Ring,
    Tile,
)
from pipec2.profile import get_profile

ROOT = Path(__file__).resolve().parents[3]


def built_manifests() -> list[Path]:
    """Locally built example/campaign packages; unbuilt artifacts contribute no cases."""
    return sorted([
        *ROOT.glob("campaigns/attention/out/packaged/*/*.plink"),
        *ROOT.glob("campaigns/megakernel/port/build/artifacts/*.plink"),
        *ROOT.glob("examples/profile-consumer/build/sm_103a/artifacts/*.plink"),
    ])


def _protocol() -> Protocol:
    return Protocol("none", None, None, None, None, None, None, None)


def _endpoint(
    layout: dict[str, Any],
    *,
    name: str = "tile",
    direction: str = "produce",
    owner_role: str = "load",
    space: str = "smem",
    dtype: str = "bf16",
    tile_shape: tuple[int, ...] = (8, 8),
    ring: Ring | None = None,
) -> Endpoint:
    return Endpoint(
        name=name,
        direction=direction,
        owner_role=owner_role,
        space=space,
        scope="cta",
        proxy="generic",
        tile=Tile(dtype, tile_shape),
        layout=load_layout(layout, space, "$.layout"),
        ring=ring,
        protocol=_protocol(),
        binds={},
    )


def _base_claim(align: int, *, domain: str = "smem") -> Claim:
    return Claim(
        domain=domain,
        handle="base",
        lifetime="kernel",
        extent=Extent(1, {}, {}),
        align=align,
        offset=None,
        roles=None,
        reg_group=None,
        values=None,
        absolute_base=None,
        alias=None,
    )


def _addr_parts(canon: Any) -> dict[str, Any]:
    return dict(canon.addr_fn[2:])


def _flatten(tree: Any) -> list[int]:
    if isinstance(tree, tuple):
        return [leaf for item in tree for leaf in _flatten(item)]
    return [tree]


def _bit_address_at(canon: Any, linear_index: int) -> int:
    """Evaluate the symbolic address at a CuTe first-mode-fastest index."""
    parts = _addr_parts(canon)
    bits = parts["origin_bits"]
    for extent, stride in zip(_flatten(parts["shape"]), _flatten(parts["stride_bits"]), strict=True):
        linear_index, coordinate = divmod(linear_index, extent)
        bits += coordinate * stride
    assert linear_index == 0
    swizzle = parts["swizzle_xor_bits"]
    if swizzle is not None:
        b, m, s = swizzle
        source = bits & (((1 << b) - 1) << (m + max(s, 0)))
        bits ^= source >> s if s >= 0 else source << -s
    return bits


@pytest.mark.parametrize("elem_bits", [4, 16])
def test_swizzle_units_preserve_cute_byte_pointer_addresses(elem_bits: int) -> None:
    profile = get_profile("sm_103a")
    columns = 1024 // elem_bits
    base = {
        "encoding": "pl-layout-v2",
        "elem_bits": elem_bits,
        "shape": [8, columns],
        "stride": [columns, 1],
    }
    swizzles = [
        {"B": 3, "M": 7 - (elem_bits.bit_length() - 1), "S": 3},
        {"B": 3, "M": 4, "S": 3, "unit": "byte"},
        {"B": 3, "M": 7, "S": 3, "unit": "bit"},
    ]
    for swizzle in swizzles:
        canon = canonicalize(
            _endpoint({**base, "swizzle": swizzle}, tile_shape=(8, columns)),
            None,
            profile,
        )
        for m in range(8):
            for n in range(columns):
                byte, subbyte_bits = divmod(elem_bits * (columns * m + n), 8)
                # cute/swizzle.hpp:67-78 and pointer_swizzle.hpp:86-104:
                # Swizzle<3,4,3> acts on byte addresses, not element offsets.
                expected = 8 * (byte ^ ((byte & 896) >> 3)) + subbyte_bits
                assert _bit_address_at(canon, m + 8 * n) == expected


@pytest.mark.parametrize(("atom", "shape", "stride", "byte_mask"), [
    ("sw128_k_bf16", (8, 64), (64, 1), 896),
    ("sw64_mn_bf16", (32, 8), (1, 32), 384),
])
def test_profile_atoms_match_cute_byte_addresses(atom, shape, stride, byte_mask) -> None:
    canon = canonicalize(
        _endpoint({"atom": atom}, tile_shape=shape), None, get_profile("sm_103a")
    )
    # SM100 reuses SM90's atoms (mma_traits_sm100.hpp:64-71).
    # mma_traits_sm90_gmma.hpp:77,84,92,104 defines the bf16 affine maps
    # plus byte Swizzle<2,4,3>/<3,4,3>. pointer_flagged.hpp:55-75,108-114
    # preserves those byte parameters when upcasting, then recasts the swizzle
    # for position-independent element indexing. For bf16 this is M=3.
    for m in range(shape[0]):
        for n in range(shape[1]):
            byte = 2 * (m * stride[0] + n * stride[1])
            cute_byte = byte ^ ((byte & byte_mask) >> 3)
            assert _bit_address_at(canon, m + shape[0] * n) == 8 * cute_byte, (atom, m, n)


def test_sub_byte_layout_keeps_half_byte_address_identity() -> None:
    layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 4,
        "shape": [2, 4],
        "stride": [4, 1],
        "swizzle": {"B": 1, "M": 1, "S": 1},
    }
    canon = canonicalize(
        _endpoint(layout, dtype="s4", tile_shape=(2, 4)),
        None,
        get_profile("sm_103a"),
    )
    parts = _addr_parts(canon)
    assert canon.elem_bits == 4
    assert canon.swizzle_bits == (1, 3, 1)
    assert parts["stride_bits"] == (16, 4)
    assert canon.footprint_bits == 32
    assert canon.align_bits is None


def test_tmem_lane_column_form_is_a_bit_address() -> None:
    layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 32,
        "lanes": [0, 128],
        "cols": 256,
    }
    canon = canonicalize(
        _endpoint(
            layout,
            space="tmem",
            owner_role="mma",
            dtype="f32",
            tile_shape=(128, 256),
        ),
        None,
        get_profile("sm_103a"),
    )
    parts = _addr_parts(canon)
    assert parts["shape"] == (128, 256)
    assert parts["stride_bits"] == ((1 << 16) * 32, 32)
    assert canon.footprint_bits == ((127 << 16) + 256) * 32
    assert canon.swizzle_bits is None
    assert canon.align_bits is None


def test_gmem_form_uses_the_same_bit_stride_algebra() -> None:
    layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 16,
        "shape": [4, 8],
        "stride": [8, 1],
    }
    canon = canonicalize(
        _endpoint(
            layout,
            space="gmem",
            dtype="bf16",
            tile_shape=(4, 8),
        ),
        None,
        get_profile("sm_103a"),
    )
    parts = _addr_parts(canon)
    assert canon.space == "gmem"
    assert parts["stride_bits"] == (128, 16)
    assert parts["swizzle_xor_bits"] is None
    assert parts["pointer_flag_bits"] is None
    assert canon.footprint_bits == 4 * 8 * 16


def test_reg_tv_layout_rejects_cross_role_pairing() -> None:
    layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 32,
        "shape": [[4], [4]],
        "stride": [[4], [1]],
    }
    profile = get_profile("sm_103a")
    producer = canonicalize(
        _endpoint(
            layout,
            space="reg",
            owner_role="mma_wg0",
            dtype="f32",
            tile_shape=(4, 4),
        ),
        None,
        profile,
    )
    consumer = canonicalize(
        _endpoint(
            layout,
            direction="consume",
            space="reg",
            owner_role="mma_wg1",
            dtype="f32",
            tile_shape=(4, 4),
        ),
        None,
        profile,
    )
    failures = unify(producer, consumer)
    assert [(failure.clause_id, failure.path) for failure in failures] == [
        ("L-PAIR", "endpoint.tv_role")
    ]


def test_dtype_mismatch_rejects_before_equal_width_address_identity() -> None:
    layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 16,
        "shape": [8, 8],
        "stride": [8, 1],
    }
    profile = get_profile("sm_103a")
    bf16 = canonicalize(_endpoint(layout, dtype="bf16"), None, profile)
    fp16 = canonicalize(
        _endpoint(layout, direction="consume", dtype="fp16"),
        None,
        profile,
    )
    assert bf16.addr_fn == fp16.addr_fn
    failures = unify(bf16, fp16)
    assert [(failure.clause_id, failure.path) for failure in failures] == [
        ("L-PAIR", "endpoint.dtype")
    ]


def test_unimplemented_atom_names_the_atom() -> None:
    profile = get_profile("sm_103a")
    sentinel = _endpoint(
        {"encoding": "pl-layout-v2", "atom": "tma_interleave_bf16"},
        tile_shape=(8, 64),
    )
    with pytest.raises(LayoutError, match="tma_interleave_bf16") as caught:
        canonicalize(sentinel, None, profile)
    assert [failure.clause_id for failure in caught.value.failures] == ["L8"]


def test_tile_congruence_is_checked_before_coalescing() -> None:
    layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 16,
        "shape": [4, 4],
        "stride": [1, 4],
    }
    with pytest.raises(LayoutError, match="not congruent") as caught:
        canonicalize(
            _endpoint(layout, tile_shape=(16,)),
            None,
            get_profile("sm_103a"),
        )
    assert caught.value.failures[0].path == "endpoints.tile.layout"
    assert caught.value.failures[0].clause_id == "L8"


def test_claim_alignment_is_converted_from_bytes_to_bits() -> None:
    layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 16,
        "shape": [8, 8],
        "stride": [8, 1],
    }
    canon = canonicalize(
        _endpoint(layout),
        _base_claim(1024),
        get_profile("sm_103a"),
    )
    assert canon.align_bits == 8192


@pytest.mark.parametrize("path", built_manifests(), ids=lambda path: path.relative_to(ROOT).as_posix())
def test_inventory_layout_json_is_lossless(path: Path) -> None:
    raw = container.read(path)[0]
    for index, endpoint in enumerate(raw["endpoints"]):
        if endpoint["space"] == "none":
            assert "layout" not in endpoint  # completion-only endpoints carry no data layout
            continue
        layout = endpoint["layout"]
        assert load_layout(layout, endpoint["space"], f"$.endpoints[{index}].layout").to_json_obj() == layout


@pytest.mark.parametrize("layout", [
    {"encoding": "pl-layout-v2", "atom": "sw128_k_bf16"},
    {"atom": "sw128_k_bf16", "shape": [[8], [64]], "swizzle": None},
    {"atom": None, "encoding": "pl-layout-v2", "elem_bits": 16, "shape": [[2], [4]], "stride": [[4], [1]]},
    {"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [[2], [4]], "stride": [[4], [1]], "swizzle": None, "pointer_flag_bits": None, "origin_bits": 0},
])
def test_authored_atom_overrides_and_explicit_nulls_are_lossless(layout: dict) -> None:
    assert load_layout(layout, "smem", "$.layout").to_json_obj() == layout


@pytest.mark.parametrize(("update", "path"), [
    ({"shape": []}, "$.layout.shape"),
    ({"stride": [[4], [True]]}, "$.layout.stride[1][0]"),
    ({"shape": [[2], 4]}, "$.layout"),
    ({"elem_bits": 0}, "$.layout.elem_bits"),
    ({"origin_bits": -1}, "$.layout.origin_bits"),
    ({"swizzle": {"B": 1, "M": 0, "S": 1, "unit": "word"}}, "$.layout.swizzle.unit"),
    ({"encoding": "other"}, "$.layout.encoding"),
    ({"opaque_extension": {}}, "$.layout"),
    ({"swizzle": {"B": 1, "M": 0, "S": True}}, "$.layout.swizzle.S"),
])
def test_layout_loader_rejects_syntax_at_the_authored_path(update: dict, path: str) -> None:
    layout = {"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [[2], [4]], "stride": [[4], [1]], **update}
    with pytest.raises(SchemaError) as caught:
        load_layout(layout, "smem", "$.layout")
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [("L8", path)]


@pytest.mark.parametrize("layout", [
    {"atom": "unknown"},
    {"atom": "sw128_k_bf16", "elem_bits": 8},
    {"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [8, 8], "stride": [8, 1], "pointer_flag_bits": None},
    {"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [8, 8], "stride": [8, 1], "pointer_flag_bits": 24, "swizzle": {"B": 3, "M": 4, "S": 3}},
])
def test_contextual_layout_errors_are_not_schema_errors(layout: dict) -> None:
    endpoint = _endpoint(layout)
    with pytest.raises(LayoutError) as caught:
        canonicalize(endpoint, None, get_profile("sm_103a"))
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [("L8", "endpoints.tile.layout")]


@pytest.mark.parametrize(("layout", "space", "path"), [
    ([], "smem", "$.layout"),
    ({"encoding": "pl-layout-v2", "elem_bits": 8, "shape": [1]}, "smem", "$.layout.stride"),
    ({"encoding": "pl-layout-v2", "elem_bits": 32, "lanes": [3, 3], "cols": 1}, "tmem", "$.layout.lanes"),
    ({"encoding": "pl-layout-v2", "elem_bits": 32, "lanes": [0, 32], "cols": 0}, "tmem", "$.layout.cols"),
])
def test_layout_shape_required_fields_and_tmem_ranges_are_syntax(layout, space, path) -> None:
    with pytest.raises(SchemaError) as caught:
        load_layout(layout, space, "$.layout")
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [("L8", path)]


def test_atom_partial_override_completeness_uses_profile_context() -> None:
    authored = {"atom": "partial", "stride": [[8], [1]]}
    endpoint = _endpoint(authored)
    inline = {"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [[8], [8]], "stride": [[8], [1]]}
    profile = replace(get_profile("sm_103a"), atoms={
        "partial": {key: value for key, value in inline.items() if key != "stride"},
    })
    assert canonicalize(endpoint, None, profile) == canonicalize(_endpoint(inline), None, profile)
    incomplete = _endpoint({"atom": "partial"})
    with pytest.raises(LayoutError) as caught:
        canonicalize(incomplete, None, profile)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [("L8", "endpoints.tile.layout")]


def test_swizzle_space_restriction_remains_contextual() -> None:
    endpoint = _endpoint({
        "encoding": "pl-layout-v2", "elem_bits": 16, "shape": [8, 8],
        "stride": [8, 1], "swizzle": None,
    }, space="gmem")
    with pytest.raises(LayoutError) as caught:
        canonicalize(endpoint, None, get_profile("sm_103a"))
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [("L8", "endpoints.tile.layout")]
