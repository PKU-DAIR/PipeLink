from dataclasses import replace

import pytest

from pipec2.profile import get_profile
from pipec2.tensor import delivery_layout, expr
from pipec2.tensor.contracts import Binding
from test_tensor_machine import _instance, _profile, _rebuild


def _fixture(*, rows=128, width=128, bases=(0, 128, 256, 384), column=None, dp=None):
    instance = _instance(threads=rows, width=width, tmem=True, profile=get_profile("sm_103a"))
    domain = expr.Domain("base", "int", values=bases)
    binding = Binding("base", "int", domain, "producer", "column", "cta", "tile")
    env = {c.name: c for c in instance.export.identity.coords} | {"base": domain}
    placement = replace(instance.export.placements[0], components={
        "dp": expr.parse(dp or ["coord", "x"], env),
        "column": expr.parse(column or ["add", ["coord", "slot"], ["bind", "base"]], env)})
    return instance, replace(instance.export, placements=(placement,)), (binding,)


def test_tmem_delivery_matches_every_model_receipt_world():
    instance, export, bindings = _fixture()
    participant, position = delivery_layout(export, bindings, instance.profile, range(128))
    assert participant == expr.parse(["coord", "x"], {c.name: c for c in export.identity.coords})
    assert position == expr.parse(["coord", "slot"], {c.name: c for c in export.identity.coords})
    instance = _rebuild(instance, export=export, bindings=bindings)
    for world, initial in zip(instance.worlds, instance.initial_cells):
        base = dict(world)["base"]
        delivered = set()
        for packet in instance.packets:
            if not base <= packet.column < base + 128:
                continue
            for lane, slot, source in packet.receipt:
                identity = initial[source]
                assert identity >= 0
                env = dict(zip(sorted(c.name for c in export.identity.coords), instance.identities[identity]))
                owner = packet.dp + lane
                assert instance.tmem_subpartition(owner) == packet.dp
                assert (expr.evaluate(participant, env), expr.evaluate(position, env)) == (
                    owner, packet.column - base + slot)
                delivered.add(identity)
        assert delivered == set(range(len(instance.identities)))


@pytest.mark.parametrize("changes,participants,mutation", [
    ({"dp": ["add", ["coord", "x"], ["bind", "base"]]}, range(128), None),
    ({}, range(255), None),
    ({}, range(32, 160), None),
    ({}, range(1, 129), None),
    ({}, range(128), "dtype"),
    ({}, range(128), "guard"),
    ({}, range(128), "capability"),
    ({"column": ["add", ["add", ["coord", "slot"], ["const", 1]], ["bind", "base"]]}, range(128), None),
    ({"rows": 16}, range(32), None),
    ({"bases": (0, 1)}, range(128), None),
    ({"width": 17}, range(128), None),
    ({"column": ["add", ["mulc", 2, ["coord", "slot"]], ["bind", "base"]]}, range(128), None),
    ({"column": ["add", ["coord", "slot"], ["mulc", 2, ["add", ["coord", "x"], ["bind", "base"]]]]}, range(128), None),
], ids=("bound-dp", "partial-eight-warps", "shifted-warp-base", "partial-warps", "non32", "guard", "no-receipt",
        "shifted-columns", "partial-dp-slice", "unaligned-world", "partial-packet", "gapped-permutation", "mixed-base"))
def test_unsupported_tmem_delivery_returns_none(changes, participants, mutation):
    instance, export, bindings = _fixture(**changes)
    if mutation == "dtype":
        export = replace(export, identity=replace(export.identity, dtype="f16"))
    elif mutation == "guard":
        export = replace(export, placements=(replace(export.placements[0], guard=expr.Expr("eq", (
            expr.Expr("const", (0,), "int"), expr.Expr("const", (0,), "int")), "pred")),))
    elif mutation == "capability":
        export = replace(export, read_capabilities=())
    assert delivery_layout(export, bindings, instance.profile, participants) is None


def test_permuted_complete_columns_and_dp_are_supported():
    instance, export, bindings = _fixture(
        dp=["sub", ["const", 127], ["coord", "x"]],
        column=["add", ["modc", ["add", ["coord", "slot"], ["const", 16]], 128], ["bind", "base"]])
    participant, position = delivery_layout(export, bindings, instance.profile, range(128))
    for row in range(128):
        for column in range(128):
            env = {"x": row, "slot": column}
            assert expr.evaluate(participant, env) == 127 - row
            assert expr.evaluate(position, env) == (column + 16) % 128


def test_register_delivery_is_the_original_placement():
    instance = _instance(threads=32, width=16)
    placement = instance.export.placements[0]
    assert delivery_layout(instance.export, (), instance.profile, range(32)) == (
        placement.components["participant"], placement.components["position"])


def test_delivery_uses_finite_model_geometry_not_target_name():
    profile = _profile(tmem_lanes=8, tmem_columns=16, warp_size=2, packet_columns=4)
    instance = _instance(threads=8, width=8, tmem=True, profile=profile)
    assert delivery_layout(instance.export, (), profile, range(8)) is not None
    assert delivery_layout(instance.export, (), get_profile("sm_90a"), range(8)) is None


def test_aligned_column_shift_and_complete_dp_subset():
    instance, export, bindings = _fixture(
        rows=32, width=16, bases=(0, 128),
        dp=["add", ["coord", "x"], ["const", 32]],
        column=["add", ["add", ["coord", "slot"], ["const", 16]], ["bind", "base"]])
    participant, position = delivery_layout(export, bindings, instance.profile, range(128))
    for row in range(32):
        for column in range(16):
            env = {"x": row, "slot": column}
            assert expr.evaluate(participant, env) == row + 32
            assert expr.evaluate(position, env) == column + 16


@pytest.mark.parametrize("coordinate", (
    expr.Domain("x", "int", 1, 33),
    expr.Domain("x", "int", values=(0, 1, 2)),
    expr.Domain("x", "int", 0, 0),
))
def test_nonzero_sparse_or_empty_identity_is_not_adopted(coordinate):
    instance = _instance(threads=32, width=16)
    export = replace(instance.export, identity=replace(
        instance.export.identity, coords=(coordinate, instance.export.identity.coords[1])))
    assert delivery_layout(export, (), instance.profile, range(32)) is None


@pytest.mark.parametrize("count,first", [(256, 0), (256, 128), (160, 0), (384, 0)])
def test_whole_warps_share_tmem_quadrants_without_replication(count, first):
    instance, export, bindings = _fixture(width=256, bases=(0, 256))
    participant, position = delivery_layout(export, bindings, instance.profile, range(first, first + count))
    delivered = set()
    for row in range(128):
        copies = (count - 1 - row) // 128 + 1
        for column in range(256):
            env = {"x": row, "slot": column}
            owner = expr.evaluate(participant, env)
            private = expr.evaluate(position, env)
            assert owner == first + row + 128 * ((column // 16) % copies)
            assert private == (column // (16 * copies)) * 16 + column % 16
            assert first <= owner < first + count
            assert owner % 128 == row
            assert (owner, private) not in delivered
            delivered.add((owner, private))
    assert len(delivered) == 128 * 256


def test_eight_warp_delivery_gives_sixteen_values_per_column_chunk():
    instance, export, bindings = _fixture(width=256, bases=(0, 256))
    participant, position = delivery_layout(export, bindings, instance.profile, range(256))
    for chunk in range(8):
        actual = {(expr.evaluate(participant, {"x": row, "slot": column}),
                   expr.evaluate(position, {"x": row, "slot": column}) - 16 * chunk)
                  for row in range(128) for column in range(32 * chunk, 32 * chunk + 32)}
        assert actual == {(thread, slot) for thread in range(256) for slot in range(16)}
