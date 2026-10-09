from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, NoReturn

from .decls import declaration_digest, normalize_declaration
from .stubgen import call_symbol, replace_declarator_identifier
from .model import (
    BarrierSide,
    CanonEndpoint,
    ClauseFailure,
    Endpoint,
    Export,
    Init,
    LinkError,
    ProtocolExpansion,
    SharedMember,
    SolveError,
    Solution,
    StageManifest,
    TxSum,
)


def _reject(path: str, message: str, **detail: Any) -> NoReturn:
    raise SolveError([ClauseFailure("L-PROTO", path, message, detail)])




def _canon_for(
    endpoint: Endpoint,
    pairs: dict[str, dict[str, tuple[str, Endpoint]]],
    canon: dict[str, CanonEndpoint],
) -> CanonEndpoint:
    stage_id, _ = pairs[endpoint.name][endpoint.direction]
    key = f"{stage_id}:{endpoint.name}:{endpoint.direction}"
    if key not in canon:
        _reject(
            f"channels.{endpoint.name}.{endpoint.direction}.layout",
            "canonical endpoint entry is missing",
            expected_key=key,
        )
    return canon[key]


def _tile_bytes(
    endpoint: Endpoint,
    pairs: dict[str, dict[str, tuple[str, Endpoint]]],
    canon: dict[str, CanonEndpoint],
) -> int:
    if endpoint.space == "none":
        return 0
    footprint = _canon_for(endpoint, pairs, canon).footprint_bits
    if footprint <= 0 or footprint % 8:
        _reject(
            f"channels.{endpoint.name}.{endpoint.direction}.protocol.tx_bytes",
            "tile footprint is not a positive byte-integral bit count",
            footprint_bits=footprint,
        )
    return footprint // 8


def _tx_value(
    value: int | str,
    endpoint: Endpoint,
    pairs: dict[str, dict[str, tuple[str, Endpoint]]],
    canon: dict[str, CanonEndpoint],
) -> int:
    if type(value) is int and value >= 0:
        return value
    if value == "tile":
        return _tile_bytes(endpoint, pairs, canon)
    _reject(
        f"channels.{endpoint.name}.{endpoint.direction}.protocol.tx_bytes",
        "invalid transaction-byte declaration",
        actual=value,
    )


def _role_warps(role: str, stages: list[StageManifest], path: str) -> int:
    declarations = [stage.roles[role].warps for stage in stages if role in stage.roles]
    if not declarations:
        _reject(path, "arrive count names a role that is not launched", role=role)
    if len(set(declarations)) != 1:
        _reject(path, "arrive count names a non-unique launch role", role=role)
    return declarations[0]


def _arrive_value(side: BarrierSide, stages: list[StageManifest], path: str) -> int:
    _role_warps(side.arrive.by, stages, path)
    raw = side.arrive.count
    if type(raw) is int:
        if raw <= 0:
            _reject(path, "arrive count must be positive", actual=raw)
        return raw
    if raw == "1":
        return 1
    total = 0
    for term in raw.split("+"):
        match = re.fullmatch(r"\s*(?:([1-9][0-9]*)\s*\*\s*)?(threads|warps)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*", term)
        if match is None:
            _reject(path, "invalid arrive count expression", actual=raw)
        coefficient, unit, role = match.groups()
        warps = _role_warps(role, stages, path)
        total += int(coefficient or 1) * warps * (32 if unit == "threads" else 1)
    return total


def _endpoint_pairs(stages: list[StageManifest]) -> dict[str, dict[str, tuple[str, Endpoint]]]:
    """One producer and one consumer per channel. Counter channels are many-to-many
    (`expand_counter`) and no other protocol reads them, so they are left out."""
    by_channel: dict[str, dict[str, tuple[str, Endpoint]]] = {}
    duplicates: set[tuple[str, str]] = set()
    for stage in sorted(stages, key=lambda item: item.id):
        for endpoint in sorted(stage.endpoints, key=lambda item: (item.name, item.direction)):
            if endpoint.protocol.kind == "counter":
                continue
            entries = by_channel.setdefault(endpoint.name, {})
            if endpoint.direction in entries:
                duplicates.add((endpoint.name, endpoint.direction))
            else:
                entries[endpoint.direction] = (stage.id, endpoint)
    if duplicates:
        channel, direction = sorted(duplicates)[0]
        _reject(
            f"channels.{channel}.{direction}",
            "protocol expansion requires one endpoint per direction",
        )
    return by_channel


def _ring_identity(endpoint: Endpoint) -> tuple[str, int]:
    if endpoint.ring is None:
        _reject(
            f"channels.{endpoint.name}.{endpoint.direction}.ring",
            "mbarrier protocol requires a ring",
        )
    depth = endpoint.ring.depth
    identity = depth.handle if depth.handle is not None else f"fixed:{depth.fixed}"
    return identity, 0 if endpoint.space == "none" else endpoint.ring.stage_stride_bytes


def _claims_for(handle: str, stages: list[StageManifest]) -> list[Any]:
    return [
        claim
        for stage in stages
        for claim in stage.claims
        if claim.handle == handle
    ]


def _check_slot_claim(
    handle: str,
    endpoint: Endpoint,
    stages: list[StageManifest],
    pipeline: Any,
    path: str,
) -> None:
    declaration = pipeline.handles.get(handle)
    if declaration is None or declaration.cls != "base" or declaration.domain != "mbarrier":
        _reject(path, "barrier slots handle is not declared in the mbarrier domain", handle=handle)
    claims = _claims_for(handle, stages)
    if not claims:
        _reject(path, "barrier slots handle has no allocation claim", handle=handle)
    depth = endpoint.ring.depth if endpoint.ring is not None else None
    for claim in claims:
        if claim.domain != "mbarrier" or claim.extent is None:
            _reject(path, "barrier slots handle is not an mbarrier extent claim", handle=handle)
        extent = claim.extent
        if depth is None:
            valid = False
        elif depth.handle is not None:
            valid = extent.const == 0 and extent.per == {depth.handle: 1} and not extent.table
        else:
            valid = extent.const == depth.fixed and not extent.per and not extent.table
        if not valid:
            _reject(
                path,
                "barrier slots extent must equal the ring depth",
                handle=handle,
                extent=extent.to_json_obj(),
            )


def _check_non_alias(
    full_handle: str,
    empty_handle: str,
    stages: list[StageManifest],
    path: str,
) -> None:
    if full_handle == empty_handle:
        _reject(path, "full and empty barrier slots may not use the same handle")
    full_aliases = {claim.alias for claim in _claims_for(full_handle, stages) if claim.alias is not None}
    empty_aliases = {claim.alias for claim in _claims_for(empty_handle, stages) if claim.alias is not None}
    shared = sorted(full_aliases & empty_aliases)
    if shared:
        _reject(path, "full and empty barrier slots may not alias", alias=shared[0])


def _init_agrees(left: Init, right: Init) -> bool:
    if (left.materializer, left.phase) != (right.materializer, right.phase):
        return False
    if left.warp is not None and right.warp is not None:
        return left.warp == right.warp
    if left.owner is not None and right.owner is not None:
        return left.owner == right.owner
    return True  # An owner and a physical warp can only be compared after solving.


def _check_init(
    producer: Endpoint,
    consumer: Endpoint,
    stages: list[StageManifest],
) -> Init:
    left = producer.protocol.init
    right = consumer.protocol.init
    path = f"channels.{producer.name}.protocol.init"
    if left is None or right is None:
        _reject(path, "mbarrier endpoints require init declarations on both sides")
    if left.materializer not in {"stage", "dispatch"} or right.materializer not in {"stage", "dispatch"}:
        _reject(path, "init materializer must be 'stage' or 'dispatch'")
    if not _init_agrees(left, right):
        _reject(
            path,
            "producer and consumer init declarations disagree",
            producer=left.to_json_obj(),
            consumer=right.to_json_obj(),
        )
    for init in (left, right):
        if init.owner is not None:
            _role_warps(init.owner, stages, f"{path}.owner")
        elif init.warp is None:
            _reject(path, "init requires an owner or a physical initializing warp")
    return replace(
        left, owner=left.owner or right.owner,
        warp=left.warp if left.warp is not None else right.warp,
    )


def check_init_warps(
    stages: Iterable[StageManifest],
    warp_map: Mapping[str, tuple[int, int]],
) -> dict[str, str]:
    """Resolve one init owner per channel and check physical warp declarations."""
    ordered_stages = sorted(stages, key=lambda item: item.id)
    owners: dict[str, str] = {}
    for stage in ordered_stages:
        for endpoint in sorted(stage.endpoints, key=lambda item: (item.name, item.direction)):
            init = endpoint.protocol.init
            if endpoint.protocol.kind != "mbarrier" or init is None:
                continue
            path = f"stages.{stage.id}.endpoints.{endpoint.name}.{endpoint.direction}.protocol.init"
            owner = init.owner
            if init.warp is not None:
                matches = sorted(role for role, (lo, hi) in warp_map.items() if lo <= init.warp < hi)
                if not matches:
                    _reject(f"{path}.warp", "initializing warp has no solved role binding", warp=init.warp)
                if len(matches) != 1:
                    _reject(
                        f"{path}.warp", "initializing warp belongs to multiple solved roles",
                        warp=init.warp, roles=matches,
                    )
                owner = matches[0]
                if init.owner is not None and init.owner != owner:
                    _reject(
                        f"{path}.warp", "declared init owner disagrees with the initializing warp's solved role",
                        owner=init.owner, derived_owner=owner, warp=init.warp,
                    )
            if owner is None:
                _reject(path, "init requires an owner or a physical initializing warp")
            previous = owners.setdefault(endpoint.name, owner)
            if previous != owner:
                _reject(
                    f"channels.{endpoint.name}.protocol.init",
                    "producer and consumer solved init owners disagree",
                    expected=previous, actual=owner,
                )

    ring_owners: dict[tuple[str, str], str] = {}
    for stage in ordered_stages:
        for endpoint in stage.endpoints:
            protocol = endpoint.protocol
            if protocol.kind != "mbarrier" or protocol.init is None:
                continue
            owner = owners[endpoint.name]
            full = protocol.full
            if isinstance(full, SharedMember):
                expected = owners.get(full.member_of)
            elif isinstance(full, BarrierSide) and protocol.empty is not None:
                expected = ring_owners.setdefault((full.slots, protocol.empty.slots), owner)
            else:
                continue
            if expected != owner:
                _reject(
                    f"channels.{endpoint.name}.protocol.init",
                    "barrier ring solved init owners disagree",
                    expected=expected, actual=owner,
                )
    return owners


def _check_ring_init_owner(
    full_handle: str,
    empty_handle: str,
    expected: Init,
    pairs: dict[str, dict[str, tuple[str, Endpoint]]],
    stages: list[StageManifest],
) -> None:
    for channel in sorted(pairs):
        sides = pairs[channel]
        if set(sides) != {"produce", "consume"}:
            continue
        producer = sides["produce"][1]
        consumer = sides["consume"][1]
        full = producer.protocol.full
        empty = producer.protocol.empty
        if (
            producer.protocol.kind != "mbarrier"
            or not isinstance(full, BarrierSide)
            or empty is None
            or (full.slots, empty.slots) != (full_handle, empty_handle)
        ):
            continue
        actual = _check_init(producer, consumer, stages)
        if not _init_agrees(actual, expected):
            _reject(
                f"channels.{channel}.protocol.init",
                "barrier ring has more than one init owner or materializer",
                expected=expected.to_json_obj(),
                actual=actual.to_json_obj(),
            )


def _root_for(
    channel: str,
    producer: Endpoint,
    consumer: Endpoint,
) -> str:
    roots: list[str] = []
    for endpoint in (producer, consumer):
        full = endpoint.protocol.full
        roots.append(full.member_of if isinstance(full, SharedMember) else channel)
    if roots[0] != roots[1]:
        _reject(
            f"channels.{channel}.protocol.full.member_of",
            "producer and consumer disagree on the shared-completion root",
            producer=roots[0],
            consumer=roots[1],
        )
    return roots[0]


def _shared_group(
    root: str,
    pairs: dict[str, dict[str, tuple[str, Endpoint]]],
    stages: list[StageManifest],
    canon: dict[str, CanonEndpoint],
) -> tuple[tuple[str, ...], dict[str, int], int]:
    if root not in pairs or set(pairs[root]) != {"produce", "consume"}:
        _reject(f"channels.{root}.protocol.full", "shared-completion root does not exist")
    root_producer = pairs[root]["produce"][1]
    root_consumer = pairs[root]["consume"][1]
    root_fulls = (root_producer.protocol.full, root_consumer.protocol.full)
    if any(not isinstance(full, BarrierSide) for full in root_fulls):
        _reject(f"channels.{root}.protocol.full", "member_of may only target a root barrier side")
    assert isinstance(root_fulls[0], BarrierSide)
    assert isinstance(root_fulls[1], BarrierSide)
    if not isinstance(root_fulls[0].tx_bytes, TxSum) or not isinstance(root_fulls[1].tx_bytes, TxSum):
        _reject(f"channels.{root}.protocol.full.tx_bytes", "shared-completion root must declare a sum")
    declared_left = tuple(sorted(root_fulls[0].tx_bytes.sum))
    declared_right = tuple(sorted(root_fulls[1].tx_bytes.sum))
    if declared_left != declared_right:
        _reject(
            f"channels.{root}.protocol.full.tx_bytes.sum",
            "producer and consumer root sums disagree",
            producer=list(declared_left),
            consumer=list(declared_right),
        )

    members: list[str] = []
    for name in sorted(pairs):
        sides = pairs[name]
        if set(sides) != {"produce", "consume"}:
            continue
        left_full = sides["produce"][1].protocol.full
        right_full = sides["consume"][1].protocol.full
        left_root = left_full.member_of if isinstance(left_full, SharedMember) else None
        right_root = right_full.member_of if isinstance(right_full, SharedMember) else None
        if left_root == root or right_root == root:
            if left_root != root or right_root != root:
                _reject(
                    f"channels.{name}.protocol.full.member_of",
                    "both member endpoints must name the same root",
                )
            members.append(name)
    group = tuple(sorted([root, *members]))
    if declared_left != group:
        _reject(
            f"channels.{root}.protocol.full.tx_bytes.sum",
            "root sum must name exactly the root and all direct members",
            declared=list(declared_left),
            actual=list(group),
        )

    root_depth = _ring_identity(root_producer)[0]
    root_scope = root_producer.scope
    root_init = _check_init(root_producer, root_consumer, stages)
    contributions: dict[str, int] = {}
    for name in group:
        left = pairs[name]["produce"][1]
        right = pairs[name]["consume"][1]
        if _ring_identity(left)[0] != root_depth or _ring_identity(right)[0] != root_depth:
            _reject(f"channels.{name}.ring.depth", "shared-completion group depth handles disagree")
        if left.scope != root_scope or right.scope != root_scope:
            _reject(f"channels.{name}.scope", "shared-completion group scopes disagree")
        if not _init_agrees(_check_init(left, right, stages), root_init):
            _reject(f"channels.{name}.protocol.init", "shared-completion group init ownership disagrees")
        if name == root:
            left_value = _tile_bytes(left, pairs, canon)
            right_value = _tile_bytes(right, pairs, canon)
        else:
            left_full = left.protocol.full
            right_full = right.protocol.full
            if not isinstance(left_full, SharedMember) or not isinstance(right_full, SharedMember):
                _reject(f"channels.{name}.protocol.full", "shared-completion member is not direct")
            left_value = _tx_value(left_full.tx_bytes, left, pairs, canon)
            right_value = _tx_value(right_full.tx_bytes, right, pairs, canon)
        if left_value != right_value:
            _reject(
                f"channels.{name}.protocol.full.tx_bytes",
                "producer and consumer shared contributions disagree",
                producer=left_value,
                consumer=right_value,
            )
        contributions[name] = left_value
    return group, {name: contributions[name] for name in sorted(contributions)}, sum(contributions.values())


def _paired_barrier_side(
    channel: str,
    label: str,
    producer_side: BarrierSide | None,
    consumer_side: BarrierSide | None,
    stages: list[StageManifest],
) -> tuple[str, int, int]:
    path = f"channels.{channel}.protocol.{label}"
    if producer_side is None or consumer_side is None:
        _reject(path, "barrier side is missing")
    if producer_side.slots != consumer_side.slots:
        _reject(
            f"{path}.slots",
            "producer and consumer slot handles disagree",
            producer=producer_side.slots,
            consumer=consumer_side.slots,
        )
    producer_arrive = _arrive_value(producer_side, stages, f"{path}.arrive.count")
    consumer_arrive = _arrive_value(consumer_side, stages, f"{path}.arrive.count")
    if producer_arrive != consumer_arrive or producer_side.arrive.by != consumer_side.arrive.by:
        _reject(
            f"{path}.arrive",
            "producer and consumer arrival declarations disagree",
            producer=[producer_side.arrive.by, producer_arrive],
            consumer=[consumer_side.arrive.by, consumer_arrive],
        )
    return producer_side.slots, producer_arrive, consumer_arrive
def _check_side_tx(
    channel: str,
    label: str,
    producer_side: BarrierSide,
    consumer_side: BarrierSide,
    producer: Endpoint,
    consumer: Endpoint,
    pairs: dict[str, dict[str, tuple[str, Endpoint]]],
    canon: dict[str, CanonEndpoint],
) -> tuple[int, int]:
    producer_value = _tx_value(producer_side.tx_bytes, producer, pairs, canon)
    consumer_value = _tx_value(consumer_side.tx_bytes, consumer, pairs, canon)
    if producer_value != consumer_value:
        _reject(
            f"channels.{channel}.protocol.{label}.tx_bytes",
            "producer and consumer transaction-byte expansions disagree",
            producer=producer_value,
            consumer=consumer_value,
        )
    return producer_value, consumer_value




def _consumer_export(
    consumer: Endpoint,
    stages: list[StageManifest],
    pairs: dict[str, dict[str, tuple[str, Endpoint]]],
    export_name: str,
) -> Export:
    stage_id, _ = pairs[consumer.name][consumer.direction]
    stage = next(stage for stage in stages if stage.id == stage_id)
    if export_name not in stage.exports:
        _reject(
            f"channels.{consumer.name}.protocol.export",
            "consumer stage does not declare the named callable export",
            stage=stage.id,
            export=export_name,
        )
    return stage.exports[export_name]


def expand(
    channel_name: str,
    producer: Endpoint,
    consumer: Endpoint,
    stages: Iterable[StageManifest],
    pipeline: Any,
    canon: dict[str, CanonEndpoint],
    profile: Any,
) -> ProtocolExpansion:
    """Expand and validate one paired channel's executable protocol declaration."""

    del profile
    ordered_stages = sorted(stages, key=lambda item: item.id)
    path = f"channels.{channel_name}.protocol"
    if producer.name != channel_name or consumer.name != channel_name:
        _reject(path, "endpoint names do not match the requested channel")
    if producer.direction != "produce" or consumer.direction != "consume":
        _reject(path, "protocol endpoints are not ordered producer then consumer")
    if producer.protocol.kind != consumer.protocol.kind:
        _reject(
            f"{path}.kind",
            "producer and consumer protocol kinds disagree",
            producer=producer.protocol.kind,
            consumer=consumer.protocol.kind,
        )
    kind = producer.protocol.kind
    if kind == "counter":
        return expand_counter(channel_name, [producer], [consumer], pipeline)
    pairs = _endpoint_pairs(ordered_stages)
    if set(pairs.get(channel_name, {})) != {"produce", "consume"}:
        _reject(path, "requested channel must have one producer and one consumer in linked stages")

    if producer.space == "none" or consumer.space == "none":
        for field in ("space", "scope", "proxy", "ring", "release"):
            left = producer.protocol.release if field == "release" else (
                _ring_identity(producer) if field == "ring" else getattr(producer, field)
            )
            right = consumer.protocol.release if field == "release" else (
                _ring_identity(consumer) if field == "ring" else getattr(consumer, field)
            )
            if left != right:
                raise SolveError([ClauseFailure(
                    "L-PAIR", f"channels.{channel_name}.{field}",
                    f"completion-only endpoint {field} declarations disagree", {},
                )])
    if kind == "mbarrier":
        if producer.ring is None or consumer.ring is None or _ring_identity(producer) != _ring_identity(consumer):
            _reject(f"channels.{channel_name}.ring", "mbarrier endpoint ring declarations disagree")
        if producer.protocol.release != consumer.protocol.release:
            _reject(f"{path}.release", "mbarrier release declarations disagree")
        root = _root_for(channel_name, producer, consumer)
        has_members = any(
            isinstance(sides["produce"][1].protocol.full, SharedMember)
            and sides["produce"][1].protocol.full.member_of == root
            for sides in pairs.values()
            if "produce" in sides
        )
        producer_full = producer.protocol.full
        consumer_full = consumer.protocol.full
        is_shared = (
            isinstance(producer_full, SharedMember)
            or isinstance(consumer_full, SharedMember)
            or isinstance(getattr(producer_full, "tx_bytes", None), TxSum)
            or isinstance(getattr(consumer_full, "tx_bytes", None), TxSum)
            or has_members
        )
        if not is_shared:
            if not isinstance(producer_full, BarrierSide) or not isinstance(consumer_full, BarrierSide):
                _reject(f"{path}.full", "ordinary mbarrier channel requires complete full sides")
            full_handle, arrive_full, _ = _paired_barrier_side(
                channel_name, "full", producer_full, consumer_full, ordered_stages
            )
            if producer.protocol.release in {"never", "external"}:
                producer_tx, consumer_tx = _check_side_tx(
                    channel_name, "full", producer_full, consumer_full, producer, consumer, pairs, canon
                )
                _check_slot_claim(full_handle, producer, ordered_stages, pipeline, f"{path}.full.slots")
                init = _check_init(producer, consumer, ordered_stages)
                return ProtocolExpansion(
                    channel_name, kind, producer_tx, consumer_tx, arrive_full, None,
                    init.owner, init.materializer,
                    (full_handle, None) if producer.space == "none" else None,
                    None, None, None, None,
                )
            empty_handle, arrive_empty, _ = _paired_barrier_side(
                channel_name, "empty", producer.protocol.empty, consumer.protocol.empty, ordered_stages
            )
            assert producer.protocol.empty is not None
            assert consumer.protocol.empty is not None
            _check_side_tx(
                channel_name,
                "empty",
                producer.protocol.empty,
                consumer.protocol.empty,
                producer,
                consumer,
                pairs,
                canon,
            )
            producer_tx = _tx_value(producer_full.tx_bytes, producer, pairs, canon)
            consumer_tx = _tx_value(consumer_full.tx_bytes, consumer, pairs, canon)
            if producer_tx != consumer_tx:
                _reject(
                    f"{path}.full.tx_bytes",
                    "producer and consumer transaction-byte expansions disagree",
                    producer=producer_tx,
                    consumer=consumer_tx,
                )
            _check_slot_claim(
                full_handle, producer, ordered_stages, pipeline, f"channels.{channel_name}.protocol.full.slots"
            )
            _check_slot_claim(
                empty_handle, producer, ordered_stages, pipeline, f"channels.{channel_name}.protocol.empty.slots"
            )
            _check_non_alias(full_handle, empty_handle, ordered_stages, path)
            init = _check_init(producer, consumer, ordered_stages)
            _check_ring_init_owner(
                full_handle,
                empty_handle,
                init,
                pairs,
                ordered_stages,
            )
            return ProtocolExpansion(
                channel_name,
                kind,
                producer_tx,
                consumer_tx,
                arrive_full,
                arrive_empty,
                init.owner,
                init.materializer,
                (full_handle, empty_handle),
                None,
                None,
                None,
                None,
            )

        members, contributions, root_tx = _shared_group(root, pairs, ordered_stages, canon)
        root_producer = pairs[root]["produce"][1]
        root_consumer = pairs[root]["consume"][1]
        assert isinstance(root_producer.protocol.full, BarrierSide)
        assert isinstance(root_consumer.protocol.full, BarrierSide)
        full_handle, arrive_full, _ = _paired_barrier_side(
            root,
            "full",
            root_producer.protocol.full,
            root_consumer.protocol.full,
            ordered_stages,
        )
        empty_handle, arrive_empty, _ = _paired_barrier_side(
            root,
            "empty",
            root_producer.protocol.empty,
            root_consumer.protocol.empty,
            ordered_stages,
        )
        assert root_producer.protocol.empty is not None
        assert root_consumer.protocol.empty is not None
        _check_side_tx(
            root,
            "empty",
            root_producer.protocol.empty,
            root_consumer.protocol.empty,
            root_producer,
            root_consumer,
            pairs,
            canon,
        )
        _check_slot_claim(
            full_handle, root_producer, ordered_stages, pipeline, f"channels.{root}.protocol.full.slots"
        )
        _check_slot_claim(
            empty_handle, root_producer, ordered_stages, pipeline, f"channels.{root}.protocol.empty.slots"
        )
        _check_non_alias(full_handle, empty_handle, ordered_stages, f"channels.{root}.protocol")
        for member in members:
            if member == root:
                continue
            member_producer = pairs[member]["produce"][1]
            member_consumer = pairs[member]["consume"][1]
            member_empty, member_arrive, _ = _paired_barrier_side(
                member,
                "empty",
                member_producer.protocol.empty,
                member_consumer.protocol.empty,
                ordered_stages,
            )
            assert member_producer.protocol.empty is not None
            assert member_consumer.protocol.empty is not None
            _check_side_tx(
                member,
                "empty",
                member_producer.protocol.empty,
                member_consumer.protocol.empty,
                member_producer,
                member_consumer,
                pairs,
                canon,
            )
            if (member_empty, member_arrive) != (empty_handle, arrive_empty):
                _reject(
                    f"channels.{member}.protocol.empty",
                    "shared-completion members must use the root empty arrival state",
                )
            _check_slot_claim(
                member_empty,
                member_producer,
                ordered_stages,
                pipeline,
                f"channels.{member}.protocol.empty.slots",
            )
        init = _check_init(
            root_producer, root_consumer, ordered_stages
        )
        _check_ring_init_owner(
            full_handle,
            empty_handle,
            init,
            pairs,
            ordered_stages,
        )
        is_root = channel_name == root
        own_producer = contributions[channel_name]
        own_consumer = contributions[channel_name]
        return ProtocolExpansion(
            channel_name,
            kind,
            root_tx if is_root else own_producer,
            root_tx if is_root else own_consumer,
            arrive_full if is_root else None,
            arrive_empty if is_root else None,
            init.owner if is_root else None,
            init.materializer if is_root else None,
            (full_handle, empty_handle) if is_root else None,
            (root, members, contributions),
            None,
            None,
            None,
        )

    if kind == "call":
        left = producer.protocol.call
        right = consumer.protocol.call
        if left is None or right is None:
            _reject(path, "call declaration is missing")
        if left.export != right.export:
            _reject(f"{path}.export", "call endpoints name different exports")
        for side, item in (("producer", left), ("consumer", right)):
            actual = declaration_digest(item.decl)
            if item.decl_sha256 != actual:
                _reject(f"{path}.decl_sha256", f"{side} call declaration hash does not match normalized text")
        exported = _consumer_export(consumer, ordered_stages, pairs, left.export)
        if (normalize_declaration(exported.decl), exported.decl_sha256) != (
            normalize_declaration(right.decl), right.decl_sha256
        ):
            _reject(path, "consumer export declaration or hash disagrees with the channel")
        export_actual = declaration_digest(exported.decl)
        if exported.decl_sha256 != export_actual:
            _reject(f"{path}.decl_sha256", "consumer export declaration hash is invalid")
        # Substitute only the expected function declarator. All other tokens,
        # including ABI typedef identifiers and literal contents, remain identity-bearing.
        placeholder = "\x00pipelink_call_identifier\x00"
        try:
            caller_normal = replace_declarator_identifier(
                left.decl, call_symbol(channel_name), placeholder)
            callee_normal = replace_declarator_identifier(
                exported.decl, exported.symbol, placeholder)
        except LinkError as error:
            _reject(f"{path}.decl", str(error))
        if normalize_declaration(caller_normal) != normalize_declaration(callee_normal):
            _reject(
                f"{path}.decl",
                "caller and export declarations differ outside the declarator identifier",
            )
        return ProtocolExpansion(
            channel_name,
            kind,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (exported.symbol, exported.decl, exported.decl_sha256),
            None,
            None,
        )

    if kind == "named_barrier":
        left = producer.protocol.named_barrier
        right = consumer.protocol.named_barrier
        if left is None or right is None:
            _reject(path, "named-barrier declaration is missing")
        declaration = pipeline.handles.get(left.id_handle)
        if declaration is None or declaration.cls != "base" or declaration.domain != "named_barrier":
            _reject(path, "named-barrier id handle is not declared in the named_barrier domain")
        if (left.id_handle, left.count) != (right.id_handle, right.count):
            _reject(path, "named-barrier id handle or thread count disagrees")
        return ProtocolExpansion(
            channel_name, kind, None, None, None, None, None, None, None, None, None,
            (left.id_handle, left.count), None,
        )

    _reject(f"{path}.kind", "unsupported protocol kind", actual=kind)


def expand_counter(
    channel_name: str,
    producers: Sequence[Endpoint],
    consumers: Sequence[Endpoint],
    pipeline: Any,
) -> ProtocolExpansion:
    """Expand one counter channel: one or more producers, zero or more consumers.

    Arrivals are additive red.release operations, so any number of producers may share a slot;
    a released slot that no stage acquires is harmless. Every endpoint agrees on scope, slot and
    release; each consumer chooses its own acquire pattern, recorded as the sorted set in use.
    """

    path = f"channels.{channel_name}.protocol"
    endpoints = [*producers, *consumers]
    if not producers or any(endpoint.protocol.counter is None for endpoint in endpoints):
        _reject(path, "counter channel needs a producer and a counter declaration on every endpoint")
    first = producers[0].protocol.counter
    assert first is not None
    declaration = pipeline.handles.get(first.slot_handle)
    if declaration is None or declaration.cls != "base" or declaration.domain != "gmem_counter":
        _reject(path, "counter slot handle is not declared in the gmem_counter domain")
    if any(endpoint.space != "gmem" for endpoint in endpoints):
        _reject(path, "counter channels must use gmem space")
    agreed = (producers[0].scope, first.slot_handle, first.release_op)
    if any((endpoint.scope, endpoint.protocol.counter.slot_handle, endpoint.protocol.counter.release_op) != agreed
           for endpoint in endpoints):
        _reject(path, "counter scope, slot or release operation disagree")
    acquires = tuple(sorted({endpoint.protocol.counter.acquire_op for endpoint in consumers}))
    return ProtocolExpansion(
        channel_name, "counter", None, None, None, None, None, None, None, None, None, None,
        (first.slot_handle, first.release_op, acquires),
    )


def resolve_native(
    expansions: Iterable[ProtocolExpansion],
    stages: Iterable[StageManifest],
    solution: Solution,
    profile: Any,
) -> list[ProtocolExpansion]:
    """Resolve physical barrier addresses, checking only bounds unavailable at load time."""
    native_by_channel: dict[str, dict[str, dict[str, int]]] = {}
    for stage in stages:
        claims = {claim.handle: claim for claim in stage.claims}
        for endpoint in stage.endpoints:
            for label in ("full", "empty"):
                side = getattr(endpoint.protocol, label)
                if not isinstance(side, BarrierSide) or side.native is None:
                    continue
                storage = side.native
                claim = claims[storage.base]
                unit_bytes = profile.domains[claim.domain].unit_scale
                base = solution.native_bases[storage.base]
                assert endpoint.ring is not None and claim.extent is not None
                depth = endpoint.ring.depth.fixed
                if depth is None:
                    depth = solution.depths[endpoint.ring.depth.handle]
                if claim.extent.per or claim.extent.table or endpoint.ring.depth.handle is not None:
                    interval = next(
                        interval for interval in solution.intervals["smem"]
                        if interval.handle == storage.base
                    )
                    end = base * unit_bytes + storage.offset_bytes + (depth - 1) * storage.stride_bytes + 8
                    if end > interval.hi:
                        _reject(f"channels.{endpoint.name}.protocol.{label}.native",
                                "native barrier slots exceed the solved backing claim extent",
                                end_bytes=end, claim_end_bytes=interval.hi)
                resolved = {
                    "base_solved_offset_units": base,
                    "unit_bytes": unit_bytes,
                    "offset_bytes": storage.offset_bytes,
                    "stride_bytes": storage.stride_bytes,
                }
                sides = native_by_channel.setdefault(endpoint.name, {})
                if label in sides:
                    other = sides[label]
                    other_address = other["base_solved_offset_units"] * other["unit_bytes"] + other["offset_bytes"]
                    if (other_address, other["stride_bytes"]) != (base * unit_bytes + storage.offset_bytes, storage.stride_bytes):
                        _reject(f"channels.{endpoint.name}.protocol.{label}.native",
                                "producer and consumer physical barrier addresses disagree")
                sides[label] = resolved
    return [replace(expansion, native=native_by_channel.get(expansion.channel)) for expansion in expansions]


__all__ = ["expand", "expand_counter", "resolve_native"]
