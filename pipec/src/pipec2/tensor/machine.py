"""Independent finite tensor-access machine.

Controllers execute per physical participant.  A collective issues only when all
members of its declared warp are at the same instruction; shared rendezvous has
separate arrivals and publication.  Payload integers are ghost identity indices,
not runtime data or observations.  EMPTY denotes an unavailable copy.

Operand cells are {resource, index, participant?}; index/participant may be C1
expressions in participant, published bindings, and retained local metadata.
SMEM indices are byte offsets; register indices are scalar positions.  The
special borrowed source names are @export:reg and @export:smem.  TMEM is accessible
only through complete packets in the issuing warp's own DP subpartition, never scalar edges or cross-warp DP reads.
Shared rounds use an address expression over participant and slot; each slot is
one ordinary scalar transition, with no implicit rendezvous or atomic batching.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, overload

from pipec2.canonjson import canonical_bytes, sha256_hex
from pipec2.model import ClauseFailure, PipeLinkError

from . import expr, ir
from .contracts import Lend, Site, TensorExport, TensorImport

if TYPE_CHECKING:
    from .plan import Derived

EMPTY = -1
_ROUND_SCALARS = {"smem_store_round": "smem_store_scalar", "smem_load_round": "smem_load_scalar",
                  "reg_copy_round": "reg_copy"}
MODEL_SEMANTICS = "tensor-machine-v6-memory-view-publication"


def _constant(value: int, sort: str = "int") -> list[Any]:
    return ["const", value] if sort == "int" else ["cbits", int(sort[5:-1]), value]


def _substitute(expression: expr.Expr, coordinates: Mapping[str, Any]) -> Any:
    if expression.op == "coord" and expression.args[0] in coordinates:
        return coordinates[expression.args[0]]
    return [expression.op, *(_substitute(arg, coordinates) if isinstance(arg, expr.Expr) else arg for arg in expression.args)]


def _bind_constants(value: Any, bindings: Mapping[str, int], domains: Mapping[str, expr.Domain]) -> Any:
    if isinstance(value, expr.Expr):
        value = expr.to_json_obj(value)
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and value[0] == "bind" and value[1] in bindings:
            name = value[1]
            return _constant(bindings[name], domains[name].sort if name in domains else "int")
        return [_bind_constants(item, bindings, domains) for item in value]
    return value



class ModelError(PipeLinkError):
    def __init__(self, issues: tuple[str, ...] | list[str], *, status: str = "UNSUPPORTED_MODEL"):
        self.status = status
        self.issues = tuple(sorted(set(issues)))
        clause_id = "T-CONTRACT" if status == "CONTRACT_INVALID" else "T-MODEL"
        super().__init__([ClauseFailure(clause_id, "$.tensor", issue, {"status": status}) for issue in self.issues])


@dataclass(frozen=True)
class Primitive:
    family: str
    participants: Literal["thread", "warp", "cta"]
    operand_specs: Mapping[str, str]
    timing: Mapping[str, str]
    indivisible_receipt: bool
    blocking: bool
    charges: Mapping[str, int]
    lowering_recipe: str | None


def _option(profile: Any, name: str, default: Any) -> Any:
    return profile.get(name, default) if isinstance(profile, Mapping) else getattr(profile, name, default)


def tmem_lane_count(profile: Any) -> int | None:
    """Physical TMEM lanes, or None when the target has no TMEM.

    One rule for real and finite test profiles alike: a target has TMEM iff it states a
    lane count (`Profile.tmem_lanes`; `tmem_dps`/`tmem_lanes` in a Mapping test profile).
    """
    return _option(profile, "tmem_dps", _option(profile, "tmem_lanes", None))


def tmem_column_count(profile: Any) -> int:
    """TMEM columns of a target that has TMEM: the `tmem` domain, or a Mapping's stated option."""
    return profile["tmem_columns"] if isinstance(profile, Mapping) else profile.domains["tmem"].capacity


CATALOGUES = frozenset({"tcgen05-readonly-v1", "warp-readonly-v1"})


def _primitive(name, participants, operands, read, visible, indivisible=False, blocking=False, charges=None, recipe=None):
    return Primitive(name, participants, MappingProxyType(operands), MappingProxyType({"read_at": read, "visible_after": visible}),
                     indivisible, blocking, MappingProxyType(charges or {}), recipe)


PRIMITIVES: Mapping[str, Primitive] = MappingProxyType({entry.family: entry for entry in (
    _primitive("tmem_packet_receipt", "warp", {"packet": "own-subpartition-packet-index", "dest": "private-resource", "offset": "scalar-position",
               "phase": "issue|wait", "handle": "local-handle"}, "completion", "completion", True, True,
               {"handles": 1, "outstanding": 1}, "tilelang.tcgen05.packet"),
    _primitive("smem_store_scalar", "thread", {"src": "reg-cell", "dst": "smem-cell"}, "issue", "rendezvous", recipe="tilelang.shared.store"),
    _primitive("smem_load_scalar", "thread", {"src": "smem-cell", "dst": "reg-cell"}, "issue", "completion", recipe="tilelang.shared.load"),
    _primitive("smem_store_round", "thread", {"width": "scalar-count", "src": "reg-resource", "dst": "smem-resource", "address": "participant-slot-expression", "offset": "private-position"}, "issue", "rendezvous", recipe="tilelang.shared.store.round"),
    _primitive("smem_load_round", "thread", {"width": "scalar-count", "src": "smem-resource", "dst": "reg-resource", "address": "participant-slot-expression", "offset": "private-position"}, "issue", "completion", recipe="tilelang.shared.load.round"),
    _primitive("reg_copy", "thread", {"src": "reg-cell", "dst": "reg-cell"}, "issue", "completion", recipe="tilelang.private.copy"),
    _primitive("reg_copy_round", "thread", {"width": "scalar-count", "src": "reg-resource", "dst": "reg-resource", "offset": "private-position", "src_offset": "private-position"}, "issue", "completion", recipe="tilelang.private.copy.round"),
    _primitive("shfl_xor", "warp", {"src": "reg-cell", "dst": "reg-cell", "mask": "lane-xor"}, "issue", "completion", True, recipe="tilelang.warp.shuffle.xor"),
    _primitive("cta_rendezvous", "cta", {"resource": "barrier-resource"}, "issue", "rendezvous", blocking=True, charges={"barriers": 1}, recipe="tilelang.cta.rendezvous"),
    _primitive("observe_ready", "thread", {"dest": "local-metadata"}, "issue", "completion", recipe="tilelang.fenced.readiness.observe"),
)})


def CATALOGUE(profile: Any) -> Mapping[str, Primitive]:
    """The target catalogue, independent from candidate controller construction.

    `tmem_packet_receipt` exists only on a target with TMEM (`tmem_lane_count`).
    """
    tmem = tmem_lane_count(profile) is not None
    return MappingProxyType({family: entry for family, entry in PRIMITIVES.items() if tmem or family != "tmem_packet_receipt"})


def _stay(state: int, label: str) -> int:
    return state


@dataclass(frozen=True)
class Monitor:
    """Finite automaton plus an explicit omega-regular acceptance condition.

    justice: infinitely-often enabled labels imply infinitely-often taken labels.
    compassion: infinitely-often premise labels imply consequence labels.
    response: accepting automaton states must recur.  terminal: predicate on the
    final machine state.  No monitor is allowed to assume adapter success.
    """
    kind: Literal["justice", "compassion", "terminal", "response"]
    states: tuple[int, ...] = (0,)
    initial: int = 0
    step: Callable[[int, str], int] = _stay
    accepting: tuple[int, ...] = (0,)
    labels: frozenset[str] = frozenset()
    premise: frozenset[str] = frozenset()
    consequence: frozenset[str] = frozenset()
    predicate: Callable[[State], bool] | None = None
    name: str = ""


@dataclass(frozen=True, order=True)
class Cell:
    resource: str
    participant: int
    index: int

    @property
    def name(self) -> str:
        return f"{self.resource}:{self.participant}:{self.index}"


@dataclass(frozen=True)
class Packet:
    dp: int
    column: int
    receipt: tuple[tuple[int, int, int], ...]  # lane, scalar offset, source-cell index


@dataclass(frozen=True, order=True)
class Pending:
    site: str
    warp: int
    handle: str
    sources: tuple[int, ...]
    destinations: tuple[int, ...]
    completed: bool = False
    _cached_hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_cached_hash", hash((self.site, self.warp, self.handle, self.sources, self.destinations, self.completed)))

    def __hash__(self) -> int:
        return self._cached_hash

    @property
    def label(self) -> str:
        return f"complete:{self.site}:{self.warp}:{self.handle}"


@dataclass(frozen=True, order=True)
class Arrival:
    resource: str
    actors: tuple[int, ...]


class _IdentityChunk(tuple):
    """Immutable identity chunk whose structural hash is computed only once."""

    def __new__(cls, values):
        result = tuple.__new__(cls, values)
        object.__setattr__(result, "_cached_hash", tuple.__hash__(result))
        return result

    def __hash__(self):
        return self._cached_hash

    def __setattr__(self, name, value):
        raise AttributeError("identity vectors are immutable")


class IdentityVector(Sequence[int]):
    """Immutable, hash-cached, copy-on-write identity cells in 64-cell chunks."""

    __slots__ = ("_chunks", "_length", "_cached_hash")
    _CHUNK = 64

    def __init__(self, values: Iterable[int]):
        values = tuple(values)
        chunks = tuple(_IdentityChunk(values[offset:offset + self._CHUNK]) for offset in range(0, len(values), self._CHUNK))
        object.__setattr__(self, "_chunks", chunks)
        object.__setattr__(self, "_length", len(values))
        object.__setattr__(self, "_cached_hash", hash((len(values), chunks)))

    @classmethod
    def _from_chunks(cls, chunks, length):
        result = object.__new__(cls)
        object.__setattr__(result, "_chunks", chunks)
        object.__setattr__(result, "_length", length)
        object.__setattr__(result, "_cached_hash", hash((length, chunks)))
        return result

    def __setattr__(self, name, value):
        raise AttributeError("identity vectors are immutable")

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[int]:
        for chunk in self._chunks:
            yield from chunk

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self)[index]
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return self._chunks[index // self._CHUNK][index % self._CHUNK]

    def __hash__(self) -> int:
        return self._cached_hash

    def __eq__(self, other):
        if not isinstance(other, IdentityVector):
            return NotImplemented
        return self._length == other._length and self._chunks == other._chunks

    def __repr__(self) -> str:
        return f"IdentityVector(cells={self._length}, chunks={len(self._chunks)})"

    def count(self, value: int) -> int:
        return sum(chunk.count(value) for chunk in self._chunks)

    def __contains__(self, value: object) -> bool:
        return any(value in chunk for chunk in self._chunks)

    def updated(self, changes: Iterable[tuple[int, int]]) -> IdentityVector:
        changed = {}
        for index, value in changes:
            if not 0 <= index < self._length:
                raise IndexError(index)
            chunk = index // self._CHUNK
            current = changed[chunk][index % self._CHUNK] if chunk in changed else self[index]
            if current == value:
                continue
            if chunk not in changed:
                changed[chunk] = list(self._chunks[chunk])
            changed[chunk][index % self._CHUNK] = value
        if not changed:
            return self
        chunks = list(self._chunks)
        for index, values in changed.items():
            chunks[index] = _IdentityChunk(values)
        return self._from_chunks(tuple(chunks), self._length)

    def intern_chunks(self, pool: dict) -> tuple[IdentityVector, int]:
        chunks = tuple(pool.setdefault(chunk, chunk) for chunk in self._chunks)
        replaced = sum(left is not right for left, right in zip(chunks, self._chunks))
        return (self._from_chunks(chunks, self._length), replaced) if replaced else (self, 0)


@dataclass(frozen=True)
class State:
    cells: Sequence[int]
    published: Sequence[int]
    outstanding: tuple[Pending, ...]
    arrivals: tuple[Arrival, ...]
    accesses: tuple[tuple[int, int, str], ...]
    owner_phase: int
    positions: tuple[int, ...]
    metadata: tuple[tuple[tuple[str, int], ...], ...]
    generations: tuple[int, ...]
    world: int = 0
    errors: tuple[str, ...] = ()
    monitor_states: tuple[int, ...] = ()
    operations: int = 0
    round_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        cells = self.cells if isinstance(self.cells, IdentityVector) else IdentityVector(self.cells)
        published = cells if self.published is self.cells else (
            self.published if isinstance(self.published, IdentityVector) else IdentityVector(self.published)
        )
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "published", published)
        if not self.round_positions:
            object.__setattr__(self, "round_positions", (0,) * len(self.positions))


@dataclass(frozen=True)
class Transition:
    site: str
    label: str
    primitive: str | None
    observation: bool
    actors: tuple[int, ...] = ()
    detail: str = ""


def is_static_expression(value: Any) -> bool:
    """A closed expression recomputable from the physical participant alone."""
    if isinstance(value, expr.Expr):
        value = expr.to_json_obj(value)
    if type(value) is int:
        return True
    if not isinstance(value, (tuple, list)) or not value or value[0] not in expr._GRAMMAR:
        return False
    if value[0] == "coord":
        return len(value) == 2 and value[1] == "participant"
    if value[0] == "bind":
        return False
    return all(is_static_expression(arg) for arg in value[1:])


def _values(domain: expr.Domain) -> tuple[int, ...] | range:
    return domain.values if domain.values is not None else range(domain.lo, domain.hi)


def _environments(domains: Mapping[str, expr.Domain], cap: int):
    names = sorted(domains)
    values = [_values(domains[name]) for name in names]
    if prod(len(value) for value in values) > cap:
        raise ModelError(["expression grounding exceeds the finite analyzer cap"])
    return tuple(dict(zip(names, point)) for point in product(*values))


def _grounded(expression: expr.Expr, domains: Mapping[str, expr.Domain], cap: int) -> tuple[tuple[str, ...], dict]:
    try:
        report = expr.domain_check(expression, domains, cap=cap)
    except (expr.ExprError, KeyError) as error:
        raise ModelError([f"expression dependency is outside this finite grounding environment: {error}"]) from error
    if report.capped:
        raise ModelError(["expression grounding exceeds the finite analyzer cap"])
    if not report.ok:
        raise ModelError(list(report.issues), status="CONTRACT_INVALID")
    def references(node):
        if node.op in {"coord", "bind"}:
            return {node.args[0]}
        return set().union(*(references(arg) for arg in node.args if isinstance(arg, expr.Expr)))
    return tuple(sorted(references(expression))), expr.ground(expression, domains)


def _lookup(table, environment):
    names, values = table
    return values[tuple(environment[name] for name in names)]


@dataclass(frozen=True)
class Instance:
    export: TensorExport
    imp: TensorImport
    sites: tuple[Site, ...]
    resources: tuple[Lend, ...]
    derived: Derived
    profile: Any = field(compare=False, repr=False)
    catalogue: Mapping[str, Primitive] = field(compare=False, repr=False)
    identities: tuple[tuple[int, ...], ...]
    cells: tuple[Cell, ...]
    packets: tuple[Packet, ...]
    demands: tuple[tuple[int, int, int], ...]
    worlds: tuple[tuple[tuple[str, int], ...], ...]
    initial_cells: tuple[tuple[int, ...], ...]
    world_demands: tuple[tuple[tuple[int, int, int], ...], ...]
    actors: tuple[tuple[str, int], ...]
    participants: Mapping[str, tuple[int, ...]]
    monitors: tuple[Monitor, ...]
    result_resource: str
    warp_size: int
    packet_columns: int
    scalar_bytes: int
    program_ir: ir.Program | None = field(default=None, compare=False, repr=False)
    _geometry_cache: dict[int, Any] = field(default_factory=dict, compare=False, repr=False)
    _round_address_cache: dict[tuple, tuple[Any, ...]] = field(default_factory=dict, compare=False, repr=False)
    _export_storage_cache: dict[str, dict[str, Any]] = field(default_factory=dict, compare=False, repr=False)
    _expressions: dict[int, tuple[expr.Expr | Exception, ...]] = field(default_factory=dict, compare=False, repr=False)
    program_errors: tuple[str, ...] = ()
    max_operations: int | None = None
    _cell_index: Mapping[Cell, int] = field(default_factory=dict, compare=False, repr=False)
    _groups: tuple[tuple[int, ...], ...] = field(default=(), compare=False, repr=False)
    _results: tuple[tuple[tuple[tuple[int, int], ...], ...], ...] = field(default=(), compare=False, repr=False)
    _invalid_returns: frozenset[tuple[str, int]] = field(default=frozenset(), compare=False, repr=False)
    _fixed_monitors: bool = field(default=False, compare=False, repr=False)
    _static_metadata: frozenset[tuple[str, str]] = field(default=frozenset(), compare=False, repr=False)
    _view_results: tuple[tuple[tuple[int, int], ...], ...] = field(default=(), compare=False, repr=False)
    _readonly_cells: frozenset[int] = field(default=frozenset(), compare=False, repr=False)
    accessed_bytes: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    @property
    def demand_kind(self) -> str:
        return self.imp.demands[0].kind

    @property
    def tmem_lanes(self) -> int | None:
        return tmem_lane_count(self.profile)

    def tmem_subpartition(self, participant: int) -> int:
        return (participant // self.warp_size) % (self.tmem_lanes // self.warp_size) * self.warp_size

    @classmethod
    def build(cls, export, imp, derived, profile) -> Instance:
        sites, resources = derived.sites, derived.resources
        cap = _option(profile, "grounding_cap", 1_000_000)
        warp_size = _option(profile, "warp_size", 32)
        packet_columns = _option(profile, "packet_columns", 16)
        if type(warp_size) is not int or warp_size <= 0 or warp_size & (warp_size - 1) or type(packet_columns) is not int or packet_columns <= 0:
            raise ModelError(["invalid finite target packet geometry"])
        lanes = tmem_lane_count(profile)
        if lanes is None and ("tmem_packet_receipt" in export.read_capabilities
                              or any(placement.space == "tmem" for placement in export.placements)):
            raise ModelError(["source placement space tmem is absent from the target profile"], status="CONTRACT_INVALID")
        if "tmem_packet_receipt" in export.read_capabilities and (lanes < warp_size or lanes % warp_size):
            raise ModelError(["TMEM target lanes must contain an integral number of warp subpartitions"])
        if derived.catalogue not in CATALOGUES:
            raise ModelError([f"unknown primitive catalogue {derived.catalogue!r}"])
        if "tmem_packet_receipt" in export.read_capabilities and (
                derived.early_reclaim or export.readiness.after == export.readiness.reuse_after):
            raise ModelError(["tmem_packet_receipt source-stability premise is not established from readiness through import completion/reclaim"])
        if export.readiness.after == imp.completion:
            raise ModelError(["source readiness cannot depend on the importing consumer's completion"])
        if export.identity != imp.identity:
            raise ModelError(["export and import logical identity domains differ"], status="CONTRACT_INVALID")
        scalar_bytes = {"fp32": 4, "float32": 4, "f32": 4, "fp16": 2, "bf16": 2, "f16": 2, "int32": 4, "uint32": 4, "i32": 4, "u32": 4}.get(export.identity.dtype)
        if scalar_bytes is None:
            raise ModelError([f"unsupported payload width {export.identity.dtype!r}"])
        if "tmem_packet_receipt" in export.read_capabilities and scalar_bytes != 4:
            raise ModelError(["32dp32b packet receipt requires a 32-bit payload"])
        if set(imp.sites) != {site.name for site in sites}:
            raise ModelError(["import sites must resolve to existing derived sites"], status="CONTRACT_INVALID")
        binding_domains = derived.binding_domains
        if not set(export.residual_bindings) <= set(binding_domains):
            raise ModelError(["residual binding domain is unavailable"])
        worlds_env = _environments(binding_domains, cap)
        participants = {}
        for site in sites:
            if site.participants.set is not None:
                members = tuple(sorted(site.participants.set))
            else:
                domains = {domain.name: domain for domain in site.participants.domain}
                table = _grounded(site.participants.expr, domains, cap)
                members = tuple(sorted(set(table[1].values())))
            if not members:
                raise ModelError([f"site {site.name!r} has no participants"], status="CONTRACT_INVALID")
            participants[site.name] = members
        actors = tuple((site.name, member) for site in sites for member in participants[site.name])
        if len({member for _, member in actors}) != len(actors):
            raise ModelError(["overlapping physical participants at concurrent sites require an owner scheduling model"])
        usable = tuple(resource for resource in resources if resource.name in derived.storage)
        for resource in usable:
            if resource.extent <= 0 or resource.align & (resource.align - 1):
                raise ModelError([f"resource {resource.name!r} has invalid extent/alignment"], status="CONTRACT_INVALID")
            if resource.location.base is not None and resource.location.base % resource.align:
                raise ModelError([f"resource {resource.name!r} has a misaligned fixed base"], status="CONTRACT_INVALID")
            if resource.domain == "smem" and resource.location.base is not None and resource.location.base % scalar_bytes:
                raise ModelError([f"resource {resource.name!r} cannot hold aligned scalar accesses"], status="CONTRACT_INVALID")
            if resource.domain in {"smem", "barrier"} and resource.scope != "cta" or resource.domain == "reg" and resource.scope != "thread":
                raise ModelError([f"resource {resource.name!r} scope has no initial-model storage interpretation"])
            if resource.name in derived.readonly_storage:
                continue  # owner backing: covered by the source-stability premise, not a lend lifetime
            if resource.lifetime["from"] not in {export.readiness.after, *(site.invocation for site in sites)}:
                raise ModelError([f"resource {resource.name!r} start event has no finite invocation/readiness interpretation"])
            if f"resource_{resource.name}_available_until_{imp.completion}" not in derived.assumptions:
                raise ModelError([f"resource {resource.name!r} lifetime through import completion is not established"])
        registers = tuple(resource for resource in usable if resource.domain == "reg" and resource.name not in derived.readonly_storage)
        kinds = {demand.kind for demand in imp.demands}
        if len(kinds) != 1:
            raise ModelError(["one import must use one result kind"])
        view = kinds == {"memory_view"}
        if view and any(demand.space != "smem" or demand.units != "byte" or set(demand.components) != {"address"}
                        for demand in imp.demands):
            raise ModelError(["memory_view requires shared byte-address demands"])
        result_names = [param.name for param in imp.callable.params if param.meaning == "dest_fragment"]
        named_results = [resource for resource in registers if resource.name in result_names or resource.location.handle in result_names]
        if view:
            if len([param for param in imp.callable.params if param.meaning == "dest_view"]) != 1:
                raise ModelError(["memory_view must identify one destination view callable parameter"])
            result_resource = ""
        elif len(named_results) == 1:
            result_resource = named_results[0].name
        elif len(registers) == 1:
            result_resource = registers[0].name
        else:
            raise ModelError(["destination fragment must identify one declared private resource"])
        coords = {domain.name: domain for domain in export.identity.coords}
        identity_envs = _environments(coords, cap)
        coord_names = sorted(coords)
        identities = tuple(tuple(environment[name] for name in coord_names) for environment in identity_envs)
        identity_index = {identity: index for index, identity in enumerate(identities)}
        locations = set()
        source_worlds = []
        raw_demands = []
        # tmem placements exist only past the boundary check above, so the bounds are the target's.
        tmem_bounds = None if lanes is None else (lanes, tmem_column_count(profile))
        for bindings in worlds_env:
            singleton_bindings = {name: expr.Domain(name, binding_domains[name].sort, values=(value,)) for name, value in bindings.items()}
            source = {}
            for placement in export.placements:
                domains = {**coords, **singleton_bindings}
                tables = {name: _grounded(value, domains, cap) for name, value in placement.components.items()}
                guard = _grounded(placement.guard, domains, cap) if placement.guard else None
                for q, environment in enumerate(identity_envs):
                    environment = {**environment, **bindings}
                    if guard and not _lookup(guard, environment):
                        continue
                    components = {name: _lookup(table, environment) for name, table in tables.items()}
                    if placement.space == "tmem" and set(components) == {"dp", "column"} and placement.units == "dp/column":
                        cell = Cell("@tmem", components["dp"], components["column"])
                    elif placement.space == "reg" and set(components) == {"participant", "position"} and placement.units in {"scalar", "register", "slot"}:
                        cell = Cell("@export:reg", components["participant"], components["position"])
                    elif placement.space == "smem" and set(components) == {"address"} and placement.units in {"byte", "bytes"}:
                        cell = Cell("@export:smem", -1, components["address"])
                    else:
                        raise ModelError([f"unsupported placement components/units in {placement.space}"])
                    if cell.index < 0 or cell.participant < -1 or placement.space == "smem" and cell.index % scalar_bytes:
                        raise ModelError(["misaligned or negative source placement"], status="CONTRACT_INVALID")
                    if placement.space == "tmem" and (cell.participant >= tmem_bounds[0] or cell.index >= tmem_bounds[1]):
                        raise ModelError(["source placement exceeds target TMEM address bounds"], status="CONTRACT_INVALID")
                    if cell in source and source[cell] != q:
                        raise ModelError(["distinct logical identities alias one source cell"], status="CONTRACT_INVALID")
                    source[cell] = q
                    locations.add(cell)
            source_worlds.append(source)
            demands = set()
            positions = {}
            for demand in imp.demands:
                domains = {domain.name: domain for domain in demand.domain}
                domains.update(singleton_bindings)
                tables = {name: _grounded(value, domains, cap) for name, value in demand.identity_coords.items()}
                receiver = None if view else _grounded(demand.participant, domains, cap)
                position = _grounded(demand.components["address"] if view else demand.position, domains, cap)
                guard = _grounded(demand.guard, domains, cap) if demand.guard else None
                for environment in _environments(domains, cap):
                    if guard and not _lookup(guard, environment):
                        continue
                    identity = tuple(_lookup(tables[name], environment) for name in coord_names)
                    if identity not in identity_index:
                        raise ModelError(["demand names an identity outside the export domain"], status="CONTRACT_INVALID")
                    member, slot, q = -1 if view else _lookup(receiver, environment), _lookup(position, environment), identity_index[identity]
                    if slot < 0 or (slot % scalar_bytes if view else member not in {member for _, member in actors}):
                        raise ModelError(["demand receiver/address is outside the declared aligned result domain"], status="CONTRACT_INVALID")
                    if (member, slot) in positions and positions[member, slot] != q:
                        raise ModelError(["one result position demands distinct identities"], status="CONTRACT_INVALID")
                    positions[member, slot] = q
                    demands.add((member, slot, q))
            raw_demands.append(tuple(sorted(demands)))
        for resource in usable:
            if resource.domain in {"reg", "smem"}:
                members = sorted({member for _, member in actors}) if resource.domain == "reg" else (-1,)
                stride = scalar_bytes if resource.domain == "smem" else 1
                for member in members:
                    for offset in range(0, resource.extent - stride + 1, stride):
                        locations.add(cls._resource_cell(resource, member, offset))
        cells = tuple(sorted(locations))
        cell_index = {cell: index for index, cell in enumerate(cells)}
        packets = []
        packet_keys = sorted({(cell.participant // warp_size * warp_size, cell.index // packet_columns * packet_columns)
                              for cell in cells if cell.resource == "@tmem"})
        for dp, column in packet_keys:
            receipt = tuple((lane, offset, cell_index.get(Cell("@tmem", dp + lane, column + offset), -1))
                            for lane in range(warp_size) for offset in range(packet_columns))
            if all(source >= 0 for _, _, source in receipt):
                packets.append(Packet(dp, column, receipt))
        initial_cells = tuple(tuple(source.get(cell, EMPTY) for cell in cells) for source in source_worlds)
        monitors = (Monitor("justice", labels=frozenset({"owner.ready"}), name="owner_eventually_ready"),)
        instance = cls(export, imp, sites, resources, derived, profile, CATALOGUE(profile), identities, cells, tuple(packets),
                       tuple(sorted(set().union(*map(set, raw_demands)))), tuple(tuple(sorted(world.items())) for world in worlds_env),
                       initial_cells, tuple(raw_demands), actors, MappingProxyType(participants), monitors, result_resource,
                       warp_size, packet_columns, scalar_bytes, _cell_index=MappingProxyType(cell_index),
                       _groups=tuple(tuple(index for index, (other_site, other) in enumerate(actors)
                                           if other_site == site and other // warp_size == member // warp_size)
                                     for site, member in actors))
        result_tables = []
        actor_index = {member: actor for actor, (_, member) in enumerate(actors)}
        for demands in instance.world_demands:
            by_actor = [[] for _ in actors]
            for member, slot, q in demands:
                if view:
                    continue
                location = instance._declared_cell(result_resource, member, slot)
                if location not in cell_index:
                    raise ModelError(["demand exceeds declared destination capacity"], status="CONTRACT_INVALID")
                by_actor[actor_index[member]].append((cell_index[location], q))
            result_tables.append(tuple(tuple(rows) for rows in by_actor))
        readonly_cells = frozenset(
            cell_index[cls._resource_cell(resource, member, offset)]
            for resource in usable if resource.name in derived.readonly_storage and resource.domain in {"reg", "smem"}
            for member in (sorted({member for _, member in actors}) if resource.domain == "reg" else (-1,))
            for offset in range(0, resource.extent - (scalar_bytes if resource.domain == "smem" else 1) + 1,
                                scalar_bytes if resource.domain == "smem" else 1)
        )
        return replace(instance, _results=tuple(result_tables), _readonly_cells=readonly_cells)

    @staticmethod
    def _resource_cell(resource: Lend, participant: int, index: int) -> Cell:
        if resource.domain == "smem":
            if resource.location.base is not None:
                return Cell("@smem", -1, resource.location.base + index)
            return Cell(resource.location.handle or resource.name, -1, index)
        return Cell(resource.location.handle or resource.name, participant, index)

    def _resource(self, name: str) -> Lend:
        resource = next((item for item in self.resources if item.name == name and item.name in self.derived.storage), None)
        if resource is None:
            raise ValueError(f"undeclared or unpermitted resource {name!r}")
        return resource

    def _available(self, resource: Lend, state: State) -> bool:
        return resource.lifetime["from"] != self.export.readiness.after or state.owner_phase >= 1

    def _declared_cell(self, name: str, participant: int, index: int) -> Cell:
        if name == "@export:reg":
            return Cell(name, participant, index)
        if name == "@export:smem":
            return Cell(name, -1, index)
        resource = self._resource(name)
        stride = self.scalar_bytes if resource.domain == "smem" else 1
        if resource.domain not in {"reg", "smem"} or index < 0 or index + stride > resource.extent or index % stride:
            raise ValueError("cell is outside a scalar-aligned payload resource")
        return self._resource_cell(resource, participant, index)

    def exported_storage(self) -> dict[str, dict[str, Any]]:
        """Borrowed pointer views, in scalar slots (reg) or relative bytes (smem)."""
        if self._export_storage_cache:
            return self._export_storage_cache
        extents = {}
        for cell in self.cells:
            if cell.resource.startswith("@export:"):
                stride = self.scalar_bytes if cell.resource == "@export:smem" else 1
                extents[cell.resource] = max(extents.get(cell.resource, 0), cell.index + stride)
        self._export_storage_cache.update({
            name: {"domain": name.split(":", 1)[1], "scalar_bytes": self.scalar_bytes,
                   "extent": extent, "base_convention": "relative_to_source_base"}
            for name, extent in sorted(extents.items())})
        return self._export_storage_cache

    def program(self, graphs: Mapping[str, ir.ControlGraph], *, view_binding: str | None = None) -> ir.Program:
        borrows = dict(self.export.fixed_bindings)
        for name in self.export.residual_bindings:
            borrows[name] = name
        return ir.Program(tuple(resource for resource in self.resources if resource.name in self.derived.storage), borrows, dict(graphs), self.demands, view_binding)

    def _compile_operands(self, program: ir.Program) -> dict[int, tuple[expr.Expr | Exception, ...]]:
        domains = self._binding_domains()
        domains["participant"] = expr.Domain("participant", "int", values=tuple(sorted({member for _, member in self.actors})))
        for graph in program.graphs.values():
            for node in graph.nodes:
                name = node.dest if isinstance(node, ir.Meta) else node.operands.get("dest") if isinstance(node, ir.Issue) and node.primitive == "observe_ready" else None
                if isinstance(name, str) and name and name not in domains:
                    domains[name] = expr.Domain(name, "int", values=(0,))
        round_domains = {**domains, "slot": expr.Domain("slot", "int", values=(0,))}
        expressions = {}

        def compile(value, *, round_address=False):
            if type(value) is int or id(value) in expressions:
                return
            parsed = []
            for world in self.worlds if round_address else (None,):
                try:
                    source = _bind_constants(value, dict(world), domains) if round_address else value
                    parsed.append(expr.parse(source, round_domains if round_address else domains))
                except (ValueError, KeyError, IndexError, TypeError, expr.ExprError) as error:
                    parsed.append(error)
            expressions[id(value)] = tuple(parsed)

        def operands(values, *, round_address=False):
            for name, value in values.items():
                if name in {"index", "participant", "packet", "offset", "mask", "address"}:
                    compile(value, round_address=round_address and name == "address")
                elif isinstance(value, Mapping):
                    operands(value)

        for graph in program.graphs.values():
            for node in graph.nodes:
                if isinstance(node, ir.Meta):
                    compile(node.expr)
                elif isinstance(node, ir.Issue):
                    operands(node.operands, round_address=node.primitive in _ROUND_SCALARS)
        return expressions

    def with_program(self, program: ir.Program, *, max_operations: int | None = None) -> Instance:
        errors = []
        if set(program.graphs) != set(self.participants):
            errors.append("program must attach exactly one graph to every declared import site")
        if program.results != self.demands:
            errors.append("program result relation differs from the frozen import")
        expected = self.program(program.graphs)
        if program.resources != expected.resources or dict(program.borrows) != dict(expected.borrows):
            errors.append("program resources/borrowed bindings differ from the derived lends")
        view_results = []
        if self.demand_kind == "memory_view":
            try:
                if program.view_binding is None:
                    raise ValueError("memory_view program has no bound view resource")
                if program.view_binding == "@export:smem":
                    if "@export:smem" not in self.exported_storage():
                        raise ValueError("alias view requires an exported shared source")
                elif self._resource(program.view_binding).domain != "smem":
                    raise ValueError("view binding must name declared shared storage")
                for demands in self.world_demands:
                    view_results.append(tuple((self._cell_index[self._declared_cell(program.view_binding, -1, address)], identity)
                                              for _, address, identity in demands))
            except (ValueError, KeyError) as error:
                errors.append(f"invalid view binding: {error}")
        elif program.view_binding is not None:
            errors.append("positions program cannot bind a memory view")
        monitors = list(self.monitors)
        labels_by_actor = {actor: set() for actor in range(len(self.actors))}
        demand_set = frozenset(self.demands)
        invalid_returns = set()
        static_metadata = {}
        for site, graph in sorted(program.graphs.items()):
            members = self.participants.get(site, ())
            for pc, node in enumerate(graph.nodes):
                if isinstance(node, ir.Meta):
                    key = (site, node.dest)
                    static_metadata[key] = static_metadata.get(key, True) and is_static_expression(node.expr)
                elif isinstance(node, ir.Issue) and node.primitive == "observe_ready":
                    static_metadata[site, node.operands.get("dest")] = False
                if isinstance(node, ir.Return) and node.results and not set(node.results) <= demand_set:
                    invalid_returns.add((site, pc))
                if isinstance(node, ir.Issue) and node.primitive == "tmem_packet_receipt":
                    handle = node.operands.get("handle", "receipt")
                    for warp in sorted({member // self.warp_size for member in members}):
                        label = f"complete:{site}:{warp}:{handle}"
                        monitor = Monitor("justice", labels=frozenset({label}), name=label)
                        if monitor not in monitors:
                            monitors.append(monitor)
            for actor, (actor_site, member) in enumerate(self.actors):
                if site == actor_site:
                    labels_by_actor[actor].add(f"step:{site}:{member}")
                    labels_by_actor[actor].update(f"collective:{site}:{member // self.warp_size}:{pc}" for pc in range(len(graph.nodes)))
        if "cta_eventually_scheduled" in self.derived.assumptions:
            monitors.extend(Monitor("justice", labels=frozenset(labels), name=f"schedule:{actor}") for actor, labels in labels_by_actor.items())
        return replace(self, program_ir=program, program_errors=tuple(errors), monitors=tuple(monitors),
                       max_operations=max_operations, _invalid_returns=frozenset(invalid_returns), _expressions=self._compile_operands(program),
                       _view_results=tuple(view_results), accessed_bytes={},
                       _static_metadata=frozenset(key for key, static in static_metadata.items() if static),
                       _fixed_monitors=all(monitor.step is _stay and monitor.initial in monitor.states for monitor in monitors))

    @property
    def ready_at_entry(self) -> bool:
        return self.derived.ready_at_entry

    def initial(self) -> State:
        positions = tuple(self.program_ir.graphs[site].entry if self.program_ir and site in self.program_ir.graphs else 0 for site, _ in self.actors)
        phase = 1 if self.ready_at_entry else 0
        world = 0 if len(self.worlds) == 1 else -1
        cells = self.initial_cells[0] if world == 0 else (EMPTY,) * len(self.cells)
        return State(cells, cells, (), (), (), phase, positions, ((),) * len(self.actors),
                     (0,) * len(self._barriers()), world, self.program_errors, tuple(monitor.initial for monitor in self.monitors))

    def _barriers(self) -> tuple[Lend, ...]:
        return tuple(resource for resource in self.resources if resource.domain == "barrier"
                     and resource.name in self.derived.storage and resource.name not in self.derived.readonly_storage)

    def _node(self, state: State, actor: int) -> ir.Node | None:
        if self.program_ir is None or state.positions[actor] < 0:
            return None
        site, _ = self.actors[actor]
        return self.program_ir.graphs[site].nodes[state.positions[actor]]

    def _members(self, actor: int) -> tuple[int, ...]:
        if self._groups:
            return self._groups[actor]
        site, member = self.actors[actor]
        return tuple(index for index, (other_site, other) in enumerate(self.actors)
                     if other_site == site and other // self.warp_size == member // self.warp_size)

    def controller_footprint(self, state: State, actor: int, pc: int, *, slot: int | None = None) -> frozenset[tuple] | None:
        """Conservative exact-address footprint, including control and lifetimes.

        A future instruction depending on mutable metadata is not projected:
        callers must retain every interleaving rather than guess its address.
        Read/read overlap is conservatively considered dependent.
        """
        if pc < 0:
            return frozenset()
        site, participant = self.actors[actor]
        node = self.program_ir.graphs[site].nodes[pc]
        footprint = {("pc", actor)}
        bindings = dict(self.worlds[state.world])

        def integer(value, who=actor, round_slot=None):
            def inspect(item):
                if isinstance(item, expr.Expr):
                    item = expr.to_json_obj(item)
                if isinstance(item, (tuple, list)):
                    if item and item[0] in {"coord", "bind"} and item[1] not in {"participant", *bindings} and not (round_slot is not None and item[:2] == ["coord", "slot"]):
                        raise ValueError("mutable metadata in future footprint")
                    for child in item:
                        inspect(child)
            inspect(value)
            return self._integer(value, state, who, slot=round_slot)

        def cell(operand, owner=participant):
            member = integer(operand.get("participant", owner))
            if member != owner:
                raise ValueError("remote scalar cell")
            location = self._declared_cell(operand["resource"], member, integer(operand["index"]))
            index = self._cell_index[location]
            footprint.add(("cell", index))
            if self._space(operand) == "smem":
                footprint.add(("epoch", index))
            if self.initial_cells[state.world][index] != EMPTY and (state.owner_phase == 0 or self.derived.early_reclaim):
                footprint.add(("owner",))

        def capacity():
            for resource in self.resources:
                if resource.domain == "metadata" and resource.name in self.derived.storage:
                    owner = participant if resource.scope == "thread" else participant // self.warp_size if resource.scope == "warp" else -1
                    footprint.add(("metadata-capacity", resource.name, owner))

        try:
            if isinstance(node, ir.Return):
                footprint.update(("cell", index) for index, _ in self._results[state.world][actor])
                if self.demand_kind == "memory_view":
                    footprint.add(("owner",))
                    footprint.update((kind, index) for index, _ in self._view_results[state.world]
                                     for kind in ("cell", "epoch"))
            elif isinstance(node, ir.Branch):
                footprint.add(("owner",) if node.observation in {"ready", self.export.readiness.after} else ("metadata", actor, node.observation))
            elif isinstance(node, ir.Meta):
                integer(node.expr)
                footprint.add(("metadata", actor, node.dest))
                capacity()
            elif isinstance(node, ir.Issue):
                operands = node.operands
                if node.primitive in {"reg_copy", "smem_store_scalar", "smem_load_scalar"}:
                    cell(operands["src"])
                    cell(operands["dst"])
                elif node.primitive in _ROUND_SCALARS:
                    footprint.add(("round", actor))
                    width = node.operands["width"]
                    if type(width) is not int or width <= 0:
                        return None
                    for element in range(width) if slot is None else (slot,):
                        # Refuse mutable metadata before projecting future slots.
                        if "address" in node.operands:
                            integer(node.operands["address"], round_slot=element)
                        scalar = self._round_operands(node, state, actor, element)
                        cell(scalar["src"])
                        cell(scalar["dst"])
                elif node.primitive == "observe_ready":
                    footprint.update({("owner",), ("metadata", actor, operands["dest"])})
                    capacity()
                elif node.primitive in {"tmem_packet_receipt", "shfl_xor"}:
                    members = self._members(actor)
                    footprint.update(("pc", member) for member in members)
                    footprint.add(("handles", site, participant // self.warp_size))
                    if node.primitive == "tmem_packet_receipt":
                        capacity()
                        if operands.get("phase", "issue") == "issue":
                            packet = self.packets[integer(operands["packet"])]
                            offset = integer(operands.get("offset", 0))
                            warp = participant // self.warp_size
                            for lane, slot, source in packet.receipt:
                                footprint.add(("cell", source))
                                footprint.add(("cell", self._cell_index[self._declared_cell(operands["dest"], warp * self.warp_size + lane, offset + slot)]))
                            if state.owner_phase == 0:
                                footprint.add(("owner",))
                    else:
                        # A shuffle can read any lane in the full declared warp.
                        for member in members:
                            for operand in (operands["src"], operands["dst"]):
                                other = self.actors[member][1]
                                location = self._declared_cell(operand["resource"], other, integer(operand["index"], member))
                                footprint.add(("cell", self._cell_index[location]))
                elif node.primitive == "cta_rendezvous":
                    footprint.add(("barrier", operands["resource"]))
                    # An arrival does not publish.  The separate ENV release
                    # touches the CTA's cells only after every participant arrives.
                else:
                    return None
            else:
                return None
        except (ValueError, KeyError, IndexError, TypeError, expr.ExprError):
            return None
        return frozenset(footprint)

    def transition_footprint(self, state: State, transition: Transition) -> frozenset[tuple] | None:
        if transition.site != "ENV":
            actor = transition.actors[0]
            return self.controller_footprint(state, actor, state.positions[actor],
                                             slot=state.round_positions[actor] if transition.primitive in _ROUND_SCALARS else None)
        if transition.label == "owner.ready":
            return frozenset({("owner",)})
        if transition.label == "owner.reclaim":
            return frozenset({("owner",), *(("cell", index) for index, value in enumerate(self.initial_cells[state.world]) if value != EMPTY)})
        if transition.label.startswith("complete:"):
            pending = next(item for item in state.outstanding if item.label == transition.label)
            return frozenset({("handles", pending.site, pending.warp), *(("cell", index) for index in pending.sources + pending.destinations)})
        if transition.label.startswith("rendezvous:"):
            footprint = {("barrier", transition.detail), *(("pc", actor) for actor in transition.actors)}
            for index, location in enumerate(self.cells):
                if location.participant == -1:
                    footprint.update({("cell", index), ("epoch", index)})
            return frozenset(footprint)
        return None

    def enabled(self, state: State) -> tuple[Transition, ...]:
        if state.errors:
            return ()
        if state.world < 0:
            return tuple(Transition("ENV", f"binding:{world}", None, False, detail=str(world)) for world in range(len(self.worlds)))
        transitions = []
        if state.owner_phase == 0:
            transitions.append(Transition("ENV", "owner.ready", None, False))

        if state.owner_phase == 1 and self.derived.early_reclaim:
            transitions.append(Transition("ENV", "owner.reclaim", None, False))
        for pending in state.outstanding:
            if not pending.completed:
                transitions.append(Transition("ENV", pending.label, "tmem_packet_receipt", False, detail=pending.label))
        for arrival in state.arrivals:
            barrier = self._resource(arrival.resource)
            expected = tuple(range(len(self.actors)))
            if arrival.actors == expected and barrier.sync is not None and barrier.sync.capacity >= 1:
                transitions.append(Transition("ENV", f"rendezvous:{arrival.resource}", "cta_rendezvous", False, expected, arrival.resource))
        for actor, (site, member) in enumerate(self.actors):
            node = self._node(state, actor)
            if node is None:
                continue
            if state.arrivals and any(actor in arrival.actors for arrival in state.arrivals):
                continue
            primitive = node.primitive if isinstance(node, ir.Issue) else "observe_ready" if isinstance(node, ir.Branch) and node.observation in {"ready", self.export.readiness.after} else None
            if primitive in {"tmem_packet_receipt", "shfl_xor"}:
                members = self._members(actor)
                if actor != members[0]:
                    continue
                if any(state.positions[other] != state.positions[actor] or self._node(state, other) != node for other in members):
                    continue
                if primitive == "tmem_packet_receipt" and node.operands.get("phase", "issue") == "wait":
                    handle = node.operands.get("handle", "receipt")
                    pending = next((pending for pending in state.outstanding if pending.site == site and pending.warp == member // self.warp_size and pending.handle == handle), None)
                    if pending is not None and not pending.completed:
                        continue
                transitions.append(Transition(site, f"collective:{site}:{member // self.warp_size}:{state.positions[actor]}", primitive, False, members))
            else:
                transitions.append(Transition(site, f"step:{site}:{member}", primitive, isinstance(node, ir.Branch) or primitive == "observe_ready",
                                              (actor,), f"round-slot:{state.round_positions[actor]}" if primitive in _ROUND_SCALARS else ""))
        return tuple(transitions)

    def apply(self, state: State, t: Transition) -> State:
        if t not in self.enabled(state):
            raise ValueError("transition is not enabled in this exact state")
        return self._apply_enabled(state, t)

    def _apply_enabled(self, state: State, t: Transition) -> State:
        """Checker-internal step after membership in enabled(state) is proven."""
        try:
            result = self._apply(state, t)
        except (ValueError, KeyError, IndexError, TypeError, expr.ExprError) as error:
            result = replace(state, errors=(str(error),))
        if self._fixed_monitors:
            monitor_states = state.monitor_states
        else:
            monitor_states = tuple(monitor.step(mstate, t.label) for monitor, mstate in zip(self.monitors, state.monitor_states))
            if any(value not in monitor.states for monitor, value in zip(self.monitors, monitor_states)):
                raise ModelError(["monitor transition escapes its declared finite states"])
        operations = state.operations + (t.site != "ENV") if self.max_operations is not None else 0
        if self.max_operations is not None and operations > self.max_operations:
            result = replace(result, errors=result.errors + ("candidate exceeds declared max_operations bound",))
        if monitor_states is result.monitor_states and operations == result.operations:
            return result
        return replace(result, monitor_states=monitor_states, operations=operations)

    def _environment(self, state: State, actor: int) -> dict[str, int]:
        environment = dict(self.worlds[state.world])
        environment.update(state.metadata[actor])
        environment["participant"] = self.actors[actor][1]
        return environment

    def _integer(self, value: Any, state: State, actor: int, *, slot: int | None = None) -> int:
        if type(value) is int:
            return value
        environment = self._environment(state, actor)
        if slot is not None:
            environment["slot"] = slot
        expression = self._expressions[id(value)][state.world if slot is not None else 0]
        if isinstance(expression, Exception):
            raise expression
        return expr.evaluate(expression, environment)

    def _cell(self, operand: Any, state: State, actor: int, *, write: bool = False) -> int:
        if not isinstance(operand, Mapping) or not {"resource", "index"} <= set(operand) or set(operand) - {"resource", "index", "participant"}:
            raise ValueError("cell operand requires resource, index, and optional participant")
        member = self._integer(operand.get("participant", self.actors[actor][1]), state, actor)
        if member != self.actors[actor][1]:
            raise ValueError("scalar action cannot access another participant's private cells")
        name = operand["resource"]
        if not name.startswith("@export:") and not self._available(self._resource(name), state):
            raise ValueError("resource accessed before its declared lifetime start")
        if write and (name.startswith("@export:") or name in self.derived.readonly_storage):
            raise ValueError("read-only export cannot be overwritten by the adapter")
        if name.startswith("@export:") and state.owner_phase != 1:
            raise ValueError("source accessed before published readiness or after reclaim")
        offset = self._integer(operand["index"], state, actor)
        cell = self._declared_cell(name, member, offset)
        if cell not in self._cell_index:
            raise ValueError("cell is outside the grounded access domain")
        index = self._cell_index[cell]
        if write and index in self._readonly_cells:
            raise ValueError("write aliases read-only export backing")
        if self.initial_cells[state.world][index] != EMPTY:
            if write:
                raise ValueError("write aliases the read-only exported source")
            if state.owner_phase != 1:
                raise ValueError("source alias accessed before readiness or after reclaim")
        if not name.startswith("@export:") and self._resource(name).domain == "smem":
            self.accessed_bytes[name] = max(self.accessed_bytes.get(name, 0), offset + self.scalar_bytes)
        return index

    def _space(self, operand: Mapping[str, Any]) -> str:
        name = operand["resource"]
        if name.startswith("@export:"):
            return name.split(":", 1)[1]
        return self._resource(name).domain

    def _read(self, state: State, cell: int, actor: int, *, shared: bool = False) -> int:
        if any(not pending.completed and cell in pending.destinations for pending in state.outstanding):
            raise ValueError("asynchronous destination read before completion")
        own_write = shared and any(index == cell and other == actor and mode == "w" for index, other, mode in state.accesses)
        value = state.cells[cell] if not shared or own_write else state.published[cell]
        if value == EMPTY:
            raise ValueError("read of an unavailable identity (empty or reclaimed last copy)")
        return value

    def _access(self, state: State, cell: int, actor: int, mode: str) -> tuple[tuple[int, int, str], ...]:
        if any(index == cell and other != actor and (mode == "w" or previous == "w") for index, other, previous in state.accesses):
            raise ValueError("same-epoch conflicting shared access involving a write")
        return tuple(sorted(set(state.accesses) | {(cell, actor, mode)}))

    def _check_metadata_capacity(self, state: State, metadata: tuple, outstanding: tuple[Pending, ...]) -> None:
        # Nested thread/warp/CTA scopes form a capacity tree.  Private demand
        # consumes its narrowest storage first; only unmet demand goes upward.
        # Handles are warp-owned and cannot borrow another lane's private slot.
        capacities = {
            scope: sum(resource.extent for resource in self.resources
                       if resource.name in self.derived.storage and resource.name not in self.derived.readonly_storage
                       and resource.domain == "metadata"
                       and resource.scope == scope and self._available(resource, state))
            for scope in ("thread", "warp", "cta")
        }
        warp_demand = {}
        for actor, (site, participant) in enumerate(self.actors):
            warp = participant // self.warp_size
            retained = sum((site, name) not in self._static_metadata for name, _ in metadata[actor])
            warp_demand[warp] = warp_demand.get(warp, 0) + max(0, retained - capacities["thread"])
        for pending in outstanding:
            warp_demand[pending.warp] = warp_demand.get(pending.warp, 0) + 1
        if sum(max(0, demand - capacities["warp"]) for demand in warp_demand.values()) > capacities["cta"]:
            raise ValueError("metadata and asynchronous handles exceed declared scoped storage capacity")

    def _metadata(self, state: State, actor: int, name: str, value: int) -> tuple:
        if not isinstance(name, str) or not name or name == "participant" or name in dict(self.worlds[state.world]):
            raise ValueError("metadata destination shadows a published input")
        local = dict(state.metadata[actor])
        domain = _option(self.profile, "metadata_domain", (0, 1))
        if value not in domain:
            raise ValueError("metadata value escapes the declared finite target domain")
        local[name] = value
        metadata = list(state.metadata)
        metadata[actor] = tuple(sorted(local.items()))
        self._check_metadata_capacity(state, tuple(metadata), state.outstanding)
        return tuple(metadata)

    def _permission(self, actor: int, primitive: str) -> None:
        site = self.actors[actor][0]
        if primitive not in self.derived.permitted[site]:
            raise ValueError(f"primitive {primitive!r} is not permitted at this site")
        if primitive not in self.catalogue:
            raise ValueError("primitive is outside the independent catalogue")

    def _round_operands(self, node: ir.Issue, state: State, actor: int, slot: int) -> dict[str, Any]:
        operands = node.operands
        private_copy = node.primitive == "reg_copy_round"
        required = {"width", "src", "dst"} | (set() if private_copy else {"address"})
        optional = {"offset", "src_offset"} if private_copy else {"offset"}
        if set(operands) - required - optional or not required <= set(operands):
            raise ValueError("round operands do not match its private/shared scalar recipe")
        width, offset = operands["width"], operands.get("offset", 0)
        if type(width) is not int or width <= 0 or type(offset) is not int or offset < 0 or not 0 <= slot < width:
            raise ValueError("round has an invalid finite slot interval")

        def private_cell(name, start):
            if type(start) is not int or start < 0:
                raise ValueError("round has an invalid private offset")
            if name == "@export:reg":
                extent = self.exported_storage()[name]["extent"]
            else:
                resource = self._resource(name)
                if resource.domain != "reg":
                    raise ValueError("round private operand must be a register resource")
                extent = resource.extent
            if start + width > extent:
                raise ValueError("round exceeds its declared private payload capacity")
            return {"resource": name, "index": start + slot}

        if private_copy:
            return {"src": private_cell(operands["src"], operands.get("src_offset", 0)),
                    "dst": private_cell(operands["dst"], offset)}
        store = node.primitive == "smem_store_round"
        private = private_cell(operands["src"] if store else operands["dst"], offset)
        shared = {"resource": operands["dst"] if store else operands["src"],
                  "index": self._integer(operands["address"], state, actor, slot=slot)}
        return {"src": private, "dst": shared} if store else {"src": shared, "dst": private}

    def _apply(self, state: State, t: Transition) -> State:
        if t.label.startswith("binding:"):
            world = int(t.detail)
            return replace(state, world=world, cells=self.initial_cells[world], published=self.initial_cells[world])
        if t.label == "owner.ready":
            return replace(state, owner_phase=1)
        if t.label == "owner.reclaim":
            cells = tuple(EMPTY if initial != EMPTY else value for initial, value in zip(self.initial_cells[state.world], state.cells))
            published = tuple(EMPTY if initial != EMPTY else value for initial, value in zip(self.initial_cells[state.world], state.published))
            return replace(state, owner_phase=2, cells=cells, published=published)
        if t.label.startswith("complete:"):
            pending = next(pending for pending in state.outstanding if pending.label == t.detail)
            values = tuple(state.cells[source] for source in pending.sources)
            if any(value == EMPTY for value in values):
                raise ValueError("deferred source read found an unavailable/reclaimed value")
            if values != tuple(self.initial_cells[state.world][source] for source in pending.sources):
                raise ValueError("source-stability premise violated before deferred completion")
            cells = state.cells.updated(zip(pending.destinations, values))
            outstanding = tuple(replace(item, completed=True) if item == pending else item for item in state.outstanding)
            return replace(state, cells=cells, outstanding=outstanding)
        if t.label.startswith("rendezvous:"):
            positions = list(state.positions)
            for actor in t.actors:
                positions[actor] = self._node(state, actor).next
            barriers = self._barriers()
            index = next(index for index, resource in enumerate(barriers) if resource.name == t.detail)
            generations = list(state.generations)
            generations[index] = (generations[index] + 1) % barriers[index].sync.generations
            return replace(state, positions=tuple(positions), published=state.cells, accesses=(),
                           generations=tuple(generations), arrivals=tuple(arrival for arrival in state.arrivals if arrival.resource != t.detail))
        actor = t.actors[0]
        node = self._node(state, actor)
        positions = list(state.positions)
        if isinstance(node, ir.Return):
            if (t.site, state.positions[actor]) in self._invalid_returns:
                raise ValueError("return result relation differs from the frozen import")
            member = self.actors[actor][1]
            if any(pending.site == t.site and pending.warp == member // self.warp_size for pending in state.outstanding):
                raise ValueError("return abandons an unreleased asynchronous handle")
            accesses = state.accesses
            if self.demand_kind == "memory_view":
                if state.owner_phase != 1:
                    raise ValueError("view returned before published readiness or after reclaim")
                if self.program_ir.view_binding == "@export:smem" and not {"smem_load_scalar", "smem_load_round"} & set(self.export.read_capabilities):
                    raise ValueError("export does not permit shared view reads")
                for index, identity in self._view_results[state.world]:
                    if state.cells[index] != identity or state.published[index] != identity:
                        raise ValueError("returned view identity is missing, wrong, or not published")
                    accesses = self._access(replace(state, accesses=accesses), index, actor, "r")
            for index, q in self._results[state.world][actor]:
                # All asynchronous destinations belong to their issuing warp.
                # The handle check above therefore establishes availability for
                # every private result without rescanning each pending receipt.
                if state.cells[index] == EMPTY:
                    raise ValueError("read of an unavailable identity (empty or reclaimed last copy)")
                if state.cells[index] != q:
                    raise ValueError("returned result contains the wrong logical identity/version")
            positions[actor] = -1
            return replace(state, positions=tuple(positions), accesses=accesses)
        if isinstance(node, ir.Branch):
            if node.observation in {"ready", self.export.readiness.after}:
                self._permission(actor, "observe_ready")
                value = state.owner_phase >= 1
            else:
                metadata = dict(state.metadata[actor])
                if node.observation not in metadata:
                    raise ValueError("branch reads unavailable local metadata or hidden state")
                value = bool(metadata[node.observation])
            positions[actor] = node.next_true if value else node.next_false
            return replace(state, positions=tuple(positions))
        if isinstance(node, ir.Meta):
            metadata = self._metadata(state, actor, node.dest, self._integer(node.expr, state, actor))
            positions[actor] = node.next
            return replace(state, metadata=metadata, positions=tuple(positions))
        if not isinstance(node, ir.Issue):
            raise ValueError("missing local controller")
        primitive = node.primitive
        self._permission(actor, primitive)
        operands = node.operands
        for member in t.actors:
            positions[member] = node.next
        if primitive == "observe_ready":
            return replace(state, positions=tuple(positions), metadata=self._metadata(state, actor, operands["dest"], int(state.owner_phase >= 1)))
        if primitive == "cta_rendezvous":
            resource = self._resource(operands["resource"])
            if resource.name in self.derived.readonly_storage:
                raise ValueError("rendezvous cannot modify read-only export backing")
            if not self._available(resource, state):
                raise ValueError("rendezvous before declared synchronization lifetime")
            if resource.domain != "barrier" or resource.sync is None or resource.sync.generations <= 0:
                raise ValueError("rendezvous requires declared finite synchronization storage")
            # Capacity counts unpublished collective generations, not arrivals.
            # This blocking CTA catalogue has at most one generation per resource:
            # every participant must leave it before a new one can be started.
            if not any(arrival.resource == resource.name for arrival in state.arrivals) and resource.sync.capacity < 1:
                raise ValueError("rendezvous exceeds outstanding synchronization-generation capacity")
            site = next(site for site in self.sites if site.name == t.site)
            if "converged" not in site.guarantees:
                raise ValueError("rendezvous requires a published convergence obligation")
            arrivals = {arrival.resource: set(arrival.actors) for arrival in state.arrivals}
            arrivals.setdefault(resource.name, set()).add(actor)
            return replace(state, arrivals=tuple(Arrival(name, tuple(sorted(members))) for name, members in sorted(arrivals.items())))
        if primitive == "tmem_packet_receipt":
            if "tmem_packet_receipt" not in self.export.read_capabilities:
                raise ValueError("export does not offer packet reads")
            if len(t.actors) != self.warp_size or {self.actors[member][1] % self.warp_size for member in t.actors} != set(range(self.warp_size)):
                raise ValueError("packet issue requires all physical lanes of a full declared warp")
            warp = self.actors[actor][1] // self.warp_size
            handle = operands.get("handle", "receipt")
            if not isinstance(handle, str):
                raise ValueError("handle must name a local finite handle slot")
            old = next((pending for pending in state.outstanding if pending.site == t.site and pending.warp == warp and pending.handle == handle), None)
            phase = operands.get("phase", "issue")
            if phase == "wait":
                if old is None:
                    raise ValueError("completion wait names no outstanding handle")
                return replace(state, positions=tuple(positions), outstanding=tuple(item for item in state.outstanding if item != old))
            if phase != "issue":
                raise ValueError("packet phase must be issue or wait")
            if state.owner_phase != 1:
                raise ValueError("packet issued before published readiness")
            if old is not None:
                raise ValueError("packet issue reuses an outstanding handle")
            packet_indices = {self._integer(operands["packet"], state, member) for member in t.actors}
            offsets = {self._integer(operands.get("offset", 0), state, member) for member in t.actors}
            if len(packet_indices) != 1 or len(offsets) != 1:
                raise ValueError("collective packet operands are not warp-uniform")
            packet_index = packet_indices.pop()
            if not 0 <= packet_index < len(self.packets):
                raise ValueError("packet index is outside the exact receipt table")
            packet = self.packets[packet_index]
            if packet.dp != self.tmem_subpartition(self.actors[actor][1]):
                raise ValueError("tmem packet is outside the issuing warp's own DP subpartition")
            offset = offsets.pop()
            destination_resource = self._resource(operands["dest"])
            if destination_resource.domain != "reg":
                raise ValueError("packet receipt destination must be private registers")
            if not self._available(destination_resource, state):
                raise ValueError("receipt destination accessed before its declared lifetime start")
            destinations = tuple(self._cell_index[self._declared_cell(operands["dest"], warp * self.warp_size + lane, offset + slot)]
                                 for lane, slot, _ in packet.receipt)
            if destination_resource.name in self.derived.readonly_storage or any(
                    index in self._readonly_cells or self.initial_cells[state.world][index] != EMPTY for index in destinations):
                raise ValueError("packet receipt aliases read-only export backing")
            if any(set(destinations) & set(pending.destinations) for pending in state.outstanding if not pending.completed):
                raise ValueError("overlapping outstanding receipt destinations")
            pending = Pending(t.site, warp, handle, tuple(source for _, _, source in packet.receipt), destinations)
            self._check_metadata_capacity(state, state.metadata, state.outstanding + (pending,))
            return replace(state, positions=tuple(positions), outstanding=tuple(sorted(state.outstanding + (pending,))))
        if primitive == "shfl_xor":
            if len(t.actors) != self.warp_size:
                raise ValueError("shuffle requires all declared warp lanes")
            masks = {self._integer(operands["mask"], state, member) for member in t.actors}
            if len(masks) != 1 or not 0 <= next(iter(masks)) < self.warp_size:
                raise ValueError("shuffle xor mask must be warp-uniform and lane-bounded")
            if self._space(operands["src"]) != "reg" or self._space(operands["dst"]) != "reg":
                raise ValueError("shuffle operands must be private registers")
            mask = masks.pop()
            reads = {self.actors[member][1]: self._read(state, self._cell(operands["src"], state, member), member) for member in t.actors}
            changes = []
            for member in t.actors:
                destination = self._cell(operands["dst"], state, member, write=True)
                changes.append((destination, reads[self.actors[member][1] ^ mask]))
            return replace(state, positions=tuple(positions), cells=state.cells.updated(changes))
        round_positions = state.round_positions
        if primitive in _ROUND_SCALARS:
            slot = state.round_positions[actor]
            operands = self._round_operands(node, state, actor, slot)
            cursors = list(state.round_positions)
            cursors[actor] = slot + 1
            if cursors[actor] == node.operands["width"]:
                cursors[actor] = 0
            else:
                positions[actor] = state.positions[actor]
            round_positions = tuple(cursors)
            primitive = _ROUND_SCALARS[primitive]
        source = self._cell(operands["src"], state, actor)
        destination = self._cell(operands["dst"], state, actor, write=True)
        source_space, destination_space = self._space(operands["src"]), self._space(operands["dst"])
        expected = {"reg_copy": ("reg", "reg"), "smem_store_scalar": ("reg", "smem"), "smem_load_scalar": ("smem", "reg")}
        if primitive not in expected or (source_space, destination_space) != expected[primitive]:
            raise ValueError("scalar primitive operand address spaces are incompatible")
        if operands["src"]["resource"].startswith("@export:"):
            read_family = "reg_copy" if source_space == "reg" else primitive
            if read_family not in self.export.read_capabilities:
                raise ValueError("source read is not an exported capability")
        accesses = state.accesses
        if source_space == "smem":
            accesses = self._access(state, source, actor, "r")
        elif destination_space == "smem":
            accesses = self._access(state, destination, actor, "w")
        if any(destination in pending.destinations and not pending.completed for pending in state.outstanding):
            raise ValueError("overwrite of an outstanding asynchronous destination")
        value = self._read(state, source, actor, shared=source_space == "smem")
        cells = state.cells.updated(((destination, value),))
        published = state.published.updated(((destination, EMPTY),)) if destination_space == "smem" else state.published
        return replace(state, positions=tuple(positions), cells=cells, published=published,
                       accesses=accesses, round_positions=round_positions)

    def terminal(self, state: State) -> bool:
        return not state.errors and all(position == -1 for position in state.positions) and not state.outstanding and not state.arrivals

    def deliveries(self, state: State) -> tuple[tuple[int, int, int], ...]:
        if self.demand_kind == "memory_view":
            return tuple((-1, address, state.cells[index]) for (_, address, _), (index, _) in
                         zip(self.world_demands[state.world], self._view_results[state.world]))
        return tuple((member, slot, state.cells[self._cell_index[self._declared_cell(self.result_resource, member, slot)]])
                     for member, slot, _ in self.world_demands[state.world])

    def view_delivery(self, state: State) -> dict[str, Any]:
        if self.demand_kind != "memory_view":
            return {}
        return {"view": {"resource": self.program_ir.view_binding,
                         "cells": [[address, state.cells[index], state.published[index]]
                                   for (_, address, _), (index, _) in
                                   zip(self.world_demands[state.world], self._view_results[state.world])]}}

    def owner_adequacy(self, *, budget) -> dict[str, Any]:
        cap = budget if type(budget) is int else budget.get("max_states", 100000)
        needed = len(self.worlds) * (1 if self.ready_at_entry else 2)
        return {"ok": cap >= needed, "prerequisite": "finite owner ready publication" if cap >= needed else "owner adequacy state cap exceeded",
                "provenance": {"owner_states": needed, "binding_worlds": len(self.worlds), "readiness": self.export.readiness.after,
                               "reclaim_after": self.export.readiness.reuse_after, "readiness_at_entry": self.ready_at_entry,
                               "witness": ["binding", "pre-call-ready" if self.ready_at_entry else "owner.ready", "quiescent-ready"],
                               "assumptions": list(self.derived.assumptions)}}

    def _binding_domains(self) -> dict[str, expr.Domain]:
        worlds = [dict(world) for world in self.worlds]
        return {
            name: expr.Domain(name, self.derived.binding_domains[name].sort if name in self.derived.binding_domains else "int",
                              values=tuple(sorted({world[name] for world in worlds})))
            for name in worlds[0]
        }

    def _source_geometry(self, placement_index: int):
        """Prove a coordinate-independent column binding displacement."""
        if placement_index in self._geometry_cache:
            return self._geometry_cache[placement_index]
        placement = self.export.placements[placement_index]
        if placement.space != "tmem" or set(placement.components) != {"dp", "column"}:
            self._geometry_cache[placement_index] = None
            return None
        coordinates = sorted(self.export.identity.coords, key=lambda domain: domain.name)
        fixed = {domain.name: _constant(value, domain.sort) for domain, value in zip(coordinates, self.identities[0])}
        origin = _substitute(placement.components["column"], fixed)
        parsed_origin = expr.parse(origin, self._binding_domains())
        worlds = [dict(world) for world in self.worlds]
        origins = tuple(expr.evaluate(parsed_origin, world) for world in worlds)
        datapaths, relative_columns = set(), set()
        for identity in self.identities:
            point = dict(zip((domain.name for domain in coordinates), identity))
            baseline = {**point, **worlds[0]}
            active = placement.guard is None or bool(expr.evaluate(placement.guard, baseline))
            dp = expr.evaluate(placement.components["dp"], baseline)
            relative = expr.evaluate(placement.components["column"], baseline) - origins[0]
            for world, base in zip(worlds[1:], origins[1:]):
                environment = {**point, **world}
                if (placement.guard is not None and bool(expr.evaluate(placement.guard, environment)) != active
                    or expr.evaluate(placement.components["dp"], environment) != dp
                    or expr.evaluate(placement.components["column"], environment) - base != relative):
                    self._geometry_cache[placement_index] = None
                    return None
            if active:
                datapaths.add(dp)
                relative_columns.add(relative)
        if not datapaths or not relative_columns:
            geometry = None
        else:
            geometry = {"origin": origin, "origins": origins, "dp_min": min(datapaths), "dp_max": max(datapaths),
                        "column_min": min(relative_columns), "column_width": max(relative_columns) - min(relative_columns) + 1,
                        "binding_displacement": tuple(value - origins[0] for value in origins)}
            coordinate_domains = {domain.name: domain for domain in coordinates}
            geometry["dp_expression"] = expr.parse(_bind_constants(placement.components["dp"], worlds[0], self._binding_domains()), coordinate_domains)
            geometry["column_expression"] = expr.parse(
                ["sub", _bind_constants(placement.components["column"], worlds[0], self._binding_domains()), ["const", origins[0]]],
                coordinate_domains,
            )
        self._geometry_cache[placement_index] = geometry
        return geometry

    def _packet_operands(self, site: Site) -> tuple[Any, ...]:
        """Absolute constants plus source-geometry row-stride/binding selectors."""
        count = len(self.packets)
        if count == 0:
            return ()
        members = self.participants[site.name]
        # The only participant occurrence below is floor(participant/warp_size):
        # one representative per physical warp is therefore an exact quotient
        # for operand range/uniformity checking, not an execution reduction.
        representatives = tuple(min(member for member in members if member // self.warp_size == warp)
                                for warp in sorted({member // self.warp_size for member in members}))
        domains = {**self._binding_domains(), "participant": expr.Domain("participant", "int", values=representatives)}
        environments = _environments(domains, _option(self.profile, "grounding_cap", 1_000_000))
        def own_subpartitions(vector):
            return all(self.packets[packet].dp == self.tmem_subpartition(environment["participant"])
                       for packet, environment in zip(vector, environments))
        operands: list[Any] = [
            packet for packet in range(count) if own_subpartitions((packet,) * len(environments))
        ]
        seen = {tuple([packet] * len(environments)) for packet in operands}
        by_dp = {}
        for packet in self.packets:
            by_dp.setdefault(packet.dp, []).append(packet.column)
        columns = tuple(next(iter(by_dp.values())))
        if any(tuple(values) != columns for values in by_dp.values()):
            return tuple(operands)
        stride = len(columns)
        shifts: list[Any] = [["const", 0]]
        for index in range(len(self.export.placements)):
            geometry = self._source_geometry(index)
            if geometry is None or not any(geometry["binding_displacement"]):
                continue
            if any(displacement % self.packet_columns for displacement in geometry["binding_displacement"]):
                continue
            shifts.append(["divc", ["sub", geometry["origin"], ["const", geometry["origins"][0]]], self.packet_columns])
        for shift in shifts:
            for offset in range(count):
                selector = ["modc", ["add", ["mulc", stride, ["divc", ["coord", "participant"], self.warp_size]],
                                     ["add", ["const", offset], shift]], count]
                parsed = expr.parse(selector, domains)
                report = expr.relation_check({"packet": parsed}, domains, {"packet": (0, count)}, cap=len(environments))
                if not report.total or not report.in_bounds or report.capped:
                    continue
                table = _grounded(parsed, domains, len(environments))
                vector = tuple(_lookup(table, environment) for environment in environments)
                if own_subpartitions(vector) and vector not in seen:
                    seen.add(vector)
                    operands.append(selector)
        return tuple(operands)

    def _round_addresses(self, site: Site, width: int, name: str, extent: int) -> tuple[Any, ...]:
        """Finite public-contract algebra plus flat/xor/cyclic bank addresses.

        These are operand expressions, not schedules or witness programs.  Full
        private-bank rounds remain scalar-interleaved in the execution model.
        """
        key = (site.name, width, name)
        if key in self._round_address_cache:
            return self._round_address_cache[key]
        members = self.participants[site.name]
        participant = ["sub", ["coord", "participant"], ["const", min(members)]]
        position = ["coord", "slot"]
        def flatten(row, column):
            return ["mulc", self.scalar_bytes, ["add", ["mulc", width, row], column]]
        candidates = [["const", offset] for offset in range(0, extent - self.scalar_bytes + 1, self.scalar_bytes)]
        candidates.extend((flatten(participant, position), flatten(["const", 0], position)))
        for shift in range(1, len(members)):
            candidates.append(flatten(["modc", ["add", participant, ["const", shift]], len(members)], position))
        for shift in range(1, width):
            candidates.append(flatten(participant, ["modc", ["add", position, ["const", shift]], width]))
        bits = max(1, (max(members) - min(members)).bit_length())
        for bit in range(bits):
            candidates.append(flatten(["toint", ["bxor", ["tobits", bits, participant], ["cbits", bits, 1 << bit]]], position))

        # Join the published demand coordinate expressions to a published source
        # placement.  This common coordinate-projection case is algebraic; no
        # T5/T6 construction, matching result, or preselected controller is used.
        for demand in self.imp.demands:
            if demand.kind != "positions" or demand.participant.op != "coord" or demand.position.op != "coord":
                continue
            receiver, local = demand.participant.args[0], demand.position.args[0]
            if receiver == local:
                continue
            indices = {receiver: ["coord", "participant"], local: position}
            for domain in demand.domain:
                if domain.name not in indices and len(_values(domain)) == 1:
                    indices[domain.name] = _constant(next(iter(_values(domain))), domain.sort)
            if any(domain.name not in indices for domain in demand.domain):
                continue
            identity = {name: _substitute(component, indices) for name, component in demand.identity_coords.items()}
            if name == "@export:smem":
                candidates.extend(_substitute(placement.components["address"], identity)
                                  for placement in self.export.placements if placement.space == "smem")
            else:
                for placement in self.export.placements:
                    if placement.space == "reg":
                        row = ["sub", _substitute(placement.components["participant"], identity), ["const", min(members)]]
                        column = _substitute(placement.components["position"], identity)
                        candidates.append(flatten(row, column))
            for placement_index in range(len(self.export.placements)):
                geometry = self._source_geometry(placement_index)
                if geometry is None or geometry["column_width"] > width:
                    continue
                row = ["sub", _substitute(geometry["dp_expression"], identity), ["const", geometry["dp_min"]]]
                column = ["sub", _substitute(geometry["column_expression"], identity), ["const", geometry["column_min"]]]
                candidates.append(flatten(row, column))

        domains = {"participant": expr.Domain("participant", "int", values=members), "slot": expr.Domain("slot", "int", 0, width)}
        cap = _option(self.profile, "grounding_cap", 1_000_000)
        results, seen = [], set()
        def has_binding(value):
            return isinstance(value, list) and (bool(value) and value[0] == "bind" or any(has_binding(child) for child in value))
        for candidate in candidates:
            encoded = canonical_bytes(candidate)
            if encoded in seen:
                continue
            seen.add(encoded)
            if candidate[0] == "const":
                # The literal family was constructed directly from aligned,
                # complete scalar addresses of this reservation.
                results.append(candidate)
                continue
            worlds = self.worlds if has_binding(candidate) else self.worlds[:1]
            if len(worlds) * len(members) * width > cap:
                raise ModelError(["round address expression analysis exceeds the finite grounding cap"])
            valid = True
            for world in worlds:
                try:
                    parsed = expr.parse(_bind_constants(candidate, dict(world), self._binding_domains()), domains)
                    report = expr.relation_check({"address": parsed}, domains, {"address": (0, extent - self.scalar_bytes + 1)}, cap=cap)
                    if report.capped:
                        raise ModelError(["round address expression analysis exceeds the finite grounding cap"])
                    if not report.total or not report.in_bounds or any(value % self.scalar_bytes for value in expr.ground(parsed, domains).values()):
                        valid = False
                        break
                except expr.ExprError:
                    valid = False
                    break
            if valid:
                results.append(candidate)
        self._round_address_cache[key] = tuple(results)
        return tuple(results)

    def action_domain(self, *, families: frozenset[str] | None = None) -> Mapping[str, tuple[ir.Node, ...]]:
        """Construct requested finite families; filtering never changes their order."""
        def wanted(family):
            return families is None or family in families
        registers = [resource for resource in self.resources if resource.name in self.derived.storage and resource.domain == "reg"]
        writable_registers = [resource for resource in registers if resource.name not in self.derived.readonly_storage]
        shared = [resource for resource in self.resources if resource.name in self.derived.storage and resource.domain == "smem"]
        exports = self.exported_storage()
        shared_sources = [(resource.name, resource.extent) for resource in shared]
        if "@export:smem" in exports and "smem_load_scalar" in self.export.read_capabilities:
            shared_sources.append(("@export:smem", exports["@export:smem"]["extent"]))
        scalar_shared = wanted("smem_store_scalar") or wanted("smem_load_scalar")
        scalar_private = scalar_shared or wanted("reg_copy") or wanted("shfl_xor")
        private_cells = [{"resource": resource.name, "index": index} for resource in registers for index in range(resource.extent)] if scalar_private else []
        private_destinations = [cell for cell in private_cells if cell["resource"] not in self.derived.readonly_storage]
        source_indices = sorted({cell.index for cell in self.cells if cell.resource == "@export:reg"})
        sources = [{"resource": "@export:reg", "index": index} for index in source_indices]
        shared_cells = [{"resource": name, "index": index} for name, extent in shared_sources
                        for index in range(0, extent - self.scalar_bytes + 1, self.scalar_bytes)] if scalar_shared else []
        result = {}
        for site in self.sites:
            permitted = self.derived.permitted[site.name]
            addresses = list(shared_cells)
            members = self.participants[site.name]
            if scalar_shared:
                for name, extent in shared_sources:
                    slots = extent // self.scalar_bytes
                    for shift in range(slots):
                        address = ["mulc", self.scalar_bytes, ["modc", ["add", ["coord", "participant"], ["const", shift]], slots]]
                        parsed = expr.parse(address, {"participant": expr.Domain("participant", "int", values=members)})
                        report = expr.relation_check({"address": parsed}, {"participant": expr.Domain("participant", "int", values=members)},
                                                     {"address": (0, extent)}, cap=len(members))
                        if report.total and report.in_bounds:
                            addresses.append({"resource": name, "index": address})
            actions: list[ir.Node] = []
            if not self.ready_at_entry and "observe_ready" in permitted:
                actions.append(ir.Branch("ready", 0, 0))
            if wanted("reg_copy") and "reg_copy" in permitted:
                actions.extend(ir.Issue("reg_copy", {"src": source, "dst": destination}, 0) for source in sources + private_cells for destination in private_destinations if source != destination)
            if wanted("smem_store_scalar") and "smem_store_scalar" in permitted:
                actions.extend(ir.Issue("smem_store_scalar", {"src": source, "dst": destination}, 0) for source in sources + private_cells for destination in addresses
                               if not destination["resource"].startswith("@export:") and destination["resource"] not in self.derived.readonly_storage)
            if wanted("smem_load_scalar") and "smem_load_scalar" in permitted:
                actions.extend(ir.Issue("smem_load_scalar", {"src": source, "dst": destination}, 0) for source in addresses for destination in private_destinations)
            banks = [(resource.name, resource.extent) for resource in writable_registers]
            source_banks = [(resource.name, resource.extent) for resource in registers]
            if source_indices and "reg_copy" in self.export.read_capabilities:
                source_banks.append(("@export:reg", exports["@export:reg"]["extent"]))
            if wanted("reg_copy_round") and "reg_copy_round" in permitted:
                for source, source_extent in source_banks:
                    for destination, destination_extent in banks:
                        width = min(source_extent, destination_extent)
                        for src_offset in range(source_extent - width + 1):
                            for offset in range(destination_extent - width + 1):
                                if source != destination or src_offset != offset:
                                    actions.append(ir.Issue("reg_copy_round", {
                                        "width": width, "src": source, "dst": destination,
                                        "offset": offset, "src_offset": src_offset}, 0))
            for family in ("smem_store_round", "smem_load_round"):
                if not wanted(family) or family not in permitted:
                    continue
                private_banks = source_banks if family == "smem_store_round" else banks
                shared_banks = [(resource.name, resource.extent) for resource in shared
                                if resource.name not in self.derived.readonly_storage] if family == "smem_store_round" else shared_sources
                for private, width in private_banks:
                    for name, extent in shared_banks:
                        for address in self._round_addresses(site, width, name, extent):
                            source, destination = (private, name) if family == "smem_store_round" else (name, private)
                            actions.append(ir.Issue(family, {"width": width, "src": source, "dst": destination, "address": address, "offset": 0}, 0))
            if wanted("tmem_packet_receipt") and "tmem_packet_receipt" in permitted:
                for packet in self._packet_operands(site):
                    for resource in writable_registers:
                        for offset in range(resource.extent - self.packet_columns + 1):
                            actions.append(ir.Issue("tmem_packet_receipt", {"packet": packet, "dest": resource.name, "offset": offset, "phase": "issue", "handle": "receipt"}, 0))
                actions.append(ir.Issue("tmem_packet_receipt", {"phase": "wait", "handle": "receipt"}, 0))
            if wanted("shfl_xor") and "shfl_xor" in permitted:
                actions.extend(ir.Issue("shfl_xor", {"src": source, "dst": destination, "mask": mask}, 0)
                               for source in private_cells for destination in private_destinations for mask in range(1, self.warp_size))
            if wanted("cta_rendezvous") and "cta_rendezvous" in permitted:
                actions.extend(ir.Issue("cta_rendezvous", {"resource": resource.name}, 0) for resource in self._barriers())
            actions.append(ir.Return(self.demands))
            result[site.name] = tuple(actions)
        return MappingProxyType(result)

    def controller_normalization(self) -> dict[str, Any]:
        ok = all(monitor.kind == "justice" and monitor.labels == frozenset({"owner.ready"})
                 and monitor.step is _stay for monitor in self.monitors)
        return {"ok": ok, "provenance": {
            "rule": "branch-free returning path canonical relabeling",
            "prerequisite": "caller verifies no Branch in the entire finite action domain",
            "justification": "fixed successors imply acyclic returning paths; PC is not observable; collective row equality and generated scheduling justice are renamed consistently",
            "pc_insensitive_base_monitors_verified": ok,
        }}

    def _native_static_region(self, families: frozenset[str] | None) -> dict[str, Any]:
        unsupported = {"ok": False, "kind": "native-synchronous-v1",
                       "prerequisite": "requires one pre-call-ready site with modeled private/shared scalar capabilities",
                       "provenance": {}}
        if not self.ready_at_entry or len(self.sites) != 1:
            return unsupported
        if any(monitor.kind != "justice" or monitor.step is not _stay or monitor.labels != frozenset({"owner.ready"})
               for monitor in self.monitors):
            return {**unsupported, "prerequisite": "custom progress monitors lack a synchronous projection"}
        if any(placement.space not in {"tmem", "reg", "smem"} for placement in self.export.placements):
            return {**unsupported, "prerequisite": "native static projection supports immutable TMEM/register/shared sources"}
        site = self.sites[0]
        actions = tuple(node for node in self.action_domain(families=families)[site.name] if isinstance(node, ir.Issue))
        supported = {"reg_copy", "reg_copy_round", "tmem_packet_receipt", "smem_store_scalar", "smem_load_scalar",
                     "smem_store_round", "smem_load_round", "shfl_xor", "cta_rendezvous", "observe_ready"}
        if any(node.primitive not in supported for node in actions):
            return {**unsupported, "prerequisite": "action family has no exact static transition descriptor"}
        participants = self.participants[site.name]
        if any(node.primitive in {"tmem_packet_receipt", "shfl_xor"} for node in actions):
            for warp in {participant // self.warp_size for participant in participants}:
                if {participant % self.warp_size for participant in participants if participant // self.warp_size == warp} != set(range(self.warp_size)):
                    return {**unsupported, "prerequisite": "warp collective projection requires every physical lane"}
        resources = tuple(resource for resource in self.resources if resource.name in self.derived.storage)
        resource_cells, strides = {}, {}
        mutable = set()
        for resource in resources:
            if resource.domain not in {"reg", "smem"}:
                continue
            stride = self.scalar_bytes if resource.domain == "smem" else 1
            owners = (-1,) if resource.domain == "smem" else participants
            resource_cells[resource.name] = {
                participant: tuple(self._cell_index[self._resource_cell(resource, participant, offset)]
                                   for offset in range(0, resource.extent - stride + 1, stride))
                for participant in owners
            }
            strides[resource.name] = stride
            if resource.name not in self.derived.readonly_storage:
                for cells in resource_cells[resource.name].values():
                    mutable.update(cells)
        source_registers = {}
        for participant in participants:
            offered = {cell.index: index for index, cell in enumerate(self.cells)
                       if cell.resource == "@export:reg" and cell.participant == participant}
            if offered:
                source_registers[participant] = tuple(offered.get(offset, -1) for offset in range(max(offered) + 1))
        if source_registers:
            resource_cells["@export:reg"] = source_registers
            strides["@export:reg"] = 1
        source_shared = {cell.index: index for index, cell in enumerate(self.cells) if cell.resource == "@export:smem"}
        if source_shared:
            resource_cells["@export:smem"] = {
                -1: tuple(source_shared.get(offset, -1) for offset in range(0, max(source_shared) + 1, self.scalar_bytes))}
            strides["@export:smem"] = self.scalar_bytes
        source_cells = {index for world in self.initial_cells for index, value in enumerate(world) if value != EMPTY}
        if mutable & (source_cells | self._readonly_cells):
            return {**unsupported, "prerequisite": "mutable lend aliases immutable exported storage"}
        kinds = tuple(("issue" if node.operands.get("phase", "issue") == "issue" else "wait")
                      if node.primitive == "tmem_packet_receipt" else
                      {"reg_copy": "copy", "cta_rendezvous": "rendezvous"}.get(node.primitive, node.primitive)
                      for node in actions)
        warp_count = len({participant // self.warp_size for participant in participants})
        costs = tuple(len(participants) * node.operands["width"] if node.primitive in _ROUND_SCALARS else
                      warp_count if node.primitive in {"tmem_packet_receipt", "shfl_xor"} else len(participants)
                      for node in actions)
        return {
            "ok": True, "kind": "native-synchronous-v1", "site": site.name,
            "prerequisite": "checked conditional native synchronous encoding",
            "participants": participants, "worlds": tuple(dict(world) for world in self.worlds),
            "binding_domains": self._binding_domains(), "resources": resources,
            "readonly_storage": self.derived.readonly_storage,
            "cells": tuple(cell.name for cell in self.cells), "cell_specs": self.cells,
            "cell_count": len(self.cells), "mutable_cells": tuple(sorted(mutable)),
            "private_cells": tuple(sorted({index for resource in resources if resource.domain == "reg"
                                           for cells in resource_cells[resource.name].values() for index in cells})),
            "shared_cells": tuple(sorted({index for resource in resources if resource.domain == "smem"
                                          for cells in resource_cells[resource.name].values() for index in cells})),
            "source_cells": tuple(sorted(source_cells)),
            "resource_cells": resource_cells, "resource_strides": strides,
            "initial_sparse_worlds": tuple(tuple((index, value) for index, value in enumerate(world) if value != EMPTY)
                                           for world in self.initial_cells),
            "result_cells_worlds": tuple(tuple((participant, position,
                                                self._cell_index[self._declared_cell(self.result_resource, participant, position)], identity)
                                               for participant, position, identity in demands)
                                          for demands in self.world_demands),
            "packets": tuple({"dp": packet.dp, "column": packet.column, "receipt": packet.receipt} for packet in self.packets),
            "packet_subpartition": None if self.tmem_lanes is None else {
                "rule": "own-issuing-warp-modulo",
                "dp_base": "(floor(participant/warp_size) % subpartitions)*warp_size",
                "subpartitions": self.tmem_lanes // self.warp_size, "dp_extent": self.warp_size},
            "tmem_lanes": self.tmem_lanes,
            "actions": actions, "action_kinds": kinds,
            "scalar_bytes": self.scalar_bytes, "warp_size": self.warp_size, "packet_columns": self.packet_columns,
            "semantics": MODEL_SEMANTICS,
            "barriers": {resource.name: {"generations": resource.sync.generations, "capacity": resource.sync.capacity,
                                         "participants": participants}
                         for resource in resources if resource.domain == "barrier" and resource.sync is not None
                         and resource.name not in self.derived.readonly_storage},
            "handle_capacity": {scope: sum(resource.extent for resource in resources
                                           if resource.domain == "metadata" and resource.scope == scope
                                           and resource.name not in self.derived.readonly_storage)
                                for scope in ("thread", "warp", "cta")},
            "metadata_domain": tuple(_option(self.profile, "metadata_domain", (0, 1))),
            "action_costs": costs, "return_cost": len(participants),
            "cost_units": "full-site non-environment transitions; round slots and rendezvous arrivals counted individually; completion/publication ENV steps excluded",
            "protocol": {"issue": "packet DP base equals (warp_index % subpartitions)*warp_size; allocate warp handle and outstanding destination set; no payload delivery",
                         "wait": "matching warp handle required; indivisible immutable-source receipt; release handle",
                         "pending": "no source-use or overwrite of pending destinations before explicit wait",
                         "capacity": "metadata thread/warp/CTA tree capacity counts retained values and live handles",
                         "return": "no live handles; all demanded private identities retained; every declared participant returns",
                         "rendezvous": "all declared participants; one active generation; separate publication; epoch reset"},
            "epoch": {"conflict": "same-cell access by distinct participants involving any write is illegal until rendezvous",
                      "visibility": "own same-epoch writes or rendezvous-published cells only",
                      "round": "slot-ordered scalar steps per participant; never atomic across participants or slots"},
            "provenance": {
                "representation": "compact C1 templates plus resource/source maps; no action-by-cell-by-world transition table",
                "checked_prerequisites": ["pre-call readiness", "finite binding/source tables", "declared reservations and physical participants",
                                          "no mutable/export alias", "known catalogue semantics", "no custom coupled monitors"],
                "encoding_obligations": ["branch-free bounded controllers", "pending destinations inaccessible until matching wait",
                                         "full collective participation and own-warp DP subpartition", "exact per-thread slot ordering", "epoch conflicts and publication",
                                         "scoped metadata/handle capacity", "fixed demanded identities at every binding world"],
                "synchronous_premise": "encoding must establish the obligations for each candidate; full check remains mandatory",
                "source_stability": "immutable source makes early hardware completion indistinguishable when pending destinations are untouched until wait",
            },
        }

    def static_region(self, *, families: frozenset[str] | None = None) -> dict[str, Any]:
        """Legacy private projection or compact conditional native transition model."""
        if self.demand_kind == "memory_view":
            return {"ok": False, "kind": "memory-view", "prerequisite": "memory views use the checked constructive classes"}
        if len(self.actors) != 1 or not self.ready_at_entry or len(self.worlds) != 1:
            return self._native_static_region(families)
        site, member = self.actors[0]
        if any(cell.resource in {"@smem", "@tmem"} or cell.participant != member for cell in self.cells):
            return self._native_static_region(families)
        domain = self.action_domain(families=families)[site]
        if any(isinstance(node, ir.Issue) and node.primitive != "reg_copy" for node in domain):
            return self._native_static_region(families)
        initial = self.initial()
        actions = []
        for node in domain:
            if not isinstance(node, ir.Issue) or node.primitive != "reg_copy":
                continue
            source = self._cell(node.operands["src"], initial, 0)
            destination = self._cell(node.operands["dst"], initial, 0, write=True)
            actions.append({"node": node, "reads": (source,), "writes": (destination,)})
        return {"ok": True, "prerequisite": "already-ready private scalar region", "site": site,
                "cells": tuple(cell.name for cell in self.cells), "initial": tuple(initial.cells), "actions": tuple(actions),
                "results": tuple((self._cell_index[self._declared_cell(self.result_resource, receiver, slot)], q) for receiver, slot, q in self.demands),
                "provenance": {"synchronous": True, "justification": "one participant; exact private scalar footprints; no environment effects or outstanding operations",
                               "region_boundaries": "single declared participant invocation/return are degenerate rendezvous delimiters; no remote effects require publication",
                               "footprints_enumerated": len(actions), "owner_readiness": "published before invocation"}}

    @property
    def model_digest(self) -> str:
        obj = {"semantics": MODEL_SEMANTICS, "export": self.export.to_json_obj(), "import": self.imp.to_json_obj(), "derived": self.derived.to_json_obj(),
               "sites": [site.to_json_obj() for site in self.sites], "resources": [resource.to_json_obj() for resource in self.resources],
               "active_actors": [list(actor) for actor in self.actors],
               "warp_size": self.warp_size, "packet_columns": self.packet_columns, "tmem_lanes": self.tmem_lanes,
               "metadata_domain": list(_option(self.profile, "metadata_domain", (0, 1)))}
        obj["primitives"] = {
            name: {"participants": primitive.participants, "operand_specs": dict(primitive.operand_specs),
                   "timing": dict(primitive.timing), "indivisible_receipt": primitive.indivisible_receipt,
                   "blocking": primitive.blocking, "charges": dict(primitive.charges), "lowering_recipe": primitive.lowering_recipe}
            for name, primitive in self.catalogue.items()
        }
        return sha256_hex(canonical_bytes(obj))
