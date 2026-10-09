"""Versioned tensor contracts, validated once at the stage-manifest boundary."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, ClassVar, Literal, Mapping

from pipec2.decls import declaration_digest, normalize_declaration
from pipec2.model import ClauseFailure, PipeLinkError, _JsonMixin, _canonicalize_model

from . import expr


class TensorContractError(PipeLinkError):
    """Malformed tensor-v2 records, with deterministic T-CONTRACT failures."""


def _encode(value: Any) -> Any:
    if isinstance(value, expr.Expr):
        return expr.to_json_obj(value)
    if isinstance(value, expr.Domain):
        return value.to_json_obj()
    if isinstance(value, _JsonMixin):
        return value.to_json_obj()
    if isinstance(value, dict):
        return {key: _encode(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


class _TensorJsonMixin(_JsonMixin):
    _reader: ClassVar[str]

    def to_json_obj(self) -> dict[str, Any]:
        obj = {
            item.name: _encode(getattr(self, item.name))
            for item in fields(self)
            if not (item.metadata.get("omit_none") and getattr(self, item.name) is None)
        }
        return _canonicalize_model(self, obj)

    @classmethod
    def from_json_obj(cls, obj: dict[str, Any], *, sort_env: expr.SortEnv | None = None):
        """Decode a record; standalone views need their external sort environment."""
        reader = _Reader(sort_env)
        result = reader.read(getattr(reader, cls._reader), obj, "$")
        reader.finish()
        return result


@dataclass(frozen=True)
class Identity(_TensorJsonMixin):
    _reader = "identity"
    result: str
    work: str
    version: str
    coords: tuple[expr.Domain, ...]
    dtype: str
    interpretation: str


@dataclass(frozen=True)
class PlacementView(_TensorJsonMixin):
    _reader = "placement"
    guard: expr.Expr | None
    space: Literal["tmem", "smem", "reg"]
    units: str
    components: dict[str, expr.Expr]


@dataclass(frozen=True)
class Readiness(_JsonMixin):
    after: str
    reuse_after: str


@dataclass(frozen=True)
class TensorExport(_TensorJsonMixin):
    _reader = "export"
    name: str
    identity: Identity
    placements: tuple[PlacementView, ...]
    read_capabilities: tuple[str, ...]
    readiness: Readiness
    fixed_bindings: dict[str, int]
    backing: tuple[Lend, ...]
    calls: dict[str, str | None]

    @property
    def residual_bindings(self) -> tuple[str, ...]:
        """Placement bindings not fixed by the owner; never serialized."""
        names: set[str] = set()

        def visit(node: expr.Expr) -> None:
            if node.op == "bind":
                names.add(node.args[0])
            for child in node.args:
                if isinstance(child, expr.Expr):
                    visit(child)

        for placement in self.placements:
            if placement.guard is not None:
                visit(placement.guard)
            for component in placement.components.values():
                visit(component)
        return tuple(sorted(names - self.fixed_bindings.keys()))


@dataclass(frozen=True)
class DemandView(_TensorJsonMixin):
    _reader = "demand"
    kind: Literal["positions", "memory_view"]
    domain: tuple[expr.Domain, ...]
    guard: expr.Expr | None
    identity_coords: dict[str, expr.Expr]
    participant: expr.Expr | None = field(default=None, metadata={"omit_none": True})
    position: expr.Expr | None = field(default=None, metadata={"omit_none": True})
    space: Literal["smem", "gmem"] | None = field(default=None, metadata={"omit_none": True})
    units: str | None = field(default=None, metadata={"omit_none": True})
    components: dict[str, expr.Expr] | None = field(default=None, metadata={"omit_none": True})


@dataclass(frozen=True)
class CallableParam(_JsonMixin):
    name: str
    meaning: Literal["source_base", "dest_fragment", "dest_view", "scratch", "workspace", "binding"]
    ctype: str


@dataclass(frozen=True)
class TensorCallable(_JsonMixin):
    symbol: str
    declaration: str
    params: tuple[CallableParam, ...]
    events: dict[str, str]

    @property
    def declaration_digest(self) -> str:
        return declaration_digest(self.declaration)


@dataclass(frozen=True)
class TensorImport(_TensorJsonMixin):
    _reader = "tensor_import"
    name: str
    identity: Identity
    demands: tuple[DemandView, ...]
    callable: TensorCallable
    sites: tuple[str, ...]
    completion: str
    lends: tuple[Lend, ...]
    rendezvous: tuple[Rendezvous, ...]


@dataclass(frozen=True)
class Participants(_TensorJsonMixin):
    _reader = "participants"
    set: tuple[int, ...] | None = field(default=None, metadata={"omit_none": True})
    expr: expr.Expr | None = field(default=None, metadata={"omit_none": True})
    domain: tuple[expr.Domain, ...] | None = field(default=None, metadata={"omit_none": True})


@dataclass(frozen=True)
class Rendezvous(_JsonMixin):
    event: str
    barrier: str
    cohort: Participants


@dataclass(frozen=True)
class Site(_TensorJsonMixin):
    _reader = "site"
    name: str
    role: str
    participants: Participants
    invocation: str
    guarantees: tuple[str, ...]
    observes: tuple[str, ...]


@dataclass(frozen=True)
class Binding(_TensorJsonMixin):
    _reader = "binding"
    name: str
    sort: str
    domain: expr.Domain
    owner: str
    units: str
    observation_scope: str
    lifetime: str


@dataclass(frozen=True)
class ResourceLocation(_JsonMixin):
    handle: str | None
    base: int | None


@dataclass(frozen=True)
class ResourceSync(_JsonMixin):
    generations: int
    capacity: int


@dataclass(frozen=True)
class Lend(_TensorJsonMixin):
    _reader = "lend"
    name: str
    domain: Literal["smem", "tmem", "barrier", "reg", "metadata"]
    extent: int
    align: int
    scope: Literal["cta", "thread", "warp"]
    binding: Literal["fixed", "relocatable", "caller_frame"]
    location: ResourceLocation
    owner: str
    lifetime: dict[str, str]
    sync: ResourceSync | None


@dataclass(frozen=True)
class StageTensor(_TensorJsonMixin):
    version: Literal["tensor-v2"]
    exports: tuple[TensorExport, ...]
    imports: tuple[TensorImport, ...]
    sites: tuple[Site, ...]
    bindings: tuple[Binding, ...]

    @classmethod
    def from_json_obj(cls, obj: dict[str, Any]) -> StageTensor:
        return load_stage_tensor(obj, "$")


_INVALID = object()
_MISSING = object()
Parser = Callable[[Any, str], Any]


class _Invalid(Exception):
    pass


class _Reader:
    """Accumulate independent boundary defects without manufacturing valid records."""

    def __init__(self, sort_env: expr.SortEnv | None = None) -> None:
        self.failures: list[ClauseFailure] = []
        self.bindings: dict[str, expr.Domain] = dict(sort_env or {})
        self.external_coords: Mapping[str, expr.Domain] = sort_env or {}
        self.paths: dict[int, str] = {}
        self.indices: dict[str, tuple[int, ...]] = {}

    def fail(self, path: str, message: str, **detail: Any) -> None:
        self.failures.append(ClauseFailure("T-CONTRACT", path, message, detail))

    def reject(self, path: str, message: str, **detail: Any) -> Any:
        self.fail(path, message, **detail)
        raise _Invalid

    def finish(self) -> None:
        if self.failures:
            raise TensorContractError(self.failures)

    def read(self, parser: Parser, value: Any, path: str) -> Any:
        if value is _MISSING:
            return _INVALID
        try:
            return parser(value, path)
        except _Invalid:
            return _INVALID

    def object(self, value: Any, path: str, keys: tuple[str, ...] | None = None) -> dict[str, Any]:
        if not isinstance(value, dict) or any(type(key) is not str or not key for key in value):
            self.reject(path, "expected an object with non-empty string keys")
        if keys is not None:
            for key in sorted(set(keys) - value.keys()):
                self.fail(f"{path}.{key}", "missing required field")
            for key in sorted(value.keys() - set(keys)):
                self.fail(f"{path}.{key}", "unknown field")
        return value

    def at(self, parser: Parser, obj: dict[str, Any], key: str, path: str) -> Any:
        return self.read(parser, obj.get(key, _MISSING), f"{path}.{key}")

    def build(self, cls: type, path: str, **values: Any) -> Any:
        if any(value is _INVALID for value in values.values()):
            raise _Invalid
        result = cls(**values)
        self.paths[id(result)] = path
        return result

    def record(self, cls: type | None, value: Any, path: str, parsers: dict[str, Parser]) -> Any:
        obj = self.object(value, path, tuple(parsers))
        values = {key: self.at(parser, obj, key, path) for key, parser in parsers.items()}
        if cls is not None:
            return self.build(cls, path, **values)
        if any(item is _INVALID for item in values.values()):
            raise _Invalid
        return values

    def string(self, value: Any, path: str) -> str:
        if type(value) is not str or not value:
            self.reject(path, "expected a non-empty string")
        return value

    def integer(self, value: Any, path: str) -> int:
        if type(value) is not int:
            self.reject(path, "expected an integer")
        return value

    def nonnegative(self, value: Any, path: str) -> int:
        value = self.integer(value, path)
        if value < 0:
            self.reject(path, "expected a nonnegative integer")
        return value

    def positive(self, value: Any, path: str) -> int:
        value = self.integer(value, path)
        if value <= 0:
            self.reject(path, "expected a positive integer")
        return value

    def choice(self, *choices: str) -> Parser:
        def parse(value: Any, path: str) -> str:
            if type(value) is not str or value not in choices:
                self.reject(path, "unexpected value", expected=list(choices), actual=value)
            return value
        return parse

    def nullable(self, parser: Parser) -> Parser:
        return lambda value, path: None if value is None else parser(value, path)

    def array(self, parser: Parser, value: Any, path: str, *, unique: bool = False) -> tuple[Any, ...]:
        if not isinstance(value, list):
            self.reject(path, "expected a JSON array")
        result = []
        indices = []
        seen = set()
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            parsed = self.read(parser, item, item_path)
            if parsed is _INVALID:
                continue
            if unique:
                name = parsed.name if is_dataclass(parsed) else parsed
                if name in seen:
                    self.fail(f"{item_path}.name" if is_dataclass(parsed) else item_path,
                              "duplicate name or value", actual=name)
                seen.add(name)
            result.append(parsed)
            indices.append(index)
        # Preserve source indices after malformed siblings for reference paths.
        self.indices[path] = tuple(indices)
        return tuple(result)

    def strings(self, value: Any, path: str) -> tuple[str, ...]:
        return self.array(self.string, value, path, unique=True)

    def mapping(self, parser: Parser, value: Any, path: str) -> dict[str, Any]:
        obj = self.object(value, path)
        result = {}
        for key in sorted(obj):
            parsed = self.read(parser, obj[key], f"{path}.{key}")
            if parsed is not _INVALID:
                result[key] = parsed
        return result

    def domain(self, value: Any, path: str) -> expr.Domain:
        obj = self.object(value, path)
        keys = ("name", "sort", "values") if "values" in obj else ("name", "sort", "lo", "hi")
        self.object(obj, path, keys)
        try:
            result = expr.Domain.from_json_obj(obj)
        except (expr.ExprError, TypeError, ValueError) as error:
            issues = getattr(error, "issues", [str(error)])
            self.reject(path, "invalid finite domain", issues=list(issues))
        if result.values == () or (result.values is None and result.lo == result.hi):
            self.reject(path, "finite domain must be nonempty")
        return result

    def domains(self, value: Any, path: str) -> tuple[expr.Domain, ...]:
        return self.array(self.domain, value, path, unique=True)

    def environment(self, domains: tuple[expr.Domain, ...], path: str) -> dict[str, expr.Domain]:
        coords = {domain.name: domain for domain in domains}
        for index, domain in zip(self.indices[path], domains):
            if domain.name in self.bindings and domain.name not in self.external_coords:
                self.fail(f"{path}[{index}].name", "coordinate name collides with a binding", name=domain.name)
        return coords

    def expression(self, value: Any, path: str, coords: Mapping[str, expr.Domain], *, predicate: bool = False) -> expr.Expr:
        try:
            result = expr.parse(value, {**self.bindings, **coords})
        except expr.ExprError as error:
            self.reject(path, "invalid expression", issues=list(error.issues))
        if (result.sort == "pred") != predicate:
            self.fail(path, "expected a predicate" if predicate else "expected a value expression")
        # The C1 environment supplies sorts. C2 additionally distinguishes the
        # namespaces: a published binding is not a free coordinate, or vice versa.
        def references(node: expr.Expr) -> None:
            if node.op in ("coord", "bind"):
                namespace = coords if node.op == "coord" else self.bindings
                if node.args[0] not in namespace:
                    self.fail(path, "undeclared expression name", kind=node.op, name=node.args[0])
            for child in node.args:
                if isinstance(child, expr.Expr):
                    references(child)
        references(result)
        return result

    def expressions(self, value: Any, path: str, coords: Mapping[str, expr.Domain]) -> dict[str, expr.Expr]:
        return self.mapping(lambda item, loc: self.expression(item, loc, coords), value, path)

    def guard(self, value: Any, path: str, coords: Mapping[str, expr.Domain]) -> expr.Expr | None:
        return None if value is None else self.expression(value, path, coords, predicate=True)

    def identity(self, value: Any, path: str) -> Identity:
        return self.record(Identity, value, path, {
            "result": self.string, "work": self.string, "version": self.string,
            "coords": self.domains, "dtype": self.string, "interpretation": self.string,
        })

    def placement(self, value: Any, path: str, coords: Mapping[str, expr.Domain] | None = None) -> PlacementView:
        coords = self.external_coords if coords is None else coords
        return self.record(PlacementView, value, path, {
            "guard": lambda item, loc: self.guard(item, loc, coords),
            "space": self.choice("tmem", "smem", "reg"), "units": self.string,
            "components": lambda item, loc: self.expressions(item, loc, coords),
        })

    def readiness(self, value: Any, path: str) -> Readiness:
        return self.record(Readiness, value, path, {"after": self.string, "reuse_after": self.string})

    def export(self, value: Any, path: str) -> TensorExport:
        obj = self.object(value, path, ("name", "identity", "placements", "read_capabilities", "readiness",
                                        "fixed_bindings", "backing", "calls"))
        identity = self.at(self.identity, obj, "identity", path)
        coords = self.environment(identity.coords, f"{path}.identity.coords") if identity is not _INVALID else {}
        return self.build(TensorExport, path,
            name=self.at(self.string, obj, "name", path), identity=identity,
            placements=self.at(lambda item, loc: self.array(lambda view, pos: self.placement(view, pos, coords), item, loc),
                               obj, "placements", path),
            read_capabilities=self.at(self.strings, obj, "read_capabilities", path),
            readiness=self.at(self.readiness, obj, "readiness", path),
            fixed_bindings=self.at(lambda item, loc: self.mapping(self.integer, item, loc), obj, "fixed_bindings", path),
            backing=self.at(lambda item, loc: self.array(self.lend, item, loc), obj, "backing", path),
            calls=self.at(lambda item, loc: self.record(None, item, loc, {
                "produce": self.string, "release": self.nullable(self.string)}), obj, "calls", path),
        )

    def demand(self, value: Any, path: str, identity: Identity | None = None) -> DemandView:
        obj = self.object(value, path)
        kind = self.at(self.choice("positions", "memory_view"), obj, "kind", path)
        common = ("kind", "domain", "guard", "identity_coords")
        if kind == "positions":
            self.object(obj, path, common + ("participant", "position"))
        elif kind == "memory_view":
            self.object(obj, path, common + ("space", "units", "components"))
        else:
            if "kind" not in obj:
                self.fail(f"{path}.kind", "missing required field")
            raise _Invalid
        domains = self.at(self.domains, obj, "domain", path)
        coords = self.environment(domains, f"{path}.domain") if domains is not _INVALID else {}
        mapped = self.at(lambda item, loc: self.expressions(item, loc, coords), obj, "identity_coords", path)
        if identity is not None and mapped is not _INVALID:
            required = {domain.name: domain for domain in identity.coords}
            for name in sorted(required.keys() - mapped.keys()):
                self.fail(f"{path}.identity_coords.{name}", "missing identity coordinate")
            for name in sorted(mapped.keys() - required.keys()):
                self.fail(f"{path}.identity_coords.{name}", "unknown identity coordinate")
            for name in sorted(required.keys() & mapped.keys()):
                if mapped[name].sort != required[name].sort:
                    self.fail(f"{path}.identity_coords.{name}", "identity coordinate sort mismatch",
                              expected=required[name].sort, actual=mapped[name].sort)
        values = {
            "kind": kind, "domain": domains, "identity_coords": mapped,
            "guard": self.at(lambda item, loc: self.guard(item, loc, coords), obj, "guard", path),
        }
        if kind == "positions":
            for key in ("participant", "position"):
                values[key] = self.at(lambda item, loc: self.expression(item, loc, coords), obj, key, path)
        else:
            values["space"] = self.at(self.choice("smem", "gmem"), obj, "space", path)
            values["units"] = self.at(self.string, obj, "units", path)
            values["components"] = self.at(lambda item, loc: self.expressions(item, loc, coords), obj, "components", path)
        return self.build(DemandView, path, **values)

    def callable_param(self, value: Any, path: str) -> CallableParam:
        return self.record(CallableParam, value, path, {
            "name": self.string,
            "meaning": self.choice("source_base", "dest_fragment", "dest_view", "scratch", "workspace", "binding"),
            "ctype": self.string,
        })

    def declaration(self, value: Any, path: str) -> str:
        normalized = normalize_declaration(self.string(value, path))
        if not normalized:
            self.reject(path, "declaration contains no tokens")
        return normalized

    def callable(self, value: Any, path: str) -> TensorCallable:
        return self.record(TensorCallable, value, path, {
            "symbol": self.string, "declaration": self.declaration,
            "params": lambda item, loc: self.array(self.callable_param, item, loc, unique=True),
            "events": lambda item, loc: self.record(None, item, loc, {"ready": self.string, "done": self.string}),
        })

    def tensor_import(self, value: Any, path: str) -> TensorImport:
        obj = self.object(value, path, ("name", "identity", "demands", "callable", "sites", "completion", "lends", "rendezvous"))
        identity = self.at(self.identity, obj, "identity", path)
        return self.build(TensorImport, path,
            name=self.at(self.string, obj, "name", path), identity=identity,
            demands=self.at(lambda item, loc: self.array(
                lambda view, pos: self.demand(view, pos, identity if identity is not _INVALID else None), item, loc),
                obj, "demands", path),
            callable=self.at(self.callable, obj, "callable", path),
            sites=self.at(self.strings, obj, "sites", path),
            completion=self.at(self.string, obj, "completion", path),
            lends=self.at(lambda item, loc: self.array(self.lend, item, loc), obj, "lends", path),
            rendezvous=self.at(lambda item, loc: self.array(self.rendezvous, item, loc), obj, "rendezvous", path),
        )

    def rendezvous(self, value: Any, path: str) -> Rendezvous:
        return self.record(Rendezvous, value, path, {
            "event": self.string, "barrier": self.string, "cohort": self.participants,
        })

    def participants(self, value: Any, path: str) -> Participants:
        obj = self.object(value, path)
        if "set" in obj:
            self.object(obj, path, ("set",))
            values = self.at(lambda item, loc: self.array(self.nonnegative, item, loc, unique=True), obj, "set", path)
            return self.build(Participants, path, set=tuple(sorted(values)) if values is not _INVALID else _INVALID)
        self.object(obj, path, ("expr", "domain"))
        domains = self.at(self.domains, obj, "domain", path)
        coords = self.environment(domains, f"{path}.domain") if domains is not _INVALID else {}
        return self.build(Participants, path, domain=domains,
                          expr=self.at(lambda item, loc: self.expression(item, loc, coords), obj, "expr", path))

    def site(self, value: Any, path: str) -> Site:
        return self.record(Site, value, path, {
            "name": self.string, "role": self.string, "participants": self.participants,
            "invocation": self.string,
            "guarantees": lambda item, loc: self.array(
                self.choice("enter_once", "single_completion", "converged", "full_warps", "no_early_return"), item, loc, unique=True),
            "observes": self.strings,
        })

    def binding(self, value: Any, path: str) -> Binding:
        result = self.record(Binding, value, path, {
            "name": self.string, "sort": self.string, "domain": self.domain,
            "owner": self.string, "units": self.string, "observation_scope": self.string,
            "lifetime": self.string,
        })
        if result.name != result.domain.name:
            self.fail(f"{path}.domain.name", "binding and domain names differ", expected=result.name)
        if result.sort != result.domain.sort:
            self.fail(f"{path}.domain.sort", "binding and domain sorts differ", expected=result.sort)
        return result

    def location(self, value: Any, path: str) -> ResourceLocation:
        return self.record(ResourceLocation, value, path, {
            "handle": self.nullable(self.string), "base": self.nullable(self.nonnegative),
        })

    def sync(self, value: Any, path: str) -> ResourceSync:
        return self.record(ResourceSync, value, path, {"generations": self.nonnegative, "capacity": self.nonnegative})

    def lifetime(self, value: Any, path: str) -> dict[str, str]:
        return self.record(None, value, path, {"from": self.string, "to": self.string})

    def lend(self, value: Any, path: str) -> Lend:
        result = self.record(Lend, value, path, {
            "name": self.string, "domain": self.choice("smem", "tmem", "barrier", "reg", "metadata"),
            "extent": self.nonnegative, "align": self.positive, "scope": self.choice("cta", "thread", "warp"),
            "binding": self.choice("fixed", "relocatable", "caller_frame"), "location": self.location,
            "owner": self.string, "lifetime": self.lifetime,
            "sync": self.nullable(self.sync),
        })
        if result.binding == "caller_frame" and (result.domain != "reg" or result.scope != "thread" or result.align != 1):
            self.fail(f"{path}.binding", "caller frame must be a thread-local register lend with alignment one")
        if result.binding == "relocatable" and result.location.handle is None:
            self.fail(f"{path}.location.handle", "relocatable lend requires a handle")
        elif result.binding == "fixed" and result.location.base is None and result.location.handle is None:
            self.fail(f"{path}.location", "fixed lend requires a base or handle")
        return result

    def references(self, values: tuple[str, ...], declared: Mapping[str, Any], path: str) -> None:
        for index, name in zip(self.indices[path], values):
            if name not in declared:
                self.fail(f"{path}[{index}]", "unknown reference", name=name)

    def resolve(self, exports: tuple[TensorExport, ...], imports: tuple[TensorImport, ...],
                sites: tuple[Site, ...]) -> None:
        site_map = {site.name: site for site in sites}
        names: set[str] = set()
        for item in exports:
            path = self.paths[id(item)]
            for name, value in item.fixed_bindings.items():
                domain = self.bindings.get(name)
                if domain is not None and not _contains(domain, value):
                    self.fail(f"{path}.fixed_bindings.{name}", "fixed value is outside the binding domain", actual=value)
            for lend in item.backing:
                if lend.name in names:
                    self.fail(f"{self.paths[id(lend)]}.name", "duplicate lend name", actual=lend.name)
                names.add(lend.name)
        for item in imports:
            self.references(item.sites, site_map, f"{self.paths[id(item)]}.sites")
            lends = {lend.name: lend for lend in item.lends}
            events = set()
            for index, rendezvous in enumerate(item.rendezvous):
                path = f"{self.paths[id(item)]}.rendezvous[{index}]"
                if rendezvous.event in events:
                    self.fail(f"{path}.event", "duplicate rendezvous event")
                events.add(rendezvous.event)
                if rendezvous.barrier not in lends or lends[rendezvous.barrier].domain != "barrier":
                    self.fail(f"{path}.barrier", "rendezvous requires an import barrier lend")
                if rendezvous.cohort.set == ():
                    self.fail(f"{path}.cohort", "rendezvous cohort must be nonempty")
            for lend in item.lends:
                if lend.name in names:
                    self.fail(f"{self.paths[id(lend)]}.name", "duplicate lend name", actual=lend.name)
                names.add(lend.name)


def _contains(domain: expr.Domain, value: int) -> bool:
    if domain.values is not None:
        return value in domain.values
    return domain.lo <= value < domain.hi


def load_stage_tensor(obj: Any, path: str) -> StageTensor:
    """Load strict tensor-v2 JSON; leave analyzer/model/backend scope to admission."""
    reader = _Reader()
    root = reader.read(lambda value, loc: reader.object(value, loc, (
        "version", "exports", "imports", "sites", "bindings",
    )), obj, path)
    if root is _INVALID:
        reader.finish()
    version = reader.at(reader.choice("tensor-v2"), root, "version", path)

    def collection(key: str, parser: Parser) -> tuple[Any, ...]:
        values = reader.at(lambda value, loc: reader.array(parser, value, loc, unique=True), root, key, path)
        return () if values is _INVALID else values

    bindings = collection("bindings", reader.binding)
    reader.bindings = {binding.name: binding.domain for binding in bindings}
    sites = collection("sites", reader.site)
    exports = collection("exports", reader.export)
    imports = collection("imports", reader.tensor_import)
    reader.resolve(exports, imports, sites)
    reader.finish()
    return StageTensor(version, exports, imports, sites, bindings)


__all__ = [
    "Identity", "PlacementView", "TensorExport", "DemandView", "TensorImport", "Site",
    "Binding", "Lend", "StageTensor", "TensorContractError",
    "Readiness", "CallableParam", "TensorCallable", "Participants", "ResourceLocation",
    "ResourceSync", "Rendezvous", "load_stage_tensor",
]
