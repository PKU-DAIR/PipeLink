"""Sort-checked expressions and exact analysis over declared finite domains.

Parsing establishes syntax and sorts; domain checking establishes the remaining
``tobits`` range obligations. Predicates are represented by Expr with the internal
sort ``pred`` and evaluate to integer truth values, not a third value sort.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from pipec2.canonjson import canonical_bytes, sha256_hex
from pipec2.model import ClauseFailure, PipeLinkError, _JsonMixin


class ExprError(PipeLinkError):
    """Deterministic syntax, sort, or finite-domain failures."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = sorted(issues)
        failures = []
        for issue in self.issues:
            path, separator, message = issue.partition(": ")
            if not separator or not path.startswith("$"):
                path, message = "$", issue
            failures.append(ClauseFailure("T-CONTRACT", path, message, {}))
        super().__init__(failures)


_BITS_SORT = re.compile(r"bits\[([1-9][0-9]*)\]\Z")


@dataclass(frozen=True)
class Domain(_JsonMixin):
    """A half-open integer interval or an explicitly ordered finite set.

    Empty domains are permitted here; a binding's nonempty-domain obligation is
    enforced by the contract loader. Explicit values must be distinct, and their
    declared order is preserved in JSON and enumeration.
    """

    name: str
    sort: str
    lo: int | None = field(default=None, metadata={"omit_none": True})
    hi: int | None = field(default=None, metadata={"omit_none": True})
    values: tuple[int, ...] | None = field(default=None, metadata={"omit_none": True})

    def __post_init__(self) -> None:
        issues = []
        if not isinstance(self.name, str) or not self.name:
            issues.append("$.name: expected a nonempty string")
        width = None
        if self.sort != "int":
            match = _BITS_SORT.fullmatch(self.sort) if isinstance(self.sort, str) else None
            if match is None:
                issues.append("$.sort: expected int or bits[N] with positive width")
            else:
                width = int(match[1])
        if self.values is None:
            if type(self.lo) is not int or type(self.hi) is not int:
                issues.append("$: interval requires integer lo and hi")
            elif self.lo > self.hi:
                issues.append("$: interval requires lo <= hi")
            elif width is not None and (
                self.lo < 0
                or self.hi > 0 and (self.hi - 1).bit_length() > width
            ):
                issues.append("$: interval is outside the declared bits width")
        else:
            if self.lo is not None or self.hi is not None:
                issues.append("$: values and interval bounds are mutually exclusive")
            if not isinstance(self.values, tuple) or any(type(v) is not int for v in self.values):
                issues.append("$.values: expected a tuple of integers")
            else:
                if len(set(self.values)) != len(self.values):
                    issues.append("$.values: domain values must be distinct")
                if width is not None and any(v < 0 or v.bit_length() > width for v in self.values):
                    issues.append("$.values: a value is outside the declared bits width")
        if issues:
            raise ExprError(issues)

    @classmethod
    def from_json_obj(cls, obj: Any) -> Domain:
        if not isinstance(obj, dict):
            raise ExprError(["$: domain must be a JSON object"])
        keys = set(obj)
        if keys == {"name", "sort", "values"}:
            if not isinstance(obj["values"], list):
                raise ExprError(["$.values: expected a JSON array"])
            return cls(obj["name"], obj["sort"], values=tuple(obj["values"]))
        if keys == {"name", "sort", "lo", "hi"}:
            return cls(obj["name"], obj["sort"], obj["lo"], obj["hi"])
        raise ExprError(["$: domain requires exactly name, sort, lo, hi or name, sort, values"])


SortEnv: TypeAlias = Mapping[str, Domain]


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Expr | int | str, ...]
    sort: str

    def to_json_obj(self) -> list[Any]:
        return to_json_obj(self)


@dataclass(frozen=True)
class DomainReport(_JsonMixin):
    ok: bool
    issues: tuple[str, ...]
    enumerated: int
    capped: bool


@dataclass(frozen=True)
class RelationReport(_JsonMixin):
    total: bool
    in_bounds: bool
    injective: bool
    covering: bool
    witnesses: tuple[str, ...]
    capped: bool


# Scalar operands are checked here along with the recursive expression operands.
_GRAMMAR = {
    "const": ("integer",), "cbits": ("width", "unsigned"),
    "coord": ("name",), "bind": ("name",),
    "add": ("int", "int"), "sub": ("int", "int"),
    "mulc": ("integer", "int"), "divc": ("int", "positive"),
    "modc": ("int", "positive"),
    "band": ("bits", "bits"), "bor": ("bits", "bits"),
    "bxor": ("bits", "bits"),
    "shlc": ("bits", "shift"), "shrc": ("bits", "shift"),
    "tobits": ("width", "int"), "toint": ("bits",),
    "select": ("pred", "value", "value"),
    "eq": ("int", "int"), "lt": ("int", "int"),
    "le": ("int", "int"), "not": ("pred",),
    "pand": ("pred", "pred"), "por": ("pred", "pred"),
}
_EXPR_OPERANDS = frozenset({"int", "bits", "pred", "value"})


def parse(obj: Any, sort_env: SortEnv) -> Expr:
    """Parse canonical list-form JSON, collecting independent errors in one pass."""
    issues: list[str] = []

    def visit(node: Any, path: str) -> Expr | None:
        start = len(issues)
        if not isinstance(node, list) or not node or not isinstance(node[0], str):
            issues.append(f"{path}: expected a nonempty expression list with an operator")
            return None
        op = node[0]
        specs = _GRAMMAR.get(op)
        if specs is None:
            issues.append(f"{path}[0]: unknown operator {op!r}")
            for index, child in enumerate(node[1:], 1):
                if isinstance(child, list):
                    visit(child, f"{path}[{index}]")
            return None
        if len(node) != len(specs) + 1:
            issues.append(f"{path}: {op} requires {len(specs)} operands, got {len(node) - 1}")
        args: list[Expr | int | str | None] = []
        for index, spec in enumerate(specs, 1):
            if index >= len(node):
                args.append(None)
                continue
            value = node[index]
            child_path = f"{path}[{index}]"
            if spec in _EXPR_OPERANDS:
                arg = visit(value, child_path)
                args.append(arg)
                if arg is not None:
                    matches = (
                        arg.sort.startswith("bits[") if spec == "bits"
                        else arg.sort != "pred" if spec == "value"
                        else arg.sort == spec
                    )
                    if not matches:
                        issues.append(f"{child_path}: {op} expects {spec}, got {arg.sort}")
            elif spec == "name":
                if not isinstance(value, str):
                    issues.append(f"{child_path}: expected a domain name")
                    args.append(None)
                elif value not in sort_env:
                    issues.append(f"{child_path}: unknown name {value!r}")
                    args.append(None)
                else:
                    args.append(value)
            elif type(value) is not int:
                issues.append(f"{child_path}: {op} requires an integer {spec} operand")
                args.append(None)
            elif spec in {"width", "positive"} and value <= 0:
                issues.append(f"{child_path}: {op} requires a positive {spec} operand")
                args.append(None)
            elif spec in {"unsigned", "shift"} and value < 0:
                issues.append(f"{child_path}: {op} requires a nonnegative {spec} operand")
                args.append(None)
            else:
                args.append(value)
        for index in range(len(specs) + 1, len(node)):
            if isinstance(node[index], list):
                visit(node[index], f"{path}[{index}]")
        sort = "int"
        if op in {"coord", "bind"} and args[0] is not None:
            sort = sort_env[args[0]].sort
        elif op in {"cbits", "tobits"} and args[0] is not None:
            sort = f"bits[{args[0]}]"
            if op == "cbits" and args[1] is not None and args[1].bit_length() > args[0]:
                issues.append(f"{path}[2]: cbits value is outside {sort}")
        elif op in {"band", "bor", "bxor", "select"}:
            left, right = args[-2:]
            if isinstance(left, Expr) and isinstance(right, Expr):
                sort = left.sort
                if left.sort != right.sort:
                    issues.append(f"{path}: {op} requires equal sorts, got {left.sort} and {right.sort}")
        elif op in {"shlc", "shrc"} and isinstance(args[0], Expr):
            sort = args[0].sort
        elif op in {"eq", "lt", "le", "not", "pand", "por"}:
            sort = "pred"
        if len(issues) != start:
            return None
        return Expr(op, tuple(args), sort)

    result = visit(obj, "$")
    if issues:
        raise ExprError(issues)
    assert result is not None
    return result


def to_json_obj(e: Expr) -> list[Any]:
    """Preserve the exact parsed tree; do not fold constants or erase replication."""
    return [e.op, *(to_json_obj(arg) if isinstance(arg, Expr) else arg for arg in e.args)]


def digest(e: Expr) -> str:
    return sha256_hex(canonical_bytes(to_json_obj(e)))


def evaluate(e: Expr, env: Mapping[str, int]) -> int:
    """Evaluate exactly; checked inputs require no domain revalidation here.

    ``tobits`` is a checked conversion, never truncation. Every operand, including
    both select arms, must be defined on the declared domain.
    """
    value = _evaluate(e, env, "$")
    assert value is not None
    return value


def _evaluate(
    e: Expr, env: Mapping[str, int], path: str, issues: list[str] | None = None,
) -> int | None:
    args = tuple(
        _evaluate(arg, env, f"{path}[{index}]", issues) if isinstance(arg, Expr) else arg
        for index, arg in enumerate(e.args, 1)
    )
    if None in args:
        return None
    match e.op:
        case "const":
            return args[0]
        case "cbits":
            return args[1]
        case "coord" | "bind":
            return env[args[0]]
        case "add":
            return args[0] + args[1]
        case "sub":
            return args[0] - args[1]
        case "mulc":
            return args[0] * args[1]
        case "divc":
            return args[0] // args[1]
        case "modc":
            return args[0] % args[1]
        case "band":
            return args[0] & args[1]
        case "bor":
            return args[0] | args[1]
        case "bxor":
            return args[0] ^ args[1]
        case "shlc" | "shrc":
            width = int(e.sort[5:-1])
            if args[1] >= width:
                return 0
            if e.op == "shrc":
                return args[0] >> args[1]
            return (args[0] << args[1]) & ((1 << width) - 1)
        case "tobits":
            width, value = args
            if value < 0 or value.bit_length() > width:
                issue = f"{path}: tobits operand {value} is outside [0, 2^{width})"
                if issues is None:
                    raise ExprError([issue])
                issues.append(issue)
                return None
            return value
        case "toint":
            return args[0]
        case "select":
            return args[1] if args[0] else args[2]
        case "eq":
            return int(args[0] == args[1])
        case "lt":
            return int(args[0] < args[1])
        case "le":
            return int(args[0] <= args[1])
        case "not":
            return int(not args[0])
        case "pand":
            return int(bool(args[0]) and bool(args[1]))
        case "por":
            return int(bool(args[0]) or bool(args[1]))
    raise AssertionError(f"unparsed expression operator {e.op!r}")


def _references(e: Expr) -> set[str]:
    if e.op in {"coord", "bind"}:
        return {e.args[0]}
    result: set[str] = set()
    for arg in e.args:
        if isinstance(arg, Expr):
            result.update(_references(arg))
    return result


def _domain_values(domain: Domain) -> range | tuple[int, ...]:
    return domain.values if domain.values is not None else range(domain.lo, domain.hi)


def _size(values: range | tuple[int, ...]) -> int:
    # len(range(...)) overflows Py_ssize_t for perfectly valid unbounded ints.
    return values.stop - values.start if isinstance(values, range) else len(values)


def _product(domains: Sequence[range | tuple[int, ...]]) -> Iterator[tuple[int, ...]]:
    """Iterate without itertools.product's eager copies of arbitrarily wide ranges."""
    point = [0] * len(domains)

    def visit(index: int) -> Iterator[tuple[int, ...]]:
        if index == len(domains):
            yield tuple(point)
            return
        for value in domains[index]:
            point[index] = value
            yield from visit(index + 1)

    if all(_size(domain) for domain in domains):
        yield from visit(0)


def _check_cap(cap: int) -> None:
    if type(cap) is not int or cap < 0:
        raise ValueError("enumeration cap must be a nonnegative integer")


def domain_check(e: Expr, sort_env: SortEnv, *, cap: int) -> DomainReport:
    """Check every environment of the referenced domains, never a prefix sample."""
    _check_cap(cap)
    names = sorted(_references(e))
    domains = [_domain_values(sort_env[name]) for name in names]
    if math.prod(_size(domain) for domain in domains) > cap:
        return DomainReport(False, ("enumeration cap exceeded",), 0, True)
    issues = []
    enumerated = 0
    for point in _product(domains):
        enumerated += 1
        row_issues: list[str] = []
        _evaluate(e, dict(zip(names, point)), "$", row_issues)
        issues.extend(f"{issue}; source {tuple(names)!r}={point!r}" for issue in row_issues)
    return DomainReport(not issues, tuple(sorted(issues)), enumerated, False)


def ground(e: Expr, sort_env: SortEnv) -> dict[tuple[int, ...], int]:
    """Ground referenced domains; keys follow lexical domain-name order."""
    names = sorted(_references(e))
    domains = [_domain_values(sort_env[name]) for name in names]
    return {point: evaluate(e, dict(zip(names, point))) for point in _product(domains)}


def relation_check(
    component_exprs: Mapping[str, Expr], sort_env: SortEnv,
    bounds: Mapping[str, tuple[int, int]], *, cap: int,
) -> RelationReport:
    """Check the map from *all* supplied source domains to the bounds box.

    Unused source dimensions still cause replication. Source and component tuple
    names are lexical. A capped report proves none of the four properties. Target
    coverage is decided by cardinality, so a huge bounds box does not require a
    huge enumeration; at most eight concrete failing tuples are retained.
    """
    _check_cap(cap)
    components = sorted(component_exprs)
    if set(components) != set(bounds):
        raise ExprError(["$: bounds must name exactly the relation components"])
    for name in components:
        bound = bounds[name]
        if (
            not isinstance(bound, tuple) or len(bound) != 2
            or any(type(value) is not int for value in bound) or bound[0] > bound[1]
        ):
            raise ExprError([f"$: component {name!r} requires half-open integer bounds"])
    names = sorted(sort_env)
    domains = [_domain_values(sort_env[name]) for name in names]
    if math.prod(_size(domain) for domain in domains) > cap:
        return RelationReport(False, False, False, False, ("enumeration cap exceeded",), True)
    total = in_bounds = injective = True
    witnesses: list[str] = []
    image: dict[tuple[int, ...], tuple[int, ...]] = {}
    bounded_image: set[tuple[int, ...]] = set()
    for point in _product(domains):
        env = dict(zip(names, point))
        source = f"source {tuple(names)!r}={point!r}"
        try:
            target = tuple(evaluate(component_exprs[name], env) for name in components)
        except ExprError as exc:
            total = False
            if len(witnesses) < 8:
                witnesses.append(f"non-total: {source}; {'; '.join(exc.issues)}")
            continue
        destination = f"components {tuple(components)!r}={target!r}"
        within = all(bounds[name][0] <= value < bounds[name][1] for name, value in zip(components, target))
        if within:
            bounded_image.add(target)
        else:
            in_bounds = False
            if len(witnesses) < 8:
                witnesses.append(f"out of bounds: {source} -> {destination}")
        if target in image:
            injective = False
            if len(witnesses) < 8:
                witnesses.append(
                    f"not injective: source {tuple(names)!r}={image[target]!r} and {point!r} -> {destination}"
                )
        else:
            image[target] = point
    box = [range(*bounds[name]) for name in components]
    covering = len(bounded_image) == math.prod(_size(domain) for domain in box)
    if not covering and len(witnesses) < 8:
        for target in _product(box):
            if target not in bounded_image:
                witnesses.append(f"not covering: components {tuple(components)!r}={target!r}")
                if len(witnesses) == 8:
                    break
    return RelationReport(total, total and in_bounds, total and injective, covering, tuple(witnesses), False)


__all__ = [
    "Domain", "SortEnv", "Expr", "ExprError", "DomainReport", "RelationReport",
    "parse", "to_json_obj", "digest", "evaluate", "domain_check", "ground", "relation_check",
]
