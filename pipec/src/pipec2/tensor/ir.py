"""Finite local Access IR: effects, local metadata, observations, and returns.

Graph cycles are intentional.  Expressions in operands are the tensor expression
language, never Python code; their site-local environments are checked by the
execution model.  No node can inspect ghost payload identities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

from pipec2.canonjson import canonical_bytes, sha256_hex
from pipec2.model import ClauseFailure, PipeLinkError

from . import expr
from .contracts import Lend


class IRError(PipeLinkError):
    pass


def _fail(path: str, message: str) -> None:
    raise IRError([ClauseFailure("T-MODEL", path, message, {})])


def _json(value: Any) -> Any:
    if isinstance(value, expr.Expr):
        return expr.to_json_obj(value)
    if isinstance(value, Lend):
        return value.to_json_obj()
    if isinstance(value, Mapping):
        return {key: _json(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    _fail("$.ir", "expected finite JSON operands")


@dataclass(frozen=True)
class Issue:
    primitive: str
    operands: Mapping[str, Any]
    next: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", _json(self.operands))


@dataclass(frozen=True)
class Meta:
    dest: str
    expr: Any
    next: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "expr", _json(self.expr))


@dataclass(frozen=True)
class Branch:
    observation: str
    next_true: int
    next_false: int


@dataclass(frozen=True)
class Return:
    results: tuple[tuple[int, int, int], ...] = ()


Node: TypeAlias = Issue | Meta | Branch | Return


@dataclass(frozen=True)
class ControlGraph:
    nodes: tuple[Node, ...]
    entry: int = 0

    def __post_init__(self) -> None:
        if not self.nodes:
            _fail("$.graphs", "a control graph must contain an entry node")
        targets = [self.entry]
        for node in self.nodes:
            if isinstance(node, (Issue, Meta)):
                targets.append(node.next)
            elif isinstance(node, Branch):
                targets.extend((node.next_true, node.next_false))
            elif not isinstance(node, Return):
                _fail("$.graphs", "unknown control node (oracle nodes are forbidden)")
        if any(type(target) is not int or not 0 <= target < len(self.nodes) for target in targets):
            _fail("$.graphs", "controller edge is outside the declared graph")


@dataclass(frozen=True)
class Program:
    resources: tuple[Lend, ...]
    borrows: Mapping[str, int | str]
    graphs: Mapping[str, ControlGraph]
    results: tuple[tuple[int, int, int], ...]
    view_binding: str | None = field(default=None, metadata={"omit_none": True})

    def to_json_obj(self) -> dict[str, Any]:
        return to_json_obj(self)

    @classmethod
    def from_json_obj(cls, obj: Any) -> Program:
        return parse(obj)


def node_to_json_obj(node: Node) -> dict[str, Any]:
    if isinstance(node, Issue):
        return {"kind": "issue", "primitive": node.primitive, "operands": _json(node.operands), "next": node.next}
    if isinstance(node, Meta):
        return {"kind": "metadata", "dest": node.dest, "expr": _json(node.expr), "next": node.next}
    if isinstance(node, Branch):
        return {"kind": "branch", "observation": node.observation, "next_true": node.next_true, "next_false": node.next_false}
    if isinstance(node, Return):
        return {"kind": "return", "results": _json(node.results)}
    _fail("$.graphs", "unknown node")


def to_json_obj(program: Program) -> dict[str, Any]:
    return {
        "resources": [_json(resource) for resource in program.resources],
        "borrows": _json(program.borrows),
        "graphs": {
            site: {"nodes": [node_to_json_obj(node) for node in graph.nodes], "entry": graph.entry}
            for site, graph in sorted(program.graphs.items())
        },
        "results": _json(program.results),
        **({"view_binding": program.view_binding} if program.view_binding is not None else {}),
    }


def _object(obj: Any, keys: set[str], path: str) -> dict[str, Any]:
    if not isinstance(obj, dict) or set(obj) != keys:
        _fail(path, f"expected exactly {sorted(keys)}")
    return obj


def _results(obj: Any, path: str, *, view: bool = False) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(obj, list) or any(
        not isinstance(row, list) or len(row) != 3 or any(type(item) is not int for item in row)
        or (row[0] != -1 if view else row[0] < 0) or any(item < 0 for item in row[1:])
        for row in obj
    ):
        _fail(path, "results require [participant, position, identity-index] triples")
    return tuple(tuple(row) for row in obj)


def parse(obj: Any) -> Program:
    keys = {"resources", "borrows", "graphs", "results"}
    obj = _object(obj, keys | ({"view_binding"} if isinstance(obj, dict) and "view_binding" in obj else set()), "$")
    if "view_binding" in obj and (not isinstance(obj["view_binding"], str) or not obj["view_binding"]):
        _fail("$.view_binding", "expected a nonempty shared resource name")
    if not isinstance(obj["resources"], list):
        _fail("$.resources", "expected an array")
    if not isinstance(obj["borrows"], dict) or any(
        not isinstance(key, str) or type(value) not in (str, int)
        for key, value in obj["borrows"].items()
    ):
        _fail("$.borrows", "expected named integer or handle bindings")
    if not isinstance(obj["graphs"], dict) or not obj["graphs"]:
        _fail("$.graphs", "expected named site graphs")
    graphs = {}
    for site, raw in sorted(obj["graphs"].items()):
        if not isinstance(site, str) or not site:
            _fail("$.graphs", "site names must be nonempty strings")
        raw = _object(raw, {"nodes", "entry"}, f"$.graphs.{site}")
        if not isinstance(raw["nodes"], list):
            _fail(f"$.graphs.{site}.nodes", "expected an array")
        nodes = []
        for index, node in enumerate(raw["nodes"]):
            path = f"$.graphs.{site}.nodes[{index}]"
            if not isinstance(node, dict):
                _fail(path, "expected a node object")
            kind = node.get("kind")
            if kind == "issue":
                node = _object(node, {"kind", "primitive", "operands", "next"}, path)
                if not isinstance(node["primitive"], str) or not isinstance(node["operands"], dict):
                    _fail(path, "issue requires a primitive name and named operands")
                nodes.append(Issue(node["primitive"], node["operands"], node["next"]))
            elif kind == "metadata":
                node = _object(node, {"kind", "dest", "expr", "next"}, path)
                if not isinstance(node["dest"], str) or not isinstance(node["expr"], list):
                    _fail(path, "metadata requires a destination and expression")
                nodes.append(Meta(node["dest"], node["expr"], node["next"]))
            elif kind == "branch":
                node = _object(node, {"kind", "observation", "next_true", "next_false"}, path)
                if not isinstance(node["observation"], str):
                    _fail(path, "branch requires a local observation name")
                nodes.append(Branch(node["observation"], node["next_true"], node["next_false"]))
            elif kind == "return":
                node = _object(node, {"kind", "results"}, path)
                nodes.append(Return(_results(node["results"], path + ".results", view="view_binding" in obj)))
            else:
                _fail(path, "unknown node kind (oracle nodes are forbidden)")
        graphs[site] = ControlGraph(tuple(nodes), raw["entry"])
    return Program(
        tuple(Lend.from_json_obj(resource) for resource in obj["resources"]),
        dict(sorted(obj["borrows"].items())), graphs, _results(obj["results"], "$.results", view="view_binding" in obj),
        obj.get("view_binding"),
    )


def digest(program: Program) -> str:
    return sha256_hex(canonical_bytes(to_json_obj(program)))
