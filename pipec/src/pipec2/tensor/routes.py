"""Route model: primitives are typed edges, a candidate class is a route.

Routes encode structural feasibility; the public checker certifies semantics.
This module owns the edge table (kind, endpoints, preconditions, realising
families), the fact derivation, permission, and the candidate enumeration with
its structural order. It proposes; ``check.check`` alone certifies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pipec2.model import _JsonMixin
from . import machine
from .contracts import Site, TensorExport, TensorImport

Space = Literal["tmem", "smem", "reg"]
ALIASABLE: frozenset[str] = frozenset({"smem"})
FACTS: tuple[str, ...] = (
    "lend:reg", "lend:smem", "avail:smem", "lend:barrier@cta", "cap:tmem_packet_receipt",
    "g:converged", "g:full_warps", "g:no_early_return",
)


@dataclass(frozen=True)
class Edge:
    name: str
    kind: Literal["transport", "local", "sync"]
    src: Space | None
    dst: Space | None
    requires: frozenset[str]
    families: tuple[str, ...]   # realising primitive families; the first is the round form
    searched: tuple[str, ...]   # subset entering the bounded search
    ops: Literal["receipt", "one", "rendezvous", "none"]  # contribution to the controller bound

    @property
    def charge(self) -> int:
        return sum(machine.PRIMITIVES[self.families[0]].charges.values())

    @property
    def visible_after(self) -> str:
        return machine.PRIMITIVES[self.families[0]].timing["visible_after"]


def _edge(name, kind, src, dst, requires, families, searched=None, ops="none"):
    return Edge(name, kind, src, dst, frozenset(requires), tuple(families),
                tuple(families if searched is None else searched), ops)


EDGES: dict[str, Edge] = {edge.name: edge for edge in (
    _edge("receipt", "transport", "tmem", "reg", {"cap:tmem_packet_receipt"}, ("tmem_packet_receipt",), ops="receipt"),
    _edge("load", "transport", "smem", "reg", {"avail:smem"}, ("smem_load_round", "smem_load_scalar"), ("smem_load_round",), "one"),
    _edge("store", "transport", "reg", "smem", {"lend:smem"}, ("smem_store_round", "smem_store_scalar"), ("smem_store_round",), "one"),
    _edge("copy", "transport", "reg", "reg", {"lend:reg"}, ("reg_copy_round", "reg_copy")),
    _edge("shfl", "local", "reg", "reg", {"g:converged", "g:full_warps"}, ("shfl_xor",)),
    _edge("rendezvous", "sync", None, None, {"lend:barrier@cta", "g:converged", "g:no_early_return"}, ("cta_rendezvous",), ops="rendezvous"),
    _edge("observe", "sync", None, None, (), ("observe_ready",)),
)}
_BRING = {"tmem": "receipt", "smem": "load", "reg": "copy"}


def facts(export: TensorExport, imp: TensorImport, site: Site) -> frozenset[str]:
    """The declared facts a site offers; backing never counts as a lend."""
    lends = {resource.domain for resource in imp.lends if resource.binding != "caller_frame"}
    found = {f"lend:{domain}" for domain in lends & {"reg", "smem"}}
    if "smem" in lends or any(resource.domain == "smem" for resource in export.backing):
        found.add("avail:smem")
    if any(resource.domain == "barrier" and resource.scope == "cta" for resource in imp.lends):
        found.add("lend:barrier@cta")
    found.update(f"cap:{capability}" for capability in export.read_capabilities if capability == "tmem_packet_receipt")
    found.update(f"g:{guarantee}" for guarantee in site.guarantees
                 if guarantee in {"converged", "full_warps", "no_early_return"})
    return frozenset(found)


def permitted_families(available: frozenset[str]) -> frozenset[str]:
    """Families of every edge whose preconditions hold (T1)."""
    return frozenset(family for edge in EDGES.values() if edge.requires <= available for family in edge.families)


@dataclass(frozen=True)
class Route(_JsonMixin):
    source: str                 # the export's placement spaces, joined with "+"
    sink: Space
    bring: tuple[str, ...]      # one transport edge per source space (joint domain)
    relay: bool
    deliver: bool
    shfl: bool
    tier: int

    @property
    def edges(self) -> tuple[str, ...]:
        names: list[str] = list(self.bring)
        if self.relay:
            names += ["store", "rendezvous", "load"]
        if self.deliver:
            names += ["store", "rendezvous"]
        if self.shfl:
            names.append("shfl")
        return tuple(names)

    @property
    def alias(self) -> bool:
        return not self.edges

    @property
    def label(self) -> str:
        if self.alias:
            return "view_alias"
        if self.deliver:
            return "view_materialize"
        base = "staged" if self.relay else "direct"
        return f"{base}_shuffle" if self.relay and self.shfl else "warp" if self.shfl else base

    @property
    def name(self) -> str:
        """Legacy class label: derived, descriptive, never dispatched on."""
        label = self.label
        if self.sink == "smem":
            return label
        if self.tier == 0:
            return f"{label}_constructed"
        return {"direct": "direct_receipt", "warp": "warp_local", "staged": "staged_materialized"}.get(
            label, f"{label}_materialized")

    @property
    def id(self) -> str:
        return f"{self.source}>{'|'.join(self.edges) or 'alias'}>{self.sink}"

    @property
    def requires(self) -> frozenset[str]:
        return frozenset().union(*(EDGES[name].requires for name in self.edges))

    @property
    def cost(self) -> tuple[int, int, str]:
        return len(self.edges), sum(EDGES[name].charge for name in self.edges), self.id

    @property
    def families(self) -> frozenset[str]:
        """Every family the constructive fold may issue on this route."""
        names = set(self.edges) | {"copy", "observe"}
        return frozenset(family for name in names for family in EDGES[name].families)

    def searched(self, available: frozenset[str]) -> frozenset[str]:
        """Tier-1 action domain: searched edge families, the permitted register copy, readiness."""
        names = set(self.edges) | {"observe"}
        if EDGES["copy"].requires <= available:
            names.add("copy")
        return frozenset(family for name in names for family in EDGES[name].searched)

    def bound(self, instance: machine.Instance, floor: int) -> int:
        """K(route): a fold over the edges' controller-level operation counts."""
        derived = instance.derived
        private_slots = next((resource.extent for resource in instance.resources
                              if resource.name == instance.result_resource), 0)
        receipt = 2 * ((private_slots + instance.packet_columns - 1) // instance.packet_columns)
        rendezvous = min(2, max((resource.sync.generations for resource in instance.resources
                                 if resource.name in derived.storage and resource.domain == "barrier"
                                 and resource.sync is not None), default=0))
        ops = {"receipt": receipt, "one": 1, "rendezvous": rendezvous, "none": 0}
        return max(floor, int(not instance.ready_at_entry) + sum(ops[EDGES[name].ops] for name in self.edges))

    def to_json_obj(self) -> dict[str, Any]:
        return {"id": self.id, "source": self.source, "sink": self.sink, "edges": list(self.edges),
                "tier": self.tier, "label": self.label, "cost": list(self.cost[:2])}


def _shapes(sources: tuple[Space, ...], sink: Space) -> list[Route]:
    """Routes for one source-space set ``sources``.

    A register source brings by copy only when it delivers straight to the
    result; a following bank step reads the fragment.
    """
    source = "+".join(sources)
    shapes = []
    if sources == (sink,) and sink in ALIASABLE:
        shapes.append(Route(source, sink, (), False, False, False, 0))
    if sink == "reg":
        for relay in (False, True):
            for shfl in (False, True):
                bring = tuple(_BRING[space] for space in sources if space != "reg" or not (relay or shfl))
                shapes.extend(Route(source, sink, bring, relay, False, shfl, tier) for tier in (0, 1))
    else:
        bring = tuple(_BRING[space] for space in sources if space != "reg")
        shapes.append(Route(source, sink, bring, False, True, False, 0))
    return shapes


def _subsets(spaces: tuple[Space, ...]):
    """Every nonempty subset in the fixed space order: singletons first, the joint set last."""
    for mask in range(1, 1 << len(spaces)):
        yield tuple(space for index, space in enumerate(spaces) if mask >> index & 1)


def candidates(sources: Space | Sequence[Space], sink: Space,
               available: frozenset[str]) -> tuple[list[Route], list[tuple[Route, frozenset[str]]]]:
    """Ordered permitted routes, and the unpermitted ones with their missing facts (T4/T5).

    An export with guarded placements in several spaces is routed from every
    nonempty subset of them: singleton routes keep each space's own envelope
    (alias stays a candidate whenever the sink space is a source, an inert
    placement never demands facts), and the joint route keeps the old global
    domain in which one controller brings from several spaces.
    """
    spaces = (sources,) if isinstance(sources, str) else tuple(sources)
    ordered = sorted((route for subset in _subsets(spaces) for route in _shapes(subset, sink)),
                     key=lambda route: (route.tier, route.cost))
    permitted = [route for route in ordered if route.requires <= available]
    rejected = [(route, route.requires - available) for route in ordered if not route.requires <= available]
    return permitted, rejected


def source_spaces(export: TensorExport) -> tuple[Space, ...]:
    """Every space some placement of the export lives in, in a fixed order."""
    return tuple(space for space in ("tmem", "smem", "reg") if any(p.space == space for p in export.placements))


def sink_space(imp: TensorImport) -> Space:
    return "smem" if any(demand.kind == "memory_view" for demand in imp.demands) else "reg"


__all__ = ["ALIASABLE", "EDGES", "FACTS", "Edge", "Route", "candidates", "facts",
           "permitted_families", "sink_space", "source_spaces"]
