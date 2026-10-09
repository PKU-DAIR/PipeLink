"""Conservative connection-local event lifetimes and physical reuse witnesses."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import prod
from typing import Any, Sequence

from pipec2.model import Claim, ClauseFailure, Pipeline, Solution, StageManifest, TensorLifetime
from pipec2.profile import get_profile

from . import expr
from .contracts import Lend, Participants, TensorExport, TensorImport
from .plan import EXPRESSION_CAP, _environments, _size

_START, _END = -1, 3


@dataclass(frozen=True)
class _Chain:
    name: str
    export_ref: str
    import_ref: str
    export: TensorExport
    imp: TensorImport
    invocations: frozenset[str]
    reasons: tuple[str, ...]
    done_reasons: tuple[str, ...]
    fanout: bool

    def position(self, endpoint: str, event: str | None) -> int | None:
        if self.reasons or event is None:
            return None
        if endpoint == self.export_ref:
            events = {self.export.identity.work: 0, self.export.readiness.after: 1,
                      self.export.readiness.reuse_after: 2}
        elif endpoint == self.import_ref:
            events = {**{name: 0 for name in self.invocations},
                      self.export.readiness.after: 1, self.imp.completion: 2}
        else:
            return None
        position = events.get(event)
        if position == 2 and (self.done_reasons or (endpoint == self.export_ref and self.fanout)):
            return None
        return position

    def interval(self, endpoint: str, start: str, end: str | None) -> tuple[int, int]:
        lo, hi = self.position(endpoint, start), self.position(endpoint, end)
        return (_START if lo is None else lo, _END if hi is None else hi)

    def report(self) -> dict[str, Any]:
        positions = {}
        for name, position in (("work", 0), ("ready", 1), ("done", 2)):
            reasons = list(self.reasons + (self.done_reasons if position == 2 else ()))
            export_reasons = reasons + (["export_fanout"] if position == 2 and self.fanout else [])
            positions[name] = {"position": position, "resolved": not reasons,
                               "reasons": reasons, "export_resolved": not export_reasons,
                               "export_reasons": export_reasons}
        return {"name": self.name, "export": self.export_ref, "import": self.import_ref,
                "positions": positions}


def _members(participants: Participants, bindings: dict[str, int]) -> set[int]:
    if participants.set is not None:
        return set(participants.set)
    domains = {domain.name: domain for domain in participants.domain or ()
               if domain.name in expr._references(participants.expr)}
    return {expr.evaluate(participants.expr, {**bindings, **point})
            for point in _environments(domains)}


def _cohort_covers(export, imp, sites, owners) -> bool:
    readiness = next((event for event in imp.rendezvous if event.event == export.readiness.after), None)
    if readiness is None:
        return False
    groups = [readiness.cohort, *(site.participants for site in sites)]
    references = set().union(*(expr._references(group.expr) for group in groups if group.expr is not None))
    domains = {binding.name: binding.domain for owner in owners for binding in owner.tensor.bindings
               if binding.name in references}
    for name, value in export.fixed_bindings.items():
        if name in domains:
            domains[name] = expr.Domain(name, domains[name].sort, values=(value,))
    count = prod(_size(domain) for domain in domains.values())
    if any(count * prod(_size(domain) for domain in group.domain or ()
                        if domain.name in references) > EXPRESSION_CAP for group in groups):
        return False
    # Compare at each binding, not unions: opposite binding-dependent cohorts
    # can have identical unions without ever rendezvousing together.
    for binding in _environments(domains):
        cohort = _members(readiness.cohort, binding)
        if any(not _members(site.participants, binding) <= cohort for site in sites):
            return False
    return True


@dataclass(frozen=True)
class _Witness:
    chain: _Chain
    event: str
    helper: str
    ending_handle: str
    left: Claim
    right: Claim


class LifetimeResolution:
    """A fresh composition's ordering permission, validation, and selected overlaps."""

    def __init__(self, pipeline: Pipeline, stages: Sequence[StageManifest]) -> None:
        by_id = {stage.id: stage for stage in stages}
        counts = Counter(connection.export for connection in pipeline.tensor_connections)
        self._chains: list[_Chain] = []
        for connection in sorted(pipeline.tensor_connections, key=lambda item: item.name):
            producer_id, export_name = connection.export.rsplit(":", 1)
            consumer_id, import_name = connection.import_.rsplit(":", 1)
            producer, consumer = by_id[producer_id], by_id[consumer_id]
            export = next(item for item in producer.tensor.exports if item.name == export_name)
            imp = next(item for item in consumer.tensor.imports if item.name == import_name)
            sites = [site for site in consumer.tensor.sites if site.name in imp.sites]
            reasons = []
            if not sites or any("enter_once" not in site.guarantees for site in sites):
                reasons.append("missing_enter_once")
            if export.readiness.reuse_after != imp.completion:
                reasons.append("early_reclaim")
            owners = (producer,) if producer.id == consumer.id else (producer, consumer)
            if not _cohort_covers(export, imp, sites, owners):
                reasons.append("readiness_cohort_not_covering")
            done_reasons = (() if sites and all("single_completion" in site.guarantees for site in sites)
                            else ("missing_single_completion",))
            self._chains.append(_Chain(connection.name, connection.export, connection.import_, export, imp,
                                       frozenset(site.invocation for site in sites), tuple(reasons),
                                       done_reasons, counts[connection.export] != 1))
        self._claims = [(stage.id, claim) for stage in stages for claim in stage.claims]
        self._resources = [(stage.id, f"{stage.id}:{endpoint.name}", direction, resource)
                           for stage in stages if stage.tensor is not None
                           for direction, endpoints in (("backing", stage.tensor.exports), ("lends", stage.tensor.imports))
                           for endpoint in endpoints for resource in getattr(endpoint, direction)]
        self.failures: list[ClauseFailure] = []
        for stage_id, endpoint, direction, resource in self._resources:
            if direction == "backing" and any(
                endpoint == chain.export_ref
                and chain.interval(endpoint, resource.lifetime["from"], resource.lifetime.get("to"))[1] < 2
                for chain in self._chains
            ):
                self._fail(stage_id, endpoint, direction, resource, None,
                           "export backing must remain live through completion")
                continue
            claims = [(owner, claim) for owner, claim in self._claims
                      if claim.handle == resource.location.handle and isinstance(claim.lifetime, TensorLifetime)]
            for owner, claim in claims:
                if not self._contains(owner, claim, endpoint, resource):
                    self._fail(stage_id, endpoint, direction, resource, claim, "resource lifetime exceeds its claim")
        profile = get_profile(pipeline.target.profile)
        handles: dict[str, list[tuple[str, Claim]]] = {}
        for owner, claim in self._claims:
            domain = profile.domains.get(claim.domain)
            if domain is not None and (domain.physical_domain or claim.domain) == "smem":
                handles.setdefault(claim.handle, []).append((owner, claim))
        self._witnesses: dict[tuple[str, str], list[_Witness]] = {}
        for left, right in combinations(sorted(handles), 2):
            witnesses = [self._ordered(a, b) for a in handles[left] for b in handles[right]]
            # The solver's permission is per handle; every declaration must agree.
            if all(witness is not None for witness in witnesses):
                self._witnesses[left, right] = witnesses
        self.ordered_pairs = frozenset(self._witnesses)

    def _fail(self, stage, endpoint, direction, resource, claim, message):
        section = "exports" if direction == "backing" else "imports"
        self.failures.append(ClauseFailure(
            "T-PLAN", f"stages.{stage}.tensor.{section}.{endpoint.rsplit(':', 1)[1]}.{direction}.{resource.name}",
            message, {"handle": resource.location.handle,
                      "claim_lifetime": None if claim is None else claim.lifetime.to_json_obj(),
                      "resource_lifetime": resource.lifetime}))

    @staticmethod
    def _interval(owner: str, claim: Claim, chain: _Chain) -> tuple[int, int]:
        if not isinstance(claim.lifetime, TensorLifetime):
            return _START, _END
        lifetime = claim.lifetime
        return chain.interval(f"{owner}:{lifetime.tensor}", lifetime.from_, lifetime.to)

    def _contains(self, owner: str, claim: Claim, endpoint: str, resource: Lend) -> bool:
        if claim.lifetime == "kernel":
            return True
        if not isinstance(claim.lifetime, TensorLifetime):
            return False
        claim_endpoint = f"{owner}:{claim.lifetime.tensor}"
        for chain in self._chains:
            if not {endpoint, claim_endpoint}.intersection((chain.export_ref, chain.import_ref)):
                continue
            lo, hi = self._interval(owner, claim, chain)
            start, end = chain.interval(endpoint, resource.lifetime["from"], resource.lifetime.get("to"))
            if not lo <= start <= end <= hi:
                return False
        return True

    def covers(self, stage_id: str, claim: Claim, resource: Lend) -> bool:
        """Whether a reservation claim contains this resource on every connection."""
        owners = [owner for owner, item in self._claims if item is claim]
        if not owners:
            owners = [owner for owner, item in self._claims if item == claim]
        endpoints = [endpoint for stage, endpoint, _, item in self._resources
                     if stage == stage_id and item == resource]
        return bool(owners and endpoints) and all(self._contains(owner, claim, endpoint, resource)
                                                  for owner in owners for endpoint in endpoints)

    def _ordered(self, left: tuple[str, Claim], right: tuple[str, Claim]) -> _Witness | None:
        a, b = left[1], right[1]
        if a.alias is not None and a.alias == b.alias:
            return None
        for chain in self._chains:
            alo, ahi = self._interval(*left, chain)
            blo, bhi = self._interval(*right, chain)
            if ahi <= blo:
                end, ending = ahi, a
            elif bhi <= alo:
                end, ending = bhi, b
            else:
                continue
            if end not in (1, 2):
                continue
            kind = "ready" if end == 1 else "done"
            event = chain.export.readiness.after if end == 1 else chain.imp.completion
            return _Witness(chain, event, chain.imp.callable.events[kind], ending.handle, a, b)
        return None

    def report(self, solution: Solution) -> dict[str, Any]:
        """Report permission reasons and only physically selected shared-memory reuse."""
        overlaps = []
        intervals = solution.intervals.get("smem", ())
        for (left, right), witnesses in sorted(self._witnesses.items()):
            for a in (interval for interval in intervals if interval.handle == left):
                for b in (interval for interval in intervals if interval.handle == right):
                    if max(a.lo, b.lo) >= min(a.hi, b.hi):
                        continue
                    witness = next(w for w in witnesses
                                   if w.left.lifetime.label() == a.phase and w.right.lifetime.label() == b.phase)
                    overlaps.append({"handles": [left, right], "ranges": [[a.lo, a.hi], [b.lo, b.hi]],
                                     "connection": witness.chain.name, "event": witness.event,
                                     "helper": witness.helper, "ending_handle": witness.ending_handle})
        return {"connections": [chain.report() for chain in self._chains], "overlaps": overlaps}


def resolve_lifetimes(pipeline: Pipeline, stages: Sequence[StageManifest]) -> LifetimeResolution:
    return LifetimeResolution(pipeline, stages)
