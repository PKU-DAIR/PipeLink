"""Exact finite product exploration and omega-regular progress checking.

Safety is checked before fairness, including prefixes of unfair executions.
State merging compares all immutable machine fields.  Additional reductions
require recorded proofs: static transport phases, nonobserving persistent sets,
arrivals-only rendezvous frontiers, and interaction-free warp-product factors.
Justice/compassion acceptance uses recurrent-subgraph elimination, not only maximal SCCs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Mapping

from pipec2.model import ClauseFailure, _JsonMixin

from . import ir
from .check_static import certify as _certify_static
from .machine import IdentityVector, Instance, ModelError, Monitor, State, Transition, _ROUND_SCALARS, _stay


@dataclass(frozen=True)
class CheckResult(_JsonMixin):
    status: str
    violations: tuple[ClauseFailure, ...]
    explored_states: int
    reductions: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]


def _scc(vertices: set[int], edges: Mapping[int, tuple[tuple[int, str], ...]]) -> list[set[int]]:
    """Iterative Kosaraju; controller depth does not consume Python call depth."""
    visited = set()
    order = []
    for root in sorted(vertices):
        if root in visited:
            continue
        visited.add(root)
        stack = [(root, iter(edges.get(root, ())))]
        while stack:
            vertex, children = stack[-1]
            child = next(children, None)
            if child is None:
                order.append(vertex)
                stack.pop()
            elif child[0] in vertices and child[0] not in visited:
                visited.add(child[0])
                stack.append((child[0], iter(edges.get(child[0], ()))))
    reverse = {vertex: [] for vertex in vertices}
    for vertex in vertices:
        for target, _ in edges.get(vertex, ()):
            if target in vertices:
                reverse[target].append(vertex)
    components = []
    visited.clear()
    for root in reversed(order):
        if root in visited:
            continue
        component = set()
        pending = [root]
        visited.add(root)
        while pending:
            vertex = pending.pop()
            component.add(vertex)
            for target in reverse[vertex]:
                if target not in visited:
                    visited.add(target)
                    pending.append(target)
        if len(component) > 1 or any(target == root for target, _ in edges.get(root, ())):
            components.append(component)
    return components


def _fair_component(vertices: set[int], edges: Mapping[int, tuple[tuple[int, str], ...]],
                    enabled: Mapping[int, frozenset[str]], monitors: tuple[Monitor, ...]) -> tuple[set[int], dict] | None:
    """Existence of a recurrent run satisfying all Streett-style assumptions.

    When Q is absent, a fair run must eventually avoid P.  Remove exactly P's
    states (justice enablement) or edges (compassion premise), decompose again,
    and continue.  A remaining strongly connected graph has a closed walk through
    every required witness; therefore union-of-witnesses acceptance is exact.
    """
    work = [(vertices, dict(edges))]
    while work:
        region, graph = work.pop()
        for component in _scc(region, graph):
            internal = {vertex: tuple((target, label) for target, label in graph.get(vertex, ()) if target in component)
                        for vertex in component}
            labels = {label for outgoing in internal.values() for _, label in outgoing}
            rejected = False
            for monitor in monitors:
                if monitor.kind == "justice":
                    triggering = {vertex for vertex in component if enabled[vertex] & monitor.labels}
                    if triggering and not labels & monitor.labels:
                        remainder = component - triggering
                        if remainder:
                            work.append((remainder, internal))
                        rejected = True
                        break
                elif monitor.kind == "compassion":
                    if labels & monitor.premise and not labels & monitor.consequence:
                        reduced = {vertex: tuple(edge for edge in outgoing if edge[1] not in monitor.premise)
                                   for vertex, outgoing in internal.items()}
                        work.append((component, reduced))
                        rejected = True
                        break
            if not rejected:
                return component, {"cycle_states": sorted(component), "recurrent_labels": sorted(labels),
                                   "assumptions": [monitor.name for monitor in monitors if monitor.kind in {"justice", "compassion"}]}
    return None


class _PersistentSet:
    """Singleton ample sets with checked future-dependency and cycle provisos.

    Enabled-only disjointness is insufficient: another actor can execute an
    independent instruction and then enable a conflicting one.  We therefore
    compare against every other action reachable while the selected actors stay
    fixed, stopping only at collective gates requiring those actors.  Acyclic
    local graphs and identity-state justice monitors discharge the cycle and
    visibility provisos.  Cyclic or unprojectable candidates retain full
    exploration unless a separately verified reduction applies.
    """

    def __init__(self, model: Instance):
        self.model = model
        self.applications = 0
        self.omitted = 0
        self.footprint_checks = 0
        self.classes = set()
        self.graph_edges = {}
        self.cache = {}
        self.projected = {}
        self.environment = {}
        self.supported = all(monitor.kind == "justice" and monitor.step is _stay
                             and monitor.states == (0,) for monitor in model.monitors)
        for site, graph in model.program_ir.graphs.items():
            edges = []
            incoming = [0] * len(graph.nodes)
            for node in graph.nodes:
                targets = (node.next,) if isinstance(node, (ir.Issue, ir.Meta)) else (
                    tuple(sorted({node.next_true, node.next_false})) if isinstance(node, ir.Branch) else ()
                )
                edges.append(targets)
                for target in targets:
                    incoming[target] += 1
            ready = deque(index for index, degree in enumerate(incoming) if degree == 0)
            order = []
            while ready:
                pc = ready.popleft()
                order.append(pc)
                for target in edges[pc]:
                    incoming[target] -= 1
                    if incoming[target] == 0:
                        ready.append(target)
            if len(order) != len(graph.nodes):
                self.supported = False
                continue
            self.graph_edges[site] = tuple(edges)

    def footprint(self, state: State, actor: int, pc: int):
        key = (state.world, state.owner_phase, actor, pc)
        if key not in self.projected:
            self.projected[key] = self.model.controller_footprint(state, actor, pc)
        return self.projected[key]

    def future(self, state: State, actor: int, selected_actor: int):
        pc = state.positions[actor]
        if pc < 0:
            return frozenset()
        selected_site, selected_participant = self.model.actors[selected_actor]
        key = (state.world, state.owner_phase, actor, pc, selected_site, selected_participant // self.model.warp_size)
        if key not in self.cache:
            footprints, visited, pending = [], set(), [pc]
            site = self.model.actors[actor][0]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                node = self.model.program_ir.graphs[site].nodes[current]
                if isinstance(node, ir.Issue) and node.primitive in {"tmem_packet_receipt", "shfl_xor"} and selected_actor in self.model._members(actor):
                    # This collective cannot fire while the selected actor stays
                    # at its noncollective instruction.  Do not cross that gate.
                    continue
                footprint = self.footprint(state, actor, current)
                if footprint is None:
                    self.cache[key] = None
                    break
                footprints.append(footprint)
                if isinstance(node, ir.Issue) and node.primitive == "cta_rendezvous":
                    # The arrival is independent; publication cannot happen until
                    # the selected (currently nonblocked) actor also arrives.
                    continue
                pending.extend(self.graph_edges[site][current])
            else:
                self.cache[key] = frozenset().union(*footprints)
        return self.cache[key]

    def participants(self, state: State, transition: Transition) -> tuple[int, ...]:
        if transition.observation or transition.primitive == "cta_rendezvous":
            return ()
        if transition.site == "ENV":
            if not transition.label.startswith("complete:"):
                return ()
            pending = next(item for item in state.outstanding if item.label == transition.label)
            group = tuple(actor for actor, (site, participant) in enumerate(self.model.actors)
                          if site == pending.site and participant // self.model.warp_size == pending.warp)
            if not group or len({state.positions[actor] for actor in group}) != 1:
                return ()
            # Never hide a premature read by selecting completion ahead of it.
            # This full warp must already be parked at THIS handle's wait.
            for actor in group:
                node = self.model._node(state, actor)
                if not isinstance(node, ir.Issue) or node.primitive != "tmem_packet_receipt" or node.operands.get("phase") != "wait" or node.operands.get("handle", "receipt") != pending.handle:
                    return ()
            return group
        if transition.primitive is None:
            return transition.actors
        primitive = self.model.catalogue.get(transition.primitive)
        if primitive is None:
            return ()
        if primitive.participants == "thread" and not primitive.blocking:
            return transition.actors
        if primitive.participants == "warp":
            if transition.primitive == "tmem_packet_receipt":
                node = self.model._node(state, transition.actors[0])
                if node.operands.get("phase", "issue") == "wait":
                    site, participant = self.model.actors[transition.actors[0]]
                    if not any(item.site == site and item.warp == participant // self.model.warp_size
                               and item.handle == node.operands.get("handle", "receipt") and item.completed for item in state.outstanding):
                        return ()
                return transition.actors
            if not primitive.blocking:
                return transition.actors
        return ()

    def choose(self, state: State, transitions: tuple[Transition, ...]):
        if not self.supported or state.world < 0 or len(transitions) < 2:
            return transitions, None
        candidates = [(transition, self.participants(state, transition)) for transition in transitions]
        for candidate, participants in sorted(candidates, key=lambda item: (item[0].site, item[1], item[0].label)):
            if not participants:
                continue
            footprint = (self.model.transition_footprint(state, candidate)
                         if candidate.site == "ENV" or candidate.primitive in _ROUND_SCALARS
                         else self.footprint(state, participants[0], state.positions[participants[0]]))
            if footprint is None:
                continue
            independent = True
            for actor in range(len(self.model.actors)):
                if actor in participants:
                    continue
                other = self.future(state, actor, participants[0])
                self.footprint_checks += 1
                if other is None or not footprint.isdisjoint(other):
                    independent = False
                    break
            if not independent:
                continue
            for transition in transitions:
                if transition.site == "ENV" and transition != candidate:
                    pending = next((item for item in state.outstanding if item.label == transition.label), None)
                    key = (state.world, transition.label, pending)
                    if key not in self.environment:
                        self.environment[key] = self.model.transition_footprint(state, transition)
                    other = self.environment[key]
                    self.footprint_checks += 1
                    if other is None or not footprint.isdisjoint(other):
                        independent = False
                        break
            if not independent:
                continue
            successor = self.model._apply_enabled(state, candidate)
            if successor.errors:
                continue
            self.applications += 1
            self.omitted += len(transitions) - 1
            self.classes.add(candidate.primitive or type(self.model._node(state, participants[0])).__name__)
            return (candidate,), successor
        return transitions, None

    def record(self):
        return {"kind": "checked-persistent-set", "verified": True, "applications": self.applications,
                "omitted_interleavings": self.omitted, "footprint_checks": self.footprint_checks,
                "transition_classes": sorted(self.classes),
                "justification": "exact grounded cells/control/metadata/handles/epochs; disjoint from every other action reachable before the selected actors move, stopping only at collectives that require those actors; enabled environment effects checked separately",
                "cycle_proviso": "all local graphs verified acyclic; bounded scalar-round cursors and finite owner/target completions cannot independently cycle",
                "visibility_proviso": "selected transition nonobserving and nonblocking at this state; warp receipts remain indivisible; ENV completion only at its full-warp wait; identity-state justice monitors unchanged"}


class _RendezvousFrontier:
    """Exact arrival-permutation quotient; never batches or publishes arrivals."""

    def __init__(self, model: Instance):
        self.model = model
        self.applications = 0
        self.omitted = 0
        self.checked_arrivals = 0
        self.resources = set()
        self.monitor_invariant = all(monitor.step is _stay for monitor in model.monitors)

    def choose(self, state: State, transitions: tuple[Transition, ...]):
        if not self.monitor_invariant or len(transitions) < 2:
            return transitions, None
        resource, waiting = None, set()
        for transition in transitions:
            if transition.site == "ENV" or transition.observation or transition.primitive != "cta_rendezvous" or len(transition.actors) != 1:
                return transitions, None
            actor = transition.actors[0]
            node = self.model._node(state, actor)
            if not isinstance(node, ir.Issue):
                return transitions, None
            current = node.operands.get("resource")
            if resource is None:
                resource = current
            if current != resource or actor in waiting:
                return transitions, None
            waiting.add(actor)
        if any(arrival.resource != resource for arrival in state.arrivals):
            return transitions, None
        arrived = {actor for arrival in state.arrivals for actor in arrival.actors}
        if arrived & waiting or arrived | waiting != set(range(len(self.model.actors))):
            return transitions, None
        selected, successor = None, None
        for transition in sorted(transitions, key=lambda item: (item.site, item.actors)):
            candidate = self.model._apply_enabled(state, transition)
            self.checked_arrivals += 1
            if candidate.errors:
                return transitions, None
            if (candidate.cells is not state.cells or candidate.published is not state.published
                or candidate.positions != state.positions or candidate.metadata != state.metadata
                or candidate.outstanding != state.outstanding or candidate.accesses != state.accesses
                or candidate.owner_phase != state.owner_phase or candidate.generations != state.generations
                or candidate.round_positions != state.round_positions or candidate.monitor_states != state.monitor_states):
                return transitions, None
            actual = {actor for arrival in candidate.arrivals for actor in arrival.actors}
            if actual != arrived | set(transition.actors):
                return transitions, None
            if selected is None:
                selected, successor = transition, candidate
        self.applications += 1
        self.omitted += len(transitions) - 1
        self.resources.add(resource)
        return (selected,), successor

    def record(self):
        return {"kind": "checked-rendezvous-frontier", "verified": True, "applications": self.applications,
                "omitted_permutations": self.omitted, "checked_scalar_arrivals": self.checked_arrivals,
                "resources": sorted(self.resources),
                "conditions": ["every enabled system transition is a nonobserving arrival at one rendezvous instance",
                               "every declared participant has arrived or has its own enabled legal arrival",
                               "each step adds exactly its actor; payload/control/monitor state otherwise unchanged"],
                "justification": "arrival set additions commute; each scalar arrival remains represented; publication is still the separate full-set ENV transition",
                "fairness": "identity-state monitors; every completed frontier contains the same transition-label multiset"}


def _check_product(instance: Instance, program: ir.Program, *, budget) -> CheckResult:
    max_states, max_operations = _limits(budget)
    model = instance.with_program(program, max_operations=max_operations)
    provenance: dict[str, Any] = {"algorithm": "exact-product-streett-v1", "model_digest": instance.model_digest,
        "ir_digest": ir.digest(program), "budget": {"max_states": max_states, "max_operations": max_operations},
        "terminal_deliveries": [], "safety": "every reachable prefix, independent of fairness",
        "assumptions": [monitor.name for monitor in model.monitors if monitor.kind in {"justice", "compassion"}]}
    states: list[State] = []
    index: dict[State, int] = {}
    predecessors: dict[int, tuple[int, Transition]] = {}
    merged = 0
    persistent = _PersistentSet(model)
    frontier = _RendezvousFrontier(model)
    vectors: dict[IdentityVector, IdentityVector] = {}
    chunks: dict[tuple[int, ...], tuple[int, ...]] = {}
    interned_chunks = 0
    interned_vectors = 0

    def intern_vector(vector: IdentityVector) -> IdentityVector:
        nonlocal interned_chunks
        known = vectors.get(vector)
        if known is not None:
            return known
        canonical, replacements = vector.intern_chunks(chunks)
        interned_chunks += replacements
        vectors[canonical] = canonical
        return canonical

    def intern(state: State) -> State:
        nonlocal interned_vectors
        cells = intern_vector(state.cells)
        published = intern_vector(state.published)
        if cells is state.cells and published is state.published:
            return state
        interned_vectors += int(cells is not state.cells) + int(published is not state.published)
        return replace(state, cells=cells, published=published)

    def trace(vertex: int) -> list[dict[str, Any]]:
        result = []
        while vertex in predecessors:
            vertex, transition = predecessors[vertex]
            result.append({"site": transition.site, "label": transition.label, "primitive": transition.primitive,
                           "actors": list(transition.actors), "detail": transition.detail,
                           "controllers": [[model.actors[actor][0], states[vertex].positions[actor]] for actor in transition.actors]})
        result.reverse()
        return result

    def result(status: str, message: str | None = None, vertex: int | None = None, detail: dict | None = None) -> CheckResult:
        provenance["accessed_bytes"] = dict(model.accessed_bytes)
        violations = ()
        if message is not None:
            data = dict(detail or {})
            if vertex is not None:
                data["trace"] = trace(vertex)
                if status == "UNSAFE" and data["trace"]:
                    data["counterexample_kind"] = "reachable_safety_prefix"
                    data["controller_rows"] = [list(row) for row in sorted({
                        tuple(row) for step in data["trace"] for row in step["controllers"] if row[1] >= 0
                    })]
            violations = (ClauseFailure("T-MODEL", "$.tensor.program", message, data),)
        reductions = ({"kind": "exact-state-merging", "verified": True,
                       "justification": "structural equality of every immutable machine/controller/monitor field",
                       "merged_states": merged},
                      {"kind": "identity-vector-interning", "verified": True,
                       "justification": "immutable equal identity vectors/chunks; copy-on-write preserves physical indices and binding worlds",
                       "unique_vectors": len(vectors), "interned_vectors": interned_vectors,
                       "unique_chunks": len(chunks), "interned_chunks": interned_chunks})
        if persistent.applications:
            reductions += (persistent.record(),)
        if frontier.applications:
            reductions += (frontier.record(),)
        return CheckResult(status, violations, len(states), reductions, provenance)

    if max_states == 0:
        return result("INCOMPLETE", "explored-state cap exhausted")
    initial = intern(model.initial())
    states.append(initial)
    index[initial] = 0
    pending = deque([0])
    edges: dict[int, tuple[tuple[int, str], ...]] = {}
    enabled: dict[int, frozenset[str]] = {}
    terminals = set()
    deadlocks = []
    while pending:
        vertex = pending.popleft()
        state = states[vertex]
        if state.errors:
            return result("UNSAFE", "; ".join(state.errors), vertex)
        if model.terminal(state):
            outputs = model.deliveries(state)
            if outputs != model.world_demands[state.world]:
                return result("UNSAFE", "terminal delivery differs from the frozen result relation", vertex)
            for monitor_index, monitor in enumerate(model.monitors):
                if monitor.kind == "terminal" and monitor.predicate is not None and not monitor.predicate(state):
                    return result("UNSAFE", f"terminal obligation {monitor.name!r} failed", vertex)
                if monitor.kind == "response" and state.monitor_states[monitor_index] not in monitor.accepting:
                    deadlocks.append((vertex, f"pending response obligation {monitor.name!r} at return"))
            terminals.add(vertex)
            record = {"bindings": dict(model.worlds[state.world]), "outputs": [list(output) for output in outputs]}
            record.update(model.view_delivery(state))
            if record not in provenance["terminal_deliveries"]:
                provenance["terminal_deliveries"].append(record)
            edges[vertex] = ()
            enabled[vertex] = frozenset()
            continue
        transitions = model.enabled(state)
        enabled[vertex] = frozenset(transition.label for transition in transitions)
        if not transitions:
            deadlocks.append((vertex, "finite maximal execution stops before all sites return"))
        transitions, selected_successor = frontier.choose(state, transitions)
        if selected_successor is None:
            transitions, selected_successor = persistent.choose(state, transitions)
        outgoing = []
        for transition in transitions:
            try:
                successor = intern(selected_successor if selected_successor is not None else model._apply_enabled(state, transition))
            except ModelError as error:
                return result("INCOMPLETE", "; ".join(error.issues), vertex)
            target = index.get(successor)
            if target is None:
                if len(states) >= max_states:
                    return result("INCOMPLETE", "explored-state cap exhausted", vertex)
                target = len(states)
                states.append(successor)
                index[successor] = target
                predecessors[target] = (vertex, transition)
                pending.append(target)
            else:
                merged += 1
            outgoing.append((target, transition.label))
        edges[vertex] = tuple(outgoing)
    if deadlocks:
        vertex, message = deadlocks[0]
        return result("NONPROGRESS", message, vertex)
    nonterminal = set(range(len(states))) - terminals
    fair = _fair_component(nonterminal, edges, enabled, model.monitors)
    if fair is not None:
        component, witness = fair
        return result("NONPROGRESS", "reachable fair infinite execution does not return all sites", min(component), witness)
    if not terminals:
        return result("INCOMPLETE", "no maximal assumption-consistent execution establishes non-vacuity")
    # Termination is required by the import; therefore any infinite run that could
    # violate a response condition has already been rejected above.  Finite
    # response states are checked explicitly at each terminal.
    provenance["complete"] = True
    provenance["reachable_terminals"] = len(terminals)
    provenance["terminal_deliveries"].sort(key=lambda record: tuple(sorted(record["bindings"].items())))
    return result("PASS")


def _limits(budget):
    if type(budget) is int:
        max_states, max_operations = budget, None
    elif isinstance(budget, Mapping):
        max_states = budget.get("max_states", 100000)
        max_operations = budget.get("max_operations")
    else:
        raise ValueError("checker budget must be a state cap or budget mapping")
    if type(max_states) is not int or max_states < 0 or max_operations is not None and (type(max_operations) is not int or max_operations < 0):
        raise ValueError("checker bounds must be nonnegative integers")
    return max_states, max_operations


def _independent_warps(instance: Instance, program: ir.Program, max_operations):
    """Verify disjoint private-resource envelopes before projecting the product."""
    if instance.demand_kind == "memory_view" or not instance.ready_at_entry or max_operations is not None:
        return ()
    groups = {}
    for actor, (_, participant) in enumerate(instance.actors):
        groups.setdefault(participant // instance.warp_size, []).append(actor)
    if len(groups) < 2:
        return ()
    if any(monitor.kind != "justice" or monitor.step is not _stay or monitor.labels != frozenset({"owner.ready"})
           for monitor in instance.monitors):
        return ()
    if any(resource.name in instance.derived.storage and resource.domain == "metadata" and resource.scope == "cta"
           for resource in instance.resources):
        return ()
    bound = instance.with_program(program)
    if bound.program_errors or bound._invalid_returns:
        return ()
    private_names = set()
    for graph in program.graphs.values():
        for node in graph.nodes:
            if not isinstance(node, ir.Issue):
                continue
            if node.primitive not in {"tmem_packet_receipt", "reg_copy", "shfl_xor", "observe_ready"}:
                return ()
            try:
                if node.primitive in {"reg_copy", "shfl_xor"}:
                    for operand in (node.operands["src"], node.operands["dst"]):
                        if instance._space(operand) != "reg":
                            return ()
                        if operand["resource"] != "@export:reg":
                            private_names.add(operand["resource"])
                elif node.primitive == "tmem_packet_receipt" and node.operands.get("phase", "issue") == "issue":
                    private_names.add(node.operands["dest"])
            except (ValueError, KeyError, TypeError):
                return ()
    physical_names = set()
    for name in private_names:
        try:
            resource = instance._resource(name)
        except ValueError:
            return ()
        physical = resource.location.handle or resource.name
        if resource.domain != "reg" or resource.scope != "thread" or physical.startswith("@"):
            return ()
        physical_names.add(physical)
    # Every legal destination of these opcodes stays in its issuing physical
    # thread/warp, regardless of the evaluated private index.  Enumerate the
    # complete declared envelopes; no metadata-dependent index is guessed.
    envelopes = {warp: set() for warp in groups}
    for index, cell in enumerate(instance.cells):
        if cell.resource in physical_names:
            warp = cell.participant // instance.warp_size
            if warp not in envelopes:
                return ()
            envelopes[warp].add(index)
    occupied = set()
    for envelope in envelopes.values():
        if occupied & envelope:
            return ()
        occupied.update(envelope)
    projections = []
    for warp, members in sorted(groups.items()):
        actors = tuple(instance.actors[actor] for actor in members)
        physical = {participant for _, participant in actors}
        participants = {site: tuple(participant for actor_site, participant in actors if actor_site == site)
                        for site in sorted({site for site, _ in actors})}
        world_demands = tuple(tuple(row for row in demands if row[0] in physical) for demands in instance.world_demands)
        demands = tuple(sorted(set().union(*map(set, world_demands))))
        projected = replace(
            instance, actors=actors, participants=participants,
            sites=tuple(site for site in instance.sites if site.name in participants),
            demands=demands, world_demands=world_demands, program_ir=None,
            _results=tuple(tuple(rows[actor] for actor in members) for rows in instance._results),
            _groups=tuple(tuple(other for other, (other_site, participant) in enumerate(actors)
                                if other_site == site and participant // instance.warp_size == member // instance.warp_size)
                          for site, member in actors),
        )
        graphs = {
            site: ir.ControlGraph(tuple(ir.Return(tuple(row for row in node.results if row[0] in physical))
                                        if isinstance(node, ir.Return) else node for node in program.graphs[site].nodes),
                                  program.graphs[site].entry)
            for site in participants
        }
        projections.append((tuple(members), projected, projected.program(graphs), len(envelopes[warp])))
    return tuple(projections)


def check(instance: Instance, program: ir.Program, *, budget) -> CheckResult:
    max_states, max_operations = _limits(budget)
    record, deliveries = _certify_static(instance, program, max_states=max_states, max_operations=max_operations)
    if deliveries is not None:
        provenance = {
            "algorithm": "verified-static-phase-v1", "model_digest": instance.model_digest,
            "ir_digest": ir.digest(program), "budget": {"max_states": max_states, "max_operations": max_operations},
            "terminal_deliveries": deliveries, "complete": True, "grounded_worlds": len(deliveries),
            "accessed_bytes": record["accessed_bytes"],
            "safety": "all interleavings by verified static-phase premises, including pre-ready polling",
            "assumptions": [monitor.name for monitor in instance.monitors],
        }
        return CheckResult("PASS", (), 0, (record,), provenance)
    result = _check_interleavings(instance, program, budget=budget)
    return replace(result, provenance={**result.provenance, "static_phase": record})


def _check_interleavings(instance: Instance, program: ir.Program, *, budget) -> CheckResult:
    max_states, max_operations = _limits(budget)
    factors = _independent_warps(instance, program, max_operations) if max_states else ()
    if not factors:
        return _check_product(instance, program, budget=budget)
    provenance = {"algorithm": "checked-independent-warp-product-v1", "model_digest": instance.model_digest,
                  "ir_digest": ir.digest(program), "budget": {"max_states": max_states, "max_operations": None},
                  "terminal_deliveries": [], "factors": [], "accessed_bytes": {}}
    reductions = [{
        "kind": "checked-independent-warp-product", "verified": True, "factors": len(factors),
        "premises": ["immutable pre-call-ready source", "private thread/warp effects only",
                     "no CTA/shared effects or CTA-scoped metadata", "no global operation bound",
                     "owner-ready and generated warp/actor justice monitors only"],
        "justification": "safety, terminal delivery, and justice of a disjoint product are the conjunction of the checked factors",
        "private_cell_envelopes": [factor[3] for factor in factors],
    }]
    explored, failures, outputs = 0, [], {}
    for number, (members, model, local_program, _) in enumerate(factors):
        local = _check_product(model, local_program, budget={"max_states": max(0, max_states - explored)})
        explored += local.explored_states
        provenance["factors"].append({"participants": [list(instance.actors[actor]) for actor in members],
                                      "status": local.status, "explored_states": local.explored_states,
                                      "model_digest": model.model_digest, "ir_digest": ir.digest(local_program)})
        reductions.extend({**record, "factor": number} for record in local.reductions)
        for name, extent in local.provenance["accessed_bytes"].items():
            provenance["accessed_bytes"][name] = max(provenance["accessed_bytes"].get(name, 0), extent)
        if local.status != "PASS":
            violations = []
            for violation in local.violations:
                detail = dict(violation.detail)
                detail["factor_participants"] = [list(instance.actors[actor]) for actor in members]
                if "trace" in detail:
                    detail["trace"] = [{**step, "actors": [members[actor] for actor in step["actors"]]} for step in detail["trace"]]
                violations.append(replace(violation, detail=detail))
            failures.append((local.status, tuple(violations)))
            if local.status in {"UNSAFE", "INCOMPLETE"}:
                return CheckResult(local.status, tuple(violations), explored, tuple(reductions), provenance)
        for delivery in local.provenance["terminal_deliveries"]:
            key = tuple(sorted(delivery["bindings"].items()))
            outputs.setdefault(key, []).extend(delivery["outputs"])
    if failures:
        return CheckResult(failures[0][0], failures[0][1], explored, tuple(reductions), provenance)
    provenance["terminal_deliveries"] = [{"bindings": dict(bindings), "outputs": sorted(rows)}
                                        for bindings, rows in sorted(outputs.items())]
    provenance["complete"] = True
    return CheckResult("PASS", (), explored, tuple(reductions), provenance)
