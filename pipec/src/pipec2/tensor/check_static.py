"""Verified straight-line transport quotient; failures retain exact exploration.

The mutable identity bank below is only a canonical execution, never a replacement
machine semantics: private operands and receipt admission use the machine's
validators, and shared effects are admitted only after the race/publication proof.
"""
from __future__ import annotations

from dataclasses import replace
from graphlib import CycleError, TopologicalSorter
from typing import Mapping

from . import expr, ir
from .machine import EMPTY, IdentityVector, Instance, State, Transition, _ROUND_SCALARS, _stay


JUSTIFICATION = (
    "under 1–8, every interleaving of the site's participants is observationally equivalent — "
    "receipts and private ops touch only private state; stores are race-free by 4; the single "
    "rendezvous orders all stores before all loads (visible_after=rendezvous); loads read only "
    "stored, immutable-per-phase cells by 5; therefore terminal delivery holds in every execution "
    "iff it holds in the canonical sequential one, which the grounded checks establish."
)
_NAMES = ("shape", "readiness", "receipts", "stores", "loads", "rendezvous", "private-phase", "source-stability")


class _Rejected(Exception):
    def __init__(self, premise: int, detail: str):
        self.premise = premise
        self.detail = detail


def _require(condition, premise, detail):
    if not condition:
        raise _Rejected(premise, detail)


def _inputs(value, bindings, *, slot=False):
    """Reject metadata-dependent expressions even if one execution looks correct."""
    if isinstance(value, expr.Expr):
        value = expr.to_json_obj(value)
    if isinstance(value, Mapping):
        return all(_inputs(child, bindings, slot=slot) for child in value.values())
    if isinstance(value, (list, tuple)):
        if value and value[0] in {"coord", "bind"}:
            return (len(value) == 2 and value[1] in
                    (bindings if value[0] == "bind" else {"participant", *(('slot',) if slot else ())}))
        return all(_inputs(child, bindings, slot=slot) for child in value)
    return True


def _shape(model, program):
    """Ground only static participant branches; retain each actor's phase order."""
    _require(not model.program_errors and not model._invalid_returns, 0,
             "; ".join(model.program_errors) or "return relation differs from the frozen import")
    paths, active = {}, {}
    bindings = set(dict(model.worlds[0]))
    barriers = []
    static_state = None
    for site, graph in program.graphs.items():
        entry, prefix = graph.entry, []
        node = graph.nodes[entry]
        if isinstance(node, ir.Issue) and node.primitive == "observe_ready":
            prefix.append(entry)
            entry, node = node.next, graph.nodes[node.next]
        if isinstance(node, ir.Branch):
            expected = graph.nodes[prefix[0]].operands.get("dest") if prefix else None
            _require(node.observation in ({expected} if prefix else {"ready", model.export.readiness.after})
                     and node.next_false == graph.entry, 1, f"{site}: guard is not a refreshing readiness loop")
            prefix.append(entry)
            entry = node.next_true
        else:
            _require(not prefix and model.ready_at_entry, 1, f"{site}: missing entry readiness guard")
        members = tuple(actor for actor, (name, _) in enumerate(model.actors) if name == site)
        selected, dependencies, categories = {}, {}, {}
        actor_barriers = []
        for actor in members:
            pc, phase, pending, previous = entry, 0, None, None
            visited, rendezvous = set(prefix), []
            while True:
                _require(pc not in visited, 0, f"{site}: non-readiness cycle")
                visited.add(pc)
                selected.setdefault(pc, []).append(actor)
                dependencies.setdefault(pc, set()).update(() if previous is None else (previous,))
                previous = pc
                node = graph.nodes[pc]
                if isinstance(node, ir.Return):
                    _require(pending is None, 2, f"{site}: receipt has no adjacent wait")
                    break
                if isinstance(node, ir.Meta):
                    _require(pending is None, 2, f"{site}: guard separates receipt issue and wait")
                    branch = graph.nodes[node.next]
                    _require((site, node.dest) in model._static_metadata and isinstance(branch, ir.Branch)
                             and branch.observation == node.dest and node.dest not in {"ready", model.export.readiness.after}, 0,
                             f"{site}:{pc}: static guard must be adjacent Meta(participant-only) / Branch")
                    _require(node.next not in visited, 0, f"{site}: cyclic static guard")
                    categories[pc] = "metadata"
                    visited.add(node.next)
                    selected.setdefault(node.next, []).append(actor)
                    dependencies.setdefault(node.next, set()).add(pc)
                    previous = node.next
                    if static_state is None:
                        static_state = replace(model.initial(), world=0, owner_phase=1)
                    value = model._integer(node.expr, static_state, actor)
                    pc = branch.next_true if value else branch.next_false
                    continue
                _require(isinstance(node, ir.Issue), 0, f"{site}:{pc}: second observation or non-static node")
                primitive = node.primitive
                _require(_inputs(node.operands, bindings, slot=primitive in _ROUND_SCALARS), 0,
                         f"{site}:{pc}: operands depend on mutable metadata or non-static coordinates")
                if pending is not None:
                    _require(primitive == "tmem_packet_receipt" and node.operands.get("phase") == "wait"
                             and node.operands.get("handle", "receipt") == pending, 2,
                             f"{site}:{pc}: receipt issue/wait are not adjacent and matching")
                if primitive == "tmem_packet_receipt":
                    _require(phase == 0, 0, f"{site}:{pc}: receipt follows the private phase")
                    if node.operands.get("phase", "issue") == "issue":
                        _require(pending is None, 2, f"{site}:{pc}: overlapping receipt issues")
                        pending = node.operands.get("handle", "receipt")
                    else:
                        _require(pending is not None and node.operands.get("phase") == "wait", 2,
                                 f"{site}:{pc}: unmatched receipt wait")
                        pending = None
                    categories[pc] = "private"
                elif primitive in {"reg_copy", "reg_copy_round", "shfl_xor"} or (
                        primitive == "smem_load_round" and node.operands.get("src") == "@export:smem"):
                    _require(phase <= 1, 0, f"{site}:{pc}: private operation follows stores")
                    phase, categories[pc] = 1, "private"
                elif primitive == "smem_store_round":
                    _require(phase <= 2, 0, f"{site}:{pc}: store follows publication or loads")
                    phase, categories[pc] = 2, "stores"
                elif primitive == "cta_rendezvous":
                    _require(not rendezvous and phase <= 2, 5, f"{site}: second or misplaced rendezvous")
                    phase, categories[pc] = 3, "rendezvous"
                    rendezvous.append(pc)
                elif primitive == "smem_load_round":
                    _require(bool(rendezvous), 5, f"{site}: scratch load precedes rendezvous")
                    phase, categories[pc] = 4, "loads"
                else:
                    raise _Rejected(0, f"{site}:{pc}: primitive {primitive!r} is outside the static shape")
                pc = node.next
            actor_barriers.append(tuple(rendezvous))
        _require(len(set(prefix) | set(selected)) == len(graph.nodes), 0, f"{site}: nodes outside all grounded participant paths")
        try:
            order = tuple(TopologicalSorter(dependencies).static_order())
        except CycleError:
            raise _Rejected(0, f"{site}: static participant paths have no common acyclic phase order")
        partition = {name: tuple(pc for pc in order if categories.get(pc) == name)
                     for name in ("private", "stores", "rendezvous", "loads", "metadata")}
        staged = bool(partition["stores"] or partition["loads"])
        _require(all(bool(rows) == staged for rows in actor_barriers) and
                 len(set(actor_barriers)) == 1, 5,
                 f"{site}: every site participant must execute the same rendezvous once")
        if staged:
            barriers.append(graph.nodes[partition["rendezvous"][0]].operands.get("resource"))
        for pc in partition["private"]:
            primitive = graph.nodes[pc].primitive
            if primitive in {"tmem_packet_receipt", "shfl_xor"}:
                chosen = set(selected[pc])
                _require(all(set(model._members(actor)) <= chosen for actor in chosen),
                         2 if primitive == "tmem_packet_receipt" else 6,
                         f"{site}:{pc}: static guard splits a collective warp")
        paths[site] = {"prefix": prefix, **partition}
        active[site] = {pc: tuple(chosen) for pc, chosen in selected.items()}
    if barriers:
        _require(len(barriers) == len(paths) and len(set(barriers)) == 1, 5,
                 "every instance actor must reach the same declared CTA rendezvous")
    return paths, active


def certify(instance: Instance, program: ir.Program, *, max_states=None, max_operations=None):
    """Return (premise record, terminal evidence or None); never certify on failure."""
    view = instance.demand_kind == "memory_view"
    names = (*_NAMES, "view-results") if view else _NAMES
    premises = [{"name": name, "ok": False, "detail": "not established because another premise failed"}
                for name in names]
    justification = JUSTIFICATION + (
        " For memory views, additional premise 9 establishes the terminal relation on the bound shared resource: alias reads require immutable "
        "published source identities; materialization stores into the declared demanded layout and the common "
        "rendezvous publishes every terminal cell. Identity and publication are checked independently at every address."
        if view else "")
    record = {"kind": "static-phase", "verified": False, "premises": premises, "justification": justification}
    counters = {"receipts": 0, "private": 0, "stores": 0, "loads": 0}
    deliveries = []
    current = 0

    def established(index, detail):
        premises[index].update(ok=True, detail=detail)

    try:
        _require(max_states is None or max_states >= len(instance.worlds) + 1, 0,
                 "state cap cannot cover the initial relation and one terminal witness per binding world")
        _require(max_operations is None, 0, "a global operation bound requires full exploration")
        _require(all(m.kind == "justice" and m.step is _stay and m.labels == frozenset({"owner.ready"})
                     and m.initial in m.states for m in instance.monitors), 0,
                 "additional monitor obligations require full exploration")
        model = instance.with_program(program)
        paths, active = _shape(model, program)
        established(0, "grounded participant-only guards; acyclic per-actor phases and full-warp collective paths")
        _require(not instance.derived.early_reclaim, 7,
            "admission does not establish immutable source stability through completion")
        established(7, "Instance.build source-stability premise; no early-reclaim owner transition")
        groups = {site: tuple(model._members(actor) for actor, (name, _) in enumerate(model.actors)
                              if name == site and model._members(actor)[0] == actor) for site in paths}
        actors = {site: tuple(actor for actor, (name, _) in enumerate(model.actors) if name == site) for site in paths}
        entry = model.initial()
        if not model.ready_at_entry:
            # Fair owner readiness cannot erase an unsafe polling prefix.
            # Observe-ready metadata must fit even before ready-start storage is
            # available, with every actor polling simultaneously.
            current = 1
            polling = replace(entry, world=0)
            for site, path in paths.items():
                for pc in path["prefix"]:
                    for actor in actors[site]:
                        polling = _at(polling, (actor,), pc)
                        polling = model._apply(polling, Transition(site, "static-poll", None, True, (actor,)))
        for world, initial in enumerate(model.initial_cells):
            vector = IdentityVector(initial)
            state = replace(entry, world=world, owner_phase=1, cells=vector, published=vector)
            bank = list(initial)
            expected = dict(model._view_results[world]) if view else dict(row for rows in model._results[world] for row in rows)
            stored = set()
            receipt_capacity = {}
            for site, path in paths.items():
                current = 1
                graph = program.graphs[site]
                for pc in path["prefix"]:
                    for actor in actors[site]:
                        state = _at(state, (actor,), pc)
                        state = model._apply(state, Transition(site, "static-prefix", None, True, (actor,)))
                current = 0
                for pc in path["metadata"]:
                    node = graph.nodes[pc]
                    for actor in active[site][pc]:
                        state = replace(state, metadata=model._metadata(
                            state, actor, node.dest, model._integer(node.expr, state, actor)))
                if path["rendezvous"]:
                    current = 5
                    resource = model._resource(graph.nodes[path["rendezvous"][0]].operands["resource"])
                    _require(resource.domain == "barrier" and resource.name not in model.derived.readonly_storage
                             and resource.sync is not None
                             and resource.sync.capacity >= 1 and resource.sync.generations > 0
                             and model._available(resource, state), 5, "rendezvous requires available synchronization generation capacity")
                    _require("converged" in next(s for s in model.sites if s.name == site).guarantees, 5,
                             f"{site}: rendezvous has no declared convergence obligation")
                    for actor in actors[site]:
                        model._permission(actor, "cta_rendezvous")
            established(1, "ready-at-entry or refreshing published-ready guard before every source read")
            for site, path in paths.items():
                graph = program.graphs[site]
                for pc in path["private"]:
                    node = graph.nodes[pc]
                    chosen = active[site][pc]
                    collective_groups = tuple(group for group in groups[site] if group[0] in chosen)
                    if node.primitive == "tmem_packet_receipt":
                        if node.operands.get("phase", "issue") == "wait":
                            continue
                        current = 2
                        state = _at(state, chosen, pc)
                        for members in collective_groups:
                            state = model._apply(state, Transition(site, "static-receipt", node.primitive, False, members))
                        for pending in state.outstanding:
                            receipt_capacity[pending.site, pending.warp] = pending
                        for pending in state.outstanding:
                            # Issue already checked full-warp ownership and every
                            # destination bound. P8 makes deferred completion a
                            # read of these immutable initial identities. Update
                            # only touched cells, not a whole vector per packet.
                            for destination, source in zip(pending.destinations, pending.sources):
                                _require(initial[destination] == EMPTY, 2, "receipt destination aliases immutable source storage")
                                _require(bank[source] != EMPTY and bank[source] == initial[source], 2,
                                         "deferred receipt source identity is unavailable or unstable")
                                bank[destination] = bank[source]
                                counters["receipts"] += 1
                        state = replace(state, outstanding=tuple(replace(item, completed=True) for item in state.outstanding))
                        for members in collective_groups:
                            state = model._apply(state, Transition(site, "static-wait", node.primitive, False, members))
                    elif node.primitive == "shfl_xor":
                        current = 6
                        for members in collective_groups:
                            _shuffle(model, state, node, members, bank)
                            counters["private"] += len(members)
                    else:
                        current = 4 if node.primitive == "smem_load_round" else 6
                        _round(model, state, node, chosen, bank, expected, stored, counters)
            current = 2
            # Any interleaving can retain one handle per receipt-bearing warp.
            # Check their simultaneous high-water mark, not just the serial run.
            model._check_metadata_capacity(state, state.metadata, tuple(receipt_capacity.values()))
            for phase in ("stores", "loads"):
                for site, path in paths.items():
                    for pc in path[phase]:
                        node = program.graphs[site].nodes[pc]
                        current = 3 if phase == "stores" else 4
                        _round(model, state, node, active[site][pc], bank, expected, stored, counters)
                if view and phase == "stores":
                    # Replay the actual publication transition on the computed
                    # bank, not an expected-output vector or a publication flag.
                    state = replace(state, cells=IdentityVector(bank))
                    for site, path in paths.items():
                        for pc in path["rendezvous"]:
                            state = _at(state, actors[site], pc)
                            for actor in actors[site]:
                                state = model._apply(state, Transition(site, "static-arrival", "cta_rendezvous", False, (actor,)))
                    releases = [transition for transition in model.enabled(state) if transition.label.startswith("rendezvous:")]
                    for transition in releases:
                        state = model.apply(state, transition)
            current = 8 if view else 6
            _require(all(bank[cell] == identity for cell, identity in expected.items()), current,
                     "canonical terminal cells differ from the frozen demanded identities")
            state = replace(state, cells=IdentityVector(bank)) if view else state
            if view:
                _require(state.owner_phase == 1, 8, "view is not ready at return")
                if program.view_binding == "@export:smem":
                    _require(bool({"smem_load_scalar", "smem_load_round"} & set(model.export.read_capabilities)), 8,
                             "export does not permit shared view reads")
                    _require(all(initial[cell] == identity for cell, identity in expected.items()), 8,
                             "alias view does not refer to immutable initial source identities")
                else:
                    _require(set(expected) <= stored, 8, "materialized view contains a cell not stored by the program")
                _require(all(state.published[cell] == identity for cell, identity in expected.items()), 8,
                         "terminal shared view is not published")
                outputs = [list(row) for row in model.deliveries(state)]
            else:
                outputs = [[participant, position, bank[model._cell_index[model._declared_cell(model.result_resource, participant, position)]]]
                           for participant, position, _ in model.world_demands[world]]
            deliveries.append({"bindings": dict(model.worlds[world]), "outputs": outputs, **model.view_delivery(state)})
        established(2, {"validated_receipt_cells": counters["receipts"], "validators": "machine issue/wait; immutable completion identities; simultaneous scoped handle capacity"})
        established(3, {"checked_cells": counters["stores"], "addresses": "in-bounds, globally distinct physical cells, declared demanded view layout" if view else
                       "in-bounds, globally distinct physical cells, scalar_bytes * identity_index"})
        established(4, {"checked_cells": counters["loads"], "identities": (
            "initialized stored-or-immutable intermediate sources; final shared view checked by premise 9" if view else
            "stored or immutable exported source; demanded destination identity")})
        established(5, {"participants": len(model.actors), "sites": len(paths), "capacity_unit": "outstanding collective generations",
                        "ordering": "one common rendezvous after every store and before every scratch load, or no scratch effects"})
        established(6, {"checked_cells": counters["private"], "identities": "initialized private sources, snapshot shuffle, sequential round slots, exact terminal relation"})
        if view:
            established(8, {"resource": program.view_binding, "terminal": "actual identity and published identity at every demanded address in every binding world"})
        record["verified"] = True
        record["accessed_bytes"] = model.accessed_bytes
        deliveries.sort(key=lambda row: tuple(sorted(row["bindings"].items())))
        return record, deliveries
    except _Rejected as error:
        premises[error.premise].update(ok=False, detail=error.detail)
    except (ValueError, KeyError, IndexError, TypeError, expr.ExprError) as error:
        premises[current].update(ok=False, detail=str(error))
    return record, None


def _at(state: State, actors, pc):
    positions = list(state.positions)
    for actor in actors:
        positions[actor] = pc
    return replace(state, positions=tuple(positions))


def _shuffle(model, state, node, members, bank):
    _require(len(members) == model.warp_size and
             {model.actors[actor][1] % model.warp_size for actor in members} == set(range(model.warp_size)),
             6, "shuffle requires every physical warp lane")
    masks = set()
    reads, destinations = {}, {}
    for actor in members:
        model._permission(actor, node.primitive)
        masks.add(model._integer(node.operands["mask"], state, actor))
        _require(model._space(node.operands["src"]) == model._space(node.operands["dst"]) == "reg", 6,
                 "shuffle operands must be private registers")
        source = model._cell(node.operands["src"], state, actor)
        destination = model._cell(node.operands["dst"], state, actor, write=True)
        _require(bank[source] != EMPTY, 6, "shuffle source has no initialized identity")
        participant = model.actors[actor][1]
        reads[participant], destinations[participant] = bank[source], destination
    _require(len(masks) == 1 and 0 <= next(iter(masks)) < model.warp_size, 6,
             "shuffle xor mask must be warp-uniform and lane-bounded")
    mask = masks.pop()
    for participant, destination in destinations.items():
        bank[destination] = reads[participant ^ mask]


def _round(model, state, node, actors, bank, expected, stored, counters):
    family = _ROUND_SCALARS.get(node.primitive, node.primitive)
    spaces = {"reg_copy": ("reg", "reg"), "smem_store_scalar": ("reg", "smem"), "smem_load_scalar": ("smem", "reg")}
    premise = {"reg_copy": 6, "smem_store_scalar": 3, "smem_load_scalar": 4}[family]
    width = 1 if node.primitive == "reg_copy" else node.operands["width"]
    _require(type(width) is int and width > 0, premise, "round width must be a positive scalar count")
    for actor in actors:
        model._permission(actor, node.primitive)
        for slot in range(width):
            operands = node.operands if node.primitive == "reg_copy" else model._round_operands(node, state, actor, slot)
            source = model._cell(operands["src"], state, actor)
            destination = model._cell(operands["dst"], state, actor, write=True)
            _require((model._space(operands["src"]), model._space(operands["dst"])) == spaces[family], premise,
                     "round scalar operand address spaces are incompatible")
            if operands["src"]["resource"].startswith("@export:"):
                capability = "reg_copy" if family in {"reg_copy", "smem_store_scalar"} else family
                _require(capability in model.export.read_capabilities, premise, "source read is not an exported capability")
            if family == "smem_load_scalar":
                _require(operands["src"]["resource"] == "@export:smem" or source in stored, 4,
                         "load address was not stored in the store phase")
            identity = bank[source]
            _require(identity != EMPTY, premise, "round source has no available identity")
            if family == "smem_store_scalar":
                _require(destination not in stored, 3, "duplicate store address in the whole instance")
                if model.demand_kind == "memory_view":
                    _require(expected.get(destination) == identity, 3,
                             "stored identity does not match the declared view address")
                stored.add(destination)
                counters["stores"] += 1
            elif family == "smem_load_scalar":
                if model.demand_kind != "memory_view":
                    _require(expected.get(destination) == identity, 4, "loaded identity differs from the demanded destination identity")
                counters["loads"] += 1
            else:
                counters["private"] += 1
            bank[destination] = identity
