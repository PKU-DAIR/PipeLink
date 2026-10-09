"""Construct transport from grounded ownership and fit one static-phase program.

The constructor proposes; the independent checker alone grants MODEL_SAT. No
candidate action vocabulary, per-cell solver, or verification oracle is used.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import check, expr, fit, ir, machine, staging
from .synth import SynthResult

Constructed = SynthResult


class _Reject(Exception):
    def __init__(self, reason, **details):
        self.record = {"reason": reason, **details}


def _runs(values):
    values = sorted(values)
    if not values:
        return []
    runs = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous - start + 1))
            start = value
        previous = value
    runs.append((start, previous - start + 1))
    return runs


class _Builder:
    def __init__(self, instance, families):
        self.instance, self.families = instance, families
        self.bindings = tuple(name for name, _ in instance.worlds[0])
        self.worlds = tuple(dict(world) for world in instance.worlds)
        self.domains = instance._binding_domains()
        self.fitted = []
        self.sources = []
        self.staging_layout = None
        self.demands = []
        self.resources = {r.name: r for r in instance.resources if r.name in instance.derived.storage}
        self.view_binding = None
        self.view_addresses = []
        if instance.demand_kind == "memory_view":
            self.view_addresses = [{identity: address for _, address, identity in demands}
                                   for demands in instance.world_demands]
        self.packet_for_cell = {}
        for packet_index, packet in enumerate(instance.packets):
            for lane, slot, cell in packet.receipt:
                self.packet_for_cell[cell] = (packet_index, lane, slot)
        for initial, demands in zip(instance.initial_cells, instance.world_demands):
            sources = {}
            for cell, identity in enumerate(initial):
                if identity != machine.EMPTY:
                    sources.setdefault(identity, []).append(cell)
            self.sources.append(sources)
            self.demands.append({(p, s): identity for p, s, identity in demands})

    def row(self, participant, slot, world):
        return (participant, slot, *(self.worlds[world][name] for name in self.bindings))

    def expression(self, rows, purpose, *, slot=True):
        if not slot:
            rows = {(point[0], *point[2:]): value for point, value in rows.items()}
        names = ("participant", "slot", *self.bindings) if slot else ("participant", *self.bindings)
        inputs = [(name, max(1, max(point[i] for point in rows).bit_length())) for i, name in enumerate(names)]
        failures = []
        fitted = fit.fit_expression(rows, inputs, failures=failures)
        if fitted is None:
            raise _Reject("fit_failed", purpose=purpose, templates=failures)
        return self.fitted_expression(fitted, purpose, len(rows))

    def fitted_expression(self, fitted, purpose, row_count):
        def typed(node):
            if node.op == "bind" and self.domains[node.args[0]].sort != "int":
                return expr.Expr("toint", (replace(node, sort=self.domains[node.args[0]].sort),), "int")
            return replace(node, args=tuple(typed(arg) if isinstance(arg, expr.Expr) else arg for arg in node.args))

        result = typed(fitted)
        self.fitted.append({"purpose": purpose, "rows": row_count, "expression": result.to_json_obj()})
        return result

    def participant_guard(self, owners, participants, purpose):
        coordinate = expr.Expr("coord", ("participant",), "int")
        constant = lambda value: expr.Expr("const", (value,), "int")
        if tuple(p for p in participants if min(owners) <= p <= max(owners)) == owners:
            lower = expr.Expr("le", (constant(min(owners)), coordinate), "pred")
            upper = expr.Expr("le", (coordinate, constant(max(owners))), "pred")
            predicate = upper if min(owners) == min(participants) else lower if max(owners) == max(participants) else expr.Expr("pand", (lower, upper), "pred")
        else:
            failures = []
            fitted = fit.fit_expression({(p,): int(p in owners) for p in participants},
                                        [("participant", max(1, max(participants).bit_length()))], failures=failures)
            if fitted is None:
                raise _Reject("fit_failed", purpose=purpose, templates=failures)
            predicate = expr.Expr("eq", (fitted, constant(1)), "pred")
        self.fitted.append({"purpose": purpose, "rows": len(participants), "expression": predicate.to_json_obj()})
        return predicate

    def allowed(self, family, site):
        return family in self.families and family in self.instance.derived.permitted[site.name]

    def issue(self, site, operations, family, **operands):
        if not self.allowed(family, site):
            raise _Reject("family_unavailable", family=family, site=site.name)
        operations.append(ir.Issue(family, operands, 0))

    def receipt(self, site, operations, rows, destination, offset):
        packet = self.expression(rows, f"receipt:{destination}:{offset}", slot=False)
        self.issue(site, operations, "tmem_packet_receipt", packet=packet, dest=destination,
                   offset=offset, phase="issue", handle="receipt")
        self.issue(site, operations, "tmem_packet_receipt", phase="wait", handle="receipt")

    def selected_sources(self, participants, *, warp=False):
        instance = self.instance
        tables = []
        tmem_rows = instance.tmem_lanes
        for world, demands in enumerate(self.demands):
            selected = {}
            for (participant, slot), identity in demands.items():
                if participant not in participants:
                    continue
                candidates = []
                for index in self.sources[world].get(identity, ()):
                    cell = instance.cells[index]
                    if cell.resource == "@export:smem":
                        candidates.append(index)
                    elif cell.resource == "@export:reg":
                        if (cell.participant // instance.warp_size == participant // instance.warp_size
                                if warp else cell.participant == participant):
                            candidates.append(index)
                    elif cell.resource == "@tmem" and index in self.packet_for_cell:
                        owner = participant % tmem_rows
                        if (cell.participant // instance.warp_size == owner // instance.warp_size
                                if warp else cell.participant == owner):
                            candidates.append(index)
                if not candidates:
                    raise _Reject("ownership_not_local", participant=participant, slot=slot, world=world, warp=warp)
                selected[participant, slot] = min(candidates)
            tables.append(selected)
        return tables

    def demand_runs(self, participants):
        shape = None
        for demands in self.demands:
            for participant in participants:
                positions = tuple(sorted(slot for p, slot in demands if p == participant))
                if shape is None:
                    shape = positions
                elif shape != positions:
                    raise _Reject("nonuniform_demand_positions", participant=participant)
        return _runs(shape or ())

    def _selected(self, site, *, warp):
        instance = self.instance
        participants = instance.participants[site.name]
        selected = self.selected_sources(set(participants), warp=warp)
        kinds = {instance.cells[index].resource for table in selected for index in table.values()}
        if len(kinds) != 1:
            raise _Reject("mixed_source_spaces")
        return participants, selected, next(iter(kinds))

    def bring_to_result(self, site):
        """Bring edge straight into the result: receipt / load / same-thread copy."""
        instance = self.instance
        participants, selected, kind = self._selected(site, warp=False)
        operations = []
        runs = self.demand_runs(participants)
        if kind == "@export:smem":
            for start, width in runs:
                rows = {self.row(p, s, w): instance.cells[table[p, start + s]].index
                        for w, table in enumerate(selected) for p in participants for s in range(width)}
                self.issue(site, operations, "smem_load_round", src=kind, dst=instance.result_resource,
                           width=width, offset=start, address=self.expression(rows, f"source_load:{start}"))
            return operations
        if kind == "@export:reg":
            for start, width in runs:
                offsets = {instance.cells[table[p, start + s]].index - s
                           for table in selected for p in participants for s in range(width)}
                if len(offsets) == 1:
                    self.issue(site, operations, "reg_copy_round", src=kind, dst=instance.result_resource,
                               width=width, offset=start, src_offset=next(iter(offsets)))
                else:
                    for slot in range(start, start + width):
                        rows = {self.row(p, 0, w): instance.cells[table[p, slot]].index
                                for w, table in enumerate(selected) for p in participants}
                        self.issue(site, operations, "reg_copy", src={"resource": kind, "index": self.expression(rows, f"copy:{slot}", slot=False)},
                                   dst={"resource": instance.result_resource, "index": slot})
            return operations
        if kind == "@tmem":
            for start, width in runs:
                if width % instance.packet_columns:
                    raise _Reject("partial_receipt_run", start=start, width=width)
                for offset in range(start, start + width, instance.packet_columns):
                    rows = {}
                    for w, table in enumerate(selected):
                        for p in participants:
                            packets = {self.packet_for_cell[table[p, offset + s]][0]
                                       for s in range(instance.packet_columns)}
                            if len(packets) != 1 or any(self.packet_for_cell[table[p, offset + s]][2] != s
                                                      for s in range(instance.packet_columns)):
                                raise _Reject("noncontiguous_receipt", participant=p, offset=offset)
                            rows[self.row(p, 0, w)] = next(iter(packets))
                    self.receipt(site, operations, rows, instance.result_resource, offset)
            return operations
        raise _Reject("unknown_source_space", kind=kind)

    def permute(self, site):
        """Shuffle adornment: per-slot cross-lane moves from an immutable source bank."""
        instance = self.instance
        participants, selected, kind = self._selected(site, warp=True)
        operations = []
        runs = self.demand_runs(participants)
        # One immutable source bank avoids in-place lane permutations. TMEM is
        # first received into a declared temporary bank, then shuffled from it.
        if kind == "@tmem":
            bank, receipts, bank_cells = self.source_bank(site)
            operations.extend(receipts)
            lookup = [{identity: (p, slot) for (p, slot), identity in table.items()} for table in bank_cells]
        else:
            bank = kind
            lookup = [{identity: (instance.cells[index].participant, instance.cells[index].index)
                       for identity, indices in sources.items() for index in indices
                       if instance.cells[index].resource == kind} for sources in self.sources]
        for start, width in runs:
            for slot in range(start, start + width):
                masks, rows = set(), {}
                for w, demands in enumerate(self.demands):
                    for p in participants:
                        owner, source_slot = lookup[w][demands[p, slot]]
                        masks.add((p % instance.warp_size) ^ (owner % instance.warp_size))
                        rows[self.row(owner, 0, w)] = source_slot
                if len(masks) != 1:
                    raise _Reject("non_xor_lane_permutation", slot=slot)
                mask = next(iter(masks))
                index = self.expression(rows, f"shuffle_source:{slot}", slot=False)
                self.issue(site, operations, "shfl_xor" if mask else "reg_copy",
                           src={"resource": bank, "index": index},
                           dst={"resource": instance.result_resource, "index": slot},
                           **({"mask": mask} if mask else {}))
        return operations

    def source_bank(self, site):
        instance = self.instance
        participants = instance.participants[site.name]
        if "@export:reg" in instance.exported_storage():
            tables = []
            for initial in instance.initial_cells:
                tables.append({(cell.participant, cell.index): initial[index]
                               for index, cell in enumerate(instance.cells)
                               if cell.resource == "@export:reg" and cell.participant in participants and initial[index] != machine.EMPTY})
            return "@export:reg", [], tables
        if "@export:smem" in instance.exported_storage():
            owner_count = max(count for count in range(1, min(len(instance.identities), len(participants)) + 1)
                              if len(instance.identities) % count == 0)
            owners = participants[:owner_count]
            width = len(instance.identities) // owner_count
            registers = sorted((r for r in self.resources.values() if r.domain == "reg" and r.extent >= width
                                and r.name not in instance.derived.readonly_storage),
                               key=lambda r: (r.extent, r.name))
            if not registers:
                raise _Reject("private_storage_too_small", required=width)
            bank = registers[0].name
            tables, rows = [], {}
            for world, sources in enumerate(self.sources):
                table = {}
                for rank, participant in enumerate(owners):
                    for slot in range(width):
                        identity = rank * width + slot
                        index = next((index for index in sources.get(identity, ())
                                      if instance.cells[index].resource == "@export:smem"), None)
                        if index is None:
                            raise _Reject("shared_source_identity_missing", identity=identity, world=world)
                        table[participant, slot] = identity
                        rows[self.row(participant, slot, world)] = instance.cells[index].index
                tables.append(table)
            operations = []
            if owners != participants:
                operations.extend((ir.Meta("source_load_active", self.participant_guard(owners, participants, "source_load_guard"), 0),
                                   ir.Branch("source_load_active", 2, 3)))
            self.issue(site, operations, "smem_load_round", src="@export:smem", dst=bank, width=width,
                       offset=0, address=self.expression(rows, "shared_source_partition"))
            return bank, operations, tables
        tmem_rows = instance.tmem_lanes
        packets = []
        width = None
        for world, initial in enumerate(instance.initial_cells):
            table = {}
            for p in participants:
                eligible = tuple(index for index, packet in enumerate(instance.packets)
                                 if packet.dp == (p % tmem_rows) // instance.warp_size * instance.warp_size
                                 and all(initial[cell] != machine.EMPTY for _, _, cell in packet.receipt))
                if width is None:
                    width = len(eligible) * instance.packet_columns
                if not eligible or len(eligible) * instance.packet_columns != width:
                    raise _Reject("nonuniform_source_receipts", participant=p, world=world)
                table[p] = eligible
            packets.append(table)
        registers = sorted((r for r in self.resources.values() if r.domain == "reg" and r.extent >= (width or 0)
                            and r.name not in instance.derived.readonly_storage),
                           key=lambda r: (r.name == instance.result_resource, r.extent, r.name))
        if not registers:
            raise _Reject("private_storage_too_small", required=width)
        bank = registers[0].name
        operations = []
        tables = [{} for _ in self.worlds]
        for offset in range(0, width, instance.packet_columns):
            rows = {}
            for w, table in enumerate(packets):
                initial = instance.initial_cells[w]
                for p in participants:
                    packet_index = table[p][offset // instance.packet_columns]
                    rows[self.row(p, 0, w)] = packet_index
                    packet = instance.packets[packet_index]
                    for lane, slot, cell in packet.receipt:
                        if lane == p % instance.warp_size:
                            tables[w][p, offset + slot] = initial[cell]
            self.receipt(site, operations, rows, bank, offset)
        return bank, operations, tables

    def staged(self, site):
        instance = self.instance
        view = instance.demand_kind == "memory_view"
        if view and any(len(addresses) != len(demands) for addresses, demands in zip(self.view_addresses, instance.world_demands)):
            raise _Reject("replicated_view_identity_requires_multiple_stores")
        extent = (max((address for demands in instance.world_demands for _, address, _ in demands), default=-instance.scalar_bytes)
                  + instance.scalar_bytes) if view else len(instance.identities) * instance.scalar_bytes
        scratch = next((r for r in self.resources.values() if r.domain == "smem" and r.extent >= extent
                        and r.name not in instance.derived.readonly_storage), None)
        if scratch is None:
            raise _Reject("scratch_too_small", required=extent,
                          available=max((r.extent for r in self.resources.values() if r.domain == "smem"
                                         and r.name not in instance.derived.readonly_storage), default=0))
        barrier = next((r for r in self.resources.values() if r.domain == "barrier" and r.sync is not None
                        and r.name not in instance.derived.readonly_storage
                        and r.sync.capacity >= 1 and r.sync.generations >= 1), None)
        if barrier is None:
            raise _Reject("rendezvous_unavailable")
        bank, operations, tables = self.source_bank(site)
        if view:
            self.view_binding = scratch.name
            tables = [{cell: identity for cell, identity in table.items() if identity in addresses}
                      for table, addresses in zip(tables, self.view_addresses)]
        participants = instance.participants[site.name]
        owners = tuple(sorted({p for p, _ in tables[0]}))
        if not owners or any({p for p, _ in table} != set(owners) for table in tables[1:]):
            raise _Reject("binding_dependent_source_owners")
        guard = None
        if owners != participants:
            predicate = self.participant_guard(owners, participants, "source_guard")
            operations.append(ir.Meta("source_active", predicate, 0))
            guard = len(operations)
            operations.append(ir.Branch("source_active", guard + 1, 0))
        shape = None
        for world, table in enumerate(tables):
            for p in owners:
                positions = tuple(sorted(s for owner, s in table if owner == p))
                if shape is None:
                    shape = positions
                elif positions != shape:
                    raise _Reject("nonuniform_source_positions", participant=p, world=world)
        store_runs = _runs(shape or ())
        load_runs = [] if view else self.demand_runs(participants)
        address_tables = None
        if not view:
            identity_tables = [
                {self.row(p, s, w): table[p, start + s]
                 for w, table in enumerate(worlds) for p in actors for s in range(width)}
                for worlds, actors, runs in ((tables, owners, store_runs), (self.demands, participants, load_runs))
                for start, width in runs
            ]
            address_tables, fitted_addresses, self.staging_layout = staging.select(identity_tables, instance.scalar_bytes, extent, self.bindings)
            if address_tables is None:
                raise _Reject("fit_failed", purpose="staging_layout")
            address_tables = iter(address_tables)
            fitted_addresses = iter(fitted_addresses)
        for start, width in store_runs:
            rows = ({self.row(p, s, w): self.view_addresses[w][table[p, start + s]]
                     for w, table in enumerate(tables) for p in owners for s in range(width)}
                    if view else next(address_tables))
            purpose = f"{'view' if view else 'staging'}_store:{start}"
            address = (self.expression(rows, purpose) if view else
                       self.fitted_expression(next(fitted_addresses), purpose, len(rows)))
            self.issue(site, operations, "smem_store_round", src=bank, dst=scratch.name, width=width,
                       offset=start, address=address)
        if guard is not None:
            operations[guard] = replace(operations[guard], next_false=len(operations))
        self.issue(site, operations, "cta_rendezvous", resource=barrier.name)
        if view:
            return operations
        for start, width in load_runs:
            rows = next(address_tables)
            self.issue(site, operations, "smem_load_round", src=scratch.name, dst=instance.result_resource,
                       width=width, offset=start,
                       address=self.fitted_expression(next(fitted_addresses), f"staging_load:{start}", len(rows)))
        return operations

    def alias(self):
        instance = self.instance
        if "@export:smem" not in instance.exported_storage():
            raise _Reject("view_alias_requires_shared_export")
        for world, demands in enumerate(instance.world_demands):
            for _, address, identity in demands:
                cell = instance._cell_index.get(machine.Cell("@export:smem", -1, address))
                actual = machine.EMPTY if cell is None else instance.initial_cells[world][cell]
                if actual != identity:
                    raise _Reject("view_alias_mismatch", world=world, address=address, expected=identity, actual=actual)
        self.view_binding = "@export:smem"
        return []

    def build(self, route):
        """Fold the route's edge templates; precedence relay > shfl, load ignores shfl."""
        if len(self.instance.sites) != 1:
            raise _Reject("multiple_sites")
        site = self.instance.sites[0]
        if route.alias:
            operations = self.alias()
        elif route.relay or route.deliver:
            operations = self.staged(site)
        elif route.shfl and route.bring != ("load",):
            operations = self.permute(site)
        else:
            operations = self.bring_to_result(site)
        nodes = [] if self.instance.ready_at_entry else [ir.Branch("ready", 1, 0)]
        prefix = len(nodes)
        nodes.extend(replace(operation, next_true=operation.next_true + prefix, next_false=operation.next_false + prefix)
                     if isinstance(operation, ir.Branch) else replace(operation, next=index + prefix + 1)
                     for index, operation in enumerate(operations))
        nodes.append(ir.Return(self.instance.demands))
        return self.instance.program({site.name: ir.ControlGraph(tuple(nodes))}, view_binding=self.view_binding)


def construct(instance: machine.Instance, route, checked: dict[str, Constructed] | None = None) -> Constructed:
    """Propose the route's program from declared cells/resources, then independently certify.

    ``checked`` memoizes verdicts by program digest: a route whose fold emits a
    program already certified inherits that verdict (route-model.md, tier 0).
    """
    builder = _Builder(instance, route.families)
    provenance: dict[str, Any] = {"generator": "constructive-v1", "mode": "static", "candidates": [],
                                  "fitted_expressions": builder.fitted, "class": route.name, "route": route.to_json_obj()}
    scope = {"families": sorted(route.families), "route": route.to_json_obj(), "model_digest": instance.model_digest,
             "quantification": "one program for every admitted binding world and participant schedule",
             "impossibility_scope": "constructive template rejection is not an impossibility proof"}
    try:
        program = builder.build(route)
    except _Reject as rejection:
        provenance.update(rejection.record)
        return Constructed("UNKNOWN", None, None, scope, provenance)
    if builder.staging_layout is not None:
        provenance["staging_layout"] = builder.staging_layout
    if program.view_binding is not None:
        provenance["view_binding"] = program.view_binding
    digest = ir.digest(program)
    if checked is not None and digest in checked:
        earlier = checked[digest]
        provenance["candidates"] = earlier.provenance["candidates"]
        provenance["deduplicated_from"] = earlier.provenance["class"]
        if earlier.status != "MODEL_SAT":
            provenance["reason"] = earlier.provenance["reason"]
        return Constructed(earlier.status, earlier.program, earlier.resource_bindings, scope, provenance)
    result = check.check(instance, program, budget={"max_states": 100_000})
    provenance["candidates"].append({"ir_digest": digest, "check": result.to_json_obj()})
    if result.status != "PASS":
        provenance["reason"] = "candidate_check_" + result.status.lower()
        outcome = Constructed("UNKNOWN", None, None, scope, provenance)
    else:
        outcome = Constructed("MODEL_SAT", program, dict(program.borrows), scope, provenance)
    if checked is not None:
        checked[digest] = outcome
    return outcome
