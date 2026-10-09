"""Bounded Z3 controller search; a model is a proposal, never a certificate.

K counts nonterminal controller slots per site; each graph also has a distinguished
Return node. action_domain restricts primitive *families*, never their semantics.
The exact model supplies every operand/observation template. Static mode is a
separate, checked specialization, not an existential approximation to concurrency.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

import z3

from pipec2.canonjson import canonical_bytes, sha256_hex
from pipec2.model import _JsonMixin
from . import check, expr, ir
from .machine import Instance, ModelError


@dataclass(frozen=True)
class Bounds(_JsonMixin):
    K: int
    H: int | None = None
    action_domain: tuple[str, ...] | None = None
    rlimit: int = 10_000_000

    def __post_init__(self) -> None:
        if type(self.K) is not int or self.K < 0:
            raise ValueError("K must be a nonnegative number of nonterminal slots")
        if self.H is not None and (type(self.H) is not int or self.H < 0):
            raise ValueError("H must be a nonnegative operation bound or None")
        if type(self.rlimit) is not int or self.rlimit < 0:
            raise ValueError("rlimit must be a nonnegative integer")
        if self.action_domain is not None:
            object.__setattr__(self, "action_domain", tuple(sorted(set(self.action_domain))))


@dataclass(frozen=True)
class SynthResult(_JsonMixin):
    status: Literal["MODEL_SAT", "BOUNDED_UNSAT", "UNKNOWN"]
    program: ir.Program | None
    resource_bindings: Mapping[str, int | str] | None
    scope_record: dict[str, Any]
    provenance: dict[str, Any]


@dataclass
class _Budget:
    limit: int
    used: int = 0

    def check(self, solver: z3.Solver) -> z3.CheckSatResult:
        remaining = self.limit - self.used
        if remaining <= 0:
            return z3.unknown
        solver.set(rlimit=remaining)
        before = solver.statistics().get_key_value("rlimit count")
        result = solver.check()
        self.used += solver.statistics().get_key_value("rlimit count") - before
        return z3.unknown if self.used > self.limit else result


def _provenance(bounds: Bounds, mode: str) -> dict[str, Any]:
    return {
        "backend": "z3", "version": z3.get_version_string(),
        "random_seed": 0, "threads": 1, "rlimit": bounds.rlimit,
        "rlimit_scope": "sum_of_check_resource_deltas",
        "mode": mode, "objective": None, "candidates": [],
        "exclusions": "exact_assignment_only" if mode == "static" else "exact_assignment_or_checked_safety_prefix",
        "resource_binding_scope": "declared resources selected by operands; published borrows retained; "
                                  "relocatable physical bases are allocated by the ordinary composition solver",
    }


def _scope(bounds: Bounds, domains: Mapping[str, tuple[Any, ...]]) -> dict[str, Any]:
    return {
        "bounds": bounds.to_json_obj(),
        "controller_bound": "K nonterminal slots plus one distinguished Return per site",
        "action_templates": {
            site: [ir.node_to_json_obj(node) for node in nodes]
            for site, nodes in sorted(domains.items())
        },
        "quantification": "one controller for every admitted binding, owner execution and schedule",
        "impossibility_scope": "only the recorded finite controller/action/resource bounds",
    }


def _domains(instance: Instance, bounds: Bounds) -> dict[str, tuple[Any, ...]]:
    allowed = None if bounds.action_domain is None else frozenset(bounds.action_domain)
    return {
        site: tuple(node for node in nodes
                    if not isinstance(node, ir.Issue) or allowed is None or node.primitive in allowed)
        for site, nodes in sorted(instance.action_domain(families=allowed).items())
    }


def _checked(instance: Instance, program: ir.Program, bounds: Bounds):
    budget = {"max_states": max(1, bounds.rlimit)}
    if bounds.H is not None:
        budget["max_operations"] = bounds.H
    return check.check(instance, program, budget=budget)


def _result(status: str, program: ir.Program | None, scope: dict[str, Any],
            provenance: dict[str, Any], budget: _Budget) -> SynthResult:
    provenance["rlimit_used"] = budget.used
    return SynthResult(status, program, None if program is None else dict(program.borrows), scope, provenance)


def synthesize(instance: Instance, bounds: Bounds, mode: str = "controller") -> SynthResult:
    """Enumerate bounded controller assignments and exactly check every proposal.

    Tool errors propagate as tool errors. Solver unknown, exhausted work, and an
    incomplete checker remain UNKNOWN; none block a possibly valid controller.
    """
    if mode == "static":
        return synthesize_static(instance, bounds)
    if mode != "controller":
        raise ValueError(f"unknown tensor synthesis mode {mode!r}")
    try:
        domains = _domains(instance, bounds)
    except ModelError as exc:
        provenance = _provenance(bounds, mode)
        provenance["reason"] = {"grounding_incomplete": list(exc.issues)}
        return _result("UNKNOWN", None, _scope(bounds, {}), provenance, _Budget(bounds.rlimit))
    scope, provenance = _scope(bounds, domains), _provenance(bounds, mode)
    scope["model_digest"] = instance.model_digest
    normalization = instance.controller_normalization()
    branch_free = all(not isinstance(node, ir.Branch) for nodes in domains.values() for node in nodes)
    normalize = normalization["ok"] and branch_free
    provenance["controller_normalization"] = {
        "applied": normalize, "branch_free_domain": branch_free, "model_premise": normalization,
        "justification": "an observation-free returning path cannot repeat a controller state; "
                         "relabel that path to a prefix and discard unreachable rows; "
                         "all action choices, operand choices, and inter-site schedules remain free",
    }
    budget = _Budget(bounds.rlimit)
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)
    solver.set(random_seed=0, threads=1)
    variables: list[z3.ArithRef] = []
    tables: dict[str, list[tuple[z3.ArithRef, z3.ArithRef, z3.ArithRef]]] = {}
    returns = {}
    state_numbers = tuple(z3.IntVal(state, ctx=ctx) for state in range(bounds.K + 1))
    zero, last = state_numbers[0], state_numbers[-1]
    for site_index, (site, templates) in enumerate(domains.items()):
        terminal_indices = [index for index, node in enumerate(templates) if isinstance(node, ir.Return)]
        if len(terminal_indices) != 1:
            raise ValueError("model action domain must contain exactly one fixed Return per site")
        return_index = terminal_indices[0]
        returns[site] = templates[return_index]
        return_number = z3.IntVal(return_index, ctx=ctx)
        action_count = z3.IntVal(len(templates), ctx=ctx)
        branch_numbers = () if branch_free else tuple(
            z3.IntVal(index, ctx=ctx) for index, node in enumerate(templates) if isinstance(node, ir.Branch)
        )
        previous_return = None
        rows = []
        for state in range(bounds.K):
            action, yes, no = (z3.Int(f"s{site_index}_q{state}_{name}", ctx=ctx)
                               for name in ("action", "next", "else"))
            variables.extend((action, yes, no))
            rows.append((action, yes, no))
            solver.add(action >= zero, action < action_count, yes >= zero, yes <= last,
                       no >= zero, no <= last)
            is_return = action == return_number
            # All non-Branch alternatives have the same unused false edge. Keep
            # the action domain intact, but state this once per controller row.
            if branch_numbers:
                solver.add(z3.Or([no == zero, *(action == number for number in branch_numbers)]))
            else:
                solver.add(no == zero)
            if normalize:
                solver.add(yes == z3.If(is_return, zero, state_numbers[state + 1]))
                if previous_return is not None:
                    solver.add(z3.Implies(previous_return, is_return))
            else:
                solver.add(z3.Implies(is_return, yes == zero))
            previous_return = is_return
        tables[site] = rows
    while True:
        outcome = budget.check(solver)
        if outcome == z3.unknown:
            provenance["reason"] = solver.reason_unknown() if budget.used < budget.limit else "rlimit exhausted"
            return _result("UNKNOWN", None, scope, provenance, budget)
        if outcome == z3.unsat:
            return _result("BOUNDED_UNSAT", None, scope, provenance, budget)
        assignment = solver.model()
        values = [assignment.eval(var, model_completion=True).as_long() for var in variables]
        graphs = {}
        for site, rows in tables.items():
            nodes = []
            for action, yes, no in rows:
                node = domains[site][assignment.eval(action).as_long()]
                if isinstance(node, ir.Branch):
                    node = replace(node, next_true=assignment.eval(yes).as_long(),
                                   next_false=assignment.eval(no).as_long())
                elif isinstance(node, (ir.Issue, ir.Meta)):
                    node = replace(node, next=assignment.eval(yes).as_long())
                nodes.append(node)
            graphs[site] = ir.ControlGraph(tuple(nodes) + (returns[site],), 0)
        program = instance.program(graphs)
        checked = _checked(instance, program, bounds)
        provenance["candidates"].append({
            "assignment_digest": sha256_hex(canonical_bytes(values)),
            "ir_digest": ir.digest(program), "check": checked.to_json_obj(),
        })
        if checked.status == "PASS":
            return _result("MODEL_SAT", program, scope, provenance, budget)
        if checked.status == "INCOMPLETE":
            provenance["reason"] = "whole-candidate checker incomplete"
            return _result("UNKNOWN", None, scope, provenance, budget)
        blocked = list(zip(variables, values))
        exclusion: dict[str, Any] = {"kind": "exact_assignment"}
        if checked.status == "UNSAFE" and checked.violations:
            evidence = checked.violations[0].detail
            if evidence.get("counterexample_kind") == "reachable_safety_prefix":
                visited = sorted({tuple(row) for row in evidence["controller_rows"]})
                if all(site in tables and 0 <= state <= bounds.K for site, state in visited):
                    prefix_variables = [var for site, state in visited if state < bounds.K
                                        for var in tables[site][state]]
                    blocked = [(var, assignment.eval(var).as_long()) for var in prefix_variables]
                    exclusion = {
                        "kind": "checked_safety_prefix", "controller_rows": [list(row) for row in visited],
                        "justification": "unchanged visited controller rows reproduce the exact reachable "
                                         "unsafe prefix under the same binding and environment choices; "
                                         "unvisited rows cannot affect that prefix",
                    }
        provenance["candidates"][-1]["exclusion"] = exclusion
        # The empty assignment has one model; blocking it correctly exhausts K=0.
        solver.add(z3.Or([z3.BoolVal(False, ctx=ctx),
                          *(var != value for var, value in blocked)]))


def synthesize_static(instance: Instance, bounds: Bounds) -> SynthResult:
    """Exact q[t,s]/action[t] encoding only after the model proves its premise.

    The model projection records rendezvous-delimited synchronous ownership and
    disjoint footprints. Copies read q[t], not a partially updated q[t+1]. NOPs
    preserve every cell and make the horizon an at-most bound. The decoded IR
    is still checked in the full execution model before acceptance.
    """
    try:
        domains = _domains(instance, bounds)
        region = instance.static_region(families=None if bounds.action_domain is None else frozenset(bounds.action_domain))
    except ModelError as exc:
        provenance = _provenance(bounds, "static")
        provenance["reason"] = {"grounding_incomplete": list(exc.issues)}
        provenance["static_premise_failed"] = True
        return _result("UNKNOWN", None, _scope(bounds, {}), provenance, _Budget(bounds.rlimit))
    scope, provenance = _scope(bounds, domains), _provenance(bounds, "static")
    budget = _Budget(bounds.rlimit)
    scope["model_digest"] = instance.model_digest
    provenance["synchronous_premise"] = {
        key: region[key] for key in ("ok", "prerequisite", "provenance")
    }
    if not region["ok"]:
        provenance["reason"] = region["prerequisite"]
        provenance["static_premise_failed"] = True
        return _result("UNKNOWN", None, scope, provenance, budget)
    if region.get("kind") == "native-synchronous-v1":
        return _synthesize_native_static(instance, bounds, region, domains, scope, provenance, budget)
    allowed = bounds.action_domain
    actions = tuple(action for action in region["actions"]
                    if allowed is None or action["node"].primitive in allowed)
    horizon = bounds.K if bounds.H is None else min(bounds.K, bounds.H)
    scope["static_horizon"] = horizon
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)
    solver.set(random_seed=0, threads=1)
    q = [[z3.Int(f"q_{step}_{cell}", ctx=ctx) for cell in range(len(region["cells"]))]
         for step in range(horizon + 1)]
    choices = [z3.Int(f"action_{step}", ctx=ctx) for step in range(horizon)]
    for cell, identity in enumerate(region["initial"]):
        solver.add(q[0][cell] == identity)
    for step, choice in enumerate(choices):
        solver.add(choice >= 0, choice <= len(actions))  # zero is NOP
        for index, action in enumerate((None, *actions)):
            writes = {} if action is None else dict(zip(action["writes"], action["reads"]))
            constraints = []
            if action is not None:
                constraints.extend(q[step][source] >= 0 for source in action["reads"])
            constraints.extend(q[step + 1][cell] == q[step][writes.get(cell, cell)]
                               for cell in range(len(region["cells"])))
            solver.add(z3.Implies(choice == index,
                                 z3.And([z3.BoolVal(True, ctx=ctx), *constraints])))
    for cell, identity in region["results"]:
        solver.add(q[horizon][cell] == identity)
    while True:
        outcome = budget.check(solver)
        if outcome == z3.unknown:
            provenance["reason"] = "static solver incomplete or rlimit exhausted"
            return _result("UNKNOWN", None, scope, provenance, budget)
        if outcome == z3.unsat:
            return _result("BOUNDED_UNSAT", None, scope, provenance, budget)
        assignment = solver.model()
        values = [assignment.eval(choice).as_long() for choice in choices]
        site = next(iter(domains))
        terminal = next(node for node in domains[site] if isinstance(node, ir.Return))
        selected = [actions[index - 1]["node"] for index in values if index]
        nodes = tuple(replace(node, next=index + 1) for index, node in enumerate(selected)) + (terminal,)
        program = instance.program({site: ir.ControlGraph(nodes, 0)})
        checked = _checked(instance, program, bounds)
        provenance["candidates"].append({
            "assignment_digest": sha256_hex(canonical_bytes(values)),
            "ir_digest": ir.digest(program), "check": checked.to_json_obj(),
        })
        if checked.status == "PASS":
            return _result("MODEL_SAT", program, scope, provenance, budget)
        if checked.status == "INCOMPLETE":
            provenance["reason"] = "whole-candidate checker incomplete"
            return _result("UNKNOWN", None, scope, provenance, budget)
        solver.add(z3.Or([z3.BoolVal(False, ctx=ctx),
                          *(choice != value for choice, value in zip(choices, values))]))


def _runs(points):
    """Exact affine runs, preserving gaps rather than interpolating across them."""
    result = []
    index = 0
    while index < len(points):
        lo, value = points[index]
        end = index
        step = 0
        if index + 1 < len(points) and points[index + 1][0] == lo + 1:
            step = points[index + 1][1] - value
            end += 1
            while end + 1 < len(points) and points[end + 1] == (points[end][0] + 1, points[end][1] + step):
                end += 1
        result.append((lo, points[end][0], value, step))
        index = end + 1
    return result


def _lookup(index, points, *, ctx, default=-1):
    """Compact exact integer lookup, with a two-dimensional regular-run case."""
    runs = _runs(points)
    if not runs:
        return z3.IntVal(default, ctx=ctx)
    if len(runs) > 2:
        width = runs[0][1] - runs[0][0] + 1
        stride = runs[1][0] - runs[0][0]
        slope = runs[0][3]
        if stride > 0 and all(hi - lo + 1 == width and lo == runs[0][0] + row * stride and delta == slope
                              for row, (lo, hi, _value, delta) in enumerate(runs)):
            relative = index - runs[0][0]
            row, column = relative / stride, relative % stride
            base = _lookup(row, [(i, run[2]) for i, run in enumerate(runs)], ctx=ctx, default=default)
            return z3.If(z3.And(relative >= 0, row < len(runs), column < width), base + slope * column, default)
    value = z3.IntVal(default, ctx=ctx)
    for lo, hi, base, stride in reversed(runs):
        condition = index == lo if lo == hi else z3.And(index >= lo, index <= hi)
        value = z3.If(condition, base + stride * (index - lo), value)
    return value


def _member(index, values, ctx):
    runs = _runs([(value, 0) for value in sorted(values)])
    return z3.Or([z3.BoolVal(False, ctx=ctx),
                  *(index == lo if lo == hi else z3.And(index >= lo, index <= hi)
                    for lo, hi, _base, _stride in runs)])


def _expression_tree(value):
    if isinstance(value, expr.Expr):
        return expr.to_json_obj(value)
    return ["const", value] if type(value) is int else value


def _expression_shape(tree):
    if tree[0] == "const":
        return ("const",)
    if tree[0] == "cbits":
        return ("cbits", tree[1])
    return tuple(_expression_shape(value) if isinstance(value, list) else value for value in tree)


def _lift_expression(choice, records, coordinates, bindings, binding_domains, ctx):
    """Compile existing C1 trees, factoring literal differences in one skeleton."""
    first = records[0][1]
    op = first[0]
    if op == "const":
        return _lookup(choice, [(index, tree[1]) for index, tree in records], ctx=ctx, default=0)
    if op == "cbits":
        return z3.Int2BV(_lookup(choice, [(index, tree[2]) for index, tree in records], ctx=ctx, default=0), first[1])
    if op == "coord":
        return coordinates[first[1]]
    if op == "bind":
        domain = binding_domains[first[1]]
        value = bindings[first[1]]
        return z3.IntVal(value, ctx=ctx) if domain.sort == "int" else z3.BitVecVal(value, int(domain.sort[5:-1]), ctx=ctx)

    def child(position):
        return _lift_expression(choice, [(index, tree[position]) for index, tree in records],
                                coordinates, bindings, binding_domains, ctx)

    if op == "add":
        return child(1) + child(2)
    if op == "sub":
        return child(1) - child(2)
    if op == "mulc":
        return first[1] * child(2)
    if op == "divc":
        return child(1) / first[2]
    if op == "modc":
        return child(1) % first[2]
    if op == "band":
        return child(1) & child(2)
    if op == "bor":
        return child(1) | child(2)
    if op == "bxor":
        return child(1) ^ child(2)
    if op == "shlc":
        return child(1) << first[2]
    if op == "shrc":
        return z3.LShR(child(1), first[2])
    if op == "tobits":
        return z3.Int2BV(child(2), first[1])
    if op == "toint":
        return z3.BV2Int(child(1), is_signed=False)
    if op == "eq":
        return child(1) == child(2)
    if op == "lt":
        return child(1) < child(2)
    if op == "le":
        return child(1) <= child(2)
    if op == "not":
        return z3.Not(child(1))
    if op == "pand":
        return z3.And(child(1), child(2))
    if op == "por":
        return z3.Or(child(1), child(2))
    if op == "select":
        return z3.If(child(1), child(2), child(3))
    raise ModelError([f"native static C1 translation unavailable for {op!r}"])


def _apply_smt(function, values, ctx):
    parameters, body = function
    replacements = tuple((parameter, value if isinstance(value, z3.ExprRef) else z3.IntVal(value, ctx=ctx))
                         for parameter, value in zip(parameters, values))
    return z3.simplify(z3.substitute(body, *replacements))


def _smt_key(value):
    return ("expr", value) if isinstance(value, z3.ExprRef) else ("int", value)


class _NativeFields:
    """Lift compact IR operand vocabulary into shared symbolic selector functions."""

    def __init__(self, region, ctx):
        self.region, self.ctx = region, ctx
        self.resources = tuple(sorted(region["resource_cells"]))
        self.resource_ids = {name: index for index, name in enumerate(self.resources)}
        self.handles = tuple(sorted({node.operands.get("handle", "receipt")
                                     for node, kind in zip(region["actions"], region["action_kinds"])
                                     if kind in {"issue", "wait"}}))
        self.barriers = tuple(sorted(region["barriers"]))
        self.records = {}
        self.functions = {}
        self.cell_functions = {}
        self.value_cache = {}
        self.cell_cache = {}
        destinations = {node.operands["dest"] for node, kind in zip(region["actions"], region["action_kinds"]) if kind == "issue"}
        self.contiguous_destinations = all(all(right == left + 1 for left, right in zip(cells, cells[1:]))
                                           for name in destinations for cells in region["resource_cells"][name].values())
        for action, (node, kind) in enumerate(zip(region["actions"], region["action_kinds"]), 1):
            operands = node.operands
            values = {}
            if kind in {"copy", "shfl_xor", "smem_store_scalar", "smem_load_scalar"}:
                for side in ("src", "dst"):
                    values[f"{side}_resource"] = self.resource_ids[operands[side]["resource"]]
                    values[f"{side}_index"] = operands[side]["index"]
                if kind == "shfl_xor":
                    values["mask"] = operands["mask"]
            elif kind in {"reg_copy_round", "smem_store_round", "smem_load_round"}:
                values = {"src_resource": self.resource_ids[operands["src"]],
                          "dst_resource": self.resource_ids[operands["dst"]],
                          "width": operands["width"], "offset": operands.get("offset", 0)}
                if kind == "reg_copy_round":
                    values["src_offset"] = operands.get("src_offset", 0)
                else:
                    values["address"] = operands["address"]
            elif kind == "issue":
                values = {"packet": operands["packet"], "dst_resource": self.resource_ids[operands["dest"]],
                          "offset": operands.get("offset", 0), "handle": self.handles.index(operands.get("handle", "receipt"))}
            elif kind == "wait":
                values = {"handle": self.handles.index(operands.get("handle", "receipt"))}
            elif kind == "rendezvous":
                values = {"barrier": self.barriers.index(operands["resource"])}
            for name, value in values.items():
                self.records.setdefault(name, []).append((action, _expression_tree(value)))
        def contains_binding(tree):
            return tree[0] == "bind" or any(contains_binding(item) for item in tree[1:] if isinstance(item, list))
        self.binding_fields = {name for name, records in self.records.items()
                               if any(contains_binding(tree) for _index, tree in records)}

    def value(self, name, action, participant, slot, world):
        world = world if name in self.binding_fields else 0
        cache_key = (name, world, _smt_key(action), _smt_key(participant), _smt_key(slot))
        if cache_key in self.value_cache:
            return self.value_cache[cache_key]
        key = (name, world)
        if key not in self.functions:
            a, p, s = z3.Ints(f"field_{name}_{world}_a field_{name}_{world}_p field_{name}_{world}_s", ctx=self.ctx)
            groups = {}
            for index, tree in self.records.get(name, ()):
                groups.setdefault(_expression_shape(tree), []).append((index, tree))
            value = z3.IntVal(0, ctx=self.ctx)
            for records in reversed(tuple(groups.values())):
                term = _lift_expression(a, records, {"participant": p, "slot": s},
                                        self.region["worlds"][world], self.region["binding_domains"], self.ctx)
                value = z3.If(_member(a, [index for index, _tree in records], self.ctx), term, value)
            self.functions[key] = ((a, p, s), value)
        result = _apply_smt(self.functions[key], (action, participant, slot), self.ctx)
        self.value_cache[cache_key] = result
        return result

    def cell(self, resource, participant, index):
        cache_key = (_smt_key(resource), _smt_key(participant), _smt_key(index))
        if cache_key in self.cell_cache:
            return self.cell_cache[cache_key]
        if "cells" not in self.cell_functions:
            r, p, i = z3.Ints("resource_cell_r resource_cell_p resource_cell_i", ctx=self.ctx)
            result = z3.IntVal(-1, ctx=self.ctx)
            for name in reversed(self.resources):
                owners = self.region["resource_cells"][name]
                stride = self.region["resource_strides"][name]
                offset = i / stride
                if -1 in owners:
                    row = _lookup(offset, list(enumerate(owners[-1])), ctx=self.ctx)
                else:
                    affine = []
                    for member, cells in sorted(owners.items()):
                        delta = cells[1] - cells[0] if len(cells) > 1 else 0
                        if not cells or any(value != cells[0] + position * delta for position, value in enumerate(cells)):
                            break
                        affine.append((member, cells[0], delta))
                    if len(affine) == len(owners) and len({item[2] for item in affine}) == 1:
                        row = _lookup(p, [(member, base) for member, base, _delta in affine], ctx=self.ctx) + affine[0][2] * offset
                        valid = z3.And(_member(p, owners, self.ctx), offset >= 0,
                                       offset < _lookup(p, [(member, len(cells)) for member, cells in sorted(owners.items())], ctx=self.ctx))
                        row = z3.If(valid, row, -1)
                    else:
                        row = z3.IntVal(-1, ctx=self.ctx)
                        for member, cells in reversed(tuple(sorted(owners.items()))):
                            row = z3.If(p == member, _lookup(offset, list(enumerate(cells)), ctx=self.ctx), row)
                result = z3.If(r == self.resource_ids[name], row, result)
            self.cell_functions["cells"] = ((r, p, i), result)
        result = _apply_smt(self.cell_functions["cells"], (resource, participant, index), self.ctx)
        self.cell_cache[cache_key] = result
        return result


def _map_function(maps, size, width, ctx, name):
    """Exact finite map compilation; recognized formulas are exhaustively verified."""
    action, index = z3.Ints(f"{name}_action {name}_index", ctx=ctx)
    groups = {}
    for key, values in maps.items():
        if all(value == values[0] for value in values):
            form, parameter = "constant", values[0]
        elif all(value == (i + values[0]) % size for i, value in enumerate(values)):
            form, parameter = "rotate", values[0]
        elif values[0] < width and all(value == i // width * width + (i % width + values[0]) % width for i, value in enumerate(values)):
            form, parameter = "column", values[0]
        elif all(value == i ^ values[0] for i, value in enumerate(values)):
            form, parameter = "xor", values[0]
        else:
            form, parameter = values, 0
        groups.setdefault(form, []).append((key, parameter))
    value = z3.IntVal(-1, ctx=ctx)
    for form, entries in reversed(tuple(groups.items())):
        parameter = _lookup(action, entries, ctx=ctx, default=0)
        if form == "constant":
            expression = parameter
        elif form == "rotate":
            expression = (index + parameter) % size
        elif form == "column":
            expression = index / width * width + (index % width + parameter) % width
        elif form == "xor":
            expression = z3.IntVal(-1, ctx=ctx)
            masks = {}
            for key, mask in entries:
                masks.setdefault(mask, []).append(key)
            for mask, keys in masks.items():
                transformed = index
                for bit in range(mask.bit_length()):
                    power = 1 << bit
                    if mask & power:
                        transformed = transformed + power * (1 - 2 * ((index / power) % 2))
                expression = z3.If(_member(action, keys, ctx), transformed, expression)
        else:
            expression = _lookup(index, list(enumerate(form)), ctx=ctx)
        value = z3.If(_member(action, [key for key, _parameter in entries], ctx), expression, value)
    return ((action, index), value)


def _operand_dependencies(value):
    if isinstance(value, expr.Expr):
        value = expr.to_json_obj(value)
    if isinstance(value, Mapping):
        return set().union(*(_operand_dependencies(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], str) and value[0] in {"coord", "bind"}:
            return {(value[0], value[1])}
        return set().union(*(_operand_dependencies(item) for item in value))
    return set()


def _constant_bindings(tree, world, domains):
    if tree[0] == "bind":
        name = tree[1]
        domain = domains[name]
        return ["const", world[name]] if domain.sort == "int" else ["cbits", int(domain.sort[5:-1]), world[name]]
    return [_constant_bindings(value, world, domains) if isinstance(value, list) else value for value in tree]


def _static_world_obligations(region):
    """Prove redundant static source constraints; never merge checker worlds."""
    participants, worlds = region["participants"], region["worlds"]
    warp_size = region["warp_size"]
    by_warp = {}
    for participant in participants:
        by_warp.setdefault(participant // warp_size, []).append(participant)
    initial = [dict(values) for values in region["initial_sparse_worlds"]]
    expressions = {}
    for index, (node, kind) in enumerate(zip(region["actions"], region["action_kinds"]), 1):
        if kind == "issue":
            tree = _expression_tree(node.operands["packet"])
            if any(op == "coord" and name != "participant" for op, name in _operand_dependencies(tree)):
                return (), tuple((index,) for index in range(len(worlds))), {"applied": False, "reason": "packet uses retained local observations"}
            expressions.setdefault(canonical_bytes(tree), (tree, []))[1].append(index)
    subpartition = region.get("packet_subpartition")
    if expressions and subpartition != {
        "rule": "own-issuing-warp-modulo",
        "dp_base": "(floor(participant/warp_size) % subpartitions)*warp_size",
        "subpartitions": region["tmem_lanes"] // warp_size, "dp_extent": warp_size,
    }:
        raise ModelError(["native static TMEM projection lacks the recognized own-issuing-warp subpartition rule"])
    rejected, signatures = [], [[] for _world in worlds]
    for tree, indices in expressions.values():
        values_by_world = []
        valid = True
        for world_index, world in enumerate(worlds):
            parsed = expr.parse(_constant_bindings(tree, world, region["binding_domains"]),
                                {"participant": expr.Domain("participant", "int", values=participants)})
            receipts = []
            for warp, members in sorted(by_warp.items()):
                choices = {expr.evaluate(parsed, {"participant": member}) for member in members}
                if len(choices) != 1 or not 0 <= next(iter(choices)) < len(region["packets"]):
                    valid = False
                    break
                packet = region["packets"][choices.pop()]
                if packet["dp"] != warp % subpartition["subpartitions"] * warp_size:
                    valid = False
                    break
                for lane, slot, source in sorted(packet["receipt"]):
                    identity = initial[world_index].get(source, -1)
                    if source < 0 or identity < 0:
                        valid = False
                        break
                    receipts.append((warp * warp_size + lane, slot, identity))
                if not valid:
                    break
            if not valid:
                break
            values_by_world.append(tuple(receipts))
        if not valid:
            rejected.extend(indices)
        else:
            for signature, values in zip(signatures, values_by_world):
                signature.append(values)
    rejected_rounds, round_cache = [], {}
    for action, node in enumerate(region["actions"], 1):
        if node.primitive != "smem_store_round":
            continue
        width = node.operands["width"]
        tree = _expression_tree(node.operands["address"])
        resource = node.operands["dst"]
        key = (canonical_bytes(tree), width, resource)
        if key not in round_cache:
            domains = {"participant": expr.Domain("participant", "int", values=participants),
                       "slot": expr.Domain("slot", "int", 0, width)}
            dependencies = _operand_dependencies(tree)
            checked_worlds = worlds if any(op == "bind" or name not in {"participant", "slot"} for op, name in dependencies) else worlds[:1]
            safe = True
            cells = region["resource_cells"][resource][-1]
            stride = region["resource_strides"][resource]
            for world in checked_worlds:
                parsed = expr.parse(_constant_bindings(tree, world, region["binding_domains"]), domains)
                owners = {}
                for participant in participants:
                    for slot in range(width):
                        address = expr.evaluate(parsed, {"participant": participant, "slot": slot})
                        if address < 0 or address % stride or address // stride >= len(cells):
                            safe = False
                            break
                        cell = cells[address // stride]
                        if cell in owners and owners[cell] != participant:
                            safe = False
                            break
                        owners[cell] = participant
                    if not safe:
                        break
                if not safe:
                    break
            round_cache[key] = safe
        if not round_cache[key]:
            rejected_rounds.append(action)
    rejected.extend(rejected_rounds)
    kinds = set(region["action_kinds"])
    # Loads from an immutable source do not communicate between private owners;
    # mutable shared relay requires both a writer and a reader.
    relay = any(kind.startswith("smem_store") for kind in kinds) and any(kind.startswith("smem_load") for kind in kinds)
    local_groups = tuple(tuple(members) for _warp, members in sorted(by_warp.items())) if "shfl_xor" in kinds else tuple((p,) for p in participants)
    counting_groups = (participants,) if relay else local_groups
    refined_bound, closed_cut, shared_required = 0, [], False
    for world, vectors in enumerate(signatures):
        per_receipt = []
        for vector in vectors:
            by_participant = {}
            for participant, _slot, identity in vector:
                by_participant.setdefault(participant, set()).add(identity)
            per_receipt.append(by_participant)
        demands_by_participant = {}
        for participant, _position, _cell, identity in region["result_cells_worlds"][world]:
            demands_by_participant.setdefault(participant, set()).add(identity)

        def coverage(members):
            required = set().union(*(demands_by_participant.get(p, set()) for p in members))
            available = {initial[world].get(cell, -1) for p in members
                         for cell in region["resource_cells"].get("@export:reg", {}).get(p, ())}
            if {"smem_load_scalar", "smem_load_round"} & kinds:
                available.update(initial[world].get(cell, -1)
                                 for cell in region["resource_cells"].get("@export:smem", {}).get(-1, ()))
            needed = required - available
            offered = [set().union(*(receipt.get(p, set()) for p in members)) for receipt in per_receipt]
            return needed, offered

        for members in counting_groups:
            needed, offered = coverage(members)
            union = set().union(*offered)
            if needed - union:
                closed_cut.append({"world": world, "participants": list(members), "missing": sorted(needed - union)[:8]})
            maximum = max((len(needed & receipt) for receipt in offered), default=0)
            if needed and maximum:
                refined_bound = max(refined_bound, (len(needed) + maximum - 1) // maximum)
        if relay:
            for members in local_groups:
                needed, offered = coverage(members)
                if needed - set().union(*offered):
                    shared_required = True
    binding_names = set(region["binding_domains"])
    independent = True
    for node in region["actions"]:
        for key, value in node.operands.items():
            if node.primitive == "tmem_packet_receipt" and key == "packet":
                continue
            if any(op == "bind" or name in binding_names for op, name in _operand_dependencies(value)):
                independent = False
    groups = {}
    register_sources = tuple(index for index in region["source_cells"]
                             if region["cell_specs"][index].resource != "@tmem")
    if independent:
        for world in range(len(worlds)):
            signature = (tuple(signatures[world]),
                         tuple((cell, initial[world].get(cell, -1)) for cell in register_sources),
                         tuple(region["result_cells_worlds"][world]))
            groups.setdefault(signature, []).append(world)
        classes = tuple(tuple(group) for group in groups.values())
    else:
        classes = tuple((index,) for index in range(len(worlds)))
    return tuple(sorted(rejected)), classes, {
        "applied": True,
        "rejected_issue_actions": sorted(set(rejected) - set(rejected_rounds)),
        "rejected_round_actions": rejected_rounds,
        "round_rule": "a branch-free full-site store round has no internal rendezvous; "
                      "different participants writing the same physical cell necessarily violate the epoch rule",
        "source_rule": "every non-padding static issue executes and must reach a matching wait; "
                       "its full immutable receipt must be offered, in range, warp-uniform, and in the issuing warp's own subpartition in every world",
        "packet_subpartition": subpartition,
        "world_classes": [list(group) for group in classes],
        "world_rule": "exact per-lane/slot receipt identity, register-source cell/value, and demand cell/identity signatures; "
                      "non-packet operands have no binding dependency; resource maps, protocol and capacities are fixed",
        "signature_digests": [sha256_hex(canonical_bytes(signature)) for signature in signatures],
        "receipt_coverage_lower_bound": refined_bound,
        "closed_private_cut": closed_cut,
        "shared_transfer_required": shared_required,
        "coverage_rule": "exact per-participant receipt identities; count maximum intersection with distinct missing demands "
                         "within each closed private communication group, or the whole site when shared relay is available",
        "checker_worlds_unchanged": True,
    }


class _NativeStaticProblem:
    """Exact branch-free q/action encoding of the model's compact projection."""

    def __init__(self, region, bounds):
        rejected, world_classes, self.obligation_reduction = _static_world_obligations(region)
        rejected_set = set(rejected)
        surviving = tuple(index for index in range(len(region["actions"])) if index + 1 not in rejected_set)
        self.obligation_reduction["surviving_template_indices"] = [index + 1 for index in surviving]
        region = {**region, **{key: tuple(region[key][index] for index in surviving)
                              for key in ("actions", "action_kinds", "action_costs")}}
        self.region, self.bounds = region, bounds
        self.ctx = z3.Context()
        self.solver = z3.Solver(ctx=self.ctx)
        self.solver.set(random_seed=0, threads=1)
        self.fields = _NativeFields(region, self.ctx)
        self.actions = region["actions"]
        self.kinds = {}
        for index, kind in enumerate(region["action_kinds"], 1):
            self.kinds.setdefault(kind, []).append(index)
        self.participants = region["participants"]
        self.warps = sorted({p // region["warp_size"] for p in self.participants})
        self.choices = [z3.Int(f"action_{step}", ctx=self.ctx) for step in range(bounds.K)]
        self.integer = z3.IntSort(ctx=self.ctx)
        self.mask_sort = z3.BitVecSort(len(self.participants), ctx=self.ctx)
        self.mask_zero = z3.BitVecVal(0, len(self.participants), ctx=self.ctx)
        source_cell = z3.Int("source_classification_cell", ctx=self.ctx)
        self.source_predicate = ((source_cell,), _member(source_cell, region["source_cells"], self.ctx))
        self.packet_source = self._packet_sources()
        receipt_bound = 0
        relay = any(kind.startswith("smem_store") for kind in self.kinds) and any(kind.startswith("smem_load") for kind in self.kinds)
        for world, demands in enumerate(region["result_cells_worlds"]):
            initial = dict(region["initial_sparse_worlds"][world])
            groups = (self.participants,) if relay else tuple(
                tuple(p for p in self.participants if p // region["warp_size"] == warp) for warp in self.warps
            )
            for members in groups:
                demanded = {identity for participant, _position, _cell, identity in demands if participant in members}
                available = {initial.get(cell, -1) for participant in members
                             for cell in region["resource_cells"].get("@export:reg", {}).get(participant, ())}
                if {"smem_load_scalar", "smem_load_round"} & set(self.kinds):
                    available.update(initial.get(cell, -1)
                                     for cell in region["resource_cells"].get("@export:smem", {}).get(-1, ()))
                needed = len(demanded - available)
                capacity = len(members) * region["packet_columns"]
                if needed and capacity:
                    receipt_bound = max(receipt_bound, (needed + capacity - 1) // capacity)
        receipt_bound = max(receipt_bound, self.obligation_reduction.get("receipt_coverage_lower_bound", 0))
        if self.obligation_reduction.get("closed_private_cut"):
            self.solver.add(z3.BoolVal(False, ctx=self.ctx))
        self.obligation_reduction["receipt_count_lower_bound"] = receipt_bound
        self.obligation_reduction["receipt_count_rule"] = (
            "distinct demanded identities not initially accessible in exported registers; "
            "one full-site issue supplies at most participants times packet_columns identities; "
            "without shared relay, count separately in each warp; every issued handle must be waited"
        )
        if receipt_bound:
            for kind in ("issue", "wait"):
                self.solver.add(z3.Sum([z3.IntVal(0, ctx=self.ctx),
                                       *(z3.If(self.kind(action, kind), 1, 0) for action in self.choices)]) >= receipt_bound)
        if self.obligation_reduction.get("shared_transfer_required"):
            for kinds in (("smem_store_scalar", "smem_store_round"), ("smem_load_scalar", "smem_load_round"), ("rendezvous",)):
                self.solver.add(z3.Sum([z3.IntVal(0, ctx=self.ctx), *(
                    z3.If(z3.Or([self.kind(action, kind) for kind in kinds]), 1, 0) for action in self.choices
                )]) >= 1)
        self.possible_kinds = self._protocol_kinds(receipt_bound)
        if self.obligation_reduction.get("closed_private_cut") or not self.obligation_reduction["protocol_language_nonempty"]:
            return
        for step, action in enumerate(self.choices):
            self.solver.add(action >= 0, action <= len(self.actions))
            if step:
                self.solver.add(z3.Implies(self.choices[step - 1] == 0, action == 0))
        if bounds.H is not None:
            costs = [(0, 0), *enumerate(region["action_costs"], 1)]
            self.solver.add(z3.Sum([z3.IntVal(region["return_cost"], ctx=self.ctx),
                                   *(_lookup(action, costs, ctx=self.ctx, default=0) for action in self.choices)]) <= bounds.H)
        coupled = any(kind.startswith("smem_store") or kind == "rendezvous"
                      or kind.startswith("smem_load") and node.operands["src"] != "@export:smem"
                      and not (isinstance(node.operands["src"], Mapping) and node.operands["src"]["resource"] == "@export:smem")
                      for node, kind in zip(self.actions, region["action_kinds"]))
        owner = {}
        for name, rows in region["resource_cells"].items():
            for participant, cells in rows.items():
                if participant == -1 or name.startswith("@export:"):
                    continue
                for cell in cells:
                    if cell in owner and owner[cell] != participant:
                        coupled = True
                    owner[cell] = participant
        if coupled:
            actor_groups = (self.participants,)
        elif "shfl_xor" in self.kinds:
            actor_groups = tuple(tuple(p for p in self.participants if p // region["warp_size"] == warp)
                                 for warp in self.warps)
        else:
            actor_groups = tuple((participant,) for participant in self.participants)
        self.obligation_reduction["mutable_state_groups"] = [list(group) for group in actor_groups]
        self.obligation_reduction["mutable_state_rule"] = (
            "no shared communication and exact disjoint private ownership: split only mutable data equations; "
            "shuffle lanes stay together, controller choices and full-site handle/resource constraints stay joint; "
            "indivisible receipts retain one shared issue/wait choice and all-lane source validation"
        )
        receipt_ssa = len(self.fields.handles) == 1 and all(
            kinds <= {"issue", "wait", "nop"} for kinds in self.possible_kinds
        ) and all(
            type(node.operands.get("offset", 0)) is int
            and node.operands.get("offset", 0) >= 0
            and all(node.operands.get("offset", 0) + region["packet_columns"] <= len(region["resource_cells"][node.operands["dest"]].get(p, ()))
                    for p in self.participants)
            for node, kind in zip(self.actions, region["action_kinds"]) if kind == "issue"
        )
        self.obligation_reduction["receipt_scalar_ssa"] = receipt_ssa
        banks = {group[0]: self._bank_projection(group[0]) for group in world_classes} if coupled and not receipt_ssa else {}
        use_banks = bool(banks) and all(bank is not None for bank in banks.values())
        for group in world_classes:
            if receipt_ssa:
                self._receipt_world(group[0], region["initial_sparse_worlds"][group[0]])
            elif use_banks:
                self._bank_world(group[0], region["initial_sparse_worlds"][group[0]], banks[group[0]])
            else:
                for index, participants in enumerate(actor_groups):
                    self._world(group[0], region["initial_sparse_worlds"][group[0]], participants, index)
        if receipt_ssa or use_banks:
            goal = z3.Goal(ctx=self.ctx)
            goal.add([z3.simplify(assertion) for assertion in self.solver.assertions()])
            if z3.Probe("is-qflia", ctx=self.ctx)(goal):
                self.solver = z3.SolverFor("QF_LIA", ctx=self.ctx)
                self.solver.set(random_seed=0, threads=1)
                self.solver.add(goal)
                self.obligation_reduction["solver_logic"] = "QF_LIA; z3 is-qflia probe"

    def _protocol_kinds(self, receipt_bound):
        letters = {("nop", None, False): [0]}
        for index, (node, kind) in enumerate(zip(self.actions, self.region["action_kinds"]), 1):
            handle = node.operands.get("handle", "receipt") if kind in {"issue", "wait"} else None
            source = node.operands.get("src")
            immutable = (source.get("resource") if isinstance(source, Mapping) else source) in {"@export:reg", "@export:smem"}
            letters.setdefault((kind, handle, immutable), []).append(index)
        capacity = sum(resource.extent for resource in self.region["resources"]
                       if resource.domain == "reg" and resource.name not in self.region["readonly_storage"])
        has_register_source = bool(self.region["resource_cells"].get("@export:reg"))
        widths = {kind: [node.operands["width"] for node, node_kind in zip(self.actions, self.region["action_kinds"]) if node_kind == kind]
                  for kind in ("reg_copy_round", "smem_store_round", "smem_load_round")}
        require_transfer = self.obligation_reduction.get("shared_transfer_required", False)
        states = [{(frozenset(), 0, False, capacity if has_register_source else 0, False, False, False)}]
        edges = []
        for _step in self.choices:
            outgoing, following = [], set()
            for state in states[-1]:
                pending, count, padded, filled, stored, published, transferred = state
                for letter in letters:
                    kind, handle, immutable = letter
                    if padded and kind != "nop" or kind == "issue" and handle in pending or kind == "wait" and handle not in pending:
                        continue
                    needed = (min(widths[kind]) if kind in {"reg_copy_round", "smem_store_round"} else
                              1 if kind in {"copy", "shfl_xor", "smem_store_scalar"} else 0) if not immutable else 0
                    if filled < needed or kind in {"smem_load_scalar", "smem_load_round"} and not immutable and not stored:
                        continue
                    gain = (self.region["packet_columns"] if kind == "wait" else
                            max(widths[kind]) if kind in {"reg_copy_round", "smem_load_round"} else
                            1 if kind in {"copy", "shfl_xor", "smem_load_scalar"} else 0)
                    target = (pending | {handle} if kind == "issue" else pending - {handle} if kind == "wait" else pending,
                              min(receipt_bound, count + (kind == "issue")), padded or kind == "nop",
                              min(capacity, filled + gain),
                              stored or kind in {"smem_store_scalar", "smem_store_round"},
                              published or kind == "rendezvous" and stored,
                              transferred or kind in {"smem_load_scalar", "smem_load_round"} and published)
                    following.add(target)
                    outgoing.append((state, target, letter))
            edges.append(outgoing)
            states.append(following)
        accepting = {state for state in states[-1] if not state[0] and state[1] >= receipt_bound and (not require_transfer or state[6])}
        possibilities = [set() for _step in self.choices]
        for step in reversed(range(len(self.choices))):
            previous = set()
            allowed = set()
            for source, target, letter in edges[step]:
                if target in accepting:
                    previous.add(source)
                    allowed.update(letters[letter])
                    possibilities[step].add(letter[0])
            self.solver.add(_member(self.choices[step], allowed, self.ctx))
            accepting = previous
        if not accepting:
            self.solver.add(z3.BoolVal(False, ctx=self.ctx))
        self.obligation_reduction["protocol_language_nonempty"] = bool(accepting)
        self.obligation_reduction["protocol_allowed_kinds"] = [sorted(kinds) for kinds in possibilities]
        self.obligation_reduction["protocol_rule"] = (
            "exact finite issue/matching-wait handles, empty at return; proved receipt/communication lower bounds; "
            "conservative initialized-private-slot upper bound and shared-write/publication reachability; "
            "no operation order or address is supplied"
        )
        return possibilities

    def kind_at(self, step, action, name):
        if name not in self.possible_kinds[step]:
            return z3.BoolVal(False, ctx=self.ctx)
        if self.possible_kinds[step] == {name}:
            return z3.BoolVal(True, ctx=self.ctx)
        return self.kind(action, name)


    def kind(self, action, name):
        return _member(action, self.kinds.get(name, ()), self.ctx)

    def require(self, active, *conditions):
        self.solver.add(z3.Implies(active, z3.And([z3.BoolVal(True, ctx=self.ctx), *conditions])))

    def field(self, name, action, participant=0, slot=0, world=0):
        return self.fields.value(name, action, participant, slot, world)

    def _packet_sources(self):
        packet, lane, slot = z3.Ints("packet_source_packet packet_source_lane packet_source_slot", ctx=self.ctx)
        bases, rows, columns = [], [], []
        fallback = []
        width, lanes = self.region["packet_columns"], self.region["warp_size"]
        regular = True
        for index, entry in enumerate(self.region["packets"]):
            cells = {(p, s): cell for p, s, cell in entry["receipt"]}
            base = cells[0, 0]
            row = cells[1, 0] - base if lanes > 1 else 0
            column = cells[0, 1] - base if width > 1 else 0
            regular &= all(cell == base + p * row + s * column for (p, s), cell in cells.items())
            bases.append((index, base))
            rows.append((index, row))
            columns.append((index, column))
            fallback.extend((index * lanes * width + p * width + s, cell) for (p, s), cell in cells.items())
        if regular:
            row_stride = rows[0][1] if rows and len({value for _index, value in rows}) == 1 else _lookup(packet, rows, ctx=self.ctx, default=0)
            column_stride = columns[0][1] if columns and len({value for _index, value in columns}) == 1 else _lookup(packet, columns, ctx=self.ctx, default=0)
            value = z3.If(z3.And(packet >= 0, packet < len(self.region["packets"])),
                          _lookup(packet, bases, ctx=self.ctx) + lane * row_stride + slot * column_stride, -1)
        else:
            value = _lookup(packet * lanes * width + lane * width + slot, sorted(fallback), ctx=self.ctx)
        return ((packet, lane, slot), value)

    def _pending_clear(self, cell, participant, live, issued, world):
        terms = []
        for handle, active in live.items():
            if z3.is_false(active):
                continue
            action = issued[handle]
            resource = self.field("dst_resource", action, participant, world=world)
            offset = self.field("offset", action, participant, world=world)
            if self.fields.contiguous_destinations:
                first = self.fields.cell(resource, participant, offset)
                last = self.fields.cell(resource, participant, offset + self.region["packet_columns"] - 1)
                overlap = z3.And(cell >= first, cell <= last)
            else:
                overlap = z3.Or([cell == self.fields.cell(resource, participant, offset + slot)
                                 for slot in range(self.region["packet_columns"])])
            terms.append(z3.Not(z3.And(active, overlap)))
        return z3.And([z3.BoolVal(True, ctx=self.ctx), *terms])

    def _capacity(self, live, retained):
        capacity = self.region["handle_capacity"]
        count = z3.Sum([z3.IntVal(0, ctx=self.ctx), *(z3.If(active, 1, 0) for active in live.values())])
        values = z3.Sum([z3.IntVal(0, ctx=self.ctx), *(z3.If(active, 1, 0) for active in retained.values())])
        private_excess = z3.If(values > capacity["thread"], values - capacity["thread"], 0)
        excess = []
        for warp in self.warps:
            members = sum(p // self.region["warp_size"] == warp for p in self.participants)
            demand = members * private_excess + count
            excess.append(z3.If(demand > capacity["warp"], demand - capacity["warp"], 0))
        self.solver.add(z3.Sum([z3.IntVal(0, ctx=self.ctx), *excess]) <= capacity["cta"])

    def _bank_projection(self, world):
        allowed = {"issue", "wait", "smem_store_round", "smem_load_round", "rendezvous"}
        if set(self.kinds) - allowed or len(self.fields.handles) > 1:
            return None
        private = [resource for resource in self.region["resources"]
                   if resource.domain == "reg" and resource.name not in self.region["readonly_storage"]]
        shared = [resource for resource in self.region["resources"]
                  if resource.domain == "smem" and resource.name not in self.region["readonly_storage"]]
        if len(private) != 1 or len(shared) != 1:
            return None
        private, shared = private[0], shared[0]
        width, size = private.extent, len(self.participants) * private.extent
        private_cells = tuple(cell for p in self.participants for cell in self.region["resource_cells"][private.name][p])
        shared_cells = self.region["resource_cells"][shared.name][-1]
        if len(shared_cells) != size or len(set(private_cells)) != size or len(set(shared_cells)) != size:
            return None
        source_rows = self.region["resource_cells"].get("@export:reg", {})
        source_cells = tuple(cell for p in self.participants for cell in source_rows.get(p, ()))
        if source_rows and (any(len(source_rows.get(p, ())) != width for p in self.participants) or any(cell < 0 for cell in source_cells)):
            return None
        immutable_stores = []
        inverse, loads = {}, {}
        domains = {"participant": expr.Domain("participant", "int", values=self.participants),
                   "slot": expr.Domain("slot", "int", 0, width)}
        for action, (node, kind) in enumerate(zip(self.actions, self.region["action_kinds"]), 1):
            operands = node.operands
            if kind == "issue":
                offset = operands.get("offset", 0)
                if operands["dest"] != private.name or type(offset) is not int or offset < 0 or offset + self.region["packet_columns"] > width:
                    return None
            if kind not in {"smem_store_round", "smem_load_round"}:
                continue
            store = kind == "smem_store_round"
            sources = {private.name, "@export:reg"} if store else {shared.name}
            if operands["src"] not in sources or operands["dst"] != (shared.name if store else private.name) or operands["width"] != width or operands.get("offset", 0) != 0:
                return None
            if store and operands["src"] == "@export:reg":
                immutable_stores.append(action)
            tree = _constant_bindings(_expression_tree(operands["address"]), self.region["worlds"][world], self.region["binding_domains"])
            parsed = expr.parse(tree, domains)
            values = []
            for participant in self.participants:
                for slot in range(width):
                    address = expr.evaluate(parsed, {"participant": participant, "slot": slot})
                    if address < 0 or address % self.region["scalar_bytes"] or address // self.region["scalar_bytes"] >= size:
                        return None
                    values.append(address // self.region["scalar_bytes"])
            if store:
                if len(set(values)) != size:
                    return None
                reversed_map = [0] * size
                for origin, destination in enumerate(values):
                    reversed_map[destination] = origin
                inverse[action] = tuple(reversed_map)
            else:
                loads[action] = tuple(values)
        if not inverse or not loads:
            return None
        return {"width": width, "size": size, "private": private.name, "shared": shared.name,
                "private_cells": private_cells, "shared_cells": shared_cells, "inverse": inverse, "loads": loads,
                "immutable_stores": tuple(immutable_stores), "source_cells": source_cells}

    def _bank_pair_relation(self, world, bank, inverse_maps, load_maps, receipt_rows, receipt_columns, history):
        """Extensional endpoint relation when semantic lower bounds leave two reads."""
        width, rows = bank["width"], len(self.participants)
        count = self.region["packet_columns"]
        if width != 2 * count or self.bounds.K > 8 or not self.obligation_reduction.get("shared_transfer_required"):
            return False
        if self.obligation_reduction.get("receipt_count_lower_bound") != 2:
            return False
        capture_classes = {}
        for first in receipt_rows:
            first_offset = self.actions[first - 1].operands.get("offset", 0)
            for second in receipt_rows:
                second_offset = self.actions[second - 1].operands.get("offset", 0)
                if {first_offset, second_offset} != {0, count} or receipt_rows[first] != receipt_rows[second]:
                    continue
                columns = [-1] * width
                for offset, action in ((first_offset, first), (second_offset, second)):
                    columns[offset:offset + count] = receipt_columns[action]
                if set(columns) != set(range(width)):
                    continue
                key = (receipt_rows[first], tuple(columns))
                capture_classes.setdefault(key, []).append((first, second))
        load_choices = {}
        for action in bank["loads"]:
            load_choices.setdefault((load_maps[0][action], load_maps[1][action]), []).append(action)
        groups = []
        for (captured_rows, captured_columns), pairs in capture_classes.items():
            inverse_rows = tuple(captured_rows.index(row) for row in range(rows))
            inverse_columns = tuple(captured_columns.index(column) for column in range(width))
            compatible_stores = {}
            for store in bank["inverse"]:
                store_forward_rows = tuple(inverse_maps[0][store].index(row) for row in range(rows))
                store_forward_columns = tuple(inverse_maps[1][store].index(column) for column in range(width))
                required = (tuple(store_forward_rows[inverse_rows[row]] for row in range(rows)),
                            tuple(store_forward_columns[inverse_columns[column]] for column in range(width)))
                loads = tuple(load_choices.get(required, ()))
                if loads:
                    compatible_stores.setdefault(loads, []).append(store)
            if compatible_stores:
                groups.append((pairs, compatible_stores))
        first_issue = z3.IntVal(0, ctx=self.ctx)
        second_issue = z3.IntVal(0, ctx=self.ctx)
        for step, action in enumerate(self.choices):
            issue = self.kind_at(step, action, "issue")
            second_issue = z3.If(z3.And(issue, first_issue != 0), action, second_issue)
            first_issue = z3.If(z3.And(issue, first_issue == 0), action, first_issue)
        last_load = z3.IntVal(0, ctx=self.ctx)
        preceding_store = z3.IntVal(0, ctx=self.ctx)
        for step in reversed(range(len(self.choices))):
            action = self.choices[step]
            preceding_store = z3.If(z3.And(last_load != 0, preceding_store == 0, self.kind_at(step, action, "smem_store_round")),
                                    action, preceding_store)
            last_load = z3.If(z3.And(last_load == 0, self.kind_at(step, action, "smem_load_round")), action, last_load)
        relations = []
        for pairs, compatible_stores in groups:
            capture = z3.Or([z3.And(first_issue == first, second_issue == second) for first, second in pairs])
            transport = z3.Or([z3.And(_member(preceding_store, stores, self.ctx), _member(last_load, loads, self.ctx))
                               for loads, stores in compatible_stores.items()])
            relations.append(z3.And(capture, transport))
        self.solver.add(z3.Or([z3.BoolVal(False, ctx=self.ctx), *relations]))
        self.obligation_reduction["bank_pair_endpoint_relation"] = {
            "capture_classes": len(capture_classes), "compatible_classes": len(groups),
            "rule": "receipt/communication lower bounds and K<=8 force exactly two matched receipts before "
                    "the first full-bank store; the last load and its preceding full-bank store determine final origins. "
                    "Checked tensor-product maps and bijective fixed demands yield an exact extensional relation; "
                    "redundant extra stores/loads/rendezvous remain selectable and all epoch/resource constraints remain",
        }
        return True


    def _immutable_bank_pair_relation(self, bank, inverse_maps, load_maps, immutable_maps):
        """Exact last-store/load relation when a private-bank relay cannot fit."""
        if self.fields.handles or not bank["immutable_stores"]:
            return False
        # A private-source store affecting the final load needs an initializing
        # immutable store/load first, then its own store/load and the required
        # publication. Smaller controllers can only read the immutable bank.
        relay_nodes = 2 * 2 + int(self.obligation_reduction.get("shared_transfer_required", False))
        if not self.obligation_reduction.get("shared_transfer_required") or self.bounds.K >= relay_nodes:
            return False
        rows, width = len(self.participants), bank["width"]
        load_choices = {}
        for action in bank["loads"]:
            load_choices.setdefault((load_maps[0][action], load_maps[1][action]), []).append(action)
        compatible = {}
        for store in bank["immutable_stores"]:
            captured_rows, captured_columns = immutable_maps[0][store], immutable_maps[1][store]
            if set(captured_rows) != set(range(rows)) or set(captured_columns) != set(range(width)):
                return False
            source_rows = tuple(captured_rows.index(row) for row in range(rows))
            source_columns = tuple(captured_columns.index(column) for column in range(width))
            forward_rows = tuple(inverse_maps[0][store].index(row) for row in range(rows))
            forward_columns = tuple(inverse_maps[1][store].index(column) for column in range(width))
            required = (tuple(forward_rows[source_rows[row]] for row in range(rows)),
                        tuple(forward_columns[source_columns[column]] for column in range(width)))
            loads = tuple(load_choices.get(required, ()))
            if loads:
                compatible.setdefault(loads, []).append(store)
        last_store = z3.IntVal(0, ctx=self.ctx)
        loaded_store = z3.IntVal(0, ctx=self.ctx)
        last_load = z3.IntVal(0, ctx=self.ctx)
        for step, action in enumerate(self.choices):
            store, load = self.kind_at(step, action, "smem_store_round"), self.kind_at(step, action, "smem_load_round")
            loaded_store = z3.If(load, last_store, loaded_store)
            last_load = z3.If(load, action, last_load)
            last_store = z3.If(store, action, last_store)
        self.solver.add(z3.Or([z3.BoolVal(False, ctx=self.ctx), *(
            z3.And(_member(loaded_store, stores, self.ctx), _member(last_load, loads, self.ctx))
            for loads, stores in compatible.items())]))
        self.obligation_reduction["immutable_bank_pair_endpoint_relation"] = {
            "compatible_classes": len(compatible), "private_relay_minimum_nodes": relay_nodes,
            "rule": "complete private bank starts empty; within this controller bound a private-source store "
                    "cannot feed the final load after a required publication. Exact bijective source and round "
                    "tensor-product maps determine the last load's preceding immutable store; all selected "
                    "operations, initializedness, epochs, and resource constraints remain unchanged",
        }
        return True


    def _bank_product_goals(self, world, bank, locations, history):
        """Factor a proved tensor-product origin relation, never its action choice."""
        width, rows = bank["width"], len(self.participants)

        def split(maps):
            row_maps, column_maps = {}, {}
            for action, mapping in maps.items():
                r = tuple(mapping[row * width] // width for row in range(rows))
                c = tuple(mapping[column] % width for column in range(width))
                if any(value != r[index // width] * width + c[index % width] for index, value in enumerate(mapping)):
                    return None
                row_maps[action], column_maps[action] = r, c
            return row_maps, column_maps

        inverse_maps, load_maps = split(bank["inverse"]), split(bank["loads"])
        if inverse_maps is None or load_maps is None:
            return False
        cell_rank = {cell: index for index, cell in enumerate(bank["private_cells"])}
        targets = {}
        for _participant, _position, cell, identity in self.region["result_cells_worlds"][world]:
            if identity in targets and targets[identity] != cell_rank[cell]:
                return False
            targets[identity] = cell_rank[cell]
        if len(targets) != bank["size"] or len(set(targets.values())) != bank["size"]:
            return False
        immutable_maps = ({}, {})
        if bank["immutable_stores"]:
            initial = dict(self.region["initial_sparse_worlds"][world])
            source_targets = tuple(targets.get(initial.get(cell, -1), -1) for cell in bank["source_cells"])
            if any(target < 0 for target in source_targets):
                return False
            immutable_maps = split({action: source_targets for action in bank["immutable_stores"]})
            if immutable_maps is None:
                return False
        if self._immutable_bank_pair_relation(bank, inverse_maps, load_maps, immutable_maps):
            return True
        receipts = {}
        count = self.region["packet_columns"]
        for identity, entries in locations.items():
            if identity not in targets:
                return False
            for action, location in entries:
                offset = self.actions[action - 1].operands.get("offset", 0)
                index = location // width * count + location % width - offset
                receipts.setdefault(action, {})[index] = targets[identity]
        receipt_rows, receipt_columns = {}, {}
        for action, mapping in receipts.items():
            if set(mapping) != set(range(rows * count)):
                return False
            r = tuple(mapping[row * count] // width for row in range(rows))
            c = tuple(mapping[column] % width for column in range(count))
            if any(value != r[index // count] * width + c[index % count] for index, value in mapping.items()):
                return False
            receipt_rows[action], receipt_columns[action] = r, c
        if not bank["immutable_stores"] and self._bank_pair_relation(world, bank, inverse_maps, load_maps, receipt_rows, receipt_columns, history):
            return True
        inverse_row = _map_function(inverse_maps[0], rows, rows, self.ctx, f"inverse_rows_{world}")
        inverse_col = _map_function(inverse_maps[1], width, width, self.ctx, f"inverse_columns_{world}")
        load_row = _map_function(load_maps[0], rows, rows, self.ctx, f"load_rows_{world}")
        load_col = _map_function(load_maps[1], width, width, self.ctx, f"load_columns_{world}")
        receipt_row = _map_function(receipt_rows, rows, rows, self.ctx, f"receipt_rows_{world}")
        receipt_col = _map_function(receipt_columns, count, width, self.ctx, f"receipt_columns_{world}")
        immutable_row = _map_function(immutable_maps[0], rows, rows, self.ctx, f"immutable_rows_{world}")
        immutable_col = _map_function(immutable_maps[1], width, width, self.ctx, f"immutable_columns_{world}")
        row_values = [z3.IntVal(row, ctx=self.ctx) for row in range(rows)]
        column_values = [z3.IntVal(column, ctx=self.ctx) for column in range(width)]
        space = z3.IntVal(0, ctx=self.ctx)
        resolved = [z3.BoolVal(False, ctx=self.ctx) for _column in range(width)]
        correct = [z3.BoolVal(False, ctx=self.ctx) for _column in range(width)]
        for step, (wait, issued_action, store, load, action) in reversed(tuple(enumerate(history))):
            was_private, was_shared = space == 0, space == 1
            if not z3.is_false(wait):
                row_correct = z3.And([_apply_smt(receipt_row, (issued_action, value), self.ctx) == row
                                     for row, value in enumerate(row_values)])
                offset = self.field("offset", issued_action, self.participants[0], world=world)
                for column, value in enumerate(column_values):
                    written = z3.And(z3.Not(resolved[column]), wait, was_private, value >= offset, value < offset + count)
                    match = z3.And(row_correct, _apply_smt(receipt_col, (issued_action, value - offset), self.ctx) == column)
                    correct[column] = z3.If(written, match, correct[column])
                    resolved[column] = z3.Or(resolved[column], written)
            if z3.is_false(store) and z3.is_false(load):
                continue
            from_load, from_store = z3.And(load, was_private), z3.And(store, was_shared)
            from_export = z3.And(from_store, _member(action, bank["immutable_stores"], self.ctx))
            if bank["immutable_stores"]:
                row_correct = z3.And([
                    _apply_smt(immutable_row, (action, _apply_smt(inverse_row, (action, value), self.ctx)), self.ctx) == row
                    for row, value in enumerate(row_values)])
                for column, value in enumerate(column_values):
                    written = z3.And(z3.Not(resolved[column]), from_export)
                    origin = _apply_smt(inverse_col, (action, value), self.ctx)
                    match = z3.And(row_correct, _apply_smt(immutable_col, (action, origin), self.ctx) == column)
                    correct[column] = z3.If(written, match, correct[column])
                    resolved[column] = z3.Or(resolved[column], written)
            for label, values, load_function, inverse_function in (
                ("row", row_values, load_row, inverse_row),
                ("column", column_values, load_col, inverse_col),
            ):
                for index, value in enumerate(values):
                    updated = z3.If(from_load, _apply_smt(load_function, (action, value), self.ctx),
                                    z3.If(from_store, _apply_smt(inverse_function, (action, value), self.ctx), value))
                    variable = z3.Int(f"product_{label}_{world}_{step}_{index}", ctx=self.ctx)
                    self.solver.add(variable == z3.simplify(updated))
                    values[index] = variable
            updated_space = z3.If(from_export, 2, z3.If(from_load, 1, z3.If(from_store, 0, space)))
            space = z3.Int(f"product_space_{world}_{step}", ctx=self.ctx)
            self.solver.add(space == z3.simplify(updated_space))
        self.solver.add(z3.And([*resolved, *correct]))
        self.obligation_reduction["bank_tensor_product"] = {
            "rows": rows, "columns": width,
            "rule": "exhaustive physical round-map and complete immutable register/receipt-map tensor-product equality; "
                    "row and column origins share the same action variables, timing, and overwrite predicates; "
                    "fixed demanded identity labels are bijective over the private bank",
        }
        return True


    def _bank_world(self, world, initial, bank):
        """Exact backward versioned origins for checked complete-bank rounds."""
        width, size = bank["width"], bank["size"]
        inverse = _map_function(bank["inverse"], size, width, self.ctx, f"store_inverse_{world}")
        loads = _map_function(bank["loads"], size, width, self.ctx, f"load_map_{world}")
        owner_classes = {}
        for action, mapping in bank["inverse"].items():
            owners = tuple(origin // width for origin in mapping)
            owner_classes.setdefault(owners, []).append(action)
        compatible = {}
        for owners, stores in owner_classes.items():
            compatible[tuple(stores)] = [action for action, mapping in bank["loads"].items()
                                         if all(owners[cell] == private // width for private, cell in enumerate(mapping))]
        store_kind = lambda action: self.kind(action, "smem_store_round")
        load_kind = lambda action: self.kind(action, "smem_load_round")
        for first in range(len(self.choices)):
            for second in range(first + 1, len(self.choices)):
                a, b = self.choices[first], self.choices[second]
                same_epoch = z3.And([z3.BoolVal(True, ctx=self.ctx),
                                    *(z3.Not(self.kind(self.choices[index], "rendezvous")) for index in range(first + 1, second))])
                store_store = z3.Or([z3.BoolVal(False, ctx=self.ctx), *(
                    z3.And(_member(a, stores, self.ctx), _member(b, stores, self.ctx))
                    for stores in owner_classes.values()
                )])
                store_load = z3.Or([z3.BoolVal(False, ctx=self.ctx), *(
                    z3.And(_member(a, stores, self.ctx), _member(b, load_actions, self.ctx))
                    for stores, load_actions in compatible.items()
                )])
                load_store = z3.Or([z3.BoolVal(False, ctx=self.ctx), *(
                    z3.And(_member(b, stores, self.ctx), _member(a, load_actions, self.ctx))
                    for stores, load_actions in compatible.items()
                )])
                self.require(z3.And(same_epoch, store_kind(a), store_kind(b)), store_store)
                self.require(z3.And(same_epoch, store_kind(a), load_kind(b)), store_load)
                self.require(z3.And(same_epoch, load_kind(a), store_kind(b)), load_store)

        values = dict(initial)
        locations_by_identity = {}
        packet_cache = {}
        domains = {"participant": expr.Domain("participant", "int", values=self.participants)}
        for action, (node, kind) in enumerate(zip(self.actions, self.region["action_kinds"]), 1):
            if kind != "issue":
                continue
            tree = _expression_tree(node.operands["packet"])
            key = canonical_bytes(tree)
            if key not in packet_cache:
                parsed = expr.parse(_constant_bindings(tree, self.region["worlds"][world], self.region["binding_domains"]), domains)
                packet_cache[key] = {p: expr.evaluate(parsed, {"participant": p}) for p in self.participants}
            seen = set()
            for actor, participant in enumerate(self.participants):
                packet = self.region["packets"][packet_cache[key][participant]]
                for lane, slot, source in packet["receipt"]:
                    if lane != participant % self.region["warp_size"]:
                        continue
                    identity = values[source]
                    if identity in seen:
                        raise ModelError(["complete-bank origin encoding requires non-replicating whole-site receipts"])
                    seen.add(identity)
                    location = actor * width + node.operands.get("offset", 0) + slot
                    locations_by_identity.setdefault(identity, []).append((action, location))

        live = z3.BoolVal(False, ctx=self.ctx)
        issued = z3.IntVal(0, ctx=self.ctx)
        initialized = [z3.BoolVal(False, ctx=self.ctx) for _slot in range(width)]
        immutable_ready = all(values.get(cell, -1) >= 0 for cell in bank["source_cells"])
        shared_initialized = z3.BoolVal(False, ctx=self.ctx)
        history = []
        generations = {name: z3.IntVal(0, ctx=self.ctx) for name in self.fields.barriers}
        for step, action in enumerate(self.choices):
            issue, wait = self.kind_at(step, action, "issue"), self.kind_at(step, action, "wait")
            store, load = self.kind_at(step, action, "smem_store_round"), self.kind_at(step, action, "smem_load_round")
            self.require(issue, z3.Not(live))
            self.require(wait, live)
            private_ready = z3.And(initialized)
            if bank["immutable_stores"]:
                immutable_store = _member(action, bank["immutable_stores"], self.ctx)
                private_ready = z3.Or(immutable_store, private_ready)
                self.require(z3.And(store, immutable_store), immutable_ready)
            self.require(store, private_ready, z3.Not(live))
            self.require(load, shared_initialized, z3.Not(live))
            history.append((wait, issued, store, load, action))
            offset = self.field("offset", issued, self.participants[0], world=world)
            initialized = [z3.Or(ready, load, z3.And(wait, offset <= slot, slot < offset + self.region["packet_columns"]))
                           for slot, ready in enumerate(initialized)]
            shared_initialized = z3.Or(shared_initialized, store)
            live = z3.simplify(z3.If(issue, True, z3.If(wait, False, live)))
            issued = z3.simplify(z3.If(issue, action, issued))
            if self.fields.handles:
                self._capacity({self.fields.handles[0]: live}, {})
            for index, name in enumerate(self.fields.barriers):
                active = z3.And(self.kind_at(step, action, "rendezvous"), self.field("barrier", action, world=world) == index)
                self.require(active, self.region["barriers"][name]["capacity"] >= 1,
                             generations[name] < self.region["barriers"][name]["generations"])
                generations[name] += z3.If(active, 1, 0)
        self.solver.add(z3.Not(live))

        self.obligation_reduction["complete_bank_origins"] = {
            "width": width, "size": size, "store_bijections": len(bank["inverse"]),
            "load_maps": len(bank["loads"]),
            "rule": "exact physical full-bank store inverses and complete private-bank loads; backward origins use "
                    "strictly earlier versions, while all-slot initializedness/pending exclusion and every same-epoch "
                    "access pair are checked; no controller order is supplied",
        }
        if self._bank_product_goals(world, bank, locations_by_identity, history):
            return
        private_indices = {cell: index for index, cell in enumerate(bank["private_cells"])}
        source_positions = {}
        for index, cell in enumerate(bank["source_cells"]):
            source_positions.setdefault(values.get(cell, -1), []).append(index)
        for goal, (_participant, _position, cell, identity) in enumerate(self.region["result_cells_worlds"][world]):
            location = z3.IntVal(private_indices[cell], ctx=self.ctx)
            space = z3.IntVal(0, ctx=self.ctx)  # private/shared/resolved immutable source
            correct = z3.BoolVal(False, ctx=self.ctx)
            for step, (wait, issued_action, store, load, action) in reversed(tuple(enumerate(history))):
                if all(z3.is_false(condition) for condition in (wait, store, load)):
                    continue
                was_private, was_shared = space == 0, space == 1
                offset = self.field("offset", issued_action, self.participants[0], world=world)
                receipt_write = z3.And(wait, was_private, location % width >= offset,
                                       location % width < offset + self.region["packet_columns"])
                expected = _lookup(issued_action, locations_by_identity.get(identity, []), ctx=self.ctx)
                next_correct = z3.If(receipt_write, z3.And(expected >= 0, location == expected), correct)
                from_load = z3.And(load, was_private)
                from_store = z3.And(store, was_shared)
                from_export = z3.And(from_store, _member(action, bank["immutable_stores"], self.ctx))
                source_position = _apply_smt(inverse, (action, location), self.ctx)
                next_correct = z3.If(from_export, _member(source_position, source_positions.get(identity, ()), self.ctx), next_correct)
                next_location = z3.If(from_load, _apply_smt(loads, (action, location), self.ctx),
                                       z3.If(from_store, _apply_smt(inverse, (action, location), self.ctx), location))
                next_space = z3.If(z3.Or(receipt_write, from_export), 2, z3.If(from_load, 1, z3.If(from_store, 0, space)))
                location = z3.Int(f"origin_location_{world}_{goal}_{step}", ctx=self.ctx)
                space = z3.Int(f"origin_space_{world}_{goal}_{step}", ctx=self.ctx)
                correct = z3.Bool(f"origin_correct_{world}_{goal}_{step}", ctx=self.ctx)
                self.solver.add(location == z3.simplify(next_location), space == z3.simplify(next_space),
                                correct == z3.simplify(next_correct))
            self.solver.add(z3.And(space == 2, correct))


    def _receipt_world(self, world, initial):
        """Exact scalar SSA when protocol analysis leaves only receipt actions.

        This reverses each validated destination scatter into a per-cell gather;
        it neither chooses receipts nor collapses issue/wait controller nodes.
        """
        source_values = dict(initial)
        deliveries = {}
        domains = {"participant": expr.Domain("participant", "int", values=self.participants)}
        decoded_packets = {}
        for index, (node, kind) in enumerate(zip(self.actions, self.region["action_kinds"]), 1):
            if kind != "issue":
                continue
            tree = _expression_tree(node.operands["packet"])
            key = canonical_bytes(tree)
            if key not in decoded_packets:
                parsed = expr.parse(_constant_bindings(tree, self.region["worlds"][world], self.region["binding_domains"]), domains)
                decoded_packets[key] = {p: expr.evaluate(parsed, {"participant": p}) for p in self.participants}
            offset = node.operands.get("offset", 0)
            for participant in self.participants:
                packet = self.region["packets"][decoded_packets[key][participant]]
                cells = self.region["resource_cells"][node.operands["dest"]][participant]
                for lane, slot, source in packet["receipt"]:
                    if lane == participant % self.region["warp_size"]:
                        deliveries.setdefault(cells[offset + slot], []).append((index, source_values[source]))
        handle = self.fields.handles[0]
        live = z3.BoolVal(False, ctx=self.ctx)
        issued = z3.IntVal(0, ctx=self.ctx)
        completions = []
        for step, action in enumerate(self.choices):
            issue = self.kind_at(step, action, "issue")
            wait = self.kind_at(step, action, "wait")
            self.require(issue, z3.Not(live))
            self.require(wait, live)
            if not z3.is_false(wait):
                completions.append((wait, issued))
            live = z3.simplify(z3.If(issue, True, z3.If(wait, False, live)))
            issued = z3.simplify(z3.If(issue, action, issued))
            self._capacity({handle: live}, {})
        self.solver.add(z3.Not(live))
        for participant, _position, cell, identity in self.region["result_cells_worlds"][world]:
            delivered_by_action = deliveries.get(cell, [])
            correct = z3.BoolVal(False, ctx=self.ctx)
            written_actions = [index for index, _value in delivered_by_action]
            correct_actions = [index for index, value in delivered_by_action if value == identity]
            for wait, action in completions:
                written = z3.And(wait, _member(action, written_actions, self.ctx))
                correct = z3.If(written, _member(action, correct_actions, self.ctx), correct)
            self.solver.add(z3.simplify(correct))
        self.obligation_reduction["receipt_endpoint_quotient"] = (
            "issue/wait-only support has no private payload reads; for each fixed demanded cell, "
            "retain exactly whether its last indivisible receipt wrote the demanded identity; "
            "no write preserves the prior predicate, and each action remains one complete receipt"
        )


    def _world(self, world, initial, participants, group):
        cell = z3.Int(f"initial_cell_{world}", ctx=self.ctx)
        source = ((cell,), _lookup(cell, list(initial), ctx=self.ctx))
        q = z3.Array(f"q_{world}_{group}_0", self.integer, self.integer)
        self.solver.add(q == z3.K(self.integer, z3.IntVal(-1, ctx=self.ctx)))

        def read_value(state, location):
            return z3.If(_apply_smt(self.source_predicate, (location,), self.ctx),
                         _apply_smt(source, (location,), self.ctx), z3.Select(state, location))

        readers = z3.K(self.integer, self.mask_zero)
        writers = z3.K(self.integer, self.mask_zero)
        live = {handle: z3.BoolVal(False, ctx=self.ctx) for handle in self.fields.handles}
        issued = {handle: z3.IntVal(0, ctx=self.ctx) for handle in self.fields.handles}
        generations = {name: z3.IntVal(0, ctx=self.ctx) for name in self.fields.barriers}
        metadata_names = sorted({node.operands["dest"] for node in self.actions if node.primitive == "observe_ready"})
        retained = {name: z3.BoolVal(False, ctx=self.ctx) for name in metadata_names}
        private_writes = []
        for step, action in enumerate(self.choices):
            next_q = q
            next_readers, next_writers = readers, writers
            old_live, old_issued = dict(live), dict(issued)

            for kind in ("copy", "reg_copy_round", "shfl_xor"):
                if kind not in self.possible_kinds[step]:
                    continue
                active, moved = self.kind_at(step, action, kind), q
                round_ = kind == "reg_copy_round"
                private_writes.append((active, action, "offset" if round_ else "dst_index", "width" if round_ else 1))
                maximum = max(node.operands["width"] for node, node_kind in zip(self.actions, self.region["action_kinds"]) if node_kind == kind) if round_ else 1
                for participant in participants:
                    source_participant = participant
                    if kind == "shfl_xor":
                        mask = self.field("mask", action, participant, world=world)
                        self.require(active, mask >= 0, mask < self.region["warp_size"])
                        source_participant = z3.BV2Int(z3.BitVecVal(participant, 64, ctx=self.ctx) ^ z3.Int2BV(mask, 64))
                    src_resource = self.field("src_resource", action, source_participant, world=world)
                    dst_resource = self.field("dst_resource", action, participant, world=world)
                    width = self.field("width", action, participant, world=world) if round_ else 1
                    for slot in range(maximum):
                        enabled = z3.And(active, slot < width)
                        src_index = self.field("src_offset" if round_ else "src_index", action, source_participant, world=world) + slot
                        dst_index = self.field("offset" if round_ else "dst_index", action, participant, world=world) + slot
                        src = self.fields.cell(src_resource, source_participant, src_index)
                        dst = self.fields.cell(dst_resource, participant, dst_index)
                        value = read_value(moved if round_ else q, src)
                        self.require(enabled, src >= 0, dst >= 0, value >= 0,
                                     self._pending_clear(src, source_participant, old_live, old_issued, world),
                                     self._pending_clear(dst, participant, old_live, old_issued, world))
                        moved = z3.If(slot < width, z3.Store(moved, dst, value), moved)
                next_q = z3.If(active, moved, next_q)

            for handle_index, handle in enumerate(self.fields.handles):
                if not {"issue", "wait"} & self.possible_kinds[step]:
                    continue
                issue = z3.simplify(z3.And(self.kind_at(step, action, "issue"), self.field("handle", action, world=world) == handle_index))
                wait = z3.simplify(z3.And(self.kind_at(step, action, "wait"), self.field("handle", action, world=world) == handle_index))
                self.require(issue, z3.Not(old_live[handle]))
                self.require(wait, old_live[handle])
                delivered = q
                if not z3.is_false(wait):
                    private_writes.append((wait, old_issued[handle], "offset", self.region["packet_columns"]))
                other_live = {other: active for other, active in old_live.items() if other != handle}
                for participant in participants:
                    if not z3.is_false(issue):
                        new_resource = self.field("dst_resource", action, participant, world=world)
                        new_offset = self.field("offset", action, participant, world=world)
                        for slot in range(self.region["packet_columns"]):
                            new_dst = self.fields.cell(new_resource, participant, new_offset + slot)
                            self.require(issue, new_dst >= 0,
                                         self._pending_clear(new_dst, participant, other_live, old_issued, world))
                    if not z3.is_false(wait):
                        previous = old_issued[handle]
                        resource = self.field("dst_resource", previous, participant, world=world)
                        offset = self.field("offset", previous, participant, world=world)
                        packet = self.field("packet", previous, participant, world=world)
                        self.require(wait, packet >= 0, packet < len(self.region["packets"]))
                        for slot in range(self.region["packet_columns"]):
                            dst = self.fields.cell(resource, participant, offset + slot)
                            src = _apply_smt(self.packet_source, (packet, participant % self.region["warp_size"], slot), self.ctx)
                            value = _apply_smt(source, (src,), self.ctx)
                            self.require(wait, src >= 0, dst >= 0, value >= 0)
                            delivered = z3.Store(delivered, dst, value)
                next_q = z3.If(wait, delivered, next_q)
                live[handle] = z3.simplify(z3.If(issue, True, z3.If(wait, False, old_live[handle])))
                issued[handle] = z3.simplify(z3.If(issue, action, old_issued[handle]))

            for kind in ("smem_store_scalar", "smem_store_round", "smem_load_scalar", "smem_load_round"):
                if kind not in self.possible_kinds[step]:
                    continue
                active = self.kind_at(step, action, kind)
                store = kind.startswith("smem_store")
                round_ = kind.endswith("_round")
                maximum = max(node.operands["width"] if round_ else 1 for node, node_kind in zip(self.actions, self.region["action_kinds"]) if node_kind == kind)
                if not store:
                    private_writes.append((active, action, "offset" if round_ else "dst_index", "width" if round_ else 1))
                moved, read_mask, write_mask = q, readers, writers
                for actor, participant in enumerate(participants):
                    bit = z3.BitVecVal(1 << actor, len(self.participants), ctx=self.ctx)
                    src_resource = self.field("src_resource", action, participant, world=world)
                    dst_resource = self.field("dst_resource", action, participant, world=world)
                    width = self.field("width", action, participant, world=world) if round_ else z3.IntVal(1, ctx=self.ctx)
                    for slot in range(maximum):
                        enabled = z3.And(active, slot < width)
                        if round_:
                            private_index = self.field("offset", action, participant, world=world) + slot
                            address = self.field("address", action, participant, slot, world)
                        else:
                            private_index = self.field("src_index" if store else "dst_index", action, participant, world=world)
                            address = self.field("dst_index" if store else "src_index", action, participant, world=world)
                        private = self.fields.cell(src_resource if store else dst_resource, participant, private_index)
                        shared = self.fields.cell(dst_resource if store else src_resource, -1, address)
                        self.require(enabled, private >= 0, shared >= 0, address % self.region["scalar_bytes"] == 0,
                                     self._pending_clear(private, participant, old_live, old_issued, world))
                        value = read_value(q, private if store else shared)
                        self.require(enabled, value >= 0)
                        if store:
                            self.require(enabled, z3.Select(read_mask, shared) & ~bit == self.mask_zero,
                                         z3.Select(write_mask, shared) & ~bit == self.mask_zero)
                            moved = z3.If(slot < width, z3.Store(moved, shared, value), moved)
                            write_mask = z3.If(slot < width, z3.Store(write_mask, shared, z3.Select(write_mask, shared) | bit), write_mask)
                        else:
                            self.require(enabled, z3.Select(writers, shared) & ~bit == self.mask_zero)
                            moved = z3.If(slot < width, z3.Store(moved, private, value), moved)
                            read_mask = z3.If(slot < width, z3.Store(read_mask, shared, z3.Select(read_mask, shared) | bit), read_mask)
                next_q = z3.If(active, moved, next_q)
                next_readers = z3.If(active, read_mask, next_readers)
                next_writers = z3.If(active, write_mask, next_writers)

            for name_index, name in enumerate(self.fields.barriers):
                active = z3.And(self.kind_at(step, action, "rendezvous"), self.field("barrier", action, world=world) == name_index)
                barrier = self.region["barriers"][name]
                self.require(active, barrier["capacity"] >= 1, generations[name] < barrier["generations"])
                generations[name] = generations[name] + z3.If(active, 1, 0)
                next_readers = z3.If(active, z3.K(self.integer, self.mask_zero), next_readers)
                next_writers = z3.If(active, z3.K(self.integer, self.mask_zero), next_writers)

            for name in metadata_names:
                indices = [index for index, node in enumerate(self.actions, 1)
                           if node.primitive == "observe_ready" and node.operands["dest"] == name]
                active = _member(action, indices, self.ctx)
                self.require(active, 1 in self.region["metadata_domain"],
                             name != "participant" and name not in self.region["worlds"][world])
                retained[name] = z3.Or(retained[name], active)
            self._capacity(live, retained)
            q = z3.Array(f"q_{world}_{group}_{step + 1}", self.integer, self.integer)
            self.solver.add(q == next_q)
            readers = z3.Array(f"readers_{world}_{group}_{step + 1}", self.integer, self.mask_sort)
            writers = z3.Array(f"writers_{world}_{group}_{step + 1}", self.integer, self.mask_sort)
            self.solver.add(readers == next_readers, writers == next_writers)
        for active in live.values():
            self.solver.add(z3.Not(active))
        for participant, _position, cell, identity in self.region["result_cells_worlds"][world]:
            if participant not in participants:
                continue
            coverage = []
            locations = [(self.fields.resource_ids[name], cells.index(cell))
                         for name, owners in self.region["resource_cells"].items()
                         if (cells := owners.get(participant)) is not None and cell in cells]
            for active, action, offset_name, width_spec in private_writes:
                resource = self.field("dst_resource", action, participant, world=world)
                offset = self.field(offset_name, action, participant, world=world)
                width = self.field(width_spec, action, participant, world=world) if isinstance(width_spec, str) else width_spec
                coverage.extend(z3.And(active, resource == resource_id, offset <= position, position < offset + width)
                                for resource_id, position in locations)
            self.solver.add(z3.Or([z3.BoolVal(False, ctx=self.ctx), *coverage]))
            self.solver.add(z3.Select(q, cell) == identity)
        self.obligation_reduction["private_write_coverage"] = (
            "every demanded private cell starts empty and must lie in at least one selected write footprint; "
            "physical aliases are matched by exact resource-cell mappings"
        )


def _singleton_source_candidates(instance, region, bounds):
    """Cheap exact identity join for one admitted immutable-source action.

    This only proposes controllers from the existing finite action domain. Every
    proposal still passes the independent interleaving checker; failure leaves
    the entire bounded symbolic search intact.
    """
    if bounds.K < 1:
        return
    for index, node in enumerate(region["actions"]):
        if node.primitive not in {"reg_copy_round", "smem_load_round"}:
            continue
        operands = node.operands
        source, destination = operands["src"], operands["dst"]
        if not source.startswith("@export:") or destination != instance.result_resource:
            continue
        if bounds.H is not None and region["action_costs"][index] + region["return_cost"] > bounds.H:
            continue
        width, offset = operands["width"], operands.get("offset", 0)
        if any(not offset <= position < offset + width for _participant, position, _identity in instance.demands):
            continue
        valid = True
        for world, demands in enumerate(instance.world_demands):
            expected = {(participant, position): identity for participant, position, identity in demands}
            environment = dict(instance.worlds[world])
            address = None
            if node.primitive == "smem_load_round":
                domains = {**instance._binding_domains(),
                           "participant": expr.Domain("participant", "int", values=region["participants"]),
                           "slot": expr.Domain("slot", "int", 0, width)}
                address = expr.parse(_constant_bindings(_expression_tree(operands["address"]), environment,
                                                       instance._binding_domains()), domains)
            for participant in region["participants"]:
                environment["participant"] = participant
                for slot in range(width):
                    environment["slot"] = slot
                    position = expr.evaluate(address, environment) if address is not None else operands.get("src_offset", 0) + slot
                    cell = instance._cell_index.get(instance._declared_cell(source, participant, position), -1)
                    identity = instance.initial_cells[world][cell] if cell >= 0 else -1
                    if identity < 0 or expected.get((participant, offset + slot), identity) != identity:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            yield index, node


def _synthesize_native_static(instance, bounds, region, domains, scope, provenance, budget):
    site = region["site"]
    terminal = next(node for node in domains[site] if isinstance(node, ir.Return))
    for index, node in _singleton_source_candidates(instance, region, bounds):
        program = instance.program({site: ir.ControlGraph((replace(node, next=1), terminal), 0)})
        checked = _checked(instance, program, bounds)
        provenance["candidates"].append({"singleton_source_action": index,
                                        "ir_digest": ir.digest(program), "check": checked.to_json_obj()})
        if checked.status == "PASS":
            scope["static_horizon"] = bounds.K
            provenance["singleton_source_search"] = "one admitted immutable-source round; exact finite identity join; full interleaving certification"
            return _result("MODEL_SAT", program, scope, provenance, budget)
    try:
        problem = _NativeStaticProblem(region, bounds)
    except ModelError as exc:
        provenance["reason"] = {"native_static_encoding": list(exc.issues)}
        provenance["static_premise_failed"] = True
        return _result("UNKNOWN", None, scope, provenance, budget)
    scope["static_horizon"] = bounds.K
    provenance["native_static_obligations"] = region["provenance"]["encoding_obligations"]
    provenance["native_static_semantics"] = region["semantics"]
    provenance["static_constraint_reduction"] = problem.obligation_reduction
    while True:
        outcome = budget.check(problem.solver)
        if outcome == z3.unknown:
            provenance["reason"] = "native static solver incomplete or rlimit exhausted"
            provenance["solver_reason_unknown"] = problem.solver.reason_unknown()
            return _result("UNKNOWN", None, scope, provenance, budget)
        if outcome == z3.unsat:
            return _result("BOUNDED_UNSAT", None, scope, provenance, budget)
        assignment = problem.solver.model()
        values = [assignment.eval(choice).as_long() for choice in problem.choices]
        selected = [problem.actions[index - 1] for index in values if index]
        nodes = tuple(replace(node, next=index + 1) for index, node in enumerate(selected)) + (terminal,)
        program = instance.program({site: ir.ControlGraph(nodes, 0)})
        checked = _checked(instance, program, bounds)
        provenance["candidates"].append({"assignment_digest": sha256_hex(canonical_bytes(values)),
                                        "ir_digest": ir.digest(program), "check": checked.to_json_obj()})
        if checked.status == "PASS":
            return _result("MODEL_SAT", program, scope, provenance, budget)
        if checked.status == "INCOMPLETE":
            provenance["reason"] = "whole-candidate checker incomplete"
            return _result("UNKNOWN", None, scope, provenance, budget)
        problem.solver.add(z3.Or([z3.BoolVal(False, ctx=problem.ctx),
                                  *(choice != value for choice, value in zip(problem.choices, values))]))
