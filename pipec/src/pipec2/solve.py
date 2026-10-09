from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, NoReturn

import z3

from .canonjson import canonical_dumps

from .model import (
    Claim,
    ClauseFailure,
    Interval,
    Launch,
    ObjectiveTier,
    Pipeline,
    ProtocolExpansion,
    RegGroup,
    Solution,
    SolveError,
    StageManifest,
)
from .profile import Profile


@dataclass(frozen=True)
class _ClaimRef:
    stage_id: str
    claim: Claim
    roles: tuple[str, ...]


@dataclass(frozen=True)
class _Record:
    handle: str
    domain: str
    physical_domain: str
    native_lo: int
    native_hi: int
    lo: int
    hi: int
    phase: str
    alias: str | None
    roles: tuple[str, ...]


def _reject(clause_id: str, path: str, message: str, **detail: Any) -> NoReturn:
    raise SolveError([ClauseFailure(clause_id, path, message, detail)])


def _extent(claim: Claim, values: dict[str, int], path: str) -> int:
    if claim.extent is None:
        _reject("L-OWNER", path, "interval claim has no extent")
    extent = claim.extent
    if extent.table:
        names: set[str] = set()
        for key in extent.table:
            names.update(part.split("=", 1)[0] for part in key.split(","))
        missing = sorted(names - values.keys())
        if missing:
            _reject("L-HANDLE", path, "extent table handle is unbound", handle=missing[0])
        key = ",".join(f"{name}={values[name]}" for name in sorted(names))
        if key not in extent.table:
            _reject("L-HANDLE", path, "extent table has no entry for the selected handle tuple", key=key)
        result = extent.table[key]
    else:
        missing = sorted(set(extent.per) - values.keys())
        if missing:
            _reject("L-HANDLE", path, "affine extent handle is unbound", handle=missing[0])
        result = extent.const + sum(coefficient * values[name] for name, coefficient in extent.per.items())
    if result <= 0:
        _reject("L-OVERLAP", path, "selected claim extent is not positive", extent=result)
    return result


def _scale(domain_name: str, pipeline: Pipeline, profile: Profile) -> int:
    domain = profile.domains[domain_name]
    if domain.unit_scale is not None:
        return domain.unit_scale
    declaration = pipeline.domains.get(domain_name)
    if declaration is None or declaration.page_bytes is None:
        _reject("L-HANDLE", f"pipeline.domains.{domain_name}", "pipeline-scaled domain has no page size")
    return declaration.page_bytes


def _native_capacity(
    domain_name: str,
    pipeline: Pipeline,
    profile: Profile,
    launch_warps: int,
) -> int:
    if domain_name == "warp":
        return launch_warps
    domain = profile.domains[domain_name]
    if domain.capacity == "pipeline":
        declaration = pipeline.domains.get(domain_name)
        if declaration is None:
            _reject("L-HANDLE", f"pipeline.domains.{domain_name}", "pipeline capacity is missing")
        return declaration.capacity
    physical = domain.physical_domain or domain_name
    if physical != domain_name:
        backing = profile.domains.get(physical)
        if backing is not None and isinstance(backing.capacity, int):
            return backing.capacity // _scale(domain_name, pipeline, profile)
    return domain.capacity


def _physical_capacity(
    physical_domain: str,
    pipeline: Pipeline,
    profile: Profile,
    launch_warps: int,
) -> int:
    if physical_domain == "warp":
        return launch_warps
    domain = profile.domains[physical_domain]
    if domain.capacity == "pipeline":
        declaration = pipeline.domains.get(physical_domain)
        if declaration is None:
            _reject("L-HANDLE", f"pipeline.domains.{physical_domain}", "pipeline capacity is missing")
        return declaration.capacity * _scale(physical_domain, pipeline, profile)
    return domain.capacity


def _phase(lifetime: Any) -> str:
    """Public interval label: ``kernel``, a phase name, or a tensor-event lifetime.

    Tensor-event lifetimes never take part in phase exclusivity; their ordering
    arrives resolved from link time as ``ordered_pairs``.
    """
    if not isinstance(lifetime, str):
        return lifetime.label()
    return lifetime.removeprefix("phase:") if lifetime.startswith("phase:") else "kernel"


def _exclusive(left: str, right: str, pipeline: Pipeline) -> bool:
    if left == "kernel" or right == "kernel" or left == right:
        return False
    return tuple(sorted((left, right))) in set(pipeline.exclusive)


def _candidate_values(
    pipeline: Pipeline,
    stages: list[StageManifest],
    profile: Profile,
) -> tuple[list[str], dict[str, tuple[int, ...]]]:
    names = sorted(name for name, handle in pipeline.handles.items() if handle.cls != "base")
    result: dict[str, tuple[int, ...]] = {}
    for name in names:
        handle = pipeline.handles[name]
        allowed = set(handle.values or [])
        for stage in stages:
            mentioned = {variant.key[name] for variant in stage.variants if name in variant.key}
            if mentioned:
                allowed &= mentioned
            for claim in stage.claims:
                if claim.domain == "regs" and claim.handle == name and claim.values is not None:
                    allowed &= set(claim.values)
        if name in pipeline.forced_bindings:
            allowed &= {pipeline.forced_bindings[name]}
        if not allowed:
            _reject("L-HANDLE", f"pipeline.handles.{name}.values", "selected-value constraints have an empty intersection")
        if handle.cls == "regs":
            launch_cap = name == "pipelink_launch_regs" and handle.owner == "reg_group:launch"
            maximum = profile.caps.launch_regs_max if launch_cap else profile.caps.regs_max
            step = 1 if launch_cap else profile.caps.regs_step
            invalid = sorted(
                value
                for value in allowed
                if not profile.caps.regs_min <= value <= maximum or value % step
            )
            if invalid:
                _reject(
                    "L-REGS",
                    f"pipeline.handles.{name}.values",
                    "launch register cap violates target range"
                    if launch_cap
                    else "register value violates target range or step",
                    value=invalid[0],
                )
        result[name] = tuple(sorted(allowed))
    return names, result


def _selected_variants(
    stages: list[StageManifest],
    values: dict[str, int],
) -> tuple[dict[str, dict[str, int]], tuple[int, ...]] | None:
    selected: dict[str, dict[str, int]] = {}
    indices: list[int] = []
    for stage in stages:
        matches = [
            variant
            for variant in stage.variants
            if all(values.get(name) == value for name, value in variant.key.items())
        ]
        if len(matches) != 1:
            return None
        variant = matches[0]
        selected[stage.id] = {name: variant.key[name] for name in sorted(variant.key)}
        indices.append(variant.index)
    return selected, tuple(indices)


def _claim_refs(stages: list[StageManifest]) -> tuple[dict[str, list[_ClaimRef]], list[_ClaimRef]]:
    intervals: dict[str, list[_ClaimRef]] = {}
    registers: list[_ClaimRef] = []
    for stage in stages:
        for claim in sorted(stage.claims, key=lambda item: (item.domain, item.handle)):
            roles = tuple(sorted(claim.roles or ()))
            if claim.domain == "warp" and not roles:
                if len(stage.roles) != 1:
                    _reject(
                        "L-TOPO",
                        f"stages.{stage.id}.claims.warp.{claim.handle}.roles",
                        "warp claim must name roles when its stage has more than one role",
                    )
                roles = (next(iter(stage.roles)),)
            ref = _ClaimRef(stage.id, claim, roles)
            if claim.domain == "regs":
                registers.append(ref)
            else:
                intervals.setdefault(claim.handle, []).append(ref)
    return intervals, registers


def _validate_owners(
    pipeline: Pipeline,
    stages: list[StageManifest],
    intervals: dict[str, list[_ClaimRef]],
    profile: Profile,
) -> None:
    for name, refs in sorted(intervals.items()):
        handle = pipeline.handles.get(name)
        if handle is None:
            _reject("L-OWNER", f"claims.{name}", "claim references an undeclared allocation handle")
        domains = {ref.claim.domain for ref in refs}
        if handle.cls != "base" or handle.binding != "bind" or handle.domain not in domains or len(domains) != 1:
            _reject(
                "L-OWNER",
                f"pipeline.handles.{name}",
                "allocation handle and its owning claims disagree",
                claim_domains=sorted(domains),
                handle_domain=handle.domain,
            )
    for name, handle in sorted(pipeline.handles.items()):
        if handle.cls == "base" and handle.binding == "bind" and name not in intervals:
            _reject("L-OWNER", f"pipeline.handles.{name}", "allocation handle has no owning claim")
    refs_by_stage_handle: dict[tuple[str, str], list[_ClaimRef]] = {}
    for name, refs in intervals.items():
        for ref in refs:
            refs_by_stage_handle.setdefault((ref.stage_id, name), []).append(ref)
    for stage in stages:
        for endpoint in sorted(stage.endpoints, key=lambda item: (item.name, item.direction)):
            handle = endpoint.binds.get("base")
            if handle is None:
                continue
            path = f"stages.{stage.id}.endpoints.{endpoint.name}.binds.base"
            refs = refs_by_stage_handle.get((stage.id, handle), [])
            if len(refs) != 1:
                _reject(
                    "L-OWNER",
                    path,
                    "endpoint base binding must resolve to exactly one stage claim",
                    handle=handle,
                    matches=len(refs),
                )
            claim_domain = refs[0].claim.domain
            domain = profile.domains[claim_domain]
            physical_domain = domain.physical_domain or claim_domain
            if physical_domain != endpoint.space:
                _reject(
                    "L-OWNER",
                    path,
                    "endpoint space and base claim physical domain disagree",
                    handle=handle,
                    claim_domain=claim_domain,
                    physical_domain=physical_domain,
                    endpoint_space=endpoint.space,
                )



def _role_widths(stages: list[StageManifest], profile: Profile) -> dict[str, int]:
    widths: dict[str, int] = {}
    for stage in stages:
        for role, declaration in sorted(stage.roles.items()):
            if role in widths:
                _reject("L-TOPO", f"roles.{role}", "launch role is declared by more than one stage")
            if declaration.warps <= 0:
                _reject("L-TOPO", f"roles.{role}", "role warp count must be positive")
            widths[role] = declaration.warps
    if not widths:
        _reject("L-TOPO", "roles", "pipeline has no launched roles")
    threads = sum(widths.values()) * 32
    if threads > profile.caps.max_threads:
        _reject("L-TOPO", "roles", "role warp counts exceed the target thread limit", threads=threads)
    return widths


def _warp_map(
    records: list[_Record],
    widths: dict[str, int],
    launch_warps: int,
) -> dict[str, tuple[int, int]] | None:
    result: dict[str, tuple[int, int]] = {}
    for record in sorted(records, key=lambda item: (item.native_lo, item.handle, item.roles)):
        if record.domain != "warp":
            continue
        if not record.roles:
            return None
        expected = sum(widths.get(role, -launch_warps) for role in record.roles)
        if expected != record.native_hi - record.native_lo:
            return None
        cursor = record.native_lo
        for role in record.roles:
            if role not in widths or role in result:
                return None
            result[role] = (cursor, cursor + widths[role])
            cursor += widths[role]
    if set(result) != set(widths):
        return None
    intervals = sorted(result.values())
    if not intervals or intervals[0][0] != 0 or intervals[-1][1] != launch_warps:
        return None
    if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        return None
    return {role: result[role] for role in sorted(result)}


def _register_groups(
    refs: list[_ClaimRef],
    values: dict[str, int],
    warp_map: dict[str, tuple[int, int]],
    pipeline: Pipeline,
    profile: Profile,
    launch_threads: int,
) -> tuple[dict[str, int], list[RegGroup], int] | None:
    grouped: dict[str, list[_ClaimRef]] = {}
    for ref in refs:
        claim = ref.claim
        assert claim.roles is not None
        name = claim.reg_group
        if name is None:
            if len(claim.roles) != 1:
                return None
            name = claim.roles[0]
        grouped.setdefault(name, []).append(ref)
    assigned_roles: set[str] = set()
    specs: list[tuple[str, list[str], int, int, str, int]] = []
    for name in sorted(grouped):
        group_refs = grouped[name]
        handles = {ref.claim.handle for ref in group_refs}
        if len(handles) != 1:
            return None
        handle = next(iter(handles))
        if handle not in values:
            return None
        declaration = pipeline.handles.get(handle)
        if (
            declaration is None
            or declaration.cls != "regs"
            or declaration.domain != "regs"
            or declaration.owner != f"reg_group:{name}"
        ):
            return None
        roles = sorted({role for ref in group_refs for role in ref.roles})
        if any(role in assigned_roles or role not in warp_map for role in roles):
            return None
        assigned_roles.update(roles)
        intervals = sorted(warp_map[role] for role in roles)
        lo, hi = intervals[0][0], intervals[-1][1]
        if lo % profile.caps.warpgroup_warps or hi % profile.caps.warpgroup_warps:
            return None
        if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
            return None
        if sum(end - start for start, end in intervals) != hi - lo:
            return None
        specs.append((name, roles, lo, hi, handle, values[handle]))
    if refs and assigned_roles != set(warp_map):
        return None
    expected_handles = {
        name
        for name, declaration in pipeline.handles.items()
        if declaration.cls == "regs" and name != "pipelink_launch_regs"
    }
    if {handle for _, _, _, _, handle, _ in specs} != expected_handles:
        return None

    pool = sum(value * (hi - lo) * 32 for _, _, lo, hi, _, value in specs)
    # This pre-link pool inequality is a sound prune, not the L-REGS gate.
    # ptxas may under-allocate below the requested cap, so observed pools
    # can be smaller than cap * threads and a pre-link pass does not guarantee
    # a post-link pass. Rejection here remains sound because the observed
    # allocation never exceeds the requested cap. The fail-closed post-link
    # observed-REG check in link.py is the definitive L-REGS gate.
    launch_handle = pipeline.handles.get("pipelink_launch_regs")
    if launch_handle is not None:
        if launch_handle.cls != "regs" or "pipelink_launch_regs" not in values:
            return None
        cap = values["pipelink_launch_regs"]
    else:
        granularity = profile.caps.reg_alloc_granularity
        cap = max(profile.caps.regs_min, (pool + launch_threads - 1) // launch_threads)
        cap = (cap + granularity - 1) // granularity * granularity
    cap_maximum = (
        profile.caps.launch_regs_max if launch_handle is not None else profile.caps.regs_max
    )
    if cap > cap_maximum or pool > cap * launch_threads:
        return None
    groups = [
        RegGroup(
            roles,
            lo,
            hi,
            handle,
            value,
            "dec" if value < cap else "inc" if value > cap else "none",
        )
        for _, roles, lo, hi, handle, value in specs
    ]
    regs = {name: value for name, _, _, _, _, value in specs}
    return regs, groups, cap


def _topology(
    pipeline: Pipeline,
    stages: list[StageManifest],
    profile: Profile,
) -> tuple[int, int, tuple[int, int, int] | str]:
    cta_groups = {stage.topology.cta_group for stage in stages}
    if len(cta_groups) != 1:
        _reject("L-TOPO", "stages.topology.cta_group", "stages disagree on compiled cta_group")
    cta_group = next(iter(cta_groups))
    if cta_group not in profile.caps.cta_group_legal:
        _reject("L-TOPO", "stages.topology.cta_group", "compiled cta_group is not target-legal")
    for stage in stages:
        if stage.topology.persistent != pipeline.topology.persistent:
            _reject("L-TOPO", f"stages.{stage.id}.topology.persistent", "stage and pipeline persistence disagree")
        if stage.topology.scheduler != pipeline.topology.scheduler:
            _reject("L-TOPO", f"stages.{stage.id}.topology.scheduler", "stage and pipeline schedulers disagree")
    fixed = {stage.topology.cluster_compiled for stage in stages if stage.topology.cluster_compiled != "dynamic"}
    if len(fixed) > 1:
        _reject("L-TOPO", "stages.topology.cluster.compiled", "compiled cluster shapes disagree")
    x_multiple = math.lcm(cta_group, *(stage.topology.cluster_launch.x_multiple_of for stage in stages))
    maximum = tuple(
        min(stage.topology.cluster_launch.max[axis] for stage in stages)
        for axis in range(3)
    )
    if x_multiple > maximum[0] or any(value < 1 for value in maximum):
        _reject("L-TOPO", "stages.topology.cluster.launch", "cluster launch constraints have an empty intersection")
    if fixed:
        shape = next(iter(fixed))
        assert isinstance(shape, tuple)
        if shape[0] % x_multiple or any(value > limit for value, limit in zip(shape, maximum)):
            _reject("L-TOPO", "stages.topology.cluster", "compiled cluster shape violates the combined launch constraints")
        cluster: tuple[int, int, int] | str = shape
    else:
        cluster = "dynamic"
    return x_multiple, maximum[0], cluster


def _effective_priority(
    declared: Iterable[str],
    eligible: Iterable[str],
) -> list[str]:
    declared_list = list(declared)
    eligible_set = set(eligible)
    return [*declared_list, *sorted(eligible_set - set(declared_list))]


def _public_intervals(records: list[_Record]) -> dict[str, list[Interval]]:
    result: dict[str, list[Interval]] = {}
    seen: set[tuple[str, str, int, int, str]] = set()
    for record in sorted(records, key=lambda item: (item.physical_domain, item.handle, item.lo, item.hi, item.phase)):
        key = (record.physical_domain, record.handle, record.lo, record.hi, record.phase)
        if key in seen:
            continue
        seen.add(key)
        result.setdefault(record.physical_domain, []).append(
            Interval(record.handle, record.lo, record.hi, record.phase)
        )
    return {domain: result[domain] for domain in sorted(result)}


# The solve allowance covers the sum of Z3 resource-counter deltas for all
# feasibility/objective checks; core extraction gets a separate bounded check.
# Exhaustion before a feasible layout rejects (L-BUDGET); exhaustion during an
# objective tier publishes the feasible incumbent with its recorded gap.
_SOLVE_RLIMIT = 10_000_000
_DIAGNOSTIC_RLIMIT = 1_000_000
_CLAUSE_PRECEDENCE = (
    "L-OVERLAP", "L-HANDLE", "L-TOPO", "L-REGS", "L-ALIGN",
    "L-PHASE", "L-OWNER", "L-PROTO",
)
# Physical domains whose high-water mark is an objective tier, in tier order.
# Warps are an exact tiling (no tier); remaining physical domains follow sorted.
_HIGH_WATER_ORDER = ("smem", "tmem", "named_barrier")


def solver_provenance() -> dict[str, Any]:
    return {
        "backend": "z3",
        "version": z3.get_version_string(),
        "package_version": "4.15.4.0",
        "objective": ["max_depths", "max_regs", "min_high_water", "min_variants", "min_finite_values"],
        "high_water_order": [*_HIGH_WATER_ORDER, "<other physical domains, sorted>"],
        "budget_policy": "reject_without_feasible_layout;publish_incumbent_with_gap",
        "random_seed": 0,
        "threads": 1,
        "solve_rlimit": _SOLVE_RLIMIT,
        "diagnostic_rlimit": _DIAGNOSTIC_RLIMIT,
        "rlimit_scope": "sum_of_check_resource_deltas;separate_diagnostic_check",
        "core_clause_precedence": list(_CLAUSE_PRECEDENCE),
    }


@dataclass
class _SMTBudget:
    limit: int
    phase: str
    used: int = 0

    def reject(self) -> NoReturn:
        _reject(
            "L-BUDGET", f"solve.{self.phase}",
            "solve budget exhausted before a feasible layout was found",
            backend="z3", phase=self.phase, limit=self.limit,
        )

    def attempt(self, solver: z3.Solver, cap: int | None = None) -> z3.CheckSatResult | None:
        """One bounded check; ``None`` when the whole allowance is exhausted,
        ``unknown`` when only this query's ``cap`` was reached."""
        remaining = self.limit - self.used
        if remaining <= 0:
            return None
        allowance = remaining if cap is None else min(cap, remaining)
        solver.set(rlimit=allowance)
        before = solver.statistics().get_key_value("rlimit count")
        result = solver.check()
        spent = solver.statistics().get_key_value("rlimit count") - before
        self.used += spent
        if self.used >= self.limit and result == z3.unknown or self.used > self.limit:
            return None
        if result == z3.unknown and allowance == remaining:
            return None
        return result

    def check(self, solver: z3.Solver) -> z3.CheckSatResult:
        result = self.attempt(solver)
        if result is None:
            self.reject()
        return result


@dataclass(frozen=True)
class _Objective:
    """One lexicographic tier. ``bound(model)`` is a valid best-possible value
    given the already fixed tiers (static ``low``/``high`` when absent)."""

    name: str
    expression: z3.ArithRef
    low: int
    high: int
    maximize: bool
    bound: Any = None
    hint: Any = None  # model -> constraints proposing a better incumbent, or None


class _Problem:
    """Canonical integer constraints plus bounded lexicographic optimization."""

    def __init__(self) -> None:
        # A fresh context also isolates Z3's resource counters from earlier calls.
        self.ctx = z3.Context()
        self.constraints: dict[tuple[str, str], list[z3.BoolRef]] = {}
        # Objective definitions (high-water variables): always satisfiable,
        # never part of an infeasibility diagnostic.
        self.definitions: list[z3.BoolRef] = []

    def integer(self, name: str) -> z3.ArithRef:
        return z3.Int(name, ctx=self.ctx)

    def number(self, value: int) -> z3.IntNumRef:
        return z3.IntVal(value, ctx=self.ctx)

    def all(self, terms: Iterable[Any]) -> z3.BoolRef:
        return z3.And([z3.BoolVal(True, ctx=self.ctx), *terms])

    def any(self, terms: Iterable[Any]) -> z3.BoolRef:
        return z3.Or([z3.BoolVal(False, ctx=self.ctx), *terms])

    def add(self, clause: str, path: str, *terms: Any) -> None:
        self.constraints.setdefault((clause, path), []).append(self.all(terms))

    def solver(self) -> z3.Solver:
        solver = z3.Solver(ctx=self.ctx)
        solver.set(random_seed=0, threads=1)
        return solver

    def diagnostic(self) -> NoReturn:
        solver = self.solver()
        labels = {}
        for index, ((clause, path), terms) in enumerate(sorted(self.constraints.items())):
            label = z3.Bool(f"constraint_{index}", ctx=self.ctx)
            labels[label.get_id()] = (clause, path)
            solver.assert_and_track(self.all(terms), label)
        result = _SMTBudget(_DIAGNOSTIC_RLIMIT, "diagnostic").check(solver)
        if result != z3.unsat:
            raise AssertionError("diagnostic constraints disagree with feasibility check")
        core = sorted(
            (labels[label.get_id()] for label in solver.unsat_core()),
            key=lambda item: (_CLAUSE_PRECEDENCE.index(item[0]), item[1]),
        )
        if not core:
            raise AssertionError("unsatisfiable encoding has no tracked conflicting set")
        clause, path = core[0]
        conflicting_set = [{"clause_id": item[0], "path": item[1]} for item in core]
        _reject(
            clause, path,
            "no feasible joint binding and placement exists; conflicting set: "
            + canonical_dumps(conflicting_set).strip(),
            backend="z3",
            conflicting_set=conflicting_set,
        )

    def optimum(
        self, objectives: list[_Objective]
    ) -> tuple[z3.ModelRef, list[ObjectiveTier], int]:
        solver = self.solver()
        for _, terms in sorted(self.constraints.items()):
            solver.add(self.all(terms))
        solver.add(*self.definitions)
        budget = _SMTBudget(_SOLVE_RLIMIT, "feasibility")
        if budget.check(solver) == z3.unsat:
            self.diagnostic()
        model = solver.model()
        tiers: list[ObjectiveTier] = []
        exhausted = False
        # Each tier improves its feasible incumbent by galloping queries away
        # from it (easy, satisfiable steps first; hard unsat proofs near the
        # bound last), then is fixed before the next tier. Every query is
        # capped at a quarter of the remaining allowance, so one hard proof
        # cannot starve later tiers. An unresolved step leaves a recorded gap.
        for objective in objectives:
            expression, maximize = objective.expression, objective.maximize
            sign = -1 if maximize else 1  # search minimizes sign * expression
            best = sign * model.eval(expression, model_completion=True).as_long()
            limit = sign * (objective.high if maximize else objective.low)
            if objective.bound is not None:
                limit = max(limit, sign * objective.bound(model))
            if best > limit and objective.hint is not None:
                proposal = objective.hint(model)
                if proposal is not None:
                    solver.push()
                    solver.add(sign * expression < best, *proposal)
                    result = budget.attempt(solver, cap=max(1, (budget.limit - budget.used) // 4))
                    if result is None:
                        exhausted = True
                    elif result == z3.sat:
                        model = solver.model()
                        best = sign * model.eval(expression, model_completion=True).as_long()
                    solver.pop()
            status = "bound_met" if best <= limit else "proved"
            step = 1
            while best > limit and not exhausted:
                target = max(limit, best - step)
                solver.push()
                solver.add(sign * expression <= target)
                result = budget.attempt(solver, cap=max(1, (budget.limit - budget.used) // 4))
                if result is None:
                    exhausted = True
                elif result == z3.sat:
                    model = solver.model()
                    best = sign * model.eval(expression, model_completion=True).as_long()
                    step *= 2
                elif result == z3.unsat:
                    limit, step = target + 1, 1
                elif step > 1:
                    step //= 2
                else:
                    solver.pop()
                    break
                solver.pop()
            value = model.eval(expression, model_completion=True).as_long()
            if best > limit:
                status = "budget_gap"
            tiers.append(ObjectiveTier(objective.name, value, sign * limit, status))
            solver.add(expression == value)
        return model, tiers, budget.used


def _symbolic_extent(
    problem: _Problem, ref: _ClaimRef, values: dict[str, z3.ArithRef]
) -> z3.ArithRef:
    claim = ref.claim
    path = f"stages.{ref.stage_id}.claims.{claim.domain}.{claim.handle}.extent"
    extent = claim.extent
    if extent is None:
        _reject("L-OWNER", path, "interval claim has no extent")
    if extent.table:
        names = {
            part.split("=", 1)[0] for key in extent.table for part in key.split(",")
        }
        missing = sorted(names - values.keys())
        if missing:
            _reject("L-HANDLE", path, "extent table handle is unbound", handle=missing[0])
        rows = [
            (
                problem.all(
                    values[name] == int(selected)
                    for name, selected in (part.split("=", 1) for part in key.split(","))
                ),
                result,
            )
            for key, result in sorted(extent.table.items())
        ]
        problem.add("L-HANDLE", path, problem.any(condition for condition, _ in rows))
        result = sum(
            (z3.If(condition, value, 0) for condition, value in rows),
            problem.number(0),
        )
    else:
        missing = sorted(extent.per.keys() - values.keys())
        if missing:
            _reject("L-HANDLE", path, "affine extent handle is unbound", handle=missing[0])
        result = sum(
            (coefficient * values[name] for name, coefficient in sorted(extent.per.items())),
            problem.number(extent.const),
        )
    problem.add("L-OVERLAP", path, result > 0)
    return result


def _encode_registers(
    problem: _Problem,
    refs: list[_ClaimRef],
    values: dict[str, z3.ArithRef],
    roles: dict[str, tuple[z3.ArithRef, z3.ArithRef]],
    widths: dict[str, int],
    pipeline: Pipeline,
    profile: Profile,
    launch_threads: int,
) -> None:
    grouped: dict[str, list[_ClaimRef]] = {}
    path = "solve.register_groups"
    for ref in refs:
        claim = ref.claim
        if claim.reg_group is None and len(ref.roles) != 1:
            problem.add("L-REGS", path, False)
            return
        grouped.setdefault(claim.reg_group or ref.roles[0], []).append(ref)
    assigned: set[str] = set()
    claimed_handles: set[str] = set()
    pool = problem.number(0)
    for name, group_refs in sorted(grouped.items()):
        handles = {ref.claim.handle for ref in group_refs}
        if len(handles) != 1:
            problem.add("L-REGS", path, False)
            return
        handle = next(iter(handles))
        declaration = pipeline.handles.get(handle)
        if (
            handle not in values or declaration is None or declaration.cls != "regs"
            or declaration.domain != "regs" or declaration.owner != f"reg_group:{name}"
        ):
            problem.add("L-REGS", path, False)
            return
        members = sorted({role for ref in group_refs for role in ref.roles})
        if not members or any(role in assigned or role not in roles for role in members):
            problem.add("L-REGS", path, False)
            return
        assigned.update(members)
        claimed_handles.add(handle)
        width = sum(widths[role] for role in members)
        lo = problem.integer(f"reg_group:{name}:lo")
        hi = lo + width
        # Warp roles are a disjoint exact launch partition. Containment plus total
        # width proves this group's interval is contiguous without enumerating
        # role permutations.
        problem.add(
            "L-REGS", path,
            lo % profile.caps.warpgroup_warps == 0,
            hi % profile.caps.warpgroup_warps == 0,
            *(problem.all((roles[role][0] >= lo, roles[role][1] <= hi)) for role in members),
        )
        pool += values[handle] * width * 32
    expected = {
        name for name, declaration in pipeline.handles.items()
        if declaration.cls == "regs" and name != "pipelink_launch_regs"
    }
    problem.add(
        "L-REGS", path,
        not refs or assigned == set(roles),
        claimed_handles == expected,
    )
    launch_handle = pipeline.handles.get("pipelink_launch_regs")
    if launch_handle is not None:
        if launch_handle.cls != "regs" or "pipelink_launch_regs" not in values:
            problem.add("L-REGS", path, False)
            return
        cap = values["pipelink_launch_regs"]
        maximum = profile.caps.launch_regs_max
    else:
        granularity = profile.caps.reg_alloc_granularity
        average = (pool + launch_threads - 1) / launch_threads
        minimum = z3.If(average < profile.caps.regs_min, profile.caps.regs_min, average)
        cap = ((minimum + granularity - 1) / granularity) * granularity
        maximum = profile.caps.regs_max
    problem.add("L-REGS", path, cap <= maximum, pool <= cap * launch_threads)


def solve(
    pipeline: Pipeline,
    stages: list[StageManifest],
    canon: dict[str, Any],
    expansions: list[ProtocolExpansion],
    profile: Profile,
    ordered_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> Solution:
    """Optimize the physical objective tiers over all feasible bindings.

    ``ordered_pairs`` holds sorted handle pairs whose lifetimes link time has
    proved ordered by a generated CTA-wide rendezvous; their smem claims may
    share addresses.
    """
    del canon
    ordered_stages = sorted(stages, key=lambda item: item.id)
    # Canonicalize complete claim/variant keys, not just domain+handle ties.
    ordered_stages = [
        dataclass_replace(
            stage,
            claims=sorted(stage.claims, key=lambda claim: canonical_dumps(claim.to_json_obj())),
            variants=sorted(stage.variants, key=lambda variant: (variant.index, sorted(variant.key.items()))),
        )
        for stage in ordered_stages
    ]
    if len({stage.id for stage in ordered_stages}) != len(ordered_stages):
        _reject("L-TOPO", "stages", "stage ids are not unique")
    # Expansion coupling has already been materialized into shared handles and
    # claims by protocol.py. The old solver only checked channel uniqueness here.
    if len({item.channel for item in expansions}) != len(expansions):
        _reject("L-PROTO", "protocol_expansions", "protocol expansion channels are not unique")
    _, _, cluster = _topology(pipeline, ordered_stages, profile)
    widths = _role_widths(ordered_stages, profile)
    launch_warps = sum(widths.values())
    launch_threads = launch_warps * 32
    interval_refs, register_refs = _claim_refs(ordered_stages)
    _validate_owners(pipeline, ordered_stages, interval_refs, profile)
    names, candidates = _candidate_values(pipeline, ordered_stages, profile)
    depth_handles = sorted(name for name, handle in pipeline.handles.items() if handle.cls == "depth" and handle.binding == "bind")
    regs_handles = sorted(name for name, handle in pipeline.handles.items() if handle.cls == "regs" and handle.binding == "bind")
    depth_order = _effective_priority(pipeline.depth_priority, depth_handles)
    regs_order = _effective_priority(pipeline.regs_priority, regs_handles)

    problem = _Problem()
    values = {name: problem.integer(f"value:{name}") for name in names}
    objectives = [
        _Objective(f"{kind}:{name}", values[name], candidates[name][0], candidates[name][-1], True)
        for kind, order in (("depth", depth_order), ("regs", regs_order)) for name in order
    ]
    for name in names:
        problem.add(
            "L-HANDLE", f"pipeline.handles.{name}.values",
            problem.any(values[name] == value for value in candidates[name]),
        )
    variant_objectives = []
    for stage in ordered_stages:
        choices = [
            (
                problem.all(
                    values[name] == value if name in values else False
                    for name, value in sorted(variant.key.items())
                ),
                variant.index,
            )
            for variant in stage.variants
        ]
        problem.add(
            "L-HANDLE", f"stages.{stage.id}.variants",
            sum((z3.If(condition, 1, 0) for condition, _ in choices), problem.number(0)) == 1,
        )
        index = sum(
            (z3.If(condition, index, 0) for condition, index in choices), problem.number(0)
        )
        indices = [variant.index for variant in stage.variants]
        variant_objectives.append(_Objective(
            f"variant:{stage.id}", index, min(indices, default=0), max(indices, default=0), False))

    bases = {}
    placement: dict[str, tuple[z3.ArithRef, int, int, int, int | None]] = {}
    symbolic: list[tuple[_ClaimRef, str, z3.ArithRef, z3.ArithRef, z3.ArithRef]] = []
    role_intervals: dict[str, tuple[z3.ArithRef, z3.ArithRef]] = {}
    for handle, refs in sorted(interval_refs.items()):
        domain_name = refs[0].claim.domain
        domain = profile.domains[domain_name]
        physical = domain.physical_domain or domain_name
        capacity = _native_capacity(domain_name, pipeline, profile, launch_warps)
        scale = _scale(domain_name, pipeline, profile)
        base = bases[handle] = problem.integer(f"base:{handle}")
        placement[handle] = (base, domain.lower_bound, domain.granularity, scale,
                             pipeline.forced_bindings.get(handle, next(
                                 (ref.claim.absolute_base - (ref.claim.offset or 0) for ref in refs
                                  if ref.claim.absolute_base is not None), None)))
        capacity_path = f"solve.capacity.{physical}"
        alignment_path = f"solve.alignment.{handle}"
        problem.add("L-OVERLAP", capacity_path, base >= domain.lower_bound, base < capacity)
        problem.add("L-ALIGN", alignment_path, base % domain.granularity == 0)
        if handle in pipeline.forced_bindings:
            problem.add("L-ALIGN", alignment_path, base == pipeline.forced_bindings[handle])
        for ref in refs:
            claim = ref.claim
            extent = _symbolic_extent(problem, ref, values)
            native_lo = base + (claim.offset or 0)
            native_hi = native_lo + extent
            lo, hi = native_lo * scale, native_hi * scale
            problem.add(
                "L-OVERLAP", capacity_path,
                native_lo >= domain.lower_bound, native_hi <= capacity,
                lo >= 0, hi > lo,
                hi <= _physical_capacity(physical, pipeline, profile, launch_warps),
            )
            problem.add("L-ALIGN", alignment_path, native_lo % claim.align == 0)
            if claim.absolute_base is not None:
                problem.add("L-ALIGN", alignment_path, native_lo == claim.absolute_base)
            symbolic.append((ref, physical, lo, hi, native_lo))
            if domain_name == "warp":
                valid = bool(ref.roles) and all(role in widths and role not in role_intervals for role in ref.roles)
                problem.add("L-TOPO", "solve.warp_map", valid)
                problem.add("L-TOPO", "solve.warp_map", extent == sum(widths.get(role, -launch_warps) for role in ref.roles))
                cursor = native_lo
                for role in ref.roles:
                    if role in widths:
                        role_intervals[role] = (cursor, cursor + widths[role])
                        cursor += widths[role]
    def may_overlap(left: _ClaimRef, right: _ClaimRef, physical: str) -> bool:
        if _exclusive(_phase(left.claim.lifetime), _phase(right.claim.lifetime), pipeline):
            return True
        pair = tuple(sorted((left.claim.handle, right.claim.handle)))
        return physical == "smem" and pair in ordered_pairs

    for index, (left, physical, lo, hi, _) in enumerate(symbolic):
        for right, other_physical, other_lo, other_hi, _ in symbolic[index + 1:]:
            if physical != other_physical:
                continue
            identical = problem.all((lo == other_lo, hi == other_hi))
            same_alias = left.claim.alias is not None and left.claim.alias == right.claim.alias
            same_handle = left.claim.handle == right.claim.handle and left.claim.domain == right.claim.domain and physical != "warp"
            if same_alias:
                problem.add("L-OVERLAP", f"solve.alias.{physical}.{left.claim.alias}", identical)
            if may_overlap(left, right, physical):
                continue
            problem.add(
                "L-OVERLAP", f"solve.capacity.{physical}",
                problem.any((hi <= other_lo, other_hi <= lo, identical if same_alias or same_handle else False)),
            )
    # Every role occurs once, has its declared width, lies within the launch and
    # is disjoint from every other role: total width then proves an exact tiling.
    problem.add("L-TOPO", "solve.warp_map", set(role_intervals) == set(widths))
    for index, (role, (lo, hi)) in enumerate(sorted(role_intervals.items())):
        problem.add("L-TOPO", "solve.warp_map", lo >= 0, hi <= launch_warps)
        for _, (other_lo, other_hi) in sorted(role_intervals.items())[index + 1:]:
            problem.add("L-TOPO", "solve.warp_map", problem.any((hi <= other_lo, other_hi <= lo)))
    _encode_registers(problem, register_refs, values, role_intervals, widths, pipeline, profile, launch_threads)
    # Every active-lifetime union is bounded because each interval lies in
    # [0, physical_capacity]; pairwise lifetime legality is encoded above.
    high_water = _high_water_objectives(
        problem, symbolic, set(depth_order) | set(regs_order), may_overlap, placement, pipeline, profile, launch_warps)
    # Finite values last, in canonical handle order: a deterministic tie-break
    # among layouts with equal physical tiers.
    finite_ties = [
        _Objective(f"finite:{name}", values[name], candidates[name][0], candidates[name][-1], False)
        for name in names
    ]
    model, tiers, rlimit_used = problem.optimum([*objectives, *high_water, *variant_objectives, *finite_ties])
    selected_values = {name: model.eval(value, model_completion=True).as_long() for name, value in values.items()}
    native_bases = {name: model.eval(base, model_completion=True).as_long() for name, base in bases.items()}
    records = []
    for ref, physical, lo, hi, native_lo in symbolic:
        claim = ref.claim
        native = model.eval(native_lo, model_completion=True).as_long()
        records.append(_Record(
            claim.handle, claim.domain, physical, native,
            native + _extent(claim, selected_values, f"claims.{claim.handle}.extent"),
            model.eval(lo, model_completion=True).as_long(),
            model.eval(hi, model_completion=True).as_long(),
            _phase(claim.lifetime), claim.alias, ref.roles,
        ))
    variants = _selected_variants(ordered_stages, selected_values)
    warp_map = _warp_map(records, widths, launch_warps)
    assert variants is not None and warp_map is not None
    register_result = _register_groups(register_refs, selected_values, warp_map, pipeline, profile, launch_threads)
    assert register_result is not None
    regs, reg_groups, regs_cap = register_result
    intervals = _public_intervals(records)
    return Solution(
        {name: selected_values[name] for name in depth_handles},
        {name: regs[name] for name in sorted(regs)},
        native_bases, intervals, variants[0],
        sorted(reg_groups, key=lambda item: (item.warp_lo, item.warp_hi, item.handle, item.roles)),
        warp_map,
        Launch(launch_threads, regs_cap, max((item.hi for item in intervals.get("smem", [])), default=0), cluster),
        tiers,
        rlimit_used,
    )


def _high_water_objectives(
    problem: _Problem,
    symbolic: list[tuple[_ClaimRef, str, z3.ArithRef, z3.ArithRef, z3.ArithRef]],
    fixed_handles: set[str],
    may_overlap: Any,
    placement: dict[str, tuple[z3.ArithRef, int, int, int, int | None]],
    pipeline: Pipeline,
    profile: Profile,
    launch_warps: int,
) -> list[_Objective]:
    """One minimized high-water tier per physical domain (warps excluded).

    Lower-bound certificate: handles that pairwise must be disjoint (no alias,
    phase exclusion or proved ordering between any of their claims) occupy at
    least the sum of their extents. Extents count only when they depend on
    constants or on depth/regs handles, which earlier tiers have fixed; others
    contribute zero, which keeps the bound valid.
    """
    by_domain: dict[str, list[tuple[_ClaimRef, z3.ArithRef, z3.ArithRef]]] = {}
    for ref, physical, lo, hi, _native in symbolic:
        if physical != "warp":
            by_domain.setdefault(physical, []).append((ref, lo, hi))
    order = [name for name in _HIGH_WATER_ORDER if name in by_domain]
    order += sorted(set(by_domain) - set(order))
    result = []
    for physical in order:
        entries = by_domain[physical]
        capacity = _physical_capacity(physical, pipeline, profile, launch_warps)
        mark = problem.integer(f"high_water:{physical}")
        # The mark is exactly the largest end, so a published (even budget-gap)
        # tier value always equals the layout's real high-water.
        problem.definitions.extend([
            mark <= capacity, *(mark >= hi for _ref, _lo, hi in entries),
            problem.any(mark == hi for _ref, _lo, hi in entries),
        ])
        refs_by_handle: dict[str, list[_ClaimRef]] = {}
        for ref, _lo, _hi in entries:
            refs_by_handle.setdefault(ref.claim.handle, []).append(ref)

        def conflict(left: str, right: str, refs_by_handle=refs_by_handle, physical=physical) -> bool:
            return not any(
                (a.claim.alias is not None and a.claim.alias == b.claim.alias) or may_overlap(a, b, physical)
                for a in refs_by_handle[left] for b in refs_by_handle[right]
            )

        def bound(model: z3.ModelRef, refs_by_handle=refs_by_handle, conflict=conflict, physical=physical) -> int:
            values = {name: model.eval(problem.integer(f"value:{name}"), model_completion=True).as_long()
                      for name in fixed_handles}
            weights = {}
            for handle, refs in refs_by_handle.items():
                weight = 0
                for ref in refs:
                    extent = ref.claim.extent
                    if extent is not None and not extent.table and set(extent.per) <= fixed_handles:
                        native = extent.const + sum(c * values[n] for n, c in extent.per.items())
                        weight = max(weight, native * _scale(ref.claim.domain, pipeline, profile))
                weights[handle] = weight
            clique: list[str] = []
            for handle in sorted(weights, key=lambda name: (-weights[name], name)):
                if all(conflict(handle, other) for other in clique):
                    clique.append(handle)
            return sum(weights[handle] for handle in clique)

        result.append(_Objective(f"high_water:{physical}", mark, 0, capacity, False, bound,
                                 _first_fit_hint(problem, entries, conflict, placement, capacity)))
    return result


def _first_fit_hint(problem, entries, conflict, placement, capacity):
    """Heuristic incumbent for a high-water tier: first-fit packing of the
    domain's handles (lexical and decreasing-size orders; the lower mark wins)
    under the current model's extents. It is only a hint: the solver checks the
    proposed bases like any other query and discards an infeasible proposal."""

    def hint(model: z3.ModelRef) -> list[z3.BoolRef] | None:
        spans: dict[str, list[tuple[int, int, int]]] = {}  # handle -> [(native offset, physical length, align)]
        for ref, lo, hi in entries:
            spans.setdefault(ref.claim.handle, []).append((
                ref.claim.offset or 0,
                model.eval(hi, model_completion=True).as_long() - model.eval(lo, model_completion=True).as_long(),
                ref.claim.align,
            ))

        def pack(order: list[str]) -> tuple[int, dict[str, int]] | None:
            placed: list[tuple[str, int, int]] = []
            chosen: dict[str, int] = {}
            for handle in order:
                _base, lower, granularity, scale, fixed = placement[handle]
                parts = spans[handle]
                blockers = [(lo, hi) for other, lo, hi in placed if conflict(handle, other)]

                def intervals(base: int) -> list[tuple[int, int]]:
                    return [((base + offset) * scale, (base + offset) * scale + length) for offset, length, _a in parts]

                def aligned(base: int) -> bool:
                    return base % granularity == 0 and all((base + offset) % align == 0 for offset, _l, align in parts)

                if fixed is not None:
                    candidates = [fixed]
                else:
                    step = math.lcm(granularity, *(align for _o, _l, align in parts))
                    starts = {lower, *(-(-hi // scale) - offset for _lo, hi in blockers for offset, _l, _a in parts)}
                    candidates = sorted({-(-max(start, lower) // step) * step for start in starts})
                for base in candidates:
                    spans_at = intervals(base)
                    if (aligned(base) and max(hi for _lo, hi in spans_at) <= capacity
                            and not any(lo < b_hi and b_lo < hi for lo, hi in spans_at for b_lo, b_hi in blockers)):
                        chosen[handle] = base
                        placed.extend((handle, lo, hi) for lo, hi in spans_at)
                        break
                else:
                    return None
            return max((hi for _h, _lo, hi in placed), default=0), chosen

        size = {handle: sum(length for _o, length, _a in parts) for handle, parts in spans.items()}
        packings = [result for order in (sorted(spans), sorted(spans, key=lambda h: (-size[h], h)))
                    if (result := pack(order)) is not None]
        if not packings:
            return None
        _mark, chosen = min(packings, key=lambda item: item[0])
        return [placement[handle][0] == base for handle, base in sorted(chosen.items())]

    return hint


__all__ = ["solve"]
