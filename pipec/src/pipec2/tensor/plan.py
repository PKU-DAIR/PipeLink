"""Contract admission and resource-bearing, model-relative tensor planning.

Stage promises are resolved once into Derived before admission. Expression and
logical coverage remain independent of synthesis; a candidate class is only
search order, never a semantic rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, groupby, product
from math import prod
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

from pipec2.canonjson import canonical_bytes, sha256_hex
from pipec2.model import Claim, ClauseFailure, Extent, Handle, Pipeline, Solution, StageManifest, TensorConnection, TensorLifetime, _JsonMixin
from pipec2.model import register_clause
from . import construct, expr, ir, machine, routes, synth
from .contracts import Binding, Lend, Site, TensorExport, TensorImport

EXPRESSION_CAP = 100_000
MODEL_STATE_BUDGET = 100_000
SEARCH_K = 4
SEARCH_RLIMIT = 10_000_000

register_clause("T-PLAN-FRAME", phase="solve", description="Register exports require sufficient caller-owned per-participant frames.")
register_clause("T-PLAN-RENDEZVOUS", phase="solve", description="Outer tensor events have declared barrier cohorts covering their callers.")


@dataclass(frozen=True)
class Derived(_JsonMixin):
    sites: tuple[Site, ...]
    resources: tuple[Lend, ...]
    storage: frozenset[str]
    readonly_storage: frozenset[str]
    ready_at_entry: bool
    early_reclaim: bool
    assumptions: tuple[str, ...]
    binding_domains: dict[str, expr.Domain]
    permitted: dict[str, frozenset[str]]
    catalogue: str
    target: str

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "sites": [site.to_json_obj() for site in self.sites],
            "resources": [resource.to_json_obj() for resource in self.resources],
            "storage": sorted(self.storage), "readonly_storage": sorted(self.readonly_storage),
            "ready_at_entry": self.ready_at_entry, "early_reclaim": self.early_reclaim,
            "assumptions": list(self.assumptions),
            "binding_domains": {name: domain.to_json_obj() for name, domain in sorted(self.binding_domains.items())},
            "permitted": {name: sorted(primitives) for name, primitives in sorted(self.permitted.items())},
            "catalogue": self.catalogue, "target": self.target,
        }


def _stated(profile, name: str) -> Any:
    """A target fact the derivation records; a Mapping test profile states every one explicitly."""
    if not isinstance(profile, Mapping):
        return getattr(profile, name)
    try:
        return profile[name]
    except KeyError:
        raise machine.ModelError([f"target profile does not state {name!r}"]) from None


def derive(export: TensorExport, imp: TensorImport, sites: Sequence[Site],
           bindings: Sequence[Binding], profile) -> Derived:
    """Translate stated owner promises into the connection's finite premises."""
    domains = {}
    for binding in bindings:
        if binding.name in domains:
            raise machine.ModelError([f"binding {binding.name!r} is declared by both stages"],
                                     status="CONTRACT_INVALID")
        domains[binding.name] = binding.domain
    for name, value in export.fixed_bindings.items():
        declared = domains.get(name)
        if declared is None:
            raise machine.ModelError([f"fixed binding {name!r} has no owner declaration"],
                                     status="CONTRACT_INVALID")
        if value not in _values(declared):
            raise machine.ModelError([f"fixed binding {name!r} is outside the composition domain"],
                                     status="CONTRACT_INVALID")
        domains[name] = expr.Domain(name, declared.sort, values=(value,))
    selected = tuple(sorted((site for site in sites if site.name in imp.sites), key=lambda site: site.name))
    resources = {}
    for resource in (*imp.lends, *export.backing):
        if resource.name in resources:
            raise machine.ModelError([f"duplicate resolved tensor resource {resource.name!r}"],
                                     status="CONTRACT_INVALID")
        resources[resource.name] = resource
    permitted = {site.name: routes.permitted_families(routes.facts(export, imp, site)) for site in selected}
    early_reclaim = export.readiness.reuse_after != imp.completion
    assumptions = set(_stated(profile, "fairness"))
    if not early_reclaim:
        # The owner keeps the source (and every storage backing it) untouched until the
        # import completes; lends are promised through completion by their own lifetime.
        assumptions.add(f"source_stable_until_{imp.completion}")
        assumptions.update(f"resource_{resource.name}_available_until_{imp.completion}" for resource in export.backing)
    assumptions.update(f"resource_{resource.name}_available_until_{imp.completion}"
                       for resource in imp.lends if resource.lifetime["to"] == imp.completion)
    # Model storage is what the site lends; owner backing enters the model only through the
    # export's own source binding (@export:*), never as additional cells to search over.
    return Derived(
        selected, tuple(resources[name] for name in sorted(resources)),
        frozenset(resource.name for resource in imp.lends if resource.binding != "caller_frame"),
        frozenset(resource.name for resource in export.backing),
        all(export.readiness.after in site.observes for site in selected), early_reclaim,
        tuple(sorted(assumptions)), domains, permitted, _stated(profile, "catalogue"), _stated(profile, "name"),
    )


@dataclass(frozen=True)
class Admission(_JsonMixin):
    status: Literal["OK", "CONTRACT_INVALID", "UNSUPPORTED_MODEL"]
    issues: tuple[ClauseFailure, ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class TensorPlan(_JsonMixin):
    status: str
    admission: Admission
    ir_digest: str | None
    program: ir.Program | None
    claims: tuple[Claim, ...]
    handles: tuple[Handle, ...]
    barrier_requests: tuple[dict[str, Any], ...]
    lowering: Any | None
    provenance: dict[str, Any]


def _failure(status: str, prerequisite: str, **detail: Any) -> ClauseFailure:
    return ClauseFailure("T-CONTRACT" if status == "CONTRACT_INVALID" else "T-MODEL",
                         "tensor.admission", prerequisite, detail)


def _values(domain: expr.Domain) -> range | tuple[int, ...]:
    return domain.values if domain.values is not None else range(domain.lo, domain.hi)


def _size(domain: expr.Domain) -> int:
    return len(domain.values) if domain.values is not None else domain.hi - domain.lo


def _environments(domains: Mapping[str, expr.Domain]):
    names = sorted(domains)
    for values in product(*(_values(domains[name]) for name in names)):
        yield dict(zip(names, values))


def _identity_header(identity) -> tuple[str, ...]:
    return (identity.result, identity.work, identity.version, identity.dtype, identity.interpretation)



def _analyze(expressions: Mapping[str, expr.Expr], domains: Mapping[str, expr.Domain],
             path: str, report: dict[str, Any]) -> tuple[str, str] | None:
    for name, expression in sorted(expressions.items()):
        checked = expr.domain_check(expression, domains, cap=EXPRESSION_CAP)
        report["analyzers"].append({"path": f"{path}.{name}", "analyzer": "expr.domain_check",
                                    "free_variables": sorted(expr._references(expression)),
                                    "result": checked.to_json_obj()})
        if checked.capped:
            return "UNSUPPORTED_MODEL", "expr.domain_check enumeration cap"
        if not checked.ok:
            return "CONTRACT_INVALID", "; ".join(checked.issues)
    return None


def _coverage(export: TensorExport, imp: TensorImport, bindings: Mapping[str, expr.Domain]) -> dict[str, Any]:
    """T1 identity join, not a transport oracle or a source of candidate edges."""
    coordinate_names = sorted(domain.name for domain in export.identity.coords)
    offered_domains = {domain.name: domain for domain in export.identity.coords}
    missing = []
    for binding in _environments(bindings):
        offered = set()
        for point in _environments(offered_domains):
            env = {**binding, **point}
            if any(view.guard is None or expr.evaluate(view.guard, env) for view in export.placements):
                offered.add(tuple(point[name] for name in coordinate_names))
        for index, demand in enumerate(imp.demands):
            for point in _environments({domain.name: domain for domain in demand.domain}):
                env = {**binding, **point}
                if demand.guard is not None and not expr.evaluate(demand.guard, env):
                    continue
                identity = tuple(expr.evaluate(demand.identity_coords[name], env) for name in coordinate_names)
                if identity not in offered:
                    missing.append({"bindings": binding, "demand": index, "point": point,
                                    "identity": list(identity)})
    return {"ok": not missing, "missing": missing, "kind": "necessary_condition_T1",
            "impossibility_premise": None if not missing else
                "a demanded typed logical identity is absent from every offered placement view",
            "not_sufficient_for_realizability": True}


def admit(export: TensorExport, imp: TensorImport, derived: Derived,
          profile) -> tuple[Admission, machine.Instance | None]:
    """Establish prerequisites and return admission evidence plus the admitted model."""
    report: dict[str, Any] = {"analyzers": [], "schema_validation": "already performed by contract loader"}

    def reject(status: str, prerequisite: str) -> tuple[Admission, None]:
        return Admission(status, (_failure(status, prerequisite, analyzer_cap=EXPRESSION_CAP),), report), None

    if _identity_header(export.identity) != _identity_header(imp.identity):
        return reject("CONTRACT_INVALID", "export/import typed logical result/work/version identity differs")
    if {domain.name for domain in export.identity.coords} != {domain.name for domain in imp.identity.coords}:
        return reject("CONTRACT_INVALID", "export/import logical coordinate names differ")
    if any(demand.kind == "memory_view" and (
            demand.space != "smem" or demand.units != "byte" or set(demand.components) != {"address"})
           for demand in imp.demands):
        return reject("UNSUPPORTED_MODEL", "memory_view requires shared byte-address demands")
    bindings = derived.binding_domains
    for index, placement in enumerate(export.placements):
        expressions = dict(placement.components)
        if placement.guard is not None:
            expressions["guard"] = placement.guard
        domains = {**bindings, **{domain.name: domain for domain in export.identity.coords}}
        issue = _analyze(expressions, domains, f"export.placements[{index}]", report)
        if issue:
            return reject(*issue)
    for index, demand in enumerate(imp.demands):
        expressions = {**demand.identity_coords, **(demand.components if demand.kind == "memory_view" else
                       {"participant": demand.participant, "position": demand.position})}
        if demand.guard is not None:
            expressions["guard"] = demand.guard
        domains = {**bindings, **{domain.name: domain for domain in demand.domain}}
        issue = _analyze(expressions, domains, f"import.demands[{index}]", report)
        if issue:
            return reject(*issue)
    for site in derived.sites:
        if site.participants.expr is not None:
            domains = {**bindings, **{domain.name: domain for domain in site.participants.domain}}
            issue = _analyze({"participants": site.participants.expr}, domains, f"sites.{site.name}", report)
            if issue:
                return reject(*issue)
    report["logical_coverage"] = _coverage(export, imp, bindings)
    try:
        instance = machine.Instance.build(export, imp, derived, profile)
        adequacy = instance.owner_adequacy(budget=MODEL_STATE_BUDGET)
    except machine.ModelError as exc:
        report["model_prerequisite"] = list(exc.issues)
        return Admission(exc.status, tuple(exc.failures), report), None
    # After the model's contract checks: a TMEM source on a target without TMEM is a
    # contract error there, not a missing family here.
    requested = set(export.read_capabilities)
    requested.update(op for permitted in derived.permitted.values() for op in permitted)
    if requested - set(instance.catalogue):
        return reject("UNSUPPORTED_MODEL", f"D-CAT primitive families unavailable: {sorted(requested - set(instance.catalogue))}")
    report["owner_adequacy"] = adequacy
    if not adequacy["ok"]:
        return reject("UNSUPPORTED_MODEL", adequacy["prerequisite"])
    return Admission("OK", (), report), instance


def _resolve(stages: Sequence[StageManifest], reference: str, direction: str):
    stage_id, name = reference.rsplit(":", 1)
    stage = next(stage for stage in stages if stage.id == stage_id)
    if stage.tensor is None:
        raise ValueError(f"stage {stage_id!r} has no tensor declarations")
    records = stage.tensor.exports if direction == "export" else stage.tensor.imports
    return stage, next(record for record in records if record.name == name)


def _derive_connection(producer: StageManifest, consumer: StageManifest,
                       export: TensorExport, imp: TensorImport, profile) -> Derived:
    stages = (producer,) if producer.id == consumer.id else (producer, consumer)
    return derive(export, imp, consumer.tensor.sites,
                  tuple(binding for stage in stages for binding in stage.tensor.bindings), profile)


def check_fanout(stages: Sequence[StageManifest],
                 connections: Sequence[TensorConnection], profile) -> tuple[ClauseFailure, ...]:
    """Require independent readers before planning imports of a shared export."""
    failures = []
    ordered = sorted(connections, key=lambda connection: (connection.export, connection.name))
    for export, members in groupby(ordered, key=lambda connection: connection.export):
        group = list(members)
        if len(group) == 1:
            continue
        resolved = []
        for connection in group:
            try:
                producer, export_record = _resolve(stages, connection.export, "export")
                consumer, imp = _resolve(stages, connection.import_, "import")
                derived = _derive_connection(producer, consumer, export_record, imp, profile)
            except (ValueError, StopIteration, machine.ModelError):
                # Per-connection admission owns unresolved contracts.
                continue
            bindings = derived.binding_domains
            participants: set[int] = set()
            for site in derived.sites:
                if site.participants.set is not None:
                    participants.update(site.participants.set)
                    continue
                domains = {**bindings,
                           **{domain.name: domain for domain in site.participants.domain}}
                domains = {name: domain for name, domain in domains.items()
                           if name in expr._references(site.participants.expr)}
                if prod(_size(domain) for domain in domains.values()) > EXPRESSION_CAP:
                    # Over the admission cap: the planner rejects this member as
                    # UNSUPPORTED_MODEL; do not enumerate it here.
                    participants = set()
                    break
                participants.update(expr.evaluate(site.participants.expr, env)
                                    for env in _environments(domains))
            resolved.append((connection, derived.early_reclaim, participants))
        for left, right in combinations(resolved, 2):
            path = f"pipeline.tensor_connections.{right[0].name}"
            detail = {"export": export, "sibling": left[0].name}
            reclaiming = [connection.name for connection, early_reclaim, _participants in (left, right)
                          if early_reclaim]
            if reclaiming:
                failures.append(ClauseFailure(
                    "T-CONTRACT", path, "fanout requires non-reclaiming readers",
                    {**detail, "early_reclaim": True, "reclaiming": reclaiming},
                ))
            overlap = left[2] & right[2]
            if overlap:
                failures.append(ClauseFailure(
                    "T-CONTRACT", path, "fanout requires disjoint participants",
                    {**detail, "participants": sorted(overlap)},
                ))
    return tuple(failures)


def _storage_handle_name(adapter: str, resource: Lend) -> str:
    return resource.location.handle or f"pipelink_tensor_{sha256_hex(canonical_bytes([adapter, resource.name]))[:16]}"


def check_fanout_storage(
    stages: Sequence[StageManifest], connections: Sequence[TensorConnection],
    planned_adapters: Mapping[str, TensorPlan], solution: Solution,
) -> tuple[ClauseFailure, ...]:
    """Compare sibling storage using admitted contracts and solved physical bases."""
    failures = []
    ordered = sorted(connections, key=lambda connection: (connection.export, connection.name))
    for export, members in groupby(ordered, key=lambda connection: connection.export):
        group = list(members)
        if len(group) == 1:
            continue
        placed = []
        for connection in group:
            producer, _export = _resolve(stages, connection.export, "export")
            consumer, imp = _resolve(stages, connection.import_, "import")
            resources = {resource.name: resource for resource in (*imp.lends, *_export.backing)
                         if resource.domain in {"smem", "barrier"}}
            readonly = {resource.name for resource in _export.backing}
            plan = planned_adapters[connection.name]
            adapter = plan.provenance["adapter_owner"]
            intervals = []
            for name, resource in sorted(resources.items()):
                extent = resource.extent if name in readonly else plan.provenance["resource_demands"][name]
                if not extent:
                    continue
                domain = "named_barrier" if resource.domain == "barrier" else resource.domain
                base = resource.location.base
                if base is None:
                    base = solution.native_bases.get(_storage_handle_name(adapter, resource))
                if base is not None:
                    intervals.append((domain, name, base, base + extent, name in readonly))
            placed.append((connection, intervals))
        for (left, left_intervals), (right, right_intervals) in combinations(placed, 2):
            for domain, name, lo, hi, readonly in left_intervals:
                for other_domain, other_name, other_lo, other_hi, other_readonly in right_intervals:
                    if not (readonly and other_readonly) and domain == other_domain and lo < other_hi and other_lo < hi:
                        failures.append(ClauseFailure(
                            "T-CONTRACT", f"pipeline.tensor_connections.{right.name}",
                            "fanout requires disjoint solved storage",
                            {"export": export, "sibling": left.name, "domain": domain,
                             "resources": [name, other_name],
                             "intervals": [[lo, hi], [other_lo, other_hi]]},
                        ))
    return tuple(failures)


def _oracle(instance: machine.Instance, imp: TensorImport, candidate: synth.SynthResult) -> dict[str, Any]:
    """Replay checker receipts against freshly evaluated frozen demand expressions.

    It checks identity delivery, not liveness (the exact checker certifies that).
    The oracle's outputs are never supplied to the candidate generator.
    """
    terminal = candidate.provenance["candidates"][-1]["check"]["provenance"].get("terminal_deliveries")
    if terminal is None:
        return {"ok": False, "incomplete": True, "kind": "verification_oracle",
                "reason": "checker did not publish terminal delivery evidence"}
    checked = 0
    failures = []
    coord_names = sorted(domain.name for domain in imp.identity.coords)
    for execution in terminal:
        delivered = {(participant, position): identity for participant, position, identity in execution["outputs"]}
        view = execution.get("view")
        if instance.demand_kind == "memory_view" and (
                view is None or view.get("resource") != candidate.program.view_binding):
            return {"ok": False, "incomplete": True, "kind": "verification_oracle",
                    "reason": "checker did not publish evidence for the bound view"}
        shared = {address: (identity, published) for address, identity, published in view["cells"]} if view else {}
        for demand in imp.demands:
            for point in _environments({domain.name: domain for domain in demand.domain}):
                env = {**execution["bindings"], **point}
                if demand.guard is not None and not expr.evaluate(demand.guard, env):
                    continue
                expected = tuple(expr.evaluate(demand.identity_coords[name], env) for name in coord_names)
                if demand.kind == "memory_view":
                    target = (-1, expr.evaluate(demand.components["address"], env))
                    identity, published = shared.get(target[1], (-1, -1))
                else:
                    target = (expr.evaluate(demand.participant, env), expr.evaluate(demand.position, env))
                    identity = delivered.get(target, -1)
                    published = identity
                actual = instance.identities[identity] if 0 <= identity < len(instance.identities) else None
                checked += 1
                if actual != expected or published != identity:
                    failures.append({"position": list(target), "expected": list(expected), "actual": actual,
                                     **({"published_identity": published} if demand.kind == "memory_view" else {})})
    return {"ok": bool(terminal) and not failures, "incomplete": False,
            "kind": "verification_oracle", "discovery_input": False,
            "checked_positions": checked, "terminal_executions": len(terminal), "failures": failures}


def _resource_demands(program: ir.Program, derived: Derived, consumer: StageManifest,
                      connection: TensorConnection, pipeline: Pipeline, accessed_bytes: Mapping[str, int]):
    used: set[str] = set()
    rendezvous_sites: dict[str, set[str]] = {}
    imp = next(item for item in consumer.tensor.imports if item.name == connection.import_.rsplit(":", 1)[1])
    outer_participants: dict[str, set[int]] = {}
    for rendezvous in imp.rendezvous:
        used.add(rendezvous.barrier)
        cohort = rendezvous.cohort
        members = set(cohort.set) if cohort.set is not None else {
            expr.evaluate(cohort.expr, env) for env in _environments(
                {**derived.binding_domains, **{domain.name: domain for domain in cohort.domain}})}
        outer_participants.setdefault(rendezvous.barrier, set()).update(members)

    def visit(value):
        if isinstance(value, str):
            used.add(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    if program.view_binding is not None:
        used.add(program.view_binding)
    for site_name, graph in sorted(program.graphs.items()):
        for node in graph.nodes:
            if isinstance(node, ir.Issue):
                visit(node.operands)
                if node.primitive == "cta_rendezvous":
                    rendezvous_sites.setdefault(node.operands["resource"], set()).add(site_name)
    participants: set[int] = set()
    site_participants: dict[str, set[int]] = {}
    for site in derived.sites:
        if site.participants.set is not None:
            members = set(site.participants.set)
        else:
            domains = {**derived.binding_domains,
                       **{domain.name: domain for domain in site.participants.domain}}
            members = {expr.evaluate(site.participants.expr, env) for env in _environments(domains)}
        site_participants[site.name] = members
        participants.update(members)
    demands = {
        resource.name: (accessed_bytes.get(resource.name, 0) if resource.domain == "smem"
                        else (1 if resource.domain == "barrier" else resource.extent)
                        if resource.name in used else 0)
        for resource in derived.resources
    }
    claims, handles, barriers = [], [], []
    adapter = f"{consumer.id}.tensor.{connection.name}"
    for resource in derived.resources:
        if not demands[resource.name] or resource.domain not in {"smem", "barrier"}:
            continue
        domain = "smem" if resource.domain == "smem" else "named_barrier"
        name = _storage_handle_name(adapter, resource)
        borrowed = name in pipeline.handles or (
            resource.binding == "fixed" and resource.location.base is None
        )
        if resource.domain == "barrier":
            arriving = set().union(*(site_participants[site] for site in rendezvous_sites.get(resource.name, ())))
            arriving.update(outer_participants.get(resource.name, ()))
            request = {"handle": name, "owner": adapter, "resource": resource.name,
                       "participants": sorted(arriving), "borrowed": borrowed,
                       "sync": None if resource.sync is None else resource.sync.to_json_obj()}
            reserved_zero = (
                resource.binding == "fixed" and resource.location.base == 0
                and bool(arriving)
            )
            if reserved_zero:
                request["reserved"] = 0
            barriers.append(request)
            if reserved_zero:
                continue
        # A fixed symbolic location is a caller-owned borrow, never a license to
        # allocate/relocate it. Existing pipeline reservations are also borrowed.
        if borrowed:
            continue
        handle = Handle(name, "base", "bind",
                        domain, None, f"claim:{adapter}:{name}")
        extent = (demands[resource.name] if resource.binding == "relocatable"
                  and resource.name in derived.storage else resource.extent)
        lifetime = (TensorLifetime(imp.name, resource.lifetime["from"], resource.lifetime["to"])
                    if resource.name in derived.storage else "kernel")
        claim = Claim(domain, name, lifetime, Extent(extent, {}, {}), resource.align,
                      0, None, None, None, resource.location.base, None)
        handles.append(handle)
        claims.append(claim)
    return (tuple(sorted(claims, key=lambda item: (item.domain, item.handle))),
            tuple(sorted(handles, key=lambda item: item.name)), tuple(barriers), adapter, demands)


def _lowering_context(instance: machine.Instance) -> dict[str, Any]:
    """Record model facts for lowering without changing absolute packet addresses."""
    referenced: set[str] = set()

    def visit(expression: expr.Expr) -> None:
        if expression.op == "bind":
            referenced.add(expression.args[0])
        for operand in expression.args:
            if isinstance(operand, expr.Expr):
                visit(operand)

    for placement in instance.export.placements:
        for expression in placement.components.values():
            visit(expression)
    residual = referenced & set(instance.export.residual_bindings)
    source_base = next(iter(residual)) if len(residual) == 1 else None
    exports = instance.exported_storage()
    domains = instance.derived.binding_domains
    return {
        "packets": [{"dp": packet.dp, "column": packet.column} for packet in instance.packets],
        "packet_columns_are_absolute": True,
        "participants": {site: list(members) for site, members in sorted(instance.participants.items())},
        "sites": {site.name: site.to_json_obj() for site in sorted(instance.sites, key=lambda item: item.name)},
        "result_resource": instance.result_resource,
        "demand_kind": instance.demand_kind,
        "view_binding": None,
        "warp_size": instance.warp_size,
        "packet_columns": instance.packet_columns,
        "scalar_bytes": instance.scalar_bytes,
        "readiness": instance.export.readiness.after,
        "ready_at_entry": instance.derived.ready_at_entry,
        "early_reclaim": instance.derived.early_reclaim,
        "storage": sorted(instance.derived.storage),
        "readonly_storage": sorted(instance.derived.readonly_storage),
        "permitted": {site: sorted(primitives) for site, primitives in instance.derived.permitted.items()},
        "binding_domains": {name: domain.to_json_obj() for name, domain in sorted(domains.items())},
        "exports": exports,
        "source_space": routes.source_spaces(instance.export)[0],
        "bindings_delivered_by_callable": [source_base] if source_base is not None else [],
        "source_base_binding": source_base,
        "source_base_reference": instance.export.fixed_bindings.get(source_base, 0),
    }


class ConnectionPlanner:
    """Resolve → admit → ordered route candidates → delivery oracle → resource claims.

    ``next()`` returns the next candidate that certifies (``MODEL_SAT``) or the
    terminal verdict once the candidates are exhausted; ``reject_lowering`` records
    a lowering failure on the current candidate so the link can resume.
    Route order is structural, never semantics.
    """

    def __init__(self, pipeline: Pipeline, stages: Sequence[StageManifest],
                 connection: TensorConnection, profile) -> None:
        self.pipeline, self.stages, self.connection, self.profile = pipeline, stages, connection, profile
        self.provenance: dict[str, Any] = {"connection": connection.name, "classes": [], "mode": "static",
                                           "search_order_not_semantics": True}
        self.instance: machine.Instance | None = None
        self.terminal: TensorPlan | None = None
        self._queue: list[routes.Route] = []
        self._checked: dict[str, construct.Constructed] = {}
        try:
            producer, export = _resolve(stages, connection.export, "export")
            self.consumer, self.imp = _resolve(stages, connection.import_, "import")
            self.export = export
            self.derived = _derive_connection(producer, self.consumer, export, self.imp, profile)
        except machine.ModelError as exc:
            admission = Admission(exc.status, tuple(exc.failures), {"model_prerequisite": list(exc.issues)})
            self.terminal = TensorPlan(admission.status, admission, None, None, (), (), (), None, {})
            return
        except (ValueError, StopIteration) as exc:
            admission = Admission("CONTRACT_INVALID", (_failure("CONTRACT_INVALID", f"unresolved tensor connection: {exc}"),), {})
            self.terminal = TensorPlan(admission.status, admission, None, None, (), (), (), None, {})
            return
        self.provenance["contract_digests"] = {
            "export": sha256_hex(canonical_bytes(export.to_json_obj())),
            "import": sha256_hex(canonical_bytes(self.imp.to_json_obj())),
            "derived": sha256_hex(canonical_bytes(self.derived.to_json_obj()))}
        self.admission, self.instance = admit(export, self.imp, self.derived, profile)
        if self.admission.status != "OK":
            self.terminal = TensorPlan(self.admission.status, self.admission, None, None, (), (), (), None, self.provenance)
            return
        instance = self.instance
        failures = []
        frames = [lend for lend in self.imp.lends if lend.binding == "caller_frame"]
        if "@export:reg" in instance.exported_storage():
            required = instance.exported_storage()["@export:reg"]["extent"]
            if len(frames) != 1 or frames[0].extent < required:
                failures.append(ClauseFailure("T-PLAN-FRAME", f"tensor.imports.{self.imp.name}.lends",
                    "register export requires exactly one sufficient caller_frame lend",
                    {"required_extent": required, "frames": {frame.name: frame.extent for frame in frames}}))
        events = {event.event for event in self.imp.rendezvous}
        if events != {export.readiness.after, self.imp.completion}:
            failures.append(ClauseFailure("T-PLAN-RENDEZVOUS", f"tensor.imports.{self.imp.name}.rendezvous",
                "readiness and completion require exactly their declared outer rendezvous",
                {"required_events": [export.readiness.after, self.imp.completion], "actual_events": sorted(events)}))
        if failures:
            self.admission = Admission("CONTRACT_INVALID", tuple(failures), self.admission.report)
            self.terminal = TensorPlan("CONTRACT_INVALID", self.admission, None, None, (), (), (), None, self.provenance)
            return
        self.provenance["model_digest"] = instance.model_digest
        self.provenance["lowering_context"] = _lowering_context(instance)
        # Routes are enumerated over the union of the sites' facts: a family a route
        # needs at only some sites is still filtered per site by the model's
        # `derived.permitted` (construct `issue`, synth `action_domain`), as before.
        available = frozenset().union(*(routes.facts(export, self.imp, site) for site in self.derived.sites))
        self.available = available
        self._queue, rejected = routes.candidates(routes.source_spaces(export), routes.sink_space(self.imp), available)
        self.provenance["facts"] = sorted(available)
        self.provenance["candidates"] = [route.to_json_obj() for route in self._queue]
        self.provenance["resource_envelope"] = [{"route": route.to_json_obj(), "missing": sorted(missing)}
                                                for route, missing in rejected]

    def reject_lowering(self, detail: Any) -> None:
        """Record that the last returned candidate did not lower; ``next`` resumes after it."""
        self.provenance["classes"][-1]["lowering"] = {"status": "LOWERING_UNSUPPORTED", "backend": detail}
        self.provenance.pop("selected", None)

    def next(self) -> TensorPlan:
        if self.terminal is not None:
            return self.terminal
        instance, derived, provenance = self.instance, self.derived, self.provenance
        while self._queue:
            route = self._queue.pop(0)
            started = perf_counter()
            entry = {"name": route.name, "route": route.to_json_obj()}
            if route.tier == 0:
                candidate = construct.construct(instance, route, self._checked)
                if "deduplicated_from" in candidate.provenance:
                    entry["deduplicated_from"] = candidate.provenance["deduplicated_from"]
                attempts = [{"mode": "static", "status": candidate.status, "scope": candidate.scope_record,
                             "provenance": candidate.provenance}]
            else:
                bounds = synth.Bounds(route.bound(instance, SEARCH_K), None, tuple(route.searched(self.available)), SEARCH_RLIMIT)
                candidate = synth.synthesize(instance, bounds, "static")
                attempts = [{"mode": "static", "status": candidate.status, "scope": candidate.scope_record,
                             "provenance": candidate.provenance}]
                if candidate.status == "UNKNOWN" and candidate.provenance.get("static_premise_failed"):
                    candidate = synth.synthesize(instance, bounds, "controller")
                    attempts.append({"mode": "controller", "status": candidate.status, "scope": candidate.scope_record,
                                     "provenance": candidate.provenance})
            entry.update(status=candidate.status, wall_s=perf_counter() - started, mode=candidate.provenance["mode"],
                         attempts=attempts, scope=candidate.scope_record, provenance=candidate.provenance)
            provenance["classes"].append(entry)
            if candidate.status != "MODEL_SAT":
                continue
            oracle = _oracle(instance, self.imp, candidate)
            provenance["admission_oracle"] = oracle
            if not oracle["ok"]:
                provenance["reason"] = "verification oracle incomplete" if oracle["incomplete"] else "delivery oracle disagrees with model certificate"
                self.terminal = TensorPlan("UNKNOWN", self.admission, None, None, (), (), (), None, provenance)
                return self.terminal
            claims, handles, barriers, adapter, demands = _resource_demands(
                candidate.program, derived, self.consumer, self.connection, self.pipeline,
                candidate.provenance["candidates"][-1]["check"]["provenance"]["accessed_bytes"])
            provenance["adapter_owner"] = adapter
            provenance["resource_demands"] = demands
            provenance["resource_lifetime"] = {
                claim.handle: (claim.lifetime.to_json_obj() if isinstance(claim.lifetime, TensorLifetime)
                               else claim.lifetime)
                for claim in claims
            }
            provenance["resource_bindings"] = candidate.resource_bindings
            provenance["mode"] = candidate.provenance["mode"]
            provenance["selected"] = len(provenance["classes"]) - 1
            provenance["lowering_context"]["view_binding"] = candidate.program.view_binding
            provenance.pop("view_binding", None)
            if candidate.program.view_binding is not None:
                provenance["view_binding"] = candidate.program.view_binding
            if candidate.provenance.get("generator") == "constructive-v1":
                provenance.update(generator="constructive-v1", fitted_expressions=candidate.provenance["fitted_expressions"],
                                  constructed_class=route.name)
            else:
                for key in ("generator", "fitted_expressions", "constructed_class"):
                    provenance.pop(key, None)
            return TensorPlan("MODEL_SAT", self.admission, ir.digest(candidate.program), candidate.program,
                              claims, handles, barriers, None, provenance)
        # Constructive rejection proves no impossibility. Only the bounded Z3
        # envelopes participate in the bounded-exhaustion aggregation (T5).
        bounded = [item for item in provenance["classes"] if item["provenance"].get("generator") != "constructive-v1"]
        if any("lowering" in item for item in provenance["classes"]):
            status = "LOWERING_UNSUPPORTED"
            provenance["reason"] = "every certified candidate was rejected by lowering"
        elif not provenance["classes"] and provenance["resource_envelope"]:
            status = "RESOURCE_ENVELOPE"
            provenance["reason"] = "no route is permitted by the declared lends and guarantees"
        elif not bounded or any(item["status"] == "UNKNOWN" for item in bounded):
            status = "UNKNOWN"
        else:
            status = "BOUNDED_UNSAT"
        provenance["aggregation"] = ("union of recorded Z3 route scopes; constructive rejections prove no impossibility; "
                                     "any incomplete bounded route is UNKNOWN; RESOURCE_ENVELOPE names the facts the cheapest route lacks")
        modes = {item["mode"] for item in provenance["classes"]}
        provenance["mode"] = next(iter(modes)) if len(modes) == 1 else "mixed" if modes else "static"
        self.terminal = TensorPlan(status, self.admission, None, None, (), (), (), None, provenance)
        return self.terminal
