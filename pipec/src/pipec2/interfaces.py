"""Link-time checks of stage entry rendezvous (L-JOIN) and callable interfaces (L-CALL).

A call channel pairs a caller endpoint with a provided export. A role-bearing caller runs in
its role (cohort = role width); a nested caller (no owner role) runs inside the exports of its
own stage named by the endpoint, in their cohort. The call graph is the export graph rooted in
roles.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from .layout import smem_param_address
from .model import CallInterface, CallParam, CallSite, ClauseFailure, Endpoint, LinkError, StageManifest
from .profile import Profile


def launch_threads(stages: Sequence[StageManifest]) -> int:
    """Launch threads of an assembly: every role's warps, as the solver packs them."""
    return 32 * sum(role.warps for stage in stages for role in stage.roles.values())


def check_entry_rendezvous(stages: Sequence[StageManifest], threads: int) -> None:
    """If any role-bearing stage joins at entry, all of them join the same full-launch barrier."""
    bearing = sorted((stage for stage in stages if stage.roles), key=lambda stage: stage.id)
    declared = {stage.entry_rendezvous for stage in bearing if stage.entry_rendezvous is not None}
    if not declared:
        return
    failures = []
    for stage in bearing:
        path = f"stages.{stage.id}.stage.entry_rendezvous"
        join = stage.entry_rendezvous
        if join is None or len(declared) > 1:
            failures.append(ClauseFailure("L-JOIN", path, "every role-bearing stage declares one identical entry rendezvous",
                                          {"declared": sorted((item.barrier, item.threads) for item in declared)}))
        elif join.threads != threads:
            failures.append(ClauseFailure("L-JOIN", f"{path}.threads", "the entry rendezvous spans the launch",
                                          {"threads": join.threads, "launch_threads": threads}))
    if failures:
        raise LinkError(failures)


@dataclass(frozen=True)
class _Call:
    channel: str
    caller: StageManifest
    site: Endpoint
    callee: StageManifest
    export: str

    @property
    def target(self) -> tuple[str, str]:
        return self.callee.id, self.export

    @property
    def origins(self) -> tuple[tuple[str, str], ...]:
        """Enclosing nodes: the caller role, or the caller stage's exports it is made from."""
        if self.site.owner_role is not None:
            return ((self.caller.id, f"role:{self.site.owner_role}"),)
        return tuple((self.caller.id, name) for name in self.site.exports or ())


def _interface(stage: StageManifest, export: str) -> CallInterface | None:
    return stage.exports[export].interface


def _enclosing_threads(call: _Call, origin: tuple[str, str]) -> int | None:
    stage_id, name = origin
    if name.startswith("role:"):
        return call.caller.roles[name.removeprefix("role:")].warps * 32
    interface = _interface(call.caller, name)
    return None if interface is None else interface.cohort.threads


def _check_params(path: str, provided: tuple[CallParam, ...], touched: tuple[CallParam, ...],
                  profile: Profile, fail: Any) -> None:
    by_index = {param.index: param for param in provided}
    for param in touched:
        site = by_index.get(param.index)
        here = f"{path}.params[{param.index}]"
        if site is None:
            fail(here, "the call site provides every parameter the export touches")
            continue
        for name in ("space", "dtype"):
            if getattr(site, name) != getattr(param, name):
                fail(f"{here}.{name}", f"call site and export {name} differ",
                     site=getattr(site, name), export=getattr(param, name))
        if site.extent < param.extent:
            fail(f"{here}.extent", "the call site provides fewer elements than the export touches",
                 site=site.extent, export=param.extent)
        if not set(param.access) <= set(site.access):
            fail(f"{here}.access", "the call site grants less access than the export takes",
                 site=site.access, export=param.access)
        if site.layout is not None and param.layout is not None and site.space == param.space == "smem":
            identities = [smem_param_address(layout, profile, f"{path}.params[{param.index}]")
                          for layout in (site.layout, param.layout)]
            if identities[0] != identities[1]:
                fail(f"{here}.layout", "call site and export shared-memory layouts differ")


def _calls(pairs: Sequence[tuple[str, Sequence[Any], Sequence[Any]]]) -> list[_Call]:
    calls = []
    for channel, producers, consumers in pairs:
        if not consumers or consumers[0].endpoint.protocol.call is None:
            continue
        callee = consumers[0]
        calls.append(_Call(channel, producers[0].stage, producers[0].endpoint, callee.stage,
                           callee.endpoint.protocol.call.export))
    return calls


def check_calls(pairs: Sequence[tuple[str, Sequence[Any], Sequence[Any]]], threads: int, profile: Profile,
                pipeline_stages: Sequence[StageManifest]) -> None:
    """Check paired call interfaces, acyclicity and single-instance barrier-bearing exports."""
    failures: dict[tuple[str, str], ClauseFailure] = {}

    def fail(path: str, message: str, **detail: Any) -> None:
        failures.setdefault((path, message), ClauseFailure("L-CALL", path, message, detail))

    calls = _calls(pairs)
    for call in calls:
        path = f"channels.{call.channel}.protocol"
        site: CallSite | None = call.site.protocol.call.site
        interface = _interface(call.callee, call.export)
        if (site is None) != (interface is None):
            fail(f"{path}.site" if site is None else f"stages.{call.callee.id}.exports.{call.export}.interface",
                 "a call site and its export state their interfaces together")
            continue
        if interface is not None and site is not None:
            if site.cohort.threads != interface.cohort.threads:
                fail(f"{path}.site.cohort.threads", "call site and export cohorts differ",
                     site=site.cohort.threads, export=interface.cohort.threads)
            _check_params(f"{path}.site", site.params, interface.params, profile, fail)
            if interface.collectives.kind == "cta" and interface.cohort.threads != threads:
                fail(f"stages.{call.callee.id}.exports.{call.export}.interface.cohort.threads",
                     "a CTA collective spans the launch", cohort=interface.cohort.threads, launch_threads=threads)
        if site is not None:
            for origin in call.origins:
                enclosing = _enclosing_threads(call, origin)
                if enclosing is None:
                    fail(f"stages.{origin[0]}.exports.{origin[1]}.interface",
                         "an export making a call with a stated site states its own interface")
                elif enclosing % site.cohort.threads:
                    fail(f"{path}.site.cohort.threads", "a call site cohort divides its enclosing cohort",
                         cohort=site.cohort.threads, enclosing=enclosing)

    # Edges from enclosing nodes (roles, exports) to callee exports.
    edges: dict[tuple[str, str], list[_Call]] = {}
    for call in calls:
        for origin in call.origins:
            edges.setdefault(origin, []).append(call)
    state: dict[tuple[str, str], int] = {}

    def visit(node: tuple[str, str]) -> None:
        state[node] = 1
        for call in edges.get(node, ()):
            if state.get(call.target) == 1:
                fail(f"channels.{call.channel}", "the pipeline call graph is acyclic",
                     export=f"{call.target[0]}:{call.target[1]}")
            elif call.target not in state:
                visit(call.target)
        state[node] = 2

    for node in sorted(edges):
        if node not in state:
            visit(node)
    if failures:
        raise LinkError(list(failures.values()))

    # Barrier-bearing exports (named/CTA collectives, transitively) run once per barrier
    # instance: every path reaching one starts in a single role and keeps the full cohort.
    bearing: dict[tuple[str, str], bool] = {}
    stages = {call.callee.id: call.callee for call in calls} | {call.caller.id: call.caller for call in calls}

    def barrier_bearing(node: tuple[str, str]) -> bool:
        if node not in bearing:
            interface = _interface(stages[node[0]], node[1])
            bearing[node] = bool(interface is not None and interface.collectives.kind in {"named", "cta"})
            bearing[node] = bearing[node] or any(barrier_bearing(call.target) for call in edges.get(node, ()))
        return bearing[node]

    def paths(node: tuple[str, str]) -> Iterator[list[tuple[_Call, tuple[str, str]]]]:
        if node[1].startswith("role:"):
            yield []
            return
        for call in calls:
            if call.target == node:
                for origin in call.origins:
                    for prefix in paths(origin):
                        yield [*prefix, (call, origin)]

    # Allocation handles are pipeline-global, and the solver lets equal handles or equal aliases
    # share one physical barrier. An instance is the equivalence class of handles and aliases
    # joined by any named-barrier claim.
    parent: dict[str, str] = {}

    def find(label: str) -> str:
        parent.setdefault(label, label)
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for stage in pipeline_stages:
        for claim in stage.claims:
            if claim.domain == "named_barrier" and claim.alias is not None:
                parent[find(f"handle:{claim.handle}")] = find(f"alias:{claim.alias}")

    def instance_key(stage_id: str, handle: str) -> str:
        return find(f"handle:{handle}")

    def instances(node: tuple[str, str]) -> set[str]:
        """Barrier instances an export synchronizes on, itself or through its callees."""
        interface = _interface(stages[node[0]], node[1])
        own = ({instance_key(node[0], interface.collectives.handle)}
               if interface is not None and interface.collectives.kind == "named" else set())
        return own.union(*(instances(call.target) for call in edges.get(node, ())))

    # Roles reaching each barrier instance, across every export (and library) that uses it.
    instance_roots: dict[str, set[tuple[str, str]]] = {}
    for node in sorted({call.target for call in calls}):
        if not barrier_bearing(node):
            continue
        roots = set()
        for path in paths(node):
            roots.add(path[0][1])
            for call, origin in path:
                site = call.site.protocol.call.site
                enclosing = _enclosing_threads(call, origin)
                if site is None or enclosing is None or site.cohort.threads != enclosing:
                    fail(f"channels.{call.channel}.protocol.site.cohort",
                         "calls reaching a barrier-bearing export keep their enclosing cohort",
                         export=f"{node[0]}:{node[1]}", enclosing=enclosing,
                         cohort=None if site is None else site.cohort.threads)
        if len(roots) > 1:
            fail(f"stages.{node[0]}.exports.{node[1]}", "a barrier-bearing export is reached from one role",
                 roles=sorted(f"{stage}:{role.removeprefix('role:')}" for stage, role in roots))
        for instance in instances(node):
            instance_roots.setdefault(instance, set()).update(roots)
    for instance, roots in sorted(instance_roots.items()):
        if len(roots) > 1:
            labels = sorted(label.removeprefix("handle:") for label in parent
                            if label.startswith("handle:") and find(label) == instance)
            fail(f"claims.{labels[0]}", "a named barrier is reached from one role across its exports",
                 handles=labels, roles=sorted(f"{owner}:{role.removeprefix('role:')}" for owner, role in roots))
    if failures:
        raise LinkError(list(failures.values()))


__all__ = ["check_calls", "check_entry_rendezvous", "launch_threads"]
