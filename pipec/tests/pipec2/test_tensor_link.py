from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import pipec2.link as link_module
from pipec2.layout import load_layout
from pipec2.model import Claim, ClauseFailure, Extent, Handle, Interval, LinkError, StageRef, TensorConnection
from pipec2.profile import get_profile
from pipec2.tensor import expr, ir, lower, plan
from pipec2.tensor.contracts import Binding, Lend, Participants, Readiness, ResourceLocation, load_stage_tensor
from test_link import (
    _paged_stage, _pipeline, _register_pipeline, _register_solution,
    _solution, _stage, _symbol_table,
)
from test_tensor_schema import _tensor_section


def _composition():
    tensor = load_stage_tensor(_tensor_section(), "$.tensor")
    target = replace(_stage().target, schema="3.0", lto_arch="lto_103a")
    export = replace(tensor.exports[0], readiness=Readiness("ready", tensor.imports[0].completion))
    producer = replace(_stage("org.producer"), target=target,
                       tensor=replace(tensor, exports=(export,), imports=()), symbols=_symbol_table())
    consumer = replace(_stage("org.consumer"), target=target,
                       tensor=replace(tensor, exports=(), bindings=()),
                       symbols=_symbol_table(undefined=(tensor.imports[0].callable.symbol,)))
    pipeline = replace(
        _pipeline(), target=target,
        stages=[StageRef("producer.plink", producer.id), StageRef("consumer.plink", consumer.id)],
        tensor_connections=(TensorConnection("acc", "org.producer:published", "org.consumer:received"),),
    )
    handle = Handle("pipelink_tensor_scratch", "base", "bind", "smem", None, "claim:adapter:scratch")
    claim = Claim("smem", handle.name, "kernel", Extent(32, {}, {}), 16, 0, None, None, None, None, None)
    barrier = Handle("pipelink_tensor_barrier", "base", "bind", "named_barrier", None, "claim:adapter:barrier")
    barrier_claim = replace(claim, domain="named_barrier", handle=barrier.name, extent=Extent(1, {}, {}), align=1)
    candidate = plan.TensorPlan(
        "MODEL_SAT", plan.Admission("OK", (), {"nonvacuous": True}), "c" * 64,
        ir.Program((), {}, {"load": ir.ControlGraph((ir.Return(),))}, ()),
        (claim, barrier_claim), (handle, barrier),
        ({"handle": barrier.name, "participants": [0], "borrowed": False},), None,
        {"contract_digests": {"export": "e" * 64, "import": "i" * 64},
         "model_digest": "m" * 64, "mode": "controller",
         "resource_demands": {},
         "classes": [{"name": "direct_receipt", "status": "MODEL_SAT", "scope": {"K": 4, "H": 1}}]},
    )
    imp = tensor.imports[0]
    adapter = lower.AdapterSource(
        "adapter_acc", imp.callable.declaration.rstrip(";") + " { dest[0] = 1.0f; }\n",
        (imp.callable.symbol, *imp.callable.events.values()), ({"kind": "test-trusted-linkage"},),
        {"declaration_digest": imp.callable.declaration_digest},
    )
    return pipeline, [producer, consumer], candidate, adapter


def _fanout_composition():
    pipeline, stages, candidate, adapter = _composition()
    tensor = stages[1].tensor
    imp = tensor.imports[0]
    symbol = f"{imp.callable.symbol}_b"
    second = replace(
        imp, name="received_b", sites=("load_b",),
        lends=tuple(replace(lend, name=f"{lend.name}_b") for lend in imp.lends),
        callable=replace(imp.callable, symbol=symbol,
                         declaration=imp.callable.declaration.replace(imp.callable.symbol, symbol),
                         events={"ready": symbol + "_ready", "done": symbol + "_done"}),
    )
    site = replace(tensor.sites[0], name="load_b", participants=Participants(set=(1,)))
    stages[1] = replace(
        stages[1], tensor=replace(
            tensor, imports=(imp, second), sites=(*tensor.sites, site),
        ),
        symbols=_symbol_table(undefined=(imp.callable.symbol, symbol)),
    )
    pipeline = replace(pipeline, tensor_connections=(
        *pipeline.tensor_connections,
        TensorConnection("acc_b", "org.producer:published", "org.consumer:received_b"),
    ))
    return pipeline, stages, replace(candidate, claims=(), handles=(), barrier_requests=()), adapter


def _native_tools(monkeypatch):
    commands = []
    links = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[0].endswith("/nvcc"):
            Path(command[-1]).write_bytes(Path(command[-3]).read_bytes())
        return SimpleNamespace(
            returncode=0,
            stdout="Function pipelink_kernel:\n REG:72 STACK:0 LOCAL:0\n"
            if "--dump-resource-usage" in command else "",
        )

    class NvJitLink:
        def __init__(self, _library):
            pass

        def link(self, inputs, options):
            links.append((inputs, options))
            return b"linked-cubin", "", ""

    monkeypatch.setenv("PIPEC_CUDA", "/cuda")
    monkeypatch.setenv("CUTLASS_INCLUDE_DIR", "/cutlass/include")
    monkeypatch.setenv("TILELANG_INCLUDE_DIR", "/tilelang/include")
    monkeypatch.delenv("PIPEC_SASS_REPORT", raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(link_module, "_NvJitLink", NvJitLink)
    return commands, links


class _Planner:
    """Scripted candidate sequence standing in for ConnectionPlanner."""

    def __init__(self, plans):
        self.plans = list(plans)
        self.rejections = []
        self.current = None

    def next(self):
        if not self.plans:
            self.current = replace(self.current, status="LOWERING_UNSUPPORTED", ir_digest=None, program=None,
                                   claims=(), handles=(), barrier_requests=())
            return self.current
        plan = self.plans.pop(0)
        if self.current is not None:
            history = self.current.provenance.get("classes", [])
            classes = [*history, *plan.provenance.get("classes", [])[len(history):]]
            plan = replace(plan, provenance={**plan.provenance, "classes": classes})
        self.current = plan
        return plan

    def reject_lowering(self, detail):
        self.rejections.append(detail)
        classes = [dict(item) for item in self.current.provenance.get("classes", [])]
        if classes:
            classes[-1]["lowering"] = {"status": "LOWERING_UNSUPPORTED", "backend": detail}
        self.current = replace(self.current, provenance={**self.current.provenance, "classes": classes})


def _install(monkeypatch, pipeline, stages, candidate, adapter):
    for stage in stages:
        if stage.tensor is not None:
            for imp in stage.tensor.imports:
                for resource in imp.lends:
                    candidate.provenance["resource_demands"].setdefault(resource.name, resource.extent)
    artifacts = [link_module._Artifact(Path(f"{stage.id}.plink"), stage, stage.id.encode(), "a" * 64)
                 for stage in stages]
    monkeypatch.setattr(link_module, "_load_assembly", lambda _path: (pipeline, artifacts))
    monkeypatch.setattr(plan, "ConnectionPlanner", lambda *_args: _Planner([candidate]))
    solved = replace(_solution(), native_bases={handle.name: 16 if handle.domain == "smem" else 1
                                              for handle in candidate.handles})
    solves = []

    def solve(composition, owners, *_args):
        solves.append((composition, owners))
        return solved

    monkeypatch.setattr(link_module, "_solve_pipeline", solve)
    monkeypatch.setattr(lower, "emit_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(link_module, "_generate_stub", lambda *_args: "// stub\n")
    monkeypatch.setattr(link_module, "_generate_dispatch", lambda *_args: "// dispatch\n")
    return solves, solved


def _install_fanout(monkeypatch, pipeline, stages, candidate, adapter):
    solves, _solved = _install(monkeypatch, pipeline, stages, candidate, adapter)
    planned, emitted = [], []

    def planner(_pipeline, _stages, connection, _profile):
        planned.append(connection.name)
        return _Planner([replace(candidate, provenance={
            **candidate.provenance,
            "adapter_owner": f"{stages[1].id}.tensor.{connection.name}",
            "resource_demands": {resource.name: resource.extent
                                 for imp in stages[1].tensor.imports if imp.name == connection.import_.rsplit(":", 1)[1]
                                 for resource in imp.lends},
        })])

    def emit(_candidate, imp, _solution, _profile, **_kwargs):
        emitted.append(imp.name)
        return replace(
            adapter, name=f"adapter_{imp.name}",
            source=imp.callable.declaration.rstrip(";") + " { dest[0] = 1.0f; }\n",
            defined_symbols=(imp.callable.symbol, *imp.callable.events.values()),
            resources={"declaration_digest": imp.callable.declaration_digest},
        )

    monkeypatch.setattr(plan, "ConnectionPlanner", planner)
    monkeypatch.setattr(lower, "emit_adapter", emit)
    return planned, emitted, solves


def test_tensor_link_composes_resources_and_links_the_generated_provider(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    commands, links = _native_tools(monkeypatch)
    solves, solved = _install(monkeypatch, pipeline, stages, candidate, adapter)
    lower_solutions = []

    def emit(_candidate, _imp, solution, _profile, **_kwargs):
        lower_solutions.append(solution)
        return adapter

    monkeypatch.setattr(lower, "emit_adapter", emit)
    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert len(solves) == 1
    composition, owners = solves[0]
    assert set(composition.handles) == {handle.name for handle in candidate.handles}
    assert owners[0].claims == []
    assert owners[1].claims == list(candidate.claims)
    assert pipeline.handles == {} and stages[1].claims == []
    assert lower_solutions == [solved]
    adapter_command = next(command for command in commands if "tensor_adapter_000.cu" in command[-3])
    assert "-gencode=arch=compute_103a,code=lto_103a" in adapter_command
    assert "-arch=lto_103" not in adapter_command
    assert "-dc" in adapter_command
    assert {arg for arg in adapter_command if arg.startswith("-D")} == {"-DNDEBUG"}
    assert "/cutlass/include" in adapter_command and "/tilelang/include" in adapter_command
    assert [name for name, _data in links[0][0]] == [
        "000-org.producer.o", "001-org.consumer.o", "binding_stub.o", "dispatch.o", "tensor_adapter_000.o",
    ]
    assert links[0][0][-1][1] == adapter.source.encode()
    record = result.tensor["connections"][0]
    assert record["status"] == record["solver_outcome"]["status"] == "MODEL_SAT"
    assert record["solver_outcome"]["resource_solver"]["solution"]["native_bases"] == solved.native_bases
    assert record["lowering"]["status"] == "LOWERED"
    assert record["native"]["status"] == "DEPLOYED"
    assert record["native"]["observed"] == [72, 0, 0]
    assert record["trusted_premises"] == list(adapter.trusted_premises)
    assert record["bounds"] == [{"class": "direct_receipt", "scope": {"K": 4, "H": 1}}]
    assert json.loads((tmp_path / "out/link.json").read_text())["tensor"] == result.tensor
    assert "tensor_adapter_000.cu" in result.generated_hashes


def test_independent_tensor_fanout_links_both_adapters(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _fanout_composition()
    commands, links = _native_tools(monkeypatch)
    planned, emitted, solves = _install_fanout(monkeypatch, pipeline, stages, candidate, adapter)
    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert planned == ["acc", "acc_b"]
    assert emitted == ["received", "received_b"]
    assert len(solves) == 1
    assert [record["native"]["status"] for record in result.tensor["connections"]] == [
        "DEPLOYED", "DEPLOYED",
    ]
    assert {Path(command[-3]).name for command in commands if command[0].endswith("/nvcc")} == {
        "binding_stub.cu", "dispatch.cu", "tensor_adapter_000.cu", "tensor_adapter_001.cu",
    }
    assert [name for name, _data in links[0][0]] == [
        "000-org.producer.o", "001-org.consumer.o", "binding_stub.o", "dispatch.o",
        "tensor_adapter_000.o", "tensor_adapter_001.o",
    ]
    assert (tmp_path / "out/pipeline.cubin").read_bytes() == b"linked-cubin"


@pytest.mark.parametrize("conflict", ["early_reclaim", "participants"])
def test_tensor_fanout_rejects_dependent_readers_before_planning(tmp_path, monkeypatch, conflict):
    pipeline, stages, candidate, adapter = _fanout_composition()
    tensor = stages[1].tensor
    if conflict == "early_reclaim":
        export = stages[0].tensor.exports[0]
        stages[0] = replace(stages[0], tensor=replace(stages[0].tensor, exports=(
            replace(export, readiness=replace(export.readiness, reuse_after="reclaimed")),)))
        detail = {"early_reclaim": True, "reclaiming": ["acc", "acc_b"]}
    else:
        tensor = replace(tensor, sites=tuple(
            replace(site, participants=Participants(set=(0,))) for site in tensor.sites
        ))
        detail = {"participants": [0]}
    stages[1] = replace(stages[1], tensor=tensor)
    pipeline = replace(pipeline, tensor_connections=tuple(reversed(pipeline.tensor_connections)))
    planned, emitted, solves = _install_fanout(monkeypatch, pipeline, stages, candidate, adapter)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("T-CONTRACT", "pipeline.tensor_connections.acc_b"),
    ]
    assert caught.value.failures[0].detail == {
        "export": "org.producer:published", "sibling": "acc", **detail,
        "tensor": {"connections": []},
    }
    assert planned == emitted == solves == []
    assert not (tmp_path / "out/pipeline.cubin").exists()


@pytest.mark.parametrize("borrowed", [False, True])
def test_tensor_fanout_rejects_overlapping_solved_storage(tmp_path, monkeypatch, borrowed):
    pipeline, stages, candidate, adapter = _fanout_composition()
    tensor = stages[1].tensor
    resources = tuple(
        Lend(
            name, "smem", 32, 16, "cta", "fixed",
            ResourceLocation(f"pipelink_{name}" if borrowed else None, None if borrowed else 0),
            stages[1].id, {"from": "ready", "to": "done"}, None,
        )
        for name in ("scratch", "scratch_b")
    )
    stages[1] = replace(stages[1], tensor=replace(
        tensor, imports=tuple(replace(imp, lends=(resource,))
                              for imp, resource in zip(tensor.imports, resources)),
    ))
    solution = _solution()
    if borrowed:
        handles = {
            resource.location.handle: Handle(
                resource.location.handle, "base", "bind", "smem", None, f"claim:consumer:{resource.name}",
            )
            for resource in resources
        }
        pipeline = replace(pipeline, handles=handles)
        stages[1] = replace(stages[1], claims=[
            Claim("smem", handle, "kernel", Extent(32, {}, {}), 16,
                  0, None, None, None, None, "shared_scratch")
            for handle in handles
        ])
        solution = replace(
            solution, native_bases={handle: 0 for handle in handles},
            intervals={"smem": [Interval(handle, 0, 32, "kernel") for handle in handles]},
        )
    planned, emitted, solves = _install_fanout(monkeypatch, pipeline, stages, candidate, adapter)

    def solve(composition, owners, *_args):
        solves.append((composition, owners))
        return solution

    monkeypatch.setattr(link_module, "_solve_pipeline", solve)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("T-CONTRACT", "pipeline.tensor_connections.acc_b"),
    ]
    assert caught.value.failures[0].detail == {
        "export": "org.producer:published", "sibling": "acc", "domain": "smem",
        "resources": ["scratch", "scratch_b"], "intervals": [[0, 32], [0, 32]],
        "tensor": caught.value.tensor,
    }
    assert planned == ["acc", "acc_b"] and len(solves) == 1
    assert [record["solver_outcome"]["reservation_validation"]["phase"]
            for record in caught.value.tensor["connections"]] == ["solved", "solved"]
    assert emitted == []
    assert not (tmp_path / "out/pipeline.cubin").exists()


def test_tensor_fanout_grounds_all_sites_over_resolved_binding_domains():
    pipeline, stages, _candidate, _adapter = _fanout_composition()
    tensor = stages[1].tensor
    lane = expr.Domain("lane", "int", values=(0, 1))
    extra = replace(tensor.sites[0], name="extra", participants=Participants(
        expr=expr.parse(["add", ["bind", "offset"], ["coord", "lane"]],
                        {"offset": expr.Domain("offset", "int", values=(4, 6)), "lane": lane}),
        domain=(lane,),
    ))
    stages[1] = replace(stages[1], tensor=replace(
        tensor, imports=(replace(tensor.imports[0], sites=("load", "extra")), tensor.imports[1]),
        sites=(tensor.sites[0], replace(tensor.sites[1], participants=Participants(set=(5, 4, 2))), extra),
        bindings=(Binding("offset", "int", expr.Domain("offset", "int", values=(4, 6)),
                          stages[1].id, "thread", "cta", "kernel"),),
    ))
    failures = plan.check_fanout(stages, pipeline.tensor_connections, get_profile("sm_103a"))
    assert [(failure.clause_id, failure.path) for failure in failures] == [
        ("T-CONTRACT", "pipeline.tensor_connections.acc_b"),
    ]
    assert failures[0].detail == {
        "export": "org.producer:published", "sibling": "acc", "participants": [4, 5],
    }


@pytest.mark.parametrize("domain,base,expected", [
    ("smem", 32, []),
    ("smem", None, []),
    ("barrier", 16, [("T-CONTRACT", "pipeline.tensor_connections.acc_b")]),
])
def test_tensor_fanout_storage_respects_allocation_boundaries(domain, base, expected):
    pipeline, stages, candidate, _adapter = _fanout_composition()
    tensor = stages[1].tensor
    resource = Lend(
        "scratch", domain, 32, 16, "cta", "fixed", ResourceLocation(None, 0),
        stages[1].id, {"from": "ready", "to": "done"}, None,
    )
    second = replace(resource, name="scratch_b", location=ResourceLocation("pipelink_scratch_b", None))
    stages[1] = replace(stages[1], tensor=replace(
        tensor, imports=(replace(tensor.imports[0], lends=(resource,)),
                         replace(tensor.imports[1], lends=(second,))),
    ))
    plans = {
        connection.name: replace(candidate, provenance={
            "adapter_owner": f"org.consumer.tensor.{connection.name}",
            "resource_demands": {"scratch": 32, "scratch_b": 32}})
        for connection in pipeline.tensor_connections
    }
    solution = replace(_solution(), native_bases={} if base is None else {"pipelink_scratch_b": base})
    failures = plan.check_fanout_storage(stages, pipeline.tensor_connections, plans, solution)
    assert [(failure.clause_id, failure.path) for failure in failures] == expected
    if expected:
        assert failures[0].detail == {
            "export": "org.producer:published", "sibling": "acc", "domain": "named_barrier",
            "resources": ["scratch", "scratch_b"], "intervals": [[0, 32], [16, 48]],
        }


def test_fanout_can_share_readonly_backing_but_not_overwrite_it():
    pipeline, stages, candidate, _adapter = _fanout_composition()
    export = stages[0].tensor.exports[0]
    source = Lend("source", "smem", 32, 16, "cta", "fixed", ResourceLocation(None, 0),
                  stages[0].id, {"from": "ready", "to": export.readiness.reuse_after}, None)
    stages[0] = replace(stages[0], tensor=replace(stages[0].tensor, exports=(
        replace(export, backing=(source,)),)))
    plans = {connection.name: replace(candidate, provenance={"adapter_owner": connection.name,
                                                            "resource_demands": {"scratch": 32}})
             for connection in pipeline.tensor_connections}
    assert plan.check_fanout_storage(stages, pipeline.tensor_connections, plans, _solution()) == ()
    first, second = stages[1].tensor.imports
    scratch = replace(source, name="scratch", owner=stages[1].id)
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=(
        first, replace(second, lends=(*second.lends, scratch)))))
    failures = plan.check_fanout_storage(stages, pipeline.tensor_connections, plans, _solution())
    assert len(failures) == 1
    assert failures[0].clause_id == "T-CONTRACT"
    assert failures[0].detail["resources"] == ["source", "scratch"]


def test_connection_rejects_binding_declared_by_both_stages():
    pipeline, stages, _candidate, _adapter = _composition()
    domain = expr.Domain("base", "int", values=(0, 4))
    stages = [
        replace(stage, tensor=replace(stage.tensor, bindings=(
            Binding("base", "int", domain, stage.id, "byte", "cta", "kernel"),)))
        for stage in stages
    ]
    result = plan.ConnectionPlanner(pipeline, stages, pipeline.tensor_connections[0], get_profile("sm_103a")).next()
    assert result.status == "CONTRACT_INVALID"
    assert result.program is None
    assert {failure.clause_id for failure in result.admission.issues} == {"T-CONTRACT"}
    assert "declared by both stages" in result.admission.issues[0].message


def test_fixed_binding_overlay_uses_partner_declaration_and_rejects_outside_domain():
    pipeline, stages, _candidate, _adapter = _composition()
    domain = expr.Domain("base", "bits[32]", values=(0, 4))
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, bindings=(
        Binding("base", domain.sort, domain, stages[1].id, "byte", "cta", "kernel"),)))
    export = replace(stages[0].tensor.exports[0], fixed_bindings={"base": 4})
    derived = plan.derive(export, stages[1].tensor.imports[0], stages[1].tensor.sites,
                          stages[1].tensor.bindings, get_profile("sm_103a"))
    assert derived.binding_domains["base"] == expr.Domain("base", "bits[32]", values=(4,))
    stages[0] = replace(stages[0], tensor=replace(stages[0].tensor, exports=(
        replace(export, fixed_bindings={"base": 2}),)))
    result = plan.ConnectionPlanner(pipeline, stages, pipeline.tensor_connections[0], get_profile("sm_103a")).next()
    assert result.status == "CONTRACT_INVALID"
    assert {failure.clause_id for failure in result.admission.issues} == {"T-CONTRACT"}


@pytest.mark.parametrize("status,clause", [
    ("CONTRACT_INVALID", "T-CONTRACT"), ("UNSUPPORTED_MODEL", "T-MODEL"),
    ("BOUNDED_UNSAT", "T-PLAN"), ("UNKNOWN", "T-PLAN"),
])
def test_tensor_planning_failure_records_outcome_before_aborting(tmp_path, monkeypatch, status, clause):
    pipeline, stages, candidate, adapter = _composition()
    admission = plan.Admission(status, (ClauseFailure(clause, "tensor.admission", "not admitted", {}),), {}) \
        if status in {"CONTRACT_INVALID", "UNSUPPORTED_MODEL"} else candidate.admission
    candidate = replace(candidate, status=status, admission=admission)
    solves, _ = _install(monkeypatch, pipeline, stages, candidate, adapter)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [failure.clause_id for failure in caught.value.failures] == [clause]
    record = caught.value.tensor["connections"][0]
    assert record["status"] == record["solver_outcome"]["status"] == status
    assert caught.value.failures[0].detail["tensor"] == caught.value.tensor
    assert solves == []
    assert not (tmp_path / "out/link.json").exists()


def test_lowering_unsupported_with_no_further_candidate_records_the_history(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    _install(monkeypatch, pipeline, stages, candidate, adapter)

    def unsupported(*_args, **_kwargs):
        raise lower.LoweringUnsupported("load:1:shfl_xor", "no_recipe", "TileLang has no matching recipe")

    monkeypatch.setattr(lower, "emit_adapter", unsupported)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [failure.clause_id for failure in caught.value.failures] == ["T-LOWER"]
    record = caught.value.tensor["connections"][0]
    assert record["status"] == record["solver_outcome"]["status"] == "LOWERING_UNSUPPORTED"
    assert record["lowering"]["status"] == "LOWERING_UNSUPPORTED"
    assert record["lowering"]["reason"] == "no_recipe"
    classes = record["solver_outcome"]["provenance"]["classes"]
    assert classes[0]["status"] == "MODEL_SAT" and classes[0]["lowering"]["status"] == "LOWERING_UNSUPPORTED"
    assert not (tmp_path / "out/pipeline.cubin").exists()


def test_lowering_rejection_resumes_with_the_next_candidate_and_resolves_again(tmp_path, monkeypatch):
    pipeline, stages, first, adapter = _composition()
    _native_tools(monkeypatch)
    # The second candidate claims a differently sized scratch: the composition and the
    # solution must be rebuilt for it, and the adapter must see the new solution.
    second_claim = replace(first.claims[0], extent=Extent(64, {}, {}))
    second = replace(first, ir_digest="d" * 64, claims=(second_claim, first.claims[1]),
                     provenance={**first.provenance, "classes": [
                         *first.provenance["classes"],
                         {"name": "staged_materialized", "status": "MODEL_SAT", "scope": {"K": 6, "H": 1}}],
                         "selected": 1})
    planner = _Planner([first, second])
    solves, solved = _install(monkeypatch, pipeline, stages, first, adapter)
    monkeypatch.setattr(plan, "ConnectionPlanner", lambda *_args: planner)
    lowered = []

    def emit(candidate, _imp, solution, _profile, **_kwargs):
        lowered.append((candidate.ir_digest, solution))
        if candidate is first:
            raise lower.LoweringUnsupported("load:1:tmem_packet_receipt", "no_recipe", "no recipe for the first route")
        return adapter

    monkeypatch.setattr(lower, "emit_adapter", emit)
    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [digest for digest, _solution in lowered] == [first.ir_digest, second.ir_digest]
    assert [owners[1].claims for _composition, owners in solves] == [list(first.claims), list(second.claims)]
    assert planner.rejections[0]["reason"] == "no_recipe"
    record = result.tensor["connections"][0]
    assert record["status"] == "MODEL_SAT" and record["plan_digest"] == second.ir_digest
    assert record["selected"] == 1
    assert record["solver_outcome"]["claims"] == [claim.to_json_obj() for claim in second.claims]
    classes = record["solver_outcome"]["provenance"]["classes"]
    assert classes[0]["lowering"]["status"] == "LOWERING_UNSUPPORTED" and "lowering" not in classes[1]
    assert record["lowering"]["status"] == "LOWERED" and record["native"]["status"] == "DEPLOYED"


def test_unavailable_lowering_tooling_fails_without_searching_further_routes(tmp_path, monkeypatch):
    pipeline, stages, first, adapter = _composition()
    planner = _Planner([first, replace(first, ir_digest="d" * 64)])
    _install(monkeypatch, pipeline, stages, first, adapter)
    monkeypatch.setattr(plan, "ConnectionPlanner", lambda *_args: planner)
    monkeypatch.delenv("PIPELINK_TILELANG_PYTHON", raising=False)
    lowered = []

    def emit(candidate, *_args, **_kwargs):
        lowered.append(candidate.ir_digest)
        lower._backend({})  # The real backend entry: every route needs the interpreter.

    monkeypatch.setattr(lower, "emit_adapter", emit)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [failure.clause_id for failure in caught.value.failures] == ["T-LOWER"]
    assert lowered == [first.ir_digest] and planner.rejections == []
    assert caught.value.failures[0].detail["backend"]["details"]["kind"] == "missing_interpreter"


def test_missing_adapter_include_root_is_native_failure(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    _commands, links = _native_tools(monkeypatch)
    monkeypatch.delenv("CUTLASS_INCLUDE_DIR")
    _install(monkeypatch, pipeline, stages, candidate, adapter)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [failure.clause_id for failure in caught.value.failures] == ["T-NATIVE"]
    assert caught.value.failures[0].detail["environment_variable"] == "CUTLASS_INCLUDE_DIR"
    assert caught.value.tensor["connections"][0]["status"] == "MODEL_SAT"
    assert links == []


def test_tensor_native_route_rejects_ordinary_lto_target_without_substitution(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    stages[0] = replace(stages[0], target=replace(stages[0].target, lto_arch="lto_103"))
    commands, links = _native_tools(monkeypatch)
    _install(monkeypatch, pipeline, stages, candidate, adapter)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [failure.clause_id for failure in caught.value.failures] == ["T-NATIVE"]
    assert caught.value.failures[0].detail["actual"]["lto_arch"] == "lto_103"
    assert commands == [] and links == []


def test_late_provider_check_prevents_linking_missing_generated_definition(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    _commands, links = _native_tools(monkeypatch)
    _install(monkeypatch, pipeline, stages, candidate, replace(adapter, defined_symbols=()))
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert {failure.clause_id for failure in caught.value.failures} == {"T-LOWER"}
    assert caught.value.tensor["connections"][0]["status"] == "MODEL_SAT"
    assert links == []


def test_tensor_connection_does_not_exempt_mismatched_ordinary_endpoints(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    endpoint = _paged_stage().endpoints[0]
    base = "pipelink_ordinary_base"
    endpoint = replace(endpoint, binds={"base": base})
    claim = replace(_paged_stage().claims[0], domain="smem", handle=base,
                    extent=Extent(128, {}, {}), align=16)
    stages[0] = replace(stages[0], endpoints=[endpoint], claims=[claim])
    mismatch = replace(endpoint, direction="consume", owner_role=None, layout=load_layout(
        {**endpoint.layout.to_json_obj(), "stride": [[1], [8]]}, endpoint.space, "$.layout"
    ))
    stages[1] = replace(stages[1], endpoints=[mismatch], claims=[claim])
    pipeline = replace(pipeline, handles={base: Handle(base, "base", "bind", "smem", None, "claim:ordinary")})
    solves, _ = _install(monkeypatch, pipeline, stages, candidate, adapter)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert {failure.clause_id for failure in caught.value.failures} == {"L8"}
    assert solves == []


def test_tensor_register_rejection_retains_model_and_composed_audit(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    pipeline = replace(pipeline, handles=_register_pipeline().handles)
    _commands, links = _native_tools(monkeypatch)
    _install(monkeypatch, pipeline, stages, candidate, adapter)
    solutions = []

    def solve(composition, owners, *_args):
        solutions.append((composition, owners))
        return _register_solution(96, 96)

    monkeypatch.setattr(link_module, "_solve_pipeline", solve)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [failure.clause_id for failure in caught.value.failures] == ["L-REGS"]
    record = caught.value.tensor["connections"][0]
    assert record["status"] == "MODEL_SAT"
    assert record["native"]["status"] == "NATIVE_RESOURCE_REJECTED"
    assert record["native"]["attempt"] == 2
    assert len(links) == 2
    assert all(inputs[-1][0] == "tensor_adapter_000.o" for inputs, _options in links)
    assert len(solutions) == 2
    assert solutions[1][0].handles["pipelink_launch_regs"].values == [48]
    assert solutions[1][1][1].claims == list(candidate.claims)
    assert {handle.name for handle in candidate.handles} <= solutions[1][0].handles.keys()
    assert not (tmp_path / "out/pipeline.cubin").exists()


@pytest.mark.parametrize("demand", [0, 16])
def test_generated_lends_reserve_only_selected_demand(tmp_path, monkeypatch, demand):
    pipeline, stages, candidate, adapter = _composition()
    scratch = Lend(
        "scratch", "smem", 32, 16, "cta", "relocatable",
        ResourceLocation(candidate.handles[0].name, None), stages[1].id,
        {"from": "ready", "to": "done"}, None,
    )
    fixed = replace(scratch, name="fixed", extent=16, binding="fixed",
                    location=ResourceLocation("pipelink_tensor_fixed", 128))
    unused = replace(scratch, name="unused", extent=16,
                     location=ResourceLocation("pipelink_tensor_unused", None))
    stages[1] = replace(
        stages[1],
        tensor=replace(stages[1].tensor, imports=(
            replace(stages[1].tensor.imports[0], lends=(scratch, fixed, unused)),)),
        symbols=_symbol_table(undefined=(adapter.defined_symbols[0],
                                        scratch.location.handle, fixed.location.handle)),
    )
    candidate = replace(
        candidate, claims=(replace(candidate.claims[0], extent=Extent(demand, {}, {})),) if demand else (),
        handles=candidate.handles[:1] if demand else (), barrier_requests=(),
        provenance={**candidate.provenance, "resource_demands": {"scratch": demand, "fixed": 0, "unused": 0}},
    )
    real_stub = link_module._generate_stub
    _commands, links = _native_tools(monkeypatch)
    _install(monkeypatch, pipeline, stages, candidate, adapter)
    monkeypatch.setattr(link_module, "_generate_stub", real_stub)
    native_bases = {
        scratch.location.handle: 16 if demand else 0,
        fixed.location.handle: 128,
        unused.location.handle: 0,
    }
    compositions = []

    def solve(composition, owners, *_args):
        compositions.append((composition, owners))
        return replace(_solution(), native_bases=native_bases, intervals={"smem": [
            Interval(claim.handle, native_bases[claim.handle],
                     native_bases[claim.handle] + claim.extent.const, "kernel")
            for owner in owners for claim in owner.claims]})

    monkeypatch.setattr(link_module, "_solve_pipeline", solve)
    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    composition, owners = compositions[0]
    assert set(composition.handles) == set(native_bases)
    assert len(owners[1].claims) == 1 + bool(demand)
    scratch_claims = [claim for claim in owners[1].claims if claim.handle == scratch.location.handle]
    assert [claim.extent.const for claim in scratch_claims] == ([demand] if demand else [])
    assert next(claim for claim in owners[1].claims
                if claim.handle == fixed.location.handle).absolute_base == 128
    stub = dict(links[0][0])["binding_stub.o"].decode()
    assert f'int pipelink_tensor_scratch() {{ return {16 if demand else 0}; }}' in stub
    assert 'int pipelink_tensor_fixed() { return 128; }' in stub
    assert unused.location.handle not in stub
    assert result.bindings[scratch.location.handle] == ((16, "solved") if demand else (0, "forced"))
    assert result.bindings[fixed.location.handle] == (128, "solved")
    assert pipeline.handles == {} and stages[1].claims == []


@pytest.mark.parametrize("extent,align,lifetime", [
    (64, 16, "kernel"),
    (128, 8, "kernel"),
    (128, 16, "phase:prepare"),
])
def test_borrowed_symbolic_scratch_requires_covering_reservation(
    tmp_path, monkeypatch, extent, align, lifetime
):
    pipeline, stages, candidate, adapter = _composition()
    handle = candidate.handles[0]
    scratch = Lend(
        "scratch", "smem", 128, 16, "cta", "fixed", ResourceLocation(handle.name, None),
        stages[1].id, {"from": "ready", "to": "done"}, None,
    )
    backing = replace(candidate.claims[0], extent=Extent(extent, {}, {}),
                      align=align, lifetime=lifetime)
    pipeline = replace(pipeline, handles={handle.name: handle}, phases=["prepare"])
    stages[1] = replace(stages[1], claims=[backing],
                        tensor=replace(stages[1].tensor, imports=(
                            replace(stages[1].tensor.imports[0], lends=(scratch,)),)))
    candidate = replace(candidate, claims=(), handles=(), barrier_requests=())
    solves, _ = _install(monkeypatch, pipeline, stages, candidate, adapter)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert {failure.clause_id for failure in caught.value.failures} == {"T-PLAN"}
    assert caught.value.tensor["connections"][0]["status"] == "MODEL_SAT"
    assert caught.value.failures[0].detail["resource"]["extent"] == 128
    assert solves == []
    assert stages[1].claims == [backing]


def test_larger_borrowed_scratch_is_not_reallocated(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    handle = candidate.handles[0]
    scratch = Lend(
        "scratch", "smem", 128, 16, "cta", "fixed", ResourceLocation(handle.name, None),
        stages[1].id, {"from": "ready", "to": "done"}, None,
    )
    backing = replace(candidate.claims[0], extent=Extent(256, {}, {}), align=32)
    pipeline = replace(pipeline, handles={handle.name: handle})
    stages[1] = replace(stages[1], claims=[backing],
                        tensor=replace(stages[1].tensor, imports=(
                            replace(stages[1].tensor.imports[0], lends=(scratch,)),)))
    candidate = replace(candidate, claims=(), handles=(), barrier_requests=())
    _native_tools(monkeypatch)
    _install(monkeypatch, pipeline, stages, candidate, adapter)
    solution = replace(_solution(), native_bases={handle.name: 128},
                       intervals={"smem": [Interval(handle.name, 128, 384, "kernel")]})
    compositions = []

    def solve(composition, owners, *_args):
        compositions.append((composition, owners))
        return solution

    monkeypatch.setattr(link_module, "_solve_pipeline", solve)
    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert result.bindings[handle.name] == (128, "solved")
    assert compositions[0][1][1].claims == [backing]
    assert compositions[0][0].handles == pipeline.handles


def test_reserved_zero_barrier_rejects_nonzero_solved_handle(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    handle = candidate.handles[1]
    barrier = Lend(
        "barrier", "barrier", 1, 1, "cta", "fixed", ResourceLocation(handle.name, 0),
        stages[1].id, {"from": "ready", "to": "done"}, None,
    )
    backing = candidate.claims[1]
    pipeline = replace(pipeline, handles={handle.name: handle})
    stages[1] = replace(stages[1], claims=[backing],
                        tensor=replace(stages[1].tensor, imports=(
                            replace(stages[1].tensor.imports[0], lends=(barrier,)),)))
    candidate = replace(
        candidate, claims=(), handles=(),
        barrier_requests=({"handle": handle.name, "resource": barrier.name, "reserved": 0},),
    )
    _install(monkeypatch, pipeline, stages, candidate, adapter)
    solution = replace(_solution(), native_bases={handle.name: 1},
                       intervals={"named_barrier": [Interval(handle.name, 1, 2, "kernel")]})
    monkeypatch.setattr(link_module, "_solve_pipeline", lambda *_args: solution)
    emitted = []
    monkeypatch.setattr(lower, "emit_adapter", lambda *_args, **_kwargs: emitted.append(True) or adapter)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert {failure.clause_id for failure in caught.value.failures} == {"T-PLAN"}
    assert caught.value.tensor["connections"][0]["status"] == "MODEL_SAT"
    assert caught.value.failures[0].detail["declared_base"] == 0
    assert caught.value.failures[0].detail["solved_base"] == 1
    assert emitted == []


def test_symbolic_borrow_extent_is_checked_after_resource_solving(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _composition()
    handle = candidate.handles[0]
    depth = Handle("pipelink_scratch_depth", "depth", "bind", "smem", [1], "scratch_depth")
    scratch = Lend(
        "scratch", "smem", 128, 16, "cta", "fixed", ResourceLocation(handle.name, None),
        stages[1].id, {"from": "ready", "to": "done"}, None,
    )
    backing = replace(candidate.claims[0], extent=Extent(0, {depth.name: 64}, {}))
    pipeline = replace(pipeline, handles={handle.name: handle, depth.name: depth})
    stages[1] = replace(stages[1], claims=[backing],
                        tensor=replace(stages[1].tensor, imports=(
                            replace(stages[1].tensor.imports[0], lends=(scratch,)),)))
    candidate = replace(candidate, claims=(), handles=(), barrier_requests=())
    _install(monkeypatch, pipeline, stages, candidate, adapter)
    solution = replace(
        _solution(), depths={depth.name: 1}, native_bases={handle.name: 0},
        intervals={"smem": [Interval(handle.name, 0, 64, "kernel")]},
    )
    monkeypatch.setattr(link_module, "_solve_pipeline", lambda *_args: solution)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert {failure.clause_id for failure in caught.value.failures} == {"T-PLAN"}
    assert caught.value.tensor["connections"][0]["solver_outcome"]["resource_solver"]["status"] == "SAT"
    assert caught.value.failures[0].detail["solved_intervals"][0]["hi"] == 64


@pytest.mark.parametrize("profile_name", ["sm_103a", "sm_90a", "sm_120a"])
@pytest.mark.parametrize("space,threads,columns", [("smem", 2, 2), ("reg", 128, 32)])
def test_pointer_export_round_plans_checks_and_emits(monkeypatch, space, threads, columns, profile_name):
    from test_tensor_lower import _backend, _kernel
    from test_tensor_synth import _pointer_fixture, _pointer_plan

    profile = get_profile(profile_name)
    planned, imp = _pointer_plan(_pointer_fixture(space, threads=threads, columns=columns), profile)
    assert planned.status == "MODEL_SAT"
    issues = [node for graph in planned.program.graphs.values() for node in graph.nodes if isinstance(node, ir.Issue)]
    assert len(issues) == 1
    assert issues[0].primitive == ("reg_copy_round" if space == "reg" else "smem_load_round")
    assert planned.provenance["admission_oracle"]["ok"]
    index = "slot" if space == "reg" else f"{columns + 4} * (int)threadIdx.x + slot"
    calls = _backend(monkeypatch, _kernel(f"  for (int slot = 0; slot < {columns}; ++slot) dest[slot] = source_base[{index}];\n",
                                          parameters="const float* source_base, float* dest", threads=threads))
    emitted = lower.emit_adapter(planned, imp, _solution(threads=threads), profile)
    assert calls[0][1]["target"] == profile_name
    bound = emitted.resources["bindings"][f"@export:{space}"]
    assert bound["domain"] == space and bound["param"] == "source_base"
    assert bound["extent"] == (columns if space == "reg" else ((threads - 1) * (columns + 4) + columns) * 4)
    assert not planned.program.borrows


def test_one_warp_of_larger_cta_plans_and_prepares_full_fragment():
    from pipec2.profile import get_profile
    from test_tensor_synth import _pointer_fixture, _pointer_plan

    raw = _pointer_fixture("smem", threads=32, columns=32, participant_start=128, rows_per_thread=4)
    planned, imp = _pointer_plan(raw)
    assert planned.status == "MODEL_SAT"
    request = lower._prepare(planned, imp, _solution(threads=160), get_profile("sm_103a"))
    assert request["context"]["participants"]["load"] == list(range(128, 160))
    assert request["resources"]["dest"]["extent"] == 128
    assert [step["primitive"] for step in request["graphs"]["load"]] == ["smem_load_round"]


def test_constructed_transport_lowers_without_entering_the_solver(monkeypatch):
    from pipec2.profile import get_profile
    from pipec2.tensor import synth
    from test_tensor_lower import _backend, _kernel
    from test_tensor_synth import _pointer_fixture, _pointer_plan

    def forbidden_solver(*_args, **_kwargs):
        raise AssertionError("constructible source must not enter bounded Z3 search")

    monkeypatch.setattr(synth, "synthesize", forbidden_solver)
    planned, imp = _pointer_plan(_pointer_fixture("reg", threads=2, columns=3))
    assert planned.status == "MODEL_SAT"
    assert planned.provenance["generator"] == "constructive-v1"
    _backend(monkeypatch, _kernel("  for (int i = 0; i < 3; ++i) dest[i] = source_base[i];\n",
                                  parameters="const float* source_base, float* dest", threads=2))
    emitted = lower.emit_adapter(planned, imp, _solution(threads=2), get_profile("sm_103a"))
    assert set(emitted.defined_symbols) == {imp.callable.symbol, *imp.callable.events.values()}
    assert planned.provenance["admission_oracle"]["checked_positions"] == 6


def test_relowering_never_keeps_a_sibling_lowering_from_the_previous_solution(tmp_path, monkeypatch):
    pipeline, stages, candidate, adapter = _fanout_composition()
    _install_fanout(monkeypatch, pipeline, stages, candidate, adapter)
    second = replace(candidate, ir_digest="d" * 64, provenance={
        **candidate.provenance, "adapter_owner": f"{stages[1].id}.tensor.acc_b",
        "classes": [*candidate.provenance["classes"],
                    {"name": "staged_materialized", "status": "MODEL_SAT", "scope": {"K": 6, "H": 1}}],
        "selected": 1})
    planners = {"acc": _Planner([replace(candidate, provenance={
        **candidate.provenance, "adapter_owner": f"{stages[1].id}.tensor.acc"})]),
                "acc_b": _Planner([replace(candidate, provenance={
                    **candidate.provenance, "adapter_owner": f"{stages[1].id}.tensor.acc_b"}), second])}
    monkeypatch.setattr(plan, "ConnectionPlanner", lambda _p, _s, connection, _f: planners[connection.name])
    solves = []

    def solve(composition, owners, *_args):
        solves.append(composition)
        if len(solves) == 2:
            raise LinkError([ClauseFailure("T-PLAN", "pipeline", "no room on the second composition", {})])
        return replace(_solution(), native_bases={})

    monkeypatch.setattr(link_module, "_solve_pipeline", solve)

    def emit(candidate_plan, imp, _solution, _profile, **_kwargs):
        if imp.name == "received_b" and candidate_plan.ir_digest != second.ir_digest:
            raise lower.LoweringUnsupported("load_b:0", "no_recipe", "first route of acc_b has no recipe")
        return replace(adapter, name=f"adapter_{imp.name}", defined_symbols=(imp.callable.symbol, *imp.callable.events.values()),
                       source=imp.callable.declaration.rstrip(";") + " { dest[0] = 1.0f; }\n",
                       resources={"declaration_digest": imp.callable.declaration_digest})

    monkeypatch.setattr(lower, "emit_adapter", emit)
    with pytest.raises(LinkError) as caught:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert len(solves) == 2
    records = {record["name"]: record for record in caught.value.tensor["connections"]}
    # `acc` lowered against the first solution; that evidence is void once `acc_b` forced a new solve.
    assert records["acc"]["lowering"] is None and records["acc"]["native"] is None
    assert records["acc"]["trusted_premises"] == []
    assert records["acc"]["solver_outcome"]["resource_solver"]["status"] == "REJECTED"
    assert records["acc_b"]["plan_digest"] == second.ir_digest and records["acc_b"]["selected"] == 1
    assert records["acc_b"]["solver_outcome"]["provenance"]["classes"][0]["lowering"]["status"] == "LOWERING_UNSUPPORTED"


@pytest.mark.parametrize("start,accepted", [("ready", True), ("delivered", False)])
def test_borrowed_event_reservation_covers_lend(tmp_path, monkeypatch, start, accepted):
    from pipec2.model import TensorLifetime
    from pipec2.tensor.contracts import Rendezvous

    pipeline, stages, candidate, adapter = _composition()
    handle = candidate.handles[0]
    imp = stages[1].tensor.imports[0]
    scratch = Lend(
        "scratch", "smem", 32, 16, "cta", "fixed", ResourceLocation(handle.name, None),
        stages[1].id, {"from": "ready", "to": imp.completion}, None,
    )
    claim = replace(candidate.claims[0], lifetime=TensorLifetime(imp.name, start))
    site = replace(stages[1].tensor.sites[0], guarantees=("enter_once", "single_completion"))
    imp = replace(imp, lends=(scratch,), rendezvous=(
        Rendezvous("ready", "outer", Participants(set=(0,))),
        Rendezvous(imp.completion, "outer", Participants(set=(0,))),
    ))
    stages[1] = replace(stages[1], claims=[claim],
                        tensor=replace(stages[1].tensor, imports=(imp,), sites=(site,)))
    pipeline = replace(pipeline, handles={handle.name: handle})
    candidate = replace(candidate, claims=(), handles=(), barrier_requests=())
    _native_tools(monkeypatch)
    _install(monkeypatch, pipeline, stages, candidate, adapter)
    solution = replace(_solution(), native_bases={handle.name: 0},
                       intervals={"smem": [Interval(handle.name, 0, 32, "tensor:received:ready..end")]})
    monkeypatch.setattr(link_module, "_solve_pipeline", lambda *_args: solution)
    if accepted:
        result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
        assert result.bindings[handle.name] == (0, "solved")
    else:
        with pytest.raises(LinkError) as caught:
            link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
        assert {failure.clause_id for failure in caught.value.failures} == {"T-PLAN"}


def test_generated_handle_shared_by_connections_keeps_kernel_extent():
    from pipec2.model import TensorLifetime

    pipeline, stages, candidate, _adapter = _fanout_composition()
    tensor = stages[1].tensor
    lends = [
        Lend(f"scratch_{index}", "smem", 64, 16, "cta", "relocatable",
             ResourceLocation("pipelink_shared_scratch", None), stages[1].id,
             {"from": "ready", "to": imp.completion}, None)
        for index, imp in enumerate(tensor.imports)
    ]
    imports = tuple(replace(imp, lends=(lend,)) for imp, lend in zip(tensor.imports, lends))
    stages[1] = replace(stages[1], tensor=replace(tensor, imports=imports))
    planned = []
    for index, (connection, imp, lend) in enumerate(zip(pipeline.tensor_connections, imports, lends)):
        size = 16 * (index + 1)
        claim = Claim("smem", lend.location.handle,
                      TensorLifetime(imp.name, "ready", imp.completion),
                      Extent(size, {}, {}), 16, 0, None, None, None, None, None)
        selected = replace(candidate, claims=(claim,), handles=(),
                           provenance={**candidate.provenance, "resource_demands": {lend.name: size}})
        planned.append(link_module._TensorConnectionPlan(
            stages[1].id, imp, selected,
            link_module._plan_record(connection, selected, get_profile(pipeline.target.profile)),
            connection, _Planner([selected]),
        ))
    _pipeline_out, owners = link_module._tensor_composition(pipeline, stages, planned)
    shared = next(claim for claim in owners[1].claims if claim.handle == "pipelink_shared_scratch")
    assert shared.lifetime == "kernel"
    assert shared.extent.const == 32


def test_relowering_recomputes_selected_overlaps_and_fence_witnesses(tmp_path, monkeypatch):
    from pipec2.model import TensorLifetime
    from pipec2.tensor.contracts import Rendezvous

    pipeline, stages, candidate, adapter = _composition()
    imp = stages[1].tensor.imports[0]
    ring = replace(candidate.claims[0], handle="pipelink_ring",
                   lifetime=TensorLifetime("published", "work", "ready"))
    ring_handle = replace(candidate.handles[0], name=ring.handle,
                          owner=f"claim:{stages[0].id}:{ring.handle}")
    stages[0] = replace(stages[0], claims=[ring])
    site = replace(stages[1].tensor.sites[0], guarantees=("enter_once", "single_completion"))
    imp = replace(imp, rendezvous=(Rendezvous("ready", "outer", Participants(set=(0,))),))
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=(imp,), sites=(site,)))
    pipeline = replace(pipeline, handles={ring.handle: ring_handle})
    scratch = replace(candidate.claims[0], lifetime=TensorLifetime(imp.name, "ready", imp.completion))
    first = replace(candidate, claims=(scratch,), handles=candidate.handles[:1], barrier_requests=())
    second = replace(first, ir_digest="d" * 64)
    _native_tools(monkeypatch)
    _install(monkeypatch, pipeline, stages, first, adapter)
    planner = _Planner([first, second])
    monkeypatch.setattr(plan, "ConnectionPlanner", lambda *_args: planner)
    solutions = iter([
        replace(_solution(), native_bases={ring.handle: 0, scratch.handle: base},
                intervals={"smem": [Interval(ring.handle, 0, 32, ring.lifetime.label()),
                                    Interval(scratch.handle, base, base + 32, scratch.lifetime.label())]})
        for base in (0, 32)
    ])
    monkeypatch.setattr(link_module, "_solve_pipeline", lambda *_args: next(solutions))
    witnesses = []

    def emit(selected, _imp, _solution, _profile, *, witness_events):
        witnesses.append(witness_events)
        if selected is first:
            raise lower.LoweringUnsupported("load:0", "no_recipe", "first route cannot lower")
        return adapter

    monkeypatch.setattr(lower, "emit_adapter", emit)
    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert witnesses == [frozenset({"ready"}), frozenset()]
    assert result.tensor["smem_reuse"]["overlaps"] == []
    assert not any(premise.get("name", "").startswith("claim_pipelink_ring_quiescent")
                   for premise in result.tensor["connections"][0]["trusted_premises"])
