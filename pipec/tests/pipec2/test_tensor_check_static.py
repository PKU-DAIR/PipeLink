from __future__ import annotations

from dataclasses import replace

import pytest

from pipec2.tensor import expr, ir, plan
from pipec2.tensor.check import _check_product, check
from pipec2.tensor.contracts import (
    CallableParam, DemandView, Identity, Lend, Participants, PlacementView, Readiness,
    ResourceLocation, ResourceSync, Site, TensorCallable, TensorExport, TensorImport,
)
from pipec2.tensor.machine import Instance, Monitor
from test_tensor_machine import _binding, _profile, _rebuild


def _instance(*, tmem=False, ready=True, worlds=False, transpose=True):
    row, column = expr.Domain("row", "int", 0, 4), expr.Domain("column", "int", 0, 1)
    domains = {"row": row, "column": column}
    identity = Identity("values", "tile", "complete", (row, column), "f32", "ieee754")
    participant = expr.parse(["coord", "row"], domains)
    position = expr.parse(["coord", "column"], domains)
    placement = PlacementView(None, "tmem" if tmem else "reg", "dp/column" if tmem else "scalar",
                              {"dp": participant, "column": position} if tmem else
                              {"participant": participant, "position": position})
    export = TensorExport("source", identity, (placement,), ("tmem_packet_receipt",) if tmem else ("reg_copy",), Readiness("ready", "done"), {}, (), {"produce": "produce", "release": None})
    demand = DemandView("positions", (row, column), None, {"row": participant, "column": position},
                        expr.parse(["sub", ["const", 3], ["coord", "row"]], domains) if transpose else participant, position)
    call = TensorCallable("access", 'extern "C" __device__ void access(float*);', (CallableParam("dest", "dest_fragment", "float*"),), {"ready": "access_ready", "done": "access_done"})
    def resource(name, domain, extent, scope, base=None, sync=None):
        return Lend(name, domain, extent, 1, scope, "fixed", ResourceLocation(name if base is None else None, base),
                    "consumer", {"from": "invoke", "to": "done"}, sync)
    resources = (resource("dest", "reg", 1, "thread"), resource("temp", "reg", 1, "thread"),
                 resource("scratch", "smem", 20, "cta", base=0),
                 resource("barrier", "barrier", 1, "cta", base=1, sync=ResourceSync(1, 1)),
                 resource("metadata", "metadata", 3, "warp"))
    imp = TensorImport("consumer", identity, (demand,), call, ("hook",), "done", resources, ())
    site = Site("hook", "consumer", Participants(set=(0, 1, 2, 3)), "invoke",
                ("enter_once", "converged", "full_warps", "no_early_return"), ("ready",) if ready else ())
    bindings = (_binding(expr.Domain("world", "int", values=(0, 1))),) if worlds else ()
    profile = _profile(warp_size=2, packet_columns=1)
    return Instance.build(export, imp, plan.derive(export, imp, (site,), bindings, profile), profile)


def _program(instance, *, store=None, load=None, synchronize=True, extra_branch=False, observe=False):
    participant = ["coord", "participant"]
    own = ["mulc", instance.scalar_bytes, participant]
    other = ["mulc", instance.scalar_bytes, ["sub", ["const", 3], participant]]
    if instance.worlds[0]:
        # Both coordinate and residual-binding inputs go through the evaluator.
        own = ["add", own, ["mulc", 0, ["bind", "world"]]]
    nodes = [ir.Issue("observe_ready", {"dest": "seen"}, 1), ir.Branch("seen", 2, 0)] if observe else [ir.Branch("ready", 1, 0)]
    if instance.export.placements[0].space == "tmem":
        nodes.extend((ir.Issue("tmem_packet_receipt", {
            "packet": ["divc", participant, instance.warp_size], "dest": "temp", "phase": "issue", "handle": "r"}, len(nodes) + 1),
            ir.Issue("tmem_packet_receipt", {"phase": "wait", "handle": "r"}, len(nodes) + 2)))
        source = "temp"
    else:
        source = "@export:reg"
    nodes.append(ir.Issue("smem_store_round", {"src": source, "dst": "scratch", "width": 1,
                                               "address": own if store is None else store}, len(nodes) + 1))
    if synchronize:
        nodes.append(ir.Issue("cta_rendezvous", {"resource": "barrier"}, len(nodes) + 1))
    if extra_branch:
        nodes.append(ir.Branch("hidden", len(nodes) + 1, len(nodes) + 1))
    nodes.append(ir.Issue("smem_load_round", {"src": "scratch", "dst": "dest", "width": 1,
                                              "address": other if load is None else load}, len(nodes) + 1))
    nodes.append(ir.Return(instance.demands))
    return instance.program({"hook": ir.ControlGraph(tuple(nodes))})


def _attempt(result):
    return next((row for row in result.reductions if row["kind"] == "static-phase"),
                result.provenance.get("static_phase"))


@pytest.mark.parametrize("tmem,ready,worlds,observe", [(False, True, True, False), (True, False, False, True)])
def test_static_two_warp_delivery_matches_exploration(tmem, ready, worlds, observe):
    instance = _instance(tmem=tmem, ready=ready, worlds=worlds)
    program = _program(instance, observe=observe)
    result = check(instance, program, budget=100000)
    exact = _check_product(instance, program, budget=100000)
    assert result.status == exact.status == "PASS"
    assert result.provenance["terminal_deliveries"] == exact.provenance["terminal_deliveries"]
    record = _attempt(result)
    assert record["verified"] is True
    assert len(record["premises"]) == 8 and all(row["ok"] for row in record["premises"])


@pytest.mark.parametrize("change,premise", [
    ({"store": ["const", 0]}, "stores"),
    ({"load": ["const", 16]}, "loads"),
    ({"synchronize": False}, "rendezvous"),
    ({"extra_branch": True}, "shape"),
    ({"load": ["mulc", 4, ["coord", "participant"]]}, "loads"),
])
def test_failed_premise_falls_back_to_true_unsafe_status(change, premise):
    instance = _instance()
    program = _program(instance, **change)
    result = check(instance, program, budget=100000)
    assert result.status == _check_product(instance, program, budget=100000).status == "UNSAFE"
    attempt = _attempt(result)
    assert attempt["verified"] is False
    assert next(row for row in attempt["premises"] if row["name"] == premise)["ok"] is False
    assert all(row["kind"] != "static-phase" for row in result.reductions)


def test_failed_shape_retains_incomplete_budget_status():
    instance = _instance()
    result = check(instance, _program(instance, extra_branch=True), budget=1)
    assert result.status == "INCOMPLETE"
    assert _attempt(result)["verified"] is False
    assert result.explored_states == 1


def test_monitor_obligations_and_operation_bounds_are_not_silently_dropped():
    instance = _instance()
    monitor = Monitor("terminal", predicate=lambda state: False, name="extra-obligation")
    monitored = replace(instance, monitors=instance.monitors + (monitor,))
    result = check(monitored, _program(monitored), budget=100000)
    assert result.status == "UNSAFE"
    assert "extra-obligation" in result.violations[0].message
    bounded = check(instance, _program(instance), budget={"max_states": 100000, "max_operations": 1})
    assert bounded.status == "UNSAFE"
    assert _attempt(bounded)["verified"] is False


@pytest.mark.parametrize("change", ["valid", "collision", "mismatch", "missing_rendezvous", "bounds"])
def test_noncanonical_xor_layout_matches_exact_checker(change):
    instance = _instance()
    def swizzle(value):
        bits = ["tobits", 3, value]
        return ["mulc", 4, ["toint", ["bxor", bits, ["shrc", bits, 1]]]]
    own = swizzle(["coord", "participant"])
    other = swizzle(["sub", ["const", 3], ["coord", "participant"]])
    if change == "collision":
        own = ["const", 0]
    elif change == "mismatch":
        other = ["mulc", 4, ["sub", ["const", 3], ["coord", "participant"]]]
    elif change == "bounds":
        own = ["const", 20]
    program = _program(instance, store=own, load=other, synchronize=change != "missing_rendezvous")
    result = check(instance, program, budget=100000)
    exact = _check_product(instance, program, budget=100000)
    assert result.status == exact.status == ("PASS" if change == "valid" else "UNSAFE")
    if change == "valid":
        assert result.provenance["terminal_deliveries"] == exact.provenance["terminal_deliveries"]


def test_static_participant_guard_stores_only_owned_sources_and_publishes_to_all():
    source = _instance(transpose=False)
    domains = {domain.name: domain for domain in source.export.identity.coords}
    placement = replace(source.export.placements[0], guard=expr.parse(["lt", ["coord", "row"], ["const", 2]], domains))
    demand = replace(source.imp.demands[0], identity_coords={
        "row": expr.parse(["modc", ["coord", "row"], 2], domains),
        "column": expr.parse(["coord", "column"], domains)})
    instance = _rebuild(source, export=replace(source.export, placements=(placement,)), imp=replace(source.imp, demands=(demand,)))
    participant = ["coord", "participant"]
    program = instance.program({"hook": ir.ControlGraph((
        ir.Meta("has_source", ["lt", participant, ["const", 2]], 1),
        ir.Branch("has_source", 2, 3),
        ir.Issue("smem_store_round", {"src": "@export:reg", "dst": "scratch", "width": 1,
                                     "address": ["mulc", 4, participant]}, 3),
        ir.Issue("cta_rendezvous", {"resource": "barrier"}, 4),
        ir.Issue("smem_load_round", {"src": "scratch", "dst": "dest", "width": 1,
                                    "address": ["mulc", 4, ["modc", participant, 2]]}, 5),
        ir.Return(),
    ))})
    result = check(instance, program, budget=100000)
    exact = _check_product(instance, program, budget=100000)
    assert result.status == exact.status == "PASS"
    assert _attempt(result)["verified"] is True
    assert result.provenance["terminal_deliveries"] == exact.provenance["terminal_deliveries"]


def test_private_copy_then_inplace_shuffle_preserves_snapshot_identity():
    source = _instance(transpose=False)
    domains = {domain.name: domain for domain in source.imp.demands[0].domain}
    receiver = ["add", ["coord", "row"], ["sub", ["const", 1], ["mulc", 2, ["modc", ["coord", "row"], 2]]]]
    demand = replace(source.imp.demands[0], participant=expr.parse(receiver, domains))
    instance = _rebuild(source, imp=replace(source.imp, demands=(demand,)))
    program = instance.program({"hook": ir.ControlGraph((
        ir.Issue("reg_copy_round", {"src": "@export:reg", "dst": "dest", "width": 1}, 1),
        ir.Issue("shfl_xor", {"src": {"resource": "dest", "index": 0},
                              "dst": {"resource": "dest", "index": 0}, "mask": 1}, 2),
        ir.Return(),
    ))})
    result = check(instance, program, budget=100000)
    assert result.status == "PASS"
    assert _attempt(result)["verified"] is True
    assert result.provenance["terminal_deliveries"] == _check_product(instance, program, budget=100000).provenance["terminal_deliveries"]


def test_receipt_premise_reuses_machine_own_subpartition_validation():
    instance = _instance(tmem=True)
    program = _program(instance)
    nodes = list(program.graphs["hook"].nodes)
    nodes[1] = replace(nodes[1], operands={**nodes[1].operands, "packet": 0})
    wrong = instance.program({"hook": ir.ControlGraph(tuple(nodes))})
    result = check(instance, wrong, budget=100000)
    assert result.status == "UNSAFE"
    assert next(row for row in _attempt(result)["premises"] if row["name"] == "receipts")["ok"] is False
    assert "own DP subpartition" in result.violations[0].message


def test_readiness_polling_metadata_must_exist_before_publication():
    source = _instance(tmem=True, ready=False)
    lends = tuple(replace(resource, lifetime={"from": "ready", "to": "done"})
                  if resource.name == "metadata" else resource for resource in source.imp.lends)
    instance = _rebuild(source, imp=replace(source.imp, lends=lends))
    result = check(instance, _program(instance, observe=True), budget=100000)
    assert result.status == "UNSAFE"
    assert next(row for row in _attempt(result)["premises"] if row["name"] == "readiness")["ok"] is False
    assert "metadata" in result.violations[0].message


def test_binding_dependent_guard_is_explored_even_when_both_branches_deliver():
    instance = _instance(worlds=True, transpose=False)
    copy = {"src": "@export:reg", "dst": "dest", "width": 1}
    program = instance.program({"hook": ir.ControlGraph((
        ir.Meta("binding_choice", ["bind", "world"], 1),
        ir.Branch("binding_choice", 2, 3),
        ir.Issue("reg_copy_round", copy, 4),
        ir.Issue("reg_copy_round", copy, 4),
        ir.Return(),
    ))})
    result = check(instance, program, budget=100000)
    assert result.status == "PASS"
    assert _attempt(result)["verified"] is False
    assert next(row for row in _attempt(result)["premises"] if row["name"] == "shape")["ok"] is False


def test_missing_convergence_cannot_certify_shared_publication():
    source = _instance()
    site = replace(source.sites[0], guarantees=tuple(guarantee for guarantee in source.sites[0].guarantees if guarantee != "converged"))
    instance = _rebuild(source, sites=(site,))
    program = _program(instance)
    result = check(instance, program, budget=100000)
    assert result.status == _check_product(instance, program, budget=100000).status == "UNSAFE"
    assert _attempt(result)["verified"] is False
    assert next(row for row in _attempt(result)["premises"] if row["name"] == "rendezvous")["ok"] is False
