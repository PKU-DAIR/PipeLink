from dataclasses import replace

import pytest

from pipec2.model import Claim, Extent, Handle, Role, TensorConnection, TensorLifetime, Variant
from pipec2.profile import get_profile
from pipec2.solve import solve
from pipec2.tensor import expr
from pipec2.tensor.contracts import Binding, Lend, Participants, Readiness, Rendezvous, ResourceLocation
from pipec2.tensor.lifetimes import resolve_lifetimes
from test_tensor_link import _composition


def _claim(handle, tensor, start, end=None, *, domain="smem", size=128, alias=None):
    return Claim(domain, handle, TensorLifetime(tensor, start, end), Extent(size, {}, {}),
                 16 if domain == "smem" else 1, 0, None, None, None, None, alias)


def _fixture():
    pipeline, stages, _, _ = _composition()
    imp = stages[1].tensor.imports[0]
    imp = replace(imp, rendezvous=(Rendezvous("ready", "barrier", Participants(set=(0,))),))
    site = replace(stages[1].tensor.sites[0], guarantees=("enter_once", "single_completion"))
    stages[0] = replace(stages[0], claims=[_claim("ring", "published", "work", "ready")],
                        variants=[Variant({}, 0, None)])
    stages[1] = replace(stages[1], claims=[_claim("scratch", "received", "ready", "delivered")],
                        variants=[Variant({}, 0, None)],
                        tensor=replace(stages[1].tensor, imports=(imp,), sites=(site,)))
    return pipeline, stages


def _allocate(pipeline, stages, ordered_pairs=frozenset()):
    # A real allocator launch with one owned warp and the two resource intervals.
    warp = Claim("warp", "warp", "kernel", Extent(1, {}, {}), 1, 0, ["load"], None, None, None, None)
    stages = [stages[0], replace(stages[1], roles={"load": Role("load", 1)},
                                 claims=[*stages[1].claims, warp])]
    handles = {claim.handle: Handle(claim.handle, "base", "bind", claim.domain, None,
                                    f"claim:{stage.id}:{claim.handle}")
               for stage in stages for claim in stage.claims}
    pipeline = replace(pipeline, handles=handles)
    return solve(pipeline, stages, {}, [], get_profile(pipeline.target.profile), ordered_pairs=ordered_pairs)


def _disjoint(solution):
    a, b = solution.intervals["smem"]
    return a.hi <= b.lo or b.hi <= a.lo


def test_ready_witness_reduces_physical_high_water_and_reports_owner():
    pipeline, stages = _fixture()
    resolution = resolve_lifetimes(pipeline, stages)
    baseline = _allocate(pipeline, stages)
    selected = _allocate(pipeline, stages, resolution.ordered_pairs)
    assert _disjoint(baseline)
    assert not _disjoint(selected)
    assert selected.launch.dyn_smem_bytes < baseline.launch.dyn_smem_bytes
    assert resolution.report(baseline)["overlaps"] == []
    overlap, = resolution.report(selected)["overlaps"]
    assert overlap["handles"] == ["ring", "scratch"]
    assert overlap["event"] == "ready"
    assert overlap["connection"] == "acc"
    assert overlap["helper"] == stages[1].tensor.imports[0].callable.events["ready"]
    assert overlap["ending_handle"] == "ring"
    assert overlap["ranges"][0] == overlap["ranges"][1]


@pytest.mark.parametrize("defect,reason", [("repeat_entry", "missing_enter_once"),
                                           ("early_reclaim", "early_reclaim"),
                                           ("cohort", "readiness_cohort_not_covering")])
def test_unresolved_ready_keeps_physical_storage_disjoint(defect, reason):
    pipeline, stages = _fixture()
    tensor = stages[1].tensor
    if defect == "repeat_entry":
        tensor = replace(tensor, sites=(replace(tensor.sites[0], guarantees=("single_completion",)),))
    elif defect == "early_reclaim":
        export = replace(stages[0].tensor.exports[0], readiness=Readiness("ready", "reclaimed"))
        stages[0] = replace(stages[0], tensor=replace(stages[0].tensor, exports=(export,)))
    else:
        imp = replace(tensor.imports[0], rendezvous=(Rendezvous("ready", "barrier", Participants(set=(1,))),))
        tensor = replace(tensor, imports=(imp,))
    stages[1] = replace(stages[1], tensor=tensor)
    resolution = resolve_lifetimes(pipeline, stages)
    solution = _allocate(pipeline, stages, resolution.ordered_pairs)
    assert _disjoint(solution)
    assert reason in resolution.report(solution)["connections"][0]["positions"]["ready"]["reasons"]


def test_every_selected_site_must_enter_once_but_unselected_sites_do_not_matter():
    pipeline, stages = _fixture()
    tensor = stages[1].tensor
    second = replace(tensor.sites[0], name="second", guarantees=())
    stages[1] = replace(stages[1], tensor=replace(tensor, sites=(*tensor.sites, second)))
    assert resolve_lifetimes(pipeline, stages).ordered_pairs == frozenset({("ring", "scratch")})
    imp = replace(tensor.imports[0], sites=("load", "second"))
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=(imp,)))
    assert not resolve_lifetimes(pipeline, stages).ordered_pairs


def test_repeated_completion_preserves_ready_order_but_not_done():
    pipeline, stages = _fixture()
    tensor = stages[1].tensor
    stages[1] = replace(stages[1], claims=[*stages[1].claims, _claim("after", "received", "delivered")],
                        tensor=replace(tensor, sites=(replace(tensor.sites[0], guarantees=("enter_once",)),)))
    resolution = resolve_lifetimes(pipeline, stages)
    assert ("ring", "scratch") in resolution.ordered_pairs
    assert ("after", "scratch") not in resolution.ordered_pairs
    assert ("after", "ring") not in resolution.ordered_pairs
    stages[1] = replace(stages[1], tensor=tensor)
    assert ("after", "scratch") in resolve_lifetimes(pipeline, stages).ordered_pairs


def test_export_done_does_not_reclaim_other_fanout_reader():
    pipeline, stages = _fixture()
    stages[0] = replace(stages[0], claims=[_claim("backing", "published", "work", "delivered")])
    stages[1] = replace(stages[1], claims=[_claim("after", "received", "delivered")])
    assert ("after", "backing") in resolve_lifetimes(pipeline, stages).ordered_pairs
    imp = replace(stages[1].tensor.imports[0], name="reader_b")
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=(*stages[1].tensor.imports, imp)))
    pipeline = replace(pipeline, tensor_connections=(*pipeline.tensor_connections,
                       TensorConnection("other", "org.producer:published", "org.consumer:reader_b")))
    resolution = resolve_lifetimes(pipeline, stages)
    solution = _allocate(pipeline, stages, resolution.ordered_pairs)
    assert _disjoint(solution)
    assert all(not row["positions"]["done"]["export_resolved"] for row in resolution.report(solution)["connections"])


def test_fanout_keeps_single_publication_ready_order():
    pipeline, stages = _fixture()
    imp = replace(stages[1].tensor.imports[0], name="reader_b")
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=(*stages[1].tensor.imports, imp)))
    pipeline = replace(pipeline, tensor_connections=(*pipeline.tensor_connections,
                       TensorConnection("other", "org.producer:published", "org.consumer:reader_b")))
    assert ("ring", "scratch") in resolve_lifetimes(pipeline, stages).ordered_pairs


def test_different_connections_do_not_establish_event_order():
    pipeline, stages = _fixture()
    export = replace(stages[0].tensor.exports[0], name="other_export")
    imp = replace(stages[1].tensor.imports[0], name="other_import")
    stages[0] = replace(stages[0], tensor=replace(stages[0].tensor, exports=(*stages[0].tensor.exports, export)))
    stages[1] = replace(stages[1], claims=[_claim("scratch", "other_import", "ready", "delivered")],
                        tensor=replace(stages[1].tensor, imports=(*stages[1].tensor.imports, imp)))
    pipeline = replace(pipeline, tensor_connections=(*pipeline.tensor_connections,
                       TensorConnection("other", "org.producer:other_export", "org.consumer:other_import")))
    resolution = resolve_lifetimes(pipeline, stages)
    assert _disjoint(_allocate(pipeline, stages, resolution.ordered_pairs))


def test_reverse_handle_order_is_allowed_but_reversed_live_intervals_are_not():
    pipeline, stages = _fixture()
    stages[0] = replace(stages[0], claims=[replace(stages[0].claims[0], handle="z_ring")])
    stages[1] = replace(stages[1], claims=[replace(stages[1].claims[0], handle="a_scratch")])
    resolution = resolve_lifetimes(pipeline, stages)
    assert ("a_scratch", "z_ring") in resolution.ordered_pairs
    selected = _allocate(pipeline, stages, resolution.ordered_pairs)
    assert resolution.report(selected)["overlaps"][0]["ending_handle"] == "z_ring"
    stages[0] = replace(stages[0], claims=[_claim("z_ring", "published", "ready", "delivered")])
    stages[1] = replace(stages[1], claims=[_claim("a_scratch", "received", "work", "delivered")])
    resolution = resolve_lifetimes(pipeline, stages)
    assert _disjoint(_allocate(pipeline, stages, resolution.ordered_pairs))


@pytest.mark.parametrize("domain,alias", [("tmem", None), ("named_barrier", None), ("smem", "shared")])
def test_only_smem_without_shared_alias_has_event_permission(domain, alias):
    pipeline, stages = _fixture()
    stages = [replace(stage, claims=[replace(claim, domain=domain, alias=alias) for claim in stage.claims])
              for stage in stages]
    assert not resolve_lifetimes(pipeline, stages).ordered_pairs


def test_shared_handle_kernel_claim_prevents_handle_wide_permission():
    pipeline, stages = _fixture()
    stages[0] = replace(stages[0], claims=[*stages[0].claims, replace(stages[0].claims[0], lifetime="kernel")])
    assert not resolve_lifetimes(pipeline, stages).ordered_pairs


@pytest.mark.parametrize("lifetime", [TensorLifetime("unbound", "work", "ready"),
                                      TensorLifetime("published", "work", "unknown"), "kernel"])
def test_unbound_or_unknown_end_cannot_order_claim(lifetime):
    pipeline, stages = _fixture()
    stages[0] = replace(stages[0], claims=[replace(stages[0].claims[0], lifetime=lifetime)])
    assert not resolve_lifetimes(pipeline, stages).ordered_pairs


def _resource(handle="scratch", start="ready", end="delivered"):
    return Lend("loan", "smem", 128, 16, "cta", "relocatable", ResourceLocation(handle, None),
                "org.consumer", {"from": start, "to": end}, None)


@pytest.mark.parametrize("claim_end,covered", [("ready", False), ("delivered", True), (None, True)])
def test_lend_must_fit_event_claim_and_borrowed_coverage(claim_end, covered):
    pipeline, stages = _fixture()
    claim = _claim("scratch", "received", "work", claim_end)
    resource = _resource()
    imp = replace(stages[1].tensor.imports[0], lends=(resource,))
    stages[1] = replace(stages[1], claims=[claim], tensor=replace(stages[1].tensor, imports=(imp,)))
    resolution = resolve_lifetimes(pipeline, stages)
    assert resolution.covers(stages[1].id, claim, resource) is covered
    assert bool(resolution.failures) is not covered
    if not covered:
        assert [(failure.clause_id, failure.path) for failure in resolution.failures] == [
            ("T-PLAN", "stages.org.consumer.tensor.imports.received.lends.loan")]


def test_cross_stage_borrowed_reservation_uses_claim_owner_events():
    pipeline, stages = _fixture()
    claim = _claim("ring", "published", "work", "delivered")
    resource = _resource("ring")
    stages[0] = replace(stages[0], claims=[claim])
    imp = replace(stages[1].tensor.imports[0], lends=(resource,))
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=(imp,)))
    assert resolve_lifetimes(pipeline, stages).covers(stages[1].id, claim, resource)


@pytest.mark.parametrize("kernel", [False, True])
def test_backing_cannot_end_before_done_even_when_claim_contains_it(kernel):
    pipeline, stages = _fixture()
    resource = _resource("ring", "work", "ready")
    export = replace(stages[0].tensor.exports[0], backing=(resource,))
    if kernel:
        stages[0] = replace(stages[0], claims=[replace(stages[0].claims[0], lifetime="kernel")])
    stages[0] = replace(stages[0], tensor=replace(stages[0].tensor, exports=(export,)))
    resolution = resolve_lifetimes(pipeline, stages)
    assert [(failure.clause_id, failure.path) for failure in resolution.failures] == [
        ("T-PLAN", "stages.org.producer.tensor.exports.published.backing.loan")]


def test_legacy_kernel_backing_with_unknown_reclaimed_event_stays_conservative():
    pipeline, stages = _fixture()
    resource = _resource("ring", "work", "reclaimed")
    export = replace(stages[0].tensor.exports[0], backing=(resource,))
    claim = replace(stages[0].claims[0], lifetime="kernel")
    stages[0] = replace(stages[0], claims=[claim], tensor=replace(stages[0].tensor, exports=(export,)))
    resolution = resolve_lifetimes(pipeline, stages)
    assert not resolution.failures
    assert resolution.covers(stages[0].id, claim, resource)
    assert not resolution.ordered_pairs


def test_expression_participants_require_coverage_at_each_binding():
    pipeline, stages = _fixture()
    domain = expr.Domain("lane", "int", values=(0, 1))
    binding = Binding("lane", "int", domain, "owner", "threads", "cta", "kernel")
    site_members = Participants(expr=expr.parse(["bind", "lane"], {"lane": domain}), domain=())
    cohort = Participants(expr=expr.parse(["sub", ["const", 1], ["bind", "lane"]], {"lane": domain}), domain=())
    tensor = stages[1].tensor
    site = replace(tensor.sites[0], participants=site_members)
    imp = replace(tensor.imports[0], rendezvous=(Rendezvous("ready", "barrier", cohort),))
    stages[1] = replace(stages[1], tensor=replace(tensor, bindings=(binding,), sites=(site,), imports=(imp,)))
    assert not resolve_lifetimes(pipeline, stages).ordered_pairs
    imp = replace(imp, rendezvous=(Rendezvous("ready", "barrier", Participants(set=(0, 1, 2))),))
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=(imp,)))
    assert ("ring", "scratch") in resolve_lifetimes(pipeline, stages).ordered_pairs


@pytest.mark.parametrize("single", [False, True])
def test_done_reuse_needs_single_completion_physical_proof(single):
    pipeline, stages = _fixture()
    stages[0] = replace(stages[0], claims=[])
    tensor = stages[1].tensor
    guarantees = ("enter_once", "single_completion") if single else ("enter_once",)
    stages[1] = replace(
        stages[1], claims=[*stages[1].claims, _claim("after", "received", "delivered")],
        tensor=replace(tensor, sites=(replace(tensor.sites[0], guarantees=guarantees),)))
    resolution = resolve_lifetimes(pipeline, stages)
    solution = _allocate(pipeline, stages, resolution.ordered_pairs)
    assert _disjoint(solution) is not single
    if single:
        overlap, = resolution.report(solution)["overlaps"]
        assert overlap["event"] == "delivered"
        assert overlap["ending_handle"] == "scratch"
        assert overlap["helper"] == tensor.imports[0].callable.events["done"]


def test_scaled_smem_view_can_reuse_bytes():
    pipeline, stages = _fixture()
    stages[0] = replace(stages[0], claims=[replace(stages[0].claims[0], domain="mbarrier",
                                                 extent=Extent(8, {}, {}), align=1)])
    resolution = resolve_lifetimes(pipeline, stages)
    solution = _allocate(pipeline, stages, resolution.ordered_pairs)
    assert not _disjoint(solution)
    assert solution.launch.dyn_smem_bytes == 128


def test_resource_start_before_claim_is_not_covered():
    pipeline, stages = _fixture()
    resource = _resource(start="work")
    imp = replace(stages[1].tensor.imports[0], lends=(resource,))
    stages[1] = replace(stages[1], tensor=replace(stages[1].tensor, imports=(imp,)))
    resolution = resolve_lifetimes(pipeline, stages)
    assert not resolution.covers(stages[1].id, stages[1].claims[0], resource)
    assert resolution.failures[0].clause_id == "T-PLAN"
