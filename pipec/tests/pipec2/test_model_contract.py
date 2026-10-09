from __future__ import annotations

import re

import pytest

from pipec2.canonjson import canonical_dumps
from pipec2.layout import load_layout
from pipec2.model import (
    CLAUSE_IDS,
    CLAUSES,
    CLAUSE_PHASES,
    Arrive,
    BarrierSide,
    CallDecl,
    CanonEndpoint,
    Claim,
    ClauseFailure,
    ClusterConstraints,
    CounterDecl,
    Depth,
    DescriptorDecl,
    Endpoint,
    Export,
    Extent,
    Handle,
    Init,
    Interval,
    Launch,
    ObjectiveTier,
    LayoutError,
    LegalitySet,
    LinkError,
    LinkResult,
    NamedBarrierDecl,
    PackError,
    ParamsDecl,
    ParamsLayout,
    Pipeline,
    PipelineDomain,
    PipelineTopology,
    ProfileError,
    Protocol,
    ProtocolExpansion,
    RegGroup,
    Ring,
    Role,
    SchemaError,
    SharedMember,
    Solution,
    SolveError,
    StageManifest,
    StageRef,
    StageTopology,
    SymbolInfo,
    SymbolTable,
    TargetStamp,
    Tile,
    TxSum,
    Variant,
    register_clause,
)


ZERO_HASH = "0" * 64


def _interval_claim(domain: str, handle: str, extent: int, *, offset: int = 0) -> Claim:
    return Claim(
        domain=domain,
        handle=handle,
        lifetime="kernel",
        extent=Extent(const=extent, per={}, table={}),
        align=1,
        offset=offset,
        roles=None,
        reg_group=None,
        values=None,
        absolute_base=None,
        alias=None,
    )


def _regs_claim(group: str, role: str, handle: str, value: int) -> Claim:
    return Claim(
        domain="regs",
        handle=handle,
        lifetime="kernel",
        extent=None,
        align=None,
        offset=None,
        roles=[role],
        reg_group=group,
        values=[value],
        absolute_base=None,
        alias=None,
    )


def _miniature_contract() -> tuple[list[object], Pipeline, list[StageManifest]]:
    target = TargetStamp("2.0", "sm_103a", "lto_103", "cuda-13.3.0")
    cluster = ClusterConstraints(1, (2, 1, 1))
    topology = StageTopology(1, (1, 1, 1), cluster, ["ab_ring"], True, "static")
    params = ParamsDecl(64, 64, "ExampleParams", ZERO_HASH)
    tile = Tile("bf16", (8, 8))
    depth = Depth(None, "pipelink_ab_depth")
    ring = Ring(depth, 128)
    tx_sum = TxSum(["ab_ring"])
    arrive_full = Arrive(1, "load")
    arrive_empty = Arrive("threads(mma)", "mma")
    full = BarrierSide("pipelink_ab_full_slots", arrive_full, tx_sum)
    empty = BarrierSide("pipelink_ab_empty_slots", arrive_empty, 0)
    init = Init("load", 0, "dispatch")
    mbarrier = Protocol("mbarrier", full, empty, init, "once-per-acquire", None, None, None)

    call_decl = CallDecl("register_compute", "int(int)", 'extern "C" int consumer_compute(int x);', ZERO_HASH)
    call = Protocol("call", None, None, None, None, call_decl, None, None)
    named_decl = NamedBarrierDecl("pipelink_named_id", 128)
    named_protocol = Protocol("named_barrier", None, None, None, None, None, named_decl, None)
    counter_decl = CounterDecl("pipelink_counter_slot", "red.release.gpu", "ld.acquire.gpu")
    counter_protocol = Protocol("counter", None, None, None, None, None, None, counter_decl)
    shared_member = SharedMember("ab_ring", "tile")

    layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 16,
        "shape": [[8], [8]],
        "stride": [[8], [1]],
    }
    reg_layout = {
        "encoding": "pl-layout-v2",
        "elem_bits": 32,
        "shape": [[128], [1]],
        "stride": [[1], [1]],
    }
    producer_ring = Endpoint(
        "ab_ring",
        "produce",
        "load",
        "smem",
        "cta",
        "async",
        tile,
        load_layout(layout, "smem", "$.layout"),
        ring,
        mbarrier,
        {
            "base": "pipelink_ab_smem_base",
            "empty": "pipelink_ab_empty_slots",
            "full": "pipelink_ab_full_slots",
        },
    )
    consumer_ring = Endpoint(
        "ab_ring",
        "consume",
        "mma",
        "smem",
        "cta",
        "async",
        tile,
        load_layout(layout, "smem", "$.layout"),
        ring,
        mbarrier,
        {
            "base": "pipelink_ab_smem_base",
            "empty": "pipelink_ab_empty_slots",
            "full": "pipelink_ab_full_slots",
        },
    )
    caller = Endpoint(
        "register_call",
        "produce",
        "mma",
        "reg",
        "cta",
        "generic",
        Tile("s32", (128, 1)),
        load_layout(reg_layout, "reg", "$.layout"),
        None,
        call,
        {},
    )
    callee = Endpoint(
        "register_call",
        "consume",
        None,
        "reg",
        "cta",
        "generic",
        Tile("s32", (128, 1)),
        load_layout(reg_layout, "reg", "$.layout"),
        None,
        call,
        {},
    )

    handles = {
        handle.name: handle
        for handle in [
            Handle("pipelink_ab_depth", "depth", "bind", None, [2, 4], "channel:ab_ring"),
            Handle("pipelink_ab_empty_slots", "base", "bind", "mbarrier", None, "claim:ab_empty"),
            Handle("pipelink_ab_full_slots", "base", "bind", "mbarrier", None, "claim:ab_full"),
            Handle("pipelink_ab_smem_base", "base", "bind", "smem", None, "claim:ab_ring"),
            Handle("pipelink_regs_load", "regs", "bind", "regs", [64], "reg_group:load_wg"),
            Handle("pipelink_regs_mma", "regs", "bind", "regs", [128], "reg_group:mma_wg"),
            Handle("pipelink_variant", "variant", "fixed", None, [1], "stage:producer.stage"),
            Handle("pipelink_warp_load", "base", "bind", "warp", None, "claim:warp_load"),
            Handle("pipelink_warp_mma", "base", "bind", "warp", None, "claim:warp_mma"),
        ]
    }

    producer_claims = [
        _interval_claim("mbarrier", "pipelink_ab_empty_slots", 4),
        _interval_claim("mbarrier", "pipelink_ab_full_slots", 4),
        _regs_claim("load_wg", "load", "pipelink_regs_load", 64),
        Claim(
            "smem",
            "pipelink_ab_smem_base",
            "phase:steady",
            Extent(0, {"pipelink_ab_depth": 128}, {}),
            128,
            0,
            None,
            None,
            None,
            None,
            None,
        ),
        _interval_claim("warp", "pipelink_warp_load", 4),
    ]
    consumer_claims = [
        _regs_claim("mma_wg", "mma", "pipelink_regs_mma", 128),
        _interval_claim("warp", "pipelink_warp_mma", 4, offset=4),
    ]

    producer = StageManifest(
        "producer.stage",
        "1.0.0",
        "producer_stage_",
        {"load": Role("producer_stage_role_load", 4)},
        {},
        "void(const void* params, unsigned variant)",
        "v2",
        [Variant({"pipelink_variant": 1}, 0, None)],
        [producer_ring, caller],
        producer_claims,
        topology,
        params,
        DescriptorDecl(128, 16),
        {"bytes_per_tile": 128},
        target,
        provenance={
            "mechanism": "pack-cu-v1",
            "inputs": {"owner.cu": "4" * 64},
            "sections": {"$.params": "compiled", "$.stage.roles": "declared",
                         "$.symbols": "observed", "$.tensor.exports[0].placements": "fitted"},
        },
    )
    consumer = StageManifest(
        "consumer.stage",
        "1.0.0",
        "consumer_stage_",
        {"mma": Role("consumer_stage_role_mma", 4)},
        {
            "register_compute": Export(
                "consumer_stage_register_compute",
                call_decl.decl,
                call_decl.decl_sha256,
            )
        },
        "void(const void* params, unsigned variant)",
        "v2",
        [Variant({}, 0, "consumer_stage_variant_0")],
        [consumer_ring, callee],
        consumer_claims,
        topology,
        ParamsDecl(32, 16, "ConsumerParams", ZERO_HASH),
        None,
        {},
        target,
    )

    pipeline = Pipeline(
        "miniature",
        [StageRef("producer.pla", producer.id), StageRef("consumer.pla", consumer.id)],
        handles,
        ["steady", "setup"],
        [("steady", "setup")],
        {
            "gmem_counter": PipelineDomain(16, None),
            "smem_page": PipelineDomain(2, 4096),
        },
        ["pipelink_ab_depth"],
        ["pipelink_regs_mma", "pipelink_regs_load"],
        {"pipelink_variant": 1},
        PipelineTopology(True, "static"),
        target,
    )

    canon = CanonEndpoint(
        (((8, 8), (128, 16)),),
        16,
        (3, 7, 3),
        1024,
        1024,
        ("pipelink_ab_depth", 128),
        "smem",
        "cta",
        "async",
        "bf16",
        (8, 8),
        None,
    )
    mbarrier_expansion = ProtocolExpansion(
        "ab_ring",
        "mbarrier",
        128,
        128,
        1,
        128,
        "load",
        "dispatch",
        ("pipelink_ab_full_slots", "pipelink_ab_empty_slots"),
        ("ab_ring", ("ab_ring",), {"ab_ring": 128}),
        None,
        None,
        None,
    )
    call_expansion = ProtocolExpansion(
        "register_call",
        "call",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        ("consumer_stage_register_compute", call_decl.decl, call_decl.decl_sha256),
        None,
        None,
    )
    all_fields_expansion = ProtocolExpansion(
        "synthetic_projection",
        "projection",
        4,
        4,
        1,
        1,
        "load",
        "stage",
        ("full", "empty"),
        ("root", ("member", "root"), {"member": 2, "root": 2}),
        ("export", "extern int export(int x);", ZERO_HASH),
        ("pipelink_named_id", 128),
        ("pipelink_counter_slot", "red.release.gpu", ("ld.acquire.gpu",)),
    )
    intervals = {
        "smem": [Interval("pipelink_ab_smem_base", 0, 512, "steady")],
        "warp": [
            Interval("pipelink_warp_load", 0, 4, "kernel"),
            Interval("pipelink_warp_mma", 4, 8, "kernel"),
        ],
    }
    launch = Launch(256, 96, 512, (1, 1, 1))
    reg_groups = [
        RegGroup(["load"], 0, 4, "pipelink_regs_load", 64, "dec"),
        RegGroup(["mma"], 4, 8, "pipelink_regs_mma", 128, "inc"),
    ]
    solution = Solution(
        {"pipelink_ab_depth": 4},
        {"load_wg": 64, "mma_wg": 128},
        {"pipelink_ab_smem_base": 0},
        intervals,
        {producer.id: {"pipelink_variant": 1}, consumer.id: {}},
        reg_groups,
        {"load": (0, 4), "mma": (4, 8)},
        launch,
        [ObjectiveTier("depth:pipelink_ab_depth", 4, 4, "bound_met")],
        0,
    )
    params_layout = ParamsLayout(
        128,
        64,
        [
            (producer.id, 0, 64, 64),
            (consumer.id, 64, 32, 16),
        ],
    )
    symbol_info = {
        "consumer_stage_register_compute": SymbolInfo("func", "weak"),
        "pipelink_ab_depth": SymbolInfo("object", "undef"),
        "producer_stage_role_load": SymbolInfo("func", "strong"),
    }
    symbols = SymbolTable(
        ("producer_stage_role_load",),
        ("consumer_stage_register_compute",),
        ("pipelink_ab_depth",),
        symbol_info,
    )
    legality = LegalitySet(1, (2, 1, 1))
    link_result = LinkResult(
        "pipelink_kernel_miniature",
        256,
        96,
        512,
        legality,
        params_layout,
        {"pipelink_ab_depth": (4, "solved"), "pipelink_variant": (1, "forced")},
        solution.variants,
        intervals,
        [call_expansion, mbarrier_expansion],
        (90, 0, 0),
        {producer.id: "1" * 64, consumer.id: "2" * 64},
        {"dispatch.cu": "3" * 64, "stub.cu": "4" * 64},
        {"channels": {"register_call": {"call": 0, "ldl": 0, "stl": 0}}},
    )
    failure = ClauseFailure("L-PROTO", "channels.ab_ring", "example", {"actual": 1})

    objects: list[object] = [
        failure,
        next(iter(handles.values())),
        next(iter(producer.roles.values())),
        next(iter(consumer.exports.values())),
        producer.variants[0],
        target,
        cluster,
        topology,
        params,
        producer.descriptor,
        tile,
        depth,
        ring,
        arrive_full,
        tx_sum,
        full,
        shared_member,
        init,
        call_decl,
        named_decl,
        counter_decl,
        mbarrier,
        call,
        named_protocol,
        counter_protocol,
        producer_ring,
        producer_claims[0].extent,
        producer_claims[0],
        producer_claims[2],
        producer,
        consumer,
        pipeline.stages[0],
        next(iter(pipeline.domains.values())),
        pipeline.topology,
        pipeline,
        canon,
        mbarrier_expansion,
        call_expansion,
        all_fields_expansion,
        intervals["smem"][0],
        launch,
        reg_groups[0],
        solution,
        params_layout,
        next(iter(symbol_info.values())),
        symbols,
        legality,
        link_result,
    ]
    return objects, pipeline, [producer, consumer]


def test_every_model_type_round_trips_canonically() -> None:
    objects, pipeline, stages = _miniature_contract()
    for value in objects:
        json_obj = value.to_json_obj()  # type: ignore[attr-defined]
        restored = type(value).from_json_obj(json_obj)  # type: ignore[attr-defined]
        assert canonical_dumps(restored.to_json_obj()) == canonical_dumps(json_obj)

    assert [item["stage_id"] for item in pipeline.to_json_obj()["stages"]] == [
        "consumer.stage",
        "producer.stage",
    ]
    assert [item["id"] for item in sorted((stage.to_json_obj() for stage in stages), key=lambda x: x["id"])] == [
        "consumer.stage",
        "producer.stage",
    ]




def test_claim_presence_rules_reject_invalid_combinations() -> None:
    with pytest.raises(ValueError):
        Claim("smem", "base", "kernel", None, 16, None, None, None, None, None, None)
    with pytest.raises(ValueError):
        Claim("mbarrier", "slots", "kernel", Extent(1, {}, {}), None, None, None, None, None, None, None)

    base = {
        "domain": "regs",
        "handle": "regs",
        "lifetime": "kernel",
        "extent": None,
        "align": None,
        "offset": None,
        "roles": ["mma"],
        "reg_group": "mma_wg",
        "values": [128],
        "absolute_base": None,
        "alias": None,
    }
    for name, bad_value in [
        ("extent", Extent(1, {}, {})),
        ("align", 8),
        ("offset", 0),
        ("absolute_base", 0),
        ("alias", "other"),
    ]:
        values = dict(base)
        values[name] = bad_value
        with pytest.raises(ValueError):
            Claim(**values)
    for name, bad_value in [("roles", None), ("roles", []), ("values", None), ("values", [])]:
        values = dict(base)
        values[name] = bad_value
        with pytest.raises(ValueError):
            Claim(**values)


def test_clause_registry_structure_and_exception_ordering() -> None:
    assert set(CLAUSE_IDS) == set(CLAUSES) == set(CLAUSE_PHASES)
    for clause_id, description in CLAUSES.items():
        assert re.fullmatch(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*", clause_id)
        assert isinstance(description, str)
        assert CLAUSE_PHASES[clause_id] in {
            "load", "pack", "layout", "protocol", "solve", "link", "launch"
        }
    failures = [
        ClauseFailure("P-IMPORT", "z", "later", {}),
        ClauseFailure("L-ALIGN", "a", "first", {}),
    ]
    for error_type in (SchemaError, ProfileError, LayoutError, PackError, SolveError, LinkError):
        error = error_type(failures)
        assert isinstance(error.failures, list)
        assert [(item.clause_id, item.path) for item in error.failures] == [
            ("L-ALIGN", "a"),
            ("P-IMPORT", "z"),
        ]
        assert str(error) == "; ".join(
            f"[{failure.clause_id}] {failure.path}: {failure.message}"
            for failure in error.failures
        )


def test_registered_clause_is_admitted_without_rebuilding_registry() -> None:
    clause_id = "P-TEST-EXTENSION"
    assert clause_id not in CLAUSE_IDS
    register_clause(clause_id, phase="pack", description="extension-owned validation")
    try:
        failure = ClauseFailure(clause_id, "$.payload", "rejected", {"reason": "invalid"})
        assert clause_id in CLAUSE_IDS
        assert ClauseFailure.from_json_obj(failure.to_json_obj()) == failure
        assert PackError([failure]).failures == [failure]
        with pytest.raises(ValueError):
            register_clause(clause_id, phase="load")
        assert CLAUSE_PHASES[clause_id] == "pack"
    finally:
        del CLAUSES[clause_id]
        del CLAUSE_PHASES[clause_id]


def test_clause_registration_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError):
        register_clause("not a clause", phase="pack")
    with pytest.raises(ValueError):
        register_clause("P-TEST-INVALID", phase="not-a-phase")
    with pytest.raises(ValueError):
        ClauseFailure("P-TEST-UNREGISTERED", "$", "rejected", {})
