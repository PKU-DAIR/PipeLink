from __future__ import annotations

from dataclasses import replace
import re

import pytest

from pipec2.decls import declaration_digest
from pipec2.dispatchgen import generate_dispatch, params_layout
from pipec2.layout import load_layout
from pipec2.model import (
    Arrive,
    BarrierSide,
    ClusterConstraints,
    Depth,
    Endpoint,
    Handle,
    Init,
    Launch,
    LinkError,
    ParamsDecl,
    Pipeline,
    PipelineTopology,
    Protocol,
    ProtocolExpansion,
    RegGroup,
    Ring,
    Role,
    Solution,
    StageManifest,
    StageRef,
    StageTopology,
    TargetStamp,
    Tile,
    Variant,
)
from pipec2.profile import SM_103A
from pipec2.stubgen import generate_stub


TARGET = TargetStamp("2.0-draft", "sm_103a", "lto_103", "cuda-13.3.0")
ZERO_HASH = "0" * 64


def _barrier_protocol() -> Protocol:
    return Protocol(
        "mbarrier",
        BarrierSide("pipelink_route_full", Arrive(1013, "load"), 1011),
        BarrierSide("pipelink_route_empty", Arrive(1014, "mma"), 0),
        Init("load", 0, "dispatch"),
        "once-per-acquire",
        None,
        None,
        None,
    )


def _endpoint(direction: str, owner: str) -> Endpoint:
    return Endpoint(
        "route_ring",
        direction,
        owner,
        "smem",
        "cluster",
        "async",
        Tile("bf16", (8, 8)),
        load_layout({"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [8, 8], "stride": [8, 1]}, "smem", "$.layout"),
        Ring(Depth(None, "pipelink_route_depth"), 128),
        _barrier_protocol(),
        {
            "full": "pipelink_route_full",
            "empty": "pipelink_route_empty",
        },
    )


def _fixture() -> tuple[Pipeline, list[StageManifest], Solution, list[ProtocolExpansion]]:
    topology = StageTopology(
        1,
        (2, 1, 1),
        ClusterConstraints(2, (2, 1, 1)),
        ["route_ring"],
        False,
        "static",
    )
    producer = StageManifest(
        "a.producer",
        "1",
        "a_producer_",
        {"load": Role("a_producer_role_load", 4)},
        {},
        "void(const void* params, unsigned variant)",
        "v2",
        [Variant({"pipelink_variant": 7}, 37, None)],
        [_endpoint("produce", "load")],
        [],
        topology,
        ParamsDecl(73, 32, "ProducerParams", ZERO_HASH),
        None,
        {},
        TARGET,
    )
    consumer = StageManifest(
        "z.consumer",
        "1",
        "z_consumer_",
        {"mma": Role("z_consumer_role_mma", 4)},
        {},
        "void(const void* params, unsigned variant)",
        "v2",
        [Variant({}, 41, "z_consumer_variant_selected")],
        [_endpoint("consume", "mma")],
        [],
        topology,
        ParamsDecl(19, 128, "ConsumerParams", ZERO_HASH),
        None,
        {},
        TARGET,
    )
    handles = {
        handle.name: handle
        for handle in [
            Handle(
                "pipelink_route_depth",
                "depth",
                "bind",
                None,
                [3],
                "channel:route_ring",
            ),
            Handle(
                "pipelink_route_full",
                "base",
                "bind",
                "mbarrier",
                None,
                "claim:route_full",
            ),
            Handle(
                "pipelink_route_empty",
                "base",
                "bind",
                "mbarrier",
                None,
                "claim:route_empty",
            ),
            Handle(
                "pipelink_named_route",
                "base",
                "bind",
                "named_barrier",
                None,
                "claim:named_route",
            ),
            Handle(
                "pipelink_counter_route",
                "base",
                "bind",
                "gmem_counter",
                None,
                "claim:counter_route",
            ),
            Handle(
                "pipelink_regs_producer",
                "regs",
                "bind",
                "regs",
                [64],
                "reg_group:producer_wg",
            ),
            Handle(
                "pipelink_regs_consumer",
                "regs",
                "bind",
                "regs",
                [128],
                "reg_group:consumer_wg",
            ),
            Handle(
                "pipelink_launch_regs",
                "regs",
                "bind",
                "regs",
                [96],
                "launch",
            ),
            Handle(
                "pipelink_variant",
                "variant",
                "fixed",
                None,
                [7],
                "stage:a.producer",
            ),
        ]
    }
    pipeline = Pipeline(
        "routing",
        [StageRef("z.pla", consumer.id), StageRef("a.pla", producer.id)],
        handles,
        ["kernel"],
        [],
        {},
        ["pipelink_route_depth"],
        ["consumer_wg", "producer_wg"],
        {"pipelink_variant": 7},
        PipelineTopology(False, "static"),
        TARGET,
    )
    solution = Solution(
        {"pipelink_route_depth": 3},
        {"producer_wg": 64, "consumer_wg": 128},
        {
            "pipelink_route_full": 11,
            "pipelink_route_empty": 17,
            "pipelink_named_route": 5,
            "pipelink_counter_route": 9,
        },
        {},
        {producer.id: {"pipelink_variant": 7}, consumer.id: {}},
        [
            RegGroup(["mma"], 4, 8, "pipelink_regs_consumer", 128, "inc"),
            RegGroup(["load"], 0, 4, "pipelink_regs_producer", 64, "dec"),
        ],
        {"load": (0, 4), "mma": (4, 8)},
        Launch(256, 96, 4096, (2, 1, 1)),
        [],
        0,
    )
    call_decl = (
        "struct RoutingFragment { float x[4]; };\n"
        'extern "C" __device__ RoutingFragment z_consumer_callable('
        "RoutingFragment fragment, float scale);"
    )
    expansions = [
        ProtocolExpansion(
            "route_named",
            "named_barrier",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ("pipelink_named_route", 1017),
            None,
        ),
        ProtocolExpansion(
            "route_call",
            "call",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ("z_consumer_callable", call_decl, declaration_digest(call_decl)),
            None,
            None,
        ),
        ProtocolExpansion(
            "route_ring",
            "mbarrier",
            1011,
            1012,
            1013,
            1014,
            "load",
            "dispatch",
            ("pipelink_route_full", "pipelink_route_empty"),
            (
                "route_ring",
                ("route_member_unique", "route_ring"),
                {"route_member_unique": 1015, "route_ring": 1016},
            ),
            None,
            None,
            None,
        ),
        ProtocolExpansion(
            "route_counter",
            "counter",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (
                "pipelink_counter_route",
                "route_release_unique",
                "route_acquire_unique",
            ),
        ),
    ]
    return pipeline, [consumer, producer], solution, expansions


def test_params_layout_is_canonical_and_aligned() -> None:
    _pipeline, stages, _solution, _expansions = _fixture()
    layout = params_layout(stages, SM_103A)
    assert layout.align == 128
    assert layout.total_bytes == 256
    assert layout.entries == [
        ("a.producer", 0, 73, 32),
        ("z.consumer", 128, 19, 128),
    ]
    assert params_layout(list(reversed(stages)), SM_103A) == layout


def test_params_layout_rejects_aggregate_limit_with_registered_clause() -> None:
    _pipeline, stages, _solution, _expansions = _fixture()
    too_large = replace(
        stages[0], params=replace(stages[0].params, bytes=SM_103A.caps.param_limit_bytes)
    )
    with pytest.raises(LinkError) as caught:
        params_layout([too_large], SM_103A)
    assert [failure.clause_id for failure in caught.value.failures] == ["L-PARAMS"]
    assert caught.value.failures[0].detail["limit"] == 32764


def test_call_declaration_is_emitted_only_in_binding_stub() -> None:
    pipeline, stages, solution, expansions = _fixture()

    params = params_layout(stages, SM_103A)
    stub = generate_stub(solution, pipeline, stages, expansions)
    dispatch = generate_dispatch(solution, pipeline, stages, expansions, params)

    call = next(item for item in expansions if item.callable is not None)
    assert call.callable is not None
    export_symbol, decl, _digest = call.callable
    assert export_symbol in stub and export_symbol not in dispatch
    assert decl in stub and decl not in dispatch


def test_generated_dispatch_has_required_control_flow_and_literals() -> None:
    pipeline, stages, solution, expansions = _fixture()
    source = generate_dispatch(
        solution, pipeline, stages, expansions, params_layout(stages, SM_103A)
    )

    assert "__maxnreg__(96)" in source
    assert "__launch_bounds__" not in source
    assert "__cluster_dims__(2, 1, 1)" in source
    assert "__grid_constant__ const PipelinkParams p" in source
    assert "__builtin_offsetof(PipelinkParams, stage_0000_a_producer) == 0" in source
    assert "__builtin_offsetof(PipelinkParams, stage_0001_z_consumer) == 128" in source

    assert "mbarrier.init.shared::cta.b64" in source
    assert "1013u" in source and "1014u" in source
    assert "fence.mbarrier_init.release.cluster" in source
    assert "barrier.cluster.arrive.relaxed.aligned" in source
    assert "barrier.cluster.wait.aligned" in source
    assert re.search(
        r"if \(warpgroup >= REG_GROUP_PRODUCER_WG_LO &&\s+"
        r"warpgroup < REG_GROUP_PRODUCER_WG_HI\)",
        source,
    )
    assert "pipelink_regs_config_producer_wg();" in source
    assert "pipelink_regs_config_consumer_wg();" in source

    init_helper = re.search(
        r"void pipelink_init_mbarrier\([^)]*\)\s*\{([^{}]*)\}", source
    )
    assert init_helper is not None
    assert init_helper.group(1).count("mbarrier.init.shared::cta.b64") == 1
    assert "constexpr unsigned INIT_DEPTH_ROUTE_RING = 3u;" in source

    kernel = source[source.index("void pipelink_kernel") :]
    # One thread (lane 0 of the owner's first warp, selected by thread index so the
    # guard stays warp-uniform for codegen) initializes each slot before publishing it.
    init = re.search(
        r"if \(threadIdx\.x == 32u \* ROLE_LOAD_LO\) \{\s*"
        r"for \(unsigned slot = 0; slot < INIT_DEPTH_ROUTE_RING; \+\+slot\) \{"
        r"(?P<slots>[^{}]*)\}\s*"
        r'asm volatile\("fence\.mbarrier_init\.release\.cluster;',
        kernel,
    )
    assert init is not None
    calls = re.findall(
        r"pipelink_init_mbarrier\(\s*"
        r"pipelink_smem \+ 16u \* \(static_cast<unsigned>\((\w+)\(\)\) \+ slot\),"
        r"\s*(\d+)u\);",
        init.group("slots"),
    )
    assert sorted(calls) == [
        ("pipelink_route_empty", "1014"),
        ("pipelink_route_full", "1013"),
    ]
    assert kernel.count("pipelink_init_mbarrier(") == 2
    fence = kernel.index("fence.mbarrier_init.release.cluster")
    cta_sync = kernel.index("__syncthreads();")
    cluster_arrive = kernel.index("barrier.cluster.arrive.relaxed.aligned")
    cluster_wait = kernel.index("barrier.cluster.wait.aligned")
    register_dispatch = kernel.index("if (warpgroup")
    role_dispatch = kernel.index("if (warp >= ROLE_LOAD_LO")
    assert (
        init.start()
        < fence
        < cta_sync
        < cluster_arrive
        < cluster_wait
        < register_dispatch
        < role_dispatch
    )

    assert "constexpr unsigned VARIANT_STAGE_0000_A_PRODUCER = 37u;" in source
    assert (
        "a_producer_role_load(static_cast<const void*>(p.stage_0000_a_producer), "
        "VARIANT_STAGE_0000_A_PRODUCER);"
    ) in source
    assert (
        "z_consumer_variant_selected(static_cast<const void*>(p.stage_0001_z_consumer));"
        in source
    )
    assert "41u" not in source
    assert source.count("a_producer_role_load(static_cast") == 1
    assert source.count("z_consumer_variant_selected(static_cast") == 1
    assert re.search(r"a_producer_role_load\(.*?\);\s+return;", source, re.DOTALL)
    assert re.search(r"z_consumer_variant_selected\(.*?\);\s+return;", source, re.DOTALL)


def test_dispatch_generation_is_byte_stable_and_cluster_attribute_is_conditional() -> None:
    pipeline, stages, solution, expansions = _fixture()
    params = params_layout(stages, SM_103A)
    expected = generate_dispatch(solution, pipeline, stages, expansions, params)
    permuted_solution = replace(solution, reg_groups=list(reversed(solution.reg_groups)))
    permuted_params = replace(params, entries=list(reversed(params.entries)))
    assert (
        generate_dispatch(
            permuted_solution,
            pipeline,
            list(reversed(stages)),
            list(reversed(expansions)),
            permuted_params,
        )
        == expected
    )

    dynamic = replace(solution, launch=replace(solution.launch, cluster="dynamic"))
    dynamic_source = generate_dispatch(dynamic, pipeline, stages, expansions, params)
    assert "__cluster_dims__" not in dynamic_source

    unbound_handles = {
        name: handle
        for name, handle in pipeline.handles.items()
        if name != "pipelink_launch_regs"
    }
    launch_bounds_source = generate_dispatch(
        solution,
        replace(pipeline, handles=unbound_handles),
        stages,
        expansions,
        params,
    )
    assert "__launch_bounds__(256, 1)" in launch_bounds_source
    assert "__maxnreg__" not in launch_bounds_source


def test_dispatch_rejects_ambiguous_variant_entry() -> None:
    pipeline, stages, solution, expansions = _fixture()
    params = params_layout(stages, SM_103A)


    selected_entry_stage = next(stage for stage in stages if stage.id == "z.consumer")
    ambiguous = replace(
        selected_entry_stage,
        roles={
            **selected_entry_stage.roles,
            "extra": Role("z_consumer_role_extra", 4),
        },
    )
    with pytest.raises(LinkError) as caught:
        generate_dispatch(
            solution,
            pipeline,
            [next(stage for stage in stages if stage.id == "a.producer"), ambiguous],
            expansions,
            params,
        )
    assert [failure.clause_id for failure in caught.value.failures] == ["L-LINK"]


def test_role_branches_are_nested_inside_their_register_group_branch() -> None:
    """Each setmaxnreg region must reach its roles without a control-flow join.

    ptxas derives per-region register budgets from setmaxnreg; a join after
    mixed .inc/.dec branches compiles everything after it to the smallest
    budget.
    """
    pipeline, stages, solution, expansions = _fixture()
    source = generate_dispatch(
        solution, pipeline, stages, expansions, params_layout(stages, SM_103A)
    )
    kernel = source[source.index("void pipelink_kernel") :]
    for group, role in (("PRODUCER_WG", "LOAD"), ("CONSUMER_WG", "MMA")):
        config = kernel.index(f"pipelink_regs_config_{group.lower()}();")
        role_branch = kernel.index(f"if (warp >= ROLE_{role}_LO", config)
        # The group branch closes only after its role branches.
        depth = 0
        closed_before_role = False
        for index, char in enumerate(kernel[config:role_branch]):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    closed_before_role = True
                    break
        assert not closed_before_role, f"{group} branch re-joins before ROLE_{role}"
    # Every dispatched role is emitted exactly once.
    assert kernel.count("if (warp >= ROLE_LOAD_LO") == 1
    assert kernel.count("if (warp >= ROLE_MMA_LO") == 1
