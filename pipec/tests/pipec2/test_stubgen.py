from __future__ import annotations

from dataclasses import replace
import re

import pytest

from pipec2.model import (
    Handle,
    Launch,
    LinkError,
    Pipeline,
    PipelineTopology,
    ProtocolExpansion,
    RegGroup,
    Solution,
    StageRef,
    TargetStamp,
)
from pipec2.stubgen import generate_stub


TARGET = TargetStamp("2.0-draft", "sm_103a", "lto_103", "cuda-13.3.0")


def _pipeline() -> Pipeline:
    handles = {
        handle.name: handle
        for handle in [
            Handle("pipelink_base", "base", "bind", "smem", None, "claim:base"),
            Handle("pipelink_depth", "depth", "bind", None, [2, 4], "channel:ring"),
            Handle(
                "pipelink_forced_variant",
                "variant",
                "fixed",
                None,
                [7],
                "stage:producer.stage",
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
                "pipelink_regs_consumer",
                "regs",
                "bind",
                "regs",
                [128],
                "reg_group:consumer_wg",
            ),
            Handle(
                "pipelink_regs_equal",
                "regs",
                "bind",
                "regs",
                [96],
                "reg_group:equal_wg",
            ),
            Handle(
                "pipelink_regs_producer",
                "regs",
                "bind",
                "regs",
                [64],
                "reg_group:producer_wg",
            ),
        ]
    }
    return Pipeline(
        "stub-fixture",
        [StageRef("producer.pla", "producer.stage")],
        handles,
        ["kernel"],
        [],
        {},
        ["pipelink_depth"],
        ["consumer_wg", "equal_wg", "producer_wg"],
        {"pipelink_forced_variant": 7},
        PipelineTopology(False, "static"),
        TARGET,
    )


def _solution() -> Solution:
    return Solution(
        {"pipelink_depth": 4},
        {"consumer_wg": 128, "equal_wg": 96, "producer_wg": 64},
        {"pipelink_base": 32},
        {},
        {"producer.stage": {"pipelink_forced_variant": 7}},
        [
            RegGroup(["producer"], 0, 4, "pipelink_regs_producer", 64, "dec"),
            RegGroup(["equal"], 4, 8, "pipelink_regs_equal", 96, "none"),
            RegGroup(["consumer"], 8, 12, "pipelink_regs_consumer", 128, "inc"),
        ],
        {"producer": (0, 4), "equal": (4, 8), "consumer": (8, 12)},
        Launch(384, 96, 0, "dynamic"),
        [],
        0,
    )


def _calls() -> list[ProtocolExpansion]:
    aggregate_decl = (
        "struct RouteAggregate { float value[4]; };\n"
        'extern "C" __device__ RouteAggregate consumer_aggregate(\n'
        "    RouteAggregate fragment, const float scalar);"
    )
    void_decl = (
        'extern "C" __device__ void consumer_notify('
        "const int* values, unsigned count);"
    )
    return [
        ProtocolExpansion(
            "z_notify",
            "call",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ("consumer_notify", void_decl, "2" * 64),
            None,
            None,
        ),
        ProtocolExpansion(
            "a_aggregate",
            "call",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ("consumer_aggregate", aggregate_decl, "1" * 64),
            None,
            None,
        ),
    ]


def test_generate_stub_routes_bindings_calls_and_register_acquires() -> None:
    source = generate_stub(_solution(), _pipeline(), [], _calls())

    binding_lines = re.findall(
        r'^extern "C" __device__ int (pipelink_[A-Za-z0-9_]+)\(\) \{ return (-?\d+); \}$',
        source,
        re.MULTILINE,
    )
    assert binding_lines == sorted(binding_lines)
    assert dict(binding_lines) == {
        "pipelink_base": "32",
        "pipelink_depth": "4",
        "pipelink_forced_variant": "7",
        "pipelink_launch_regs": "96",
        "pipelink_regs_consumer": "128",
        "pipelink_regs_equal": "96",
        "pipelink_regs_producer": "64",
    }

    for _symbol, decl, digest in (item.callable for item in _calls()):  # type: ignore[misc]
        assert decl in source
        assert f"decl-sha256 {digest}" in source
    assert source.index("pipelink_call_a_aggregate") < source.index("pipelink_call_z_notify")
    assert re.search(
        r'extern "C" __attribute__\(\(used\)\) __forceinline__ __device__\s+'
        r"RouteAggregate pipelink_call_a_aggregate\(.*?\) \{\s+"
        r"return consumer_aggregate\(fragment, scalar\);",
        source,
        re.DOTALL,
    )
    assert "return consumer_notify(values, count);" in source

    assert (
        'asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(128));'
        in source
    )
    assert (
        'asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(64));'
        in source
    )
    equal_body = re.search(
        r"void pipelink_regs_config_equal_wg\(\) \{(?P<body>.*?)\}",
        source,
        re.DOTALL,
    )
    assert equal_body is not None
    assert "setmaxnreg" not in equal_body.group("body")


def test_generate_stub_is_byte_stable_under_input_permutation() -> None:
    pipeline = _pipeline()
    solution = _solution()
    expected = generate_stub(solution, pipeline, [], _calls())
    permuted = replace(solution, reg_groups=list(reversed(solution.reg_groups)))
    assert generate_stub(permuted, pipeline, [], list(reversed(_calls()))) == expected
    assert expected.endswith("\n") and not expected.endswith("\n\n")


def test_call_declaration_requires_the_named_export() -> None:
    expansion = replace(
        _calls()[0],
        callable=(
            "missing_export",
            'extern "C" __device__ int another_export(int value);',
            "3" * 64,
        ),
    )
    with pytest.raises(LinkError):
        generate_stub(_solution(), _pipeline(), [], [expansion])
