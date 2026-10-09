from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import pipec2.link as link_module
from pipec2.canonjson import canonical_bytes
from pipec2.decls import declaration_digest
from pipec2.launcher import _select_cluster
from pipec2.layout import load_layout
from pipec2.model import (
    CallDecl,
    Claim,
    ClusterConstraints,
    Endpoint,
    Extent,
    Handle,
    Interval,
    Launch,
    LegalitySet,
    LayoutError,
    LinkError,
    MathMode,
    NVCC_DEFAULT_MATH,
    ParamsDecl,
    ParamsLayout,
    Pipeline,
    PipelineDomain,
    PipelineTopology,
    Protocol,
    RegGroup,
    SymbolInfo,
    SymbolTable,
    Solution,
    StageManifest,
    StageRef,
    StageTopology,
    Tile,
    TargetStamp,
)
from pipec2.profile import get_profile
from pipec2.symbols import cuda_home
from pipec2.schema import load_stage_manifest


TARGET = TargetStamp("2.0-draft", "sm_103a", "lto_103", "cuda-13.3.0")
TOPOLOGY = StageTopology(
    1,
    "dynamic",
    ClusterConstraints(1, (4, 2, 1)),
    [],
    False,
    "static",
)
PARAMS = ParamsDecl(8, 8, "Opaque", "0" * 64)


def _stage(stage_id: str = "org.example") -> StageManifest:
    return StageManifest(
        stage_id,
        "1.0.0",
        stage_id.replace(".", "_") + "_",
        {},
        {},
        "void(const void* params, unsigned variant)",
        "v2",
        [],
        [],
        [],
        TOPOLOGY,
        PARAMS,
        None,
        {},
        TARGET,
    )


def _pipeline(handles: dict[str, Handle] | None = None) -> Pipeline:
    return Pipeline(
        "test",
        [],
        handles or {},
        [],
        [],
        {},
        [],
        [],
        {},
        PipelineTopology(False, "static"),
        TARGET,
    )


def _paged_stage() -> StageManifest:
    # Load an actual source manifest without pipeline context. A page claim is
    # valid stage syntax even though its physical alignment is not yet known.
    declaration = 'extern "C" void pipelink_call_page_handoff();'
    return load_stage_manifest({
        "pipelink_abi": {"schema": TARGET.schema, "target": {
            "profile": TARGET.profile, "lto_arch": TARGET.lto_arch, "toolkit": TARGET.toolkit,
        }},
        "stage": {
            "id": "org.example", "version": "1.0.0", "symbol_prefix": "org_example_",
            "entry_abi": "v2", "entry_signature": "void(const void* params, unsigned variant)",
            "roles": {"load": {"entry": "org_example_role_load", "warps": 4}},
            "exports": {}, "variants": [{"key": {}, "index": 0}],
        },
        "endpoints": [{
            "name": "page_handoff", "direction": "produce", "owner_role": "load",
            "space": "smem", "scope": "cta", "proxy": "generic",
            "tile": {"dtype": "bf16", "shape": [8, 8]},
            "layout": {
                "encoding": "pl-layout-v2", "elem_bits": 16,
                "shape": [[8], [8]], "stride": [[8], [1]],
            },
            "protocol": {
                "kind": "call", "export": "noop", "signature": "void()",
                "decl": declaration, "decl_sha256": declaration_digest(declaration),
            },
            "binds": {"base": "pipelink_page_base"},
        }],
        "claims": [{
            "domain": "smem_page", "handle": "pipelink_page_base", "lifetime": "kernel",
            "extent": {"const": 1, "per": {}}, "align": 1,
        }],
        "topology": {
            "cta_group": {"fixed": 1},
            "cluster": {"compiled": "dynamic", "launch": {"x_multiple_of": 1, "max": [4, 2, 1]}},
            "multicast": [], "persistent": False, "scheduler": "static",
        },
        "params": {"bytes": 8, "align": 8, "type": "Opaque", "sha256_of_layout_probe": "0" * 64},
    })


def _solution(
    *,
    regs_cap: int = 96,
    groups: list[RegGroup] | None = None,
    threads: int = 256,
) -> Solution:
    return Solution(
        {},
        {},
        {},
        {"smem": [Interval("base", 0, 8, "kernel")]},
        {},
        groups or [],
        {},
        Launch(threads, regs_cap, 1024, "dynamic"),
        [],
        0,
    )


def _patch_link_chain(
    monkeypatch: pytest.MonkeyPatch,
    stage: StageManifest,
    solution: Solution,
    calls: list[str],
) -> None:
    artifact = link_module._Artifact(Path("stage.plink"), stage, b"object", "a" * 64)
    monkeypatch.setattr(link_module, "_load_assembly", lambda _path: (_pipeline(), [artifact]))
    monkeypatch.setattr(link_module, "_check_symbols", lambda _artifacts: calls.append("symbols"))
    monkeypatch.setattr(link_module, "_solve_pipeline", lambda *_args: solution)
    monkeypatch.setattr(
        link_module,
        "_params_layout",
        lambda *_args: ParamsLayout(8, 8, [(stage.id, 0, 8, 8)]),
    )
    monkeypatch.setattr(
        link_module, "_generate_stub", lambda *_args: calls.append("stub") or "// stub\n"
    )


def _symbol_table(
    *,
    strong: tuple[str, ...] = (),
    weak: tuple[str, ...] = (),
    undefined: tuple[str, ...] = (),
) -> SymbolTable:
    return SymbolTable(
        strong,
        weak,
        undefined,
        {
            **{name: SymbolInfo("func", "strong") for name in strong},
            **{name: SymbolInfo("func", "weak") for name in weak},
            **{name: SymbolInfo("func", "undefined") for name in undefined},
        },
    )


def _importing_stage(stage_id: str, imports: tuple[str, ...]) -> StageManifest:
    endpoints = []
    for symbol in imports:
        declaration = f'extern "C" void pipelink_call_{symbol}();'
        endpoints.append(Endpoint(
            symbol, "produce", "work", "reg", "cta", "generic", Tile("bf16", (1,)),
            load_layout({"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [1], "stride": [1]}, "reg", "$.layout"), None,
            Protocol("call", None, None, None, None, CallDecl(
                symbol, "void()", declaration, declaration_digest(declaration)
            ), None, None),
            {},
        ))
    return replace(_stage(stage_id), endpoints=endpoints)


def test_unreferenced_exports_drop_out_with_the_endpoints_they_own() -> None:
    from pipec2.model import CounterDecl

    reg = load_layout({"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [1], "stride": [1]}, "reg", "$.layout")
    counter_layout = load_layout({"encoding": "pl-layout-v2", "elem_bits": 32, "shape": [1], "stride": [1]}, "gmem", "$.layout")

    def call(name: str, direction: str, export: str | None = None) -> Endpoint:
        declaration = f'extern "C" void pipelink_call_{name}();'
        return Endpoint(name, direction, "work" if direction == "produce" else None, "reg", "cta", "generic",
                        Tile("bf16", (1,)), reg, None,
                        Protocol("call", None, None, None, None,
                                 CallDecl(export or name, "void()", declaration, declaration_digest(declaration)),
                                 None, None), {})

    def counter(name: str, direction: str, exports: list[str] | None) -> Endpoint:
        return Endpoint(name, direction, None, "gmem", "grid", "generic", Tile("i32", (1,)), counter_layout, None,
                        Protocol("counter", None, None, None, None, None, None,
                                 CounterDecl("pipelink_slot", "red.release.gpu", "ld.acquire.gpu")), {}, exports)

    caller = replace(_stage("org.caller"), endpoints=[call("fwd", "produce"), counter("shared", "consume", None)])
    provider = replace(_stage("org.provider"), endpoints=[
        call("fwd", "consume"), call("bwd", "consume"),
        counter("bwd_only", "produce", ["bwd"]),        # owned only by the unreferenced export
        counter("shared", "produce", ["fwd", "bwd"]),   # still used by a referenced export
    ])
    with pytest.raises(LinkError):
        link_module._pair_channels([caller, provider])
    pruned, dropped = link_module._prune_unreferenced_exports([caller, provider])
    assert dropped == [
        {"stage": "org.provider", "endpoint": "bwd", "reason": "uncalled export channel"},
        {"stage": "org.provider", "endpoint": "bwd_only", "reason": "owned by unreferenced exports"},
    ]
    assert sorted(channel for channel, _, _ in link_module._pair_channels(pruned)) == ["fwd", "shared"]
    # An export stays live while any of its channels is called: the uncalled alias channel goes,
    # the endpoint associated with the export stays.
    aliased = replace(provider, endpoints=[
        call("fwd", "consume"), call("fwd_alias", "consume", export="fwd"),
        counter("fwd_only", "produce", ["fwd"]),
    ])
    caller_of_fwd = replace(caller, endpoints=[call("fwd", "produce"), counter("fwd_only", "consume", None)])
    _, dropped = link_module._prune_unreferenced_exports([caller_of_fwd, aliased])
    assert dropped == [{"stage": "org.provider", "endpoint": "fwd_alias", "reason": "uncalled export channel"}]
    # A host transfer keeps its endpoint in the owner's inventory: dropping it is an error, not a dangling reference.
    transfer = {"device_endpoint": {"stage": "org.provider", "endpoint": "bwd_only"}}
    hosted = replace(provider, host={"transfers": [transfer]})
    with pytest.raises(LinkError) as caught:
        link_module._prune_unreferenced_exports([caller, hosted])
    assert [(item.clause_id, item.path) for item in caught.value.failures] == [
        ("L-PAIR", "stages.org.provider.endpoints.bwd_only")]
    # An unresolved import still fails: a caller without an export.
    orphan = replace(_stage("org.orphan"), endpoints=[call("missing", "produce")])
    with pytest.raises(LinkError):
        link_module._pair_channels(link_module._prune_unreferenced_exports([orphan, provider])[0])


def test_assembly_validates_the_pruned_interface() -> None:
    from test_schema import _digest, _valid_pipeline, _valid_stage
    from pipec2.schema import load_pipeline, validate_assembly

    raw = _valid_stage()
    decl = 'extern "C" int producer_stage_dead_compute(int x);'
    raw["stage"]["exports"]["dead_compute"] = {"symbol": "producer_stage_dead_compute", "decl": decl,
                                                "decl_sha256": _digest(decl)}
    caller = next(item for item in raw["endpoints"] if item["name"] == "register_call")
    raw["endpoints"].append({**copy.deepcopy(caller), "name": "dead_call", "direction": "consume", "owner_role": None,
                             "protocol": {**caller["protocol"], "export": "dead_compute", "decl": decl,
                                          "decl_sha256": _digest(decl)}})
    counter = next(item for item in raw["endpoints"] if item["name"] == "grid_counter")
    # Used only by the uncalled export; its slot handle is absent from this pipeline.
    raw["endpoints"].append({**copy.deepcopy(counter), "name": "dead_counter", "exports": ["dead_compute"],
                             "protocol": {**counter["protocol"], "slot": "pipelink_dead_slot"}})
    stage = load_stage_manifest(raw)
    pipeline = load_pipeline(_valid_pipeline())
    assert [failure.clause_id for failure in validate_assembly(pipeline, [stage])] == ["L-HANDLE"]
    [artifact] = link_module._assemble(pipeline, [link_module._Artifact(Path("stage.plink"), stage, b"", "a" * 64)])
    assert "dead_counter" not in {endpoint.name for endpoint in artifact.stage.endpoints}
    assert artifact.dropped_endpoints == (
        {"stage": "producer.stage", "endpoint": "dead_call", "reason": "uncalled export channel"},
        {"stage": "producer.stage", "endpoint": "dead_counter", "reason": "owned by unreferenced exports"},
    )


def test_counter_channels_pair_many_producers_and_optional_consumers() -> None:
    from pipec2.model import CounterDecl, SolveError

    gmem = load_layout({"encoding": "pl-layout-v2", "elem_bits": 32, "shape": [1], "stride": [1]}, "gmem", "$.layout")

    def counter(name: str, direction: str, acquire: str = "ld.acquire.gpu", release: str = "red.release.gpu") -> Endpoint:
        return Endpoint(name, direction, None, "gmem", "grid", "generic", Tile("i32", (1,)), gmem, None,
                        Protocol("counter", None, None, None, None, None, None,
                                 CounterDecl("pipelink_slot", release, acquire)), {})

    fence = "ld.relaxed.gpu+fence.acquire.gpu"
    stages = [
        replace(_stage("org.a"), endpoints=[counter("done", "produce"), counter("sink", "produce")]),
        replace(_stage("org.b"), endpoints=[counter("done", "produce"), counter("done", "consume")]),
        replace(_stage("org.c"), endpoints=[counter("done", "consume", fence)]),
    ]
    pairs = link_module._pair_channels(stages)
    assert [(channel, [p.stage.id for p in producers], [c.stage.id for c in consumers])
            for channel, producers, consumers in pairs] == [
        ("done", ["org.a", "org.b"], ["org.b", "org.c"]), ("sink", ["org.a"], [])]
    pipeline = replace(_pipeline(), handles={
        "pipelink_slot": Handle("pipelink_slot", "base", "bind", "gmem_counter", None, "claim:pipelink_slot")})
    profile = get_profile("sm_103a")
    expansions = {channel: link_module._expand_protocol(channel, producers, consumers, stages, pipeline, {}, profile)
                  for channel, producers, consumers in pairs}
    # One record per channel (solve requires unique channels); acquires are the consumers' sorted set.
    assert {channel: item.counter for channel, item in expansions.items()} == {
        "done": ("pipelink_slot", "red.release.gpu", ("ld.acquire.gpu", fence)),
        "sink": ("pipelink_slot", "red.release.gpu", ()),
    }
    # Every producer and consumer still agrees on scope, slot and release.
    mixed = [stages[0], replace(stages[1], endpoints=[counter("done", "produce", release="red.release.sys")]), stages[2]]
    with pytest.raises(SolveError):
        for channel, producers, consumers in link_module._pair_channels(mixed):
            link_module._expand_protocol(channel, producers, consumers, mixed, pipeline, {}, profile)
    # ... and on the physical interface: every producer unifies with every consumer.
    link_module._unify_pairs(pairs, link_module._canonical_endpoints(stages, pipeline, profile))
    pair_layout = load_layout({"encoding": "pl-layout-v2", "elem_bits": 32, "shape": [2], "stride": [1]}, "gmem", "$.layout")
    wide = [stages[0], replace(stages[1], endpoints=[
        replace(counter("done", "produce"), tile=Tile("i32", (2,)), layout=pair_layout)]), stages[2]]
    with pytest.raises(LinkError) as caught:
        link_module._unify_pairs(link_module._pair_channels(wide), link_module._canonical_endpoints(wide, pipeline, profile))
    assert {item.path.split(".")[1] for item in caught.value.failures} == {"done"}
    # A consumer without any producer would wait forever.
    with pytest.raises(LinkError):
        link_module._pair_channels([replace(_stage("org.c"), endpoints=[counter("orphan", "consume")])])


def test_check_symbols_rejects_undeclared_import_even_when_sibling_defines_it() -> None:
    consumer = link_module._Artifact(
        Path("consumer.plink"),
        _stage("org.consumer"),
        b"consumer",
        "a" * 64,
    )
    provider = link_module._Artifact(
        Path("provider.plink"),
        _stage("org.provider"),
        b"provider",
        "b" * 64,
    )
    tables = {
        b"consumer": _symbol_table(
            undefined=("org_provider_export", "org_provider_weak")
        ),
        b"provider": _symbol_table(
            strong=("org_provider_export",),
            weak=("org_provider_weak",),
        ),
    }
    consumer = replace(consumer, stage=replace(consumer.stage, symbols=tables[b"consumer"]))
    provider = replace(provider, stage=replace(provider.stage, symbols=tables[b"provider"]))

    with pytest.raises(LinkError) as caught:
        link_module._check_symbols([consumer, provider])

    assert [failure.detail["symbol"] for failure in caught.value.failures] == [
        "org_provider_export",
        "org_provider_weak",
    ]


def test_check_symbols_resolves_call_trampolines_not_logical_export_keys() -> None:
    imports = ("org_provider_export", "org_provider_weak")
    consumer = link_module._Artifact(
        Path("consumer.plink"),
        _importing_stage("org.consumer", imports),
        b"consumer",
        "a" * 64,
    )
    provider = link_module._Artifact(
        Path("provider.plink"),
        _stage("org.provider"),
        b"provider",
        "b" * 64,
    )
    tables = {
        b"consumer": _symbol_table(
            undefined=tuple(f"pipelink_call_{symbol}" for symbol in imports),
        ),
        b"provider": _symbol_table(
            strong=("org_provider_export",),
            weak=("org_provider_weak",),
        ),
    }
    consumer = replace(consumer, stage=replace(consumer.stage, symbols=tables[b"consumer"]))
    provider = replace(provider, stage=replace(provider.stage, symbols=tables[b"provider"]))

    link_module._check_symbols([consumer, provider])
    bypass = replace(
        consumer,
        stage=replace(consumer.stage, symbols=_symbol_table(undefined=imports)),
    )
    with pytest.raises(LinkError) as caught:
        link_module._check_symbols([bypass, provider])
    assert [(failure.clause_id, failure.detail["symbol"]) for failure in caught.value.failures] == [
        ("P-IMPORT", "org_provider_export"),
        ("P-IMPORT", "org_provider_weak"),
    ]


def test_check_symbols_rejects_import_defined_nowhere() -> None:
    artifact = link_module._Artifact(
        Path("consumer.plink"),
        _importing_stage("org.consumer", ("missing_export",)),
        b"consumer",
        "a" * 64,
    )
    artifact = replace(artifact, stage=replace(
        artifact.stage,
        symbols=_symbol_table(undefined=("missing_export",)),
    ))

    with pytest.raises(LinkError) as caught:
        link_module._check_symbols([artifact])

    assert [(failure.clause_id, failure.detail["symbol"]) for failure in caught.value.failures] == [
        ("P-IMPORT", "missing_export")
    ]


def test_check_symbols_requires_repacking_legacy_artifact() -> None:
    artifact = link_module._Artifact(
        Path("legacy.plink"), _stage("org.example"), b"object", "a" * 64,
    )
    with pytest.raises(LinkError) as caught:
        link_module._check_symbols([artifact])
    assert [failure.clause_id for failure in caught.value.failures] == ["L-LINK"]


def test_canonical_endpoints_resolves_pipeline_page_scale() -> None:
    pipeline = replace(
        _pipeline(),
        domains={"smem_page": PipelineDomain(capacity=4, page_bytes=256)},
    )

    canon = link_module._canonical_endpoints(
        [_paged_stage()],
        pipeline,
        get_profile("sm_103a"),
    )

    assert canon["org.example:page_handoff:produce"].align_bits == 256 * 8


def test_canonical_endpoints_rejects_missing_pipeline_page_scale() -> None:
    pipeline = replace(
        _pipeline(),
        domains={"smem_page": PipelineDomain(capacity=4, page_bytes=None)},
    )

    with pytest.raises(LayoutError) as exc:
        link_module._canonical_endpoints(
            [_paged_stage()],
            pipeline,
            get_profile("sm_103a"),
        )

    assert [(failure.clause_id, failure.path) for failure in exc.value.failures] == [
        ("L8", "endpoints.page_handoff.layout")
    ]


def test_pipeline_declared_stage_order_controls_device_link_inputs() -> None:
    alpha = link_module._Artifact(
        Path("alpha.plink"), _stage("org.alpha"), b"alpha", "a" * 64
    )
    zeta = link_module._Artifact(
        Path("zeta.plink"), _stage("org.zeta"), b"zeta", "b" * 64
    )
    pipeline = replace(
        _pipeline(),
        stages=[
            StageRef("zeta.plink", "org.zeta"),
            StageRef("alpha.plink", "org.alpha"),
        ],
    )

    ordered = link_module._artifacts_in_pipeline_order(pipeline, [alpha, zeta])

    assert [artifact.stage.id for artifact in ordered] == ["org.zeta", "org.alpha"]


def test_compile_and_link_options_preserve_lto_target_and_register_budget(
    tmp_path: Path,
) -> None:
    profile = get_profile("sm_103a")
    compile_command = link_module._compile_command(
        Path("/cuda/bin/nvcc"), profile, Path("stub.cu"), Path("stub.o")
    )
    assert {"-rdc=true", "-dc", "-arch=lto_103"} <= set(compile_command)
    assert {"-lto", "-arch=sm_103a", "-maxrregcount=96"} <= set(
        link_module._link_options(profile, 96, NVCC_DEFAULT_MATH)
    )


def test_final_link_applies_the_stages_shared_math_mode() -> None:
    from pipec2.pack.emit import math_mode

    fast = math_mode(["-O3", "--use_fast_math", "--fmad=false"])
    assert fast == MathMode(ftz=True, prec_div=False, prec_sqrt=False, fmad=False)
    assert math_mode([]) == NVCC_DEFAULT_MATH
    stages = [replace(_stage(stage_id), target=replace(_stage(stage_id).target, math=fast))
              for stage_id in ("org.a", "org.b")]
    assert {"-ftz=1", "-prec-div=0", "-prec-sqrt=0", "-fma=0"} <= set(
        link_module._link_options(get_profile("sm_103a"), 96, link_module._math_mode(stages, None)))
    # A stage without recorded switches was compiled with nvcc's defaults, which differ.
    mixed = [*stages, _stage("org.c")]
    with pytest.raises(LinkError) as caught:
        link_module._math_mode(mixed, None)
    assert [(item.clause_id, item.path) for item in caught.value.failures] == [("L-LINK", "stages.target.math")]
    # The pipeline's target.math chooses the image's mode over disagreeing stages.
    assert link_module._math_mode(mixed, NVCC_DEFAULT_MATH) == NVCC_DEFAULT_MATH


def test_compile_link_and_resource_commands_use_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    link_calls: list[tuple[Path, list[tuple[str, bytes]], list[str]]] = []

    class FakeNvJitLink:
        def __init__(self, library: Path) -> None:
            self.library = library

        def link(
            self,
            inputs: list[tuple[str, bytes]],
            options: list[str],
        ) -> tuple[bytes, str, str]:
            link_calls.append((self.library, inputs, options))
            return b"linked-cubin", "info log", ""


    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        if command[0].endswith("/nvcc"):
            Path(command[-1]).write_bytes(b"generated-object")
        output = (
            "Function pipelink_kernel:\n REG:72 STACK:0 SHARED:0 LOCAL:0\n"
            if "--dump-resource-usage" in command
            else ""
        )
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setenv("PIPEC_CUDA", "/cuda")
    monkeypatch.delenv("PIPEC_SASS_REPORT", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(link_module, "_NvJitLink", FakeNvJitLink)
    zeta = _stage("org.zeta")
    alpha = _stage("org.alpha")
    artifacts = [
        link_module._Artifact(Path("zeta.plink"), zeta, b"zeta", "b" * 64),
        link_module._Artifact(Path("alpha.plink"), alpha, b"alpha", "a" * 64),
    ]
    kernel, observed, report = link_module._compile_and_measure(
        tmp_path,
        artifacts,
        {"binding_stub.cu": "// stub\n", "dispatch.cu": "// dispatch\n"},
        get_profile("sm_103a"),
        _solution(),
        "pipelink_kernel",
        NVCC_DEFAULT_MATH,
    )
    assert (kernel, observed, report) == ("pipelink_kernel", (72, 0, 0), None)
    assert all("-dc" in command for command in commands[:2])
    assert commands[2][-2:] == [
        "--dump-resource-usage",
        str(tmp_path / "pipeline.cubin"),
    ]
    assert len(link_calls) == 1
    library, inputs, options = link_calls[0]
    assert library == Path("/cuda/lib64/libnvJitLink.so.13")
    assert [name for name, _data in inputs] == [
        "000-org.zeta.o",
        "001-org.alpha.o",
        "binding_stub.o",
        "dispatch.o",
    ]
    assert {"-lto", "-arch=sm_103a", "-maxrregcount=96"} <= set(options)
    assert (tmp_path / "pipeline.cubin").read_bytes() == b"linked-cubin"


def test_all_v2_generates_dispatch_and_canonical_link_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    stage = _stage()
    solution = _solution()
    _patch_link_chain(monkeypatch, stage, solution, calls)
    monkeypatch.setattr(
        link_module,
        "_generate_dispatch",
        lambda *_args: calls.append("dispatch") or "// dispatch\n",
    )
    observed_sources: dict[str, str] = {}

    def compile_link(
        _out: Path,
        _artifacts: object,
        sources: dict[str, str],
        _profile: object,
        _solution: object,
        expected: str,
        _math: object,
    ) -> tuple[str, tuple[int, int, int], None]:
        observed_sources.update(sources)
        assert expected == "pipelink_kernel"
        return "pipelink_kernel", (96, 0, 0), None

    monkeypatch.setattr(link_module, "_compile_and_measure", compile_link)
    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert calls == ["symbols", "stub", "dispatch"]
    assert sorted(observed_sources) == ["binding_stub.cu", "dispatch.cu"]
    assert result.generated_hashes == {
        name: link_module.sha256_hex(source.encode())
        for name, source in sorted(observed_sources.items())
    }
    assert (tmp_path / "out" / "link.json").read_bytes() == canonical_bytes(result.to_json_obj())
    assert "tensor" not in result.to_json_obj()




def _register_pipeline() -> Pipeline:
    return _pipeline(
        {
            "pipelink_launch_regs": Handle(
                "pipelink_launch_regs", "regs", "bind", "regs", [48, 96], "launch"
            ),
            "pipelink_regs_group": Handle(
                "pipelink_regs_group", "regs", "bind", "regs", [48, 96], "reg_group:g"
            ),
        }
    )


def _register_solution(value: int, cap: int) -> Solution:
    return _solution(
        regs_cap=cap,
        groups=[RegGroup(["work"], 0, 8, "pipelink_regs_group", value, "none")],
    )


def test_observed_register_violation_resolves_once_under_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _stage()
    artifact = link_module._Artifact(Path("stage.plink"), stage, b"object", "c" * 64)
    pipeline = _register_pipeline()
    solutions = iter([_register_solution(96, 96), _register_solution(48, 48)])
    solved_pipelines: list[Pipeline] = []
    monkeypatch.setattr(link_module, "_load_assembly", lambda _path: (pipeline, [artifact]))
    monkeypatch.setattr(link_module, "_check_symbols", lambda _artifacts: None)

    def solve(candidate: Pipeline, *_args: object) -> Solution:
        solved_pipelines.append(candidate)
        return next(solutions)

    monkeypatch.setattr(link_module, "_solve_pipeline", solve)
    monkeypatch.setattr(link_module, "_generate_stub", lambda *_args: "// stub\n")
    monkeypatch.setattr(link_module, "_generate_dispatch", lambda *_args: "// dispatch\n")
    monkeypatch.setattr(
        link_module,
        "_compile_and_measure",
        lambda *_args: ("pipelink_kernel", (48, 0, 0), None),
    )
    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert len(solved_pipelines) == 2
    assert solved_pipelines[1].handles["pipelink_launch_regs"].values == [48]
    assert result.regs_cap == 48
    assert result.bindings == {
        "pipelink_launch_regs": (48, "solved"),
        "pipelink_regs_group": (48, "solved"),
    }


def test_observed_register_violation_after_second_link_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _stage()
    artifact = link_module._Artifact(Path("stage.plink"), stage, b"object", "d" * 64)
    pipeline = _register_pipeline()
    monkeypatch.setattr(link_module, "_load_assembly", lambda _path: (pipeline, [artifact]))
    monkeypatch.setattr(link_module, "_check_symbols", lambda _artifacts: None)
    monkeypatch.setattr(
        link_module, "_solve_pipeline", lambda *_args: _register_solution(96, 96)
    )
    monkeypatch.setattr(link_module, "_generate_stub", lambda *_args: "// stub\n")
    monkeypatch.setattr(link_module, "_generate_dispatch", lambda *_args: "// dispatch\n")
    monkeypatch.setattr(
        link_module,
        "_compile_and_measure",
        lambda *_args: ("pipelink_kernel", (48, 0, 0), None),
    )
    with pytest.raises(LinkError) as exc:
        link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert [failure.clause_id for failure in exc.value.failures] == ["L-REGS"]
    assert not (tmp_path / "out" / "link.json").exists()


def test_resource_usage_is_kernel_specific_and_fail_closed() -> None:
    output = """
Resource usage:
 Function helper:
  REG:12 STACK:3 LOCAL:4
 Function pipelink_kernel:
  REG:75 STACK:0 SHARED:1024 LOCAL:8
"""
    assert link_module._parse_resource_usage(output, "pipelink_kernel") == (
        "pipelink_kernel",
        (75, 0, 8),
    )
    with pytest.raises(LinkError):
        link_module._parse_resource_usage(output, "missing")




def test_launcher_validates_fixed_and_dynamic_cluster_legality() -> None:
    legality = LegalitySet(2, (4, 2, 1))
    assert _select_cluster(legality, None) == (2, 1, 1)
    assert _select_cluster(legality, (4, 2, 1)) == (4, 2, 1)
    with pytest.raises(LinkError):
        _select_cluster(legality, (3, 1, 1))
    assert _select_cluster((2, 1, 1), None) == (2, 1, 1)
    with pytest.raises(LinkError):
        _select_cluster((2, 1, 1), (1, 1, 1))




@pytest.fixture
def cuda_toolchain() -> Path:
    try:
        cuda = cuda_home()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if not (cuda / "bin" / "nvcc").is_file():
        pytest.skip("configured CUDA compiler is unavailable")
    return cuda




def test_nvjitlink_lto_device_link_produces_cubin(tmp_path: Path, cuda_toolchain: Path) -> None:
    if not (cuda_toolchain / "lib64" / "libnvJitLink.so.13").is_file():
        pytest.skip("configured nvJitLink library is unavailable")
    profile = get_profile("sm_103a")
    nvcc = cuda_toolchain / "bin" / "nvcc"
    provider_source = tmp_path / "provider.cu"
    kernel_source = tmp_path / "kernel.cu"
    provider_object = tmp_path / "provider.o"
    kernel_object = tmp_path / "kernel.o"
    cubin = tmp_path / "linked.cubin"
    provider_source.write_text(
        'extern "C" __device__ int helper() { return 7; }\n',
        encoding="utf-8",
        newline="\n",
    )
    kernel_source.write_text(
        'extern "C" __device__ int helper();\n'
        'extern "C" __global__ void pipelink_kernel(int* out) { *out = helper(); }\n',
        encoding="utf-8",
        newline="\n",
    )
    env = link_module._tool_env(cuda_toolchain)
    for source, output in (
        (provider_source, provider_object),
        (kernel_source, kernel_object),
    ):
        subprocess.run(
            link_module._compile_command(nvcc, profile, source, output),
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    cubin_bytes, _info_log, _error_log = link_module._NvJitLink(
        cuda_toolchain / "lib64" / "libnvJitLink.so.13"
    ).link(
        [
            (provider_object.name, provider_object.read_bytes()),
            (kernel_object.name, kernel_object.read_bytes()),
        ],
        link_module._link_options(profile, 96, NVCC_DEFAULT_MATH),
    )
    cubin.write_bytes(cubin_bytes)
    assert cubin.read_bytes().startswith(b"\x7fELF")


FIXTURES = Path(__file__).parent / "fixtures" / "link"


@pytest.mark.parametrize("profile_name", ["sm_90a", "sm_120a"])
def test_two_stage_warp_profile_pack_and_link_produces_profile_cubin(
    tmp_path: Path, cuda_toolchain: Path, profile_name: str
) -> None:
    """pack --profile -> link on a TMEM-free profile: every arch spelling flows from the profile, no GPU."""
    from pipec2 import container
    from pipec2.pack import pack_cu

    if (
        not (cuda_toolchain / "lib64" / "libnvJitLink.so.13").is_file()
        or shutil.which("nm") is None
    ):
        pytest.skip("configured CUDA link tools are unavailable")
    profile = get_profile(profile_name)
    target = {"profile": profile_name, "lto_arch": profile.arch.lto, "toolkit": "cuda-13.3.0"}
    schemas = set()
    for name in ("provider", "caller"):
        pack_cu(FIXTURES / f"sm90_{name}.cu", tmp_path / f"{name}.plink", profile=profile_name)
        manifest, _payload = container.read(tmp_path / f"{name}.plink")
        assert manifest["pipelink_abi"]["target"] == {**target, "math": NVCC_DEFAULT_MATH.to_json_obj()}
        assert manifest["topology"]["cta_group"] == {"fixed": 1}
        assert manifest["topology"]["scheduler"] == "static"
        schemas.add(manifest["pipelink_abi"]["schema"])
    (schema,) = schemas
    (tmp_path / "pipeline.json").write_bytes(canonical_bytes({
        "pipelink_abi": {"schema": schema, "target": target},
        "pipeline": {
            "name": "sm90-two-stage",
            "stages": [
                {"stage_id": "org.sm90.provider", "artifact_path": "provider.plink"},
                {"stage_id": "org.sm90.caller", "artifact_path": "caller.plink"},
            ],
            "handles": {
                "pipelink_call_warps": {
                    "class": "base", "binding": "bind", "domain": "warp", "owner": "claim:call",
                },
            },
            "phases": ["call"],
            "exclusive": [],
            "domains": {},
            "depth_priority": [],
            "regs_priority": [],
            "forced_bindings": {},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }))

    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")

    assert result.kernel_symbol == "pipelink_kernel"
    assert [expansion.channel for expansion in result.protocol_expansions] == ["scale"]
    cubin = tmp_path / "out" / "pipeline.cubin"
    assert cubin.read_bytes().startswith(b"\x7fELF")
    elf = subprocess.run(
        [str(cuda_toolchain / "bin" / "cuobjdump"), "--dump-elf", str(cubin)],
        env=link_module._tool_env(cuda_toolchain),
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ).stdout
    header = next(line for line in elf.splitlines() if line.startswith("64-bit ELF:"))
    assert f"sm={profile_name.removeprefix('sm_')}," in header
    assert f"-maxrregcount {result.regs_cap} -arch {profile_name}" in elf
    # Observed-REG gate ran against the profile's register granularity.
    observed_reg, _stack, _local = result.observed
    assert 0 < link_module._observed_register_allocation(observed_reg, profile) <= result.regs_cap
    assert result.regs_cap % profile.caps.regs_step == 0
    link_json = json.loads((tmp_path / "out" / "link.json").read_text(encoding="utf-8"))
    assert link_json["observed"] == list(result.observed)
    assert link_json["regs_cap"] == result.regs_cap


@pytest.mark.parametrize(("scope", "arrival_count", "cluster_x"), [
    ("cta", 1, 1),
    ("cluster", 2, 2),
])
def test_two_stage_sm90_signal_pack_and_link(
    tmp_path: Path, cuda_toolchain: Path, scope: str, arrival_count: int, cluster_x: int
) -> None:
    """Link a full-only completion barrier, including cluster publication, without a GPU."""
    from pipec2 import container
    from pipec2.pack import pack_cu

    if (
        not (cuda_toolchain / "lib64" / "libnvJitLink.so.13").is_file()
        or shutil.which("nm") is None
    ):
        pytest.skip("configured CUDA link tools are unavailable")
    target = {"profile": "sm_90a", "lto_arch": "lto_90a", "toolkit": "cuda-13.3.0"}
    full = "pipelink_signal_full"
    schemas = set()
    for role in ("producer", "consumer"):
        artifact = tmp_path / f"{role}.plink"
        pack_cu(
            FIXTURES / f"sm90_signal_{role}.cu",
            artifact,
            profile="sm_90a",
            defines=(
                f'SIGNAL_SCOPE="{scope}"',
                f"SIGNAL_ARRIVAL_COUNT={arrival_count}",
                f"SIGNAL_CLUSTER_X={cluster_x}",
            ),
        )
        manifest, _payload = container.read(artifact)
        assert manifest["pipelink_abi"]["target"] == {**target, "math": NVCC_DEFAULT_MATH.to_json_obj()}
        schemas.add(manifest["pipelink_abi"]["schema"])
        (endpoint,) = manifest["endpoints"]
        assert endpoint["space"] == "none"
        assert endpoint["scope"] == scope
        assert endpoint["binds"] == {"full": full}
        assert "tile" not in endpoint and "layout" not in endpoint
        (claim,) = [claim for claim in manifest["claims"] if claim["domain"] == "mbarrier"]
        assert claim["handle"] == full
        assert claim["extent"] == {"const": 1, "per": {}}
    (schema,) = schemas
    (tmp_path / "pipeline.json").write_bytes(canonical_bytes({
        "pipelink_abi": {"schema": schema, "target": target},
        "pipeline": {
            "name": f"sm90-signal-{scope}",
            "stages": [
                {"stage_id": f"org.sm90.signal.{role}", "artifact_path": f"{role}.plink"}
                for role in ("producer", "consumer")
            ],
            "handles": {
                full: {
                    "class": "base", "binding": "bind", "domain": "mbarrier",
                    "owner": "claim:signal_full",
                },
                **{
                    f"pipelink_{role}_warps": {
                        "class": "base", "binding": "bind", "domain": "warp",
                        "owner": f"claim:{role}",
                    }
                    for role in ("producer", "consumer")
                },
            },
            "phases": ["producer", "consumer"],
            "exclusive": [],
            "domains": {},
            "depth_priority": [],
            "regs_priority": [],
            "forced_bindings": {},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }))

    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")

    (expansion,) = result.protocol_expansions
    assert expansion.channel == "completion"
    assert expansion.kind == "mbarrier"
    assert expansion.tx_bytes_producer == expansion.tx_bytes_consumer == 0
    assert expansion.arrive_full == arrival_count
    assert expansion.arrive_empty is None
    assert expansion.init_slots == (full, None)
    assert expansion.init_owner == "producer"
    assert expansion.init_materializer == "dispatch"
    dispatch = (tmp_path / "out" / "generated" / "dispatch.cu").read_text(encoding="utf-8")
    assert dispatch.count("mbarrier.init.shared::cta.b64") == 1
    kernel = dispatch[dispatch.index("void pipelink_kernel") :]
    assert kernel.count("pipelink_init_mbarrier(") == 1
    assert re.findall(
        r"pipelink_init_mbarrier\(\s*"
        r"pipelink_smem \+ 16u \* \(static_cast<unsigned>\((\w+)\(\)\) \+ slot\),"
        r"\s*(\d+)u\);",
        kernel,
    ) == [(full, str(arrival_count))]
    assert "constexpr unsigned INIT_DEPTH_COMPLETION = 1u;" in dispatch
    if scope == "cluster":
        assert "__cluster_dims__(2, 1, 1)" in dispatch
        assert (
            kernel.index("pipelink_init_mbarrier(")
            < kernel.index("fence.mbarrier_init.release.cluster")
            < kernel.index("barrier.cluster.arrive.relaxed.aligned")
            < kernel.index("barrier.cluster.wait.aligned")
        )

    cubin = tmp_path / "out" / "pipeline.cubin"
    assert cubin.read_bytes().startswith(b"\x7fELF")
    elf = subprocess.run(
        [str(cuda_toolchain / "bin" / "cuobjdump"), "--dump-elf", str(cubin)],
        env=link_module._tool_env(cuda_toolchain),
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ).stdout
    header = next(line for line in elf.splitlines() if line.startswith("64-bit ELF:"))
    assert "sm=90a," in header
    assert f"-maxrregcount {result.regs_cap} -arch sm_90a" in elf
