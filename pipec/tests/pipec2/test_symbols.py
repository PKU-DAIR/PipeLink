from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pipec2.symbols as symbols_module
from pipec2.decls import declaration_digest
from pipec2.model import CLAUSE_IDS, Export, Role, SymbolInfo, SymbolTable
from pipec2.profile import get_profile
from pipec2.symbols import (
    TOOLCHAIN_ALLOWLIST,
    advisory_report,
    check_generated_providers,
    check_imports,
    check_pack,
    declared_imports,
    exact_import_patterns,
    import_seed,
    link_assembly_check,
    tensor_import_symbols,
)


def _table(
    *,
    strong: tuple[str, ...] = (),
    weak: tuple[str, ...] = (),
    undefined: tuple[str, ...] = (),
) -> SymbolTable:
    info = {
        **{name: SymbolInfo("func", "strong") for name in strong},
        **{name: SymbolInfo("func", "weak") for name in weak},
        **{name: SymbolInfo("func", "undefined") for name in undefined},
    }
    return SymbolTable(strong, weak, undefined, info)


def _pack_stage(
    symbol_prefix: str,
    *,
    exports: dict[str, Export] | None = None,
    roles: dict[str, Role] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(symbol_prefix=symbol_prefix, exports=exports or {}, roles=roles or {})


def test_packed_symbol_section_round_trips_kinds_and_provenance() -> None:
    table = SymbolTable(
        ("stage_entry",),
        ("weak_helper",),
        ("imported_function", "imported_variable"),
        {
            "stage_entry": SymbolInfo("func", "strong"),
            "weak_helper": SymbolInfo("func", "weak"),
            "imported_function": SymbolInfo("func", "undefined"),
            "imported_variable": SymbolInfo("object", "undefined"),
        },
        mechanism="future-probe-provenance",
    )
    assert SymbolTable.from_manifest(table.to_manifest()) == table


def test_check_pack_enforces_prefix_on_undeclared_non_pipelink_strong_globals() -> None:
    table = _table(
        strong=(
            "wrong_export",
            "org_example_stage_role_load",
            "pipelink_internal_hook",
        ),
        weak=("unprefixed_comdat",),
    )

    failures = check_pack(table, _pack_stage("org_example_stage_"))

    assert len(failures) == 1
    assert failures[0].clause_id == "L-SYM"
    assert failures[0].detail == {
        "symbol": "wrong_export",
        "required_prefix": "org_example_stage_",
    }
    assert failures[0].clause_id in CLAUSE_IDS


def test_weak_and_comdat_definitions_are_advisory_never_failures() -> None:
    table = _table(weak=("z_comdat", "a_linkonce"))

    assert check_pack(table, _pack_stage("required_prefix_")) == []
    assert advisory_report(table) == {
        "weak_comdat": [
            {"symbol": "a_linkonce", "kind": "func", "binding": "weak"},
            {"symbol": "z_comdat", "kind": "func", "binding": "weak"},
        ]
    }


def test_check_pack_admits_exact_declared_unprefixed_exports_and_role_entries() -> None:
    declaration = 'extern "C" __device__ void plx_produce(const void* raw_params);'
    stage = _pack_stage(
        "tensor_producer_",
        exports={"produce": Export("plx_produce", declaration, declaration_digest(declaration))},
        roles={"tensor_caller": Role("plx_role_tensor_caller", 4)},
    )

    assert check_pack(
        _table(strong=("plx_produce", "plx_role_tensor_caller")),
        stage,
    ) == []
    failures = check_pack(
        _table(
            strong=(
                "plx_produce_extra",
                "plx_role_tensor_caller_extra",
                "produce",
                "tensor_caller",
                "pl_import",
            ),
        ),
        stage,
    )
    assert [(failure.clause_id, failure.detail["symbol"]) for failure in failures] == [
        ("L-SYM", "pl_import"),
        ("L-SYM", "plx_produce_extra"),
        ("L-SYM", "plx_role_tensor_caller_extra"),
        ("L-SYM", "produce"),
        ("L-SYM", "tensor_caller"),
    ]


def test_declared_unprefixed_pack_exports_still_collide_at_link() -> None:
    declaration = 'extern "C" __device__ void plx_produce(const void* raw_params);'
    exports = {"produce": Export("plx_produce", declaration, declaration_digest(declaration))}
    alpha = _table(strong=("plx_produce",))
    beta = _table(strong=("plx_produce",))

    assert check_pack(alpha, _pack_stage("alpha_", exports=exports)) == []
    assert check_pack(beta, _pack_stage("beta_", exports=exports)) == []
    failures = link_assembly_check(
        {"alpha": alpha, "beta": beta},
        {"alpha": "alpha_", "beta": "beta_"},
    )
    assert [(failure.clause_id, failure.detail) for failure in failures] == [
        ("L-SYM", {"symbol": "plx_produce", "stages": ["alpha", "beta"]}),
    ]


def test_check_imports_accepts_pipelink_and_toolchain_symbol_families() -> None:
    table = _table(
        undefined=(
            "pipelink_call_output",
            "__nv_sinf",
            "__device_builtin__foo",
            "__cudaPushCallConfiguration",
            "__assertfail",
            "vprintf",
            "malloc",
            "free",
            "memcpy",
            "memset",
            "memmove",
            "llvm.nvvm.barrier0",
        )
    )

    assert check_imports(table, TOOLCHAIN_ALLOWLIST) == []


def test_check_imports_rejects_each_undeclared_external_deterministically() -> None:
    table = _table(undefined=("z_other_stage", "a_libc_call", "pipelink_ok"))

    failures = check_imports(table, TOOLCHAIN_ALLOWLIST)

    assert [failure.detail["symbol"] for failure in failures] == [
        "a_libc_call",
        "z_other_stage",
    ]
    assert all(failure.clause_id == "P-IMPORT" for failure in failures)
    assert all(failure.clause_id in CLAUSE_IDS for failure in failures)


def test_manifest_declared_call_imports_resolve_symbols_not_logical_export_keys() -> None:
    declaration = 'extern "C" __device__ void plx_release(const void* raw_params);'
    exported = Export("plx_release", declaration, declaration_digest(declaration))
    caller = SimpleNamespace(
        endpoints=[
            SimpleNamespace(
                name="tensor_release",
                direction="produce",
                protocol=SimpleNamespace(kind="call", call=SimpleNamespace(export="release")),
            ),
        ],
        exports={},
        tensor=None,
    )
    callee = SimpleNamespace(
        endpoints=[
            SimpleNamespace(
                name="tensor_release",
                direction="consume",
                protocol=SimpleNamespace(kind="call", call=SimpleNamespace(export="release")),
            ),
        ],
        exports={"release": exported},
        tensor=None,
    )

    assert declared_imports(caller) == ("pipelink_call_tensor_release",)
    assert declared_imports(callee) == ("plx_release",)
    failures = check_imports(
        _table(undefined=("release", "plx_release", "plx_release_extra")),
        exact_import_patterns(declared_imports(callee)),
    )
    assert [failure.detail["symbol"] for failure in failures] == [
        "plx_release_extra",
        "release",
    ]


def _cuda_home(tmp_path: Path) -> Path:
    cuda = tmp_path / "cuda"
    for relative in ("bin/nvcc", "lib64/libnvJitLink.so.13"):
        path = cuda / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return cuda


def test_symbol_probe_retains_declarations_not_fence_or_comment_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda = _cuda_home(tmp_path)
    ptx = """
// .visible .func comment_only() {}
/* .visible .global .u32 comment_storage; */
.visible .func (.param .b32 return_value) retained_import()
{
    fence.release.cta;
    ret;
}
.extern .func external_import();
.visible .global .align 4 .u32 retained_storage;
.visible .const .align 8 .b8 retained_constant[8] = {0};
"""

    def compile_object(_nvcc, _source, output, _env, *, lto_arch):
        output.write_bytes(b"probe object")

    def link(_inputs, *, arch, emit_ptx=False):
        return 0, "", ptx if emit_ptx else ""

    monkeypatch.setattr(symbols_module, "_compile", compile_object)
    monkeypatch.setattr(
        symbols_module,
        "_ProbeLinker",
        lambda _library: SimpleNamespace(link=link),
    )
    actual = symbols_module._undefined_symbols(
        b"stage object",
        (),
        cuda,
        {},
        {
            "release": "function",
            "comment_only": "function",
            "comment_storage": "variable",
            "return_value": "function",
            "retained_import": "function",
            "external_import": "function",
            "retained_storage": "variable",
            "retained_constant": "variable",
        },
        profile=get_profile("sm_103a"),
    )

    assert actual == {
        "retained_import": "function",
        "external_import": "function",
        "retained_storage": "variable",
        "retained_constant": "variable",
    }


class _Function:
    """Stands in for one ctypes foreign function: callable with settable argtypes/restype."""

    def __init__(self, implementation=lambda *_args: 0) -> None:
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


@pytest.mark.parametrize(
    ("profile_name", "link_arch", "probe_arch"),
    [("sm_103a", "sm_103a", "lto_103"), ("sm_90a", "sm_90a", "lto_90"), ("sm_120a", "sm_120a", "lto_120")],
)
def test_symbol_probe_arch_follows_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_name: str,
    link_arch: str,
    probe_arch: str,
) -> None:
    cuda = _cuda_home(tmp_path)
    commands: list[list[str]] = []
    link_options: list[list[bytes]] = []

    def run(command, _env, _what):
        commands.append(command)
        Path(command[-1]).write_bytes(b"probe object")
        return ""

    def create(_handle, _count, options):
        link_options.append(list(options))
        return 0

    class Library:
        def __init__(self, _path: str) -> None:
            self.nvJitLinkCreate = _Function(create)
            for name in (
                "nvJitLinkAddData",
                "nvJitLinkComplete",
                "nvJitLinkGetErrorLogSize",
                "nvJitLinkGetErrorLog",
                "nvJitLinkGetLinkedPtxSize",
                "nvJitLinkGetLinkedPtx",
                "nvJitLinkDestroy",
            ):
                setattr(self, name, _Function())

    monkeypatch.setattr(symbols_module, "_run", run)
    monkeypatch.setattr(symbols_module.ctypes, "CDLL", Library)
    actual = symbols_module._undefined_symbols(
        b"stage object",
        ("plx_defined",),
        cuda,
        {},
        profile=get_profile(profile_name),
    )

    assert actual == {}
    assert [command[0] for command in commands] == [str(cuda / "bin" / "nvcc")]
    assert [
        argument for command in commands for argument in command if argument.startswith("-arch=")
    ] == [f"-arch={probe_arch}"]
    assert [
        option for options in link_options for option in options if option.startswith(b"-arch=")
    ] == [f"-arch={link_arch}".encode()]


def test_link_assembly_rejects_duplicate_strong_globals_and_prefixes() -> None:
    tables = {
        "org.example.beta": _table(strong=("shared_global", "beta_only")),
        "org.example.alpha": _table(strong=("alpha_only", "shared_global")),
        "org.example.gamma": _table(weak=("shared_global",)),
    }
    prefixes = {
        "org.example.gamma": "gamma_",
        "org.example.beta": "collision_",
        "org.example.alpha": "collision_",
    }

    failures = link_assembly_check(tables, prefixes)

    assert len(failures) == 2
    prefix_failure = next(
        failure for failure in failures if "prefix" in failure.detail
    )
    duplicate_failure = next(
        failure for failure in failures if "symbol" in failure.detail
    )
    assert prefix_failure.clause_id == "L-SYM"
    assert prefix_failure.detail == {
        "prefix": "collision_",
        "stages": ["org.example.alpha", "org.example.beta"],
    }
    assert duplicate_failure.clause_id == "L-SYM"
    assert duplicate_failure.detail == {
        "symbol": "shared_global",
        "stages": ["org.example.alpha", "org.example.beta"],
    }


def test_link_assembly_results_do_not_depend_on_mapping_insertion_order() -> None:
    alpha = _table(strong=("duplicate",))
    beta = _table(strong=("duplicate",))
    forward = link_assembly_check(
        {"alpha": alpha, "beta": beta},
        {"alpha": "same_", "beta": "same_"},
    )
    reverse = link_assembly_check(
        {"beta": beta, "alpha": alpha},
        {"beta": "same_", "alpha": "same_"},
    )

    assert forward == reverse
    assert [(failure.clause_id, failure.path) for failure in forward] == sorted(
        (failure.clause_id, failure.path) for failure in forward
    )


def test_link_assembly_allows_shared_weak_names_and_distinct_prefixes() -> None:
    tables = {
        "alpha": _table(strong=("alpha_entry",), weak=("common_comdat",)),
        "beta": _table(strong=("beta_entry",), weak=("common_comdat",)),
    }

    assert link_assembly_check(
        tables,
        {"alpha": "alpha_", "beta": "beta_"},
    ) == []


def _tensor_stage(*symbols: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="org.consumer",
        endpoints=(),
        claims=(),
        variants=(),
        tensor=SimpleNamespace(
            imports=tuple(
                SimpleNamespace(callable=SimpleNamespace(symbol=symbol,
                    events={"ready": symbol + "_ready", "done": symbol + "_done"}))
                for symbol in symbols
            ),
        ),
    )




def test_tensor_imports_are_admitted_by_pack_without_prefix_wildcards() -> None:
    stage = _tensor_stage("pl_import")
    table = _table(strong=("consumer_entry",), undefined=("pl_import", "pl_import_ready", "pl_import_done"))
    allowlist = (
        *TOOLCHAIN_ALLOWLIST,
        *exact_import_patterns(declared_imports(stage)),
    )

    assert check_pack(table, _pack_stage("consumer_")) == []
    assert check_imports(table, allowlist) == []
    assert import_seed(stage) == {name: "function" for name in (
        "pl_import", "pl_import_ready", "pl_import_done", "pipelink_params_org_consumer")}
    failures = check_imports(
        _table(undefined=("pl_import_extra", "pl_import")),
        allowlist,
    )
    assert [(failure.clause_id, failure.detail["symbol"]) for failure in failures] == [
        ("P-IMPORT", "pl_import_extra"),
    ]


def test_early_tensor_admission_requires_no_packed_provider() -> None:
    stage = _tensor_stage("pl_import")
    table = _table(undefined=("pl_import",))
    required = {"pl_import": "consumer"}
    allowlist = (
        *TOOLCHAIN_ALLOWLIST,
        *exact_import_patterns(tensor_import_symbols(stage)),
    )

    assert check_imports(table, allowlist) == []
    assert link_assembly_check(
        {"consumer": table},
        {"consumer": "consumer_"},
        tensor_imports=required,
    ) == []
    failures = check_imports(_table(undefined=("undeclared_import",)), allowlist)
    assert [(failure.clause_id, failure.detail["symbol"]) for failure in failures] == [
        ("P-IMPORT", "undeclared_import"),
    ]


def test_early_tensor_admission_rejects_packed_strong_and_weak_providers() -> None:
    tables = {
        "packed_strong": _table(strong=("pl_import",)),
        "consumer": _table(undefined=("pl_import",)),
        "packed_weak": _table(weak=("pl_import",)),
    }
    failures = link_assembly_check(
        tables,
        {},
        tensor_imports={"pl_import": "consumer"},
    )

    assert [(failure.clause_id, failure.path, failure.detail) for failure in failures] == [
        (
            "T-LOWER",
            '$.symbols.tensor_imports["pl_import"].packed',
            {
                "symbol": "pl_import",
                "importing_stage": "consumer",
                "stages": ["packed_strong", "packed_weak"],
            },
        ),
    ]
    assert failures == link_assembly_check(
        dict(reversed(tables.items())),
        {},
        tensor_imports={"pl_import": "consumer"},
    )


def test_generated_tensor_provider_is_required_even_without_undefined_reference() -> None:
    failures = check_generated_providers(
        {"consumer": _table(strong=("consumer_entry",))},
        {},
        {"pl_import": "consumer"},
    )

    assert [(failure.clause_id, failure.path, failure.detail) for failure in failures] == [
        (
            "T-LOWER",
            '$.symbols.tensor_imports["pl_import"].generated',
            {"symbol": "pl_import", "adapters": [], "importing_stage": "consumer"},
        ),
    ]


def test_generated_tensor_provider_definitions_must_be_unique() -> None:
    failures = check_generated_providers(
        {},
        {"z_adapter": ("pl_import",), "a_adapter": ("pl_import",)},
        {"pl_import": "consumer"},
    )

    assert [(failure.clause_id, failure.detail) for failure in failures] == [
        (
            "T-LOWER",
            {
                "symbol": "pl_import",
                "adapters": ["a_adapter", "z_adapter"],
                "importing_stage": "consumer",
            },
        ),
    ]


def test_generated_tensor_helpers_cannot_collide_with_packed_definitions() -> None:
    tables = {
        "consumer": _table(strong=("consumer_helper",), undefined=("pl_import",)),
    }
    failures = check_generated_providers(
        tables,
        {"adapter": ("pl_import", "consumer_helper")},
        {"pl_import": "consumer"},
    )

    assert [(failure.clause_id, failure.detail) for failure in failures] == [
        (
            "T-LOWER",
            {
                "symbol": "consumer_helper",
                "adapters": ["adapter"],
                "stages": ["consumer"],
            },
        ),
    ]


def test_generated_tensor_provider_diagnostics_have_deterministic_order() -> None:
    tables = {
        "z_packed": _table(strong=("pl_import_z", "helper")),
        "a_packed": _table(strong=("helper",), weak=("pl_import_z",)),
    }
    adapters = {
        "z_adapter": ("pl_import_z", "helper"),
        "a_adapter": ("helper", "pl_import_z"),
    }
    required = {"pl_import_z": "z_consumer", "pl_import_a": "a_consumer"}
    forward = check_generated_providers(tables, adapters, required)
    reverse = check_generated_providers(
        dict(reversed(tables.items())),
        {name: tuple(reversed(symbols)) for name, symbols in reversed(adapters.items())},
        dict(reversed(required.items())),
    )

    assert forward == reverse
    assert [(failure.clause_id, failure.path) for failure in forward] == sorted(
        (failure.clause_id, failure.path) for failure in forward
    )
    assert all(failure.clause_id == "T-LOWER" for failure in forward)


def test_generated_tensor_providers_accept_one_definition_per_required_symbol() -> None:
    assert check_generated_providers(
        {
            "consumer_a": _table(undefined=("pl_import_a",), weak=("helper",)),
            "consumer_z": _table(undefined=("pl_import_z",)),
        },
        {
            "z_adapter": ("pl_import_z",),
            "a_adapter": ("helper", "pl_import_a"),
        },
        {"pl_import_z": "consumer_z", "pl_import_a": "consumer_a"},
    ) == []
