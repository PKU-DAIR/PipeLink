from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .model import (
    SYMBOL_MECHANISM as MECHANISM,
    ClauseFailure,
    PackError,
    StageManifest,
    SymbolInfo,
    SymbolTable,
    register_clause,
)
from .profile import Profile


ALLOWLIST_VERSION = "cuda-13.3-device-runtime-v1"
# Carried byte-for-byte from the legacy symbol probe. These are
# regular-expression patterns because the CUDA device-runtime surface includes symbol families.
TOOLCHAIN_ALLOWLIST = (
    r"^__nv_",
    r"^__device_builtin__",
    r"^__cuda",
    r"^__assertfail$",
    r"^vprintf$",
    r"^malloc$",
    r"^free$",
    r"^mem(?:cpy|set|move)$",
    r"^llvm\.",
)
NVJITLINK_INPUT_OBJECT = 5
_UNRESOLVED_RE = re.compile(
    r"Unresolved extern (function|variable) ['\"]([^'\"]+)['\"]"
)
_HOST_STUB_RE = re.compile(r"(?:^|__)device_stub__")
_C_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_PTX_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_PTX_DECLARATION_RE = re.compile(
    r"^[ \t]*(?:(?:\.visible|\.extern|\.weak|\.common)\s+)*"
    r"(?:\.func\s+(?:\([^)]*\)\s*)?"
    r"|\.(?:global|const)\s+(?:\.align\s+\d+\s+)?"
    r"(?:\.v[24]\s+)?\.(?:[busf]\d+|pred)\s+)"
    r"([A-Za-z_$][A-Za-z0-9_$]*)(?=\s*[(\[=;])",
    re.MULTILINE,
)

register_clause(
    "P-SYMBOLS",
    phase="pack",
    description="artifact symbol extraction completes successfully",
)


PARAMS_BASE_HANDLE = "pipelink_internal_params_base_cell"
# Cell layout: the params-image address (8 bytes).
PARAMS_BASE_BYTES = 8


def params_accessor_symbol(stage_id: str) -> str:
    """Canonical, stage-owned parameter accessor symbol."""
    return "pipelink_params_" + re.sub(r"[^0-9A-Za-z]", "_", stage_id)


def params_accessor_stages(stages: Iterable[StageManifest]) -> list[StageManifest]:
    return [
        stage for stage in sorted(stages, key=lambda stage: stage.id)
        if stage.symbols is not None
        and params_accessor_symbol(stage.id) in stage.symbols.undefined
    ]


def host_view_stage(stage: StageManifest) -> bool:
    """A role-less member of a host params group with residue or shared scalars.

    It has no dispatched role view; each call into it passes a `pipelink_host_view`
    of its owner's records as the export's first argument (a view call).
    """
    host = stage.host
    return (
        host is not None and host.get("params_group") is not None and not stage.roles
        and (host["residue"] is not None or host["scalars"] is not None)
    )


def check_params_accessors(stages: Iterable[StageManifest]) -> list[ClauseFailure]:
    failures = []
    owners: dict[str, str] = {}
    for stage in sorted(stages, key=lambda stage: stage.id):
        expected = params_accessor_symbol(stage.id)
        if expected in owners:
            failures.append(ClauseFailure(
                "L-PARAMS", stage.id, "canonical params accessor collision",
                {"symbol": expected, "stages": [owners[expected], stage.id]},
            ))
        owners[expected] = stage.id
        if stage.symbols is None:
            continue
        for symbol in stage.symbols.undefined:
            if not symbol.startswith("pipelink_params_"):
                continue
            if host_view_stage(stage):
                failures.append(ClauseFailure(
                    "L-PARAMS", stage.id,
                    "a role-less host-group stage receives its owner's view as the first "
                    "argument of each called export, not through a params accessor",
                    {"symbol": symbol},
                ))
            elif symbol != expected or not stage.params.bytes:
                failures.append(ClauseFailure(
                    "L-PARAMS", stage.id,
                    "params accessor must reference the stage's own nonzero params",
                    {"symbol": symbol, "expected": expected, "bytes": stage.params.bytes},
                ))
        for symbol in (*stage.symbols.defined_strong, *stage.symbols.defined_weak):
            if symbol.startswith("pipelink_params_"):
                failures.append(ClauseFailure(
                    "L-PARAMS", stage.id, "params accessors are linker-owned",
                    {"symbol": symbol},
                ))
    return failures

def cuda_home(env: Mapping[str, str] = os.environ) -> Path:
    configured = env.get("PIPEC_CUDA") or env.get("PIPELINK_CUDA_HOME")
    if not configured:
        raise RuntimeError("CUDA toolchain not configured; source tools/env.sh or set PIPEC_CUDA")
    return Path(configured)


def _raise_extract_error(message: str) -> None:
    raise PackError([ClauseFailure("P-SYMBOLS", "$.symbols", message, {})])


class _ProbeLinker:
    def __init__(self, library: Path) -> None:
        try:
            self.lib = ctypes.CDLL(str(library))
        except OSError as exc:
            _raise_extract_error(f"cannot load pinned nvJitLink library: {exc}")
        self.create = self.lib.nvJitLinkCreate
        self.create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.create.restype = ctypes.c_int
        self.add_data = self.lib.nvJitLinkAddData
        self.add_data.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]
        self.add_data.restype = ctypes.c_int
        self.complete = self.lib.nvJitLinkComplete
        self.complete.argtypes = [ctypes.c_void_p]
        self.complete.restype = ctypes.c_int
        self.get_error_size = self.lib.nvJitLinkGetErrorLogSize
        self.get_error_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.get_error_size.restype = ctypes.c_int
        self.get_error = self.lib.nvJitLinkGetErrorLog
        self.get_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.get_error.restype = ctypes.c_int
        self.get_ptx_size = self.lib.nvJitLinkGetLinkedPtxSize
        self.get_ptx_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.get_ptx_size.restype = ctypes.c_int
        self.get_ptx = self.lib.nvJitLinkGetLinkedPtx
        self.get_ptx.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.get_ptx.restype = ctypes.c_int
        self.destroy = self.lib.nvJitLinkDestroy
        self.destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.destroy.restype = ctypes.c_int

    def _error_log(self, handle: ctypes.c_void_p) -> str:
        size = ctypes.c_size_t()
        if self.get_error_size(handle, ctypes.byref(size)) != 0 or size.value == 0:
            return ""
        buffer = ctypes.create_string_buffer(size.value)
        if self.get_error(handle, buffer) != 0:
            return ""
        return buffer.value.decode("utf-8", errors="replace")

    def link(
        self, inputs: list[tuple[str, bytes]], *, arch: str, emit_ptx: bool = False
    ) -> tuple[int, str, str]:
        options = [b"-lto", f"-arch={arch}".encode(), b"-O0"]
        if emit_ptx:
            options.append(b"-ptx")
        option_array = (ctypes.c_char_p * len(options))(*options)
        handle = ctypes.c_void_p()
        result = self.create(ctypes.byref(handle), len(options), option_array)
        if result != 0:
            _raise_extract_error(f"nvJitLinkCreate returned result {result}")
        keepalive: list[Any] = []
        try:
            for name, data in inputs:
                buffer = ctypes.create_string_buffer(data, len(data))
                keepalive.append(buffer)
                result = self.add_data(
                    handle,
                    NVJITLINK_INPUT_OBJECT,
                    buffer,
                    len(data),
                    name.encode("utf-8"),
                )
                if result != 0:
                    _raise_extract_error(
                        f"nvJitLinkAddData({name}) returned result {result}: "
                        f"{self._error_log(handle).rstrip()}"
                    )
            result = self.complete(handle)
            error_log = self._error_log(handle).rstrip()
            ptx = ""
            if result == 0 and emit_ptx:
                size = ctypes.c_size_t()
                if self.get_ptx_size(handle, ctypes.byref(size)) != 0:
                    _raise_extract_error("nvJitLinkGetLinkedPtxSize failed")
                buffer = ctypes.create_string_buffer(size.value)
                if self.get_ptx(handle, buffer) != 0:
                    _raise_extract_error("nvJitLinkGetLinkedPtx failed")
                ptx = buffer.value.decode("utf-8")
            return result, error_log, ptx
        finally:
            if handle.value is not None:
                self.destroy(ctypes.byref(handle))


def _run(command: list[str], env: Mapping[str, str], what: str) -> str:
    try:
        completed = subprocess.run(
            command,
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        _raise_extract_error(f"cannot execute {what}: {exc}")
    if completed.returncode != 0:
        _raise_extract_error(
            f"{what} failed ({completed.returncode}): {completed.stdout.rstrip()}"
        )
    return completed.stdout


def _external_definitions(
    object_path: Path,
    env: Mapping[str, str],
) -> tuple[set[str], set[str]]:
    nm = env.get("PIPEC_NM", "nm")
    output = _run(
        [nm, "-g", "--defined-only", "--format=posix", str(object_path)],
        env,
        "nm definition enumeration",
    )
    strong: set[str] = set()
    weak: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        name, symbol_type = fields[0], fields[1]
        if _HOST_STUB_RE.search(name) or name.startswith("_GLOBAL__sub_I_"):
            continue
        # Legacy device-companion rule: external CUDA device functions have a
        # companion global text symbol; weak/linkonce companions use W. Host-side
        # V/D/B metadata is deliberately excluded.
        if symbol_type == "T":
            strong.add(name)
        elif symbol_type == "W":
            weak.add(name)
    weak.difference_update(strong)
    return strong, weak


def _consumer_source(definitions: Sequence[str]) -> str:
    lines = [
        "template <class T>",
        "__device__ __noinline__ void pipelink_keep_symbol(T value) {",
        '  asm volatile("" : : "l"(value));',
        "}",
    ]
    aliases: list[str] = []
    for index, symbol in enumerate(definitions):
        if _C_IDENTIFIER_RE.fullmatch(symbol):
            alias = symbol
            lines.append(f'extern "C" __device__ void {alias}();')
        else:
            alias = f"pipelink_probe_ref_{index}"
            lines.append(
                f'extern "C" __device__ void {alias}() asm({json.dumps(symbol)});'
            )
        aliases.append(alias)
    lines.append('extern "C" __global__ void pipelink_probe_consumer() {')
    lines.extend(f"  pipelink_keep_symbol(&{alias});" for alias in aliases)
    lines.extend(["}", ""])
    return "\n".join(lines)


def _provider_source(symbols: Mapping[str, str]) -> str:
    lines: list[str] = []
    for index, symbol in enumerate(sorted(symbols)):
        kind = symbols[symbol]
        simple = _C_IDENTIFIER_RE.fullmatch(symbol) is not None
        alias = symbol if simple else f"pipelink_probe_def_{index}"
        asm_label = "" if simple else f" asm({json.dumps(symbol)})"
        if kind == "variable":
            if asm_label:
                lines.append(f'extern "C" __device__ int {alias}{asm_label};')
            lines.append(f'extern "C" __device__ int {alias} = 0;')
        else:
            if asm_label:
                lines.append(
                    f'extern "C" __device__ __noinline__ int {alias}(){asm_label};'
                )
            lines.extend(
                [
                    f'extern "C" __device__ __noinline__ int {alias}() {{',
                    "  return 0;",
                    "}",
                ]
            )
    return "\n".join(lines) + ("\n" if lines else "")


def _compile(
    nvcc: Path, source: Path, output: Path, env: Mapping[str, str], *, lto_arch: str
) -> None:
    _run(
        [
            str(nvcc),
            "-std=c++17",
            "-O0",
            "-rdc=true",
            "-dc",
            f"-arch={lto_arch}",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--frandom-seed=0x50495052",
            str(source),
            "-o",
            str(output),
        ],
        env,
        "symbol-probe compile",
    )


def _undefined_symbols(
    object_bytes: bytes,
    definitions: Sequence[str],
    cuda: Path,
    env: Mapping[str, str],
    seed: Mapping[str, str] | None = None,
    *,
    profile: Profile,
) -> dict[str, str]:
    nvcc = cuda / "bin" / "nvcc"
    library = cuda / "lib64" / "libnvJitLink.so.13"
    if not nvcc.is_file() or not library.is_file():
        _raise_extract_error("pinned nvcc or nvJitLink library is unavailable")

    with tempfile.TemporaryDirectory(prefix="pipec2-symprobe-") as temp_name:
        temp = Path(temp_name)
        inputs = [("stage.o", object_bytes)]
        if definitions:
            consumer_source = temp / "consumer.cu"
            consumer_object = temp / "consumer.o"
            consumer_source.write_text(
                _consumer_source(definitions),
                encoding="utf-8",
                newline="\n",
            )
            _compile(nvcc, consumer_source, consumer_object, env, lto_arch=profile.arch.lto_base)
            inputs.append(("consumer.o", consumer_object.read_bytes()))

        seeded = {
            name: kind for name, kind in (seed or {}).items() if name not in definitions
        }
        if seeded:
            provider_source = temp / "seed.cu"
            provider_object = temp / "seed.o"
            provider_source.write_text(_provider_source(seeded), encoding="utf-8")
            _compile(nvcc, provider_source, provider_object, env, lto_arch=profile.arch.lto_base)
            inputs.append(("seed.o", provider_object.read_bytes()))

        linker = _ProbeLinker(library)
        undefined: dict[str, str] = {}
        # The pinned linker reports one unresolved symbol at a time. Iterative
        # providers expose the next one; the finite bound makes corruption and
        # diagnostic drift fail closed.
        for _ in range(4097):
            result, error_log, ptx = linker.link(inputs, arch=profile.arch.sm)
            if result == 0:
                # PTX-only linking does not diagnose unresolved externs, so it
                # must follow the successful normal link above. CUDA 13.3 drops
                # unused function stubs but retains variable storage: explicit
                # variable seeds record the admitted set under the v2 mechanism.
                if seeded:
                    ptx_result, ptx_error, ptx = linker.link(
                        inputs, arch=profile.arch.sm, emit_ptx=True
                    )
                    if ptx_result != 0:
                        _raise_extract_error(f"linked PTX extraction failed: {ptx_error}")
                # Only declarations establish that a seed survived linking.
                # Instruction modifiers such as fence.release are not symbols.
                retained = {
                    match.group(1)
                    for match in _PTX_DECLARATION_RE.finditer(
                        _PTX_COMMENT_RE.sub(" ", ptx)
                    )
                }
                return {
                    **{name: kind for name, kind in seeded.items() if name in retained},
                    **undefined,
                }
            matches = _UNRESOLVED_RE.findall(error_log)
            if not matches:
                _raise_extract_error(
                    "nvJitLink failed without a parseable unresolved-symbol "
                    f"diagnostic: {error_log}"
                )
            progress = False
            for kind, symbol in matches:
                previous = undefined.get(symbol)
                if previous is not None and previous != kind:
                    _raise_extract_error(
                        f"nvJitLink reported inconsistent kinds for symbol {symbol!r}"
                    )
                if previous is None:
                    undefined[symbol] = kind
                    progress = True
                    provider_source = temp / f"provider_{len(undefined)}.cu"
                    provider_object = provider_source.with_suffix(".o")
                    provider_source.write_text(_provider_source({symbol: kind}), encoding="utf-8")
                    _compile(
                        nvcc, provider_source, provider_object, env, lto_arch=profile.arch.lto_base
                    )
                    inputs.append((provider_object.name, provider_object.read_bytes()))
            if not progress:
                _raise_extract_error(
                    f"unresolved-symbol probe made no progress: {sorted(undefined)}"
                )
    _raise_extract_error("unresolved-symbol enumeration exceeded its fail-closed bound")


def extract(
    object_bytes: bytes,
    *,
    profile: Profile,
    tools_env: Mapping[str, str] | None = None,
    manifest: StageManifest | None = None,
) -> SymbolTable:
    """Extract externally visible device definitions and unresolved imports.

    ``profile`` names the probe architecture: unresolved symbols are enumerated
    by linking under ``profile.arch.sm`` with providers compiled for
    ``profile.arch.lto_base``. ``tools_env`` may supply a complete subprocess
    environment; ``PIPEC_CUDA`` overrides ``PIPELINK_CUDA_HOME``, and
    ``PIPEC_NM`` selects the symbol tool.
    """

    if not isinstance(object_bytes, bytes):
        _raise_extract_error("object payload must be bytes")
    env = os.environ.copy()
    if tools_env is not None:
        env.update(tools_env)
    cuda = cuda_home(env)
    env["PATH"] = str(cuda / "bin") + (f":{env['PATH']}" if env.get("PATH") else "")
    env["LD_LIBRARY_PATH"] = str(cuda / "lib64") + (
        f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
    )

    with tempfile.TemporaryDirectory(prefix="pipec2-symbols-") as temp_name:
        object_path = Path(temp_name) / "stage.o"
        object_path.write_bytes(object_bytes)
        strong, weak = _external_definitions(object_path, env)
    undefined_kinds = _undefined_symbols(
        object_bytes,
        sorted(strong | weak),
        cuda,
        env,
        import_seed(manifest) if manifest is not None else None,
        profile=profile,
    )
    overlap = (strong | weak) & set(undefined_kinds)
    if overlap:
        _raise_extract_error(
            f"symbols reported as both defined and undefined: {sorted(overlap)}"
        )

    info: dict[str, SymbolInfo] = {}
    info.update({name: SymbolInfo("func", "strong") for name in strong})
    info.update({name: SymbolInfo("func", "weak") for name in weak})
    info.update(
        {
            name: SymbolInfo("object" if kind == "variable" else "func", "undefined")
            for name, kind in undefined_kinds.items()
        }
    )
    return SymbolTable(
        defined_strong=tuple(strong),
        defined_weak=tuple(weak),
        undefined=tuple(undefined_kinds),
        info=info,
    )


def _symbol_path(collection: str, symbol: str) -> str:
    return f"$.symbols.{collection}[{json.dumps(symbol)}]"


def check_pack(table: SymbolTable, manifest: StageManifest) -> list[ClauseFailure]:
    """Require prefixes except for the stage's explicitly declared callable surface.

    ``manifest`` has passed schema validation, including export declaration
    digests. Export symbols and role entries are exact-name admissions; other
    strong globals still require the stage or PipeLink namespace prefix.
    """

    symbol_prefix = manifest.symbol_prefix
    declared = {export.symbol for export in manifest.exports.values()}
    declared.update(role.entry for role in manifest.roles.values())
    failures = [
        ClauseFailure(
            clause_id="L-SYM",
            path=_symbol_path("defined_strong", symbol),
            message=f"strong global {symbol!r} does not start with {symbol_prefix!r}",
            detail={"symbol": symbol, "required_prefix": symbol_prefix},
        )
        for symbol in table.defined_strong
        if symbol not in declared
        and not symbol.startswith("pipelink_")
        and not symbol.startswith(symbol_prefix)
    ]
    return sorted(failures, key=lambda item: (item.clause_id, item.path))


def advisory_report(table: SymbolTable) -> dict[str, Any]:
    """Return the disclosed, non-gating weak/COMDAT ODR boundary."""

    return {
        "weak_comdat": [
            {
                "symbol": symbol,
                "kind": table.info[symbol].kind,
                "binding": table.info[symbol].binding,
            }
            for symbol in table.defined_weak
        ]
    }


def tensor_import_symbols(stage: StageManifest) -> tuple[str, ...]:
    """Return the exact callable symbols whose providers must be generated."""

    tensor = stage.tensor
    if tensor is None:
        return ()
    return tuple(sorted({symbol for imp in tensor.imports
                         for symbol in (imp.callable.symbol, *imp.callable.events.values())}))


def declared_imports(manifest: StageManifest) -> tuple[str, ...]:
    """Resolve call-channel declarations to real symbols, never logical export keys."""
    from .stubgen import call_symbol

    admitted = set(tensor_import_symbols(manifest))
    for endpoint in manifest.endpoints:
        if endpoint.protocol.kind != "call" or endpoint.protocol.call is None:
            continue
        call = endpoint.protocol.call
        if endpoint.direction == "produce":
            admitted.add(call_symbol(endpoint.name))
        elif endpoint.direction == "consume":
            exported = manifest.exports.get(call.export)
            if exported is not None:
                admitted.add(exported.symbol)
    return tuple(sorted(admitted))


def import_seed(manifest: StageManifest) -> dict[str, str]:
    """Return only declared imports and namespace handles admitted at pack."""
    from .stubgen import reg_config_symbol

    admitted = declared_imports(manifest)
    names = set(admitted)
    for claim in manifest.claims:
        names.add(claim.handle)
        if claim.extent is not None:
            names.update(claim.extent.per)
            for key in claim.extent.table:
                names.update(part.split("=", 1)[0] for part in key.split(","))
        if claim.reg_group is not None:
            names.add(reg_config_symbol(claim.reg_group))
    for variant in manifest.variants:
        names.update(variant.key)
    for endpoint in manifest.endpoints:
        names.update(endpoint.binds.values())
        if endpoint.ring is not None and endpoint.ring.depth.handle is not None:
            names.add(endpoint.ring.depth.handle)
        protocol = endpoint.protocol
        if protocol.init is not None and isinstance(protocol.init.phase, str):
            names.add(protocol.init.phase)
        for side in (protocol.full, protocol.empty):
            if side is not None and hasattr(side, "slots"):
                names.add(side.slots)
        if protocol.named_barrier is not None:
            names.add(protocol.named_barrier.id_handle)
        if protocol.counter is not None:
            names.add(protocol.counter.slot_handle)
    # The linker defines the stage's params accessor; a stage whose code reads params imports it.
    names.add(params_accessor_symbol(manifest.id))
    return {
        name: "function"
        for name in sorted(names)
        if name.startswith("pipelink_") or name in admitted
    }


def exact_import_patterns(symbols: Iterable[str]) -> tuple[str, ...]:
    """Encode literal admitted symbol names for ``check_imports``."""

    return tuple(f"^{re.escape(symbol)}$" for symbol in sorted(set(symbols)))


def check_imports(
    table: SymbolTable,
    allowlist: Sequence[str],
) -> list[ClauseFailure]:
    """Reject imports outside PipeLink's namespace and the supplied admissions."""

    failures: list[ClauseFailure] = []
    for symbol in table.undefined:
        if symbol.startswith("pipelink_"):
            continue
        if any(re.search(pattern, symbol) is not None for pattern in allowlist):
            continue
        failures.append(
            ClauseFailure(
                clause_id="P-IMPORT",
                path=_symbol_path("undefined", symbol),
                message=f"undefined symbol {symbol!r} is not an admitted import",
                detail={
                    "symbol": symbol,
                    "allowlist_version": ALLOWLIST_VERSION,
                },
            )
        )
    return sorted(failures, key=lambda item: (item.clause_id, item.path))


def _check_packed_tensor_providers(
    tables: Mapping[str, SymbolTable],
    required: Mapping[str, str],
) -> list[ClauseFailure]:
    owners: dict[str, list[str]] = {}
    for stage_id, table in sorted(tables.items()):
        for symbol in set(table.defined_strong) | set(table.defined_weak):
            if symbol in required:
                owners.setdefault(symbol, []).append(stage_id)
    return [
        ClauseFailure(
            clause_id="T-LOWER",
            path=f"{_symbol_path('tensor_imports', symbol)}.packed",
            message=f"tensor import {symbol!r} must not be defined by a packed stage",
            detail={
                "symbol": symbol,
                "importing_stage": required[symbol],
                "stages": stages,
            },
        )
        for symbol, stages in sorted(owners.items())
    ]


def link_assembly_check(
    tables: Mapping[str, SymbolTable],
    prefixes: Mapping[str, str],
    *,
    tensor_imports: Mapping[str, str] | None = None,
) -> list[ClauseFailure]:
    """Check packed definitions, prefixes, and reserved tensor-import symbols.

    ``tensor_imports`` maps callable symbols to importing stage IDs. These symbols
    may be unresolved here, but no packed stage may define them (even weakly).
    Generated-provider completeness is deferred to ``check_generated_providers``.
    """

    failures: list[ClauseFailure] = []
    prefix_owners: dict[str, list[str]] = {}
    for stage_id in sorted(tables):
        if stage_id in prefixes:
            prefix_owners.setdefault(prefixes[stage_id], []).append(stage_id)
    for prefix, owners in sorted(prefix_owners.items()):
        if len(owners) > 1:
            failures.append(
                ClauseFailure(
                    clause_id="L-SYM",
                    path=f"$.symbol_prefixes[{json.dumps(prefix)}]",
                    message=f"normalized symbol prefix {prefix!r} is shared by multiple artifacts",
                    detail={"prefix": prefix, "stages": sorted(owners)},
                )
            )

    symbol_owners: dict[str, list[str]] = {}
    for stage_id, table in sorted(tables.items()):
        for symbol in table.defined_strong:
            symbol_owners.setdefault(symbol, []).append(stage_id)
    for symbol, owners in sorted(symbol_owners.items()):
        unique_owners = sorted(set(owners))
        if len(unique_owners) > 1:
            failures.append(
                ClauseFailure(
                    clause_id="L-SYM",
                    path=_symbol_path("defined_strong", symbol),
                    message=f"strong global {symbol!r} is defined by multiple artifacts",
                    detail={"symbol": symbol, "stages": unique_owners},
                )
            )
    if tensor_imports:
        failures.extend(_check_packed_tensor_providers(tables, tensor_imports))
    return sorted(failures, key=lambda item: (item.clause_id, item.path))


def check_generated_providers(
    tables: Mapping[str, SymbolTable],
    adapters: Mapping[str, tuple[str, ...]],
    required: Mapping[str, str],
) -> list[ClauseFailure]:
    """Check late tensor-provider completeness over actual generated symbol sets.

    ``tables`` maps packed stage IDs to symbol tables; ``adapters`` maps generated
    adapter names to their externally defined symbols; ``required`` maps each
    tensor-import callable symbol to its importing stage ID. Each required symbol
    needs exactly one generated provider. All generated definitions must be
    unique and must not collide with packed strong globals.
    Declaration matching is the lowering caller's separate obligation.
    """

    failures: list[ClauseFailure] = []
    generated: dict[str, list[str]] = {}
    for adapter, symbols in sorted(adapters.items()):
        for symbol in set(symbols):
            generated.setdefault(symbol, []).append(adapter)
    for symbol in sorted(set(required) | generated.keys()):
        owners = generated.get(symbol, [])
        if len(owners) != 1:
            detail: dict[str, Any] = {"symbol": symbol, "adapters": owners}
            if symbol in required:
                detail["importing_stage"] = required[symbol]
            failures.append(
                ClauseFailure(
                    clause_id="T-LOWER",
                    path=f"{_symbol_path('tensor_imports', symbol)}.generated",
                    message=f"tensor adapter symbol {symbol!r} requires exactly one generated definition",
                    detail=detail,
                )
            )

    collisions: dict[str, list[str]] = {}
    for stage_id, table in sorted(tables.items()):
        for symbol in set(table.defined_strong):
            if symbol in generated and symbol not in required:
                collisions.setdefault(symbol, []).append(stage_id)
    for symbol, stages in sorted(collisions.items()):
        failures.append(
            ClauseFailure(
                clause_id="T-LOWER",
                path=_symbol_path("defined_strong", symbol),
                message=f"generated adapter symbol {symbol!r} collides with a packed strong global",
                detail={
                    "symbol": symbol,
                    "adapters": generated[symbol],
                    "stages": stages,
                },
            )
        )
    return sorted(failures, key=lambda item: (item.clause_id, item.path))


__all__ = [
    "ALLOWLIST_VERSION",
    "MECHANISM",
    "TOOLCHAIN_ALLOWLIST",
    "advisory_report",
    "declared_imports",
    "exact_import_patterns",
    "check_generated_providers",
    "check_imports",
    "check_pack",
    "cuda_home",
    "extract",
    "import_seed",
    "link_assembly_check",
    "tensor_import_symbols",
]
