from __future__ import annotations

import ctypes
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .canonjson import canonical_bytes, sha256_hex
from .model import (
    CanonEndpoint,
    Claim,
    ClauseFailure,
    Endpoint,
    Extent,
    LegalitySet,
    Handle,
    LinkError,
    LinkResult,
    MathMode,
    NVCC_DEFAULT_MATH,
    ParamsLayout,
    Pipeline,
    PipeLinkError,
    ProtocolExpansion,
    Solution,
    StageManifest,
    SymbolTable,
    TensorLifetime,
    register_clause,
)
from .profile import Profile, get_profile
from .host.contracts import host_link, host_params_layout, resolve_host
from .solve import solver_provenance
from .symbols import _external_definitions, cuda_home

if TYPE_CHECKING:
    from .tensor.contracts import TensorImport
    from .tensor.lower import AdapterSource
    from .tensor.plan import ConnectionPlanner, TensorPlan


_KERNEL_SYMBOL = "pipelink_kernel"
_RESOURCE_FIELD_RE = re.compile(r"\b(REG|STACK|LOCAL):(\d+)\b")
_RESOURCE_FUNCTION_RE = re.compile(r"^\s*Function\s+(.+):\s*$", re.MULTILINE)
_NVJITLINK_INPUT_OBJECT = 5

register_clause("T-NATIVE-SYMBOLS", phase="link",
                description="Generated adapter objects define no uncontracted strong device symbols.")


@dataclass(frozen=True)
class _Artifact:
    path: Path
    stage: StageManifest
    object_bytes: bytes
    artifact_sha256: str
    # Endpoints of this stage dropped with unreferenced exports (A4); `stage` is already pruned.
    dropped_endpoints: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class _EndpointRecord:
    stage: StageManifest
    endpoint: Endpoint

    @property
    def key(self) -> str:
        return f"{self.stage.id}:{self.endpoint.name}:{self.endpoint.direction}"


def _operation_error(message: str) -> LinkError:
    return LinkError([ClauseFailure("L-LINK", "link", message, {})])


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _operation_error(f"cannot read pipeline {path}: {exc}") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        value = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _operation_error(f"invalid pipeline JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _operation_error(f"pipeline JSON root must be an object: {path}")
    return value


def _load_assembly(pipeline_path: Path) -> tuple[Pipeline, list[_Artifact]]:
    from .container import read as read_container
    from .schema import load_pipeline, load_stage_manifest

    pipeline = load_pipeline(_read_json_object(pipeline_path))
    base = pipeline_path.parent
    artifacts: list[_Artifact] = []
    for ref in sorted(pipeline.stages, key=lambda item: (item.stage_id, item.artifact_path)):
        candidate = Path(ref.artifact_path)
        artifact_path = candidate if candidate.is_absolute() else base / candidate
        manifest_obj, object_bytes = read_container(artifact_path)
        try:
            artifact_digest = sha256_hex(artifact_path.read_bytes())
        except OSError as exc:
            raise _operation_error(f"cannot hash artifact {artifact_path}: {exc}") from exc
        artifacts.append(
            _Artifact(
                path=artifact_path,
                stage=load_stage_manifest(manifest_obj),
                object_bytes=object_bytes,
                artifact_sha256=artifact_digest,
            )
        )
    artifacts.sort(key=lambda item: (item.stage.id, str(item.path)))
    return pipeline, _assemble(pipeline, artifacts)


def _assemble(pipeline: Pipeline, artifacts: Sequence[_Artifact]) -> list[_Artifact]:
    """Prune unreferenced exports, then validate the pipeline against the pruned interface:
    endpoints that only unreferenced exports use (and the slot handles they name) are not part
    of this assembly."""
    from .schema import validate_assembly

    stages, dropped = _prune_unreferenced_exports([item.stage for item in artifacts])
    failures = validate_assembly(pipeline, stages)
    if failures:
        raise LinkError(failures)
    return [
        replace(item, stage=stage,
                dropped_endpoints=tuple(record for record in dropped if record["stage"] == stage.id))
        for item, stage in zip(artifacts, stages)
    ]


def _artifacts_in_pipeline_order(
    pipeline: Pipeline,
    artifacts: Sequence[_Artifact],
) -> list[_Artifact]:
    """Restore the declared order used only for nvJitLink input addition."""

    by_stage_id = {artifact.stage.id: artifact for artifact in artifacts}
    try:
        return [by_stage_id[ref.stage_id] for ref in pipeline.stages]
    except KeyError as exc:
        raise _operation_error(f"validated pipeline stage {exc.args[0]!r} has no loaded artifact") from exc


def _check_symbols(artifacts: Sequence[_Artifact]) -> None:
    from .symbols import (
        TOOLCHAIN_ALLOWLIST,
        check_imports,
        check_params_accessors,
        declared_imports,
        exact_import_patterns,
        link_assembly_check,
        tensor_import_symbols,
    )

    tables: dict[str, SymbolTable] = {}
    for artifact in sorted(artifacts, key=lambda item: item.stage.id):
        if artifact.stage.symbols is None:
            raise _operation_error(f"{artifact.path}: packed manifest is missing symbols; "
            "repack this artifact with pipec2 pack")
        tables[artifact.stage.id] = artifact.stage.symbols
    stages = {artifact.stage.id: artifact.stage for artifact in artifacts}
    definition_owners: dict[str, set[str]] = {}
    for stage_id, table in tables.items():
        for symbol in (*table.defined_strong, *table.defined_weak):
            definition_owners.setdefault(symbol, set()).add(stage_id)

    failures = check_params_accessors(stages.values())
    for stage_id in sorted(tables):
        resolved_declarations = {
            symbol
            for symbol in declared_imports(stages[stage_id])
            if any(owner != stage_id for owner in definition_owners.get(symbol, ()))
        }
        resolved_declarations.update(tensor_import_symbols(stages[stage_id]))
        import_allowlist = (
            *TOOLCHAIN_ALLOWLIST,
            *exact_import_patterns(resolved_declarations),
        )
        failures.extend(check_imports(tables[stage_id], import_allowlist))
    tensor_imports = {
        symbol: stage_id
        for stage_id, stage in sorted(stages.items())
        for symbol in tensor_import_symbols(stage)
    }
    prefixes = {
        artifact.stage.id: artifact.stage.symbol_prefix
        for artifact in sorted(artifacts, key=lambda item: item.stage.id)
    }
    failures.extend(
        link_assembly_check(tables, prefixes, tensor_imports=tensor_imports)
        if tensor_imports else link_assembly_check(tables, prefixes)
    )
    if failures:
        raise LinkError(failures)


def _prune_unreferenced_exports(
    stages: Sequence[StageManifest],
) -> tuple[list[StageManifest], list[dict[str, str]]]:
    """Drop provided call channels nobody calls, and the endpoints associated only with
    exports none of whose channels is called.

    As in any linker, an unreferenced export is not an error. An export is live while any of
    its provided channels has a caller. An endpoint that declares the exports whose bodies use
    it leaves the interface when all of them are unreferenced. Host transfers stay in the
    owner's host inventory, so an endpoint one of them references cannot leave (L-PAIR).
    """
    called = {
        endpoint.name for stage in stages for endpoint in stage.endpoints
        if endpoint.protocol.kind == "call" and endpoint.direction == "produce"
    }
    transfer_endpoints = {
        (transfer["device_endpoint"]["stage"], transfer["device_endpoint"]["endpoint"])
        for stage in stages for transfer in (stage.host or {}).get("transfers", [])
    }
    pruned: list[StageManifest] = []
    dropped: list[dict[str, str]] = []
    failures: list[ClauseFailure] = []
    for stage in stages:
        provided = [
            endpoint for endpoint in stage.endpoints
            if endpoint.protocol.kind == "call" and endpoint.direction == "consume"
        ]
        unreferenced = (
            {endpoint.protocol.call.export for endpoint in provided}
            - {endpoint.protocol.call.export for endpoint in provided if endpoint.name in called}
        )
        kept = []
        for endpoint in stage.endpoints:
            if endpoint.protocol.kind == "call" and endpoint.direction == "consume":
                live, reason = endpoint.name in called, "uncalled export channel"
            else:
                live = endpoint.exports is None or not set(endpoint.exports) <= unreferenced
                reason = "owned by unreferenced exports"
            if live:
                kept.append(endpoint)
            elif (stage.id, endpoint.name) in transfer_endpoints:
                failures.append(ClauseFailure(
                    "L-PAIR", f"stages.{stage.id}.endpoints.{endpoint.name}",
                    "an endpoint referenced by a host transfer cannot leave the interface with unreferenced exports",
                    {"exports": sorted(endpoint.exports or ())},
                ))
            else:
                dropped.append({"stage": stage.id, "endpoint": endpoint.name, "reason": reason})
        pruned.append(stage if len(kept) == len(stage.endpoints) else replace(stage, endpoints=kept))
    if failures:
        raise LinkError(failures)
    dropped.sort(key=lambda item: (item["stage"], item["endpoint"]))
    return pruned, dropped


def _pair_channels(
    stages: Sequence[StageManifest],
) -> list[tuple[str, list[_EndpointRecord], list[_EndpointRecord]]]:
    """Group each channel's producers and consumers: exactly one of each, except counter
    channels, which take one or more producers and any number of consumers."""
    channels: dict[str, dict[str, list[_EndpointRecord]]] = {}
    for stage in sorted(stages, key=lambda item: item.id):
        for endpoint in sorted(stage.endpoints, key=lambda item: (item.name, item.direction)):
            channels.setdefault(endpoint.name, {}).setdefault(endpoint.direction, []).append(
                _EndpointRecord(stage, endpoint)
            )

    failures: list[ClauseFailure] = []
    pairs: list[tuple[str, list[_EndpointRecord], list[_EndpointRecord]]] = []
    for channel in sorted(channels):
        directions = channels[channel]
        producers = sorted(directions.get("produce", []), key=lambda item: item.key)
        consumers = sorted(directions.get("consume", []), key=lambda item: item.key)
        other = sorted(
            record.key
            for direction, records in directions.items()
            if direction not in {"produce", "consume"}
            for record in records
        )
        counter = producers and not other and all(
            record.endpoint.protocol.kind == "counter" for record in (*producers, *consumers)
        )
        if not counter and (len(producers) != 1 or len(consumers) != 1 or other):
            failures.append(
                ClauseFailure(
                    "L-PAIR",
                    f"channels.{channel}",
                    "channel requires exactly one producer and one consumer",
                    {
                        "producers": [item.key for item in producers],
                        "consumers": [item.key for item in consumers],
                        "other": other,
                    },
                )
            )
            continue
        pairs.append((channel, producers, consumers))
    if failures:
        raise LinkError(failures)
    return pairs


def _base_claim(record: _EndpointRecord):
    handle = record.endpoint.binds.get("base")
    if handle is None:
        return None
    matches = sorted(
        (claim for claim in record.stage.claims if claim.handle == handle),
        key=lambda claim: (claim.domain, claim.handle, claim.lifetime),
    )
    if len(matches) != 1:
        raise LinkError(
            [
                ClauseFailure(
                    "L-HANDLE",
                    f"stages.{record.stage.id}.endpoints.{record.endpoint.name}.binds.base",
                    "endpoint base binding must resolve to exactly one stage claim",
                    {"handle": handle, "matches": len(matches)},
                )
            ]
        )
    return matches[0]


def _layout_profile(pipeline: Pipeline, profile: Profile) -> Profile:
    domains = dict(profile.domains)
    for name, domain in profile.domains.items():
        if domain.unit_scale is not None:
            continue
        declaration = pipeline.domains.get(name)
        if declaration is None or declaration.page_bytes is None:
            continue
        domains[name] = replace(domain, unit_scale=declaration.page_bytes)
    return replace(profile, domains=domains)


def _canonical_endpoints(
    stages: Sequence[StageManifest],
    pipeline: Pipeline,
    profile: Profile,
) -> dict[str, CanonEndpoint]:
    from .layout import canonicalize

    layout_profile = _layout_profile(pipeline, profile)
    result: dict[str, CanonEndpoint] = {}
    records = [
        _EndpointRecord(stage, endpoint)
        for stage in sorted(stages, key=lambda item: item.id)
        for endpoint in sorted(stage.endpoints, key=lambda item: (item.name, item.direction))
    ]
    for record in records:
        result[record.key] = canonicalize(
            record.endpoint,
            _base_claim(record),
            layout_profile,
        )
    return result


def _unify_pairs(
    pairs: Sequence[tuple[str, list[_EndpointRecord], list[_EndpointRecord]]],
    canon: Mapping[str, CanonEndpoint],
) -> None:
    from .layout import unify

    failures: list[ClauseFailure] = []
    for channel, producers, consumers in pairs:
        for producer in producers:
            for consumer in consumers:
                failures.extend(
                    ClauseFailure(
                        failure.clause_id,
                        f"channels.{channel}.{failure.path}",
                        failure.message,
                        failure.detail,
                    )
                    for failure in unify(canon[producer.key], canon[consumer.key])
                )
    if failures:
        raise LinkError(failures)


def _expand_protocol(
    channel: str,
    producers: Sequence[_EndpointRecord],
    consumers: Sequence[_EndpointRecord],
    stages: Sequence[StageManifest],
    pipeline: Pipeline,
    canon: Mapping[str, CanonEndpoint],
    profile: Profile,
) -> ProtocolExpansion:
    from .protocol import expand, expand_counter

    if producers[0].endpoint.protocol.kind == "counter":
        return expand_counter(
            channel, [item.endpoint for item in producers], [item.endpoint for item in consumers], pipeline)
    return expand(channel, producers[0].endpoint, consumers[0].endpoint, stages, pipeline, dict(canon), profile)


def _solve_pipeline(
    pipeline: Pipeline,
    stages: Sequence[StageManifest],
    canon: Mapping[str, CanonEndpoint],
    expansions: Sequence[ProtocolExpansion],
    profile: Profile,
    ordered_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> Solution:
    from .solve import solve

    return solve(pipeline, list(stages), dict(canon), list(expansions), profile, ordered_pairs)


def _generate_stub(
    solution: Solution,
    pipeline: Pipeline,
    stages: Sequence[StageManifest],
    expansions: Sequence[ProtocolExpansion],
    view_calls: Mapping[str, Any],
) -> str:
    from .stubgen import generate_stub

    tensor_resources = [(path, resource) for stage in stages
                        for path, resource in _tensor_resources(stage)
                        if resource.location.handle is not None]
    if tensor_resources:
        referenced = {
            symbol for stage in stages if stage.symbols is not None
            for symbol in stage.symbols.undefined
        }
        suppressed = {resource.location.handle for _path, resource in tensor_resources} - referenced
        bases = {name: value for name, value in solution.native_bases.items() if name not in suppressed}
        for path, resource in tensor_resources:
            name, base = resource.location.handle, resource.location.base
            if name not in referenced or resource.binding != "fixed" or base is None:
                continue
            if name in bases and bases[name] != base:
                raise LinkError([ClauseFailure(
                    "T-CONTRACT", f"{path}.location",
                    "fixed tensor binding disagrees with its solved native base",
                    {"handle": name, "declared": base, "solved": bases[name]},
                )])
            bases[name] = base
        solution = replace(solution, native_bases=bases)
        pipeline = replace(
            pipeline,
            forced_bindings={name: value for name, value in pipeline.forced_bindings.items()
                             if name not in suppressed},
        )
    return generate_stub(solution, pipeline, list(stages), list(expansions), view_calls)


def _params_layout(stages: Sequence[StageManifest], profile: Profile) -> ParamsLayout:
    from .dispatchgen import params_layout

    return params_layout(list(stages), profile)


def _generate_dispatch(
    solution: Solution,
    pipeline: Pipeline,
    stages: Sequence[StageManifest],
    expansions: Sequence[ProtocolExpansion],
    params: ParamsLayout,
    host: dict[str, Any] | None = None,
) -> str:
    from .dispatchgen import generate_dispatch

    if host is not None:
        return generate_dispatch(solution, pipeline, list(stages), list(expansions), params, host=host)
    return generate_dispatch(solution, pipeline, list(stages), list(expansions), params)


def _tool_env(cuda: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(cuda / "bin") + (f":{env['PATH']}" if env.get("PATH") else "")
    env["LD_LIBRARY_PATH"] = str(cuda / "lib64") + (
        f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
    )
    return env


def _compile_command(nvcc: Path, profile: Profile, source: Path, output: Path) -> list[str]:
    return [
        str(nvcc),
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-rdc=true",
        "-dc",
        f"-arch={profile.lto_arch}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-I", str(Path(__file__).parent / "pack" / "include"),
        str(source),
        "-o",
        str(output),
    ]


def _adapter_compile_command(
    nvcc: Path, profile: Profile, source: Path, output: Path
) -> list[str]:
    """Use the declared adapter route, never the ordinary stub LTO route."""
    route = profile.adapter_compile
    if route.forced_macros:
        raise LinkError([ClauseFailure(
            "T-NATIVE", "profile.adapter_compile.forced_macros",
            "adapter compilation cannot force architecture feature macros",
            {"forced_macros": list(route.forced_macros)},
        )])
    includes = ["-I", str(Path(__file__).parent / "pack" / "include")]
    for name in route.include_roots_env:
        value = os.environ.get(name)
        if not value:
            raise LinkError([ClauseFailure(
                "T-NATIVE", f"profile.adapter_compile.include_roots_env.{name}",
                f"missing adapter include-root environment variable {name}",
                {"environment_variable": name},
            )])
        includes.extend(("-I", value))
    frontend, lto = route.gencode
    return [
        str(nvcc), "-std=c++17", "-O3", "-DNDEBUG", "-rdc=true", "-dc",
        f"-gencode=arch={frontend},code={lto}",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        *includes, str(source), "-o", str(output),
    ]


def _check_adapter_targets(stages: Sequence[StageManifest], profile: Profile) -> None:
    route = profile.adapter_compile
    arch = profile.arch
    expected = {
        "profile": profile.name,
        "lto_arch": arch.lto,
        "toolkit": route.toolkit,
    }
    failures = []
    # Arch spellings come from the route itself (Profile.arch); only the toolkit can disagree.
    if route.toolkit != profile.toolkit:
        failures.append(ClauseFailure(
            "T-NATIVE", "profile.adapter_compile",
            "adapter frontend/toolkit route is incompatible with the target profile",
            {"profile": profile.to_json_obj(), "adapter_compile": route.to_json_obj()},
        ))
    for stage in stages:
        actual = {name: getattr(stage.target, name) for name in expected}
        if actual != expected:
            failures.append(ClauseFailure(
                "T-NATIVE", f"stages.{stage.id}.target",
                "packed stage target is incompatible with the adapter compile route",
                {"expected": expected, "actual": actual},
            ))
    if failures:
        raise LinkError(failures)


def _link_options(profile: Profile, regs_cap: int, math: MathMode) -> list[str]:
    return [
        "-lto",
        f"-arch={profile.arch.sm}",
        f"-maxrregcount={regs_cap}",
        # LTO code generation runs here; nvJitLink applies only these switches, not the stages'.
        f"-ftz={int(math.ftz)}",
        f"-prec-div={int(math.prec_div)}",
        f"-prec-sqrt={int(math.prec_sqrt)}",
        f"-fma={int(math.fmad)}",
    ]


def _math_mode(stages: Sequence[StageManifest], override: MathMode | None) -> MathMode:
    """The one floating-point codegen mode of the image: the pipeline's `target.math` when it
    sets one, otherwise the mode every stage was packed with."""
    if override is not None:
        return override
    modes = {stage.id: stage.target.math or NVCC_DEFAULT_MATH for stage in stages}
    if len(set(modes.values())) > 1:
        raise LinkError([ClauseFailure(
            "L-LINK", "stages.target.math",
            "stages were compiled with different floating-point codegen switches",
            {"modes": {stage_id: mode.to_json_obj() for stage_id, mode in sorted(modes.items())}},
        )])
    return next(iter(modes.values()), NVCC_DEFAULT_MATH)


class _NvJitLink:
    """The final-link vehicle retained from the legacy implementation."""

    def __init__(self, library: Path) -> None:
        try:
            self.lib = ctypes.CDLL(str(library))
        except OSError as exc:
            raise _operation_error(f"cannot load {library}: {exc}") from exc

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
        self.get_cubin_size = self.lib.nvJitLinkGetLinkedCubinSize
        self.get_cubin_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.get_cubin_size.restype = ctypes.c_int
        self.get_cubin = self.lib.nvJitLinkGetLinkedCubin
        self.get_cubin.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.get_cubin.restype = ctypes.c_int
        self.get_error_size = self.lib.nvJitLinkGetErrorLogSize
        self.get_error_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.get_error_size.restype = ctypes.c_int
        self.get_error = self.lib.nvJitLinkGetErrorLog
        self.get_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.get_error.restype = ctypes.c_int
        self.get_info_size = self.lib.nvJitLinkGetInfoLogSize
        self.get_info_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.get_info_size.restype = ctypes.c_int
        self.get_info = self.lib.nvJitLinkGetInfoLog
        self.get_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.get_info.restype = ctypes.c_int
        self.destroy = self.lib.nvJitLinkDestroy
        self.destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.destroy.restype = ctypes.c_int

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise _operation_error(f"{operation} returned nvJitLinkResult {result}")

    def _log(self, handle: ctypes.c_void_p, *, info: bool) -> str:
        size = ctypes.c_size_t()
        size_fn = self.get_info_size if info else self.get_error_size
        get_fn = self.get_info if info else self.get_error
        self._check(size_fn(handle, ctypes.byref(size)), "nvJitLinkGetLogSize")
        if size.value == 0:
            return ""
        buffer = ctypes.create_string_buffer(size.value)
        self._check(get_fn(handle, buffer), "nvJitLinkGetLog")
        return buffer.value.decode("utf-8", errors="replace")

    def link(
        self,
        inputs: list[tuple[str, bytes]],
        options: list[str],
    ) -> tuple[bytes, str, str]:
        handle = ctypes.c_void_p()
        encoded_options = [option.encode("utf-8") for option in options]
        option_array = (ctypes.c_char_p * len(encoded_options))(*encoded_options)
        self._check(
            self.create(ctypes.byref(handle), len(encoded_options), option_array),
            "nvJitLinkCreate",
        )
        keepalive: list[Any] = []
        try:
            for name, data in inputs:
                buffer = ctypes.create_string_buffer(data, len(data))
                keepalive.append(buffer)
                result = self.add_data(
                    handle,
                    _NVJITLINK_INPUT_OBJECT,
                    buffer,
                    len(data),
                    name.encode("utf-8"),
                )
                if result != 0:
                    error_log = self._log(handle, info=False)
                    raise _operation_error(f"nvJitLinkAddData({name}) returned nvJitLinkResult {result}\n"
                    f"{error_log}")
            result = self.complete(handle)
            info_log = self._log(handle, info=True)
            error_log = self._log(handle, info=False)
            if result != 0:
                raise _operation_error(f"nvJitLinkComplete returned nvJitLinkResult {result}\n"
                f"info log:\n{info_log}\nerror log:\n{error_log}")
            size = ctypes.c_size_t()
            self._check(
                self.get_cubin_size(handle, ctypes.byref(size)),
                "nvJitLinkGetLinkedCubinSize",
            )
            cubin = ctypes.create_string_buffer(size.value)
            self._check(
                self.get_cubin(handle, cubin),
                "nvJitLinkGetLinkedCubin",
            )
            return bytes(cubin.raw), info_log, error_log
        finally:
            if handle.value is not None:
                self.destroy(ctypes.byref(handle))


def _run_checked(command: Sequence[str], env: Mapping[str, str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        raise _operation_error(f"cannot execute {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        raise _operation_error(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")
    return completed.stdout


def _write_source(path: Path, source: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise _operation_error(f"cannot write generated source {path}: {exc}") from exc


def _safe_object_name(index: int, stage_id: str) -> str:
    return f"{index:03d}-{re.sub(r'[^A-Za-z0-9_.-]', '_', stage_id)}.o"


def _parse_resource_usage(
    output: str, expected_kernel: str
) -> tuple[str, tuple[int, int, int]]:
    records: dict[str, tuple[int, int, int]] = {}
    matches = list(_RESOURCE_FUNCTION_RE.finditer(output))
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(output)
        fields = {name: int(value) for name, value in _RESOURCE_FIELD_RE.findall(output[match.end():body_end])}
        if all(name in fields for name in ("REG", "STACK", "LOCAL")):
            records[match.group(1)] = (fields["REG"], fields["STACK"], fields["LOCAL"])

    if expected_kernel not in records:
        raise _operation_error(f"cuobjdump did not report REG/STACK/LOCAL for kernel {expected_kernel!r}")
    return expected_kernel, records[expected_kernel]


def _requested_sass_report(
    cuobjdump: Path, cubin: Path, env: Mapping[str, str]
) -> dict[str, Any] | None:
    requested = os.environ.get("PIPEC_SASS_REPORT", "").lower() in {"1", "true", "yes"}
    if not requested or not cuobjdump.is_file():
        return None
    command = [str(cuobjdump), "--dump-sass", str(cubin)]
    try:
        completed = subprocess.run(
            command,
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    encoded = completed.stdout.encode("utf-8")
    return {
        "tool": "cuobjdump",
        "output_bytes": len(encoded),
        "output_sha256": sha256_hex(encoded),
    }


def _compile_and_measure(
    out_dir: Path,
    artifacts: Sequence[_Artifact],
    generated_sources: Mapping[str, str],
    profile: Profile,
    solution: Solution,
    expected_kernel: str,
    math: MathMode,
    *,
    adapters: Mapping[str, AdapterSource] | None = None,
    tensor_required: Mapping[str, str] | None = None,
) -> tuple[str, tuple[int, int, int], dict[str, Any] | None]:
    cuda = cuda_home()
    nvcc = cuda / "bin" / "nvcc"
    cuobjdump = cuda / "bin" / "cuobjdump"
    library = cuda / "lib64" / "libnvJitLink.so.13"
    env = _tool_env(cuda)
    generated_dir = out_dir / "generated"
    object_dir = out_dir / ".objects"
    try:
        generated_dir.mkdir(parents=True, exist_ok=True)
        object_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _operation_error(f"cannot create linker output directories: {exc}") from exc
    for filename in ({"binding_stub.cu", "dispatch.cu"} - set(generated_sources)):
        for stale in (
            generated_dir / filename,
            object_dir / f"{Path(filename).stem}.o",
        ):
            try:
                stale.unlink(missing_ok=True)
            except OSError as exc:
                raise _operation_error(f"cannot remove stale generated file {stale}: {exc}") from exc

    link_inputs = [
        (_safe_object_name(index, artifact.stage.id), artifact.object_bytes)
        for index, artifact in enumerate(artifacts)
    ]

    if adapters:
        _check_adapter_targets([artifact.stage for artifact in artifacts], profile)

    filenames = sorted(set(generated_sources) - set(adapters or ()))
    filenames.extend(sorted(adapters or ()))
    for filename in filenames:
        source_path = generated_dir / filename
        object_path = object_dir / f"{Path(filename).stem}.o"
        _write_source(source_path, generated_sources[filename])
        command = (
            _adapter_compile_command(nvcc, profile, source_path, object_path)
            if adapters and filename in adapters
            else _compile_command(nvcc, profile, source_path, object_path)
        )
        try:
            _run_checked(command, env)
        except LinkError as exc:
            if not adapters or filename not in adapters:
                raise
            raise LinkError([ClauseFailure(
                "T-NATIVE", f"generated.{filename}",
                "adapter compiler failed on the declared native route",
                {"command": command, "failures": [failure.to_json_obj() for failure in exc.failures]},
            )]) from exc
        if adapters and filename in adapters:
            try:
                strong, _weak = _external_definitions(object_path, env)
            except PipeLinkError as exc:
                raise LinkError([ClauseFailure(
                    "T-NATIVE-SYMBOLS", f"generated.{filename}.symbols",
                    "cannot inspect generated adapter device definitions",
                    {"failures": [failure.to_json_obj() for failure in exc.failures]},
                )]) from exc
            unexpected = sorted(strong - set(adapters[filename].defined_symbols))
            if unexpected:
                raise LinkError([ClauseFailure(
                    "T-NATIVE-SYMBOLS", f"generated.{filename}.symbols",
                    "adapter defines strong device symbols outside its contract",
                    {"unexpected": unexpected, "contract": sorted(adapters[filename].defined_symbols)},
                )])
        try:
            object_bytes = object_path.read_bytes()
        except OSError as exc:
            raise _operation_error(f"cannot read generated object {object_path}: {exc}") from exc
        link_inputs.append((object_path.name, object_bytes))

    if tensor_required is not None:
        from .symbols import check_generated_providers

        failures = check_generated_providers(
            {artifact.stage.id: artifact.stage.symbols for artifact in artifacts},
            {name: adapter.defined_symbols for name, adapter in (adapters or {}).items()},
            tensor_required,
        )
        if failures:
            raise LinkError(failures)

    cubin = out_dir / "pipeline.cubin"
    cubin_bytes, _info_log, _error_log = _NvJitLink(library).link(
        link_inputs,
        _link_options(profile, solution.launch.regs_cap, math),
    )
    try:
        cubin.write_bytes(cubin_bytes)
    except OSError as exc:
        raise _operation_error(f"cannot write linked cubin {cubin}: {exc}") from exc
    resource_output = _run_checked(
        [str(cuobjdump), "--dump-resource-usage", str(cubin)], env
    )
    kernel_symbol, observed = _parse_resource_usage(resource_output, expected_kernel)
    return kernel_symbol, observed, _requested_sass_report(cuobjdump, cubin, env)


def _observed_register_allocation(observed_reg: int, profile: Profile) -> int:
    granularity = profile.caps.reg_alloc_granularity
    if observed_reg <= 0:
        raise _operation_error(f"cuobjdump reported invalid REG value {observed_reg}")
    return math.ceil(observed_reg / granularity) * granularity


def _register_pool(
    solution: Solution, observed_reg: int, profile: Profile
) -> tuple[bool, int, int, int]:
    allocation = _observed_register_allocation(observed_reg, profile)
    demand = sum(
        group.value * (group.warp_hi - group.warp_lo) * 32
        for group in solution.reg_groups
    )
    capacity = allocation * solution.launch.threads
    return demand <= capacity, demand, capacity, allocation


def _register_failure(
    solution: Solution, observed_reg: int, profile: Profile, *, after_resolve: bool
) -> ClauseFailure:
    _fits, demand, capacity, allocation = _register_pool(solution, observed_reg, profile)
    return ClauseFailure(
        "L-REGS",
        "link.observed.REG",
        "observed register pool remains illegal after deterministic re-solve"
        if after_resolve
        else "observed register pool is illegal",
        {
            "observed_reg": observed_reg,
            "observed_allocation": allocation,
            "demand": demand,
            "capacity": capacity,
        },
    )


def _pipeline_with_register_bound(
    pipeline: Pipeline, observed_allocation: int
) -> Pipeline:
    name = "pipelink_launch_regs"
    handles = dict(pipeline.handles)
    handle = handles.get(name)
    if handle is None:
        handles[name] = Handle(
            name,
            "regs",
            "bind",
            "regs",
            [observed_allocation],
            "launch",
        )
        return replace(pipeline, handles={key: handles[key] for key in sorted(handles)})

    values = [value for value in (handle.values or []) if value <= observed_allocation]
    forced = pipeline.forced_bindings.get(name)
    if not values or (forced is not None and forced not in values):
        raise LinkError(
            [
                ClauseFailure(
                    "L-REGS",
                    f"pipeline.handles.{name}.values",
                    "no legal launch register value remains under the observed bound",
                    {
                        "observed_allocation": observed_allocation,
                        "declared": sorted(handle.values or []),
                        "forced": forced,
                    },
                )
            ]
        )
    handles[name] = replace(handle, values=values)
    return replace(pipeline, handles={key: handles[key] for key in sorted(handles)})


def _cluster_result(solution: Solution, stages: Sequence[StageManifest]) -> tuple[int, int, int] | LegalitySet:
    if solution.launch.cluster != "dynamic":
        return solution.launch.cluster
    x_multiple = 1
    maximum: list[int] | None = None
    for stage in sorted(stages, key=lambda item: item.id):
        constraint = stage.topology.cluster_launch
        x_multiple = math.lcm(
            x_multiple, stage.topology.cta_group, constraint.x_multiple_of
        )
        maximum = (
            list(constraint.max)
            if maximum is None
            else [min(lhs, rhs) for lhs, rhs in zip(maximum, constraint.max, strict=True)]
        )
    if maximum is None:
        maximum = [1, 1, 1]
    return LegalitySet(x_multiple, tuple(maximum))  # type: ignore[arg-type]


def _binding_values(pipeline: Pipeline, solution: Solution) -> dict[str, tuple[int, str]]:
    candidates: dict[str, set[int]] = {name: set() for name in pipeline.handles}

    def add(name: str, value: int) -> None:
        if name in candidates:
            candidates[name].add(value)

    for source in (solution.depths, solution.native_bases, solution.regs):
        for name, value in sorted(source.items()):
            add(name, value)
    for group in solution.reg_groups:
        add(group.handle, group.value)
    for stage_id in sorted(solution.variants):
        for name, value in sorted(solution.variants[stage_id].items()):
            add(name, value)
    for name, handle in sorted(pipeline.handles.items()):
        if name == "pipelink_launch_regs" or (
            handle.cls == "regs" and handle.owner in {"launch", "kernel", "launch:kernel"}
        ):
            add(name, solution.launch.regs_cap)
        if name in pipeline.forced_bindings:
            add(name, pipeline.forced_bindings[name])
        if not candidates[name] and handle.values is not None and len(handle.values) == 1:
            add(name, handle.values[0])

    failures = [
        ClauseFailure(
            "L-HANDLE",
            f"link.bindings.{name}",
            "resolved handle must have exactly one binding value",
            {"values": sorted(values)},
        )
        for name, values in sorted(candidates.items())
        if len(values) != 1
    ]
    if failures:
        raise LinkError(failures)
    return {
        name: (
            next(iter(candidates[name])),
            "forced" if name in pipeline.forced_bindings else "solved",
        )
        for name in sorted(candidates)
    }


@dataclass
class _TensorConnectionPlan:
    stage_id: str
    imp: TensorImport
    plan: TensorPlan
    record: dict[str, Any]
    connection: TensorConnection
    planner: ConnectionPlanner

    def advance(self, detail: Any) -> bool:
        """Record the lowering rejection and move to the next certified candidate; False when exhausted."""
        self.planner.reject_lowering(detail)
        self.plan = self.planner.next()
        return self.plan.status == "MODEL_SAT"


class _Relower(Exception):
    """A candidate did not lower and its planner has moved on; recompose from the original assembly."""


def _tensor_error(
    failures: Sequence[ClauseFailure], records: Sequence[dict[str, Any]]
) -> LinkError:
    """Keep failed-link tensor evidence on both the exception and CLI failures."""
    tensor = {"connections": list(records)}
    error = LinkError([
        replace(failure, detail={**failure.detail, "tensor": tensor})
        for failure in failures
    ])
    error.tensor = tensor
    return error


def _plan_record(connection: TensorConnection, plan: TensorPlan, profile: Profile) -> dict[str, Any]:
    """The link.json record of one connection's current candidate; solution-derived fields start empty."""
    provenance = plan.provenance
    return {
        "name": connection.name,
        "contract_digests": provenance.get("contract_digests", {}),
        "model_digest": provenance.get("model_digest"),
        "bounds": [{"class": item["name"], "scope": item["scope"]} for item in provenance.get("classes", [])],
        "selected": provenance.get("selected"),
        "mode": provenance.get("mode"),
        "status": plan.status,
        "admission": plan.admission.to_json_obj(),
        "plan_digest": plan.ir_digest,
        "solver_outcome": {
            "status": plan.status, "provenance": provenance,
            "profile_digest": sha256_hex(canonical_bytes(profile.to_json_obj())),
            "claims": [claim.to_json_obj() for claim in plan.claims],
            "handles": [handle.to_json_obj() for handle in plan.handles],
            "barrier_requests": list(plan.barrier_requests),
        },
        "lowering": None,
        "trusted_premises": [],
        "native": None,
    }


_PLAN_CLAUSES = {
    "CONTRACT_INVALID": "T-CONTRACT", "UNSUPPORTED_MODEL": "T-MODEL",
    "BOUNDED_UNSAT": "T-PLAN", "UNKNOWN": "T-PLAN", "RESOURCE_ENVELOPE": "T-PLAN",
    "LOWERING_UNSUPPORTED": "T-LOWER",
}


def _plan_failure(connection: TensorConnection, plan: TensorPlan) -> list[ClauseFailure]:
    return list(plan.admission.issues) or [ClauseFailure(
        _PLAN_CLAUSES.get(plan.status, "T-PLAN"),
        f"pipeline.tensor_connections.{connection.name}",
        f"tensor planning returned {plan.status}",
        {"status": plan.status, "provenance": plan.provenance},
    )]


def _plan_tensors(
    pipeline: Pipeline, stages: Sequence[StageManifest], profile: Profile
) -> tuple[list[_TensorConnectionPlan], list[dict[str, Any]]]:
    from .tensor.plan import ConnectionPlanner, check_fanout

    failures = check_fanout(stages, pipeline.tensor_connections, profile)
    if failures:
        raise _tensor_error(failures, [])

    planned = []
    records = []
    failures = []
    by_id = {stage.id: stage for stage in stages}
    for connection in pipeline.tensor_connections:
        planner = ConnectionPlanner(pipeline, stages, connection, profile)
        plan = planner.next()
        record = _plan_record(connection, plan, profile)
        records.append(record)
        if plan.status != "MODEL_SAT":
            failures.extend(_plan_failure(connection, plan))
            continue
        stage_id, import_name = connection.import_.rsplit(":", 1)
        imp = next(imp for imp in by_id[stage_id].tensor.imports if imp.name == import_name)
        planned.append(_TensorConnectionPlan(stage_id, imp, plan, record, connection, planner))
    if failures:
        raise _tensor_error(failures, records)
    return planned, records


def _tensor_resources(stage: StageManifest):
    """Enumerate declared loans and export backing with their owner paths."""
    if stage.tensor is None:
        return
    for direction, records, field in (("exports", stage.tensor.exports, "backing"),
                                       ("imports", stage.tensor.imports, "lends")):
        for record in records:
            for resource in getattr(record, field):
                yield f"stages.{stage.id}.tensor.{direction}.{record.name}.{field}.{resource.name}", resource


def _check_tensor_resource_bindings(
    pipeline: Pipeline, stages: Sequence[StageManifest],
    planned: Sequence[_TensorConnectionPlan], borrowed_handles: frozenset[str],
    solution: Solution | None = None,
) -> None:
    """Validate declaration-to-allocation refinement, reporting T-PLAN on failure.

    MODEL_SAT is not an allocation certificate. Borrowed intervals are never
    enlarged or relocated to make this check pass. Event lifetimes must contain
    the lend on every resolved connection. Symbolic extents are checked on solved intervals.
    Published runtime TMEM bindings and private cells are not ordinary allocations.
    """
    from .tensor.lifetimes import resolve_lifetimes

    lifetimes = resolve_lifetimes(pipeline, stages)
    backing: dict[str, list[Claim]] = {}
    for stage in stages:
        for claim in stage.claims:
            backing.setdefault(claim.handle, []).append(claim)
    generated = {
        f"stages.{item.stage_id}.tensor.imports.{item.imp.name}.lends.{resource.name}":
            item.plan.provenance["resource_demands"][resource.name]
        for item in planned for resource in item.imp.lends
        if resource.binding == "relocatable" and resource.location.handle not in borrowed_handles
        and resource.domain in {"smem", "barrier"}
    }
    failures = list(lifetimes.failures)
    for stage in stages:
        if stage.tensor is None:
            continue
        published = {binding.name for binding in stage.tensor.bindings}
        for path, resource in _tensor_resources(stage):
            name, fixed = resource.location.handle, resource.location.base
            if name is None:
                continue
            selected_base = pipeline.forced_bindings.get(name)
            if solution is not None:
                selected_base = solution.native_bases.get(name, selected_base)
            if (resource.binding == "fixed" and fixed is not None
                    and selected_base is not None and selected_base != fixed):
                failures.append(ClauseFailure(
                    "T-PLAN", path, "solved handle contradicts the declared fixed tensor base",
                    {"resource": resource.to_json_obj(), "handle": name,
                     "declared_base": fixed, "solved_base": selected_base},
                ))
                continue
            if resource.domain in {"reg", "metadata"}:
                continue
            required = generated.get(path, resource.extent)
            if path not in generated and name not in borrowed_handles:
                if resource.binding != "fixed" or fixed is not None:
                    continue
                if resource.domain == "tmem" and name in published:
                    continue
            domain = "named_barrier" if resource.domain == "barrier" else resource.domain
            handle = pipeline.handles.get(name)
            claims = backing.get(name, [])
            if path in generated and required == 0 and not claims:
                if (handle is None or handle.binding != "fixed" or selected_base != 0):
                    failures.append(ClauseFailure(
                        "T-PLAN", path, "unused generated lend must resolve to zero", {"handle": name}))
                continue
            covering = [
                claim for claim in claims
                if claim.domain == domain and claim.extent is not None
                and claim.align is not None and claim.align >= resource.align
                and claim.align % resource.align == 0
                and lifetimes.covers(stage.id, claim, resource) and (claim.offset or 0) <= 0
                and (claim.offset or 0) % resource.align == 0
                and (claim.extent.per or claim.extent.table
                     or claim.extent.const + (claim.offset or 0) >= required)
            ]
            valid = (
                handle is not None and handle.cls == "base" and handle.binding == "bind"
                and handle.domain == domain and bool(covering)
            )
            if solution is not None:
                base = solution.native_bases.get(name)
                valid = valid and base is not None and base % resource.align == 0 and any(
                    interval.handle == name
                    and interval.lo <= base and interval.hi >= base + required
                    for interval in solution.intervals.get(domain, ())
                )
            if not valid:
                failures.append(ClauseFailure(
                    "T-PLAN", path,
                    "tensor reservation lacks covering domain, extent, alignment, or lifetime",
                    {"resource": resource.to_json_obj(),
                     "handle": None if handle is None else handle.to_json_obj(),
                     "backing_claims": [claim.to_json_obj() for claim in claims],
                     "solved_base": None if solution is None else solution.native_bases.get(name),
                     "solved_intervals": [] if solution is None else [
                         interval.to_json_obj() for interval in solution.intervals.get(domain, ())
                         if interval.handle == name
                     ]},
                ))
    for item in planned:
        outcome = item.record["solver_outcome"]
        if solution is not None:
            outcome["resource_solver"] = {"status": "SAT", "solution": solution.to_json_obj()}
            for resource in item.imp.lends:
                if resource.domain not in {"smem", "barrier"}:
                    continue
                domain = "named_barrier" if resource.domain == "barrier" else resource.domain
                intervals = [interval for interval in solution.intervals.get(domain, ())
                             if interval.handle == resource.location.handle]
                outcome["reservation_realization"][resource.name]["allocated_extent"] = max(
                    (interval.hi - interval.lo for interval in intervals), default=0)
        outcome["reservation_validation"] = {
            "status": "REJECTED" if failures else "PASS",
            "phase": "composition" if solution is None else "solved",
            "failures": [failure.to_json_obj() for failure in failures],
        }
    if failures:
        raise _tensor_error(failures, [item.record for item in planned])


def _view_calls(
    pairs: Sequence[tuple[str, list[_EndpointRecord], list[_EndpointRecord]]], host: dict | None,
) -> dict[str, Any]:
    """Resolve each call into a role-less host-group stage (see `stubgen.ViewCall`).

    The caller's own view - a dispatched role view or, for a role-less host-group caller,
    the view it received - must hold a record; its scalars (image offset 0) are preferred.
    """
    from .stubgen import ViewCall, first_parameter_type
    from .symbols import host_view_stage

    calls: dict[str, ViewCall] = {}
    failures: list[ClauseFailure] = []
    for channel, producers, consumers in pairs:
        if not consumers or consumers[0].endpoint.protocol.call is None:
            continue
        callee = consumers[0]
        if not host_view_stage(callee.stage):
            continue
        caller = producers[0]
        path = f"$.channels.{channel}"
        if host is None:
            failures.append(ClauseFailure("L-PARAMS", path, "view call needs a host-params link", {}))
            continue
        groups = {stage_id: group for group in host["groups"] for stage_id in group["stage_ids"]}

        def records(group: dict, readable: Any) -> list[tuple[str, int]]:
            return [(key, group[key]["offset"]) for key in ("scalars", "residue")
                    if group[key] is not None and group[key]["bytes"] and readable(group[key])]

        group = groups.get(caller.stage.id)
        role = caller.endpoint.owner_role
        if group is None:
            sources = []
        elif host_view_stage(caller.stage):
            sources = records(group, lambda record: True)
        else:
            sources = records(group, lambda record: f"{caller.stage.id}:{role}" in record["read_roles"])
        if not sources:
            failures.append(ClauseFailure(
                "L-PARAMS", path, "a view call's caller must hold a host params view with a record",
                {"caller": caller.stage.id, "role": role},
            ))
            continue
        export = callee.stage.exports[callee.endpoint.protocol.call.export]
        export_symbol, decl = export.symbol, export.decl
        if first_parameter_type(decl, export_symbol) != "const void*":
            failures.append(ClauseFailure(
                "L-PARAMS", path, "an export of a role-less host-group stage takes its view as a "
                "leading const void* parameter", {"export": export_symbol},
            ))
            continue
        callee_group = groups[callee.stage.id]
        offsets = {key: offset for key, offset in records(callee_group, lambda record: True)}
        calls[channel] = ViewCall(*sources[0], offsets.get("residue"), offsets.get("scalars"))
    if failures:
        raise LinkError(failures)
    return calls


def _params_composition(
    pipeline: Pipeline, stages: Sequence[StageManifest],
) -> tuple[Pipeline, list[StageManifest]]:
    from .symbols import PARAMS_BASE_BYTES, PARAMS_BASE_HANDLE, params_accessor_stages

    users = params_accessor_stages(stages)
    if not users:
        return pipeline, list(stages)
    name = PARAMS_BASE_HANDLE
    if name in pipeline.handles or any(
        claim.handle == name for stage in stages for claim in stage.claims
    ):
        raise _operation_error(f"{name} is reserved for params publication")
    owner = users[0].id
    extent = PARAMS_BASE_BYTES
    handle = Handle(name, "base", "bind", "smem", None, f"claim:{owner}:{name}")
    claim = Claim("smem", name, "kernel", Extent(extent, {}, {}), 8,
                  0, None, None, None, None, None)
    return replace(pipeline, handles={**pipeline.handles, name: handle}), [
        replace(stage, claims=[*stage.claims, claim]) if stage.id == owner else stage
        for stage in stages
    ]


def _tensor_composition(
    pipeline: Pipeline, stages: Sequence[StageManifest],
    planned: Sequence[_TensorConnectionPlan],
) -> tuple[Pipeline, list[StageManifest]]:
    """Realize selected lend demands; preserve frozen backing and explicit claims."""
    handles = dict(pipeline.handles)
    forced = dict(pipeline.forced_bindings)
    claims = {stage.id: list(stage.claims) for stage in stages}
    existing_claims: dict[str, list[Claim]] = {}
    for stage in stages:
        for claim in stage.claims:
            existing_claims.setdefault(claim.handle, []).append(claim)
    frozen = set(pipeline.handles) | set(existing_claims)
    frozen.update(resource.location.handle for stage in stages if stage.tensor is not None
                  for export in stage.tensor.exports for resource in export.backing)
    demands: dict[str, int] = {}
    lending_connections: dict[str, set[str]] = {}
    lending_lifetimes: dict[str, TensorLifetime] = {}
    for item in planned:
        for resource in item.imp.lends:
            name = resource.location.handle
            if name is not None:
                lending_connections.setdefault(name, set()).add(item.connection.name)
                lending_lifetimes[name] = TensorLifetime(
                    item.imp.name, resource.lifetime["from"], resource.lifetime.get("to"))
            if (name is not None and name not in frozen and resource.binding == "relocatable"
                    and resource.domain in {"smem", "barrier"}):
                demands[name] = max(demands.get(name, 0),
                                    item.plan.provenance["resource_demands"][resource.name])
    declared_handles: set[str] = set()
    failures = []

    def generated_lifetime(name: str) -> str | TensorLifetime:
        if len(lending_connections.get(name, ())) == 1:
            return lending_lifetimes[name]
        return "kernel"

    def add_handle(handle: Handle) -> None:
        previous = handles.get(handle.name)
        if previous is not None and previous != handle:
            if handle.name in declared_handles and replace(handle, owner=previous.owner) == previous:
                return
            failures.append(ClauseFailure(
                "T-CONTRACT", f"pipeline.handles.{handle.name}",
                "tensor reservation conflicts with an existing resource binding",
                {"existing": previous.to_json_obj(), "generated": handle.to_json_obj()},
            ))
        else:
            handles[handle.name] = handle

    def reservation_key(claim: Claim) -> tuple:
        return (claim.domain, claim.handle, claim.extent, claim.align, claim.offset or 0,
                claim.absolute_base, claim.lifetime, claim.alias)

    def add_claim(stage_id: str, claim: Claim) -> None:
        previous = existing_claims.get(claim.handle, [])
        if previous:
            if any(reservation_key(item) == reservation_key(claim) for item in previous):
                return
            failures.append(ClauseFailure(
                "T-CONTRACT", f"stages.{stage_id}.claims.{claim.handle}",
                "tensor reservation disagrees with an existing physical claim",
                {"existing": [item.to_json_obj() for item in previous],
                 "generated": claim.to_json_obj()},
            ))
            return
        claims[stage_id].append(claim)
        existing_claims[claim.handle] = [claim]

    for stage in stages:
        for path, resource in _tensor_resources(stage):
            name, base = resource.location.handle, resource.location.base
            if name is None or (resource.binding == "fixed" and base is None):
                continue
            if resource.domain in {"reg", "metadata"}:
                if resource.binding == "relocatable":
                    failures.append(ClauseFailure(
                        "T-MODEL", path,
                        "relocatable private tensor storage has no ordinary allocator domain",
                        {"resource": resource.to_json_obj()},
                    ))
                continue
            # Hardware CTA barrier zero is reserved, not a named-barrier allocation.
            # A declared fixed handle for it can still receive a constant provider.
            if resource.domain == "barrier" and resource.binding == "fixed" and base == 0:
                continue
            domain = "named_barrier" if resource.domain == "barrier" else resource.domain
            declared_handles.add(name)
            if name in pipeline.handles:
                # Existing reservations are borrows: prove coverage, never resize.
                continue
            extent = demands.get(name, resource.extent)
            if extent == 0:
                add_handle(Handle(name, "base", "fixed", domain, None, f"claim:{stage.id}:{name}"))
                forced[name] = 0
                continue
            add_handle(Handle(name, "base", "bind", domain, None, f"claim:{stage.id}:{name}"))
            add_claim(stage.id, Claim(
                domain, name, generated_lifetime(name), Extent(extent, {}, {}), resource.align,
                0, None, None, None, base if resource.binding == "fixed" else None, None,
            ))
    for item in planned:
        for handle in item.plan.handles:
            add_handle(handle)
        for claim in item.plan.claims:
            add_claim(item.stage_id, replace(
                claim,
                extent=Extent(demands[claim.handle], {}, {})
                if claim.handle in demands else claim.extent,
                lifetime=generated_lifetime(claim.handle)
                if claim.handle in lending_connections else claim.lifetime,
            ))
        item.record["solver_outcome"]["declared_reservations"] = [
            resource.to_json_obj()
            for stage in stages if stage.id == item.stage_id
            for _path, resource in _tensor_resources(stage)
        ]
        item.record["solver_outcome"]["provenance"] = {
            **item.plan.provenance,
            "resource_lifetime": {
                claim.handle: claim.to_json_obj()["lifetime"]
                for claim in claims[item.stage_id]
                if claim.handle in lending_connections
            },
        }
        item.record["solver_outcome"]["resource_demands"] = item.plan.provenance["resource_demands"]
        item.record["solver_outcome"]["reservation_realization"] = {
            resource.name: {"declared_extent": resource.extent,
                            "required_extent": item.plan.provenance["resource_demands"][resource.name],
                            "allocated_extent": None,
                            "forced_binding": forced.get(resource.location.handle)}
            for resource in item.imp.lends if resource.domain in {"smem", "barrier"}
        }
    for item in planned:
        for request in item.plan.barrier_requests:
            name = request["handle"]
            handle = handles.get(name)
            if request.get("reserved") == 0:
                continue
            if handle is None or handle.domain != "named_barrier":
                failures.append(ClauseFailure(
                    "T-CONTRACT", f"pipeline.handles.{name}",
                    "adapter barrier request has no declared named-barrier allocation",
                    {"request": request},
                ))
    if failures:
        raise _tensor_error(failures, [item.record for item in planned])
    composition = replace(pipeline, handles=handles, forced_bindings=forced)
    owners = [
        replace(stage, claims=claims[stage.id]) if claims[stage.id] != stage.claims else stage
        for stage in stages
    ]
    _check_tensor_resource_bindings(composition, owners, planned, frozenset(pipeline.handles))
    return composition, owners


def _lower_tensors(
    planned: Sequence[_TensorConnectionPlan], solution: Solution, profile: Profile,
    smem_reuse: dict[str, Any],
) -> dict[str, AdapterSource]:
    from .tensor.lower import LoweringToolingUnavailable, LoweringUnsupported, emit_adapter

    adapters = {}
    for index, item in enumerate(planned):
        try:
            overlaps = [overlap for overlap in smem_reuse["overlaps"]
                        if overlap["connection"] == item.connection.name]
            adapter = emit_adapter(
                item.plan, item.imp, solution, profile,
                witness_events=frozenset(overlap["event"] for overlap in overlaps),
            )
        except LoweringUnsupported as exc:
            detail = exc.to_json_obj()
            # Unrunnable tooling fails every route alike: searching on would only spend solver time.
            if not isinstance(exc, LoweringToolingUnavailable) and item.advance(detail):
                raise _Relower() from exc
            item.record.clear()
            item.record.update(_plan_record(item.connection, item.plan, profile))
            item.record["lowering"] = detail
            raise _tensor_error([ClauseFailure(
                "T-LOWER", f"pipeline.tensor_connections.{item.record['name']}",
                f"tensor adapter lowering unsupported: {exc}",
                {"status": "LOWERING_UNSUPPORTED", "backend": detail,
                 "candidates": item.plan.provenance.get("classes", [])},
            )], [entry.record for entry in planned]) from exc
        filename = f"tensor_adapter_{index:03d}.cu"
        adapters[filename] = adapter
        item.record["lowering"] = {
            "status": "LOWERED", "name": adapter.name, "source": filename,
            "source_digest": sha256_hex(adapter.source.encode("utf-8")),
            "defined_symbols": list(adapter.defined_symbols), "resources": adapter.resources,
        }
        item.record["trusted_premises"] = list(adapter.trusted_premises)
        item.record["trusted_premises"].extend(
            {"name": name, "scope": item.connection.name,
             "justification": "owner event-lifetime quiescence declaration"}
            for name in sorted({
                f"claim_{overlap['ending_handle']}_quiescent_before_{item.connection.name}.{overlap['event']}"
                for overlap in overlaps
            })
        )
    return adapters


def _link_result(
    pipeline: Pipeline,
    stages: Sequence[StageManifest],
    artifacts: Sequence[_Artifact],
    solution: Solution,
    params: ParamsLayout,
    expansions: Sequence[ProtocolExpansion],
    kernel_symbol: str,
    observed: tuple[int, int, int],
    generated_sources: Mapping[str, str],
    sass_report: dict[str, Any] | None,
    host: dict[str, Any] | None = None,
) -> LinkResult:
    advisory = (
        None
        if sass_report is None
        else {
            **sass_report,
            "channels": sorted(expansion.channel for expansion in expansions),
        }
    )
    return LinkResult(
        kernel_symbol=kernel_symbol,
        threads=solution.launch.threads,
        regs_cap=solution.launch.regs_cap,
        dyn_smem_bytes=solution.launch.dyn_smem_bytes,
        cluster=_cluster_result(solution, stages),
        params=params,
        bindings=_binding_values(pipeline, solution),  # type: ignore[arg-type]
        variants={key: solution.variants[key] for key in sorted(solution.variants)},
        intervals={key: solution.intervals[key] for key in sorted(solution.intervals)},
        protocol_expansions=sorted(expansions, key=lambda item: item.channel),
        observed=observed,
        input_hashes={
            artifact.stage.id: artifact.artifact_sha256
            for artifact in sorted(artifacts, key=lambda item: item.stage.id)
        },
        generated_hashes={
            filename: sha256_hex(source.encode("utf-8"))
            for filename, source in sorted(generated_sources.items())
        },
        sass_report=advisory,
        host=host,
        solver={**solver_provenance(),
                "tiers": [tier.to_json_obj() for tier in solution.objective],
                "rlimit_used": solution.rlimit_used},
    )


def link(pipeline_path: str | Path, out_dir: str | Path) -> LinkResult:
    """Resolve, generate, device-link, audit, and publish one ABI-v2 pipeline."""

    pipeline_file = Path(pipeline_path).resolve()
    output_dir = Path(out_dir).resolve()
    pipeline, artifacts = _load_assembly(pipeline_file)
    link_artifacts = _artifacts_in_pipeline_order(pipeline, artifacts)
    stages = [artifact.stage for artifact in artifacts]
    dropped_endpoints = [record for artifact in artifacts for record in artifact.dropped_endpoints]
    profile = get_profile(pipeline.target.profile)
    _check_symbols(artifacts)
    math = _math_mode(stages, pipeline.math)
    planned: list[_TensorConnectionPlan] = []
    tensor_records: list[dict[str, Any]] = []
    tensor_required: dict[str, str] = {}
    tensor_borrowed_handles: frozenset[str] = frozenset()
    if pipeline.tensor_connections:
        planned, tensor_records = _plan_tensors(pipeline, stages, profile)
        tensor_borrowed_handles = frozenset(pipeline.handles)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _operation_error(f"cannot create output directory {output_dir}: {exc}") from exc
    link_json = output_dir / "link.json"
    cubin = output_dir / "pipeline.cubin"
    for stale in (link_json, cubin):
        try:
            stale.unlink(missing_ok=True)
        except OSError as exc:
            raise _operation_error(f"cannot remove stale linker output {stale}: {exc}") from exc

    # Composition → solve → lowering is a function of the original
    # assembly and the connections' current candidates. A candidate that does not
    # lower advances its planner and the whole block is redone from `assembly`.
    assembly = (pipeline, stages)
    while True:
        pipeline, stages = assembly
        if planned:
            # Every record's solution-derived evidence belongs to the previous
            # iteration's solution; rebuild all of them, not only the advanced one.
            for item in planned:
                item.record.clear()
                item.record.update(_plan_record(item.connection, item.plan, profile))
            pipeline, stages = _tensor_composition(pipeline, stages, planned)
            tensor_required = {
                symbol: stage.id
                for stage in stages if stage.tensor is not None
                for imp in stage.tensor.imports
                for symbol in (imp.callable.symbol, *imp.callable.events.values())
            }
        pipeline, stages = _params_composition(pipeline, stages)
        from .tensor.lifetimes import resolve_lifetimes

        lifetimes = resolve_lifetimes(pipeline, stages)
        if lifetimes.failures:
            raise _tensor_error(lifetimes.failures, tensor_records)
        try:
            pairs = _pair_channels(stages)
            canon = _canonical_endpoints(stages, pipeline, profile)
            _unify_pairs(pairs, canon)
            expansions = [
                _expand_protocol(channel, producers, consumers, stages, pipeline, canon, profile)
                for channel, producers, consumers in pairs
            ]
            expansions.sort(key=lambda item: item.channel)

            params = (
                ParamsLayout(total_bytes=0, align=profile.caps.param_align, entries=[])
                if host_link(stages)
                else _params_layout(stages, profile)
            )
            solve_pipeline = pipeline
            try:
                solution = _solve_pipeline(
                    solve_pipeline, stages, canon, expansions, profile, lifetimes.ordered_pairs)
            except PipeLinkError as exc:
                if tensor_records:
                    for record in tensor_records:
                        record["solver_outcome"]["resource_solver"] = {
                            "status": "REJECTED", "failures": [failure.to_json_obj() for failure in exc.failures],
                        }
                    raise _tensor_error(exc.failures, tensor_records) from exc
                raise

            final: tuple[
                Pipeline,
                Solution,
                str,
                tuple[int, int, int],
                dict[str, str],
                dict[str, Any] | None,
            ] | None = None
            for attempt in range(2):
                from .protocol import check_init_warps, resolve_native

                init_owners = check_init_warps(stages, solution.warp_map)
                expansions = [
                    replace(expansion, init_owner=init_owners[expansion.channel])
                    if expansion.init_materializer is not None else expansion
                    for expansion in expansions
                ]
                expansions = resolve_native(expansions, stages, solution, profile)
                if planned:
                    _check_tensor_resource_bindings(
                        solve_pipeline, stages, planned, tensor_borrowed_handles, solution,
                    )
                    from .tensor.plan import check_fanout_storage

                    failures = check_fanout_storage(
                        stages, solve_pipeline.tensor_connections,
                        {item.record["name"]: item.plan for item in planned}, solution,
                    )
                    if failures:
                        raise _tensor_error(failures, [item.record for item in planned])
                smem_reuse = lifetimes.report(solution)
                adapters = _lower_tensors(planned, solution, profile, smem_reuse) if planned else {}
                generation_pipeline = (
                    solve_pipeline
                    if "pipelink_launch_regs" in pipeline.handles
                    else pipeline
                )
                host = resolve_host(stages, solution, generation_pipeline)
                if host is not None:
                    params = host_params_layout(host)
                generated_sources = {
                    "binding_stub.cu": _generate_stub(
                        solution, generation_pipeline, stages, expansions, _view_calls(pairs, host)
                    )
                }
                if host is None:
                    generated_sources["dispatch.cu"] = _generate_dispatch(
                        solution, generation_pipeline, stages, expansions, params
                    )
                else:
                    generated_sources["dispatch.cu"] = _generate_dispatch(
                        solution, generation_pipeline, stages, expansions, params, host
                    )
                generated_sources.update({name: adapter.source for name, adapter in adapters.items()})
                compile_kwargs = (
                    {"adapters": adapters, "tensor_required": tensor_required}
                    if planned else {}
                )

                try:
                    kernel_symbol, observed, sass_report = _compile_and_measure(
                        output_dir,
                        link_artifacts,
                        generated_sources,
                        profile,
                        solution,
                        _KERNEL_SYMBOL,
                        math,
                        **compile_kwargs,
                    )
                except Exception as exc:
                    cubin.unlink(missing_ok=True)
                    if tensor_records and isinstance(exc, LinkError):
                        for record in tensor_records:
                            failure_record = {
                                "failures": [failure.to_json_obj() for failure in exc.failures],
                            }
                            if any(failure.clause_id == "T-LOWER" for failure in exc.failures):
                                record["lowering"] = {
                                    **(record["lowering"] or {}), **failure_record,
                                    "status": "LOWERING_UNSUPPORTED", "reason": "generated_provider",
                                }
                            else:
                                record["native"] = {"status": "TOOLING_FAILURE", **failure_record}
                        raise _tensor_error(exc.failures, tensor_records) from exc
                    raise
                fits, _demand, _capacity, observed_allocation = _register_pool(
                    solution, observed[0], profile
                )
                for record in tensor_records:
                    record["native"] = {
                        "status": "DEPLOYED" if fits else "NATIVE_RESOURCE_REJECTED",
                        "attempt": attempt + 1,
                        "observed": list(observed),
                        "register_allocation": observed_allocation,
                        "register_demand": _demand,
                        "register_capacity": _capacity,
                        "launch": solution.launch.to_json_obj(),
                        "adapter_compile": profile.adapter_compile.to_json_obj(),
                        "profile_digest": sha256_hex(canonical_bytes(profile.to_json_obj())),
                        "stage_targets": {
                            artifact.stage.id: artifact.stage.target.to_json_obj()
                            for artifact in artifacts
                        },
                    }
                if fits:
                    final = (
                        solve_pipeline,
                        solution,
                        kernel_symbol,
                        observed,
                        generated_sources,
                        sass_report,
                    )
                    break
                cubin.unlink(missing_ok=True)
                if attempt == 1:
                    failures = [_register_failure(solution, observed[0], profile, after_resolve=True)]
                    raise _tensor_error(failures, tensor_records) if tensor_records else LinkError(failures)
                try:
                    solve_pipeline = _pipeline_with_register_bound(pipeline, observed_allocation)
                    solution = _solve_pipeline(
                        solve_pipeline, stages, canon, expansions, profile, lifetimes.ordered_pairs)
                except PipeLinkError as exc:
                    if tensor_records:
                        raise _tensor_error(exc.failures, tensor_records) from exc
                    raise

        except _Relower:
            continue
        break

    if final is None:
        raise AssertionError("register-audit loop did not produce a terminal result")
    _final_pipeline, solution, kernel_symbol, observed, generated_sources, sass_report = final
    result = _link_result(
        pipeline,
        stages,
        artifacts,
        solution,
        params,
        expansions,
        kernel_symbol,
        observed,
        generated_sources,
        sass_report,
        host,
    )
    if tensor_records:
        result = replace(result, tensor={"connections": tensor_records, "smem_reuse": smem_reuse})
    if dropped_endpoints:
        result = replace(result, dropped_endpoints=dropped_endpoints)
    if pipeline.math is not None:
        result = replace(result, math_override={
            "mode": math.to_json_obj(),
            "packed": {stage.id: (stage.target.math or NVCC_DEFAULT_MATH).to_json_obj()
                       for stage in sorted(stages, key=lambda item: item.id)
                       if (stage.target.math or NVCC_DEFAULT_MATH) != math},
        })
    try:
        link_json.write_bytes(canonical_bytes(result.to_json_obj()))
    except (OSError, TypeError, ValueError) as exc:
        raise _operation_error(f"cannot write link result {link_json}: {exc}") from exc
    return result


__all__ = ["link"]
