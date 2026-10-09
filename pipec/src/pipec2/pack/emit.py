"""Compile one annotated CUDA TU twice, never projecting it through host shims."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from ..canonjson import canonical_bytes, sha256_hex
from ..model import NVCC_DEFAULT_MATH, ClauseFailure, MathMode, PackError, register_clause
from ..symbols import cuda_home


register_clause("P-COMPILE", phase="pack", description="The annotated CUDA translation unit compiles and emits a registry.")


@dataclass(frozen=True)
class Compilation:
    registry: dict[str, Any]
    object_path: Path
    provenance_inputs: dict[str, str]
    compile: dict[str, Any]
    math: MathMode


_MATH_SWITCH = re.compile(r"--?(ftz|prec-div|prec-sqrt|fmad)=(true|false)")


def math_mode(nvcc_flags: Sequence[str]) -> MathMode:
    """The floating-point codegen switches the payload is compiled with, in nvcc's order."""
    mode = asdict(NVCC_DEFAULT_MATH)
    for flag in nvcc_flags:
        if flag in {"--use_fast_math", "-use_fast_math"}:
            mode.update(ftz=True, prec_div=False, prec_sqrt=False, fmad=True)
        elif (match := _MATH_SWITCH.fullmatch(flag)) is not None:
            mode[match.group(1).replace("-", "_")] = match.group(2) == "true"
    return MathMode(**mode)


def _run(command: list[str], env: Mapping[str, str], operation: str) -> bytes:
    try:
        result = subprocess.run(command, env=dict(env), capture_output=True, check=False)
    except OSError as error:
        raise PackError([ClauseFailure("P-COMPILE", operation, f"cannot execute compiler or emit program: {error}", {})]) from error
    if result.returncode:
        raise PackError([ClauseFailure("P-COMPILE", operation,
                                      f"{operation} failed ({result.returncode}): {result.stderr.decode('utf-8', errors='replace').rstrip()}",
                                      {})])
    return result.stdout


def compile_registry(
    source: Path, scratch: Path, *, includes: Sequence[Path] = (), defines: Sequence[str] = (),
    nvcc_flags: Sequence[str] = (), profile: str = "sm_103a", toolkit: str = "cuda-13.3.0",
    env: Mapping[str, str],
) -> Compilation:
    """Emit registry first, then produce the single LTO payload for the same defines."""
    from ..profile import get_profile

    selected = get_profile(profile)
    if toolkit != selected.toolkit:
        raise ValueError(f"toolkit must match {profile}: {selected.toolkit}")
    seen = set()
    for definition in defines:
        name, separator, _ = definition.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None or name == "PIPELINK_EMIT":
            raise ValueError("-D definitions require K=V with a non-reserved C++ macro name")
        if name in seen:
            raise ValueError(f"duplicate definition for {name!r}")
        seen.add(name)
    source = source.resolve()
    if source.suffix != ".cu":
        raise ValueError("the stage source must end with .cu")
    package = Path(__file__).parent
    runner = package / "runner.cpp"
    include_dir = package / "include"
    host_include_dir = package.parent / "host" / "include"
    compiler = cuda_home(env) / "bin" / "nvcc"
    # Extra nvcc flags follow the fixed recipe so a stage may override e.g. -std=c++20.
    flags = ["-std=c++17", f"-I{include_dir}", f"-I{host_include_dir}", *(f"-I{path.resolve()}" for path in includes),
             *(f"-D{definition}" for definition in defines), *nvcc_flags]
    arch = selected.arch
    emit_flags = ["-x", "cu", "-DPIPELINK_EMIT", "--expt-relaxed-constexpr", "-gencode",
                  f"arch={arch.compute},code={arch.sm}", *flags]
    # The payload recipe matches the linker's adapter compile and the one-TU parity control
    # (-O3 -DNDEBUG): release asserts/traps in CUTLASS change register allocation across the seam.
    object_flags = ["-gencode", f"arch={arch.compute},code={arch.lto}", "-dc", "-rdc=true", "-O3", "-DNDEBUG", *flags]
    executable = scratch / "emit-0"
    object_path = scratch / "stage-0.o"
    tool_env = {**env, "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(scratch)}
    compiler_version = _run([str(compiler), "--version"], tool_env, "query nvcc version").decode("utf-8")
    emit_command = [str(compiler), *emit_flags, str(source), str(runner), "-o", str(executable)]
    _run(emit_command, tool_env, "compile emit program")
    output = _run([str(executable)], tool_env, "run emit program")
    try:
        registry = json.loads(output)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackError([ClauseFailure("P-COMPILE", "registry", "emit program did not print UTF-8 registry JSON", {})]) from error
    if not isinstance(registry, dict):
        raise PackError([ClauseFailure("P-COMPILE", "registry", "emit program must print one JSON object", {})])
    device_command = [str(compiler), *object_flags, str(source), "-o", str(object_path)]
    _run(device_command, tool_env, "compile device object")
    recipe = {"compiler": str(compiler), "version": compiler_version, "emit_flags": emit_flags,
              "object_flags": object_flags, "profile": profile, "toolkit": toolkit}
    inputs = {"tu": sha256_hex(source.read_bytes()), "recipe": sha256_hex(canonical_bytes(recipe)),
              "header": sha256_hex((include_dir / "pipelink" / "stage.hpp").read_bytes()),
              "runner": sha256_hex(runner.read_bytes())}
    return Compilation(registry, object_path, inputs, {
        "device": device_command, "emit": emit_command,
        "includes": [str(include_dir), str(host_include_dir), *(str(path.resolve()) for path in includes)],
        "defines": dict(definition.split("=", 1) for definition in defines),
    }, math_mode(nvcc_flags))
