from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence




def _write_canonical(path: Path, value: Any) -> None:
    from .canonjson import canonical_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _pack_command(args: argparse.Namespace) -> None:
    from .canonjson import canonical_dumps
    from .pack import pack_cu

    output_path = Path(args.out).resolve()
    report_path = (
        Path(args.report).resolve()
        if args.report is not None
        else Path(str(output_path) + ".report.json")
    )
    if output_path == report_path:
        raise ValueError("--out and --report must name different files")
    if not args.source.endswith(".cu"):
        raise ValueError("the positional stage source must end with .cu")
    report = pack_cu(
        Path(args.source), output_path,
        includes=[Path(path) for path in args.include], defines=args.define,
        nvcc_flags=args.nvcc_flag, profile=args.profile, toolkit=args.toolkit,
    )
    _write_canonical(report_path, report)
    sys.stdout.write(canonical_dumps(report))


def _inspect_command(args: argparse.Namespace) -> None:
    from .canonjson import canonical_dumps, sha256_hex
    from .container import _raise_pack_error, read
    from .schema import load_stage_manifest

    manifest_obj, object_bytes = read(Path(args.plink).resolve())
    manifest = load_stage_manifest(manifest_obj)
    expected = manifest_obj["payload"]["sha256"]
    actual = sha256_hex(object_bytes)
    if actual != expected:
        _raise_pack_error(
            "$.payload.sha256",
            f"payload SHA-256 mismatch: expected {expected}, got {actual}",
        )
    if args.json:
        sys.stdout.write(canonical_dumps(manifest_obj))
        return
    payload = manifest_obj["payload"]
    sys.stdout.write(
        f"{manifest.id}: {len(object_bytes)} bytes, sha256={payload['sha256']}\n"
    )


def _link_command(args: argparse.Namespace) -> None:
    from .canonjson import canonical_dumps
    from .link import link

    result = link(Path(args.pipeline), Path(args.out_dir))
    sys.stdout.write(canonical_dumps(result.to_json_obj()))




def _parse_stage_path_args(values: Sequence[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        stage_id, separator, raw_path = value.partition("=")
        if not separator or not stage_id or not raw_path:
            raise ValueError(f"{option} must have the form STAGE_ID=PATH")
        if stage_id in result:
            raise ValueError(f"duplicate {option} stage ID {stage_id!r}")
        result[stage_id] = Path(raw_path)
    return {key: result[key] for key in sorted(result)}


def _parse_blob_args(values: Sequence[str]) -> dict[str, Path]:
    return _parse_stage_path_args(values, "--blob")


def _parse_provider_args(values: Sequence[str]) -> dict[str, Path]:
    return _parse_stage_path_args(values, "--provider")


def _parse_provider_arg_args(values: Sequence[str]) -> dict[str, Path]:
    return _parse_stage_path_args(values, "--provider-args")


def _parse_buffer_args(values: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    buffers = {}
    for value in values:
        name, separator, spec = value.partition("=")
        if not separator or not name or not spec:
            raise ValueError("--buffer must have the form NAME=FILE.npy or NAME=SHAPE:DTYPE")
        if name in buffers:
            raise ValueError(f"duplicate --buffer name {name!r}")
        if spec.endswith(".npy"):
            buffers[name] = np.load(spec, allow_pickle=False)
            continue
        shape, separator, dtype = spec.rpartition(":")
        if not separator or not shape or not dtype:
            raise ValueError("--buffer allocation must have the form NAME=64x64:float32")
        try:
            extents = tuple(int(extent) for extent in shape.split("x"))
            if any(extent < 0 for extent in extents):
                raise ValueError("negative extent")
            element_type = np.dtype(dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid --buffer spec {spec!r}; use x-separated nonnegative dimensions "
                             "and a NumPy dtype (uint16 for raw bf16 bits)") from exc
        if element_type.hasobject or not element_type.itemsize:
            raise ValueError("--buffer requires a fixed non-object NumPy dtype")
        buffers[name] = (extents, element_type)
    return buffers


def _parse_dimensions(value: str, option: str) -> tuple[int, int, int]:
    try:
        dimensions = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{option} requires X,Y,Z positive integer dimensions") from exc
    if len(dimensions) != 3 or any(extent <= 0 for extent in dimensions):
        raise ValueError(f"{option} requires X,Y,Z positive integer dimensions")
    return dimensions


def _load_params_builder(spec: str) -> Any:
    import runpy

    filename, separator, function = spec.rpartition(":")
    if not separator or not filename or not function:
        raise ValueError("--params must have the form PY_FILE:FUNCTION")
    builder = runpy.run_path(filename).get(function)
    if not callable(builder):
        raise ValueError(f"--params function {function!r} is not callable in {filename}")
    return builder


def _launch_command(args: argparse.Namespace) -> None:
    from .launcher import launch

    link_json = Path(args.link_json)
    blobs = _parse_blob_args(args.blob)
    cluster = None if args.cluster is None else _parse_dimensions(args.cluster, "--cluster")
    providers = _parse_provider_args(args.provider)
    provider_args = _parse_provider_arg_args(args.provider_args)
    completed = launch(
        link_json,
        blobs,
        cluster,
        provider_paths=providers,
        provider_arg_paths=provider_args,
        host_inputs_path=args.host_inputs,
        device=args.device,
        buffers=_parse_buffer_args(args.buffer),
        params=None if args.params is None else _load_params_builder(args.params),
        grid=None if args.grid is None else _parse_dimensions(args.grid, "--grid"),
        output_paths=_parse_stage_path_args(args.output, "--output"),
    )
    sys.stdout.write(completed.stdout)


def _clauses_command(args: argparse.Namespace) -> None:
    import importlib
    import pkgutil

    from . import __name__ as package, __path__ as package_path
    from .canonjson import canonical_dumps
    from .model import CLAUSES

    # Clauses register at module import time; import every module so the catalogue is complete.
    for module in pkgutil.walk_packages(package_path, f"{package}."):
        if not module.name.endswith(".__main__"):
            importlib.import_module(module.name)
    if args.json:
        sys.stdout.write(canonical_dumps(CLAUSES))
        return
    for clause_id in sorted(CLAUSES):
        sys.stdout.write(f"{clause_id}: {CLAUSES[clause_id]}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipec2",
        description="Pack, inspect, resolve, link, and launch PipeLink ABI-v2 artifacts.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    pack = subcommands.add_parser(
        "pack", help="compile and pack an annotated CUDA stage"
    )
    pack.add_argument("source", metavar="stage.cu")
    pack.add_argument("-o", "--out", required=True)
    pack.add_argument("-I", "--include", action="append", default=[], metavar="DIR")
    pack.add_argument("-D", "--define", action="append", default=[], metavar="K=V")
    pack.add_argument("--nvcc-flag", action="append", default=[], metavar="FLAG",
                      help="extra nvcc flag appended to both the emit and payload recipes (repeatable; use --nvcc-flag=--flag)")
    pack.add_argument("--profile", default="sm_103a")
    pack.add_argument("--toolkit", default="cuda-13.3.0")
    pack.add_argument("--report")
    pack.set_defaults(handler=_pack_command)

    inspect = subcommands.add_parser("inspect", help="validate and inspect a .plink")
    inspect.add_argument("plink")
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_inspect_command)

    link = subcommands.add_parser("link", help="resolve and link a pipeline manifest")
    link.add_argument("pipeline")
    link.add_argument("--out-dir", required=True)
    link.set_defaults(handler=_link_command)


    launch = subcommands.add_parser("launch", help="launch a linked cubin")
    launch.add_argument("link_json")
    launch.add_argument(
        "--blob",
        action="append",
        default=[],
        metavar="STAGE_ID=PATH",
        help="already-built opaque stage params blob",
    )
    launch.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="GROUP=SO",
        help="owner provider shared object keyed by params group",
    )
    launch.add_argument(
        "--provider-args",
        action="append",
        default=[],
        metavar="GROUP=PATH",
        help="raw caller-owned argument bytes passed to a selected host provider",
    )
    launch.add_argument("--host-inputs", help="caller scalar and fixture-resource JSON")
    launch.add_argument("--device", type=int, default=0, help="visible CUDA device ordinal")
    launch.add_argument("--cluster", metavar="X,Y,Z", help="admitted cluster")
    launch.add_argument("--grid", metavar="X,Y,Z", help="host-less grid; block size comes from the link")
    launch.add_argument("--buffer", action="append", default=[], metavar="NAME=SPEC",
                        help="host-less buffer: FILE.npy or SHAPE:DTYPE, e.g. out=64x64:float32")
    launch.add_argument("--params", metavar="PY_FILE:FUNCTION",
                        help="host-less Python builder: device addresses -> stage bytes")
    launch.add_argument("--output", action="append", default=[], metavar="NAME=FILE.npy",
                        help="save a host-less buffer after one launch")
    launch.set_defaults(handler=_launch_command)

    clauses = subcommands.add_parser("clauses", help="enumerate stable validation clauses")
    clauses.add_argument("--json", action="store_true")
    clauses.set_defaults(handler=_clauses_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .model import PipeLinkError

    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        args.handler(args)
    except (OSError, PipeLinkError, ValueError) as exc:
        print(f"pipec2: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
