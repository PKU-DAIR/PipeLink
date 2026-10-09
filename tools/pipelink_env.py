#!/usr/bin/env python3
"""Load PipeLink's machine-local environment configuration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import tempfile
from typing import Mapping


_BASE_KEYS = (
    "PIPELINK_CUDA_HOME",
    "PIPELINK_SCRATCH_ROOT",
    "PIPELINK_GPUS",
    "PIPELINK_LOCKED_GPUS",
    "PIPELINK_MAX_CONCURRENT_GPUS",
    "PIPELINK_LOCK_SM_MHZ",
    "PIPELINK_LOCK_STATE_DIR",
    "PIPELINK_CHECKPOINT_ROOT",
    "PIPELINK_ALLOW_UNLOCKED",
    "PIPELINK_NO_CACHE",
)


def _defaults(repo_root: Path) -> dict[str, str]:
    scratch_root = Path(tempfile.gettempdir()) / f"pipelink-{os.getuid()}"
    return {
        "PIPELINK_CUDA_HOME": "",
        "PIPELINK_SCRATCH_ROOT": str(scratch_root),
        "PIPELINK_GPUS": "",
        "PIPELINK_LOCKED_GPUS": "",
        "PIPELINK_MAX_CONCURRENT_GPUS": "4",
        "PIPELINK_LOCK_SM_MHZ": "",
        "PIPELINK_LOCK_STATE_DIR": str(scratch_root / "gpulockd"),
        "PIPELINK_CHECKPOINT_ROOT": str(repo_root / "checkpoints"),
        "PIPELINK_ALLOW_UNLOCKED": "0",
        "PIPELINK_NO_CACHE": "0",
    }


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in _BASE_KEYS:
            raise ValueError(f"{path}:{line_number}: unknown PipeLink key {key!r}")
        values[key] = value.strip()
    return values


def _validate_gpu_list(key: str, raw: str) -> str:
    if not raw:
        return ""
    fields = raw.split(",")
    if any(not field.strip() for field in fields):
        raise ValueError(f"{key} must be a comma-separated list of GPU indices")
    try:
        gpus = [int(field.strip()) for field in fields]
    except ValueError as exc:
        raise ValueError(f"{key} must be a comma-separated list of GPU indices") from exc
    if any(gpu < 0 for gpu in gpus):
        raise ValueError(f"{key} GPU indices must be non-negative integers")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"{key} must not contain duplicate GPU indices")
    return ",".join(str(gpu) for gpu in gpus)


def _prepend_path(directory: str, current: str) -> str:
    if not directory:
        return current
    entries = [entry for entry in current.split(os.pathsep) if entry and entry != directory]
    return os.pathsep.join((directory, *entries))


def _derive(values: dict[str, str], environment: Mapping[str, str]) -> None:
    scratch_root = Path(values["PIPELINK_SCRATCH_ROOT"])
    values["UV_CACHE_DIR"] = str(scratch_root / "uv-cache")
    values["CUDA_HOME"] = values["PIPELINK_CUDA_HOME"]

    cuda_home = values["PIPELINK_CUDA_HOME"]
    values["PATH"] = _prepend_path(
        str(Path(cuda_home) / "bin") if cuda_home else "",
        environment.get("PATH", ""),
    )
    values["LD_LIBRARY_PATH"] = _prepend_path(
        str(Path(cuda_home) / "lib64") if cuda_home else "",
        environment.get("LD_LIBRARY_PATH", ""),
    )


def load(repo_root: Path | str | None = None) -> dict[str, str]:
    """Return validated config with process environment taking precedence."""
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    values = _defaults(root)
    file_values = _read_env_file(root / ".env")
    values.update(file_values)
    values.update({key: os.environ[key] for key in _BASE_KEYS if key in os.environ})

    values["PIPELINK_GPUS"] = _validate_gpu_list(
        "PIPELINK_GPUS", values["PIPELINK_GPUS"]
    )
    values["PIPELINK_LOCKED_GPUS"] = _validate_gpu_list(
        "PIPELINK_LOCKED_GPUS", values["PIPELINK_LOCKED_GPUS"]
    )
    try:
        max_concurrent = int(values["PIPELINK_MAX_CONCURRENT_GPUS"])
    except ValueError as exc:
        raise ValueError("PIPELINK_MAX_CONCURRENT_GPUS must be a positive integer") from exc
    if max_concurrent <= 0:
        raise ValueError("PIPELINK_MAX_CONCURRENT_GPUS must be a positive integer")
    for key in ("PIPELINK_ALLOW_UNLOCKED", "PIPELINK_NO_CACHE"):
        if values[key] not in {"0", "1"}:
            raise ValueError(f"{key} must be 0 or 1")

    if "PIPELINK_LOCK_STATE_DIR" not in file_values and "PIPELINK_LOCK_STATE_DIR" not in os.environ:
        values["PIPELINK_LOCK_STATE_DIR"] = str(
            Path(values["PIPELINK_SCRATCH_ROOT"]) / "gpulockd"
        )
    _derive(values, os.environ)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        action="store_true",
        help="print shell export statements",
    )
    args = parser.parse_args()
    values = load()
    if args.export:
        for key, value in values.items():
            print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
