"""Dependency-free command, packaging, and evidence carrier for stage builds."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def path_key(path: Path, root: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class CommandRecorder:
    """Run with explicit owner cwd/environment; retain logs even on command failure."""

    def __init__(self, *, cwd: Path, env: Mapping[str, str], root: Path,
                 timeout: float | None = None, record_path: Path | None = None,
                 metadata: Mapping[str, object] | None = None):
        self.cwd = cwd
        self.env = env
        self.root = root
        self.timeout = timeout
        self.record_path = record_path
        self.metadata = metadata or {}
        self.records: list[dict] = []

    def run(self, command: Sequence[str | Path], log_path: Path, *,
            env: Mapping[str, str] | None = None) -> bytes:
        argv = [str(value) for value in command]
        print(shlex.join(argv), flush=True)
        started = time.monotonic()
        result = subprocess.run(argv, cwd=self.cwd, env=self.env if env is None else env,
                                capture_output=True, timeout=self.timeout)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(result.stdout + result.stderr)
        self.records.append({"argv": argv, "cwd": str(self.cwd), "returncode": result.returncode,
                             "log": path_key(log_path, self.root), "log_sha256": digest(log_path),
                             "wall_seconds": time.monotonic() - started})
        if self.record_path is not None:
            write_json(self.record_path, {**self.metadata, "commands": self.records})
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): {shlex.join(argv)}\n"
                               f"log: {log_path}\n" + result.stderr.decode(errors="replace"))
        return result.stdout


def dependency_hashes(paths: Iterable[Path], *, cwd: Path, root: Path) -> dict[str, str]:
    """Hash host Make depfiles' dependencies (one rule, no drive-letter prefixes)."""
    dependencies = set()
    for path in paths:
        _, body = path.read_text().replace("\\\n", " ").split(":", 1)
        dependencies.update((cwd / item).resolve() for item in shlex.split(body))
    return {path_key(path, root): digest(path) for path in sorted(dependencies)}


def write_hash_index(out: Path, stage_key: str, artifacts: Mapping[str, object]) -> None:
    """Replace one stage namespace without losing cooperating builders' evidence."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".hashes.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out / "HASHES.json"
        hashes = json.loads(path.read_text()) if path.exists() else {}
        hashes[stage_key] = artifacts
        write_json(path, hashes)


def pack_cu(recorder: CommandRecorder, source: Path, out_plink: Path, *,
            includes: list[Path], defines: dict[str, str], log: Path,
            env: Mapping[str, str], nvcc_flags: Sequence[str] = (),
            profile: str | None = None) -> dict:
    """Pack one stage; `profile` names a non-default pipec2 target profile."""
    report = Path(f"{out_plink}.report.json")
    recorder.run(["uv", "run", "--project", "pipec", "--locked", "pipec2", "pack",
                  source, "-o", out_plink,
                  *[arg for path in includes for arg in ("-I", path)],
                  *[arg for key, value in defines.items() for arg in ("-D", f"{key}={value}")],
                  *[f"--nvcc-flag={flag}" for flag in nvcc_flags],
                  "--report", report,
                  *(() if profile is None else ("--profile", profile))], log, env=env)
    return json.loads(report.read_text())


def manifest_of(plink: Path) -> dict:
    from pipec2.container import read

    return read(plink)[0]
