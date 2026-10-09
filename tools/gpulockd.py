#!/usr/bin/env python3
"""Hold CUDA primary contexts while enforcing configured SM clock locks.

Exit codes:
  0  The requested operation succeeded (and ``status`` is healthy).
  1  Any other operational failure, unhealthy status, or interruption.
  3  A requested SM clock lock was not honored within the allowed tolerance.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence


CLOCK_TOLERANCE_MHZ = 15
MONITOR_INTERVAL_SECONDS = 5.0
COMMAND_TIMEOUT_SECONDS = 15.0
QUERY_TIMEOUT_SECONDS = 10.0
DAEMON_READY_TIMEOUT_SECONDS = 15.0
STOP_TIMEOUT_SECONDS = 10.0
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_LOCK_NOT_HONORED = 3
SLOWDOWN_REASON_MASK = 0x4 | 0x8 | 0x40 | 0x80
# SW power cap. Some parts (e.g. consumer Blackwell) assert it permanently, even idle at P8;
# there it carries no throttling signal and is exempted per GPU at start (with a warning).
SW_POWER_CAP_REASON = 0x4
IDLE_WAIT_SECONDS = 10.0
IDLE_POLL_SECONDS = 0.5


class GpuLockError(RuntimeError):
    """An expected gpulockd operational failure."""


class LockNotHonoredError(GpuLockError):
    """A requested SM clock lock was applied but its readback did not match."""

    def __init__(
        self,
        gpu: int,
        requested_mhz: int,
        observed_mhz: int,
        tolerance_mhz: int,
        *,
        detail: str | None = None,
    ) -> None:
        self.gpu = gpu
        self.requested_mhz = requested_mhz
        self.observed_mhz = observed_mhz
        self.tolerance_mhz = tolerance_mhz
        message = (
            f"GPU {gpu} clocks.sm readback {observed_mhz} MHz differs from "
            f"requested {requested_mhz} MHz by more than {tolerance_mhz} MHz"
        )
        super().__init__(detail if detail is not None else message)

    def payload(self) -> dict[str, Any]:
        return {
            "error": "lock-not-honored",
            "gpu": self.gpu,
            "requested_mhz": self.requested_mhz,
            "observed_mhz": self.observed_mhz,
            "tolerance_mhz": self.tolerance_mhz,
        }


@dataclass(frozen=True)
class GpuIdentity:
    index: int
    uuid: str
    pci_bus_id: str
    driver_version: str


@dataclass
class HeldGpu:
    identity: GpuIdentity
    sm_mhz: int
    cuda_device: int
    context: ctypes.c_void_p
    policy_token: str
    started_at: str
    violations: int = 0
    exempt_reasons: int = 0


class CudaDriver:
    """Minimal CUDA Driver API binding needed to retain primary contexts."""

    class _Uuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    def __init__(self) -> None:
        try:
            self._lib = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:
            raise GpuLockError(f"cannot load libcuda.so.1: {exc}") from None

        try:
            self._cu_init = self._lib.cuInit
            self._get_by_pci = self._lib.cuDeviceGetByPCIBusId
            self._get_uuid = self._lib.cuDeviceGetUuid_v2
            self._retain = self._lib.cuDevicePrimaryCtxRetain
            self._release = getattr(
                self._lib, "cuDevicePrimaryCtxRelease_v2", None
            )
            if self._release is None:
                self._release = self._lib.cuDevicePrimaryCtxRelease
        except AttributeError as exc:
            raise GpuLockError(f"required CUDA Driver API symbol is missing: {exc}") from None

        self._cu_init.argtypes = [ctypes.c_uint]
        self._cu_init.restype = ctypes.c_int
        self._get_by_pci.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p]
        self._get_by_pci.restype = ctypes.c_int
        self._get_uuid.argtypes = [ctypes.POINTER(self._Uuid), ctypes.c_int]
        self._get_uuid.restype = ctypes.c_int
        self._retain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self._retain.restype = ctypes.c_int
        self._release.argtypes = [ctypes.c_int]
        self._release.restype = ctypes.c_int

        self._error_name = getattr(self._lib, "cuGetErrorName", None)
        self._error_string = getattr(self._lib, "cuGetErrorString", None)
        for function in (self._error_name, self._error_string):
            if function is not None:
                function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
                function.restype = ctypes.c_int

        self._check(self._cu_init(0), "cuInit")

    def _describe_error(self, result: int) -> str:
        details: list[str] = []
        for function in (self._error_name, self._error_string):
            if function is None:
                continue
            text = ctypes.c_char_p()
            if function(result, ctypes.byref(text)) == 0 and text.value:
                details.append(text.value.decode("utf-8", errors="replace"))
        return ": ".join(details) if details else f"CUDA error {result}"

    def _check(self, result: int, operation: str) -> None:
        if result != 0:
            raise GpuLockError(f"{operation} failed: {self._describe_error(result)}")

    def retain(self, identity: GpuIdentity) -> tuple[int, ctypes.c_void_p]:
        device = ctypes.c_int()
        self._check(
            self._get_by_pci(ctypes.byref(device), identity.pci_bus_id.encode("ascii")),
            f"cuDeviceGetByPCIBusId({identity.pci_bus_id})",
        )

        cuda_uuid = self._Uuid()
        self._check(
            self._get_uuid(ctypes.byref(cuda_uuid), device),
            f"cuDeviceGetUuid_v2({identity.pci_bus_id})",
        )
        expected_uuid = _uuid_bytes(identity.uuid)
        actual_uuid = bytes(cuda_uuid.bytes)
        if actual_uuid != expected_uuid:
            raise GpuLockError(
                f"GPU {identity.index} identity mismatch: nvidia-smi UUID "
                f"{identity.uuid} does not match CUDA UUID GPU-{actual_uuid.hex()}"
            )

        context = ctypes.c_void_p()
        self._check(
            self._retain(ctypes.byref(context), device),
            f"cuDevicePrimaryCtxRetain(GPU {identity.index})",
        )
        return device.value, context

    def release(self, device: int, gpu: int) -> None:
        self._check(self._release(device), f"cuDevicePrimaryCtxRelease(GPU {gpu})")


def _run(command: Sequence[str], *, timeout: float, operation: str) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise GpuLockError(f"{operation}: executable not found: {exc.filename}") from None
    except subprocess.TimeoutExpired:
        raise GpuLockError(f"{operation}: command timed out after {timeout:g} seconds") from None
    except OSError as exc:
        raise GpuLockError(f"{operation}: cannot execute command: {exc}") from None

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise GpuLockError(
            f"{operation}: command exited {completed.returncode}: {detail}"
        )
    return completed.stdout


def _nvidia_smi(arguments: Sequence[str], *, operation: str) -> str:
    return _run(
        ["nvidia-smi", *arguments],
        timeout=QUERY_TIMEOUT_SECONDS,
        operation=operation,
    )


def _sudo_nvidia_smi(arguments: Sequence[str], *, operation: str) -> None:
    _run(
        ["sudo", "nvidia-smi", *arguments],
        timeout=COMMAND_TIMEOUT_SECONDS,
        operation=operation,
    )


def _parse_gpu_csv(value: str, *, label: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise GpuLockError(f"{label} must be a non-empty comma-separated GPU index list")
    indices: list[int] = []
    for part in parts:
        try:
            index = int(part, 10)
        except ValueError:
            raise GpuLockError(f"{label} contains a non-integer GPU index: {part!r}") from None
        if index < 0:
            raise GpuLockError(f"{label} contains a negative GPU index: {index}")
        if index in indices:
            raise GpuLockError(f"{label} contains duplicate GPU index {index}")
        indices.append(index)
    return tuple(indices)


def _positive_frequency(value: str, *, label: str) -> int:
    try:
        frequency = int(value, 10)
    except (TypeError, ValueError):
        raise GpuLockError(f"{label} must be an integer MHz value") from None
    if frequency <= 0:
        raise GpuLockError(f"{label} must be greater than zero")
    return frequency


def _uuid_bytes(uuid: str) -> bytes:
    value = uuid.strip()
    if value[:4].lower() != "gpu-":
        raise GpuLockError(f"unsupported nvidia-smi GPU UUID format: {uuid!r}")
    compact = value[4:].replace("-", "")
    if len(compact) != 32 or not re.fullmatch(r"[0-9A-Fa-f]{32}", compact):
        raise GpuLockError(f"malformed nvidia-smi GPU UUID: {uuid!r}")
    return bytes.fromhex(compact)


def _query_identities() -> dict[int, GpuIdentity]:
    output = _nvidia_smi(
        [
            "--query-gpu=index,uuid,pci.bus_id,driver_version",
            "--format=csv,noheader",
        ],
        operation="query GPU identities",
    )
    identities: dict[int, GpuIdentity] = {}
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 4:
            raise GpuLockError(
                f"query GPU identities: malformed row {line_number}: {raw_line!r}"
            )
        try:
            index = int(fields[0], 10)
        except ValueError:
            raise GpuLockError(
                f"query GPU identities: invalid index in row {line_number}: {fields[0]!r}"
            ) from None
        _uuid_bytes(fields[1])
        if index in identities:
            raise GpuLockError(f"query GPU identities: duplicate index {index}")
        identities[index] = GpuIdentity(index, fields[1], fields[2], fields[3])
    if not identities:
        raise GpuLockError("query GPU identities returned no GPUs")
    return identities


def _parse_numeric_field(value: str, *, label: str) -> int:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if match is None:
        raise GpuLockError(f"{label}: expected a numeric value, got {value!r}")
    return int(round(float(match.group(0))))


def _read_clock(gpu: int) -> int:
    output = _nvidia_smi(
        [
            "-i",
            str(gpu),
            "--query-gpu=clocks.sm",
            "--format=csv,noheader,nounits",
        ],
        operation=f"read GPU {gpu} SM clock",
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise GpuLockError(
            f"read GPU {gpu} SM clock: expected one row, received {len(rows)}"
        )
    return _parse_numeric_field(rows[0], label=f"GPU {gpu} clocks.sm")


def _verify_clock(gpu: int, target_mhz: int) -> int:
    last_error: GpuLockError | None = None
    for attempt in range(3):
        try:
            actual_mhz = _read_clock(gpu)
            if abs(actual_mhz - target_mhz) <= CLOCK_TOLERANCE_MHZ:
                return actual_mhz
            last_error = LockNotHonoredError(
                gpu,
                target_mhz,
                actual_mhz,
                CLOCK_TOLERANCE_MHZ,
            )
        except GpuLockError as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(1.0)
    assert last_error is not None
    raise last_error


def _set_clock(gpu: int, sm_mhz: int) -> None:
    _sudo_nvidia_smi(
        ["-i", str(gpu), "-lgc", f"{sm_mhz},{sm_mhz}"],
        operation=f"lock GPU {gpu} SM clock",
    )


def _reset_clock(gpu: int) -> None:
    _sudo_nvidia_smi(
        ["-i", str(gpu), "-rgc"],
        operation=f"reset GPU {gpu} SM clock",
    )


def _policy_token(identity: GpuIdentity, sm_mhz: int) -> str:
    payload = f"{identity.uuid}|{sm_mhz}|{identity.driver_version}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _marker_path(state_dir: Path, gpu: int) -> Path:
    return state_dir / f"gpu{gpu}.json"


def _marker_payload(held: HeldGpu, pid: int) -> dict[str, Any]:
    return {
        "gpu": held.identity.index,
        "uuid": held.identity.uuid,
        "sm_mhz": held.sm_mhz,
        "pid": pid,
        "started_at": held.started_at,
        "driver_version": held.identity.driver_version,
        "policy_token": held.policy_token,
        "violations": held.violations,
        "exempt_reasons": held.exempt_reasons,
    }


def _load_marker(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            marker = json.load(stream)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuLockError(f"cannot read marker {path}: {exc}") from None
    if not isinstance(marker, dict):
        raise GpuLockError(f"marker {path} is not a JSON object")

    integer_fields = ("gpu", "sm_mhz", "pid", "violations", "exempt_reasons")
    string_fields = ("uuid", "started_at", "driver_version", "policy_token")
    for field in integer_fields:
        if not isinstance(marker.get(field), int) or isinstance(marker.get(field), bool):
            raise GpuLockError(f"marker {path} has invalid {field!r}")
    for field in string_fields:
        if not isinstance(marker.get(field), str) or not marker[field]:
            raise GpuLockError(f"marker {path} has invalid {field!r}")
    if marker["gpu"] < 0 or marker["sm_mhz"] <= 0 or marker["pid"] <= 0:
        raise GpuLockError(f"marker {path} contains an out-of-range numeric field")
    if marker["violations"] < 0 or marker["exempt_reasons"] < 0:
        raise GpuLockError(f"marker {path} has a negative violations or exempt_reasons field")
    _uuid_bytes(marker["uuid"])
    if not re.fullmatch(r"[0-9a-f]{64}", marker["policy_token"]):
        raise GpuLockError(f"marker {path} has an invalid policy_token")
    expected_name = f"gpu{marker['gpu']}.json"
    if path.name != expected_name:
        raise GpuLockError(
            f"marker {path} names GPU {marker['gpu']}; expected filename {expected_name}"
        )
    return marker


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise GpuLockError(f"cannot inspect pid {pid}: {exc}") from None
    return True


def _discover_marker_paths(state_dir: Path) -> list[Path]:
    try:
        return sorted(state_dir.glob("gpu*.json"), key=lambda path: path.name)
    except OSError as exc:
        raise GpuLockError(f"cannot list marker directory {state_dir}: {exc}") from None


class _StateOperationLock:
    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._descriptor: int | None = None

    def __enter__(self) -> _StateOperationLock:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        path = self._state_dir / ".operation.lock"
        try:
            self._descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
            raise GpuLockError("another gpulockd state-changing operation is in progress") from None
        except OSError as exc:
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
            raise GpuLockError(f"cannot lock state directory {self._state_dir}: {exc}") from None
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self._descriptor is not None:
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._descriptor)
                self._descriptor = None


class LockManager:
    def __init__(self, state_dir: Path, identities: dict[int, GpuIdentity], sm_mhz: int,
                 exempt_reasons: dict[int, int] | None = None) -> None:
        self.state_dir = state_dir
        self.identities = identities
        self.sm_mhz = sm_mhz
        self.exempt_reasons = exempt_reasons or {}
        self.cuda: CudaDriver | None = None
        self.held: list[HeldGpu] = []
        self.stop_event = threading.Event()
        self.owner_pid = os.getpid()
        self._cleaned = False

    def acquire_all(self, gpus: Iterable[int]) -> None:
        if self.cuda is not None or self.held:
            raise GpuLockError("GPU locks have already been acquired")
        self.cuda = CudaDriver()
        for gpu in gpus:
            if self.stop_event.is_set():
                raise GpuLockError("start interrupted while acquiring GPU locks")
            self.held.append(self._acquire_one(self.identities[gpu]))

    def _acquire_one(self, identity: GpuIdentity) -> HeldGpu:
        if self.cuda is None:
            raise GpuLockError("CUDA driver is not initialized")
        cuda = self.cuda
        clock_touched = False
        retained_device: int | None = None
        held: HeldGpu | None = None
        try:
            retained_device, context = cuda.retain(identity)
            clock_touched = True
            _set_clock(identity.index, self.sm_mhz)
            _verify_clock(identity.index, self.sm_mhz)
            held = HeldGpu(
                identity=identity,
                sm_mhz=self.sm_mhz,
                cuda_device=retained_device,
                context=context,
                policy_token=_policy_token(identity, self.sm_mhz),
                started_at=datetime.now(timezone.utc).isoformat(),
                exempt_reasons=self.exempt_reasons.get(identity.index, 0),
            )
            _atomic_write_json(
                _marker_path(self.state_dir, identity.index),
                _marker_payload(held, self.owner_pid),
            )
            return held
        except BaseException:
            errors: list[str] = []
            if clock_touched:
                try:
                    _reset_clock(identity.index)
                except GpuLockError as exc:
                    errors.append(str(exc))
            if retained_device is not None:
                try:
                    cuda.release(retained_device, identity.index)
                except GpuLockError as exc:
                    errors.append(str(exc))
            if held is not None:
                try:
                    _marker_path(self.state_dir, identity.index).unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    errors.append(f"cannot remove GPU {identity.index} marker: {exc}")
            if errors:
                print("gpulockd rollback: " + "; ".join(errors), file=sys.stderr)
            raise

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_event.set()

    def monitor(self) -> None:
        while not self.stop_event.wait(MONITOR_INTERVAL_SECONDS):
            try:
                self._sample()
            except GpuLockError as exc:
                print(f"gpulockd monitor: {exc}", file=sys.stderr)

    def _sample(self) -> None:
        gpu_arg = ",".join(str(held.identity.index) for held in self.held)
        output = _nvidia_smi(
            [
                "-i",
                gpu_arg,
                "--query-gpu=index,clocks.sm,clocks_event_reasons.active",
                "--format=csv,noheader",
            ],
            operation="sample locked GPU state",
        )
        samples: dict[int, tuple[int, int]] = {}
        for line_number, raw_line in enumerate(output.splitlines(), start=1):
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.split(",")]
            if len(fields) != 3:
                raise GpuLockError(
                    f"sample locked GPU state: malformed row {line_number}: {raw_line!r}"
                )
            try:
                gpu = int(fields[0], 10)
                reason_mask = int(fields[2], 0)
            except ValueError:
                raise GpuLockError(
                    f"sample locked GPU state: invalid index or reason mask in row "
                    f"{line_number}: {raw_line!r}"
                ) from None
            clock_mhz = _parse_numeric_field(fields[1], label=f"GPU {gpu} clocks.sm")
            samples[gpu] = (clock_mhz, reason_mask)

        held_by_gpu = {held.identity.index: held for held in self.held}
        missing = set(held_by_gpu).difference(samples)
        unexpected = set(samples).difference(held_by_gpu)
        if missing or unexpected:
            raise GpuLockError(
                "sample locked GPU state returned an unexpected GPU set "
                f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})"
            )

        deviating = [
            gpu
            for gpu, (clock_mhz, _reason_mask) in samples.items()
            if abs(clock_mhz - held_by_gpu[gpu].sm_mhz) > CLOCK_TOLERANCE_MHZ
        ]
        loaded = self._query_loaded_gpus(deviating) if deviating else set()
        for gpu, (clock_mhz, reason_mask) in samples.items():
            held = held_by_gpu[gpu]
            reason_violation = bool(reason_mask & SLOWDOWN_REASON_MASK & ~held.exempt_reasons)
            clock_violation = (
                gpu in loaded
                and abs(clock_mhz - held.sm_mhz) > CLOCK_TOLERANCE_MHZ
            )
            if reason_violation or clock_violation:
                held.violations += 1
                _atomic_write_json(
                    _marker_path(self.state_dir, gpu),
                    _marker_payload(held, self.owner_pid),
                )

    def _query_loaded_gpus(self, gpus: Sequence[int]) -> set[int]:
        gpu_arg = ",".join(str(gpu) for gpu in gpus)
        output = _nvidia_smi(
            [
                "-i",
                gpu_arg,
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            operation="query GPU load for clock deviation",
        )
        loaded: set[int] = set()
        seen: set[int] = set()
        for line_number, raw_line in enumerate(output.splitlines(), start=1):
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.split(",")]
            if len(fields) != 2:
                raise GpuLockError(
                    f"query GPU load: malformed row {line_number}: {raw_line!r}"
                )
            try:
                gpu = int(fields[0], 10)
            except ValueError:
                raise GpuLockError(
                    f"query GPU load: invalid index in row {line_number}: {fields[0]!r}"
                ) from None
            utilization = _parse_numeric_field(
                fields[1], label=f"GPU {gpu} utilization.gpu"
            )
            seen.add(gpu)
            if utilization > 0:
                loaded.add(gpu)
        expected = set(gpus)
        if seen != expected:
            raise GpuLockError(
                "query GPU load returned an unexpected GPU set "
                f"(missing={sorted(expected - seen)}, unexpected={sorted(seen - expected)})"
            )
        return loaded

    def cleanup(self) -> list[str]:
        if self._cleaned or os.getpid() != self.owner_pid:
            return []
        self._cleaned = True
        errors: list[str] = []
        for held in reversed(self.held):
            gpu = held.identity.index
            reset_ok = False
            try:
                _reset_clock(gpu)
                reset_ok = True
            except GpuLockError as exc:
                errors.append(str(exc))
            try:
                if self.cuda is None:
                    raise GpuLockError("CUDA driver is not initialized")
                self.cuda.release(held.cuda_device, gpu)
            except GpuLockError as exc:
                errors.append(str(exc))
            if reset_ok:
                path = _marker_path(self.state_dir, gpu)
                try:
                    current = _load_marker(path)
                    if (
                        current["pid"] == self.owner_pid
                        and current["policy_token"] == held.policy_token
                    ):
                        path.unlink()
                except FileNotFoundError:
                    pass
                except (GpuLockError, OSError) as exc:
                    errors.append(f"cannot remove GPU {gpu} marker: {exc}")
        return errors


def _load_config() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        import pipelink_env
    except ImportError as exc:
        raise GpuLockError(f"cannot import tools/pipelink_env.py: {exc}") from None
    try:
        return pipelink_env.load(repo_root=repo_root)
    except Exception as exc:
        raise GpuLockError(f"cannot load PipeLink environment: {exc}") from None


def _configured_state(config: dict[str, str]) -> tuple[tuple[int, ...], Path]:
    allowed_text = config["PIPELINK_LOCKED_GPUS"]
    if not allowed_text:
        raise GpuLockError("PIPELINK_LOCKED_GPUS must be a non-empty comma-separated GPU index list")
    allowed = tuple(map(int, allowed_text.split(",")))
    state_text = config["PIPELINK_LOCK_STATE_DIR"].strip()
    if not state_text:
        raise GpuLockError("PIPELINK_LOCK_STATE_DIR is empty")
    return allowed, Path(state_text).expanduser()


def _select_gpus(requested: str | None, allowed: tuple[int, ...]) -> tuple[int, ...]:
    selected = (
        _parse_gpu_csv(requested, label="--gpus") if requested is not None else allowed
    )
    unauthorized = sorted(set(selected).difference(allowed))
    if unauthorized:
        raise GpuLockError(
            f"refusing unauthorized GPU indices {unauthorized}; "
            f"PIPELINK_LOCKED_GPUS permits {list(allowed)}"
        )
    return selected


def _ensure_startable(state_dir: Path, gpus: Iterable[int]) -> None:
    for gpu in gpus:
        path = _marker_path(state_dir, gpu)
        try:
            marker = _load_marker(path)
        except FileNotFoundError:
            continue
        except GpuLockError as exc:
            raise GpuLockError(f"GPU {gpu} has an unusable existing marker: {exc}") from None
        state = "live" if _pid_alive(marker["pid"]) else "stale"
        raise GpuLockError(
            f"GPU {gpu} already has a {state} marker at {path}; run gpulockd.py stop first"
        )


def _redirect_daemon_stdio(state_dir: Path) -> None:
    input_fd = os.open(os.devnull, os.O_RDONLY)
    try:
        log_fd = os.open(
            state_dir / "gpulockd.log",
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
    except BaseException:
        os.close(input_fd)
        raise
    try:
        os.dup2(input_fd, sys.stdin.fileno())
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())
    finally:
        if input_fd > sys.stderr.fileno():
            os.close(input_fd)
        if log_fd > sys.stderr.fileno():
            os.close(log_fd)


def _daemonize(manager: LockManager, gpus: Sequence[int]) -> bool:
    read_fd, write_fd = os.pipe()
    try:
        first_pid = os.fork()
    except OSError as exc:
        os.close(read_fd)
        os.close(write_fd)
        raise GpuLockError(f"cannot fork daemon: {exc}") from None

    if first_pid > 0:
        os.close(write_fd)
        ready_timeout = DAEMON_READY_TIMEOUT_SECONDS + len(gpus) * (
            2 * COMMAND_TIMEOUT_SECONDS + 2 * QUERY_TIMEOUT_SECONDS + 2.0
        )
        try:
            with os.fdopen(read_fd, "r", encoding="utf-8") as pipe:
                readable, _, _ = select.select([read_fd], [], [], ready_timeout)
                message = pipe.read() if readable else ""
        finally:
            try:
                os.waitpid(first_pid, 0)
            except ChildProcessError:
                pass
        if not message:
            raise GpuLockError(
                "cannot start daemon: daemon did not report readiness before the timeout"
            )
        try:
            response = json.loads(message)
        except json.JSONDecodeError as exc:
            raise GpuLockError(f"cannot start daemon: invalid startup response: {exc}") from None
        if not response:
            return True
        if response.get("error") == "lock-not-honored":
            raise LockNotHonoredError(
                response["gpu"],
                response["requested_mhz"],
                response["observed_mhz"],
                response["tolerance_mhz"],
                detail=response["detail"],
            )
        raise GpuLockError(f"cannot start daemon: ERROR {response['detail']}")

    os.close(read_fd)
    response: dict[str, Any] = {}
    exit_code = EXIT_SUCCESS
    try:
        os.setsid()
        if os.fork() > 0:
            os._exit(0)
        os.umask(0o027)
        _redirect_daemon_stdio(manager.state_dir)
        manager.owner_pid = os.getpid()
        manager.acquire_all(gpus)
    except BaseException as exc:
        cleanup_errors = manager.cleanup()
        detail = str(exc)
        if cleanup_errors:
            detail += "; cleanup: " + "; ".join(cleanup_errors)
        print(f"gpulockd startup: {detail}", file=sys.stderr, flush=True)
        if isinstance(exc, LockNotHonoredError):
            response = exc.payload()
            exit_code = EXIT_LOCK_NOT_HONORED
        else:
            exit_code = EXIT_ERROR
        response["detail"] = detail
    try:
        with os.fdopen(write_fd, "w", encoding="utf-8") as pipe:
            json.dump(response, pipe, separators=(",", ":"))
    except OSError:
        _atexit_cleanup(manager)
        os._exit(EXIT_ERROR)
    if exit_code != EXIT_SUCCESS:
        os._exit(exit_code)
    return False


def _idle_reason_exemptions(gpus: Sequence[int]) -> dict[int, int]:
    """Exempt SW power cap on GPUs that assert it while idle, before their clocks are locked.

    A just-finished workload settles to P8 within seconds, so each GPU gets up to IDLE_WAIT_SECONDS to read idle;
    a GPU that never does keeps the full violation mask.
    """
    exemptions: dict[int, int] = {}
    pending = list(gpus)
    deadline = time.monotonic() + IDLE_WAIT_SECONDS
    while pending:
        output = _nvidia_smi(
            [
                "-i",
                ",".join(str(gpu) for gpu in pending),
                "--query-gpu=index,utilization.gpu,pstate,clocks_event_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            operation="sample idle clock event reasons",
        )
        for raw_line in output.splitlines():
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.split(",")]
            if len(fields) != 4:
                raise GpuLockError(f"sample idle clock event reasons: malformed row {raw_line!r}")
            gpu = int(fields[0], 10)
            # Idle means no work and the lowest performance state; right after a workload the
            # GPU reads 0% at P0/P5 for a few seconds before settling.
            if _parse_numeric_field(fields[1], label=f"GPU {gpu} utilization.gpu") or fields[2] != "P8":
                continue
            pending.remove(gpu)
            if int(fields[3], 0) & SW_POWER_CAP_REASON:
                exemptions[gpu] = SW_POWER_CAP_REASON
                print(
                    f"gpulockd warning: GPU {gpu} asserts SW power cap (0x4) while idle; it is not "
                    "counted as a lock violation (clock deviation under load still is)",
                    file=sys.stderr,
                )
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(IDLE_POLL_SECONDS)
    return exemptions


def _start(args: argparse.Namespace) -> int:
    config = _load_config()
    allowed, state_dir = _configured_state(config)
    gpus = _select_gpus(args.gpus, allowed)
    configured_frequency = config.get("PIPELINK_LOCK_SM_MHZ", "").strip()
    if args.sm_mhz is None and not configured_frequency:
        raise GpuLockError(
            "SM lock frequency is required: pass --sm-mhz or set PIPELINK_LOCK_SM_MHZ"
        )
    sm_mhz = (
        args.sm_mhz
        if args.sm_mhz is not None
        else _positive_frequency(configured_frequency, label="PIPELINK_LOCK_SM_MHZ")
    )

    identities = _query_identities()
    missing = [gpu for gpu in gpus if gpu not in identities]
    if missing:
        raise GpuLockError(f"configured GPU indices are not visible to nvidia-smi: {missing}")

    manager = LockManager(state_dir, identities, sm_mhz, _idle_reason_exemptions(gpus))
    atexit.register(_atexit_cleanup, manager)
    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    for sig in previous_handlers:
        signal.signal(sig, manager.request_stop)

    try:
        with _StateOperationLock(state_dir):
            _ensure_startable(state_dir, gpus)
            if args.foreground:
                manager.acquire_all(gpus)
                parent_process = False
            else:
                parent_process = _daemonize(manager, gpus)
        if args.foreground:
            print(
                f"gpulockd running in foreground as pid {os.getpid()} for GPUs "
                + ",".join(str(gpu) for gpu in gpus)
            )
        elif parent_process:
            print(
                "gpulockd started for GPUs "
                + ",".join(str(gpu) for gpu in gpus)
                + f" at {sm_mhz} MHz"
            )
            return 0
        manager.monitor()
        return 0
    finally:
        errors = manager.cleanup()
        if errors:
            print("gpulockd cleanup: " + "; ".join(errors), file=sys.stderr)
        if os.getpid() == manager.owner_pid:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def _probe_one(
    cuda: CudaDriver, identity: GpuIdentity, sm_mhz: int
) -> dict[str, Any]:
    retained_device: int | None = None
    clock_touched = False
    result: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    try:
        retained_device, _context = cuda.retain(identity)
        clock_touched = True
        _set_clock(identity.index, sm_mhz)
        try:
            observed_mhz = _verify_clock(identity.index, sm_mhz)
            honored = True
        except LockNotHonoredError as exc:
            observed_mhz = exc.observed_mhz
            honored = False
        result = {
            "gpu": identity.index,
            "requested_mhz": sm_mhz,
            "observed_mhz": observed_mhz,
            "tolerance_mhz": CLOCK_TOLERANCE_MHZ,
            "honored": honored,
        }
    except BaseException as exc:
        primary_error = exc

    cleanup_errors: list[str] = []
    if clock_touched:
        try:
            _reset_clock(identity.index)
        except GpuLockError as exc:
            cleanup_errors.append(str(exc))
    if retained_device is not None:
        try:
            cuda.release(retained_device, identity.index)
        except GpuLockError as exc:
            cleanup_errors.append(str(exc))

    if primary_error is not None:
        if cleanup_errors:
            print(
                "gpulockd probe rollback: " + "; ".join(cleanup_errors),
                file=sys.stderr,
            )
        raise primary_error
    if cleanup_errors:
        raise GpuLockError("probe cleanup: " + "; ".join(cleanup_errors))
    assert result is not None
    return result


def _probe(args: argparse.Namespace) -> int:
    config = _load_config()
    allowed, state_dir = _configured_state(config)
    gpus = _select_gpus(args.gpus, allowed)
    identities = _query_identities()
    missing = [gpu for gpu in gpus if gpu not in identities]
    if missing:
        raise GpuLockError(f"configured GPU indices are not visible to nvidia-smi: {missing}")

    with _StateOperationLock(state_dir):
        _ensure_startable(state_dir, gpus)
        cuda = CudaDriver()
        results = [_probe_one(cuda, identities[gpu], args.sm_mhz) for gpu in gpus]
    print(json.dumps({"gpus": results}, separators=(",", ":")))
    return EXIT_SUCCESS


def _atexit_cleanup(manager: LockManager) -> None:
    errors = manager.cleanup()
    if errors:
        print("gpulockd cleanup: " + "; ".join(errors), file=sys.stderr)


def _selected_marker_paths(
    state_dir: Path, requested: str | None, allowed: tuple[int, ...]
) -> list[Path]:
    selected = _select_gpus(requested, allowed) if requested is not None else None
    paths = _discover_marker_paths(state_dir)
    if selected is None:
        return paths
    selected_set = set(selected)
    return [path for path in paths if _gpu_from_marker_filename(path) in selected_set]


def _gpu_from_marker_filename(path: Path) -> int:
    match = re.fullmatch(r"gpu(\d+)\.json", path.name)
    if match is None:
        raise GpuLockError(f"unexpected marker filename: {path}")
    return int(match.group(1), 10)


def _read_authorized_markers(
    paths: Iterable[Path], allowed: tuple[int, ...]
) -> list[tuple[Path, dict[str, Any]]]:
    allowed_set = set(allowed)
    markers: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        marker = _load_marker(path)
        if marker["gpu"] not in allowed_set:
            raise GpuLockError(
                f"refusing marker for unauthorized GPU {marker['gpu']}: {path}"
            )
        markers.append((path, marker))
    return markers


def _stop(args: argparse.Namespace) -> int:
    config = _load_config()
    allowed, state_dir = _configured_state(config)
    with _StateOperationLock(state_dir):
        paths = _selected_marker_paths(state_dir, args.gpus, allowed)
        if not paths:
            raise GpuLockError("no gpulockd markers found")
        markers = _read_authorized_markers(paths, allowed)
        stale_pids: set[int] = set()
        had_error = False
        for pid in sorted({marker["pid"] for _path, marker in markers}):
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"sent SIGTERM to gpulockd pid {pid}")
            except ProcessLookupError:
                stale_pids.add(pid)
            except OSError as exc:
                print(f"cannot signal gpulockd pid {pid}: {exc}", file=sys.stderr)
                had_error = True

        deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
        while markers:
            pending = []
            for path, marker in markers:
                gpu = marker["gpu"]
                if marker["pid"] in stale_pids:
                    try:
                        _reset_clock(gpu)
                        path.unlink(missing_ok=True)
                        print(f"GPU {gpu}: stale marker reset and removed")
                    except (GpuLockError, OSError) as exc:
                        print(f"GPU {gpu}: stale cleanup failed: {exc}", file=sys.stderr)
                        had_error = True
                elif path.exists():
                    pending.append((path, marker))
            markers = pending
            if not markers or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        if markers:
            print(
                "timed out waiting for markers to disappear: "
                + ", ".join(str(path) for path, _marker in markers),
                file=sys.stderr,
            )
            had_error = True
    return EXIT_ERROR if had_error else EXIT_SUCCESS


def _status(args: argparse.Namespace) -> int:
    config = _load_config()
    allowed, state_dir = _configured_state(config)
    paths = _selected_marker_paths(state_dir, args.gpus, allowed)
    print("gpu  sm_mhz  pid       alive  clocks.sm  violations  healthy")
    if not paths:
        print("-    -       -         no     -          -           no")
        return 1

    healthy = True
    try:
        markers = _read_authorized_markers(paths, allowed)
    except GpuLockError as exc:
        print(f"status error: {exc}", file=sys.stderr)
        return 1
    for _path, marker in sorted(markers, key=lambda item: item[1]["gpu"]):
        gpu = marker["gpu"]
        alive = _pid_alive(marker["pid"])
        try:
            current_clock: int | None = _read_clock(gpu)
            clock_text = str(current_clock)
        except GpuLockError as exc:
            current_clock = None
            clock_text = "error"
            print(f"GPU {gpu}: {exc}", file=sys.stderr)
        row_healthy = (
            alive
            and current_clock is not None
            and abs(current_clock - marker["sm_mhz"]) <= CLOCK_TOLERANCE_MHZ
            and marker["violations"] == 0
        )
        healthy = healthy and row_healthy
        if marker["exempt_reasons"]:
            print(f"gpulockd warning: GPU {gpu} clock event reasons {marker['exempt_reasons']:#x} "
                  "were asserted while idle and are not counted as violations", file=sys.stderr)
        print(
            f"{gpu:<4} {marker['sm_mhz']:<7} {marker['pid']:<9} "
            f"{'yes' if alive else 'no':<6} {clock_text:<10} "
            f"{marker['violations']:<11} {'yes' if row_healthy else 'no'}"
        )
    return 0 if healthy else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lock configured GPU SM clocks while retaining CUDA primary contexts. "
            "Machine-specific policy is loaded from the PipeLink environment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0  requested operation succeeded (and status is healthy)\n"
            "  1  any other failure, unhealthy status, or interruption\n"
            "  3  requested SM clock lock was not honored within tolerance"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser(
        "start", help="lock GPUs and run the context-holding monitor daemon"
    )
    start.add_argument(
        "--gpus",
        metavar="INDEXES",
        help="comma-separated GPU indices (default: PIPELINK_LOCKED_GPUS)",
    )
    start.add_argument(
        "--sm-mhz",
        metavar="MHZ",
        type=lambda value: _positive_frequency(value, label="--sm-mhz"),
        help="SM lock frequency (default: PIPELINK_LOCK_SM_MHZ)",
    )
    start.add_argument(
        "--foreground",
        action="store_true",
        help="do not daemonize; monitor until SIGTERM or SIGINT",
    )
    start.set_defaults(handler=_start)
    probe = subparsers.add_parser(
        "probe",
        help="temporarily lock, sample, and unlock authorized GPUs; report JSON",
    )
    probe.add_argument(
        "--sm-mhz",
        metavar="F",
        required=True,
        type=lambda value: _positive_frequency(value, label="--sm-mhz"),
        help="SM frequency to probe",
    )
    probe.add_argument(
        "--gpus",
        metavar="INDEXES",
        help="comma-separated GPU indices (default: PIPELINK_LOCKED_GPUS)",
    )
    probe.set_defaults(handler=_probe)


    stop = subparsers.add_parser(
        "stop", help="signal live daemons and clean stale configured markers"
    )
    stop.add_argument(
        "--gpus", metavar="INDEXES", help="only stop markers for these configured GPUs"
    )
    stop.set_defaults(handler=_stop)

    status = subparsers.add_parser(
        "status", help="show marker, process, clock, and violation health"
    )
    status.add_argument(
        "--gpus", metavar="INDEXES", help="only show markers for these configured GPUs"
    )
    status.set_defaults(handler=_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.handler(args))
    except LockNotHonoredError as exc:
        print(json.dumps(exc.payload(), separators=(",", ":")), flush=True)
        print(f"gpulockd: {exc}", file=sys.stderr, flush=True)
        return EXIT_LOCK_NOT_HONORED
    except (GpuLockError, OSError) as exc:
        print(f"gpulockd: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("gpulockd: interrupted", file=sys.stderr)
        return EXIT_ERROR
    except SystemExit as exc:
        return EXIT_SUCCESS if exc.code == 0 else EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
