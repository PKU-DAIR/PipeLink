"""Configured GPU discovery, assignment, locking, and objective failover."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable

from .constants import (
    ABSENT_PROBE_MARKERS,
    DRIVER_FAILOVER_STATUSES,
    FAILOVER_PROBE_INTERVAL_NS,
    GPU_MEMORY_ELIGIBILITY_MIB,
    MAX_CONCURRENT_PHYSICAL_GPUS,
    RUNTIME_FAILOVER_STATUSES,
    SCHEMA_VERSIONS,
)
from .gpu_qualifier import QUALIFIER_SCHEMA
from .util import utc_now, write_json_atomic
from tools.pipelink_env import load

_ENV = load(Path(__file__).resolve().parents[3])
_CONFIGURED_GPUS = tuple(
    int(item) for item in _ENV["PIPELINK_GPUS"].split(",") if item
)
_LOCK_STATE_DIR = Path(_ENV["PIPELINK_LOCK_STATE_DIR"]).expanduser()
_LOCK_SM_MHZ = (
    int(_ENV["PIPELINK_LOCK_SM_MHZ"])
    if _ENV["PIPELINK_LOCK_SM_MHZ"]
    else None
)
_ALLOW_UNLOCKED = _ENV["PIPELINK_ALLOW_UNLOCKED"] == "1"
_LOCKED_CLOCK_WINDOW_FRACTION = 0.005
_LOCKED_SLOWDOWN_REASON_MASK = 0x4 | 0x8 | 0x40 | 0x80
_LOCKED_QUALIFIER_SCHEMA = "gpulockd-policy-v1"
_LOCK_ROOT = Path(_ENV["PIPELINK_SCRATCH_ROOT"]) / "parity-gpu-locks"

SnapshotProvider = Callable[[int | None], list[dict[str, Any]]]


def _query_snapshot(index: int | None = None) -> list[dict[str, Any]]:
    command = ["nvidia-smi"]
    if index is not None:
        command += ["-i", str(index)]
    command += [
        "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,"
        "utilization.gpu,clocks.sm,clocks_event_reasons.active,"
        "temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError(f"GPU discovery failed: {completed.stderr.strip()}")
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 11:
            raise RuntimeError(f"malformed GPU discovery row: {line!r}")
        try:
            temperature_c: int | None = int(fields[8])
        except ValueError:
            temperature_c = None
        try:
            power_draw_w: float | None = float(fields[9])
        except ValueError:
            power_draw_w = None
        try:
            power_limit_w: float | None = float(fields[10])
        except ValueError:
            power_limit_w = None
        rows.append({
            "physical_index": int(fields[0]),
            "uuid": fields[1],
            "pci_bus_id": fields[2],
            "memory_used_mib": int(fields[3]),
            "memory_total_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
            "sm_clock_mhz": int(fields[6]),
            "reason_mask": int(fields[7], 0),
            "temperature_c": temperature_c,
            "power_draw_w": power_draw_w,
            "power_limit_w": power_limit_w,
        })
    return rows


def _ineligible_reasons(row: dict[str, Any]) -> list[str]:
    reasons = []
    if row["physical_index"] not in _CONFIGURED_GPUS:
        reasons.append("physical-index-not-configured")
    if row["utilization_percent"] != 0:
        reasons.append("utilization-not-zero")
    if row["memory_used_mib"] >= GPU_MEMORY_ELIGIBILITY_MIB:
        reasons.append("memory-used-not-below-1024-MiB")
    return reasons


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno == errno.EPERM:
            return True
        raise RuntimeError(f"cannot inspect gpulockd pid {pid}: {error}") from None
    return True


def _valid_gpu_uuid(value: object) -> bool:
    if not isinstance(value, str) or value[:4].lower() != "gpu-":
        return False
    compact = value[4:].replace("-", "")
    return len(compact) == 32 and re.fullmatch(r"[0-9A-Fa-f]{32}", compact) is not None


def _lock_marker_state(
    state_dir: Path,
    gpu: int,
    expected_uuid: str | None,
) -> dict[str, Any]:
    path = state_dir / f"gpu{gpu}.json"
    state: dict[str, Any] = {
        "path": str(path),
        "present": False,
        "valid": False,
        "live": False,
        "uuid_matches_snapshot": False,
    }
    try:
        with path.open("r", encoding="utf-8") as stream:
            marker = json.load(stream)
    except FileNotFoundError:
        state["validation_reasons"] = ["marker-missing"]
        return state
    except (OSError, json.JSONDecodeError) as error:
        state.update({
            "present": True,
            "validation_reasons": [f"marker-read-failure:{error}"],
        })
        return state
    state["present"] = True
    state["marker"] = marker
    if not isinstance(marker, dict):
        state["validation_reasons"] = ["marker-not-json-object"]
        return state
    reasons = []
    for field in ("gpu", "sm_mhz", "pid", "violations"):
        if not isinstance(marker.get(field), int) or isinstance(marker.get(field), bool):
            reasons.append(f"marker-invalid-{field}")
    for field in ("uuid", "started_at", "driver_version", "policy_token"):
        if not isinstance(marker.get(field), str) or not marker[field]:
            reasons.append(f"marker-invalid-{field}")
    marker_gpu = marker.get("gpu")
    marker_sm_mhz = marker.get("sm_mhz")
    marker_pid = marker.get("pid")
    marker_violations = marker.get("violations")
    if isinstance(marker_gpu, int) and not isinstance(marker_gpu, bool):
        if marker_gpu != gpu:
            reasons.append("marker-gpu-or-filename-mismatch")
    if isinstance(marker_sm_mhz, int) and not isinstance(marker_sm_mhz, bool):
        if marker_sm_mhz <= 0:
            reasons.append("marker-sm-mhz-not-positive")
    if isinstance(marker_pid, int) and not isinstance(marker_pid, bool):
        if marker_pid <= 0:
            reasons.append("marker-pid-not-positive")
        else:
            state["live"] = _pid_alive(marker_pid)
    if isinstance(marker_violations, int) and not isinstance(marker_violations, bool):
        if marker_violations < 0:
            reasons.append("marker-violations-negative")
    if not _valid_gpu_uuid(marker.get("uuid")):
        reasons.append("marker-invalid-uuid")
    policy_token = marker.get("policy_token")
    if (
        not isinstance(policy_token, str)
        or re.fullmatch(r"[0-9a-f]{64}", policy_token) is None
    ):
        reasons.append("marker-invalid-policy-token")
    if (
        not reasons
        and isinstance(marker.get("uuid"), str)
        and isinstance(marker_sm_mhz, int)
        and isinstance(marker.get("driver_version"), str)
    ):
        expected_policy_token = hashlib.sha256(
            (
                f"{marker['uuid']}|{marker_sm_mhz}|"
                f"{marker['driver_version']}"
            ).encode("utf-8")
        ).hexdigest()
        if policy_token != expected_policy_token:
            reasons.append("marker-policy-token-content-mismatch")
    state["uuid_matches_snapshot"] = (
        expected_uuid is not None and marker.get("uuid") == expected_uuid
    )
    state["validation_reasons"] = list(dict.fromkeys(reasons))
    state["valid"] = not state["validation_reasons"]
    return state


def _lock_marker_states(
    initial: list[dict[str, Any]],
    state_dir: Path,
) -> dict[int, dict[str, Any]]:
    uuid_by_index = {
        row["physical_index"]: row.get("uuid")
        for row in initial
        if row.get("physical_index") in _CONFIGURED_GPUS
    }
    return {
        gpu: _lock_marker_state(state_dir, gpu, uuid_by_index.get(gpu))
        for gpu in _CONFIGURED_GPUS
    }


def _locked_mode(
    marker_states: dict[int, dict[str, Any]],
) -> tuple[bool, str]:
    if _ALLOW_UNLOCKED:
        return False, "PIPELINK_ALLOW_UNLOCKED=1 forces unlocked idle discovery"
    matched_live = [
        gpu
        for gpu, state in marker_states.items()
        if state["live"] and state["uuid_matches_snapshot"]
    ]
    if matched_live:
        return (
            True,
            "live gpulockd markers match configured GPU identities and "
            "PIPELINK_ALLOW_UNLOCKED is not 1",
        )
    return False, "no live gpulockd marker matches a configured GPU identity"


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _locked_ineligible_reasons(
    row: dict[str, Any],
    marker_state: dict[str, Any] | None,
) -> list[str]:
    reasons = []
    if row["physical_index"] not in _CONFIGURED_GPUS:
        reasons.append("physical-index-not-configured")
        return reasons
    if marker_state is None or not marker_state["present"]:
        reasons.append("lock-marker-missing")
        return reasons
    if not marker_state["valid"]:
        reasons.append("lock-marker-invalid")
    if not marker_state["live"]:
        reasons.append("lock-marker-pid-not-live")
    marker = marker_state.get("marker")
    if not isinstance(marker, dict):
        return list(dict.fromkeys(reasons))
    if _LOCK_SM_MHZ is None or marker.get("sm_mhz") != _LOCK_SM_MHZ:
        reasons.append("lock-marker-target-mismatch")
    if marker.get("uuid") != row.get("uuid"):
        reasons.append("lock-marker-device-identity-mismatch")
    if marker.get("violations") != 0:
        reasons.append("lock-marker-violations-nonzero")
    clock_mhz = row.get("sm_clock_mhz")
    clock_tolerance_mhz = (
        0.0
        if _LOCK_SM_MHZ is None
        else _LOCK_SM_MHZ * _LOCKED_CLOCK_WINDOW_FRACTION
    )
    if (
        not _finite_number(clock_mhz)
        or _LOCK_SM_MHZ is None
        or abs(float(clock_mhz) - _LOCK_SM_MHZ) > clock_tolerance_mhz
    ):
        reasons.append("sm-clock-not-at-lock")
    reason_mask = row.get("reason_mask")
    # gpulockd exempts reasons a GPU asserted while idle (and warned at start).
    exempt_reasons = marker.get("exempt_reasons", 0)
    if (
        not isinstance(reason_mask, int)
        or isinstance(reason_mask, bool)
        or reason_mask & _LOCKED_SLOWDOWN_REASON_MASK & ~exempt_reasons
    ):
        reasons.append("cap-thermal-or-hw-slowdown-reason-active")
    temperature_c = row.get("temperature_c")
    if (
        not _finite_number(temperature_c)
        or not 0 <= float(temperature_c) <= 150
    ):
        reasons.append("temperature-not-sane")
    power_draw_w = row.get("power_draw_w")
    if not _finite_number(power_draw_w) or float(power_draw_w) < 0:
        reasons.append("power-draw-not-sane")
    power_limit_w = row.get("power_limit_w")
    if not _finite_number(power_limit_w) or float(power_limit_w) <= 0:
        reasons.append("power-limit-not-sane")
    return list(dict.fromkeys(reasons))


def _map_assigned_device(
    value: str,
    initial: list[dict[str, Any]],
    source: str,
) -> int:
    rows = [
        row for row in initial
        if row.get("physical_index") in _CONFIGURED_GPUS
    ]
    if re.fullmatch(r"\d+", value):
        index = int(value, 10)
        if any(row["physical_index"] == index for row in rows):
            return index
        raise RuntimeError(
            f"{source}={value!r} does not name a configured visible physical GPU"
        )
    normalized = value.lower()
    matches = [
        row["physical_index"]
        for row in rows
        if isinstance(row.get("uuid"), str)
        and (
            row["uuid"].lower() == normalized
            or (
                normalized.startswith("gpu-")
                and row["uuid"].lower().startswith(normalized)
            )
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{source}={value!r} cannot be mapped uniquely to a configured "
            "physical GPU by UUID"
        )
    return matches[0]


def _externally_assigned_gpu(
    initial: list[dict[str, Any]],
) -> dict[str, Any] | None:
    explicit = os.environ.get("PIPELINK_ASSIGNED_GPU", "").strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = visible.strip() if visible is not None else ""
    assignments = []
    if explicit:
        if "," in explicit:
            raise RuntimeError("PIPELINK_ASSIGNED_GPU must name exactly one GPU")
        assignments.append((
            "PIPELINK_ASSIGNED_GPU",
            explicit,
            _map_assigned_device(explicit, initial, "PIPELINK_ASSIGNED_GPU"),
        ))
    if visible and "," not in visible:
        assignments.append((
            "CUDA_VISIBLE_DEVICES",
            visible,
            _map_assigned_device(visible, initial, "CUDA_VISIBLE_DEVICES"),
        ))
    if not assignments:
        return None
    indices = {assignment[2] for assignment in assignments}
    if len(indices) != 1:
        raise RuntimeError(
            "PIPELINK_ASSIGNED_GPU and CUDA_VISIBLE_DEVICES select different GPUs"
        )
    return {
        "physical_index": assignments[0][2],
        "sources": [
            {"environment_variable": source, "value": value}
            for source, value, _index in assignments
        ],
        "selection": "external-single-device-assignment",
    }


def ordered_eligible_gpus(eligible: Iterable[int]) -> tuple[int, ...]:
    requested = set(eligible)
    if not requested or any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in requested
    ):
        raise ValueError("eligible GPUs must be integer physical indices")
    unconfigured = requested - set(_CONFIGURED_GPUS)
    if unconfigured:
        raise ValueError(
            f"eligible GPUs are not listed in PIPELINK_GPUS: {sorted(unconfigured)}"
        )
    return tuple(sorted(requested)[:MAX_CONCURRENT_PHYSICAL_GPUS])


def assign_cases(case_ids: Iterable[str], eligible_gpus: Iterable[int]) -> dict[str, int]:
    cases = list(case_ids)
    if not cases or any(not isinstance(case, str) or not case for case in cases):
        raise ValueError("case IDs must be non-empty strings")
    if len(set(cases)) != len(cases):
        raise ValueError("case IDs must be unique")
    devices = ordered_eligible_gpus(eligible_gpus)
    ordered_cases = sorted(cases, key=lambda case: case.encode("utf-8"))
    return {case: devices[index % len(devices)] for index, case in enumerate(ordered_cases)}


class DiscoveryLease:
    def __init__(
        self,
        archive: Path,
        rows: list[dict[str, Any]],
        locks: list[tuple[dict[str, Any], Any, Path]],
        discovery_mode: str,
    ) -> None:
        self.archive = archive
        self.discovery_rows = rows
        self._locks = locks
        self.ordered_devices = tuple(row for row, _handle, _path in locks)
        self.discovery_mode = discovery_mode
        self.closed = False

    def assign(self, case_ids: Iterable[str]) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("GPU discovery lease is closed")
        cases = list(case_ids)
        indices = [row["physical_index"] for row in self.ordered_devices]
        mapping = assign_cases(cases, indices)
        by_index = {row["physical_index"]: row for row in self.ordered_devices}
        record = {
            "schema_version": SCHEMA_VERSIONS["assignment"],
            "utc_time": utc_now(),
            "discovery_record": str((self.archive / "gpu-discovery.json").resolve()),
            "discovery_mode": self.discovery_mode,
            "ordered_eligible_gpus": list(indices),
            "case_order": sorted(cases, key=lambda case: case.encode("utf-8")),
            "assignments": {
                case: {
                    "physical_index": gpu,
                    "uuid": by_index[gpu]["uuid"],
                    "pci_bus_id": by_index[gpu]["pci_bus_id"],
                    **(
                        {"policy_token": by_index[gpu]["policy_token"]}
                        if isinstance(by_index[gpu].get("policy_token"), str)
                        else {}
                    ),
                }
                for case, gpu in mapping.items()
            },
            "predicate": (
                "bytewise-lexicographic cases round-robin over numeric "
                + (
                    "gpulockd-policy-qualified GPUs"
                    if self.discovery_mode == "locked"
                    else "idle-and-telemetry-qualified GPUs"
                )
            ),
        }
        write_json_atomic(self.archive / "assignment.json", record)
        return record

    def close(self) -> None:
        if self.closed:
            return
        for _row, handle, _path in self._locks:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        self.closed = True

    def __enter__(self) -> "DiscoveryLease":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def discover_idle_gpus(
    archive: Path,
    max_devices: int | None = None,
    *,
    snapshot_provider: SnapshotProvider = _query_snapshot,
    qualifier: Callable[[dict[str, Any], str, Path], dict[str, Any]] | None = None,
    lock_root: Path = _LOCK_ROOT,
    lock_state_dir: Path = _LOCK_STATE_DIR,
    epoch_id: str | None = None,
) -> DiscoveryLease:
    """Discover qualified devices while holding exclusive per-GPU leases.

    Live gpulockd policy supplies qualification in locked mode. Unlocked mode
    requires the caller's existing telemetry qualifier seam; the retired pinned
    workload is not a default or fallback. A candidate is never selected without
    a passing qualification result. ``max_devices`` defaults to the configured
    concurrency cap, read at call time.
    """
    if max_devices is None:
        max_devices = MAX_CONCURRENT_PHYSICAL_GPUS
    if (
        not isinstance(max_devices, int)
        or isinstance(max_devices, bool)
        or max_devices <= 0
    ):
        raise ValueError("max_devices must be a positive integer")
    requested_max_devices = max_devices
    selection_notes = []
    if max_devices > MAX_CONCURRENT_PHYSICAL_GPUS:
        max_devices = MAX_CONCURRENT_PHYSICAL_GPUS
        selection_notes.append(
            f"max_devices clamped from {requested_max_devices} to configured "
            f"maximum {MAX_CONCURRENT_PHYSICAL_GPUS}"
        )
    archive = archive.resolve()
    archive.mkdir(parents=True, exist_ok=True)
    discovery_path = archive / "gpu-discovery.json"
    if discovery_path.exists():
        raise ValueError(f"refusing to overwrite discovery record: {discovery_path}")
    epoch_id = epoch_id or utc_now()
    initial = list(snapshot_provider(None))
    marker_states = (
        {
            gpu: {
                "path": str(lock_state_dir / f"gpu{gpu}.json"),
                "inspection": "skipped-because-PIPELINK_ALLOW_UNLOCKED=1",
            }
            for gpu in _CONFIGURED_GPUS
        }
        if _ALLOW_UNLOCKED
        else _lock_marker_states(initial, lock_state_dir)
    )
    use_locked_mode, mode_reason = _locked_mode(marker_states)
    if not use_locked_mode and qualifier is None:
        raise RuntimeError(
            "unlocked discovery requires an explicit telemetry qualifier; "
            "the pinned workload qualifier is retired. Use live gpulockd policy "
            "for current example discovery."
        )
    discovery_mode = "locked" if use_locked_mode else "unlocked-configured"
    external_assignment = _externally_assigned_gpu(initial)
    assigned_index = (
        external_assignment["physical_index"]
        if external_assignment is not None
        else None
    )
    lock_root.mkdir(parents=True, exist_ok=True)
    locked: list[tuple[dict[str, Any], Any, Path]] = []
    handles: list[Any] = []
    decisions: list[dict[str, Any]] = []
    try:
        for row in initial:
            physical_index = row["physical_index"]
            marker_state = marker_states.get(physical_index)
            reasons = (
                _locked_ineligible_reasons(row, marker_state)
                if use_locked_mode
                else _ineligible_reasons(row)
            )
            if assigned_index is not None and physical_index != assigned_index:
                reasons.append("not-externally-assigned-device")
            decision: dict[str, Any] = {
                "discovery_mode": discovery_mode,
                "initial": row,
                "initial_rejection_reasons": list(dict.fromkeys(reasons)),
            }
            if use_locked_mode:
                decision["initial_lock_marker"] = marker_state
            if reasons:
                decisions.append(decision)
                continue
            lock_path = lock_root / f"gpu-{physical_index}.lock"
            handle = lock_path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                decision.update({
                    "lock_acquired": False,
                    "post_lock_rejection_reasons": [
                        "exclusive-m3-lock-unavailable"
                    ],
                })
                decisions.append(decision)
                continue
            handles.append(handle)
            post_rows = snapshot_provider(physical_index)
            post = next(
                (
                    item for item in post_rows
                    if item["physical_index"] == physical_index
                ),
                None,
            )
            post_marker_state = (
                _lock_marker_state(
                    lock_state_dir,
                    physical_index,
                    post.get("uuid") if post is not None else None,
                )
                if use_locked_mode
                else None
            )
            post_reasons = (
                ["post-lock-device-missing"]
                if post is None
                else (
                    _locked_ineligible_reasons(post, post_marker_state)
                    if use_locked_mode
                    else _ineligible_reasons(post)
                )
            )
            if post is not None and (
                post["uuid"] != row["uuid"]
                or post["pci_bus_id"] != row["pci_bus_id"]
            ):
                post_reasons.append("post-lock-device-identity-changed")
            decision.update({
                "lock_acquired": True,
                "lock_path": str(lock_path),
                "post_lock": post,
                "post_lock_rejection_reasons": list(dict.fromkeys(post_reasons)),
            })
            if use_locked_mode:
                decision["post_lock_marker"] = post_marker_state
            if post_reasons:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            else:
                candidate = post
                if use_locked_mode:
                    marker = post_marker_state["marker"]
                    candidate = {
                        **post,
                        "lock_marker_path": post_marker_state["path"],
                        "policy_token": marker["policy_token"],
                        "qualification_basis": "gpulockd-policy",
                    }
                locked.append((candidate, handle, lock_path))
            decisions.append(decision)
        write_json_atomic(
            archive / "gpu-discovery-initial.json",
            {
                "schema_version": "m3-gpu-discovery-initial-v1",
                "epoch_id": epoch_id,
                "discovery_mode": discovery_mode,
                "mode_reason": mode_reason,
                "lock_marker_state_dir": str(lock_state_dir),
                "configured_lock_marker_states": [
                    marker_states[gpu] for gpu in _CONFIGURED_GPUS
                ],
                "external_assignment": external_assignment,
                "requested_max_devices": requested_max_devices,
                "effective_max_devices": max_devices,
                "selection_notes": selection_notes,
                "initial_snapshot": initial,
                "post_lock_decisions": decisions,
                "raw_locked_candidates": [
                    row for row, _handle, _path in locked
                ],
                "sealed_before_qualification": True,
            },
        )
        if not locked:
            if use_locked_mode:
                raise RuntimeError(
                    "locked discovery found no policy-healthy lockable candidate"
                )
            raise RuntimeError(
                "dynamic discovery found no lockable raw candidate"
            )
        qualified: list[tuple[dict[str, Any], Any, Path]] = []
        decision_by_index = {
            item["initial"]["physical_index"]: item for item in decisions
        }

        def qualify_one(item: tuple[dict[str, Any], Any, Path]) -> dict[str, Any]:
            row, _handle, _lock_path = item
            try:
                return qualifier(
                    row,
                    epoch_id,
                    archive / "qualification" / f"gpu-{row['physical_index']}",
                )
            except Exception as error:
                qualification = {
                    "schema_version": "m3-gpu-telemetry-qualification-result-v2",
                    "qualifier_version": QUALIFIER_SCHEMA,
                    "physical_index": row["physical_index"],
                    "gpu_uuid": row["uuid"],
                    "status": "INELIGIBLE_TELEMETRY_STATE",
                    "reasons": [f"qualifier-execution-failure:{error}"],
                    "attempt_count": 1,
                    "retry_permitted_same_epoch": False,
                    "telemetry_only": True,
                }
                write_json_atomic(
                    archive / "qualification" / f"gpu-{row['physical_index']}" / "qualification.json",
                    qualification,
                )
                return qualification

        if use_locked_mode:
            qualifications = []
            for row, _handle, _lock_path in locked:
                decision = decision_by_index[row["physical_index"]]
                marker_state = decision["post_lock_marker"]
                marker = marker_state["marker"]
                qualification = {
                    "schema_version": "m3-gpu-daemon-policy-qualification-v1",
                    "qualifier_version": _LOCKED_QUALIFIER_SCHEMA,
                    "epoch_id": epoch_id,
                    "physical_index": row["physical_index"],
                    "gpu_uuid": row["uuid"],
                    "pci_bus_id": row["pci_bus_id"],
                    "status": "PASS",
                    "reasons": [],
                    "qualification_satisfied_by": "gpulockd-daemon-policy",
                    "policy_token": marker["policy_token"],
                    "marker_path": marker_state["path"],
                    "marker": marker,
                    "health_snapshot": row,
                    "load_soak_performed": False,
                    "load_soak_skip_reason": (
                        "live gpulockd policy enforces the configured SM lock; "
                        "unlocked-DVFS FA4 soak is not applicable"
                    ),
                    "attempt_count": 1,
                    "retry_permitted_same_epoch": False,
                    "telemetry_only": True,
                }
                write_json_atomic(
                    archive / "qualification" / f"gpu-{row['physical_index']}" / "qualification.json",
                    qualification,
                )
                qualifications.append(qualification)
        else:
            with ThreadPoolExecutor(
                max_workers=min(MAX_CONCURRENT_PHYSICAL_GPUS, len(locked))
            ) as executor:
                qualifications = list(executor.map(qualify_one, locked))

        for (row, handle, lock_path), qualification in zip(
            locked, qualifications, strict=True
        ):
            decision = decision_by_index[row["physical_index"]]
            decision["qualification"] = qualification
            decision["qualification_status"] = qualification.get("status")
            if qualification.get("status") == "PASS":
                qualified.append((row, handle, lock_path))
            else:
                decision["telemetry_rejection"] = "INELIGIBLE_TELEMETRY_STATE"
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        ordered_qualified = sorted(
            qualified, key=lambda item: item[0]["physical_index"]
        )
        selected = ordered_qualified[:max_devices]
        for row, handle, _lock_path in ordered_qualified[max_devices:]:
            decision_by_index[row["physical_index"]][
                "post_qualification_rejection"
            ] = "configured-max-selection-limit"
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        for row, _handle, _path in selected:
            decision_by_index[row["physical_index"]]["selected"] = True
        if not selected:
            raise RuntimeError(
                "telemetry qualification produced an empty eligible GPU set"
            )
        eligibility = {
            "discovery_mode": discovery_mode,
            "mode_reason": mode_reason,
            "configured_physical_gpus": list(_CONFIGURED_GPUS),
            "requested_max_devices": requested_max_devices,
            "max_devices": max_devices,
            "selection_notes": selection_notes,
        }
        if use_locked_mode:
            eligibility.update({
                "lock_state_dir": str(lock_state_dir),
                "lock_sm_mhz": _LOCK_SM_MHZ,
                "clock_readback_tolerance_fraction": _LOCKED_CLOCK_WINDOW_FRACTION,
                "forbidden_slowdown_reason_mask": _LOCKED_SLOWDOWN_REASON_MASK,
                "temperature_c_sane_range_inclusive": [0, 150],
                "power_draw_w_minimum_inclusive": 0,
                "power_limit_w_minimum_exclusive": 0,
                "qualifier_version": _LOCKED_QUALIFIER_SCHEMA,
                "qualification_satisfied_by": "gpulockd-daemon-policy",
                "load_soak_performed": False,
                "utilization_and_resident_memory_gating": False,
            })
        else:
            eligibility.update({
                "utilization_percent": 0,
                "memory_used_mib_strictly_below": GPU_MEMORY_ELIGIBILITY_MIB,
                "qualifier_version": QUALIFIER_SCHEMA,
                "one_attempt_per_epoch": True,
            })
        record = {
            "schema_version": SCHEMA_VERSIONS["discovery"],
            "utc_time": utc_now(),
            "epoch_id": epoch_id,
            "discovery_mode": discovery_mode,
            "mode_reason": mode_reason,
            "external_assignment": external_assignment,
            "selection_notes": selection_notes,
            "eligibility": eligibility,
            "configured_lock_marker_states": [
                marker_states[gpu] for gpu in _CONFIGURED_GPUS
            ],
            "initial_snapshot": initial,
            "decisions": decisions,
            "qualified_before_numeric_order": [
                row for row, _handle, _path in qualified
            ],
            "ordered_selected": [
                row for row, _handle, _path in selected
            ],
            "numeric_order": True,
            "locks_held": True,
        }
        write_json_atomic(discovery_path, record)
        return DiscoveryLease(archive, initial, selected, discovery_mode)
    except Exception:
        for handle in handles:
            if not handle.closed:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        raise


def build_assignment_record(case_ids: Iterable[str], eligible_gpus: Iterable[int]) -> dict[str, Any]:
    devices = ordered_eligible_gpus(eligible_gpus)
    assignments = assign_cases(case_ids, devices)
    return {
        "schema_version": SCHEMA_VERSIONS["assignment"],
        "utc_time": utc_now(),
        "ordered_eligible_gpus": list(devices),
        "case_order": sorted(assignments, key=lambda case: case.encode("utf-8")),
        "assignments": assignments,
        "predicate": "bytewise-lexicographic case IDs round-robin over numeric discovered GPUs",
    }


def write_assignment(path: Path, case_ids: Iterable[str], eligible_gpus: Iterable[int]) -> dict[str, Any]:
    result = build_assignment_record(case_ids, eligible_gpus)
    write_json_atomic(path, result)
    return result


def query_device_identity(gpu: int) -> dict[str, Any]:
    if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu not in _CONFIGURED_GPUS:
        raise ValueError("physical GPU index must be listed in PIPELINK_GPUS")
    rows = _query_snapshot(gpu)
    row = next((item for item in rows if item["physical_index"] == gpu), None)
    if row is None:
        raise RuntimeError(f"physical GPU {gpu} is absent")
    return {
        "physical_index": gpu,
        "uuid": row["uuid"],
        "pci_bus_id": row["pci_bus_id"],
        "name": "NVIDIA GPU",
        "driver_version": "queried-by-discovery",
    }


def _probe_pair_valid(probes: object, validator: Any) -> bool:
    if not isinstance(probes, list) or len(probes) != 2 or any(not isinstance(probe, dict) for probe in probes):
        return False
    first = probes[0].get("timestamp_ns")
    second = probes[1].get("timestamp_ns")
    return isinstance(first, int) and isinstance(second, int) and second - first == FAILOVER_PROBE_INTERVAL_NS and all(validator(probe) for probe in probes)


def _absence_probe(probe: dict[str, Any]) -> bool:
    output = probe.get("output")
    return probe.get("uuid_present") is False or (isinstance(output, str) and any(marker in output for marker in ABSENT_PROBE_MARKERS))


def _status_pair(api: str) -> dict[str, int]:
    if api == "runtime":
        return RUNTIME_FAILOVER_STATUSES
    if api == "driver":
        return DRIVER_FAILOVER_STATUSES
    raise ValueError("CUDA status api must be runtime or driver")


def objective_failover_trigger(record: dict[str, Any]) -> tuple[bool, str]:
    trigger = record.get("trigger")
    interval = record.get("job_interval")
    device = record.get("assigned_device")
    if not isinstance(trigger, dict):
        return False, "missing-trigger"
    if not isinstance(interval, dict) or not isinstance(interval.get("start_ns"), int) or not isinstance(interval.get("end_ns"), int) or interval["end_ns"] <= interval["start_ns"] or not isinstance(device, dict) or not isinstance(device.get("uuid"), str) or not isinstance(device.get("pci_bus_id"), str):
        return False, "missing-device-or-job-interval"
    kind = trigger.get("class")
    if kind == "device-absent-two-probes":
        valid = _probe_pair_valid(trigger.get("probes"), _absence_probe)
        return valid, "eligible-device-absence" if valid else "absence-probes-not-exact"
    if kind == "xid-in-job-interval":
        timestamp = trigger.get("timestamp_ns")
        evidence = trigger.get("raw_xid_evidence")
        bound = isinstance(evidence, str) and (device["uuid"] in evidence or device["pci_bus_id"] in evidence)
        valid = isinstance(timestamp, int) and interval["start_ns"] <= timestamp <= interval["end_ns"] and bound
        return valid, "eligible-xid" if valid else "xid-not-bound-inside-job"
    if kind == "cuda-status-reproduced":
        try:
            statuses = _status_pair(trigger.get("api"))
        except ValueError:
            return False, "unknown-cuda-api"
        name = trigger.get("status_string")
        status_valid = isinstance(name, str) and statuses.get(name) == trigger.get("status_numeric")
        def reproduced(probe: dict[str, Any]) -> bool:
            probe_name = probe.get("status_string")
            return probe.get("fresh_context") is True and isinstance(probe_name, str) and statuses.get(probe_name) == probe.get("status_numeric")
        valid = status_valid and _probe_pair_valid(trigger.get("fresh_context_probes"), reproduced)
        return valid, "eligible-cuda-status" if valid else "cuda-status-not-exactly-reproduced"
    return False, "ineligible-trigger-class"


def build_failover_record(record: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(record)
    result["schema_version"] = SCHEMA_VERSIONS["failover"]
    result.setdefault("utc_time", utc_now())
    required = {"case_id", "eligible_gpu_order", "assigned_device", "next_device", "job_interval", "trigger", "partial_evidence_archive", "raw_stderr_evidence"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"failover record misses {sorted(missing)}")
    eligible_order = list(ordered_eligible_gpus(result["eligible_gpu_order"]))
    assigned = result["assigned_device"]
    next_device = result["next_device"]
    for label, device in (("assigned_device", assigned), ("next_device", next_device)):
        if not isinstance(device, dict) or {"physical_index", "uuid", "pci_bus_id"} - set(device):
            raise ValueError(f"{label} must bind physical index, UUID, and PCI bus ID")
        if device["physical_index"] not in _CONFIGURED_GPUS:
            raise ValueError(f"{label} physical index is not listed in PIPELINK_GPUS")
    assigned_index = eligible_order.index(assigned["physical_index"]) if assigned["physical_index"] in eligible_order else -1
    if assigned_index < 0 or assigned_index + 1 >= len(eligible_order) or next_device["physical_index"] != eligible_order[assigned_index + 1]:
        raise ValueError("failover must use the immediate next archived eligible device")
    eligible, reason = objective_failover_trigger(result)
    if not eligible:
        raise ValueError(f"ineligible failover: {reason}")
    if not isinstance(result["partial_evidence_archive"], str) or not result["partial_evidence_archive"]:
        raise ValueError("partial evidence archive path is required")
    if not isinstance(result["raw_stderr_evidence"], str):
        raise ValueError("raw stderr evidence must be a string")
    result["objective_trigger"] = {"eligible": True, "reason": reason}
    result["restart"] = {"scope": "entire-case-and-all-direct-comparison-series", "attempt": 0, "combine_pairs_across_devices": False}
    return result


def write_failover_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    result = build_failover_record(record)
    write_json_atomic(path, result)
    return result
