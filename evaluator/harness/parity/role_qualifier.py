"""Generic ratio-free 10-second role telemetry producer for case/device blocks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from .gap_eligibility import freeze_policy
from .runner import Protocol
from .telemetry import CLOCK_REASON_FIELD, ClockSampler
from .util import sha256_file, write_json_atomic
from .case_device import (
    F2_SCHEMA,
    F2_SCHEMA_V1,
    F2_SCHEMA_V2,
    F2_SCHEMA_V3,
    GAP_SCHEMA,
    evaluate_case_device_qualification,
)


def run_image_telemetry_qualifier(
    *,
    device: dict[str, Any],
    epoch_id: str,
    case_id: str,
    role_id: str,
    image_spec: dict[str, Any],
    case_config: dict[str, Any],
    policy_tuple: dict[str, Any],
    archive: Path,
) -> dict[str, Any]:
    """Run five warmups + one 10/1 telemetry-only role observation."""
    policy = freeze_policy(policy_tuple)
    command = image_spec["command"]
    if not isinstance(command, list) or not command:
        raise ValueError("role qualifier ImageSpec command is required")
    image_path = Path(image_spec["image_path"])
    if sha256_file(image_path) != image_spec["image_sha256"]:
        raise ValueError("role qualifier image hash mismatch")
    protocol = Protocol.f2_cross_source(
        archive, case_config, mode="fast"
    ).solo_protocol().record()
    common = {
        "schema_version": "m3-parity-slot-request-v1",
        "case_id": case_id,
        "image_role": "reference",
        "image": image_spec,
        "physical_gpu": device["physical_index"],
        "cuda_visible_device": 0,
        "case_config": case_config,
        "protocol": protocol,
    }
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device["physical_index"])
    warmup = subprocess.run(
        command, cwd=image_spec.get("cwd"), env=env,
        input=json.dumps({**common, "action": "warmup"}) + "\n",
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
    )
    if warmup.returncode:
        raise RuntimeError(f"role qualifier warmup failed: {warmup.stderr}")
    sampler = ClockSampler(device["physical_index"], CLOCK_REASON_FIELD)
    sampler.start()
    try:
        sampler.wait_ready()
        slot = subprocess.run(
            command, cwd=image_spec.get("cwd"), env=env,
            input=json.dumps({
                **common, "action": "slot", "attempt": 0,
                "order_position": 0, "case_device_qualification": True,
            }) + "\n",
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=240,
        )
    finally:
        sampler.stop()
    if slot.returncode:
        raise RuntimeError(f"role qualifier slot failed: {slot.stderr}")
    output = json.loads(slot.stdout)
    samples = [
        sample.to_json() for sample in sampler.samples()
        if output["sampling_window_start_ns"]
        <= sample.time_ns <= output["sampling_window_end_ns"]
    ]
    role = {
        "role_id": role_id,
        "image_sha256": image_spec["image_sha256"],
        "input_sha256": case_config.get("input_sha256", case_config["stream_map_sha256"]),
        "stream_map_sha256": case_config["stream_map_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "epoch_id": epoch_id,
        "gpu_uuid": device["uuid"],
        "pci_bus_id": device["pci_bus_id"],
        "samples": [
            {
                "query_ok": sample["query_ok"],
                "time_ns": sample["time_ns"],
                "clock_mhz": sample["clock_mhz"],
                "reason_mask": sample["reason_mask"],
                "policy_sha256": policy["policy_sha256"],
            }
            for sample in samples
        ],
        "slot_start_ns": output["slot_start_ns"],
        "observation_start_ns": output["sampling_window_start_ns"],
        "observation_end_ns": output["sampling_window_end_ns"],
        "telemetry_only": True,
    }
    archive.mkdir(parents=True, exist_ok=True)
    write_json_atomic(archive / f"{role_id}.json", role)
    return role


def run_case_device_qualification(
    *,
    schema_version: str,
    device: dict[str, Any],
    epoch_id: str,
    case_id: str,
    role_images: dict[str, dict[str, Any]],
    case_config: dict[str, Any],
    policy_tuple: dict[str, Any],
    archive: Path,
) -> dict[str, Any]:
    expected_count = (
        3
        if schema_version in {F2_SCHEMA_V1, F2_SCHEMA_V2, F2_SCHEMA_V3}
        else 2 if schema_version == GAP_SCHEMA
        else 0
    )
    if not expected_count or len(role_images) != expected_count:
        raise ValueError("case/device qualification role count/schema mismatch")
    role_order = sorted(role_images, key=lambda role: role.encode("utf-8"))
    roles = [
        run_image_telemetry_qualifier(
            device=device,
            epoch_id=epoch_id,
            case_id=case_id,
            role_id=role,
            image_spec=role_images[role],
            case_config=case_config,
            policy_tuple=policy_tuple,
            archive=archive / "roles",
        )
        for role in role_order
    ]
    source = {
        "schema_version": schema_version,
        "phase": "fast-prepair",
        "case_id": case_id,
        "assignment_epoch_id": epoch_id,
        "physical_index": device["physical_index"],
        "gpu_uuid": device["uuid"],
        "pci_bus_id": device["pci_bus_id"],
        "policy_sha256": roles[0]["policy_sha256"],
        "role_order": role_order,
        "roles": roles,
        "pair_attempt_directories": [],
    }
    write_json_atomic(archive / "qualification-input.json", source)
    result = evaluate_case_device_qualification(source)
    write_json_atomic(archive / "qualification-result.json", result)
    return result


def run_gap_case_device_qualification(
    *,
    device: dict[str, Any],
    epoch_id: str,
    case_id: str,
    reference_image: dict[str, Any],
    linked_image: dict[str, Any],
    case_config: dict[str, Any],
    policy_tuple: dict[str, Any],
    archive: Path,
) -> dict[str, Any]:
    return run_case_device_qualification(
        schema_version=GAP_SCHEMA,
        device=device,
        epoch_id=epoch_id,
        case_id=case_id,
        role_images={
            reference_image["image_id"]: reference_image,
            linked_image["image_id"]: linked_image,
        },
        case_config=case_config,
        policy_tuple=policy_tuple,
        archive=archive,
    )
