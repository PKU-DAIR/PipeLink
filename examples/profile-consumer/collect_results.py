#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipec2.profile import PROFILES
from tools.stagebuild import manifest_of

ELEMENTS = 64
COUNTER_SLOT = 17

INTERPRETER_ID = "org.pipelink.profile-consumer.0-interpreter"


def u32(value: int) -> int:
    return value & 0xFFFFFFFF


def input_value(index: int) -> int:
    value = u32(index ^ 0x7F4A7C15)
    value = u32(value ^ (value >> 16))
    value = u32(value * 0x045D9F3B)
    value = u32(value ^ (value >> 16))
    return u32(value ^ 0xA5C31E27)


def expected_output() -> list[int]:
    result: list[int] = []
    for index in range(ELEMENTS):
        first = u32(input_value(index) * 3 + 17)
        result.append(u32((first ^ 0x9E3779B9) + 23))
    return result


def parse_resource(path: Path, symbol: str) -> dict[str, int]:
    text = path.read_text()
    match = re.search(
        rf"Function {re.escape(symbol)}:\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) "
        rf"LOCAL:(\d+) CONSTANT\[0\]:(\d+) TEXTURE:(\d+) SURFACE:(\d+) SAMPLER:(\d+)",
        text,
    )
    if match is None:
        raise RuntimeError(f"resource usage for {symbol} not found in {path}")
    names = ("registers", "stack_bytes", "static_shared_bytes", "local_bytes",
             "constant0_bytes", "textures", "surfaces", "samplers")
    return dict(zip(names, (int(value) for value in match.groups()), strict=True))


def sass_evidence(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    release = [line for line in lines if "REDG" in line and "GPU" in line]
    acquire = [line for line in lines if "LDG" in line and "STRONG.GPU" in line]
    return {
        "release_present": bool(release),
        "acquire_present": bool(acquire),
        "release_sample": release[:2],
        "acquire_sample": acquire[:2],
    }


def parse_one_tu_log(path: Path) -> dict[str, Any]:
    text = path.read_text().strip()
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return {
        "stdout": re.sub(r" digest=[0-9a-f]+", "", text),
        "count": int(fields["count"]),
        "counter": int(fields["counter"]),
        "errors": int(fields["errors"]),
        "mismatches": int(fields["mismatches"]),
        "status": fields["status"],
    }


def binding_value(link: dict[str, Any], name: str) -> int:
    value = link["bindings"][name]
    if isinstance(value, list):
        return int(value[0])
    if isinstance(value, dict):
        return int(value["value"])
    raise RuntimeError(f"unknown binding encoding for {name}: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect profile-consumer Gate A evidence.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    args = parser.parse_args()
    build = Path(__file__).resolve().parent / "build" / args.profile
    stages = {
        name: manifest_of(build / "artifacts" / f"{name}.plink")
        for name in ("interpreter", "instruction1", "instruction2")
    }
    descriptor = stages["instruction1"]["stage"]["descriptor"]
    launch = json.loads((build / "logs" / "linked-runtime.log").read_text())
    readback = {name: bytes.fromhex(data) for name, data in launch["readback_bytes"].items()}
    linked_input_bytes = readback["input"]
    linked_output_bytes = readback["output"]
    linked_input = list(struct.unpack(f"<{ELEMENTS}I", linked_input_bytes))
    linked_output = list(struct.unpack(f"<{ELEMENTS}I", linked_output_bytes))
    linked_counter = struct.unpack(f"<{ELEMENTS}I", readback["counters"])[COUNTER_SLOT]
    linked_errors = struct.unpack("<I", readback["errors"])[0]

    one_tu_bytes = (build / "one_tu.output.bin").read_bytes()
    if len(one_tu_bytes) != ELEMENTS * 4:
        raise RuntimeError("one-TU output has the wrong byte count")
    one_tu_output = list(struct.unpack(f"<{ELEMENTS}I", one_tu_bytes))
    one_tu_input_bytes = (build / "one_tu.input.bin").read_bytes()
    if len(one_tu_input_bytes) != ELEMENTS * 4:
        raise RuntimeError("one-TU input has the wrong byte count")
    one_tu_input = list(struct.unpack(f"<{ELEMENTS}I", one_tu_input_bytes))
    expected = expected_output()
    one_tu_log = parse_one_tu_log(build / "logs" / "one-tu-runtime.log")
    link = json.loads((build / "linked" / "link.json").read_text())
    expansions = {
        expansion["channel"]: expansion
        for expansion in link["protocol_expansions"]
    }

    linked_sass = sass_evidence(build / "logs" / "linked.sass")
    one_tu_sass = sass_evidence(build / "logs" / "one-tu.sass")
    resource_usage = {
        "linked": parse_resource(
            build / "logs" / "linked.resources.txt", "pipelink_kernel"
        ),
        "one_tu": parse_resource(
            build / "logs" / "one-tu.resources.txt",
            "org_pipelink_profile_consumer_one_tu_kernel",
        ),
    }

    page1 = binding_value(link, "pipelink_profile_consumer_page_stage1")
    page2 = binding_value(link, "pipelink_profile_consumer_page_stage2")
    counter_binding = binding_value(link, "pipelink_profile_consumer_counter_slot")
    scheduler_base = binding_value(link, "pipelink_profile_consumer_scheduler_smem_base")
    exact_equal = linked_output_bytes == one_tu_bytes
    same_input = linked_input_bytes == one_tu_input_bytes
    linked_mismatches = sum(actual != expected_value
                            for actual, expected_value in zip(linked_output, expected, strict=True))
    correct = (
        linked_mismatches == 0
        and one_tu_output == expected
        and exact_equal
        and same_input
        and linked_counter == 1
        and linked_errors == 0
        and one_tu_log["counter"] == 1
        and one_tu_log["errors"] == 0
        and one_tu_log["mismatches"] == 0
        and one_tu_log["status"] == "PASS"
    )
    constructs_ok = (
        page1 == page2 == 1
        and scheduler_base == 0
        and counter_binding == COUNTER_SLOT
        and linked_sass["release_present"]
        and linked_sass["acquire_present"]
        and int(link["threads"]) == 96
        and int(link["dyn_smem_bytes"]) == 512
    )
    passed = correct and constructs_ok

    result: dict[str, Any] = {
        "artifact": "pipelink-profile-consumer-results-v1",
        "date": date.today().isoformat(),
        "host_abi": "linux-lp64-le-v2",
        "gate_a": correct,
        "status": "PASS" if passed else "FAIL",
        "toolchain": {
            "cuda": stages["interpreter"]["pipelink_abi"]["target"]["toolkit"],
            "profile": stages["interpreter"]["pipelink_abi"]["target"]["profile"],
            "lto_arch": stages["interpreter"]["pipelink_abi"]["target"]["lto_arch"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_ordinal": 0,
        },
        "builds": {
            "linked": {
                "entry": link["kernel_symbol"],
                "launch": "pipec2 launch with a host ABI v3 owner and caller-supplied fixtures",
                "session": launch,
                "stage_artifacts": [
                    "artifacts/interpreter.plink",
                    "artifacts/instruction1.plink",
                    "artifacts/instruction2.plink",
                ],
                "link_json": "linked/link.json",
            },
            "one_tu": {
                "entry": "org_pipelink_profile_consumer_one_tu_kernel",
                "control_source": "src/one_tu.cu",
                "includes_exact_stage_sources": [
                    "src/interpreter.cu",
                    "src/instruction1.cu",
                    "src/instruction2.cu",
                ],
            },
        },
        "correctness": {
            "input": {
                "elements": ELEMENTS,
                "descriptor": {
                    **descriptor,
                    "multiply": 3,
                    "add": 17,
                    "xor_mask": "0x9e3779b9",
                    "final_add": 23,
                    "page_bytes": 256,
                },
                "values": linked_input,
                "linked_vs_one_tu_exact": same_input,
            },
            "linked": {
                "counter": linked_counter,
                "errors": linked_errors,
                "mismatches": linked_mismatches,
                "output": linked_output,
            },
            "one_tu": {**one_tu_log, "output": one_tu_output},
            "expected_output": expected,
            "linked_matches_expected": linked_output == expected,
            "one_tu_matches_expected": one_tu_output == expected,
            "linked_vs_one_tu_exact": exact_equal,
        },
        "constructs": {
            "descriptor": {
                "instruction_stage_declarations": {
                    stages[name]["stage"]["id"]: stages[name]["stage"]["descriptor"]
                    for name in ("instruction1", "instruction2")
                },
                "queue_owner": INTERPRETER_ID,
                "fetch": "interpreter params descriptor pointer to gmem descriptor[0]",
                "params_layout": link["params"],
            },
            "smem_page": {
                "capacity_pages": 2,
                "page_bytes": 256,
                "instruction1_binding": page1,
                "instruction2_binding": page2,
                "physical_byte_offset": page1 * 256,
                "phase_exclusive_reuse": page1 == page2,
                "channel": "page_handoff",
                "physical_intervals": [
                    interval
                    for interval in link["intervals"]["smem"]
                    if interval["handle"] in {
                        "pipelink_profile_consumer_page_stage1",
                        "pipelink_profile_consumer_page_stage2",
                    }
                ],
                "link_protocol": expansions["page_handoff"],
            },
            "gmem_counter": {
                "capacity_slots": 64,
                "bound_slot": counter_binding,
                "channel": "dependency_counter",
                "release_ptx": "red.release.gpu",
                "acquire_ptx": "ld.acquire.gpu",
                "linked_sass": linked_sass,
                "one_tu_sass": one_tu_sass,
                "physical_interval": link["intervals"]["gmem_counter"],
                "link_protocol": expansions["dependency_counter"],
            },
            "interpreter_as_stage": {
                "stage_id": INTERPRETER_ID,
                "role": "sched",
                "scheduler_smem_byte_base": scheduler_base,
            },
        },
        "resource_usage": resource_usage,
        "manifest_validation": {
            "schema": stages["interpreter"]["pipelink_abi"]["schema"],
            "pack_reports": [
                "artifacts/interpreter.plink.report.json",
                "artifacts/instruction1.plink.report.json",
                "artifacts/instruction2.plink.report.json",
            ],
            "link_completed": True,
        },
        "findings": [],
    }
    (build / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"gate_a": correct, "status": result["status"],
                      "profile": args.profile, "results": str(build / "results.json")}))

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
