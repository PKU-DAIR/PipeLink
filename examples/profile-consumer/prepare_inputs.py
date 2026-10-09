#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

STAGE1_OPCODE = 0x13579BDF
STAGE2_OPCODE = 0x2468ACE0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)

    (build / "instruction1.params.bin").write_bytes(
        struct.pack("<4I", STAGE1_OPCODE, 0, 0, 0)
    )
    (build / "instruction2.params.bin").write_bytes(
        struct.pack("<4I", STAGE2_OPCODE, 0, 0, 0)
    )
    inputs = []
    for index in range(64):
        value = index ^ 0x7F4A7C15
        value ^= value >> 16
        value = (value * 0x045D9F3B) & 0xFFFFFFFF
        value ^= value >> 16
        inputs.append(value ^ 0xA5C31E27)
    initial = {
        "descriptors": [64, 3, 17, 0x9E3779B9, 23, 256, STAGE1_OPCODE, STAGE2_OPCODE],
        "input": inputs,
        "intermediate": [0xDEADBEEF] * 64,
        "output": [0xDEADBEEF] * 64,
        "counters": [0] * 64,
        "errors": [0],
    }
    resources = {}
    for name, values in initial.items():
        data = f"initial.{name}.bin"
        (build / data).write_bytes(struct.pack(f"<{len(values)}I", *values))
        resources[name] = {
            "data": data,
            "allocation_bytes": len(values) * 4,
            "extents": [len(values)],
            "bit_strides": [32],
            "dtype_id": 5,
            "alignment": 16 if name == "descriptors" else 4,
            "origin_byte_offset": 0,
            "origin_bit_offset": 0,
            "output": f"linked.{name}.bin",
        }
    (build / "host-inputs.json").write_text(
        json.dumps({"scalars": {}, "resources": resources}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
