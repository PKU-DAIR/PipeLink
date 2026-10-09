"""Normalized SASS diagnostic: linked image vs same-source one-TU control.

    uv run --project pipec --locked python campaigns/mok/port/sass.py [--build-dir DIR] [--cell ID]

Reports all matrix cells by default to campaigns/mok/evidence/p2-sass.json. Selecting
--cell reports only that cell. When the default cell is included, its P1 diagnostic is
also written to p1-sass.json beside the matrix report. Root-index-only builds retain
the P1 output. --out overrides the primary report path.

Diagnostic only: acceptance is numeric parity and linked/one-TU bitwise outputs.
Each CALL target is listed so a seam-attributable call is visible.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from evaluator.gates import comparison, normalization  # noqa: E402

CALL = re.compile(r"\bCALL\.\S*\s+(\S+?)\s*[;(]|\bCALL\.\S*\s+`\(([^)]+)\)")


def tool(name: str, *args: str) -> str:
    cuda = Path(os.environ["PIPELINK_CUDA_HOME"])
    return subprocess.run([str(cuda / "bin" / name), *args], check=True, capture_output=True, text=True).stdout


def report_path(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def image(cubin: Path) -> dict:
    dump = tool("cuobjdump", "-sass", str(cubin))
    sections = normalization.sections(dump)
    usage = {}
    for section in tool("cuobjdump", "--dump-resource-usage", str(cubin)).split(" Function ")[1:]:
        name, body = section.split(":\n", 1)
        usage[name.strip()] = {key: int(value) for key, value in re.findall(r"\b(REG|STACK|LOCAL|SHARED):(\d+)", body)}
    calls = Counter()
    for line in dump.splitlines():
        if " CALL" in line:
            match = re.search(r"CALL\S*\s+(.*?)\s*;", line)
            calls[match.group(1) if match else line.strip()] += 1
    return {
        "normalized": [(name, normalization.canonicalize(body)) for name, body in sections],
        "record": {"path": report_path(cubin), "resource_usage": usage,
                   "instructions": sum(len(body) for _, body in sections),
                   "opcodes": {op: sum(1 for _, body in sections for line in body if re.search(rf"\b{op}\b", line))
                               for op in ("CALL", "STL", "LDL", "STS", "LDS")},
                   "call_targets": dict(sorted(calls.items()))},
    }


def cell_report(index_path: Path, metadata: dict) -> dict:
    index = json.loads(index_path.read_text())
    images = {role: image(index_path.parent / index[key])
              for role, key in (("control", "one_tu"), ("control_lto", "one_tu_lto"), ("linked", "linked"))}
    pairs = {}
    for control in ("control", "control_lto"):
        compared = comparison.compare_streams(images[control]["normalized"], images["linked"]["normalized"])
        deltas = compared.pop("deltas")
        pairs[f"linked_vs_{control}"] = {
            **compared,
            "normalized_sass_equal": images[control]["normalized"] == images["linked"]["normalized"],
            "N": len(deltas), "deltas": deltas[:200], "deltas_truncated": len(deltas) > 200,
        }
    return {
        **metadata,
        "schema": "pipelink-mok-sass-diagnostic-v1",
        "images": {role: data["record"] for role, data in images.items()},
        "comparisons": pairs,
    }


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def summary(payload: dict) -> dict:
    return {
        "images": {role: {key: record[key] for key in
                         ("resource_usage", "instructions", "opcodes", "call_targets")}
                   for role, record in payload["images"].items()},
        "comparisons": {name: {key: value for key, value in pair.items() if key != "deltas"}
                        for name, pair in payload["comparisons"].items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=REPO / "campaigns/mok/port/build")
    parser.add_argument("--out", type=Path, help="Primary report path (defaults to evidence/p2-sass.json for a matrix)")
    parser.add_argument("--cell", help="Report only this matrix link ID (default: all cells)")
    args = parser.parse_args()
    matrix_path = args.build_dir / "matrix.json"
    matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else None
    if matrix is None:
        if args.cell:
            parser.error("--cell requires build/matrix.json")
        cells = []
    else:
        cells = [cell for cell in matrix["cells"] if args.cell is None or cell["id"] == args.cell]
        if not cells:
            parser.error(f"No matrix cells match {args.cell!r}")
    out = args.out or REPO / "campaigns/mok/evidence" / ("p2-sass.json" if matrix is not None else "p1-sass.json")
    p1_out = out.with_name("p1-sass.json")
    if matrix is not None and out.resolve() == p1_out.resolve():
        parser.error("The matrix --out must differ from the reserved default-cell p1-sass.json path")
    metadata = {
        "diagnostic_only": True,
        "N_definition": "operand-delta count; structural differences are separate checks",
        "tool_versions": {name: tool(name, "--version").strip().splitlines()[-1] for name in ("nvcc", "cuobjdump")},
        "comparator_sources": [report_path(Path(module.__file__)) for module in (comparison, normalization)],
    }
    if matrix is None:
        payload = cell_report(args.build_dir / "index.json", metadata)
        write_report(out, payload)
        print(json.dumps(summary(payload), indent=1))
        return 0
    reports = []
    summaries = []
    for cell in cells:
        index_path = args.build_dir / cell["index"]
        report = cell_report(index_path, metadata)
        reports.append({**cell, "index": report_path(index_path), **report})
        summaries.append({"id": cell["id"], **summary(report)})
        if cell["id"] == matrix["default_cell"]:
            write_report(p1_out, report)
    write_report(out, {
        **metadata,
        "schema": "pipelink-mok-sass-matrix-diagnostic-v2",
        "default_cell": matrix["default_cell"],
        "cells": reports,
    })
    print(json.dumps({"cells": summaries}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
