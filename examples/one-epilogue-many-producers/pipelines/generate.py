#!/usr/bin/env python3
"""Compose two manifest-declared exports at one epilogue site."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

EXAMPLE = Path(__file__).resolve().parents[1]
REPO = EXAMPLE.parents[1]
sys.path.insert(0, str(REPO / "pipec/src"))
from pipec2.container import read as read_package


def pipeline(cell: str, packages: tuple[Path, ...], destination: Path) -> dict:
    """Compose the owners; three packages form a cell, one package is a native control."""
    owners = [read_package(path)[0] for path in packages]
    handles = {}
    # Same claim merge as tensor-import/pipelines/generate.py, extended to three owners.
    for owner in owners:
        for claim in owner["claims"]:
            handle = {"class": "base", "binding": "bind", "domain": claim["domain"],
                      "owner": f"claim:{claim['handle']}"}
            if handles.setdefault(claim["handle"], handle) != handle:
                raise ValueError(f"Conflicting handle: {claim['handle']}")
    # The packed owners carry the profile/toolkit/schema stamp; do not restate it.
    target = copy.deepcopy(owners[0]["pipelink_abi"])
    connections = []
    if len(owners) == 3:
        acc, scale, consumer = owners
        for name, provider in (("acc", acc), ("scale", scale)):
            export, = provider["tensor"]["exports"]
            imp, = (item for item in consumer["tensor"]["imports"] if item["identity"] == export["identity"])
            connections.append({"name": name,
                                "export": f"{provider['stage']['id']}:{export['name']}",
                                "import": f"{consumer['stage']['id']}:{imp['name']}"})
    return {"pipelink_abi": target, "pipeline": {
        "name": f"pipelink.one_epilogue.{cell}",
        "stages": [{"stage_id": owner["stage"]["id"],
                    "artifact_path": os.path.relpath(path, destination.parent)}
                   for owner, path in zip(owners, packages)],
        "handles": handles, "phases": ["tensor_work"], "exclusive": [], "domains": {},
        "depth_priority": [], "regs_priority": [], "forced_bindings": {},
        "topology": {"persistent": False, "scheduler": "static"},
        "tensor_connections": connections,
    }}


def generate(cells: dict[str, tuple[Path, ...]], out: Path) -> dict[str, Path]:
    """Additional provider/consumer recipes add cells without changing composition."""
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, packages in cells.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(pipeline(name, packages, path), indent=2) + "\n")
        paths[name] = path
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True,
                        help="JSON mapping cell names to [acc, scale, consumer] package paths")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate({name: tuple(Path(p).resolve() for p in paths)
              for name, paths in json.loads(args.cells.read_text()).items()}, args.out)
