"""Compose two opaque owners through their declared tensor identities."""
import copy
import json
import os
from pathlib import Path
from pipec2.container import read
from pipec2.profile import get_profile


def generate(name, packages, destination):
    destination = Path(destination)
    owners = [read(Path(path))[0] for path in packages]
    producer, consumer = owners
    export, = producer["tensor"]["exports"]
    imported, = (item for item in consumer["tensor"]["imports"] if item["identity"] == export["identity"])
    handles = {}
    caps = get_profile(producer["pipelink_abi"]["target"]["profile"]).caps
    for owner in owners:
        for claim in owner["claims"]:
            handle = {"class": "base", "binding": "bind", "domain": claim["domain"],
                      "owner": f"claim:{claim['handle']}"}
            if claim["handle"] == "pipelink_launch_regs" and claim["domain"] == "regs":
                handle = {"class": "regs", "binding": "bind", "domain": "regs",
                          "values": list(range(caps.regs_min, caps.launch_regs_max + 1)),
                          "owner": "reg_group:launch"}
            if handles.setdefault(claim["handle"], handle) != handle:
                raise ValueError(f"conflicting claim {claim['handle']}")
    pipeline = {"pipelink_abi": copy.deepcopy(producer["pipelink_abi"]), "pipeline": {
        "name": f"coda.{name}",
        "stages": [{"stage_id": owner["stage"]["id"],
                    "artifact_path": os.path.relpath(path, destination.parent)}
                   for owner, path in zip(owners, packages)],
        "handles": handles, "phases": ["tile_work"], "exclusive": [], "domains": {},
        "depth_priority": [], "regs_priority": [name for name, h in handles.items() if h["class"] == "regs"],
        "forced_bindings": {},
        "topology": {"persistent": False, "scheduler": "static"},
        "tensor_connections": [{"name": "acc", "export": f"{producer['stage']['id']}:{export['name']}",
                                "import": f"{consumer['stage']['id']}:{imported['name']}"}]}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(pipeline, indent=2) + "\n")
    return destination
