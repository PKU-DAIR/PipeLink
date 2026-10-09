"""Compiled-object evidence; symbol probing stays in the existing CUDA mechanism."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping

from ..canonjson import canonical_bytes
from ..model import ClauseFailure, PackError, SymbolInfo, SymbolTable
from ..profile import Profile, get_profile
from .. import symbols


@dataclass(frozen=True)
class ObjectFacts:
    symbols: SymbolTable
    max_threads: int | None = None
    cluster_dim: tuple[int, int, int] | None = None


def parse_elf(text: str) -> tuple[int | None, tuple[int, int, int] | None]:
    """Read CUDA ELF integer attributes, leaving absent LTO-only attributes absent."""
    limits = []
    clusters = set()
    for match in re.finditer(r"Attribute:\s*(EIATTR_\w+)(.*?)(?=Attribute:|\Z)", text, re.DOTALL):
        name, body = match.groups()
        if name not in {"EIATTR_MAX_THREADS", "EIATTR_CLUSTER_DIM", "EIATTR_CTA_PER_CLUSTER"}:
            continue
        value = re.search(r"Value:\s*([^\n]*(?:\n[ \t]+0x[^\n]*)*)", body)
        numbers = [] if value is None else [int(item, 16) if item.lower().startswith("0x") else int(item)
                                             for item in re.findall(r"0x[0-9a-fA-F]+|\b[0-9]+\b", value[1])]
        if len(numbers) != 3 or any(item <= 0 for item in numbers):
            clause = "P-LAUNCH" if name == "EIATTR_MAX_THREADS" else "P-TOPO"
            raise PackError([ClauseFailure(clause, "$.object.elf", f"cannot parse {name}", {"attribute": body.strip()})])
        if name == "EIATTR_MAX_THREADS":
            # CUDA emits a three-dimensional maximum block shape; nvcc's
            # __launch_bounds__ uses (maxThreads, 1, 1).
            limit = 1
            for item in numbers:
                limit *= item
            limits.append(limit)
        else:
            clusters.add(tuple(numbers))
    if len(clusters) > 1:
        raise PackError([ClauseFailure("P-TOPO", "$.object.elf", "object contains conflicting compiled cluster dimensions",
                                      {"dimensions": [list(shape) for shape in sorted(clusters)]})])
    return min(limits) if limits else None, next(iter(clusters), None)


def observe(object_path: Path, scratch: Path, seed: Mapping[str, str], env: Mapping[str, str],
            profile: Profile) -> ObjectFacts:
    """Run the existing probe in isolation so its nested tempfiles stay in scratch.

    ``seed`` holds every import the stage declares (``symbols.import_seed``): seeded symbols are
    resolved in one probe link, so only undeclared imports pay one probe link each.
    """
    seed_path = scratch / "symbol-seed.json"
    seed_path.write_bytes(canonical_bytes(dict(seed)))
    child_env = {**env, "TMPDIR": str(scratch), "CUDA_VISIBLE_DEVICES": ""}
    command = [sys.executable, "-m", "pipec2.pack.objectfacts", str(object_path), str(seed_path), profile.name]
    try:
        result = subprocess.run(command, env=child_env, capture_output=True, text=True, check=False)
    except OSError as error:
        raise PackError([ClauseFailure("P-SYM", "$.object", f"cannot run object probe: {error}", {})]) from error
    if result.returncode:
        raise PackError([ClauseFailure("P-SYM", "$.object", f"object probe failed: {result.stderr.rstrip()}", {})])
    obj = json.loads(result.stdout)
    maximum, cluster = parse_elf(obj["elf"])
    return ObjectFacts(SymbolTable.from_manifest(obj["symbols"]), maximum, cluster)


def _main() -> None:
    object_path, seed_path = map(Path, sys.argv[1:3])
    profile = get_profile(sys.argv[3])
    env = os.environ
    cuda = symbols.cuda_home(env)
    strong, weak = symbols._external_definitions(object_path, env)
    undefined = symbols._undefined_symbols(object_path.read_bytes(), sorted(strong | weak), cuda, env,
                                            json.loads(seed_path.read_text(encoding="utf-8")), profile=profile)
    overlap = (strong | weak) & undefined.keys()
    if overlap:
        raise ValueError(f"symbols reported as both defined and undefined: {sorted(overlap)}")
    info = {name: SymbolInfo("func", "strong") for name in strong}
    info.update({name: SymbolInfo("func", "weak") for name in weak})
    info.update({name: SymbolInfo("object" if kind == "variable" else "func", "undefined") for name, kind in undefined.items()})
    table = SymbolTable(tuple(strong), tuple(weak), tuple(undefined), info)
    elf = symbols._run([str(cuda / "bin" / "cuobjdump"), "--dump-elf", str(object_path)], env, "CUDA ELF attribute extraction")
    sys.stdout.buffer.write(canonical_bytes({"symbols": table.to_manifest(), "elf": elf}))


if __name__ == "__main__":
    _main()
