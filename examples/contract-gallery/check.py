#!/usr/bin/env python3
"""GPU-free contract rejections and stage swaps against one T2 producer package."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RING = HERE.parent / "ring-pipeline"
sys.path.insert(0, str(REPO))

from pipec2.profile import PROFILES
from tools.pipelink_env import load

# Read clause tokens from the CLI's structured diagnostic prefix, never its prose.
CLAUSE = re.compile(r"(?:^pipec2: |; )\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*)\] ", re.MULTILINE)
CASES = (
    ("baseline", RING / "src/consumer.cu", "link", None),
    ("protocol", HERE / "consumers/protocol.cu", "link", "L-PROTO"),
    ("topology", HERE / "consumers/topology.cu", "link", "L-TOPO"),
    ("handle", HERE / "consumers/handle.cu", "link", "L-HANDLE"),
    ("init", HERE / "consumers/init.cu", "pack", "P-INIT"),
    ("params", HERE / "consumers/params.cu", "link", "L-PARAMS"),
    ("phase", HERE / "consumers/phase.cu", "link", "L-PHASE"),
    ("swap", HERE / "consumers/swap.cu", "link", None),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    args = parser.parse_args()
    out = HERE / "out" / args.profile
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": ""}
    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "contract-gallery" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    ring_pipeline = runpy.run_path(str(RING / "build.py"))["pipeline"]

    def command(arguments: list[str], name: str) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["uv", "run", "--project", "pipec", "--locked", "pipec2", *arguments],
            cwd=REPO, env=env, capture_output=True, text=True,
        )
        (logs / f"{name}.log").write_text(result.stdout + result.stderr)
        return result

    def require(result: subprocess.CompletedProcess, name: str) -> dict:
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {logs / (name + '.log')}\n{result.stderr}")
        return json.loads(result.stdout)

    def pack(source: Path, name: str) -> subprocess.CompletedProcess:
        package = out / f"{name}.plink"
        # A stale package must not make a rejected pack appear to produce output.
        package.unlink(missing_ok=True)
        return command(["pack", str(source), "--profile", args.profile, "-o", str(package)], f"{name}.pack")

    catalogue = require(command(["clauses", "--json"], "clauses"), "clauses")
    missing = {clause for _, _, _, clause in CASES if clause} - catalogue.keys()
    if missing:
        raise RuntimeError(f"Expected clauses absent from pipec2 clauses --json: {sorted(missing)}")
    started = time.perf_counter()
    require(pack(RING / "src/producer.cu", "producer"), "producer.pack")
    producer = require(command(["inspect", str(out / "producer.plink"), "--json"], "producer.inspect"), "producer.inspect")
    rows = []
    print(f"{args.profile}: one producer package, reused unchanged in every pipeline")
    print(f"{'consumer':<12} {'expected':<17} {'observed':<17} {'package':<8} result")
    for name, source, expected_phase, expected_clause in CASES:
        result = pack(source, name)
        phase = "pack"
        package_exists = (out / f"{name}.plink").is_file()
        if result.returncode == 0 and package_exists and expected_phase == "link":
            path = out / f"{name}.pipeline.json"
            stages = [
                {"stage_id": producer["stage"]["id"], "artifact_path": "producer.plink"},
                {"stage_id": "org.ring.consumer", "artifact_path": f"{name}.plink"},
            ]
            path.write_text(json.dumps(ring_pipeline(producer["pipelink_abi"], stages, 2), indent=2) + "\n")
            result = command(["link", str(path), "--out-dir", str(out / name)], f"{name}.link")
            phase = "link"
        clauses = CLAUSE.findall(result.stderr)
        passed = (
            phase == expected_phase
            and (package_exists if expected_phase == "link" else not package_exists)
            and ((result.returncode == 0 and not clauses) if expected_clause is None
                 else (result.returncode != 0 and clauses == [expected_clause]))
        )
        expected = f"{expected_phase}:{expected_clause or 'OK'}"
        observed = f"{phase}:{','.join(clauses) or ('OK' if result.returncode == 0 else 'ERROR')}"
        row = {
            "consumer": name, "expected_phase": expected_phase, "expected_clause": expected_clause,
            "observed_phase": phase, "observed_clauses": clauses, "returncode": result.returncode,
            "package_produced": package_exists, "passed": passed, "diagnostic": result.stderr.strip(),
        }
        rows.append(row)
        print(f"{name:<12} {expected:<17} {observed:<17} {'yes' if package_exists else 'no':<8} {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(result.stderr or f"See {logs}")
    elapsed = time.perf_counter() - started
    passed = all(row["passed"] for row in rows)
    (out / "results.json").write_text(json.dumps({
        "profile": args.profile, "date": datetime.now(timezone.utc).isoformat(),
        "producer_packs": 1, "gpu_used": False, "elapsed_seconds": elapsed,
        "passed": passed, "cases": rows,
    }, indent=2) + "\n")
    print(f"{sum(row['passed'] for row in rows)}/{len(rows)} cases passed; {elapsed:.2f}s; {out / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
