#!/usr/bin/env python3
"""Pack, link, and build the host ABI v3 provider and one-TU control without a GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from pipec2.profile import PROFILES
from tools.pipelink_env import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    args = parser.parse_args()
    from pipec2.container import read as read_package
    from pipec2.host.abi import ABI_TYPES
    from pipec2.host.provider import build_provider

    env = {**os.environ, **load(ROOT), "CUDA_VISIBLE_DEVICES": ""}
    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "profile-consumer" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    os.environ.update(env)
    build = HERE / "build" / args.profile
    artifacts = build / "artifacts"
    logs = build / "logs"
    artifacts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    def command(argv: list[str | Path], log: str) -> None:
        result = subprocess.run(list(map(str, argv)), cwd=ROOT, env=env,
                                capture_output=True, text=True)
        (logs / log).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Command failed; see {logs / log}\n{result.stderr}")

    cli = [sys.executable, "-m", "pipec2.cli"]
    for stage in ("interpreter", "instruction1", "instruction2"):
        package = artifacts / f"{stage}.plink"
        command([*cli, "pack", HERE / "src" / f"{stage}.cu", "--profile", args.profile,
                 "-o", package, "-I", HERE / "src",
                 "-I", ROOT / "pipec/src/pipec2/host/include",
                 "--report", artifacts / f"{stage}.plink.report.json"], f"{stage}-pack.log")
    manifest, _ = read_package(artifacts / "interpreter.plink")
    host = manifest["host"]
    inventory = host["provider"]["abi_types"]
    cuda = Path(env["PIPELINK_CUDA_HOME"])
    nvcc = cuda / "bin/nvcc"
    observed = build_provider(
        [HERE / "src/provider.cu"], build / "provider.so",
        owner="interpreter",
        owner_contract_digest=host["provider"]["owner_contract_digest"],
        profile=manifest["pipelink_abi"]["target"]["profile"],
        dtypes_digest=host["target"]["dtypes"],
        type_ids={row["name"]: row["id"] for row in inventory if row["name"] not in ABI_TYPES},
        extra_headers=[str(HERE / "src/consumer.cuh")],
        extra_types={row["name"]: row["fields"] for row in inventory
                     if row["name"] not in ABI_TYPES},
        compiler=str(nvcc),
        compile_flags=["-x", "cu", f"-arch={args.profile}", "-O2", "-DNDEBUG",
                       "-I" + str(HERE / "src"),
                       "-I" + str(ROOT / "pipec/src/pipec2/host/include")],
    )
    if observed != inventory:
        raise RuntimeError("interpreter provider ABI inventory differs from packed source")
    (build / "provider-inventory.json").write_text(json.dumps(observed, indent=2) + "\n")

    pipeline = json.loads((HERE / "manifests/pipeline.json").read_text())
    pipeline["pipelink_abi"] = manifest["pipelink_abi"]
    for stage in pipeline["pipeline"]["stages"]:
        stage["artifact_path"] = "artifacts/" + Path(stage["artifact_path"]).name
    pipeline_path = build / "pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline, indent=2) + "\n")
    command([*cli, "link", pipeline_path, "--out-dir", build / "linked"], "link.log")
    command([nvcc, "-std=c++17", "-O3", "-DNDEBUG", f"-arch={args.profile}",
             "-I" + str(HERE / "src"), "-I" + str(ROOT / "pipec/src/pipec2/pack/include"),
             "-I" + str(ROOT / "pipec/src/pipec2/host/include"), "--cubin",
             HERE / "src/one_tu.cu", "-o", build / "one_tu.cubin"], "one-tu-build.log")
    command([nvcc, "-std=c++17", "-O2", "-DNDEBUG", f"-arch={args.profile}",
             "-I" + str(HERE / "src"), HERE / "src/runner.cu",
             "-L" + str(cuda / "lib64/stubs"), "-lcuda", "-o", build / "runner"],
            "runner-build.log")
    print(json.dumps({"profile": args.profile, "link": str(build / "linked/link.json"),
                      "provider": str(build / "provider.so"),
                      "one_tu": str(build / "one_tu.cubin")}, indent=2))


if __name__ == "__main__":
    main()
