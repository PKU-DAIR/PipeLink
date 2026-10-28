#!/usr/bin/env python3
"""Pack the hand consumer, or build its provider against an existing consumer package."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from tools.pipelink_env import load
from pipec2.container import read as read_package
from pipec2.host.abi import ABI_TYPES, header_path
from pipec2.host.provider import build_provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("sm_103a", "sm_90a", "sm_120a"))
    parser.add_argument("--package", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    env = {**os.environ, **load(ROOT), "CUDA_VISIBLE_DEVICES": ""}
    os.environ.update(env)
    out = args.out or HERE / "out" / args.profile
    out.mkdir(parents=True, exist_ok=True)
    package = args.package or out / "epilogue.plink"
    if args.package is None:
        profile = {"sm_103a": 103, "sm_90a": 90, "sm_120a": 120}[args.profile]
        command = [sys.executable, "-m", "pipec2.cli", "pack", str(HERE / "epilogue.cu"),
                   "--profile", args.profile, "-D", f"CODA_PROFILE={profile}", "-o", str(package)]
        result = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True)
        (out / "pack.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    manifest, _ = read_package(package)
    host = manifest["host"]
    local = [row for row in host["provider"]["abi_types"] if row["name"] not in ABI_TYPES]
    role_view, = (row["name"] for row in local if row["id"] == 201)
    build_provider([HERE / "provider.cpp"], out / "provider.so", owner="coda_epilogue",
        owner_contract_digest=host["provider"]["owner_contract_digest"], profile=args.profile,
        dtypes_digest=host["target"]["dtypes"],
        type_ids={row["name"]: row["id"] for row in local},
        extra_types={row["name"]: row["fields"] for row in local},
        extra_headers=[str(HERE / "params.h")],
        compile_flags=["-O3", f"-I{header_path().parent}", f"-DCODA_ROLE_VIEW={role_view}"])
    print(json.dumps({"package": str(package), "provider": str(out / "provider.so"),
                      "host_group": "coda.epilogue.params"}))


if __name__ == "__main__":
    main()
