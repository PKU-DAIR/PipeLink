#!/usr/bin/env python3
"""GPU-free packages, providers, linked image, and same-source controls.

The build also repacks dp.red with an intentionally conflicting domain and
requires the public linker to reject that composition with L-HOSTCHAN. Logs
and the negative result are retained beneath the selected artifact root.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from tools.pipelink_env import load  # noqa: E402

# Match the public CLI's diagnostic prefix, not its explanatory prose.
CLAUSE = re.compile(r"(?:^pipec2: |; )\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*)\] ", re.MULTILINE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("sm_103a",), default="sm_103a")
    parser.add_argument("--out", type=Path, help="artifact root (default: out/<profile>)")
    args = parser.parse_args()
    env = {**os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": ""}
    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "multimem-domains" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    os.environ.update(env)

    from pipec2.container import read as read_package
    from pipec2.host.abi import ABI_TYPES, header_path
    from pipec2.host.provider import build_provider

    out = args.out.resolve() if args.out is not None else HERE / "out" / args.profile
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sdk = header_path().parent.parent.parent
    includes = [HERE, sdk / "pack/include", header_path().parent]
    cli = [sys.executable, "-m", "pipec2.cli"]
    nvcc = Path(env["PIPELINK_CUDA_HOME"]) / "bin/nvcc"

    def command(argv, log, *, check=True):
        result = subprocess.run(list(map(str, argv)), cwd=REPO, env=env,
                                capture_output=True, text=True)
        (logs / log).write_text(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError(f"Command failed: {logs / log}\n{result.stderr}")
        return result

    def pack(owner, package, *, defines=()):
        command([*cli, "pack", HERE / f"{owner}.cu", "--profile", args.profile,
                 "-o", package, *[flag for define in defines for flag in ("-D", define)],
                 *[flag for root in includes for flag in ("-I", root)]],
                f"{package.parent.name}.{package.stem}.pack.log")

    manifests = {}
    for owner in ("mc", "dp"):
        package = out / f"{owner}.plink"
        pack(owner, package)
        manifests[owner], _ = read_package(package)
        host = manifests[owner]["host"]
        inventory = host["provider"]["abi_types"]
        local = [row for row in inventory if row["name"] not in ABI_TYPES]
        try:
            observed = build_provider(
                [HERE / f"provider_{owner}.cpp"], out / f"{owner}.so", owner=owner,
                owner_contract_digest=host["provider"]["owner_contract_digest"],
                profile=args.profile, dtypes_digest=host["target"]["dtypes"],
                type_ids={row["name"]: row["id"] for row in local},
                extra_types={row["name"]: row["fields"] for row in local},
                extra_headers=[HERE / "abi.h"],
                compile_flags=["-O3", f"-I{HERE}", f"-I{header_path().parent}"],
            )
        except subprocess.CalledProcessError as exc:
            log = logs / f"{owner}.provider.log"
            log.write_text((exc.stdout or "") + (exc.stderr or ""))
            raise RuntimeError(f"Provider compilation failed: {log}") from exc
        if observed != inventory:
            raise RuntimeError(f"{owner}.so: independently compiled ABI differs from its package")

    def pipeline(directory, dp_package):
        handles = {}
        for owner in ("mc", "dp"):
            for suffix, domain in (("warp", "warp"), ("barrier", "named_barrier")):
                name = f"pipelink_{owner}_{suffix}"
                handles[name] = {"class": "base", "binding": "bind", "domain": domain,
                                 "owner": f"claim:{name}"}
        return {"pipelink_abi": manifests["mc"]["pipelink_abi"], "pipeline": {
            "name": "multimem-domains",
            "stages": [
                {"stage_id": manifests[owner]["stage"]["id"],
                 "artifact_path": os.path.relpath(package, directory)}
                for owner, package in (("mc", out / "mc.plink"), ("dp", dp_package))
            ],
            "handles": handles, "phases": ["steady"], "exclusive": [], "domains": {},
            "depth_priority": [], "regs_priority": [], "forced_bindings": {},
            "topology": {key: manifests["mc"]["topology"][key]
                         for key in ("persistent", "scheduler")},
        }}

    link_dir = out / "link"
    link_dir.mkdir(parents=True, exist_ok=True)
    (link_dir / "pipeline.json").write_text(
        json.dumps(pipeline(link_dir, out / "dp.plink"), indent=2) + "\n")
    command([*cli, "link", link_dir / "pipeline.json", "--out-dir", link_dir], "link.log")
    link = json.loads((link_dir / "link.json").read_text())

    # As in comm-overlap, namespace dispatch's generated owner-view declarations
    # so the control includes the exact device sources without redefining views.
    generated = link_dir / "generated"
    control = link_dir / "one-tu.cu"
    control.write_text(
        f'#include "{HERE / "mc.cu"}"\n'
        f'#include "{HERE / "dp.cu"}"\n'
        f'#include "{generated / "binding_stub.cu"}"\n'
        'namespace multimem_domains_control_dispatch {\n'
        f'#include "{generated / "dispatch.cu"}"\n'
        '}\n')
    regs = f"-maxrregcount={link['regs_cap']}"
    flags = ["-std=c++17", "-O3", "-DNDEBUG", regs, *[f"-I{root}" for root in includes]]
    command([nvcc, *flags, f"-arch={args.profile}", "--cubin", control,
             "-o", link_dir / "one-tu.cubin"], "one-tu.log")
    compute = args.profile.replace("sm_", "compute_")
    lto = args.profile.replace("sm_", "lto_")
    command([nvcc, *flags, f"-gencode=arch={compute},code={lto}", "-dc", "-rdc=true",
             control, "-o", link_dir / "one-tu-lto.o"], "one-tu-lto.log")
    command([nvcc, "-dlink", "-dlto", f"-arch={args.profile}", regs, "-cubin",
             link_dir / "one-tu-lto.o", "-o", link_dir / "one-tu-lto.cubin"],
            "one-tu-lto-link.log")
    command([nvcc, "-std=c++17", "-O3", f"-arch={args.profile}", "--cubin",
             HERE / "helpers.cu", "-o", out / "helpers.cubin"], "helpers.log")

    # A real repack preserves the ordinary owner contract machinery: only the
    # dp.red annotation switches its domain, while its caller source stays red.
    negative = out / "negative-domain"
    negative.mkdir(parents=True, exist_ok=True)
    pack("dp", negative / "dp.plink", defines=("MULTIMEM_DOMAINS_BAD_DP_RED=1",))
    (negative / "pipeline.json").write_text(
        json.dumps(pipeline(negative, negative / "dp.plink"), indent=2) + "\n")
    result = command([*cli, "link", negative / "pipeline.json", "--out-dir", negative / "link"],
                     "negative-domain.link.log", check=False)
    clauses = CLAUSE.findall(result.stderr)
    passed = result.returncode != 0 and clauses == ["L-HOSTCHAN"]
    report = {
        "profile": args.profile, "gpu_used": False, "phase": "link",
        "caller_source": "red", "channels": {"mc.red": "tp", "dp.red": "dp"},
        "expected_clause": "L-HOSTCHAN", "observed_clauses": clauses,
        "returncode": result.returncode, "passed": passed,
        "diagnostic": result.stderr.strip(), "log": "logs/negative-domain.link.log",
    }
    (negative / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"Expected L-HOSTCHAN domain rejection; see {negative / 'result.json'}")

    index = {
        "profile": args.profile, "link": "link/link.json", "linked": "link/pipeline.cubin",
        "one_tu": "link/one-tu.cubin", "one_tu_lto": "link/one-tu-lto.cubin",
        "helpers": "helpers.cubin", "providers": {"mc.owner": "mc.so", "dp.owner": "dp.so"},
        "negative_demo": "negative-domain/result.json",
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
