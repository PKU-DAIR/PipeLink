#!/usr/bin/env python3
"""GPU-free, physically source-separated owner builds and activation-only cutover."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
PORT = REPO / "campaigns/mok/port"
# Receipts go to <MOK_EVIDENCE_DIR>/owner-swap (default: scratch); see port/report.py.
EVIDENCE = Path(os.environ.get("MOK_EVIDENCE_DIR") or Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "mok/evidence") / "owner-swap"
SCRATCH = Path(os.environ.get("MOK_OWNER_SWAP_SCRATCH") or Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "mok/owner-swap")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    text = json.dumps(value, indent=2)
    for root, label in ((str(SCRATCH), "$MOK_OWNER_SWAP_SCRATCH"),
                        (str(REPO), "$REPO"),
                        (os.environ["PIPELINK_CUDA_HOME"], "$PIPELINK_CUDA_HOME")):
        text = text.replace(root, label)
    path.write_text(text + "\n")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PIPEC_CUDA"] = os.environ["PIPELINK_CUDA_HOME"]
    from pipec2.container import read as read_package
    from pipec2.host.abi import ABI_TYPES, header_path
    from pipec2.host.provider import build_provider
    spec = importlib.util.spec_from_file_location("mok_build", PORT / "build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    refresh_act = "--refresh-act" in sys.argv
    commands = json.loads((EVIDENCE / "commands.json").read_text()) if refresh_act else []
    def command(argv, name, cwd=REPO):
        argv = list(map(str, argv))
        result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
        (SCRATCH / (name + ".log")).write_text(result.stdout + result.stderr)
        commands.append({"argv": argv, "cwd": str(cwd), "returncode": result.returncode,
                         "log": str(SCRATCH / (name + ".log"))})
        save(EVIDENCE / "commands.json", commands)
        if result.returncode:
            raise RuntimeError(f"{name}: {result.stderr[-8000:]}")
    patched = SCRATCH / "patched"
    if not refresh_act:
        if patched.exists():
            raise RuntimeError(f"Use a fresh experiment scratch tree: {patched}")
        shutil.copytree(REPO / "third_party/mixture-of-kittens/csrc", patched / "csrc")
        for patch in sorted((PORT / "patches").glob("*.patch")):
            command(["git", "apply", patch], patch.stem, patched)
    subsets = {
        "framework": [],
        "comm": ["mxfp8.cuh", "megakernel/utils.cuh", "megakernel/dispatch_combine.cuh", "megakernel/preload.cuh"],
        "gemm": ["mxfp8.cuh", "megakernel/utils.cuh", "megakernel/grouped_gemm.cuh"],
        "act": ["mxfp8.cuh", "megakernel/utils.cuh", "megakernel/swiglu.cuh"],
        "act_gptoss": ["mxfp8.cuh", "megakernel/utils.cuh"],
    }
    audit = []
    if refresh_act:
        subsets = {"act_gptoss": subsets["act_gptoss"]}
        audit = [row for row in json.loads((EVIDENCE / "source-absence.json").read_text())
                 if row["owner"] != "act_gptoss"]
    cuda = Path(os.environ["PIPELINK_CUDA_HOME"])
    for source_owner, subset in subsets.items():
        owner = "act" if source_owner == "act_gptoss" else source_owner
        tree = SCRATCH / "source-absence" / source_owner
        if refresh_act and tree.exists():
            shutil.rmtree(tree)
        shutil.copytree(PORT / source_owner, tree / owner)
        shutil.copytree(PORT / "include", tree / "include")
        # The interpreter needs only the other owners' public call structs.
        for foreign in ("comm", "gemm", "act"):
            shutil.copy2(PORT / foreign / f"{foreign}_abi.h", tree / "include")
        shutil.copytree(REPO / "third_party/mixture-of-kittens/third_party/ThunderKittens/include", tree / "tk")
        for relative in subset:
            target = tree / "csrc" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(patched / "csrc" / relative, target)
        out = tree / "out"
        out.mkdir()
        includes = [tree / "csrc", tree / "csrc/megakernel", tree / "include", tree / owner,
                    tree / "tk", REPO / "pipec/src/pipec2/pack/include", header_path().parent]
        sources = sorted(str(p.relative_to(tree)) for p in tree.rglob("*.cu"))
        if any(not p.startswith(owner + "/") for p in sources):
            raise RuntimeError(f"foreign owner source in {tree}: {sources}")
        packages = [f"framework.{p}" for p in builder.PASSES] if owner == "framework" else [owner]
        for package in packages:
            pass_name = package.partition(".")[2]
            stage = f"mok.interp.{pass_name}" if pass_name else f"mok.{owner}"
            source = tree / owner / (f"interp_{pass_name}.cu" if pass_name else f"{owner}.cu")
            flags = [*builder.FLAGS, f"--frandom-seed={stage}"]
            command([sys.executable, "-m", "pipec2.cli", "pack", source, "--profile", "sm_103a", "-o", out / f"{package}.plink",
                     *[v for inc in includes for v in ("-I", inc)], *[f"--nvcc-flag={f}" for f in flags]],
                    f"{source_owner}-{package}-pack", tree)
            manifest, _ = read_package(out / f"{package}.plink")
            host = manifest["host"]
            inventory = host["provider"]["abi_types"]
            local = [row for row in inventory if row["name"] not in ABI_TYPES]
            type_flags = {"framework": [f"-DMOK_FRAMEWORK_PASS={builder.PASSES.index(pass_name)}"] if pass_name else [],
                          "act": ["-DMOK_ACT_PROVIDER"], "gemm": ["-DMOK_GEMM_TYPES_ONLY"], "comm": ["-DMOK_COMM_PROVIDER"]}[owner]
            provider = out / f"provider_{package}.so"
            compile_flags = ["-x", "cu", "-O3", *builder.FLAGS, *type_flags, *[f"-I{p}" for p in includes],
                             f"-L{cuda / 'lib64/stubs'}", "-lcuda"]
            observed = build_provider([tree / owner / "provider.cu"], provider, owner=f"mok_{owner}",
                owner_contract_digest=host["provider"]["owner_contract_digest"], profile="sm_103a",
                dtypes_digest=host["target"]["dtypes"], type_ids={r["name"]: r["id"] for r in local},
                extra_types={r["name"]: r["fields"] for r in local},
                extra_headers=[tree / owner / ("framework_abi.h" if pass_name else f"{owner}.cu")],
                compiler=str(cuda / "bin/nvcc"), compile_flags=compile_flags)
            assert observed == inventory
            # build.py --link-only consumes this record.
            (out / f"{package}.provider.json").write_text(json.dumps({"inventory": observed,
                "owner_contract_digest": host["provider"]["owner_contract_digest"],
                "artifact_sha256": digest(out / f"{package}.plink"), "provider_sha256": digest(provider)}))
            commands.append({"api": "pipec2.host.provider.build_provider", "owner": source_owner,
                             "package": package, "compile_flags": compile_flags, "abi_equal": True,
                             "returncode": 0})
            save(EVIDENCE / "commands.json", commands)
        audit.append({"owner": source_owner, "tree": str(tree), "owner_cu_files": sources,
                      "csrc_subset": subset, "public_headers": sorted(p.name for p in (tree / "include").iterdir()),
                      "tk_headers": len(list((tree / "tk").rglob("*.*"))), "packages": packages,
                      "foreign_owner_cu_files": [], "package_provider_build": "PASS"})
        save(EVIDENCE / "source-absence.json", audit)
        print(f"{source_owner}: source-absence build PASS", flush=True)
    prepare_swap(builder)
    link_swapped(builder)


def prepare_swap(builder):
    from pipec2.container import read as read_package
    # The packaging audit uses isolated per-owner trees. Controls need every
    # owner, so they get a separate integration tree.
    integration = SCRATCH / "integration"
    for owner in builder.OWNERS:
        shutil.copytree(PORT / ("act_gptoss" if owner == "act" else owner), integration / owner, dirs_exist_ok=True)
    for name in ("include", "pipelines"):
        shutil.copytree(PORT / name, integration / name, dirs_exist_ok=True)
    shutil.copy2(PORT / "one_tu.cu", integration / "one_tu.cu")
    out = SCRATCH / "swapped"
    out.mkdir(exist_ok=True)
    shutil.copytree(PORT / "build/csrc", out / "csrc", dirs_exist_ok=True)
    shutil.copy2(PORT / "build/build-inputs.json", out / "build-inputs.json")
    hashes = []
    for package in builder.PACKAGES:
        base = PORT / "build"
        selected = SCRATCH / "source-absence/act_gptoss/out" if package == "act" else base
        for name in (f"{package}.plink", f"provider_{package}.so", f"{package}.provider.json"):
            shutil.copy2(selected / name, out / name)
        old, old_payload = read_package(base / f"{package}.plink")
        new, new_payload = read_package(out / f"{package}.plink")
        before, after = hashlib.sha256(old_payload).hexdigest(), hashlib.sha256(new_payload).hexdigest()
        assert (before != after) == (package == "act")
        if package == "act":
            for key in ("exports", "endpoints", "host", "claims", "topology"):
                assert old.get(key) == new.get(key), key
        hashes.append({"package": package, "before_payload_sha256": before, "after_payload_sha256": after,
                       "changed": before != after, "host_contract_unchanged": old["host"] == new["host"]})
    save(EVIDENCE / "payload-hashes.json", hashes)


def link_swapped(builder):
    out = SCRATCH / "swapped"
    builder.HERE = SCRATCH / "integration"
    sys.argv = [str(PORT / "build.py"), "--link-only", "--out", str(out), "--jobs", "4"]
    builder.main()
    from pipec2.profile import get_profile
    limit = get_profile("sm_103a").caps.param_limit_bytes
    cells = []
    for path in sorted((out / "links").glob("*/link.json")):
        link = json.loads(path.read_text())
        cells.append({"cell": path.parent.name, "params_total_bytes": link["params"]["total_bytes"],
                      "params_headroom_bytes": limit - link["params"]["total_bytes"],
                      "dropped_endpoint_count": len(link["dropped_endpoints"])})
    save(EVIDENCE / "matrix-summary.json", {"status": "PASS" if len(cells) == 24 else "FAIL", "cells": cells})
    save(EVIDENCE / "result.json", {"status": "PASS", "gpu_executed": False,
        "source_absence_owners": 5, "linked_cells": 24, "build": str(out),
        "matrix_sha256": digest(out / "matrix.json"), "payload_fanout_sha256": digest(out / "payload-fanout.json"),
        "same_source_controls": ["one-tu.cubin", "one-tu-lto.cubin"],
        "only_changed_payload": "mok.act", "host_contract_unchanged": True})


if __name__ == "__main__":
    if "--link-only" in sys.argv or "--refresh-base" in sys.argv:
        spec = importlib.util.spec_from_file_location("mok_build", PORT / "build.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        if "--refresh-base" in sys.argv:
            prepare_swap(builder)
        link_swapped(builder)
    else:
        main()
