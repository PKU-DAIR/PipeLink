#!/usr/bin/env python3
"""Compiler include graph and private-owner token audit of P4 clean trees."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from run_p4 import REPO, PORT, SCRATCH, EVIDENCE, digest, save


def main():
    from pipec2.host.abi import header_path
    spec = importlib.util.spec_from_file_location("mok_build", PORT / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    rows = json.loads((EVIDENCE / "source-absence.json").read_text())
    result = []
    for row in rows:
        source_owner = row["owner"]
        owner = "act" if source_owner == "act_gptoss" else source_owner
        tree = SCRATCH / "source-absence" / source_owner
        includes = [tree / "csrc", tree / "csrc/megakernel", tree / "include", tree / owner,
                    tree / "tk", REPO / "pipec/src/pipec2/pack/include", header_path().parent]
        private = re.compile(r"\bmok_(?:" + "|".join(o for o in build.OWNERS if o != owner) + r")_owner\b")
        violations = []
        for directory in (tree / owner, tree / "include", tree / "csrc"):
            for path in directory.rglob("*"):
                if path.is_file() and private.search(path.read_text()):
                    violations.append(str(path.relative_to(tree)))
        dependencies = []
        for relative in row["owner_cu_files"]:
            source = tree / relative
            passes = build.PASSES if owner == "framework" and source.name == "provider.cu" else (None,)
            for pass_name in passes:
                flags = [*build.FLAGS, *[f"-I{i}" for i in includes]]
                if pass_name:
                    flags.append(f"-DMOK_FRAMEWORK_PASS={build.PASSES.index(pass_name)}")
                argv = [str(Path(os.environ["PIPELINK_CUDA_HOME"]) / "bin/nvcc"),
                        *flags, "-arch=sm_103a", "-M", str(source)]
                run = subprocess.run(argv, cwd=tree, text=True, capture_output=True)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                paths = sorted({Path(p).resolve() for p in shlex.split(run.stdout.replace("\\\n", " ").partition(":")[2])})
                forbidden = [str(p) for p in paths if p.is_relative_to(PORT) or p.is_relative_to(REPO / "third_party/mixture-of-kittens")]
                if forbidden:
                    raise RuntimeError(f"nonisolated include: {forbidden}")
                dependencies.append({"source": relative, "pass": pass_name, "argv": argv,
                    "returncode": run.returncode, "paths": list(map(str, paths)),
                    "local_inputs": {str(p.relative_to(tree)): digest(p) for p in paths if p.is_relative_to(tree)}})
        if violations:
            raise RuntimeError(f"foreign private namespace tokens: {violations}")
        result.append({"owner": source_owner, "include_audit": "PASS", "token_audit": "PASS",
                       "forbidden_tokens": private.pattern, "foreign_owner_cu_files": [], "dependencies": dependencies})
    save(EVIDENCE / "include-token-audit.json", {"status": "PASS", "owners": result})
    print("P4 compiler include/private-token audit: PASS", flush=True)


if __name__ == "__main__":
    main()
