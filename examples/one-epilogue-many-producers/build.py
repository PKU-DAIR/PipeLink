#!/usr/bin/env python3
"""GPU-free packing, linking, and same-source one-TU controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipec/src"))
from pipec2.container import read as read_package
from params import PROFILE_SOURCES
from pipelines.generate import generate
from evaluator.harness.parity.example import onetu_control
from tools.pipelink_env import load as load_environment
from tools.stagebuild import pack_cu


class Commands:
    def __init__(self, env):
        self.env = env

    def run(self, command, log_path, *, env=None):
        result = subprocess.run(list(map(str, command)), cwd=REPO, env=env or self.env,
                                capture_output=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Command failed: {log_path}\n{result.stderr.decode(errors='replace')}")
        return result.stdout


def assert_consumer_invariance(recipes, scale, profile, out, env, runner):
    """Lower against each opaque partner; compare source bytes, not a rebuild ledger."""
    sources = []
    for name, row in recipes.items():
        manifest = read_package(Path(row["consumer"]["package"]))[0]
        if manifest["stage"]["id"] != "tensor.consumer.tilelang-epilogue":
            continue
        destination = out / "tilelang" / "consumer-partners" / name
        python = env["PIPELINK_TILELANG_PYTHON"]
        runner.run([python, "-m", "pipelink_tilelang", "--kernel",
                    HERE / "consumer-tilelang/epilogue_contract.py", "--profile", profile,
                    "--out", destination], out / "logs" / f"{name}.consumer-lowering.log",
                   env={**env, "PIPELINK_ACC_PARTNER": row["acc"]["package"],
                        "PIPELINK_SCALE_PARTNER": scale["package"]})
        source = destination / "generation/tilelang-epilogue/roles.cu"
        if source.read_bytes() != Path(row["consumer"]["source"]).read_bytes():
            raise AssertionError(f"Contract-authored consumer changed for partner {name}")
        sources.append(source)
    if sources:
        baseline = sources[0].read_bytes()
        others = {other: sorted((HERE / "out" / other / "tilelang/consumer-partners").glob(
            "*/generation/tilelang-epilogue/roles.cu")) for other in sorted(PROFILE_SOURCES) if other != profile}
        for source in [*sources[1:], *(path for paths in others.values() for path in paths)]:
            if source.read_bytes() != baseline:
                raise AssertionError(f"Contract-authored consumer differs: {source}")
        generated = ", ".join(f"{len(paths)} {other}" for other, paths in others.items())
        print(f"Consumer source bytes identical across {len(sources)} {profile} partners"
              f" and generated partners ({generated})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_SOURCES), required=True)
    parser.add_argument("--extra-cells", type=Path,
                        help="JSON mapping extra cell names to {acc,consumer} build recipes; recipe keys: package,source,includes,defines")
    args = parser.parse_args()
    profile = args.profile
    out = HERE / "out" / profile
    scratch = Path(load_environment(REPO)["PIPELINK_SCRATCH_ROOT"]) / "one-epilogue" / profile
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **load_environment(REPO), "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(scratch)}
    if args.extra_cells:
        for key in ("PIPELINK_TILELANG_PYTHON", "TILELANG_INCLUDE_DIR"):
            if not env.get(key):
                parser.error(f"--extra-cells requires the pinned TileLang environment: set {key}")
    else:
        print("Building hand-CUDA cells only; TileLang cells and their source-invariance assertion "
              "require the pinned TileLang environment and --extra-cells", flush=True)
    runner = Commands(env)
    includes = [REPO / "third_party/cutlass/include", REPO / "pipec/src/pipec2/pack/include"]

    def pack(name, source, defines=None, extra_includes=()):
        package = out / "packed" / f"{name}.plink"
        package.parent.mkdir(parents=True, exist_ok=True)
        recipe = {"package": str(package), "source": str(source),
                  "includes": list(map(str, [*includes, *extra_includes])), "defines": defines or {}}
        pack_cu(runner, source, package, includes=[Path(p) for p in recipe["includes"]],
                      defines=recipe["defines"], log=out / "logs" / f"{name}.pack.log",
                      env=env, profile=profile)
        return recipe

    consumer = pack("epilogue", HERE / "consumer/epilogue.cu")
    scale = pack("rowscale", HERE / "providers/rowscale/rowscale.cu")
    sources = PROFILE_SOURCES[profile]
    recipes = {placement: {"acc": pack(placement, HERE / sources["provider"],
                                       {**sources["defines"], f"PL_EXPORT_{placement.upper()}": "1"}),
                           "consumer": consumer} for placement in sources["placements"]}
    # TileLang recipes plug in here; the three-stage graph and control assembly are unchanged.
    if args.extra_cells:
        recipes.update(json.loads(args.extra_cells.read_text()))
    assert_consumer_invariance(recipes, scale, profile, out, env, runner)
    # Native fused controls: one hand-fused single-stage package per mainloop (not a one-TU parity denominator).
    natives = {"native-cuda": pack("native-cuda", HERE / sources["native"])}
    if args.extra_cells:
        tilelang = out / "tilelang" / "native"
        runner.run(["uv", "run", "--project", "pipec", "--locked", "python",
                    HERE / "native/generate_tilelang.py",
                    "--provider-source", out / "tilelang/provider/roles.cu",
                    "--provider-package", out / "packed/tl-reg.plink", "--out", tilelang],
                   out / "logs" / "native-tilelang.generate.log")
        natives["native-tilelang"] = pack("native-tilelang", tilelang / "native_tilelang.cu",
                                          extra_includes=[env["TILELANG_INCLUDE_DIR"]])
    cells = {name: tuple(Path(recipe["package"]) for recipe in (row["acc"], scale, row["consumer"]))
             for name, row in recipes.items()}
    cells.update({name: (Path(recipe["package"]),) for name, recipe in natives.items()})
    pipelines = generate(cells, out / "pipelines")
    index = {"profile": profile, "cells": {}, "natives": {}}
    rel = lambda path: str(Path(path).relative_to(REPO))  # A synced out/ runs elsewhere.
    for name, pipeline in pipelines.items():
        link_dir = out / "links" / name
        runner.run(["uv", "run", "--project", "pipec", "--locked", "pipec2", "link", pipeline,
                    "--out-dir", link_dir], out / "logs" / f"{name}.link.log")
        link = json.loads((link_dir / "link.json").read_text())
        if name in natives:
            index["natives"][name] = {"link": rel(link_dir / "link.json"),
                                      "linked": rel(link_dir / "pipeline.cubin")}
            continue
        stages = [recipes[name]["acc"], scale, recipes[name]["consumer"]]
        controls = onetu_control(link_dir, stages, profile=profile, env=env,
                                 extra_includes=([env["TILELANG_INCLUDE_DIR"]]
                                                 if env.get("TILELANG_INCLUDE_DIR") else ()),
                                 log_dir=out / "logs" / name)
        native = "native-tilelang" if name.startswith("tl-reg") else "native-cuda"
        index["cells"][name] = {"link": rel(link_dir / "link.json"),
                               "linked": rel(link_dir / "pipeline.cubin"),
                               **{key: rel(path) for key, path in controls.items()},
                               "regs_cap": link["regs_cap"],
                               **({"native": native} if native in natives else {})}
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
