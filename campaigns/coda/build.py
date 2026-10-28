#!/usr/bin/env python3
"""Pack CODA owners, link hand/TileLang cells, build controls and native baselines."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipec/src"))
from params import PROFILES
from pipelines.generate import generate
from evaluator.harness.parity.example import onetu_control
from tools.pipelink_env import load
from tools.stagebuild import CommandRecorder, pack_cu, manifest_of, write_json


def relative(path):
    return os.path.relpath(path, REPO)


def pack(recipe, package, profile, runner, env, logs):
    source = REPO / recipe["source"]
    includes = [REPO / "pipec/src/pipec2/pack/include", *[REPO / path for path in recipe.get("includes", ())]]
    flags = [f"-std={recipe.get('std', 'c++17')}", *recipe.get("compile_flags", ())]
    package.parent.mkdir(parents=True, exist_ok=True)
    pack_cu(runner, source, package, includes=includes, defines=recipe.get("defines", {}),
            nvcc_flags=flags, profile=profile, env=env, log=logs / (package.stem + ".pack.log"))
    return {**recipe, "source": str(source), "includes": list(map(str, includes)),
            "defines": recipe.get("defines", {}), "package": str(package)}


def install_provider(recipe, label, profile, consumer, package, out, runner):
    """Execute the owner's build, then distribute only its DSO and public header."""
    substitute = lambda value: str(value).replace("{profile}", profile).replace(
        "{consumer}", consumer).replace("{package}", str(package))
    provider = recipe["provider"]
    runner.run([substitute(arg) for arg in provider["build"]], out / f"logs/{label}.provider.log")
    destination = out / "providers" / f"{label}.so"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / substitute(provider["artifact"]), destination)
    return {recipe["host_group"]: relative(destination)}, substitute(provider["header"])


def source_withheld(name, packages, providers, headers, out, env):
    """Link again in a distribution tree that contains no producer source files."""
    work = Path(tempfile.mkdtemp(prefix=f"coda-{name}-opaque-", dir=env["PIPELINK_SCRATCH_ROOT"]))
    received = work / "received"
    received.mkdir()
    copied = []
    for index, package in enumerate(packages):
        target = received / f"stage-{index}.plink"
        shutil.copyfile(package, target)
        copied.append(target)
    opaque_providers = {}
    for index, (group, provider) in enumerate(providers.items()):
        target = received / f"owner-{index}.so"
        shutil.copyfile(REPO / provider, target)
        opaque_providers[group] = target.name
    for index, header in enumerate(headers):
        shutil.copyfile(REPO / header, received / f"owner-{index}.h")
    pipeline = generate(name, copied, work / "pipeline.json")
    runner = CommandRecorder(cwd=work, env=env, root=work)
    runner.run([sys.executable, "-m", "pipec2.cli", "link", pipeline, "--out-dir", work / "link"],
               out / "logs" / f"{name}.source-withheld.log")
    distribution = out / "source-withheld" / name
    distribution.parent.mkdir(parents=True, exist_ok=True)
    bundle = shutil.make_archive(str(distribution), "zip", work)
    received_names = [p.name for p in received.iterdir()]
    shutil.rmtree(work)
    return {"bundle": relative(bundle), "provider_files": opaque_providers,
            "received": received_names, "link_status": "PASS", "launch_status": "NOT_RUN"}


def build(profile, *, consumers=("hand", "tilelang"), out=None):
    architecture = PROFILES[profile]
    out = Path(out).resolve() if out else HERE / "out" / profile
    env = {**os.environ, **load(REPO), "CUDA_VISIBLE_DEVICES": ""}
    scratch = Path(env["PIPELINK_SCRATCH_ROOT"]) / "coda" / profile
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "pipec/src"), env.get("PYTHONPATH", "")])
    runner = CommandRecorder(cwd=REPO, env=env, root=REPO, record_path=out / "commands.json")
    recipe_path = HERE / "producers" / profile / "build.json"
    recipe = json.loads(recipe_path.read_text())
    if recipe["profile"] != architecture:
        raise ValueError(f"producer recipe profile mismatch: {recipe_path}")
    source, = recipe["sources"]
    producer = pack({**recipe, "source": source}, out / "packed/producer.plink", architecture,
                    runner, env, out / "logs")
    producer_providers, producer_header = install_provider(
        recipe, "producer", architecture, "", producer["package"], out, runner)
    consumer_owner = json.loads((HERE / "consumer/build.json").read_text())
    index = {"profile": profile, "architecture": architecture, "cells": {}, "natives": {},
             "native_builder": f"campaigns/coda/natives/{profile}/params.py"}
    for name in consumers:
        if name == "hand":
            consumer_recipe = {"source": relative(HERE / "consumer/epilogue.cu"),
                               "includes": consumer_owner.get("includes", []),
                               "defines": {"CODA_PROFILE": profile.removeprefix("sm")},
                               "std": recipe["std"], "compile_flags": recipe["compile_flags"]}
        else:
            destination = out / "tilelang"
            runner.run([env["PIPELINK_TILELANG_PYTHON"], "-m", "pipelink_tilelang", "--kernel",
                        HERE / "consumer-tilelang/swiglu.py", "--profile", architecture, "--out", destination],
                       out / "logs/tilelang.lower.log",
                       env={**env, "PIPELINK_ACC_PARTNER": producer["package"]})
            sources = list((destination / "generation").glob("*/roles.cu"))
            generated, = sources
            consumer_recipe = {"source": relative(generated), "includes": [env["TILELANG_INCLUDE_DIR"],
                                "third_party/cutlass/include", "pipec/src/pipec2/pack/include",
                                *consumer_owner.get("includes", [])],
                               "defines": {}, "std": recipe["std"], "compile_flags": recipe["compile_flags"]}
        consumer = pack(consumer_recipe, out / f"packed/{name}.plink", architecture, runner, env, out / "logs")
        consumer_providers, consumer_header = install_provider(
            consumer_owner, name, architecture, name, consumer["package"], out, runner)
        providers = {**producer_providers, **consumer_providers}
        packages = [Path(producer["package"]), Path(consumer["package"])]
        pipeline = generate(name, packages, out / f"pipelines/{name}.json")
        link_dir = out / "links" / name
        runner.run([sys.executable, "-m", "pipec2.cli", "link", pipeline, "--out-dir", link_dir],
                   out / f"logs/{name}.link.log")
        controls = onetu_control(link_dir, [producer, consumer], profile=architecture, env=env,
                                 extra_includes=[env["TILELANG_INCLUDE_DIR"], env["CUTLASS_INCLUDE_DIR"]],
                                 log_dir=out / "logs" / name)
        manifest = manifest_of(packages[1])
        cell = {"link": relative(link_dir / "link.json"), "linked": relative(link_dir / "pipeline.cubin"),
                **{key: relative(path) for key, path in controls.items()}, "providers": providers,
                "consumer": manifest, "source_withheld": source_withheld(
                    name, packages, providers, [producer_header, consumer_header], out, env)}
        index["cells"][name] = cell
        write_json(out / "index.json", index)
    # Only the native baselines read this sidecar; packing, linking and linked launch ignore it.
    native_records = json.loads((HERE / f"launch-{profile}.json").read_text())["natives"]
    nvcc = Path(env["PIPELINK_CUDA_HOME"]) / "bin/nvcc"
    for kind, record in native_records.items():
        cubin = out / "natives" / f"{kind}.cubin"
        cubin.parent.mkdir(parents=True, exist_ok=True)
        command = [nvcc, f"-std={record['std']}", "-O3", "-DNDEBUG", f"-arch={architecture}",
                   *record["compile_flags"], *[f"-I{REPO / path}" for path in record["includes"]],
                   *[f"-D{key}={value}" for key, value in record["defines"].items()],
                   "--cubin", REPO / record["source"], "-o", cubin]
        runner.run(command, out / f"logs/native-{kind}.log")
        index["natives"][kind] = {**record, "cubin": relative(cubin)}
    standalone = out / "natives/swiglu.cubin"
    runner.run([nvcc, "-std=c++17", "-O3", f"-arch={architecture}", "--cubin",
                HERE / "consumer/swiglu.cu", "-o", standalone], out / "logs/swiglu.log")
    index["standalone"] = {"cubin": relative(standalone), "kernel": "coda_swiglu", "threads": 256,
                            "cluster": [1, 1, 1], "dyn_smem": 0}
    write_json(out / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--consumers", nargs="+", choices=("hand", "tilelang"), default=["hand", "tilelang"])
    parser.add_argument("--out", type=Path, help="build directory (default: campaigns/coda/out/<profile>)")
    args = parser.parse_args()
    print(json.dumps(build(args.profile, consumers=args.consumers, out=args.out), indent=2))
