#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PIPEC = REPO / "pipec"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipec" / "src"))

from tools.stagebuild import CommandRecorder, manifest_of, pack_cu, write_json

CUTLASS = REPO / "third_party" / "cutlass"
HOPPER = REPO / "third_party" / "flash-attention" / "hopper"
# The comm owner's shared device code (push_gather.cuh) used by kv-gather.
COMM = REPO / "examples" / "comm-overlap" / "comm"
SPECS = {"d128-sm90-c": "ATTENTION_BUILD_D128_SM90_C"}
PROFILE = "sm_90a"
# Stages, controls and slot are compiled and code-generated with fast math, like upstream FA3.
SM90_DEFINES = ("-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DCUTLASS_ENABLE_GDC_FOR_SM90")
SM90_FLAGS = ("--use_fast_math", *SM90_DEFINES)
STAGES = ("tma-load", "core", "epilogue", "empty", "mask-default", "mask-window",
          "score-identity", "score-softcap", "block-load", "mask-phi",
          "msa-load", "epilogue-partial", "cp-load", "kv-gather")
SOURCE_BY_STAGE = {stage: f"sm90/stage_{stage.replace('-', '_')}.cu" for stage in STAGES}
STAGES_BY_SPEC = {spec: STAGES for spec in SPECS}
PIPELINES_BY_SPEC = {
    spec: ("selected", "window", "softcap", "window-softcap", "phi", "msa", "cp") for spec in SPECS
}
STEPS = ("pack", "freeze", "link", "controls", "slot", "stock")


class BuildError(RuntimeError):
    pass


def command_record(command: list[str], cwd: Path, log: Path, env: dict[str, str]) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    log.write_text(completed.stdout, encoding="utf-8")
    record = {
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "log": str(log.relative_to(HERE)),
    }
    if completed.returncode != 0:
        raise BuildError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"log: {log}\n{completed.stdout[-4000:]}"
        )
    return record


def pool_run(items: list[tuple[list[str], Path, Path]], jobs: int,
             env: dict[str, str]) -> list[dict[str, Any]]:
    if not items:
        return []
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(jobs, len(items))) as pool:
        futures = [pool.submit(command_record, command, cwd, log, env)
                   for command, cwd, log in items]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda item: item["log"])


def common_flags() -> list[str]:
    return [
        "-std=c++17", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr",
        "--expt-extended-lambda", "--threads=4",
        "-Isrc", "-I../../third_party/cutlass/include",
        "-I../../third_party/cutlass/tools/util/include",
        f"-I{PIPEC / 'src' / 'pipec2' / 'pack' / 'include'}",
        "-I../../third_party/flash-attention/hopper", "-I../../examples/comm-overlap/comm",
        *SM90_FLAGS,
    ]


def pack_stage(spec: str, stage: str, env: dict[str, str]) -> list[dict[str, Any]]:
    output = HERE / "out" / "packaged" / spec / f"{stage}.plink"
    recorder = CommandRecorder(cwd=REPO, env=env, root=HERE)
    seed = "Attention" + hashlib.sha256(f"{spec}:{stage}".encode()).hexdigest()[:16]
    report = pack_cu(
        recorder, HERE / "src" / SOURCE_BY_STAGE[stage], output,
        includes=[HERE / "src", CUTLASS / "include", CUTLASS / "tools" / "util" / "include", HOPPER, COMM],
        defines={SPECS[spec]: "1", "__CUDA_ARCH_FEAT_SM90_ALL": "1"},
        nvcc_flags=["--expt-relaxed-constexpr", "--expt-extended-lambda", f"--frandom-seed={seed}",
                    *SM90_FLAGS],
        profile=PROFILE,
        log=HERE / "out" / "logs" / f"pack-{spec}-{stage}.log", env=env,
    )
    write_json(HERE / "out" / "manifests" / spec / f"{stage}.json", manifest_of(output))
    write_json(HERE / "out" / "pack-reports" / spec / f"{stage}.json", report)
    return recorder.records


def selected_configs(value: str, parser: argparse.ArgumentParser) -> list[str]:
    configs = list(SPECS) if value == "all" else [item for item in value.split(",") if item]
    unknown = sorted(set(configs) - set(SPECS))
    if unknown:
        parser.error("unknown configs: " + ",".join(unknown))
    if not configs:
        parser.error("at least one config is required")
    return configs


def selected_stages(value: str, configs: list[str],
                    parser: argparse.ArgumentParser) -> set[str] | None:
    if value == "all":
        return None
    stages = {item for item in value.split(",") if item}
    known = {stage for spec in configs for stage in STAGES_BY_SPEC[spec]}
    unknown = sorted(stages - known)
    if unknown:
        parser.error("unknown stages for selected configs: " + ",".join(unknown))
    if not stages:
        parser.error("at least one stage is required")
    return stages


def publish_artifacts(configs: list[str], stages: set[str] | None = None) -> None:
    """Publish selected packages; retain every other package as it is."""
    for spec in configs:
        for stage in STAGES_BY_SPEC[spec]:
            if stages is not None and stage not in stages:
                continue
            source = HERE / "out" / "packaged" / spec / f"{stage}.plink"
            target = HERE / "frozen" / spec / f"{stage}.plink"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def control_source(spec: str, pipeline: str) -> str:
    declaration = json.loads(
        (HERE / "pipelines" / spec / f"{pipeline}.json").read_text(encoding="utf-8"))
    sources = [SOURCE_BY_STAGE[stage["stage_id"].rsplit(".", 1)[1]]
               for stage in declaration["pipeline"]["stages"]]
    return "\n".join(f'#include "{source}"'
                     for source in ("binding_stub.cu", *sources, "dispatch.cu")) + "\n"


def require_sm90_inputs(configs: list[str], first: int, stages: set[str] | None) -> None:
    """P2a can build the stock adapter without P3 family packages or pipelines."""
    for spec in configs:
        if first <= STEPS.index("pack"):
            required = [HERE / "src" / SOURCE_BY_STAGE[stage]
                        for stage in STAGES_BY_SPEC[spec] if stages is None or stage in stages]
        elif first <= STEPS.index("controls"):
            required = [HERE / "pipelines" / spec / f"{pipeline}.json"
                        for pipeline in PIPELINES_BY_SPEC[spec]]
        else:
            continue
        missing = [str(path.relative_to(HERE)) for path in required if not path.is_file()]
        if missing:
            raise BuildError(
                f"{spec}: P3 family inputs are not available: {', '.join(missing)}; "
                "use --from slot or --from stock for the P2a stock baseline")


def build_stock(cuda: Path, env: dict[str, str], jobs: int) -> list[dict[str, Any]]:
    """Build single-entry FA3 images and record the actual cubin entry symbols."""
    source = Path("src/sm90/stock/fa3_hdim128_bf16.cu")
    # Upstream FA3's own build flags include --use_fast_math.
    flags = [
        "-arch=sm_90a", "-O3", "-std=c++17", "--use_fast_math", *SM90_DEFINES, "-DNDEBUG",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-Isrc", "-I../../third_party/cutlass/include",
        "-I../../third_party/cutlass/tools/util/include",
        "-I../../third_party/flash-attention/hopper",
    ]
    images = []
    items = []
    for causal, is_local in ((True, False), (False, True)):
        name = f"fa3-hdim128-bf16-{'c' if causal else 'local'}-cluster1"
        output = HERE / "out" / "stock" / "sm90" / f"{name}.cubin"
        output.parent.mkdir(parents=True, exist_ok=True)
        image_flags = [*flags, f"-DFA3_STOCK_CAUSAL={int(causal)}",
                       "-DFA3_STOCK_CLUSTER_M=1", f"-DFA3_STOCK_IS_LOCAL={int(is_local)}"]
        command = [str(cuda / "bin" / "nvcc"), "-cubin", *image_flags, str(source),
                   "-o", str(output)]
        items.append((command, HERE, HERE / "out" / "logs" / f"stock-{name}-compile.log"))
        images.append((output, causal, is_local, image_flags))
    records = pool_run(items, jobs, env)
    for output, causal, is_local, image_flags in images:
        log = HERE / "out" / "logs" / f"stock-{output.stem}-symbols.log"
        records.append(command_record(
            [str(cuda / "bin" / "cuobjdump"), "--dump-elf-symbols", str(output)],
            HERE, log, env))
        # cuobjdump marks kernel ELF symbols with STT_FUNC and STO_ENTRY; helper
        # device functions must not be mistaken for launchable kernel entries.
        symbols = [line.split()[-1] for line in log.read_text().splitlines()
                   if "STT_FUNC" in line and "STO_ENTRY" in line]
        if len(symbols) != 1 or "device_kernel" not in symbols[0]:
            raise BuildError(f"{output}: expected one FA3 device_kernel entry, got {symbols}")
        write_json(output.with_suffix(".json"), {
            "schema": "pipelink-attention-stock-image-v1",
            "kernel_symbol": symbols[0], "causal": causal, "cluster_m": 1,
            "dtype": "bf16", "is_local": is_local, "head_dim": 128, "source": str(source),
            "flags": image_flags,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="all")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    parser.add_argument("--from", dest="from_step", choices=STEPS, default="pack")
    parser.add_argument(
        "--stages", default="all",
        help="comma-separated stage suffixes to pack/freeze; retain other frozen packages "
             "(link, controls and slot remain unfiltered)")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    configs = selected_configs(args.configs, parser)
    stages = selected_stages(args.stages, configs, parser)

    cuda_text = os.environ.get("PIPELINK_CUDA_HOME")
    if not cuda_text:
        raise BuildError("PIPELINK_CUDA_HOME is not set; source tools/env.sh")
    cuda = Path(cuda_text).resolve()
    nvcc = cuda / "bin" / "nvcc"
    if not nvcc.is_file():
        raise BuildError(f"nvcc is missing under PIPELINK_CUDA_HOME: {nvcc}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PATH"] = str(cuda / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(cuda / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    for directory in ("out/logs", "out/manifests", "out/packaged",
                      "out/pack-reports", "out/links", "out/controls", "out/bin", "frozen"):
        (HERE / directory).mkdir(parents=True, exist_ok=True)

    first = STEPS.index(args.from_step)
    require_sm90_inputs(configs, first, stages)
    commands: list[dict[str, Any]] = []
    started = time.monotonic()
    flags = common_flags()
    generation = uuid4().hex
    report = {
        "schema": "pipelink-attention-build-v1",
        "build_generation": generation,
        "configs": configs,
        "jobs": args.jobs,
        "from_step": args.from_step,
        "stages": sorted(stages) if stages is not None else "all",
        "gpu_executed": False,
    }
    # Invalidate prior sanitizer evidence before any build output can change,
    # including when this invocation later fails.
    write_json(HERE / "out" / "build-report.json", report)


    if first <= STEPS.index("pack"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(pack_stage, spec, stage, env)
                       for spec in configs for stage in STAGES_BY_SPEC[spec]
                       if stages is None or stage in stages]
            for future in concurrent.futures.as_completed(futures):
                commands.extend(future.result())

    if first <= STEPS.index("freeze"):
        publish_artifacts(configs, stages)

    if first <= STEPS.index("link"):
        items = []
        for spec in configs:
            for pipeline_name in PIPELINES_BY_SPEC[spec]:
                output = HERE / "out" / "links" / spec / pipeline_name
                output.mkdir(parents=True, exist_ok=True)
                command = [
                    "uv", "run", "--project", str(PIPEC), "pipec2", "link",
                    str(HERE / "pipelines" / spec / f"{pipeline_name}.json"),
                    "--out-dir", str(output),
                ]
                items.append((command, REPO, HERE / "out" / "logs" / f"link-{spec}-{pipeline_name}.log"))
        commands.extend(pool_run(items, args.jobs, env))

    if first <= STEPS.index("controls"):
        items = []
        for spec in configs:
            for pipeline_name in PIPELINES_BY_SPEC[spec]:
                link_dir = HERE / "out" / "links" / spec / pipeline_name
                generated = link_dir / "generated"
                output = HERE / "out" / "controls" / spec / f"{pipeline_name}.o"
                output.parent.mkdir(parents=True, exist_ok=True)
                source = output.with_name(f"{pipeline_name}-control.cu")
                source.write_text(control_source(spec, pipeline_name), encoding="utf-8")
                seed = "AttentionControl" + hashlib.sha256(
                    f"{spec}:{pipeline_name}".encode()).hexdigest()[:16]
                # Reuse the exact TU, bindings and seed; only the core seam is omitted.
                for suffix in ("", "-no-score-seam"):
                    control_name = f"{pipeline_name}{suffix}"
                    # Isolate nvcc's --objdir-as-tempdir intermediates for the same TU.
                    control_dir = output.parent / suffix.removeprefix("-")
                    control_dir.mkdir(parents=True, exist_ok=True)
                    control_object = control_dir / f"{control_name}.o"
                    # Match pack_cu's architecture-specific LTO target.
                    command = [
                        str(nvcc), *flags, "--objdir-as-tempdir",
                        "-rdc=true", "-dc", f"-arch={PROFILE.replace('sm_', 'lto_')}",
                        "-D__CUDA_ARCH_FEAT_SM90_ALL=1",
                        f"-D{SPECS[spec]}=1", f"--frandom-seed={seed}",
                        *(["-DPIPELINK_SM90_NO_SCORE_SEAM"] if suffix else []),
                        f"-I{generated.relative_to(HERE)}", str(source.relative_to(HERE)),
                        "-o", str(control_object.relative_to(HERE)),
                    ]
                    items.append((command, HERE, HERE / "out" / "logs" /
                                  f"control-{spec}-{control_name}-compile.log"))
        commands.extend(pool_run(items, args.jobs, env))
        items = []
        for spec in configs:
            for pipeline_name in PIPELINES_BY_SPEC[spec]:
                link_result = json.loads(
                    (HERE / "out" / "links" / spec / pipeline_name /
                     "link.json").read_text(encoding="utf-8"))
                for suffix in ("", "-no-score-seam"):
                    control_name = f"{pipeline_name}{suffix}"
                    control_dir = HERE / "out" / "controls" / spec
                    control_object = (control_dir / suffix.removeprefix("-") /
                                      f"{control_name}.o")
                    output = control_dir / f"{control_name}.cubin"
                    command = [
                        sys.executable, str(HERE / "link_control.py"),
                        str(control_object), str(output),
                        "--maxrregcount", str(link_result["regs_cap"]),
                    ]
                    items.append((command, HERE, HERE / "out" / "logs" /
                                  f"control-{spec}-{control_name}-link.log"))
        commands.extend(pool_run(items, args.jobs, env))

    if first <= STEPS.index("slot"):
        items = []
        for name, source in (("attention-slot-sm90", "slot_sm90.cu"),
                             ("attention-slot-cp-sm90", "slot_cp_sm90.cu")):
            command = [
                str(nvcc), *flags, "-arch=sm_90a", f"--frandom-seed={name}",
                str(HERE / "src" / source),
                f"-L{cuda / 'lib64'}", f"-L{cuda / 'lib64' / 'stubs'}", "-lcuda",
                "-o", str(HERE / "out" / "bin" / name),
            ]
            items.append((command, HERE, HERE / "out" / "logs" / f"{name}-compile.log"))
        commands.extend(pool_run(items, args.jobs, env))

    if first <= STEPS.index("stock"):
        commands.extend(build_stock(cuda, env, args.jobs))

    elapsed = time.monotonic() - started
    report.update(wall_seconds=elapsed, commands=commands)
    write_json(HERE / "out" / "build-report.json", report)
    print(json.dumps({"configs": configs, "wall_seconds": elapsed,
                      "commands": len(commands)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        print(f"attention build: {error}", file=sys.stderr)
        raise SystemExit(2)
