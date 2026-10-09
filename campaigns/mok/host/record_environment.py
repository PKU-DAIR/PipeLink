"""Record the current execution host/toolchain without leaking device ordinals."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-label", default=platform.node())
    parser.add_argument("--check-idle-only", action="store_true")
    args = parser.parse_args()
    selected = os.environ["PIPELINK_GPUS"]
    physical = [int(value) for value in selected.split(",")]
    if len(physical) != 4 or len(set(physical)) != 4:
        raise RuntimeError("The upstream baseline requires four distinct authorized GPUs")
    uuids = set(command("nvidia-smi", "-i", selected, "--query-gpu=uuid", "--format=csv,noheader").splitlines())
    processes = command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")
    if any(line.split(",", 1)[0].strip() in uuids for line in processes.splitlines()):
        raise RuntimeError("An authorized GPU is occupied; do not disturb its existing process")
    if args.check_idle_only:
        print("Authorized GPUs have no compute processes.")
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    import torch

    cuda = Path(os.environ["PIPELINK_CUDA_HOME"])
    logs = Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "mok/logs"
    logs.mkdir(parents=True, exist_ok=True)
    report = {
        "execution_host": args.host_label,
        "python": platform.python_version(),
        "uv": command("uv", "--version"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_git_revision": torch.version.git_version,
        "cuda_bindings": importlib.metadata.version("cuda-bindings"),
        "nccl_package": importlib.metadata.version("nvidia-nccl-cu13"),
        "nvcc": command(str(cuda / "bin/nvcc"), "--version"),
        "ptxas": command(str(cuda / "bin/ptxas"), "--version"),
        "compute_sanitizer": command(str(cuda / "bin/compute-sanitizer"), "--version"),
        "host_compiler": command("g++", "--version").splitlines()[0],
        "driver": sorted(set(command("nvidia-smi", "-i", selected, "--query-gpu=driver_version", "--format=csv,noheader").splitlines())),
        "visible_device_count": torch.cuda.device_count(),
        "devices": [
            {"rank": rank, "architecture": "Blackwell/sm103",
             "compute_capability": list(torch.cuda.get_device_capability(rank)),
             "total_memory_bytes": torch.cuda.get_device_properties(rank).total_memory,
             "sm_count": torch.cuda.get_device_properties(rank).multi_processor_count}
            for rank in range(torch.cuda.device_count())
        ],
    }
    topology = command("nvidia-smi", "topo", "-m")
    rows = {line.split()[0]: line.split()[1:] for line in topology.splitlines()
            if line.startswith("GPU") and " X " in line.replace("\t", " ")}
    links = [[rows[f"GPU{source}"][target] for target in physical] for source in physical]
    (logs / "toolchain.json").write_text(json.dumps(report, indent=2) + "\n")
    (logs / "topology.json").write_text(json.dumps({"logical_rank_order": list(range(4)), "links": links}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
