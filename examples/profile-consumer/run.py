#!/usr/bin/env python3
"""Run the host ABI v3 and one-TU paths, preserve Gate A, and sanitize both images."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--gpu", type=int, help="physical GPU index; default: first PIPELINK_GPUS entry")
    args = parser.parse_args()
    config = load(ROOT)
    gpu = args.gpu
    if gpu is None:
        if not config["PIPELINK_GPUS"]:
            parser.error("set PIPELINK_GPUS in .env or pass --gpu")
        gpu = int(config["PIPELINK_GPUS"].split(",")[0])
    scratch = Path(config["PIPELINK_SCRATCH_ROOT"]) / "profile-consumer" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(config)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env = dict(os.environ)
    from evaluator.harness.parity.example import sanitize, sass_diagnostic

    build = HERE / "build" / args.profile
    logs = build / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results_path = build / "results.json"
    results_path.unlink(missing_ok=True)

    def command(argv: list[str | Path], log: str, *, check: bool = True) -> int:
        result = subprocess.run(list(map(str, argv)), cwd=ROOT, env=env,
                                capture_output=True, text=True)
        (logs / log).write_text(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError(f"Command failed; see {logs / log}\n{result.stderr}")
        return result.returncode

    command([sys.executable, HERE / "prepare_inputs.py", build], "prepare-inputs.log")
    linked = [sys.executable, "-m", "pipec2.cli", "launch", str(build / "linked/link.json"),
              "--provider", f"org.pipelink.profile-consumer.params={build / 'provider.so'}",
              "--host-inputs", str(build / "host-inputs.json"),
              "--blob", f"org.pipelink.profile-consumer.1-instruction={build / 'instruction1.params.bin'}",
              "--blob", f"org.pipelink.profile-consumer.2-instruction={build / 'instruction2.params.bin'}",
              "--device", "0", "--cluster", "1,1,1"]
    one_tu = [str(build / "runner"), str(build / "one_tu.cubin"),
              "org_pipelink_profile_consumer_one_tu_kernel", "192", "0", "64", "128", "512",
              str(build / "one_tu.output.bin"), str(build / "one_tu.input.bin")]
    launches = {
        "linked": command(linked, "linked-runtime.log", check=False),
        "one_tu": command(one_tu, "one-tu-runtime.log", check=False),
    }

    # Both tools run on both actual launch routes, even after a numerical failure.
    sanitizer = []
    for image, argv in (("linked", linked), ("one-tu", one_tu)):
        sanitizer.extend({"image": image, **report} for report in sanitize(
            argv, env=env, log_dir=build / "sanitize" / image,
        ))
    cuobjdump = Path(config["PIPELINK_CUDA_HOME"]) / "bin/cuobjdump"
    for name, image in (("linked", build / "linked/pipeline.cubin"),
                        ("one-tu", build / "one_tu.cubin")):
        command([cuobjdump, "--dump-resource-usage", image], f"{name}.resources.txt")
        command([cuobjdump, "--dump-sass", image], f"{name}.sass")
    command([sys.executable, HERE / "collect_results.py", "--profile", args.profile],
            "collect-results.log", check=False)
    results = json.loads(results_path.read_text())
    results["sanitize"] = sanitizer
    results["launch_returncodes"] = launches
    results["sass"] = sass_diagnostic(
        build / "linked/pipeline.cubin", build / "one_tu.cubin", env=env, out_dir=build / "sass",
    )
    passed = (results["status"] == "PASS" and all(code == 0 for code in launches.values())
              and all(report["status"] == "PASS" for report in sanitizer))
    results["status"] = "PASS" if passed else "FAIL"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"profile": args.profile, "gpu": gpu, "gate_a": results["gate_a"],
                      "status": results["status"], "sanitize": sanitizer,
                      "sass": results["sass"], "results": str(results_path)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
