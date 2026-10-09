#!/usr/bin/env python3
"""Check callable outputs, measure same-source parity, and sanitize both images."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from params import COUNT, build
from pipec2.profile import PROFILES
from tools.pipelink_env import load

GRID = (1024, 1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--gpu", type=int, help="physical GPU index; default: first PIPELINK_GPUS entry")
    parser.add_argument("--single", choices=("linked", "one-tu"),
                        help="one launch plus oracle check (compute-sanitizer subprocess)")
    args = parser.parse_args()
    config = load(REPO)
    gpu = args.gpu
    if gpu is None:
        if not config["PIPELINK_GPUS"]:
            parser.error("set PIPELINK_GPUS in .env or pass --gpu")
        gpu = int(config["PIPELINK_GPUS"].split(",")[0])
    scratch = Path(config["PIPELINK_SCRATCH_ROOT"]) / "hello-callable" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(config)
    os.environ["TMPDIR"] = str(scratch)
    # Select the physical GPU before CUDA initializes, including in sanitizer children.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from pipec2.launcher import open_hostless, run_hostless
    from evaluator.harness.parity.example import compare, sanitize, sass_diagnostic, time_pair

    out = HERE / "out" / args.profile
    link = out / "link" / "link.json"
    controls = json.loads((out / "controls.json").read_text())
    images = {"linked": out / "link" / "pipeline.cubin", "one-tu": out / controls["one_tu"]}
    # 64 MiB per array makes this a memory-throughput workload, not an empty-launch benchmark.
    inputs = np.random.default_rng(2026).uniform(-1.0, 1.0, COUNT).astype(np.float32)
    expected = inputs * np.float32(2.0)
    buffers = {"input": inputs, "output": np.zeros_like(inputs)}
    launch_args = {"buffers": buffers, "params": build, "grid": GRID, "device": 0}

    def launch(image: str) -> np.ndarray:
        return run_hostless(link, **launch_args, cubin=images[image], outputs=("output",))["output"]

    if args.single:
        check = compare(launch(args.single), expected, rtol=0.0, atol=0.0)
        print(json.dumps({"image": args.single, "oracle": check}))
        return 0 if check["bitwise"] else 1

    actual = launch("linked")
    control = launch("one-tu")
    numeric = {
        "linked_oracle": compare(actual, expected, rtol=0.0, atol=0.0),
        "one_tu_oracle": compare(control, expected, rtol=0.0, atol=0.0),
        "linked_vs_one_tu": compare(actual, control, rtol=0.0, atol=0.0),
    }
    passed = all(check["bitwise"] for check in numeric.values())
    results = {"profile": args.profile, "gpu": gpu, "count": COUNT, "grid": list(GRID),
               "numeric": numeric}
    if passed:
        with open_hostless(link, **launch_args, cubin=images["linked"]) as linked_session, \
                linked_session.with_image(images["one-tu"]) as control_session:
            results["timing"] = time_pair(linked_session, control_session, repetitions=20, gpu=gpu)
        passed &= results["timing"]["status"] == "PASS"
    else:
        results["timing"] = {"status": "NOT_RUN_NUMERICAL_FAILURE"}

    # Always run both tools on both images; --single prevents recursive sanitization.
    sanitizer = []
    for image in images:
        reports = sanitize(
            [sys.executable, str(HERE / "run.py"), "--profile", args.profile,
             "--gpu", str(gpu), "--single", image],
            env=dict(os.environ), log_dir=out / "sanitize" / image,
        )
        sanitizer.extend({"image": image, **report} for report in reports)
    results["sanitize"] = sanitizer
    passed &= all(report["status"] == "PASS" for report in sanitizer)
    results["sass"] = sass_diagnostic(
        images["linked"], images["one-tu"], env=dict(os.environ), out_dir=out / "sass",
        backend_matched=out / controls["one_tu_lto"],
    )
    results["status"] = "PASS" if passed else "FAIL"
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
