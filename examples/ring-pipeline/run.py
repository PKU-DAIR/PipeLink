#!/usr/bin/env python3
"""Exercise both bound ring depths, compare outputs, time parity, and sanitize."""
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

from params import BIAS, COLUMNS, ROWS, SCALE, build
from pipec2.profile import PROFILES
from tools.pipelink_env import load

GRID = (ROWS // 128, 1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--gpu", type=int, help="physical GPU index; default: first PIPELINK_GPUS entry")
    parser.add_argument("--single", choices=("linked", "one-tu"),
                        help="one launch plus oracle check (compute-sanitizer subprocess)")
    parser.add_argument("--depth", type=int, choices=(2, 4), help="depth for --single only")
    args = parser.parse_args()
    if bool(args.single) != (args.depth is not None):
        parser.error("--single and --depth must be supplied together")
    config = load(REPO)
    gpu = args.gpu
    if gpu is None:
        if not config["PIPELINK_GPUS"]:
            parser.error("set PIPELINK_GPUS in .env or pass --gpu")
        gpu = int(config["PIPELINK_GPUS"].split(",")[0])
    scratch = Path(config["PIPELINK_SCRATCH_ROOT"]) / "ring-pipeline" / args.profile
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(config)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from pipec2.launcher import open_hostless, run_hostless
    from evaluator.harness.parity.example import compare, sanitize, sass_diagnostic, time_pair

    out = HERE / "out" / args.profile
    # These dyadic inputs and scale/bias are exact in bf16/f32, including fused FMA.
    values = np.random.default_rng(2026).integers(-128, 128, (ROWS, COLUMNS), dtype=np.int16).astype(np.float32)
    values *= np.float32(1 / 32)
    inputs = (values.view(np.uint32) >> 16).astype(np.uint16)
    expected = np.maximum(values * np.float32(SCALE) + np.float32(BIAS), np.float32(0))
    buffers = {"input": inputs, "output": np.full_like(expected, np.nan)}
    launch_args = {"buffers": buffers, "params": build, "grid": GRID, "device": 0}
    results = {"profile": args.profile, "gpu": gpu, "shape": [ROWS, COLUMNS],
               "grid": list(GRID), "tiles_per_cta": COLUMNS // 64, "depths": []}
    passed = True
    for depth in ((args.depth,) if args.single else (2, 4)):
        link_dir = out / f"depth-{depth}"
        link = link_dir / "link.json"
        controls = json.loads((link_dir / "controls.json").read_text())
        images = {"linked": link_dir / "pipeline.cubin", "one-tu": out / controls["one_tu"]}

        def launch(image: str) -> np.ndarray:
            return run_hostless(link, **launch_args, cubin=images[image], outputs=("output",))["output"]

        if args.single:
            check = compare(launch(args.single), expected, rtol=0.0, atol=0.0)
            print(json.dumps({"depth": depth, "image": args.single, "oracle": check}))
            return 0 if check["bitwise"] else 1

        actual, control = launch("linked"), launch("one-tu")
        numeric = {
            "linked_oracle": compare(actual, expected, rtol=0.0, atol=0.0),
            "one_tu_oracle": compare(control, expected, rtol=0.0, atol=0.0),
            "linked_vs_one_tu": compare(actual, control, rtol=0.0, atol=0.0),
        }
        depth_passed = all(check["bitwise"] for check in numeric.values())
        record = {"depth": depth, "numeric": numeric}
        if depth_passed:
            with open_hostless(link, **launch_args, cubin=images["linked"]) as linked_session, \
                    linked_session.with_image(images["one-tu"]) as control_session:
                record["timing"] = time_pair(linked_session, control_session, repetitions=20, gpu=gpu)
            depth_passed &= record["timing"]["status"] == "PASS"
        else:
            record["timing"] = {"status": "NOT_RUN_NUMERICAL_FAILURE"}

        sanitizer = []
        for image in images:
            reports = sanitize(
                [sys.executable, str(HERE / "run.py"), "--profile", args.profile,
                 "--gpu", str(gpu), "--single", image, "--depth", str(depth)],
                env=dict(os.environ), log_dir=link_dir / "sanitize" / image,
            )
            sanitizer.extend({"image": image, **report} for report in reports)
        record["sanitize"] = sanitizer
        depth_passed &= all(report["status"] == "PASS" for report in sanitizer)
        record["sass"] = sass_diagnostic(
            images["linked"], images["one-tu"], env=dict(os.environ), out_dir=link_dir / "sass",
            backend_matched=out / controls["one_tu_lto"],
        )
        ratio = record["timing"].get("ratio_linked_over_onetu")
        record["phase_b_parity"] = ratio is not None and 0.97 <= ratio <= 1.03
        depth_passed &= record["phase_b_parity"]
        record["status"] = "PASS" if depth_passed else "FAIL"
        results["depths"].append(record)
        passed &= depth_passed
    results["status"] = "PASS" if passed else "FAIL"
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
