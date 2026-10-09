#!/usr/bin/env python3
"""Check Hopper/sm90 GEMM host ABI sessions, CUDA graphs, and source-free launch."""
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

SHAPE = (256, 384, 512)
ALPHAS = (1.25, -0.75)
REPEATS = 2


class _OracleAdapter:
    """Expose caller-owned resource loans to the existing cuBLAS oracle."""

    def __init__(self, backend, resources):
        self.addresses = {name: resource.address for name, resource in resources.items()}
        self.synchronize = backend.synchronize


def _providers(out):
    return {"gemm.owner.params": out / "owner/provider.so",
            "gemm.user.params": out / "consumer/user.so"}


def _scalars(alpha):
    return dict(zip(("M", "N", "K"), SHAPE), alpha=alpha)


def _specs(inputs):
    from pipec2.host.profile import DTYPES

    M, N, K = SHAPE
    return {
        "a": {"allocation_bytes": inputs["a"].nbytes, "extents": (M, K),
              "bit_strides": (K * 16, 16), "dtype_id": DTYPES["bf16"]["id"], "alignment": 128},
        "b": {"allocation_bytes": inputs["b"].nbytes, "extents": (N, K),
              "bit_strides": (K * 16, 16), "dtype_id": DTYPES["bf16"]["id"], "alignment": 128},
        "output": {"allocation_bytes": M * N * 4, "extents": (M, N),
                   "bit_strides": (N * 32, 32), "dtype_id": DTYPES["f32"]["id"], "alignment": 128},
    }


def _check(actual, expected, baseline=None):
    from evaluator.harness.parity.example import compare
    from oracle import RTOL, ATOL

    numeric = compare(actual, expected, rtol=RTOL, atol=ATOL)
    record = {"cublas_scaled": numeric}
    passed = numeric["allclose"]
    if baseline is not None:
        record["reused_session"] = compare(actual, baseline, rtol=0, atol=0)
        passed &= record["reused_session"]["bitwise"]
    record["status"] = "PASS" if passed else "FAIL"
    return record


def _run_image(out, result, image, inputs, config):
    import numpy as np
    from pipec2.host.broker import open_host
    from pipec2.host.cuda import CudaBackend
    from pipec2.host.fixture import Fixture
    from oracle import Cublas

    images = {"linked": "pipeline.cubin", "one-tu": "one-tu.cubin"}
    report = {"image": image, "cases": []}
    outputs, expected_outputs = {}, {}
    cluster = tuple(result.cluster)
    poison = np.full(SHAPE[:2], np.nan, dtype=np.float32).tobytes()
    with CudaBackend(out / "link" / images[image], result.kernel_symbol, device=0,
                     dynamic_smem_bytes=result.dyn_smem_bytes, cluster=cluster,
                     block=(result.threads, 1, 1)) as backend, Fixture(backend) as fixture:
        specs = _specs(inputs)
        for name, spec in specs.items():
            fixture.tensor(name, data=poison if name == "output" else inputs[name].tobytes(), **spec)
        # Oracle storage is caller-owned too, but is not a declared provider resource.
        oracle = fixture.tensor("oracle", data=None, **specs["output"])
        loans = {name: fixture.resources[name] for name in specs}
        # cuBLAS binds the current context; make it this backend's device context.
        with backend.current(), Cublas(_OracleAdapter(backend, fixture.resources), SHAPE,
                                       config["PIPELINK_CUDA_HOME"]) as blas:
            blas.launch()
            backend.synchronize()
            product = np.frombuffer(backend.readback(oracle.address, oracle.allocation_bytes),
                                    dtype=np.float32).reshape(SHAPE[:2])
            report["oracle"] = {"library": blas.library_name, "version": blas.version,
                                "operation": "float32 cuBLAS(A_bf16 @ B_bf16.T), then float32 alpha"}
        report["identity"] = dict(backend.identity)
        stream = backend.stream_token
        output = loans["output"]

        def launch_readback(launch):
            # Poison/readback belong to the caller, never to the capture body.
            backend.upload(output.address, poison)
            launch(stream)
            backend.synchronize()
            return np.frombuffer(backend.readback(output.address, output.allocation_bytes),
                                 dtype=np.float32).reshape(SHAPE[:2])

        for alpha in ALPHAS:
            expected = product * np.float32(alpha)
            expected_outputs[alpha] = expected
            case = {"alpha": alpha, "direct": [], "graph": [], "fresh": []}
            scalars = _scalars(alpha)

            def open_session():
                return open_host(result, _providers(out), {}, loans, scalars, backend, cluster)

            # Scalars are immutable session inputs. Reopen for each alpha, rather
            # than mutating a pinned params image or building inside CUDA capture.
            with open_session() as session:
                case["geometry"] = dict(session.geometry)
                case["transfer_reports"] = session.transfer_reports
                case["geometry_status"] = "PASS" if (
                    tuple(session.geometry["grid"]) == (SHAPE[0] // 128, SHAPE[1] // 128, 1)
                    and tuple(session.geometry["block"]) == (320, 1, 1)
                    and session.geometry["launch_required"]
                ) else "FAIL"
                baseline = None
                for _ in range(REPEATS):
                    actual = launch_readback(session.launch)
                    case["direct"].append(_check(actual, expected, baseline))
                    if baseline is None:
                        baseline = actual
                outputs[alpha] = baseline
                with backend.capture(stream, lambda: session.launch(stream)) as graph:
                    for _ in range(REPEATS):
                        actual = launch_readback(graph.launch)
                        case["graph"].append(_check(actual, expected, baseline))
            for _ in range(REPEATS):
                with open_session() as fresh:
                    actual = launch_readback(fresh.launch)
                    check = _check(actual, expected, baseline)
                    check["geometry"] = dict(fresh.geometry)
                    case["fresh"].append(check)
            case["status"] = "PASS" if case["geometry_status"] == "PASS" and all(
                check["status"] == "PASS" for route in ("direct", "graph", "fresh")
                for check in case[route]
            ) else "FAIL"
            report["cases"].append(case)
    report["status"] = "PASS" if all(case["status"] == "PASS" for case in report["cases"]) else "FAIL"
    return report, outputs, expected_outputs


def _prepare_cli(out, inputs):
    import numpy as np

    directory = out / "cli"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("a", "b"):
        (directory / f"{name}.bin").write_bytes(inputs[name].tobytes())
    (directory / "poison.bin").write_bytes(np.full(SHAPE[:2], np.nan, dtype=np.float32).tobytes())
    invocations = []
    for index, alpha in enumerate(ALPHAS):
        resources = {}
        output = directory / f"output-{index}.bin"
        output.unlink(missing_ok=True)
        for name, spec in _specs(inputs).items():
            resources[name] = {**spec, "data": "poison.bin" if name == "output" else f"{name}.bin",
                               "origin_byte_offset": 0, "origin_bit_offset": 0}
        resources["output"]["output"] = output.name
        invocation = directory / f"host-inputs-{index}.json"
        invocation.write_text(json.dumps({"scalars": _scalars(alpha), "resources": resources}, indent=2) + "\n")
        invocations.append((alpha, output, invocation))
    return invocations


def _cli(out, inputs, expected, baseline):
    import numpy as np

    reports = []
    for index, (alpha, output, invocation) in enumerate(_prepare_cli(out, inputs)):
        # This is the installed pipec2 CLI entry point, not a custom launcher.
        command = [sys.executable, "-m", "pipec2.cli", "launch", str(out / "link/link.json")]
        for group, provider in _providers(out).items():
            command.extend(("--provider", f"{group}={provider}"))
        command.extend(("--host-inputs", str(invocation), "--device", "0"))
        completed = subprocess.run(command, env=dict(os.environ), text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log = out / "cli" / f"launch-{index}.log"
        log.write_text(completed.stdout)
        record = {"alpha": alpha, "returncode": completed.returncode,
                  "host_inputs": str(invocation.relative_to(out)), "log": str(log.relative_to(out)),
                  "status": "FAIL"}
        if completed.returncode == 0:
            actual = np.frombuffer(output.read_bytes(), dtype=np.float32).reshape(SHAPE[:2])
            record.update(_check(actual, expected[alpha], baseline[alpha]))
        reports.append(record)
    return {"cases": reports, "status": "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("sm_90a",), required=True)
    parser.add_argument("--gpu", type=int, help="physical GPU index; default: first PIPELINK_GPUS entry")
    parser.add_argument("--artifacts", type=Path, help="consumer workspace root containing owner/, consumer/, link/")
    parser.add_argument("--prepare-inputs", action="store_true",
                        help="write caller fixture files and host-inputs JSON without launching")
    parser.add_argument("--single", action="store_true",
                        help="linked persistent-session/graph checks only; no sanitizer recursion")
    args = parser.parse_args()
    from tools.pipelink_env import load

    config = load(REPO)
    gpu = args.gpu
    if gpu is None:
        if not config["PIPELINK_GPUS"]:
            parser.error("set PIPELINK_GPUS in .env or pass --gpu")
        gpu = int(config["PIPELINK_GPUS"].split(",")[0])
    if gpu < 0:
        parser.error("--gpu must be nonnegative")
    if not config["PIPELINK_CUDA_HOME"]:
        parser.error("set PIPELINK_CUDA_HOME in .env")
    os.environ.update(config)
    # Set physical selection before importing or constructing any CUDA consumer.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    scratch = Path(config["PIPELINK_SCRATCH_ROOT"]) / "hopper-gemm-split" / f"host-{args.profile}"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["CUDA_CACHE_PATH"] = str(scratch / "cuda-cache")

    from pipec2.model import LinkResult
    from evaluator.harness.parity.example import compare, sanitize, sass_diagnostic
    from oracle import _inputs, RTOL, ATOL

    out = args.artifacts.resolve() if args.artifacts else HERE / "out" / f"host-{args.profile}"
    result = LinkResult.from_json_obj(json.loads((out / "link/link.json").read_text()))
    inputs = _inputs(SHAPE)
    if args.prepare_inputs:
        files = [str(path) for _, _, path in _prepare_cli(out, inputs)]
        print(json.dumps({"shape": SHAPE, "host_inputs": files}))
        return 0
    if args.single:
        report, _, _ = _run_image(out, result, "linked", inputs, config)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1

    results = {
        "profile": args.profile, "gpu": gpu, "device_ordinal": 0, "host_abi": "v3",
        "shape": list(SHAPE), "tile": [128, 128, 64], "scalar_cases": list(ALPHAS),
        "rtol": RTOL, "atol": ATOL, "output_poison": "NaN before every direct launch and graph replay",
        "session_policy": "two direct launches and two graph replays per immutable scalar case; two fresh sessions per case",
        "scalar_update_policy": "close/open and provider build before capture for each alpha",
        "stages": {"gemm.load": {"warp": 8}, "gemm.mma": {"warps": list(range(8))},
                   "gemm.epilogue.scaled-store": {"warp": 9}},
    }
    # Both tools run unconditionally, even if a preceding tool reports failure.
    results["sanitize"] = sanitize(
        [sys.executable, str(HERE / "run_host.py"), "--profile", args.profile,
         "--gpu", str(gpu), "--artifacts", str(out), "--single"],
        env=dict(os.environ), log_dir=out / "sanitize/linked",
    )
    for record in results["sanitize"]:
        record["log"] = str(Path(record["log"]).relative_to(out))
    passed = all(record["status"] == "PASS" for record in results["sanitize"])
    image_outputs, expected = {}, None
    for image in ("linked", "one-tu"):
        try:
            report, image_outputs[image], reference = _run_image(out, result, image, inputs, config)
            if image == "linked":
                expected = reference
            results[image] = report
        except Exception as exc:
            results[image] = {"status": "FAIL", "error": str(exc)}
        passed &= results[image]["status"] == "PASS"
    if len(image_outputs) == 2:
        checks = [{"alpha": alpha, **compare(image_outputs["linked"][alpha], image_outputs["one-tu"][alpha],
                                             rtol=0, atol=0)} for alpha in ALPHAS]
        results["linked_vs_one_tu"] = {"cases": checks,
            "status": "PASS" if all(check["bitwise"] for check in checks) else "FAIL"}
        passed &= results["linked_vs_one_tu"]["status"] == "PASS"
    if expected is not None:
        try:
            results["cli"] = _cli(out, inputs, expected, image_outputs["linked"])
        except Exception as exc:
            results["cli"] = {"status": "FAIL", "error": str(exc)}
    else:
        results["cli"] = {"status": "FAIL", "error": "linked oracle/session prerequisite failed"}
    passed &= results["cli"]["status"] == "PASS"
    try:
        results["sass"] = sass_diagnostic(
            out / "link/pipeline.cubin", out / "link/one-tu.cubin", env=dict(os.environ),
            out_dir=out / "sass", backend_matched=out / "link/one-tu-lto.cubin",
        )
        results["sass"]["comparison"] = str(Path(results["sass"]["comparison"]).relative_to(out))
    except Exception as exc:
        results["sass"] = {"status": "FAIL", "error": str(exc), "diagnostic_only": True}
        passed = False
    results["status"] = "PASS" if passed else "FAIL"
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
