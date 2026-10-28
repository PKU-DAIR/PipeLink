#!/usr/bin/env python3
"""Standalone SwiGLU correctness and locked CUDA-graph bandwidth measurement."""
from pathlib import Path
import argparse
import ctypes as C
import json
import os
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from tools.pipelink_env import load
from evaluator.harness.parity.example import clock_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    env = {**os.environ, **load(ROOT)}
    os.environ.update(env)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def copy_kernel(src, dst, count: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(dst + index, tl.load(src + index, index < count), index < count)

    out = HERE / "out/benchmark"
    out.mkdir(parents=True, exist_ok=True)
    nvcc = Path(env["PIPELINK_CUDA_HOME"]) / "bin/nvcc"
    library = out / "swiglu.so"
    subprocess.run([str(nvcc), "-O3", "-std=c++20", "-arch=sm_103a", "-shared", "-Xcompiler=-fPIC",
                    str(HERE / "benchmark_launch.cu"), "-o", str(library)], check=True)
    lib = C.CDLL(str(library))
    launch = lib.coda_swiglu_launch
    launch.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p]
    launch.restype = C.c_int

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                fn()
        samples = []
        for _ in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 60)
        return {"ms": statistics.median(samples), "samples_ms": samples}

    def expression(x):
        gate, up = x[:, 0::2].float(), x[:, 1::2].float()
        return (gate / (1.0 + torch.exp(-gate)) * up).to(torch.bfloat16)

    compiled = torch.compile(expression, fullgraph=True)
    report = {"torch": torch.__version__, "triton": triton.__version__, "launch": {
        "threads": 256, "outputs_per_thread": 4, "grid": "ceil(M*N/2048)", "dyn_smem": 0}, "shapes": {}}
    torch.manual_seed(20261015)
    with clock_lease(args.gpu, env=env, log_dir=out / "clock") as lease:
        for M in (4096, 16384):
            N = 51200
            x = torch.randn((M, N), device="cuda", dtype=torch.bfloat16)
            y = torch.empty((M, N // 2), device="cuda", dtype=torch.bfloat16)
            copied = torch.empty_like(x)

            def ours():
                error = launch(x.data_ptr(), y.data_ptr(), M, N, torch.cuda.current_stream().cuda_stream)
                if error:
                    raise RuntimeError(f"CUDA launch error {error}")

            ours()
            actual_compiled = compiled(x)
            # Check every output against f64 arithmetic, without allocating full f64 activations.
            maximum = 0.0
            for row in range(0, M, 32):
                gate = x[row:row+32, 0::2].double()
                up = x[row:row+32, 1::2].double()
                expected = (gate / (1 + torch.exp(-gate)) * up).to(torch.bfloat16)
                torch.testing.assert_close(y[row:row+32], expected, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(actual_compiled[row:row+32], expected, rtol=1e-2, atol=1e-2)
                maximum = max(maximum, (y[row:row+32].float() - expected.float()).abs().max().item())
            result = {"numeric": "PASS", "max_abs": maximum,
                      "cuda": timed(ours), "torch_compile": timed(lambda: compiled(x)),
                      "copy": timed(lambda: copy_kernel[(triton.cdiv(x.numel(), 4096),)](x, copied, x.numel(), 4096))}
            for name in ("cuda", "torch_compile", "copy"):
                traffic = x.numel() * (4 if name == "copy" else 3)
                result[name]["GB_s"] = traffic / result[name]["ms"] / 1e6
            for name in ("cuda", "torch_compile"):
                result[name]["copy_fraction"] = result[name]["GB_s"] / result["copy"]["GB_s"]
            report["shapes"][str(M)] = result
            print(json.dumps({"M": M, **result}), flush=True)
            del x, y, copied, actual_compiled
    report["clock"] = lease
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if not lease["healthy"]:
        raise RuntimeError("Unhealthy clock lease")


if __name__ == "__main__":
    main()
