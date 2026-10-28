"""Native correctness and exploratory placement measurements (run under a clock lease)."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from evaluator.harness.parity.native import NativeSession
from params import arguments
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=16384)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261015)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    shapes = [(256, 512, 64), (256, 512, 320), (256, 1024, 1024), (512, 1024, 5120)] if args.small else [(args.M, 51200, 5120)]
    for M, N, K in shapes:
        x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        w = (torch.randn((N, K), device="cuda", dtype=torch.float32) * 0.02).bfloat16()
        if args.small:
            rows, cols = torch.arange(M, device="cuda"), torch.arange(N // 2, device="cuda")
        else:
            rows = torch.arange(256, device="cuda") % 128 + torch.randint(M // 128, (256,), device="cuda") * 128
            cols = torch.arange(256, device="cuda") % 128 + torch.randint(N // 256, (256,), device="cuda") * 128
        selected = torch.stack((2 * cols, 2 * cols + 1), dim=1).flatten()
        reference = x[rows].double() @ w[selected].double().T
        for kind in ("unfused_gemm", "fused"):
            output = torch.full((M, N if kind == "unfused_gemm" else N // 2), float("nan"), device="cuda", dtype=torch.bfloat16)
            addresses = {"x": x.data_ptr(), "w": w.data_ptr(), "out": output.data_ptr(), "d": output.data_ptr()}
            with NativeSession(ROOT / "campaigns/coda/out/sm_90a/producer/native.cubin", "coda_sm90_" + kind,
                    buffers={}, arguments=lambda _: arguments(kind, addresses, M, N, K),
                    grid=((M // 128) * (N // 256), 1, 1),
                    threads=384, dyn_smem=197632 if kind == "unfused_gemm" else 230400,
                    stream=torch.cuda.current_stream().cuda_stream) as session:
                session.launch()
                session.synchronize()
                if kind == "unfused_gemm":
                    expected = reference.bfloat16()
                    actual = output[rows[:, None], selected[None, :]]
                else:
                    gate, up = reference[:, 0::2], reference[:, 1::2]
                    expected = (gate / (1 + torch.exp(-gate)) * up).bfloat16()
                    actual = output[rows[:, None], cols[None, :]]
                finite = bool(torch.isfinite(output).all())
                correct = torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)
                record = {"shape": [M, N, K], "kind": kind, "allclose": correct, "finite": finite,
                          "max_abs": float((actual.float() - expected.float()).abs().max()), "checked_outputs": actual.numel()}
                if not finite or not correct:
                    print(json.dumps(record), flush=True)
                    raise RuntimeError("Native correctness failure")
                for _ in range(5):
                    session.launch()
                session.synchronize()
                trials = [session.time(20) / 20 for _ in range(7)]
                record.update(ms=statistics.median(trials), trials_ms=trials)
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
