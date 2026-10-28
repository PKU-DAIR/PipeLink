#!/usr/bin/env python3
"""Native correctness and locked-clock producer-quality measurements."""
import argparse
import ctypes as ct
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from tools.pipelink_env import load
from evaluator.harness.parity.example import clock_lease, time_pair
from evaluator.harness.parity.native import NativeSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--clock", type=int, default=2017)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--unlocked", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **load(REPO), "PIPELINK_LOCK_SM_MHZ": str(args.clock)}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(20261015)
    torch.cuda.init()
    device = torch.cuda.get_device_properties(0)
    reference_library = ct.CDLL(str(HERE / "out/reference.so"))
    gate_reference = reference_library.coda_sm120_gate_reference
    gate_reference.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
    gate_reference.restype = ct.c_int
    report = {"gpu": args.gpu, "device": device.name,
              "clock_requested_mhz": None if args.unlocked else args.clock,
              "mode": "unlocked; device power-capped below every lockable clock (≥1200 MHz)" if args.unlocked else "locked",
              "torch": torch.__version__, "rows": [], "telemetry": []}
    stop = threading.Event()
    phase = {"name": "correctness"}
    owned_pids = {os.getpid()}
    def telemetry():
        while not stop.is_set():
            sampled_phase = phase["name"]
            p = subprocess.run(["nvidia-smi",
                "--query-gpu=index,clocks.sm,clocks.mem,power.draw,utilization.gpu,clocks_event_reasons.active",
                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            all_gpus = p.stdout.strip().splitlines()
            selected = next((line.split(",",1)[1].strip() for line in all_gpus
                             if int(line.split(",",1)[0]) == args.gpu), "")
            processes = subprocess.run(["nvidia-smi", "-i", str(args.gpu),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            pids = [int(line.strip()) for line in processes.stdout.splitlines() if line.strip().isdigit()]
            report["telemetry"].append({"time": time.time(), "sample": selected, "all_gpus": all_gpus,
                "phase": sampled_phase if phase["name"] == sampled_phase else "transition",
                "compute_pids": pids, "foreign_pids": [pid for pid in pids if pid not in owned_pids]})
            stop.wait(0.5)
    def timed(fn, label):
        phase["name"] = label
        for _ in range(3): fn()
        torch.cuda.synchronize()
        values = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10): fn()
            end.record(); end.synchronize()
            values.append(start.elapsed_time(end) / 10)
        phase["name"] = "correctness"
        return {"ms": statistics.median(values), "trials_ms": values, "telemetry_phase": label}
    class Timer:
        def __init__(self, fn): self.launch = fn
        def time(self, n):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n): self.launch()
            end.record(); end.synchronize()
            return start.elapsed_time(end)
    def paired(session, reference, label):
        phase["name"] = label
        pair = time_pair(session, Timer(reference), repetitions=10, gpu=args.gpu, sample_clocks=True)
        pair["clock_locked"] = False
        pair["mode"] = report["mode"]
        phase["name"] = "correctness"
        if pair["status"] != "PASS": raise RuntimeError(f"Paired timing rejected: {pair['status']}")
        def arm_summary(arm):
            durations = [ms/pair["repetitions"] for block in pair["blocks"]
                         for a,ms in zip(block["order"],block["milliseconds"]) if a == arm]
            telemetry = pair["telemetry"][arm]
            return {"ms":statistics.median(durations), "clock_samples":telemetry["samples"],
                    "sm_clock_min_mhz":telemetry["clock_min_mhz"],
                    "sm_clock_median_mhz":telemetry["clock_median_mhz"],
                    "sw_power_cap_samples":telemetry["sw_power_cap_samples"],
                    "foreign_pids":pair["foreign_pids"]}
        native, baseline = arm_summary("A"), arm_summary("B")
        native["paired_placement"] = pair
        return native, baseline
    shapes = [(256,512,64), (256,512,320), (256,1024,1024), (512,1024,5120)]
    if not args.small_only: shapes += [(4096,51200,5120), (16384,51200,5120)]
    worker = threading.Thread(target=telemetry)
    try:
        lease_context = (nullcontext({"mode":"unlocked","healthy":None,"stopped":True}) if args.unlocked
                         else clock_lease(args.gpu, env=env, log_dir=args.out / "clock"))
        with lease_context as lease:
            if not args.unlocked:
                marker = Path(env["PIPELINK_LOCK_STATE_DIR"]).expanduser()/f"gpu{args.gpu}.json"
                report["lease_marker_start"] = json.loads(marker.read_text())
                owned_pids.add(report["lease_marker_start"]["pid"])
            worker.start()
            for M,N,K in shapes:
                a = torch.randn((M,K), device="cuda", dtype=torch.bfloat16)
                b = (torch.randn((N,K), device="cuda", dtype=torch.float32) * .02).bfloat16()
                d = torch.empty((M,N), device="cuda", dtype=torch.bfloat16)
                out = torch.empty((M,N//2), device="cuda", dtype=torch.bfloat16)
                if M <= 512:
                    ref = a.double() @ b.double().T
                    rows = cols = None
                else:
                    count = 65536
                    gen = torch.Generator(device="cuda").manual_seed(20261015 + M)
                    index = torch.arange(count, device="cuda")
                    rows = torch.randint(M//128, (count,), device="cuda", generator=gen)*128 + index%128
                    cols = torch.randint(N//256, (count,), device="cuda", generator=gen)*128 + (index//128)%128
                    ref = torch.empty((count,2), device="cuda", dtype=torch.float64)
                    for base in range(0,count,512):
                        rr, cc = rows[base:base+512], cols[base:base+512]
                        aa = a[rr].double()
                        ref[base:base+512,0] = (aa*b[cc*2].double()).sum(1)
                        ref[base:base+512,1] = (aa*b[cc*2+1].double()).sum(1)
                gate, up = (ref[:,0::2],ref[:,1::2]) if rows is None else (ref[:,0],ref[:,1])
                fused_ref = (gate/(1+torch.exp(-gate))*up).bfloat16()
                rounded = ref.bfloat16().double()
                rg, ru = (rounded[:,0::2],rounded[:,1::2]) if rows is None else (rounded[:,0],rounded[:,1])
                unfused_ref = (rg/(1+torch.exp(-rg))*ru).bfloat16()
                semantic = (fused_ref.float()-unfused_ref.float()).abs()
                row = {"M":M,"N":N,"K":K,"semantic_rounding_max_abs":semantic.max().item(),
                       "semantic_rounding_mean_abs":semantic.mean().item(), "kernels":{}}
                def gate_launch():
                    status = gate_reference(a.data_ptr(),b.data_ptr(),d.data_ptr(),M,N,K,
                                            torch.cuda.current_stream().cuda_stream)
                    if status: raise RuntimeError(f"CUTLASS reference status {status}")
                paired_references = {}
                specs = [("gemm","coda_sm120_gemm",128,256),
                         ("fused","coda_sm120_fused",128,256),
                         ("gemm_256x128","coda_sm120_gemm_256x128",256,128)]
                for name,symbol,tm,tn in specs:
                    target = d if name.startswith("gemm") else out
                    target.fill_(float("nan"))
                    values = [ct.c_void_p(a.data_ptr()),ct.c_void_p(b.data_ptr()),ct.c_void_p(target.data_ptr()),
                              ct.c_int(M),ct.c_int(N),ct.c_int(K)]
                    grid = (M//tm)*(N//tn)
                    with NativeSession(HERE/"out/kernels.cubin",symbol,buffers={},arguments=lambda _:values,
                            grid=(grid,1,1),threads=256,dyn_smem=73728,
                            stream=torch.cuda.current_stream().cuda_stream) as session:
                        session.launch(); torch.cuda.synchronize()
                        finite = bool(torch.isfinite(target).all())
                        actual = target if rows is None else (target[rows[:,None],cols[:,None]*2+torch.arange(2,device="cuda")]
                                  if name.startswith("gemm") else target[rows,cols])
                        expected = ref.bfloat16() if name.startswith("gemm") else fused_ref
                        correct = bool(torch.allclose(actual,expected,rtol=.01,atol=.01))
                        error = (actual.float()-expected.float()).abs().max().item()
                        row["kernels"][name] = {"allclose":correct,"finite":finite,"max_abs":error}
                        if not correct or not finite: raise RuntimeError(f"{name} {M,N,K}: {row['kernels'][name]}")
                        if M > 512:
                            if args.unlocked:
                                reference = gate_launch if name == "gemm" else lambda:torch.mm(a,b.T,out=d)
                                measured, baseline = paired(session, reference, f"{M}/{name}")
                                measured["paired_reference"] = "gate_256x128" if name == "gemm" else "cublas"
                                row["kernels"][name].update(measured)
                                if name == "gemm": paired_references["gate_256x128"] = baseline
                                if name == "fused": paired_references["cublas"] = baseline
                            else:
                                row["kernels"][name].update(timed(session.launch, f"{M}/{name}"))
                d.fill_(float("nan"))
                torch.mm(a,b.T,out=d)
                actual = d if rows is None else d[rows[:,None],cols[:,None]*2+torch.arange(2,device="cuda")]
                ok = bool(torch.allclose(actual,ref.bfloat16(),rtol=.01,atol=.01))
                finite = bool(torch.isfinite(d).all())
                if not ok or not finite: raise RuntimeError("cuBLAS correctness failed")
                row["kernels"]["cublas"] = {"allclose":ok,"finite":finite}
                if M > 512:
                    row["kernels"]["cublas"].update(paired_references["cublas"] if args.unlocked else
                                                  timed(lambda:torch.mm(a,b.T,out=d), f"{M}/cublas"))
                d.fill_(float("nan"))
                gate_launch()
                actual = d if rows is None else d[rows[:,None],cols[:,None]*2+torch.arange(2,device="cuda")]
                ok = bool(torch.allclose(actual,ref.bfloat16(),rtol=.01,atol=.01))
                finite = bool(torch.isfinite(d).all())
                if not ok or not finite: raise RuntimeError("CUTLASS reference correctness failed")
                row["kernels"]["gate_256x128"] = {"allclose":ok,"finite":finite}
                if M > 512:
                    row["kernels"]["gate_256x128"].update(paired_references["gate_256x128"] if args.unlocked else
                                                        timed(gate_launch, f"{M}/gate_256x128"))
                    row["producer_speedup_over_256x128"] = (row["kernels"]["gemm"]["paired_placement"]["ratio_onetu_over_linked"]
                        if args.unlocked else row["kernels"]["gate_256x128"]["ms"]/row["kernels"]["gemm"]["ms"])
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
                del a,b,d,out,ref,rounded,gate,up,rg,ru,fused_ref,unfused_ref,semantic,actual,expected,target
                gc.collect(); torch.cuda.empty_cache()
            report["lease"] = lease
            if not args.unlocked: report["lease_marker_end"] = json.loads(marker.read_text())
    finally:
        stop.set()
        if worker.is_alive(): worker.join()
        report["lease"] = locals().get("lease")
        for row in report["rows"]:
            for kernel in row["kernels"].values():
                if "telemetry_phase" not in kernel: continue
                samples = [s for s in report["telemetry"] if s["phase"] == kernel["telemetry_phase"] and s["sample"]]
                fields = [s["sample"].split(",") for s in samples]
                clocks = [int(f[0]) for f in fields]
                kernel["clock_samples"] = len(clocks)
                kernel["sm_clock_min_mhz"] = min(clocks) if clocks else None
                kernel["sm_clock_median_mhz"] = statistics.median(clocks) if clocks else None
                kernel["sw_power_cap_samples"] = sum(bool(int(f[4],16)&4) for f in fields)
                kernel["foreign_pids"] = sorted({pid for s in samples for pid in s["foreign_pids"]})
        report["foreign_pids"] = sorted({pid for s in report["telemetry"] for pid in s["foreign_pids"]})
        (args.out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"lease":report["lease"],"results":str(args.out/"results.json")}),flush=True)


if __name__ == "__main__": main()
