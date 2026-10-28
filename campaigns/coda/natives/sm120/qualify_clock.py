#!/usr/bin/env python3
"""Descending supported-clock qualification of the full CODA sm120 workload."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from tools.pipelink_env import load
from evaluator.harness.parity.example import clock_lease
from evaluator.harness.parity.native import NativeSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--max-clock", type=int, default=2017)
    parser.add_argument("--min-clock", type=int, default=1200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **load(REPO)}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(20261015)
    M,N,K = 16384,51200,5120
    a = torch.randn((M,K),device="cuda",dtype=torch.bfloat16)
    b = (torch.randn((N,K),device="cuda")*.02).bfloat16()
    d = torch.empty((M,N),device="cuda",dtype=torch.bfloat16)
    out = torch.empty((M,N//2),device="cuda",dtype=torch.bfloat16)
    values = [ct.c_void_p(a.data_ptr()),ct.c_void_p(b.data_ptr()),ct.c_void_p(out.data_ptr()),
              ct.c_int(M),ct.c_int(N),ct.c_int(K)]
    raw = subprocess.run(["nvidia-smi","-i",str(args.gpu),"--query-supported-clocks=gr",
                          "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout
    clocks = sorted({int(line.strip()) for line in raw.splitlines()
                     if args.min_clock <= int(line.strip()) <= args.max_clock},reverse=True)
    report = {"gpu":args.gpu,"shape":[M,N,K],"protocol":"two independent healthy leases; cuBLAS and fused, each >=60 event seconds; reject early on daemon violation", "clocks":[]}
    (args.out/"supported-clocks.txt").write_text(raw)
    def save(): (args.out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    tool = [sys.executable,str(REPO/"tools/gpulockd.py")]
    with NativeSession(HERE/"out/kernels.cubin","coda_sm120_fused",buffers={},arguments=lambda _:values,
            grid=(M//128*(N//256),1,1),threads=256,dyn_smem=73728,
            stream=torch.cuda.current_stream().cuda_stream) as session:
        for clock in clocks:
            candidate = {"mhz":clock,"runs":[]}
            report["clocks"].append(candidate)
            for run in range(2):
                directory = args.out/f"{clock}-r{run+1}"
                directory.mkdir(exist_ok=True)
                entry = {"phases":[],"telemetry":[]}
                candidate["runs"].append(entry)
                stop = threading.Event()
                phase = {"name": "setup"}
                owned_pids = {os.getpid()}
                def telemetry():
                    while not stop.is_set():
                        p = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,clocks.sm,clocks.mem,power.draw,utilization.gpu,clocks_event_reasons.active",
                            "--format=csv,noheader,nounits"],capture_output=True,text=True)
                        all_gpus = p.stdout.strip().splitlines()
                        selected = next((line.split(",",1)[1].strip() for line in all_gpus
                                         if int(line.split(",",1)[0]) == args.gpu), "")
                        processes = subprocess.run(["nvidia-smi","-i",str(args.gpu),
                            "--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True)
                        pids = [int(line.strip()) for line in processes.stdout.splitlines() if line.strip().isdigit()]
                        entry["telemetry"].append({"time":time.time(),"sample":selected,"all_gpus":all_gpus,
                            "phase":phase["name"],"compute_pids":pids,"foreign_pids":[pid for pid in pids if pid not in owned_pids]})
                        stop.wait(.5)
                worker = threading.Thread(target=telemetry)
                try:
                    with clock_lease(args.gpu,env={**env,"PIPELINK_LOCK_SM_MHZ":str(clock)},log_dir=directory) as lease:
                        marker = Path(env["PIPELINK_LOCK_STATE_DIR"]).expanduser()/f"gpu{args.gpu}.json"
                        entry["lease_marker_start"] = json.loads(marker.read_text())
                        owned_pids.add(entry["lease_marker_start"]["pid"])
                        worker.start()
                        failed = False
                        for name,launch in [("cublas",lambda:torch.mm(a,b.T,out=d)),
                                            ("fused",session.launch)]:
                            phase["name"] = name
                            elapsed, iterations, last_status = 0.0,0,0.0
                            while elapsed < 60000:
                                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(50): launch()
                                end.record(); end.synchronize()
                                elapsed += start.elapsed_time(end); iterations += 50
                                if elapsed-last_status >= 10000:
                                    status = subprocess.run([*tool,"status","--gpus",str(args.gpu)],env=env,capture_output=True,text=True)
                                    with (directory/"sustained-status.log").open("a") as log:
                                        log.write(status.stdout+status.stderr)
                                    last_status = elapsed
                                    if status.returncode:
                                        failed = True
                                        break
                            entry["phases"].append({"name":name,"milliseconds":elapsed,"iterations":iterations,"mean_ms":elapsed/iterations})
                            if failed: break
                        entry["complete"] = not failed and len(entry["phases"]) == 3
                        entry["lease_marker_end"] = json.loads(marker.read_text())
                finally:
                    stop.set()
                    if worker.is_alive(): worker.join()
                    entry["foreign_pids"] = sorted({pid for s in entry["telemetry"] for pid in s["foreign_pids"]})
                    if entry["foreign_pids"]: entry["complete"] = False
                    entry["lease"] = locals().get("lease")
                    save()
                print(json.dumps({"mhz":clock,"run":run+1,"lease":entry["lease"],"phases":entry["phases"]}),flush=True)
                if not entry["complete"] or not entry["lease"]["healthy"]: break
            if len(candidate["runs"]) == 2 and all(r["complete"] and r["lease"]["healthy"] for r in candidate["runs"]):
                report["selected_mhz"] = clock
                save()
                print(json.dumps({"selected_mhz":clock}),flush=True)
                return
    raise RuntimeError("No supported clock admitted")


if __name__ == "__main__": main()
