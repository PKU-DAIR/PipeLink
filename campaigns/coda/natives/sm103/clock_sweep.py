"""Find the highest clean supported SM clock for sustained CODA work."""
import argparse
from contextlib import ExitStack
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"]="1"
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from campaigns.coda.natives.sm103.measure import Telemetry
from campaigns.coda.natives.sm103.params import arguments
from evaluator.harness.parity.native import NativeSession
from tools.pipelink_env import load
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true",help="qualify public sequences and the fused native")
    parser.add_argument("--max-clock",type=int,default=1665)
    args=parser.parse_args()
    kind="fused"
    suffix="-all" if args.all else ""
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.manual_seed(20261015)
    M,N,K=16384,51200,5120
    x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
    w=(torch.randn(N,K,device="cuda")*.02).to(torch.bfloat16)
    d=torch.empty(M,N,device="cuda",dtype=torch.bfloat16)
    out=torch.empty(M,N//2,device="cuda",dtype=torch.bfloat16)
    addresses={k:v.data_ptr() for k,v in dict(x=x,w=w,d=d,out=out).items()}
    here=Path(__file__).resolve().parent
    rows=[]
    lock=[sys.executable,str(ROOT/"tools/gpulockd.py")]
    def lease(op,clock):
        p=subprocess.run(lock+[op,"--gpus","1"],env={**os.environ,"PIPELINK_LOCK_SM_MHZ":str(clock)},capture_output=True,text=True,cwd=ROOT)
        print(op,clock,p.returncode,p.stdout,p.stderr,flush=True)
        return p
    probe=Telemetry(1)
    count=C.c_uint(0)
    probe.lib.nvmlDeviceGetSupportedMemoryClocks(probe.device,C.byref(count),None)
    memory=(C.c_uint*count.value)()
    probe.lib.nvmlDeviceGetSupportedMemoryClocks(probe.device,C.byref(count),memory)
    count=C.c_uint(0)
    probe.lib.nvmlDeviceGetSupportedGraphicsClocks(probe.device,max(memory),C.byref(count),None)
    clocks=(C.c_uint*count.value)()
    probe.lib.nvmlDeviceGetSupportedGraphicsClocks(probe.device,max(memory),C.byref(count),clocks)
    supported=sorted(set(c for c in clocks if 900<=c<=1665),reverse=True)
    (here/"out/supported-clocks.json").write_text(json.dumps(supported))
    with ExitStack() as lifetime:
        launches={}
        for native_kind in (kind,):
            grid=(M//128*(N//256),1,1)
            session=lifetime.enter_context(NativeSession(here/"out/native.cubin","coda_sm103_"+native_kind,buffers={},
                arguments=lambda _,k=native_kind:arguments(k,addresses,M,N,K),grid=grid,
                threads=256,dyn_smem=99328,cluster=(2,1,1),device=0,
                stream=torch.cuda.current_stream().cuda_stream))
            def native_launch(s=session):
                s._launches[0][0].stream=torch.cuda.current_stream().cuda_stream
                s.launch()
            launches[native_kind]=native_launch
        if args.all:
            from campaigns.coda.public.run import sequences
            public=sequences(x,w,d,out)
            launches.update({name:public[name] for name in ("gemm_interleaved","compile","liger")})
        else:
            native=launches[kind]
            def pair():
                torch.mm(x,w.T,out=d)
                native()
            launches={"gemm+"+kind:pair}
        graphs={}
        for name,launch in launches.items():
            launch()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):launch()
            graphs[name]=graph
        def test(clock):
            passes=[]
            for run in range(2):
                start=lease("start",clock)
                if start.returncode: raise RuntimeError(start.stderr)
                telemetry=Telemetry(1)
                telemetry.start()
                try:
                    begin=time.monotonic();batches=0;workloads={}
                    for name,graph in graphs.items():
                        work_begin=time.monotonic();count=0
                        while time.monotonic()-work_begin<(15 if args.all else 60):
                            graph.replay();torch.cuda.synchronize();count+=1
                        elapsed=time.monotonic()-work_begin
                        workloads[name]={"seconds":elapsed,"launches":count*10,"wall_ms":elapsed*1000/(count*10)}
                        batches+=count
                    duration=time.monotonic()-begin
                    health=lease("status",clock)
                    marker=json.loads((Path(load(ROOT)["PIPELINK_LOCK_STATE_DIR"])/"gpu1.json").read_text())
                finally:
                    samples=telemetry.stop()
                    lease("stop",clock)
                row=dict(clock=clock,run=run,seconds=duration,pairs=batches*10,workloads=workloads,
                    pair_ms=duration*1000/(batches*10),healthy=health.returncode==0,
                    violations=marker["violations"],clock_min=min(s[0] for s in samples),
                    clock_max=max(s[0] for s in samples),power_max=max(s[1] for s in samples),
                    reasons=sorted(set(s[2] for s in samples)),samples=samples)
                rows.append(row)
                (here/"out"/f"clock-sweep{suffix}.json").write_text(json.dumps(rows,indent=2)+"\n")
                print(json.dumps({k:v for k,v in row.items() if k!="samples"}),flush=True)
                passes.append(row["healthy"])
            return all(passes)
        last_failed=None
        for target in range(args.max_clock,899,-75):
            clock=max(c for c in supported if c<=target)
            if test(clock):
                winner=clock
                if last_failed is not None:
                    # Refine the coarse bracket to the highest supported clean setting.
                    choices=sorted(c for c in supported if winner<c<last_failed)
                    while choices:
                        candidate=choices[len(choices)//2]
                        if test(candidate):
                            winner=candidate;choices=[c for c in choices if c>candidate]
                        else: choices=[c for c in choices if c<candidate]
                print("SELECTED_CLOCK",winner,flush=True)
                (here/"out"/f"selected-clock{suffix}.json").write_text(json.dumps({"sm_mhz":winner,"criterion":"two 60-second runs, zero daemon violations"})+"\n")
                return
            last_failed=clock
        raise RuntimeError("No clean clock in supported sweep")

if __name__=="__main__":main()
