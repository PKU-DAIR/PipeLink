"""Native correctness matrix and leased-clock measurements (CUDA torch required)."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from campaigns.coda.natives.sm103.params import arguments
from evaluator.harness.parity.native import NativeSession
from tools.pipelink_env import load

def foreign_pids(gpu):
    output=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid",
        "--format=csv,noheader,nounits"],text=True)
    owned={os.getpid()}
    marker=Path(load(ROOT)["PIPELINK_LOCK_STATE_DIR"])/f"gpu{gpu}.json"
    if marker.exists():
        owned.add(json.loads(marker.read_text())["pid"])
    return {int(line) for line in output.splitlines() if line.strip() and int(line) not in owned}

class Telemetry:
    def __init__(self,gpu):
        self.lib=C.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlInit_v2()
        self.device=C.c_void_p()
        self.lib.nvmlDeviceGetHandleByIndex_v2(gpu,C.byref(self.device))
        self.samples=[]
        self.stop_event=threading.Event()
    def loop(self):
        while not self.stop_event.is_set():
            clock,power,reasons=C.c_uint(),C.c_uint(),C.c_uint64()
            self.lib.nvmlDeviceGetClockInfo(self.device,1,C.byref(clock))
            self.lib.nvmlDeviceGetPowerUsage(self.device,C.byref(power))
            self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.device,C.byref(reasons))
            self.samples.append([clock.value,power.value/1000,reasons.value])
            self.stop_event.wait(.005)
    def start(self):
        self.thread=threading.Thread(target=self.loop);self.thread.start()
    def stop(self):
        self.stop_event.set();self.thread.join()
        return self.samples


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int,default=1)
    parser.add_argument("--small",action="store_true")
    parser.add_argument("--kind",choices=["fused","unfused_gemm"])
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.manual_seed(20261015)
    here=Path(__file__).resolve().parent
    records=json.loads((ROOT/"campaigns/coda/launch-sm103.json").read_text())["natives"]
    shapes=[(256,512,64),(256,512,320),(256,1024,1024),(512,1024,5120)]
    if not args.small: shapes += [(4096,51200,5120),(16384,51200,5120)]
    result=[]
    for M,N,K in shapes:
        x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
        w=(torch.randn(N,K,device="cuda")*.02).to(torch.bfloat16)
        out=torch.empty(M,N//2,device="cuda",dtype=torch.bfloat16)
        d=torch.empty(M,N,device="cuda",dtype=torch.bfloat16)
        addresses={k:v.data_ptr() for k,v in {"x":x,"w":w,"out":out,"d":d}.items()}
        full=M>512
        if full:
            gen=torch.Generator().manual_seed(12345)
            # Four random grid positions for every tile-local output coordinate.
            residues=torch.arange(16384).repeat(4)
            rows=(torch.randint(M//128,(65536,),generator=gen)*128+residues//128).cuda()
            cols=(torch.randint((N//2)//128,(65536,),generator=gen)*128+residues%128).cuda()
            dots=[]
            for start in range(0,len(rows),512):
                r,c=rows[start:start+512],cols[start:start+512]
                a=x[r].double()
                dots.append(torch.stack([(a*w[2*c].double()).sum(1),(a*w[2*c+1].double()).sum(1)],1))
            ref=torch.cat(dots)
        else:
            ref=x.double()@w.double().T
        fused=(torch.nn.functional.silu(ref[...,0::2])*ref[...,1::2]).to(torch.bfloat16)
        for kind in ([args.kind] if args.kind else records):
            rec=records[kind]
            dest=d if kind=="unfused_gemm" else out
            selected=rec["stagger_ns"]
            for stagger in ([0,selected] if full else [selected]):
                dest.fill_(float("nan"))
                grid=eval(rec["grid"],{},dict(M=M,N=N,num_sms=torch.cuda.get_device_properties(0).multi_processor_count))
                with NativeSession(here/"out/native.cubin",rec["kernel"],buffers={},
                    arguments=lambda _:arguments(kind,addresses,M,N,K,stagger),grid=grid,
                    threads=256,dyn_smem=rec["dyn_smem"],cluster=(2,1,1),device=0,
                    stream=torch.cuda.current_stream().cuda_stream) as session:
                    def launch():
                        session._launches[0][0].stream=torch.cuda.current_stream().cuda_stream
                        session.launch()
                    launch();torch.cuda.synchronize()
                    finite=bool(torch.isfinite(dest).all())
                    if full:
                        actual=torch.stack([d[rows,2*cols],d[rows,2*cols+1]],1) if kind=="unfused_gemm" else out[rows,cols,None]
                    else: actual=dest
                    expected=ref.to(torch.bfloat16) if kind=="unfused_gemm" else fused
                    close=bool(torch.allclose(actual.float(),expected.float(),rtol=.01,atol=.01))
                    row=dict(shape=[M,N,K],kind=kind,stagger_ns=stagger,finite=finite,allclose=close,
                        max_abs=float((actual.float()-expected.float()).abs().max()),samples=actual.numel())
                    print(json.dumps(row),flush=True)
                    if not finite or not close: raise RuntimeError(row)
                    if full:
                        for _ in range(3):launch()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(10):launch()
                        times=[]
                        foreign=foreign_pids(args.gpu)
                        telemetry=Telemetry(args.gpu);telemetry.start()
                        for _ in range(7):
                            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                            begin.record();graph.replay();graph.replay();end.record();end.synchronize()
                            times.append(begin.elapsed_time(end)/20)
                        samples=telemetry.stop()
                        foreign.update(foreign_pids(args.gpu))
                        row.update(ms=statistics.median(times),trials_ms=times,telemetry=samples,
                            clock_summary={"min_mhz":min(s[0] for s in samples),
                                "median_mhz":statistics.median(s[0] for s in samples),
                                "powercap_samples":sum(bool(s[2]&4) for s in samples),
                                "samples":len(samples)},foreign_pids=sorted(foreign))
                        if foreign: raise RuntimeError(f"Foreign CUDA processes: {sorted(foreign)}")
                        print(json.dumps(row),flush=True)
                    result.append(row)
        del x,w,out,d,ref,fused
    (here/"out"/("small.json" if args.small else "measurements.json")).write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":main()
