"""Persistent F/B/R sessions with upstream scheduling, epilogues and references."""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import dataclass
import math
import torch
import torch.distributed as dist
from mok import functional, ops, _C
from tests.utils import (BF16_TOLERANCE, MXFP8_TOLERANCE, generate_inputs,
                         run_reference_bf16, run_forward_reference_bf16,
                         run_fwd_epilogue_reference)
from resources import validate_inventory


@dataclass(frozen=True)
class Shape:
    name: str
    experts: int
    hidden: int
    intermediate: int
    topk: int
    tokens: int


SANITY = Shape("sanity", 4, 256, 256, 1, 512)
GRAD_NAMES = ("d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down",
              "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down")


class InheritedScheduleCapacityError(RuntimeError):
    def __init__(self, required, capacity):
        self.evidence = {"required_padded_rows":required,"schedule_capacity":capacity,
                         "kernel":"scheduler::schedule_kernel","source":"csrc/scheduler.cuh:86",
                         "upstream_error":"trap -> CUDA_ERROR_LAUNCH_FAILED (719)",
                         "confirmation":"p2-deepseek-upstream-memcheck; upstream own test_forward_bf16"}
        super().__init__(f"upstream schedule trap: {required} padded rows exceed capacity {capacity}")


@dataclass(frozen=True)
class Inputs:
    x: torch.Tensor
    experts: torch.Tensor
    router: torch.Tensor

    def copy_(self, other):
        self.x.copy_(other.x)
        self.experts.copy_(other.experts)
        self.router.copy_(other.router)


def config_for(shape, *, benchmark=False):
    if shape == SANITY:
        return functional.MoKConfig(fwd_num_comm_sms=2, bwd_num_comm_sms=2, minibatch_size=256,
            macrobatch_size=256, schedule_capacity_multiplier=1.5, all_gather_top_experts_chunk_bytes=16)
    return functional.MoKConfig(fwd_num_comm_sms=36 if benchmark else 40,
                                schedule_capacity_multiplier=0.5 if benchmark else 1.5)


def build_schedule(workspace, config, inputs, schedule, scratch, routes_i32):
    routes_i32.copy_(inputs.experts)
    ops.all_gather_top_experts(routes_i32, workspace.all_gather_top_experts_buffer,
        workspace.all_gather_top_experts_buffer_multicast_ptr, workspace.ep_rank,
        config.all_gather_top_experts_chunk_bytes)
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                    workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    ops.schedule_out(workspace.all_gather_top_experts_buffer, schedule.tokens_per_expert.numel(),
        workspace.schedule_capacity, workspace.ep_rank, schedule.peer_rank, schedule.peer_token_idx,
        schedule.num_tokens, schedule.tokens_per_expert, scratch)
    return schedule


class MoKHost:
    def __init__(self, shape, cells, *, benchmark=False, image="linked", config=None):
        self.shape, self.cells, self.image = shape, {c["pass"]: c for c in cells}, image
        axes = cells[0]
        self.precision, self.clamp = axes["precision"], axes["clamp"]
        self.limit = 4.0 if self.clamp else None
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if self.world != axes["world"] or shape.experts % self.world:
            raise ValueError("shape/process group disagrees with selected link")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.config = config or config_for(shape, benchmark=benchmark)
        self.stack, self.sessions, self.controls = ExitStack(), {}, {}
        E,L,H,I,k = shape.experts//self.world,shape.tokens,shape.hidden,shape.intermediate,shape.topk
        generated = generate_inputs(self.rank, self.device, shape.experts,E,k,L,H,I)
        self.inputs = Inputs(*generated[:3])
        self.shared_weights, self.routed_weights_bf16, self.grad_output = generated[3:6],generated[6:9],generated[9]
        self.workspace = functional.create_workspace(self.config, dist.group.WORLD, device=self.device,
                                                     num_local_tokens=L, hidden_size=H, topk=k)
        if self.world==1:
            # Classify the confirmed upstream EP1 trap without poisoning this
            # process. This is setup evidence, never a changed capacity or kernel.
            padding = axes["manifest"]["host"]["facts"]["mok.framework.EXPERT_PADDING"]
            counts = torch.bincount(self.inputs.experts.flatten(),minlength=E)
            required = int((((counts+padding-1)//padding)*padding).sum().item())
            if required>self.workspace.schedule_capacity:
                raise InheritedScheduleCapacityError(required,self.workspace.schedule_capacity)
        self.schedule = functional.build_schedule(self.workspace,self.config,self.inputs.experts,num_local_experts=E)
        self.schedule_scratch = torch.empty(E*self.world,dtype=torch.int32,device=self.device)
        self.routes_i32 = torch.empty((L,k),dtype=torch.int32,device=self.device)
        self.scalars = dict(num_local_tokens=L,hidden=H,intermediate=I,num_local_experts=E,
            schedule_capacity=self.workspace.schedule_capacity,macrobatch_size=self.config.macrobatch_size,
            minibatch_size=self.config.minibatch_size,topk=k,swiglu_limit=self.limit or 0.0)
        self.specs = {p: validate_inventory(c,self.scalars) for p,c in self.cells.items()}
        self.buffers = {}
        dtypes = dict(bf16=torch.bfloat16,u8=torch.uint8,i32=torch.int32,f32=torch.float32)
        for specs in self.specs.values():
            for key,spec in specs.items():
                if spec.active and key not in self.buffers:
                    if spec.peer:
                        self.buffers[key] = getattr(self.workspace,key)
                    elif key.startswith("schedule_"):
                        self.buffers[key] = getattr(self.schedule,key.removeprefix("schedule_"))
                    elif key in ("num_tokens","tokens_per_expert"):
                        self.buffers[key] = getattr(self.schedule,key)
                    elif not key.startswith("w_"):
                        self.buffers[key] = torch.empty(spec.shape,dtype=dtypes[spec.dtype],device=self.device)
        self.routed_weights, self.backward_weights = [], []
        for name,shared,routed in zip(("gate","up","down"),self.shared_weights,self.routed_weights_bf16):
            self.buffers[f"w_shared_{name}"] = shared
            if self.precision == "mxfp8":
                normal,sc,transposed,tsc = ops.mxfp8_quantize(routed,True,True)
                self.routed_weights.append((normal,sc))
                self.backward_weights.append((normal,sc,transposed,tsc) if name != "down" else (transposed,tsc))
                for suffix,tensor in (("fp8",normal),("sc",sc),("t_fp8",transposed),("t_sc",tsc)):
                    self.buffers[f"w_routed_{name}_{suffix}"] = tensor.view(torch.uint8)
            else:
                self.buffers[f"w_routed_{name}_bf16"] = routed
                self.routed_weights.append(routed); self.backward_weights.append(routed)
        b = self.buffers
        def context_value(key, transposed=False):
            if self.precision == "mxfp8":
                key += "_t" if transposed else ""
                return (b[key+"_fp8"].view(torch.float8_e4m3fn),b[key+"_sc"])
            return b[key if key in ("gate_routed","up_routed") else key+"_bf16"]
        self.forward_context = functional.MoKForwardContext(x_routed=context_value("x_routed",True),
            gate_shared=b["gate_shared"],gate_routed=context_value("gate_routed"),
            up_shared=b["up_shared"],up_routed=context_value("up_routed"),
            hidden_shared=b["hidden_shared"],hidden_routed=context_value("hidden_routed",True))
        # Buffers each pass's kernel writes: everything it uses except host inputs and,
        # for backward, the forward context it reads. Upstream allocates all of these
        # with at::empty, so a kernel may not depend on their prior contents.
        inputs = {"x_buffer","router_weight_buffer","d_y_buffer","num_tokens","tokens_per_expert"}
        context = {t.data_ptr() for t in self.context_tensors().values()}
        self.poison = False
        self.written = {p: [b[k] for k,spec in specs.items()
                            if spec.active and k not in inputs and not k.startswith(("w_","schedule_"))
                            and not (p=="bwd" and b[k].data_ptr() in context)]
                        for p,specs in self.specs.items()}
        self.open_sessions()

    def open_sessions(self):
        from pipec2.host.broker import open_host
        from pipec2.host.cuda import CudaBackend
        from pipec2.host.fixture import Domain, Resource
        from pipec2.host.profile import DTYPES
        from pipec2.model import LinkResult
        for p,cell in self.cells.items():
            result = LinkResult.from_json_obj(cell["manifest"])
            root = cell["directory"]
            geometry = dict(device=torch.cuda.current_device(),dynamic_smem_bytes=result.dyn_smem_bytes,
                            cluster=(2,1,1),block=(256,1,1))
            backend = self.stack.enter_context(CudaBackend(root/cell["linked"],result.kernel_symbol,**geometry))
            self.controls[p] = self.stack.enter_context(CudaBackend(root/cell["one_tu"],result.kernel_symbol,**geometry))
            resources = {}
            for key,spec in self.specs[p].items():
                tensor = self.buffers[key] if spec.active else None
                peers = (tuple(getattr(self.workspace,key+"_ptrs")) if spec.active else (0,)*self.world) if spec.peer else ()
                resources[key] = Resource(tensor.data_ptr() if tensor is not None else 0,
                    tensor.numel()*tensor.element_size() if tensor is not None else 0,
                    spec.extents,spec.strides,DTYPES[spec.dtype]["id"],spec.alignment,
                    domain="ep" if spec.peer else None,peers=peers,lender=(tensor,self.workspace))
            scalars = dict(self.scalars,num_comm_sms=self.config.bwd_num_comm_sms if p == "bwd" else self.config.fwd_num_comm_sms)
            self.sessions[p] = self.stack.enter_context(open_host(result,
                {k:root/v for k,v in cell["providers"].items()}, {},resources,scalars,backend,(2,1,1),
                domains={"ep": Domain(self.rank, self.world)}))

    def barrier(self):
        w = self.workspace
        ops.barrier_all(w.barrier_buffer,w.barrier_buffer_ptrs,w.barrier_buffer_multicast_ptr,w.barrier_target)

    def schedule_into(self,inputs):
        return build_schedule(self.workspace,self.config,inputs,self.schedule,self.schedule_scratch,self.routes_i32)

    def megakernel(self, pass_name="fwd", image=None):
        image = image or self.image
        self.sessions[pass_name].launch(torch.cuda.current_stream().cuda_stream,
                                       image=self.controls[pass_name] if image == "one-tu" else None)

    def poison_outputs(self,pass_name):
        """Checks only: NaN-fill (0xFF for MXFP8 bytes) what the pass writes, so an
        omitted store cannot inherit a correct value from upstream or the other image.
        Every rank fills before the pre-launch barrier, ahead of remote peer writes."""
        if self.poison:
            for tensor in self.written[pass_name]:
                tensor.fill_(255 if tensor.dtype==torch.uint8 else math.nan)

    def forward(self,inputs,*,image=None):
        self.poison_outputs("fwd")
        self.schedule_into(inputs)
        w = self.workspace
        w.x_buffer.copy_(inputs.x); w.router_weight_buffer.copy_(inputs.router)
        self.barrier(); self.megakernel("fwd",image); self.barrier()
        return {"output":ops.fwd_epilogue(self.buffers["y_shared"],w.combine_buffer,w.router_weight_buffer),
                **self.context_tensors()}

    def recompute(self,inputs,*,image=None):
        self.poison_outputs("rec")
        self.workspace.x_buffer.copy_(inputs.x)
        self.barrier(); self.megakernel("rec",image); self.barrier()
        return self.context_tensors()

    def backward(self,inputs,*,image=None):
        w,b = self.workspace,self.buffers
        self.poison_outputs("bwd")
        w.d_y_buffer.copy_(self.grad_output); w.x_buffer.copy_(inputs.x); w.router_weight_buffer.copy_(inputs.router)
        self.barrier(); self.megakernel("bwd",image)
        _C.zero_empty_routed_wgrads(b["d_w_routed_gate"],b["d_w_routed_up"],b["d_w_routed_down"],self.schedule.tokens_per_expert)
        self.barrier()
        return {"d_x":ops.bwd_epilogue(b["d_x_shared"],w.d_x_routed_buffer),
                "d_router_weights":w.d_router_weight_buffer.clone(),**{k:b[k] for k in GRAD_NAMES[2:]}}

    def context_tensors(self):
        result = {}
        for name in ("x_routed","gate_shared","gate_routed","up_shared","up_routed","hidden_shared","hidden_routed"):
            value = getattr(self.forward_context,name)
            if isinstance(value,tuple):
                result[name+"_data"],result[name+"_scale"] = value
            else:
                result[name] = value
        return result

    def sequence(self,inputs,pass_name,*,image=None,checkpoint=False):
        if pass_name == "fwd":
            return self.forward(inputs,image=image)
        if checkpoint or pass_name == "rec":
            self.schedule_into(inputs)
            self.recompute(inputs,image=image)
        else:
            self.forward(inputs,image=image)
        context = {k:v.clone() for k,v in self.context_tensors().items()} if pass_name == "rec" else {}
        return {**context, **self.backward(inputs,image=image)}

    def reference(self,inputs,pass_name):
        result = {}
        if pass_name!="fwd":
            values = run_reference_bf16(inputs.x,inputs.experts,inputs.router,*self.shared_weights,
                                       *self.routed_weights_bf16,self.grad_output,self.limit)
            result.update(zip(GRAD_NAMES,values[1:]))
        if pass_name!="bwd":
            combine,gate,up,hidden,y = run_forward_reference_bf16(inputs.x,inputs.experts,*self.shared_weights,
                                                               *self.routed_weights_bf16,self.limit)
            result.update(gate_shared=gate,up_shared=up,hidden_shared=hidden)
            if pass_name=="fwd":
                result["output"] = run_fwd_epilogue_reference(y,combine,inputs.router)
        return result

    def upstream(self,inputs,pass_name):
        schedule = functional.build_schedule(self.workspace,self.config,inputs.experts,
                                              num_local_experts=self.scalars["num_local_experts"])
        if pass_name == "rec":
            context = functional.recompute_forward_context(self.config,self.workspace,schedule,inputs.x,
                *self.shared_weights[:2],*self.routed_weights[:2],self.limit)
        else:
            output,context = functional.forward(self.config,self.workspace,schedule,inputs.x,inputs.router,
                                                *self.shared_weights,*self.routed_weights,self.limit)
            if pass_name == "fwd":
                return {"output":output}
        values = functional.backward(self.config,self.workspace,schedule,context,self.grad_output,inputs.x,
            inputs.router,*self.shared_weights,*self.backward_weights,self.limit)
        return dict(zip(GRAD_NAMES,values))

    def new_inputs(self,generation):
        s = self.shape
        gen = torch.Generator(device=self.device).manual_seed(1234+self.rank+generation*self.world)
        logits = torch.randn(s.tokens,s.experts,generator=gen,device=self.device)
        values,experts = torch.topk(logits,s.topk,dim=1)
        if generation % 3 == 0:
            # Legal, concentrated routing exercises empty experts and empty ranks.
            experts.copy_(torch.arange(s.topk,device=self.device).expand(s.tokens,-1))
        x = torch.randn(s.tokens,s.hidden,generator=gen,device=self.device,dtype=torch.bfloat16)
        return Inputs(x,experts,torch.softmax(values.float(),dim=-1))

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        torch.cuda.synchronize()
        self.stack.close()


def error_stats(expected,actual):
    """tests/utils.py's global L1/max metrics, with bounded FP32 temporaries."""
    expected,actual = expected.reshape(-1),actual.reshape(-1)
    sums = torch.zeros(3,device=expected.device,dtype=torch.float64)
    maximum = torch.zeros((),device=expected.device,dtype=torch.float32)
    sums[2] = expected.numel()
    for start in range(0,expected.numel(),1<<26):
        reference = expected[start:start+(1<<26)].float()
        result = actual[start:start+(1<<26)].float()
        difference = (reference-result).abs()
        sums[0].add_(difference.sum(dtype=torch.float64))
        sums[1].add_(reference.abs().sum(dtype=torch.float64))
        torch.maximum(maximum,difference.max(),out=maximum)
    dist.all_reduce(sums,op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
    return (sums[0]/sums[2]).item(),maximum.item(),(sums[0]/sums[1]).item()


def tolerance_report(expected,actual,precision):
    mean,maximum,relative = error_stats(expected,actual)
    atol,rtol = MXFP8_TOLERANCE if precision == "mxfp8" else BF16_TOLERANCE
    return dict(abs_mean=mean,abs_max=maximum,relative=relative,absolute_tolerance=atol,relative_tolerance=rtol,
        passed=all(math.isfinite(v) for v in (mean,maximum,relative)) and maximum<=atol and relative<=rtol)


def bitwise_report(left,right):
    equal = torch.tensor(int(torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))),device=left.device)
    dist.all_reduce(equal,op=dist.ReduceOp.MIN)
    return bool(equal.item())


def check_host(host,pass_name,*,steps=3,images=("linked","one-tu"),upstream_only=False):
    host.poison = True
    records,graphs = [],[]
    def compare(values,expected):
        return {k:tolerance_report(expected[k],v,host.precision) for k,v in values.items() if k in expected}
    def accept_inherited(reports,upstream_reports,equal):
        for key,report in reports.items():
            inherited = (set(images)=={"linked","one-tu"} and not report["passed"]
                         and key in upstream_reports and not upstream_reports[key]["passed"]
                         and equal.get(key,False))
            report["inherited_upstream_miss"] = inherited
            report["accepted"] = report["passed"] or inherited
            if inherited and host.clamp and pass_name in ("bwd","rec"):
                report["inherited_cause"] = (
                    "MoK uses inclusive clamp-boundary gradients (torch 2.10 behavior); "
                    "torch 2.14 reference clamp autograd uses strict boundary gradients")
    def snapshot(values):
        rows = min(int(host.schedule.num_tokens.item()),host.config.macrobatch_size)
        result = {}
        for key,value in values.items():
            if key.startswith(("x_routed","gate_routed","up_routed","hidden_routed")):
                count = rows//128 if key.endswith("_scale") else rows
                transposed = host.precision=="mxfp8" and key.startswith(("x_routed","hidden_routed"))
                value = value[:, :count] if transposed else value[:count]
            result[key] = value.clone()
        return result
    def recompute_equal(inputs,image,values):
        forward = snapshot(host.forward(inputs,image=image))
        return all(bitwise_report(value,forward[key]) for key,value in values.items()
                   if key not in GRAD_NAMES)
    for generation in range(steps):
        inputs = host.new_inputs(generation+1)
        expected = host.reference(inputs,pass_name)
        original = functional.build_schedule(host.workspace,host.config,inputs.experts,
                                              num_local_experts=host.scalars["num_local_experts"])
        host.schedule_into(inputs)
        valid = original.peer_rank>=0
        schedule_equal = all([bitwise_report(original.peer_rank,host.schedule.peer_rank),
            bitwise_report(original.peer_token_idx[valid],host.schedule.peer_token_idx[valid]),
            bitwise_report(original.num_tokens,host.schedule.num_tokens),
            bitwise_report(original.tokens_per_expert,host.schedule.tokens_per_expert)])
        upstream = host.upstream(inputs,pass_name)
        row = {"generation":generation,"schedule_equal":schedule_equal,
               "live_tokens":int(host.schedule.num_tokens.item()),
               "empty_experts":int((host.schedule.tokens_per_expert==0).sum().item()),
               "invalid_padded_routes":int((host.schedule.peer_rank<0).sum().item()),
               "upstream_reference":compare(upstream,expected),"images":{}}
        outputs = {}
        for image in (() if upstream_only else images):
            values = host.sequence(inputs,pass_name,image=image)
            outputs[image] = snapshot(values)
            row.setdefault("upstream_bitwise_diagnostic",{})[image] = {
                k:bitwise_report(upstream[k],v) for k,v in outputs[image].items() if k in upstream}
            if pass_name=="rec":
                row.setdefault("recompute_exact",{})[image] = recompute_equal(inputs,image,outputs[image])
            row["images"][image] = compare(values,expected)
            accept_inherited(row["images"][image],row["upstream_reference"],
                             row["upstream_bitwise_diagnostic"][image])
        row["linked_onetu_bitwise"] = all(bitwise_report(outputs["linked"][k],outputs["one-tu"][k]) for k in outputs.get("linked",{})) if len(outputs)==2 else True
        records.append(row)
        del expected,upstream,outputs
    static = host.new_inputs(steps+1)
    captured = {}
    for image in (("upstream",) if upstream_only else images):
        step = (lambda:host.upstream(static,pass_name)) if image=="upstream" else (lambda:host.sequence(static,pass_name,image=image))
        stream = torch.cuda.Stream(device=host.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize(); dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): values = step()
        captured[image] = graph,values
    for generation in range(2):
        inputs = host.new_inputs(steps+2+generation)
        expected = host.reference(inputs,pass_name)
        upstream = host.upstream(inputs,pass_name)
        upstream_reports = compare(upstream,expected)
        static.copy_(inputs)
        snapshots = {}
        for image,(graph,values) in captured.items():
            graph.replay()
            snapshots[image] = snapshot(values)
            reports = compare(values,expected)
            equal = {k:bitwise_report(upstream[k],v) for k,v in snapshots[image].items() if k in upstream}
            accept_inherited(reports,upstream_reports,equal)
            graphs.append({"generation":generation,"image":image,"reference":reports,
                           "upstream_reference":upstream_reports,"upstream_bitwise_diagnostic":equal})
            if pass_name=="rec" and image!="upstream":
                graphs[-1]["recompute_exact"] = recompute_equal(inputs,image,snapshots[image])
        if len(snapshots)==2:
            equal = all(bitwise_report(snapshots["linked"][k],snapshots["one-tu"][k]) for k in snapshots["linked"])
            graphs[-1]["linked_onetu_bitwise"] = equal
        del expected,upstream,snapshots
    passed = all(r["schedule_equal"] and r["linked_onetu_bitwise"] and all(r.get("recompute_exact",{}).values())
                 and all(v["accepted"] for im in r["images"].values() for v in im.values())
                 and all(v["passed"] or (bool(r["images"]) and all(im.get(k,{}).get("inherited_upstream_miss",False)
                         for im in r["images"].values())) for k,v in r["upstream_reference"].items())
                 for r in records)
    passed &= all(r.get("linked_onetu_bitwise",True) and r.get("recompute_exact",True)
                  and all(v["accepted"] for v in r["reference"].values()) for r in graphs)
    inherited = any(v["inherited_upstream_miss"] for r in records for im in r["images"].values() for v in im.values())
    inherited |= any(v["inherited_upstream_miss"] for r in graphs for v in r["reference"].values())
    return {"status":("FAIL-inherited" if inherited else "PASS") if passed else "FAIL","steps":records,"graphs":graphs,
            "sequence":"R->B" if pass_name=="rec" else "F->B" if pass_name=="bwd" else "F",
            "metric_meaning":"relative = global sum absolute error / global sum absolute reference"}


class AllRank:
    """CUDA event intervals reduced MAX across ranks; preparation is untimed."""
    def __init__(self,issue,prepare,sections=None):
        self.issue,self.prepare,self.sections = issue,prepare,sections
        self.elapsed = torch.empty((),device="cuda",dtype=torch.float64)
        self.records = []

    def launch(self):
        self.prepare(); self.issue()

    def time(self,n):
        dist.barrier()
        if self.sections is not None:
            self.sections.clear()
        pairs = [(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(n)]
        for start,end in pairs:
            self.prepare(); start.record(); self.issue(); end.record()
        pairs[-1][1].synchronize()
        local = sum(a.elapsed_time(b) for a,b in pairs)
        self.elapsed.fill_(local); dist.all_reduce(self.elapsed,op=dist.ReduceOp.MAX)
        maximum = self.elapsed.item()
        self.records.append({"n":n,"local_ms":local,"max_rank_ms":maximum})
        if self.sections is not None:
            section_local = sum(a.elapsed_time(b) for a,b in self.sections)
            self.elapsed.fill_(section_local); dist.all_reduce(self.elapsed,op=dist.ReduceOp.MAX)
            self.records[-1].update(launch_region_local_ms=section_local,
                                    launch_region_max_rank_ms=self.elapsed.item())
        return maximum


def time_host(host,pass_name,repetitions=20,*,sections=False):
    from evaluator.harness.parity.example import time_pair
    inputs = host.inputs
    def prepare(image):
        if pass_name=="bwd": host.forward(inputs,image=image)
        elif pass_name=="rec": host.schedule_into(inputs)
        host.barrier()
    def issue(image):
        return {"fwd":host.forward,"bwd":host.backward,"rec":host.recompute}[pass_name](inputs,image=image)
    a = AllRank(lambda:issue("linked"),lambda:prepare("linked"))
    b = AllRank(lambda:issue("one-tu"),lambda:prepare("one-tu"))
    paired = time_pair(a,b,repetitions=repetitions)
    control_ratio = paired["ratio_linked_over_onetu"]
    passed = paired["status"]=="PASS" and 0.995 <= control_ratio <= 1.005
    result = {"status":"PASS" if passed else "FAIL",
        "parity_gate":{"maximum_linked_over_onetu":1.03,"passed":control_ratio<=1.03},
        "control_band":{"minimum":0.995,"maximum":1.005,"passed":passed},
        "whole_step":paired,"samples":{"linked":a.records,"one_tu":b.records},
        "measurement":"isolated pass whole step; preparation outside events; MAX rank total"}
    if not passed:
        return result
    schedule = context = None
    def upstream_schedule():
        return functional.build_schedule(host.workspace,host.config,inputs.experts,
            num_local_experts=host.scalars["num_local_experts"])
    def upstream_forward():
        return functional.forward(host.config,host.workspace,schedule,inputs.x,inputs.router,
            *host.shared_weights,*host.routed_weights,host.limit)
    def prepare_upstream():
        nonlocal schedule,context
        if pass_name != "fwd":
            schedule = upstream_schedule()
        if pass_name == "bwd":
            _,context = upstream_forward()
        host.barrier()
    def issue_upstream():
        nonlocal schedule
        if pass_name == "fwd":
            schedule = upstream_schedule()
            return upstream_forward()
        if pass_name == "rec":
            return functional.recompute_forward_context(host.config,host.workspace,schedule,inputs.x,
                *host.shared_weights[:2],*host.routed_weights[:2],host.limit)
        return functional.backward(host.config,host.workspace,schedule,context,host.grad_output,inputs.x,
            inputs.router,*host.shared_weights,*host.backward_weights,host.limit)
    linked_events,upstream_events = [],[]
    enabled = False
    def instrument(original,events):
        def call(*args,**kwargs):
            if not enabled:
                return original(*args,**kwargs)
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            value = original(*args,**kwargs)
            end.record()
            events.append((start,end))
            return value
        return call
    def timed_issue(call):
        nonlocal enabled
        enabled = True
        try:
            return call()
        finally:
            enabled = False
    with ExitStack() as instrumentation:
        if sections:
            op_name = (f"recompute_forward_context_{host.precision}" if pass_name=="rec" else
                       f"dispatch_mlp_swiglu_combine_{pass_name}_{host.precision}")
            for owner,name,events in ((host,"megakernel",linked_events),(functional,op_name,upstream_events)):
                original = getattr(owner,name)
                instrumentation.callback(setattr,owner,name,original)
                setattr(owner,name,instrument(original,events))
        linked_issue = lambda:issue("linked")
        linked = AllRank((lambda:timed_issue(linked_issue)) if sections else linked_issue,
                         lambda:prepare("linked"),linked_events if sections else None)
        upstream = AllRank((lambda:timed_issue(issue_upstream)) if sections else issue_upstream,
                           prepare_upstream,upstream_events if sections else None)
        placement = time_pair(linked,upstream,repetitions=repetitions)
    if sections:
        result["section_method"] = ("Diagnostic nested CUDA events, enabled only inside issue: linked session launch includes counter reset; "
            "upstream dispatch op includes allocations/counter initialization and, for B, wrapper zero-empty launch. "
            "Section max-rank totals are separately reduced; subtracting from full MAX totals is descriptive, not same-rank attribution.")
    # time_pair's historical B label is one-TU; name the actual B arm explicitly.
    result["upstream_pair"] = {"arms":{"A":"linked","B":"upstream"},**placement,
        "ratio_linked_over_upstream":placement["ratio_linked_over_onetu"]}
    result["samples"].update(placement_linked=linked.records,upstream=upstream.records)
    operations = {
        "fwd":"route int64->int32, all-gather, barrier, schedule, x/router copies, barrier, forward megakernel, barrier, forward epilogue",
        "bwd":"gradient/x/router copies, barrier, backward megakernel, zero-empty routed weight gradients, barrier, backward epilogue, router-gradient clone",
        "rec":"x copy, barrier, recompute megakernel, barrier",
    }
    result["interval"] = {"both_arms":operations[pass_name],
        "outside_events":"B: matching own forward including schedule/context restoration; R: own schedule; all: pre-interval barrier",
        "linked_and_one_tu":"persistent schedule/context/output resources; session counter reset included",
        "upstream":"original functional API, allocation-returning schedule/kernel wrappers and their counter initialization included; weights quantized outside timing"}
    result["status"] = "PASS" if placement["status"]=="PASS" else "FAIL"
    return result
