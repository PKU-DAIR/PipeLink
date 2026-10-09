"""Bounded phase markers for EP4 initialization, symmetric memory and MoK barrier."""

from datetime import timedelta
import json
import os
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cuda.bindings import driver

from mok import functional, ops


def mark(phase, **fields):
    print(json.dumps({"rank": int(os.environ["RANK"]), "time": time.time(), "phase": phase, **fields}), flush=True)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    mark("set_device_start", local_rank=local_rank, visible_count=torch.cuda.device_count())
    torch.cuda.set_device(device)
    error, cuda_device = driver.cuDeviceGet(local_rank)
    assert error == driver.CUresult.CUDA_SUCCESS
    error, multicast = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cuda_device,
    )
    assert error == driver.CUresult.CUDA_SUCCESS
    mark("driver_attributes", multicast_supported=multicast)
    mark("nccl_init_start")
    dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(seconds=120))
    mark("nccl_init_complete")
    value = torch.ones(1, device=device)
    dist.all_reduce(value)
    torch.cuda.synchronize()
    mark("nccl_all_reduce_complete", value=value.item())
    backend = os.environ.get("MOK_SYMM_BACKEND")
    if backend:
        symm_mem.set_backend(backend)
    mark("symm_empty_start", backend=symm_mem.get_backend(device))
    buffer = symm_mem.empty(512, 256, dtype=torch.bfloat16, device=device)
    mark("symm_empty_complete", backend=symm_mem.get_backend(device))
    handle = symm_mem.rendezvous(buffer, dist.group.WORLD)
    mark("symm_rendezvous_complete", multicast_available=bool(handle.multicast_ptr), peer_count=len(handle.buffer_ptrs))
    config = functional.MoKConfig(
        fwd_num_comm_sms=2, bwd_num_comm_sms=2, minibatch_size=256,
        macrobatch_size=256, schedule_capacity_multiplier=1.5,
        all_gather_top_experts_chunk_bytes=16,
    )
    mark("mok_workspace_start")
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=device,
        num_local_tokens=512, hidden_size=256, topk=1,
    )
    mark("mok_workspace_complete")
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                    workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    torch.cuda.synchronize()
    mark("mok_barrier_complete", target=workspace.barrier_target.item())
    dist.barrier()
    functional.clear_workspace_cache()
    del handle, buffer
    dist.destroy_process_group()
    mark("complete")


if __name__ == "__main__":
    main()
