"""Single-process POSIX-FD multicast binding, independent of torch and NCCL."""

import argparse
import json
import time

from cuda.bindings import driver as cuda


def checked(call, *args):
    result = call(*args)
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{call.__name__}: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def mark(phase, **fields):
    print(json.dumps({"time": time.time(), "phase": phase, **fields}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3", help="Logical ordinals within CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    ordinals = [int(value) for value in args.devices.split(",")]
    checked(cuda.cuInit, 0)
    visible = checked(cuda.cuDeviceGetCount)
    assert len(set(ordinals)) == len(ordinals) and all(0 <= index < visible for index in ordinals)
    contexts = []
    devices = []
    allocations = []
    bound = []
    multicast = None
    try:
        for ordinal in ordinals:
            device = checked(cuda.cuDeviceGet, ordinal)
            context = checked(cuda.cuDevicePrimaryCtxRetain, device)
            contexts.append(context)
            devices.append(device)
        prop = cuda.CUmulticastObjectProp()
        prop.numDevices = len(devices)
        prop.size = 1
        prop.handleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        prop.flags = 0
        granularity = checked(cuda.cuMulticastGetGranularity, prop,
                              cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED)
        prop.size = granularity
        checked(cuda.cuCtxSetCurrent, contexts[0])
        mark("multicast_create_start", logical_devices=ordinals, size_bytes=granularity, handle_type="POSIX_FILE_DESCRIPTOR")
        multicast = checked(cuda.cuMulticastCreate, prop)
        mark("multicast_create_complete")
        for ordinal, device in zip(ordinals, devices):
            checked(cuda.cuMulticastAddDevice, multicast, device)
            mark("multicast_add_device_complete", logical_device=ordinal)
        for ordinal, device, context in zip(ordinals, devices, contexts):
            checked(cuda.cuCtxSetCurrent, context)
            memory_prop = cuda.CUmemAllocationProp()
            memory_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            memory_prop.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            memory_prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            memory_prop.location.id = int(device)
            allocation_granularity = checked(cuda.cuMemGetAllocationGranularity, memory_prop,
                                             cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)
            allocation_size = ((granularity + allocation_granularity - 1) // allocation_granularity) * allocation_granularity
            memory = checked(cuda.cuMemCreate, allocation_size, memory_prop, 0)
            allocations.append(memory)
            mark("multicast_bind_start", logical_device=ordinal, allocation_bytes=allocation_size)
            checked(cuda.cuMulticastBindMem, multicast, 0, memory, 0, granularity, 0)
            bound.append(device)
            mark("multicast_bind_complete", logical_device=ordinal)
        mark("complete")
    finally:
        for device in bound:
            checked(cuda.cuMulticastUnbind, multicast, device, 0, granularity)
        for allocation in allocations:
            checked(cuda.cuMemRelease, allocation)
        if multicast is not None:
            checked(cuda.cuMemRelease, multicast)
        for device in devices:
            checked(cuda.cuDevicePrimaryCtxRelease, device)


if __name__ == "__main__":
    main()
