"""Persistent plain-CUDA cubin sessions, using only ctypes and the driver API."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import ctypes as C
import math
from pathlib import Path

import numpy as np


class _AttributeValue(C.Union):
    _fields_ = [("storage", C.c_uint64 * 8), ("cluster", C.c_uint32 * 3)]


class _Attribute(C.Structure):
    _fields_ = [("id", C.c_int), ("pad", C.c_uint32), ("value", _AttributeValue)]


class _Config(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("gx", "gy", "gz", "bx", "by", "bz", "smem")] + [
        ("stream", C.c_void_p), ("attrs", C.POINTER(_Attribute)), ("count", C.c_uint32)]


class NativeSession:
    """Own loaded modules and one buffer set; optionally share another session's stream.

    ``arguments(addresses)`` runs with the context current and returns the kernel
    arguments in order, as ctypes scalars or by-value byte images (e.g.
    CUtensorMap). ``following`` lists further kernels (cubin, kernel, arguments,
    grid, threads, dyn_smem, cluster). Each launch submits the whole sequence on
    one stream over the same buffers. Setup is untimed; ``time(n)`` returns the
    total event milliseconds of n sequences. A borrowed stream must outlive the
    session.
    """

    def __init__(self, cubin, kernel, *, buffers, arguments, grid, threads=128,
                 dyn_smem=0, cluster=(1, 1, 1), device=0, stream=None, following=()):
        self.driver = C.CDLL("libcuda.so.1")
        self._functions = {}
        self._lifetime = ExitStack()
        self._buffers = {}
        self.addresses = {}
        self.context = C.c_void_p()
        self._closed = False
        try:
            self._call("cuInit", [C.c_uint], 0)
            dev = C.c_int()
            self._call("cuDeviceGet", [C.POINTER(C.c_int), C.c_int], C.byref(dev), device)
            self._call("cuCtxGetCurrent", [C.POINTER(C.c_void_p)], C.byref(self.context))
            if self.context.value:
                current_device = C.c_int()
                self._call("cuCtxGetDevice", [C.POINTER(C.c_int)], C.byref(current_device))
                if current_device.value != dev.value:
                    self.context = C.c_void_p()
            if not self.context.value:
                self._call("cuDevicePrimaryCtxRetain", [C.POINTER(C.c_void_p), C.c_int],
                           C.byref(self.context), dev)
                self._lifetime.callback(self._call, "cuDevicePrimaryCtxRelease_v2", [C.c_int], dev)
            with self.current():
                if stream is None:
                    owned_stream = C.c_void_p()
                    self._call("cuStreamCreate", [C.POINTER(C.c_void_p), C.c_uint], C.byref(owned_stream), 0)
                    self.stream = owned_stream.value
                    self._lifetime.callback(self._release, "cuStreamDestroy_v2", [C.c_void_p], owned_stream)
                else:
                    self.stream = stream
                for name, value in buffers.items():
                    host = np.ascontiguousarray(value)
                    address = C.c_uint64()
                    self._call("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t], C.byref(address), host.nbytes)
                    self._lifetime.callback(self._release, "cuMemFree_v2", [C.c_uint64], address)
                    self.addresses[name] = address.value
                    self._buffers[name] = (host.shape, host.dtype)
                    self._call("cuMemcpyHtoD_v2", [C.c_uint64, C.c_void_p, C.c_size_t],
                               address, host.ctypes.data, host.nbytes)
                self._launches = [
                    self._prepare(cubin, kernel, arguments=arguments, grid=grid,
                                  threads=threads, dyn_smem=dyn_smem, cluster=cluster),
                    *(self._prepare(**spec) for spec in following),
                ]
        except BaseException:
            self.close()
            raise

    def _prepare(self, cubin, kernel, *, arguments, grid, threads=128, dyn_smem=0,
                 cluster=(1, 1, 1)):
        module, function = C.c_void_p(), C.c_void_p()
        image = C.create_string_buffer(Path(cubin).read_bytes())
        self._call("cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_void_p], C.byref(module), image)
        self._lifetime.callback(self._release, "cuModuleUnload", [C.c_void_p], module)
        self._call("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                   C.byref(function), module, kernel.encode())
        self._call("cuFuncSetAttribute", [C.c_void_p, C.c_int, C.c_int], function, 8, dyn_smem)
        values = [C.create_string_buffer(value, len(value)) if isinstance(value, bytes) else value
                  for value in arguments(self.addresses)]
        pointers = (C.c_void_p * len(values))(*(C.addressof(value) for value in values))
        attribute = _Attribute()
        attribute.id = 4  # CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attribute.value.cluster[:] = cluster
        config = _Config(*grid, threads, 1, 1, dyn_smem, self.stream, C.pointer(attribute), 1)
        # Return the ctypes argument storage and attribute too, so they live until the module closes.
        return config, function, pointers, values, attribute

    def _call(self, name, types, *args):
        function = self._functions.get(name)
        if function is None:
            function = getattr(self.driver, name)
            function.argtypes, function.restype = types, C.c_int
            self._functions[name] = function
        status = function(*args)
        if status:
            message = C.c_char_p()
            describe = self.driver.cuGetErrorString
            describe.argtypes, describe.restype = [C.c_int, C.POINTER(C.c_char_p)], C.c_int
            describe(status, C.byref(message))
            raise RuntimeError(f"{name}: CUDA {status}: {(message.value or b'unknown').decode()}")

    @contextmanager
    def current(self):
        self._call("cuCtxPushCurrent_v2", [C.c_void_p], self.context)
        try:
            yield
        finally:
            prior = C.c_void_p()
            self._call("cuCtxPopCurrent_v2", [C.POINTER(C.c_void_p)], C.byref(prior))

    def _release(self, name, types, *args):
        with self.current():
            self._call(name, types, *args)

    def launch(self):
        with self.current():
            for config, function, arguments, _values, _attribute in self._launches:
                self._call("cuLaunchKernelEx", [C.POINTER(_Config), C.c_void_p,
                                              C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)],
                           C.byref(config), function, arguments, None)

    def synchronize(self):
        with self.current():
            self._call("cuCtxSynchronize", [])

    def read(self, name):
        self.synchronize()
        shape, dtype = self._buffers[name]
        result = np.empty(shape, dtype=dtype)
        with self.current():
            self._call("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t],
                       result.ctypes.data, self.addresses[name], result.nbytes)
        return result

    def write(self, name, value):
        self.synchronize()
        shape, dtype = self._buffers[name]
        host = np.ascontiguousarray(value, dtype=dtype).reshape(shape)
        with self.current():
            self._call("cuMemcpyHtoD_v2", [C.c_uint64, C.c_void_p, C.c_size_t],
                       self.addresses[name], host.ctypes.data, host.nbytes)

    def time(self, repetitions):
        if type(repetitions) is not int or repetitions <= 0:
            raise ValueError("repetitions must be a positive integer")
        with ExitStack() as events:
            start, end = C.c_void_p(), C.c_void_p()
            for event in (start, end):
                self._release("cuEventCreate", [C.POINTER(C.c_void_p), C.c_uint], C.byref(event), 0)
                events.callback(self._release, "cuEventDestroy_v2", [C.c_void_p], event)
            self._release("cuEventRecord", [C.c_void_p, C.c_void_p], start, self.stream)
            for _ in range(repetitions):
                self.launch()
            self._release("cuEventRecord", [C.c_void_p, C.c_void_p], end, self.stream)
            self._release("cuEventSynchronize", [C.c_void_p], end)
            elapsed = C.c_float()
            self._release("cuEventElapsedTime", [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
                          C.byref(elapsed), start, end)
            if not math.isfinite(elapsed.value) or elapsed.value <= 0:
                raise RuntimeError(f"CUDA event batch duration must be finite and positive: {elapsed.value}")
            return elapsed.value

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.context.value:
                self.synchronize()
        finally:
            self._lifetime.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_native(cubin, kernel, **kwargs):
    return NativeSession(cubin, kernel, **kwargs)
