"""CUDA driver execution for the host broker; no frontend/kernel type dependency."""
from __future__ import annotations

import ctypes as C
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


class CudaError(RuntimeError):
    def __init__(self, operation: str, status: int, message: str):
        self.status = status
        super().__init__(f"{operation} failed with CUDA status {status}: {message}")


class _AttributeValue(C.Union):
    _fields_ = [("storage", C.c_uint64 * 8), ("cluster", C.c_uint32 * 3)]


class _Attribute(C.Structure):
    _fields_ = [("id", C.c_int), ("pad", C.c_uint32), ("value", _AttributeValue)]


class _Config(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "gx", "gy", "gz", "bx", "by", "bz", "smem"
    )] + [("stream", C.c_void_p), ("attrs", C.POINTER(_Attribute)), ("count", C.c_uint32)]


_END_CAPTURE = [C.c_void_p, C.POINTER(C.c_void_p)]
_PEER_ACCESS_ALREADY_ENABLED = 704  # CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED


class CudaBackend:
    """A loaded kernel, its device context, and one setup stream.

    A current context of the requested device is borrowed; otherwise the
    device primary context is retained, so ordinary runtime caller allocations
    remain interoperable. The context is never left pushed: every method runs
    inside `current()`, so several backends (one per device) coexist in one
    thread. Context-loss paths do not free possibly in-flight addresses.

    `stream` is the backend's own stream: the setup stream for open-time work
    and the default launch stream for the CLI. `zero` and `launch` run on the
    caller's stream; `synchronize` drains the whole context so open/close order
    against every stream an integrator may have launched on. `capture` records
    a body issued on one stream into a replayable `Graph`.
    """

    def __init__(self, cubin: str | Path, symbol: str, *, device: int = 0,
                 dynamic_smem_bytes: int = 0, cluster: tuple[int, int, int] = (1, 1, 1),
                 block: tuple[int, int, int] = (1, 1, 1)):
        self.driver = C.CDLL("libcuda.so.1")
        self._functions: dict[str, Any] = {}
        self._allocations: dict[int, int] = {}
        self._graphs: list[Graph] = []
        self.context = C.c_void_p()
        self.stream = C.c_void_p()
        self.module = C.c_void_p()
        self.function = C.c_void_p()
        self.device = C.c_int()
        self._retained = False
        self.lost = False
        self.closed = False
        try:
            self._call("cuInit", [C.c_uint], 0)
            self._call("cuDeviceGet", [C.POINTER(C.c_int), C.c_int], C.byref(self.device), device)
            self._call("cuCtxGetCurrent", [C.POINTER(C.c_void_p)], C.byref(self.context))
            if self.context.value:
                current_device = C.c_int()
                self._call("cuCtxGetDevice", [C.POINTER(C.c_int)], C.byref(current_device))
                if current_device.value != self.device.value:
                    self.context = C.c_void_p()
            if not self.context.value:
                self._call("cuDevicePrimaryCtxRetain", [C.POINTER(C.c_void_p), C.c_int],
                           C.byref(self.context), self.device)
                self._retained = True
            with self.current():
                self._setup(cubin, symbol, device, dynamic_smem_bytes, cluster, block)
        except BaseException:
            self.close()
            raise

    def _setup(self, cubin: str | Path, symbol: str, device: int, dynamic_smem_bytes: int,
               cluster: tuple[int, int, int], block: tuple[int, int, int]) -> None:
        self._call("cuStreamCreate", [C.POINTER(C.c_void_p), C.c_uint], C.byref(self.stream), 0)
        self._call("cuModuleLoad", [C.POINTER(C.c_void_p), C.c_char_p],
                   C.byref(self.module), str(cubin).encode())
        self._call("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                   C.byref(self.function), self.module, symbol.encode())
        self._call("cuFuncSetAttribute", [C.c_void_p, C.c_int, C.c_int],
                   self.function, 8, dynamic_smem_bytes)
        version = C.c_int()
        self._call("cuDriverGetVersion", [C.POINTER(C.c_int)], C.byref(version))
        driver_text = Path("/proc/driver/nvidia/version").read_text()
        build = re.search(r"Kernel Module.*?\s(\d+\.\d+\.\d+)\s", driver_text)
        if build is None:
            raise ValueError("cannot establish exact NVIDIA driver build identity")
        arch = f"sm{self._attribute(75)}{self._attribute(76)}"
        self.observations = {
            "device_ordinal": device, "observed_sm_count": self._attribute(16),
            "arch": arch, "toolkit": "13.3", "driver_api": version.value,
            "driver_identity": build.group(1),
            "max_grid": tuple(self._attribute(i) for i in (5, 6, 7)),
            "max_block": tuple(self._attribute(i) for i in (2, 3, 4)),
            "max_threads": self._attribute(1),
            **{f"cluster_{axis}": value for axis, value in zip("xyz", cluster)},
        }
        self.identity = {"target": arch, "toolkit": "13.3", "driver_api": version.value,
                         "driver_build": build.group(1)}
        config, attribute = self._config(cluster, block, cluster, dynamic_smem_bytes, self.stream_token)
        active = C.c_int()
        self._call("cuOccupancyMaxActiveClusters", [C.POINTER(C.c_int), C.c_void_p, C.POINTER(_Config)],
                   C.byref(active), self.function, C.byref(config))
        self.observations["max_active_clusters"] = active.value

    @property
    def stream_token(self) -> int:
        return self.stream.value or 0

    @contextmanager
    def current(self) -> Iterator[None]:
        """Make this backend's context current for the body (raw driver calls, providers)."""
        self._call("cuCtxPushCurrent_v2", [C.c_void_p], self.context)
        try:
            yield
        finally:
            prior = C.c_void_p()
            self._call("cuCtxPopCurrent_v2", [C.POINTER(C.c_void_p)], C.byref(prior))

    @staticmethod
    def enable_peer_access(backends: Sequence[CudaBackend]) -> None:
        """Let every backend's context address every other device's allocations."""
        for backend in backends:
            for other in backends:
                if other.device.value == backend.device.value:
                    continue
                can = C.c_int()
                backend._call("cuDeviceCanAccessPeer", [C.POINTER(C.c_int), C.c_int, C.c_int],
                              C.byref(can), backend.device, other.device)
                if not can.value:
                    raise ValueError(f"device {backend.device.value} cannot access peer device {other.device.value}")
                with backend.current():
                    status = backend._function("cuCtxEnablePeerAccess", [C.c_void_p, C.c_uint])(other.context, 0)
                    if status not in (0, _PEER_ACCESS_ALREADY_ENABLED):
                        backend._raise("cuCtxEnablePeerAccess", status)

    def _function(self, name: str, types: list[Any]) -> Any:
        function = self._functions.get(name)
        if function is None:
            function = getattr(self.driver, name)
            function.argtypes = types
            function.restype = C.c_int
            self._functions[name] = function
        return function

    def _call(self, name: str, types: list[Any], *args: Any) -> None:
        status = self._function(name, types)(*args)
        if status:
            self._raise(name, status)

    def _raise(self, name: str, status: int) -> None:
        message = C.c_char_p()
        describe = self.driver.cuGetErrorString
        describe.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
        describe.restype = C.c_int
        describe(status, C.byref(message))
        raise CudaError(name, status, (message.value or b"unknown").decode())

    def _attribute(self, number: int) -> int:
        value = C.c_int()
        self._call("cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int],
                   C.byref(value), number, self.device)
        return value.value

    def allocate(self, size: int, align: int) -> int:
        if type(size) is not int or size < 0 or type(align) is not int or align <= 0 or align & (align - 1):
            raise ValueError("allocation size/alignment invalid")
        if not size:
            return 0
        address = C.c_uint64()
        with self.current():
            self._call("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t],
                       C.byref(address), size + align - 1)
        aligned = (address.value + align - 1) & -align
        self._allocations[aligned] = address.value
        return aligned

    def free(self, address: int) -> None:
        if address and not self.lost:
            base = self._allocations[address]
            with self.current():
                self._call("cuMemFree_v2", [C.c_uint64], base)
            del self._allocations[address]

    def zero(self, address: int, size: int, stream: int) -> None:
        if size:
            with self.current():
                self._call("cuMemsetD8Async", [C.c_uint64, C.c_ubyte, C.c_size_t, C.c_void_p],
                           address, 0, size, stream)

    def upload(self, address: int, data: bytes) -> None:
        if data:
            with self.current():
                self._call("cuMemcpyHtoD_v2", [C.c_uint64, C.c_void_p, C.c_size_t],
                           address, C.c_char_p(data), len(data))

    def readback(self, address: int, size: int) -> bytes:
        data = C.create_string_buffer(size)
        if size:
            with self.current():
                self._call("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t], data, address, size)
        return data.raw

    def synchronize(self) -> None:
        with self.current():
            self._call("cuCtxSynchronize", [])

    def event_create(self) -> int:
        event = C.c_void_p()
        with self.current():
            self._call("cuEventCreate", [C.POINTER(C.c_void_p), C.c_uint], C.byref(event), 0)
        return event.value

    def event_record(self, event: int, stream: int) -> None:
        with self.current():
            self._call("cuEventRecord", [C.c_void_p, C.c_void_p], event, stream)

    def event_synchronize(self, event: int) -> None:
        with self.current():
            self._call("cuEventSynchronize", [C.c_void_p], event)

    def event_elapsed_time(self, start: int, end: int) -> float:
        milliseconds = C.c_float()
        with self.current():
            self._call("cuEventElapsedTime", [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
                       C.byref(milliseconds), start, end)
        return milliseconds.value

    def event_destroy(self, event: int) -> None:
        with self.current():
            self._call("cuEventDestroy_v2", [C.c_void_p], event)

    def abandon(self) -> None:
        # The caller receives an error and cannot reuse the failed context.
        # Do not pretend a failed synchronization made its allocations reusable.
        self.lost = True

    def _config(self, grid: tuple[int, int, int], block: tuple[int, int, int],
                cluster: tuple[int, int, int], smem: int, stream: int) -> tuple[_Config, _Attribute]:
        attr = _Attribute()
        attr.id = 4
        attr.value.cluster[:] = cluster
        return _Config(*grid, *block, smem, stream, C.pointer(attr), 1), attr

    def launch(self, params: Any, geometry: Any, stream: int) -> None:
        dims = lambda v: (v.x, v.y, v.z)
        config, attr = self._config(dims(geometry.grid), dims(geometry.block),
                                    dims(geometry.cluster), geometry.dynamic_smem_bytes, stream)
        arguments = (C.c_void_p * 1)(params.address) if params.size else None
        with self.current():
            self._call("cuLaunchKernelEx", [C.POINTER(_Config), C.c_void_p,
                                          C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)],
                       C.byref(config), self.function, arguments, None)

    def capture(self, stream: int, body: Callable[[], None]) -> Graph:
        graph, executable = C.c_void_p(), C.c_void_p()
        with self.current():
            self._call("cuStreamBeginCapture_v2", [C.c_void_p, C.c_uint], stream, 0)
            try:
                body()
            except BaseException:
                self._function("cuStreamEndCapture", _END_CAPTURE)(stream, C.byref(graph))
                self._destroy_graph(graph)
                raise
            try:
                self._call("cuStreamEndCapture", _END_CAPTURE, stream, C.byref(graph))
                self._call("cuGraphInstantiateWithFlags", [C.POINTER(C.c_void_p), C.c_void_p, C.c_ulonglong],
                           C.byref(executable), graph, 0)
            except BaseException:
                self._destroy_graph(graph)
                raise
        result = Graph(self, graph.value, executable.value)
        self._graphs.append(result)
        return result

    def _destroy_graph(self, graph: C.c_void_p) -> None:
        if graph.value:
            self._function("cuGraphDestroy", [C.c_void_p])(graph)

    def __enter__(self) -> CudaBackend:
        return self

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        errors: list[Exception] = []
        if self.context.value:
            try:
                with self.current():
                    self._release_context_objects(errors)
            except Exception as exc:
                errors.append(exc)
        if self._retained:
            # The context is no longer on this thread's stack when its reference is dropped.
            operations = []
            if self.lost:
                # This backend retained the primary context. Failed stream work
                # is disposed through context teardown, never individual frees.
                operations.append(("cuDevicePrimaryCtxReset_v2", [C.c_int], (self.device,)))
            operations.append(("cuDevicePrimaryCtxRelease_v2", [C.c_int], (self.device,)))
            for name, types, args in operations:
                try:
                    self._call(name, types, *args)
                except Exception as exc:
                    errors.append(exc)
        self._allocations.clear()
        if errors:
            raise ExceptionGroup("CUDA backend cleanup failed", errors)

    def _release_context_objects(self, errors: list[Exception]) -> None:
        """Drain, then destroy graphs, allocations, module and stream (this context is current)."""
        if self.stream.value and not self.lost:
            try:
                self.synchronize()
            except Exception as exc:
                self.abandon()
                errors.append(exc)
        for graph in self._graphs[::-1]:
            try:
                graph.close()
            except Exception as exc:
                errors.append(exc)
        if self.lost:
            return
        operations = [("cuMemFree_v2", [C.c_uint64], (base,)) for base in self._allocations.values()]
        if self.module.value:
            operations.append(("cuModuleUnload", [C.c_void_p], (self.module,)))
        if self.stream.value:
            operations.append(("cuStreamDestroy_v2", [C.c_void_p], (self.stream,)))
        for name, types, args in operations:
            try:
                self._call(name, types, *args)
            except Exception as exc:
                errors.append(exc)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.close()
        except Exception as cleanup:
            if exc is not None:
                raise ExceptionGroup("CUDA operation and cleanup failed", [exc, cleanup]) from None
            raise


class Graph:
    """An instantiated CUDA graph replaying exactly the work captured on one stream."""

    def __init__(self, backend: CudaBackend, graph: int, executable: int):
        self._backend = backend
        self._graph = graph
        self.executable = executable
        self.closed = False

    def launch(self, stream: int) -> None:
        with self._backend.current():
            self._backend._call("cuGraphLaunch", [C.c_void_p, C.c_void_p], self.executable, stream)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._backend._graphs.remove(self)
        if self._backend.lost:
            return
        with self._backend.current():
            try:
                self._backend._call("cuGraphExecDestroy", [C.c_void_p], self.executable)
            finally:
                self._backend._call("cuGraphDestroy", [C.c_void_p], self._graph)

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class _MemLocation(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]


class _AllocationProp(C.Structure):
    _fields_ = [("type", C.c_int), ("handle_types", C.c_int), ("location", _MemLocation),
                ("win32", C.c_void_p), ("compression", C.c_ubyte), ("rdma", C.c_ubyte),
                ("usage", C.c_ushort), ("reserved", C.c_ubyte * 4)]


class _AccessDesc(C.Structure):
    _fields_ = [("location", _MemLocation), ("flags", C.c_int)]


class _MulticastProp(C.Structure):
    _fields_ = [("devices", C.c_uint), ("size", C.c_size_t), ("handle_types", C.c_ulonglong),
                ("flags", C.c_ulonglong)]


_POSIX_FD, _PINNED, _DEVICE, _READ_WRITE, _RECOMMENDED = 1, 1, 1, 3, 1
_U64, _SIZE = C.c_uint64, C.c_size_t


class MulticastGroup:
    """In-process caller fabrication of one multicast object over N backends' devices.

    Each device gets a VMM backing of `size` bytes (rounded to the multicast granularity),
    mapped once with read/write access for every member device, so `unicast` doubles as
    the peer table; every device also maps the multicast object (`multicast[r]`). The
    backing is bound once at its granularity; channels lend subviews through `views`.
    POSIX_FD handle types match the evidenced driver path; nothing is exported.
    """

    def __init__(self, backends: Sequence[CudaBackend], size: int):
        self.backends = list(backends)
        self.unicast: list[int] = []
        self.multicast: list[int] = []
        self._handles: list[int] = []
        self._maps: list[tuple[CudaBackend, int]] = []
        self._bound: list[CudaBackend] = []
        self._mc = 0
        lead = self.backends[0]
        try:
            prop = _MulticastProp(len(self.backends), size, _POSIX_FD, 0)
            granularity = _SIZE()
            with lead.current():
                lead._call("cuMulticastGetGranularity", [C.POINTER(_SIZE), C.POINTER(_MulticastProp), C.c_int],
                           C.byref(granularity), C.byref(prop), _RECOMMENDED)
            self.granularity = granularity.value
            self.size = prop.size = -(-size // self.granularity) * self.granularity
            mc = _U64()
            with lead.current():
                lead._call("cuMulticastCreate", [C.POINTER(_U64), C.POINTER(_MulticastProp)], C.byref(mc), C.byref(prop))
            self._mc = mc.value
            for backend in self.backends:
                with backend.current():
                    backend._call("cuMulticastAddDevice", [_U64, C.c_int], self._mc, backend.device)
            access = (_AccessDesc * len(self.backends))(
                *(_AccessDesc(_MemLocation(_DEVICE, b.device.value), _READ_WRITE) for b in self.backends))
            for backend in self.backends:
                location = _MemLocation(_DEVICE, backend.device.value)
                with backend.current():
                    handle = _U64()
                    backend._call("cuMemCreate", [C.POINTER(_U64), _SIZE, C.POINTER(_AllocationProp), C.c_ulonglong],
                                  C.byref(handle), self.size, C.byref(_AllocationProp(_PINNED, _POSIX_FD, location)), 0)
                    self._handles.append(handle.value)
                    self.unicast.append(self._map(backend, handle.value, access))
            for backend in self.backends:
                with backend.current():
                    backend._call("cuMulticastBindMem", [_U64, _SIZE, _U64, _SIZE, _SIZE, C.c_ulonglong],
                                  self._mc, 0, self._handles[len(self._bound)], 0, self.size, 0)
                    self._bound.append(backend)
            for backend in self.backends:
                own = (_AccessDesc * 1)(_AccessDesc(_MemLocation(_DEVICE, backend.device.value), _READ_WRITE))
                with backend.current():
                    self.multicast.append(self._map(backend, self._mc, own))
        except BaseException:
            self.close()
            raise

    def _map(self, backend: CudaBackend, handle: int, access: Any) -> int:
        address = _U64()
        backend._call("cuMemAddressReserve", [C.POINTER(_U64), _SIZE, _SIZE, _U64, C.c_ulonglong],
                      C.byref(address), self.size, self.granularity, 0, 0)
        self._maps.append((backend, address.value))
        backend._call("cuMemMap", [_U64, _SIZE, _SIZE, _U64, C.c_ulonglong], address, self.size, 0, handle, 0)
        backend._call("cuMemSetAccess", [_U64, _SIZE, C.POINTER(_AccessDesc), _SIZE],
                      address, self.size, access, len(access))
        return address.value

    def views(self, offset: int, allocation_bytes: int, *, domain: str, extents: tuple[int, ...],
              bit_strides: tuple[int, ...], dtype_id: int, alignment: int) -> list:
        """One Resource per group rank for the subview [offset, offset+allocation_bytes)."""
        from .fixture import Resource
        if offset < 0 or offset + allocation_bytes > self.size:
            raise ValueError("multicast subview exceeds the bound backing")
        peers = tuple(base + offset for base in self.unicast)
        return [Resource(peers[r], allocation_bytes, extents, bit_strides, dtype_id, alignment, domain=domain,
                         peers=peers, multicast=self.multicast[r] + offset, lender=self)
                for r in range(len(self.backends))]

    def close(self) -> None:
        """Unmap, unbind and release; the caller drained every member first."""
        errors: list[Exception] = []
        def attempt(backend: CudaBackend, name: str, types: list[Any], *args: Any) -> None:
            try:
                with backend.current():
                    backend._call(name, types, *args)
            except Exception as exc:
                errors.append(exc)
        for backend, address in reversed(self._maps):
            attempt(backend, "cuMemUnmap", [_U64, _SIZE], address, self.size)
            attempt(backend, "cuMemAddressFree", [_U64, _SIZE], address, self.size)
        for backend in self._bound:
            attempt(backend, "cuMulticastUnbind", [_U64, C.c_int, _SIZE, _SIZE], self._mc, backend.device, 0, self.size)
        for backend, handle in zip(self.backends, self._handles):
            attempt(backend, "cuMemRelease", [_U64], handle)
        if self._mc:
            attempt(self.backends[0], "cuMemRelease", [_U64], self._mc)
        self._maps, self._bound, self._handles, self._mc = [], [], [], 0
        if errors:
            raise ExceptionGroup("multicast group cleanup failed", errors)

    def __enter__(self) -> MulticastGroup:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
