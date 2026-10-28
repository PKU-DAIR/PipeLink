from __future__ import annotations

import ctypes as C
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipec2.host import cuda


class Driver:
    """In-memory driver resources with context and handle-lifetime enforcement."""

    def __init__(self):
        self.contexts = [None]
        self.next_handle = 100
        self.modules = {}
        self.functions = {}
        self.streams = set()
        self.memory = {}
        self.graphs = set()
        self.executables = set()
        self.retains = 0
        self.pending = False
        self.launches = []
        self.loaded_paths = []
        self.unloaded = []
        self.reload_old = set()
        self.failure = None

    def handle(self):
        self.next_handle += 1
        return self.next_handle

    def __getattr__(self, name):
        def operation(*args):
            self.call(name, *args)
            return 0
        return operation

    def call(self, name, *args):
        if self.failure is not None and self.failure[0] == name:
            if self.failure[1] == 0:
                self.failure = None
                raise cuda.CudaError(name, 999, 'injected failure')
            self.failure = (name, self.failure[1] - 1)
        if name == 'cuInit':
            return
        if name == 'cuDeviceGet':
            args[0]._obj.value = args[1]
        elif name == 'cuCtxGetCurrent':
            args[0]._obj.value = self.contexts[-1]
        elif name == 'cuCtxGetDevice':
            args[0]._obj.value = 0
        elif name == 'cuDevicePrimaryCtxRetain':
            args[0]._obj.value = 7
            self.retains += 1
        elif name == 'cuDevicePrimaryCtxRelease_v2':
            self.retains -= 1
        elif name == 'cuCtxPushCurrent_v2':
            self.contexts.append(args[0].value)
        elif name == 'cuCtxPopCurrent_v2':
            args[0]._obj.value = self.contexts.pop()
        else:
            assert self.contexts[-1] == 7, f'{name} requires the backend context'
            self.context_call(name, *args)

    def context_call(self, name, *args):
        if name == 'cuStreamCreate':
            args[0]._obj.value = self.handle()
            self.streams.add(args[0]._obj.value)
        elif name == 'cuModuleLoad':
            assert not (self.reload_old & self.modules.keys()), 'unload all old modules first'
            args[0]._obj.value = self.handle()
            self.modules[args[0]._obj.value] = args[1]
            self.loaded_paths.append(args[1])
        elif name == 'cuModuleGetFunction':
            assert args[1].value in self.modules
            args[0]._obj.value = self.handle()
            self.functions[args[0]._obj.value] = {
                'module': args[1].value, 'symbol': args[2], 'smem': None, 'loaded': False,
            }
        elif name == 'cuFuncSetAttribute':
            assert args[1] == 8
            self.functions[args[0].value]['smem'] = args[2]
        elif name == 'cuFuncLoad':
            self.functions[args[0].value]['loaded'] = True
        elif name == 'cuDriverGetVersion':
            args[0]._obj.value = 13030
        elif name == 'cuDeviceGetAttribute':
            args[0]._obj.value = {75: 10, 76: 0}.get(args[1], 128)
        elif name == 'cuOccupancyMaxActiveClusters':
            args[0]._obj.value = 16
        elif name == 'cuMemAlloc_v2':
            address = self.handle() * 256
            args[0]._obj.value = address
            self.memory[address] = bytearray(args[1])
        elif name == 'cuMemcpyHtoD_v2':
            self.memory[args[0]][:args[2]] = C.string_at(args[1], args[2])
        elif name == 'cuMemcpyDtoH_v2':
            C.memmove(args[0], bytes(self.memory[args[1]][:args[2]]), args[2])
        elif name == 'cuLaunchKernelEx':
            config = args[0]._obj
            function = args[1].value
            assert self.functions[function]['module'] in self.modules
            assert config.stream in self.streams
            self.launches.append((function, config.stream))
            self.pending = True
        elif name == 'cuCtxSynchronize':
            self.pending = False
        elif name == 'cuModuleUnload':
            assert not self.pending, 'synchronize before unloading kernels'
            module = args[0].value
            del self.modules[module]
            self.unloaded.append(module)
            self.functions = {key: value for key, value in self.functions.items()
                              if value['module'] != module}
        elif name == 'cuMemFree_v2':
            del self.memory[args[0]]
        elif name == 'cuStreamDestroy_v2':
            self.streams.remove(args[0].value)
        elif name == 'cuStreamBeginCapture_v2':
            assert args[0] in self.streams
        elif name == 'cuStreamEndCapture':
            args[1]._obj.value = self.handle()
            self.graphs.add(args[1]._obj.value)
        elif name == 'cuGraphInstantiateWithFlags':
            assert args[1].value in self.graphs
            args[0]._obj.value = self.handle()
            self.executables.add(args[0]._obj.value)
        elif name == 'cuGraphExecDestroy':
            self.executables.remove(args[0])
        elif name == 'cuGraphDestroy':
            self.graphs.remove(args[0])
        else:
            raise AssertionError(f'unhandled driver operation: {name}')


@pytest.fixture
def driver(monkeypatch):
    driver = Driver()
    monkeypatch.setattr(cuda.C, 'CDLL', lambda _: driver)
    read_text = Path.read_text

    def driver_version(path, *args, **kwargs):
        if str(path) == '/proc/driver/nvidia/version':
            return 'NVRM version: NVIDIA UNIX Open Kernel Module 580.65.06 test'
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', driver_version)
    return driver


def launch(backend):
    dims = SimpleNamespace(x=1, y=1, z=1)
    geometry = SimpleNamespace(grid=dims, block=dims, cluster=dims, dynamic_smem_bytes=64)
    backend.launch(SimpleNamespace(address=0, size=0), geometry, backend.stream_token)


@pytest.mark.parametrize('borrowed', [False, True])
def test_reload_preserves_resources_and_launches_new_function(driver, borrowed):
    driver.contexts[-1] = 7 if borrowed else None
    with ExitStack() as stack:
        backends = [stack.enter_context(cuda.CudaBackend(f'{name}.cubin', name, dynamic_smem_bytes=64))
                    for name in ('linked', 'control')]
        resources = []
        for backend in backends:
            address = backend.allocate(8, 16)
            backend.upload(address, b'preserve')
            resources.append((backend.context, backend.stream, backend._allocations, address,
                              backend._retained, backend.observations, backend.identity))
            launch(backend)
        old_modules = [backend.module.value for backend in backends]
        old_functions = [backend.function.value for backend in backends]
        old_paths = driver.loaded_paths[-2:]
        driver.reload_old = set(old_modules)
        cuda.reload_modules(tuple(reversed(backends)))
        assert driver.contexts == [7 if borrowed else None]
        assert driver.unloaded == list(reversed(old_modules))
        assert driver.loaded_paths[-2:] == list(reversed(old_paths))
        for backend, old_function, resource in zip(backends, old_functions, resources):
            context, stream, allocations, address, retained, observations, identity = resource
            assert backend.context is context
            assert backend.stream is stream
            assert backend._retained == retained
            assert backend.observations is observations
            assert backend.identity is identity
            assert backend.readback(address, 8) == b'preserve'
            assert backend.function.value != old_function
            function = driver.functions[backend.function.value]
            assert function['loaded']
            assert function['symbol'] == backend._symbol.encode()
            assert function['smem'] == 64
            launch(backend)
            assert driver.launches[-1] == (backend.function.value, stream.value)
    assert not driver.modules and not driver.streams and not driver.memory
    assert driver.retains == 0


@pytest.mark.parametrize('invalid', ['duplicate', 'closed', 'lost', 'context', 'no_context', 'graph'])
def test_reload_validates_entire_sequence_before_mutation(driver, invalid):
    with ExitStack() as stack:
        first = stack.enter_context(cuda.CudaBackend('first.cubin', 'first'))
        second = stack.enter_context(cuda.CudaBackend('second.cubin', 'second'))
        sequence = (first, second)
        if invalid == 'duplicate':
            sequence = (first, second, first)
        elif invalid == 'closed':
            second.close()
        elif invalid == 'lost':
            second.abandon()
        elif invalid == 'context':
            second.context.value = 8
        elif invalid == 'no_context':
            second.context.value = None
        else:
            second.capture(second.stream_token, lambda: launch(second))
        modules = driver.modules.copy()
        functions = {key: value.copy() for key, value in driver.functions.items()}
        unloads = driver.unloaded.copy()
        loads = driver.loaded_paths.copy()
        launch(first)
        try:
            with pytest.raises(ValueError):
                cuda.reload_modules(sequence)
            assert driver.modules == modules and driver.functions == functions
            assert driver.unloaded == unloads and driver.loaded_paths == loads
            assert driver.pending, 'validation must precede even synchronization'
            assert driver.contexts == [None]
        finally:
            # Restore deliberately injected invalid state so real cleanup remains usable.
            second.lost = False
            second.context.value = 7
    assert not driver.modules and not driver.streams and not driver.memory
    assert not driver.graphs and not driver.executables


def test_closed_graph_does_not_prevent_reload(driver):
    with cuda.CudaBackend('kernel.cubin', 'kernel') as backend:
        with backend.capture(backend.stream_token, lambda: launch(backend)):
            pass
        cuda.reload_modules((backend,))
        assert driver.functions[backend.function.value]['loaded']


@pytest.mark.parametrize(('operation', 'successful_calls'), [
    ('cuCtxSynchronize', 0),
    *[(operation, count) for operation in (
        'cuModuleUnload', 'cuModuleLoad', 'cuModuleGetFunction', 'cuFuncSetAttribute', 'cuFuncLoad',
    ) for count in (0, 1)],
])
def test_partial_reload_failure_leaves_all_resources_cleanable(driver, operation, successful_calls):
    with ExitStack() as stack:
        backends = [stack.enter_context(cuda.CudaBackend(f'{name}.cubin', name))
                    for name in ('first', 'second')]
        addresses = [backend.allocate(8, 16) for backend in backends]
        for backend, address in zip(backends, addresses):
            backend.upload(address, b'preserve')
        old_modules = {backend.module.value for backend in backends}
        driver.failure = (operation, successful_calls)
        with pytest.raises(cuda.CudaError, match=operation):
            cuda.reload_modules(backends)
        assert driver.contexts == [None]
        assert {backend.module.value for backend in backends if backend.module.value} == driver.modules.keys()
        for backend, address in zip(backends, addresses):
            assert backend.readback(address, 8) == b'preserve'
        # ExitStack now owns every old or partially loaded module and every allocation.
    assert not driver.modules and not driver.functions and not driver.memory and not driver.streams
    assert driver.retains == 0
    assert old_modules <= set(driver.unloaded)
    assert len(driver.unloaded) == len(set(driver.unloaded))
    for backend in backends:
        backend.close()  # Ordinary close remains idempotent after partial reload.


def test_empty_reload_is_noop(driver):
    cuda.reload_modules(())
    assert driver.contexts == [None]
    assert not driver.loaded_paths and not driver.unloaded
