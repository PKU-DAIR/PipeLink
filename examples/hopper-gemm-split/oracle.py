"""Independent cuBLAS oracle and BF16 inputs shared by both launch routes."""
from __future__ import annotations
import ctypes as C
import math
from pathlib import Path
import numpy as np

RTOL, ATOL = 1e-3, 1e-3

def _bind(library, name, types):
    function = getattr(library, name)
    function.argtypes = types
    function.restype = C.c_int
    return function


def _check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with status {status}")


class Cublas:
    """cuBLAS on borrowed allocations in the caller's current context.

    The caller exposes addresses and synchronize(). Synchronize before using
    the null stream, then time it with public CUDA driver events. This helper
    owns neither a CUDA context nor any device allocation.
    """

    def __init__(self, session, shape, cuda_home):
        self.session = session
        self.session.synchronize()
        libraries = sorted((Path(cuda_home) / "lib64").glob("libcublas.so*"))
        if not libraries:
            raise RuntimeError("PIPELINK_CUDA_HOME/lib64 has no libcublas.so library")
        self.library_name = libraries[0].name
        self.library = C.CDLL(str(libraries[0]))
        self.driver = C.CDLL("libcuda.so.1")
        pointer, integer = C.c_void_p, C.c_int
        create = _bind(self.library, "cublasCreate_v2", [C.POINTER(pointer)])
        self.destroy = _bind(self.library, "cublasDestroy_v2", [pointer])
        set_stream = _bind(self.library, "cublasSetStream_v2", [pointer, pointer])
        get_version = _bind(self.library, "cublasGetVersion_v2", [pointer, C.POINTER(integer)])
        self.gemm = _bind(self.library, "cublasGemmEx", [
            pointer, integer, integer, integer, integer, integer,
            pointer, pointer, integer, integer, pointer, integer, integer,
            pointer, pointer, integer, integer, integer, integer,
        ])
        self.event_create = _bind(self.driver, "cuEventCreate", [C.POINTER(pointer), C.c_uint])
        self.event_record = _bind(self.driver, "cuEventRecord", [pointer, pointer])
        self.event_sync = _bind(self.driver, "cuEventSynchronize", [pointer])
        self.event_elapsed = _bind(self.driver, "cuEventElapsedTime", [C.POINTER(C.c_float), pointer, pointer])
        self.event_destroy = _bind(self.driver, "cuEventDestroy_v2", [pointer])
        self.handle = pointer()
        _check(create(C.byref(self.handle)), "cublasCreate_v2")
        try:
            _check(set_stream(self.handle, None), "cublasSetStream_v2")
            version = integer()
            _check(get_version(self.handle, C.byref(version)), "cublasGetVersion_v2")
            self.version = version.value
        except BaseException:
            self.close()
            raise
        self.alpha, self.beta = C.c_float(1.0), C.c_float(0.0)
        M, N, K = shape
        addresses = session.addresses
        # Row-major C[M,N] is column-major C^T[N,M]. Thus compute B @ A^T
        # as op_T(B_col[K,N]) @ op_N(A_col[K,M]), leading dimensions K,K,N.
        self.arguments = (
            self.handle, 1, 0, N, M, K, C.byref(self.alpha),
            addresses["b"], 14, K, addresses["a"], 14, K,
            C.byref(self.beta), addresses["oracle"], 0, N, 68, 99,
        )  # CUDA_R_16BF -> CUDA_R_32F; CUBLAS_COMPUTE_32F; DEFAULT_TENSOR_OP.

    def launch(self):
        _check(self.gemm(*self.arguments), "cublasGemmEx")

    def time(self, repetitions):
        self.session.synchronize()
        start, end = C.c_void_p(), C.c_void_p()
        _check(self.event_create(C.byref(start), 0), "cuEventCreate")
        try:
            _check(self.event_create(C.byref(end), 0), "cuEventCreate")
            try:
                _check(self.event_record(start, None), "cuEventRecord")
                for _ in range(repetitions):
                    self.launch()
                _check(self.event_record(end, None), "cuEventRecord")
                _check(self.event_sync(end), "cuEventSynchronize")
                elapsed = C.c_float()
                _check(self.event_elapsed(C.byref(elapsed), start, end), "cuEventElapsedTime")
                if not math.isfinite(elapsed.value) or elapsed.value <= 0:
                    raise RuntimeError("cuBLAS event duration must be finite and positive")
                return elapsed.value
            finally:
                _check(self.event_destroy(end), "cuEventDestroy_v2")
        finally:
            _check(self.event_destroy(start), "cuEventDestroy_v2")

    def close(self):
        if self.handle.value:
            _check(self.destroy(self.handle), "cublasDestroy_v2")
            self.handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _inputs(shape):
    M, N, K = shape
    rng = np.random.default_rng(2026)

    def bf16(rows):
        # Round finite float32 inputs to BF16, ties to even; upload the raw bits.
        values = rng.uniform(-1.0, 1.0, (rows, K)).astype(np.float32)
        bits = values.view(np.uint32)
        return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

    return {
        "a": bf16(M), "b": bf16(N),
        "bias": rng.uniform(-0.5, 0.5, N).astype(np.float32),
        "scale": rng.uniform(-1.5, 1.5, M).astype(np.float32),
        "output": ((M, N), np.float32),
    }
