"""Encode CUDA tiled tensor maps after the caller makes a context current."""
from __future__ import annotations

import ctypes as C
from collections.abc import Sequence


def encode_tiled(address: int, *, dtype: int, dims: Sequence[int],
                 strides_bytes: Sequence[int], box: Sequence[int], swizzle: int,
                 l2_promotion: int = 3, oob: int = 0) -> bytes:
    """Return a 128-byte CUtensorMap using CUDA enum values.

    Dimensions are innermost-first; strides omit the contiguous first dimension.
    Element strides are one, interleave is NONE, and the defaults are L2_256B
    and NONE OOB fill. CUDA validates addresses, extents and layout constraints.
    No driver library is loaded at module import time.
    """
    rank = len(dims)
    if not 1 <= rank <= 5 or len(strides_bytes) != rank - 1 or len(box) != rank:
        raise ValueError("Expected rank 1..5, rank-1 byte strides, and rank box extents")
    storage = C.create_string_buffer(128 + 127)
    aligned = (C.addressof(storage) + 127) & -128
    driver = C.CDLL("libcuda.so.1")
    encode = driver.cuTensorMapEncodeTiled
    encode.argtypes = [
        C.c_void_p, C.c_int, C.c_uint32, C.c_void_p,
        C.POINTER(C.c_uint64), C.POINTER(C.c_uint64),
        C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
        C.c_int, C.c_int, C.c_int, C.c_int,
    ]
    encode.restype = C.c_int
    status = encode(aligned, dtype, rank, address,
                    (C.c_uint64 * rank)(*dims),
                    (C.c_uint64 * (rank - 1))(*strides_bytes),
                    (C.c_uint32 * rank)(*box), (C.c_uint32 * rank)(*([1] * rank)),
                    0, swizzle, l2_promotion, oob)
    if status:
        describe = driver.cuGetErrorString
        describe.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
        describe.restype = C.c_int
        message = C.c_char_p()
        describe(status, C.byref(message))
        raise RuntimeError(f"cuTensorMapEncodeTiled failed with CUDA status {status}: "
                           f"{(message.value or b'unknown').decode()}")
    return C.string_at(aligned, 128)
