"""Host mirror of params.h; encode the TMA descriptor after device allocation."""
from __future__ import annotations

import ctypes
from typing import Mapping

ROWS = 32768
COLUMNS = 1024
SCALE = 2.0
BIAS = 1.0


class ConsumerParams(ctypes.Structure):
    _fields_ = [("output", ctypes.c_void_p), ("columns", ctypes.c_int),
                ("scale", ctypes.c_float), ("bias", ctypes.c_float)]


def build(addresses: Mapping[str, int], *, rows: int = ROWS,
          columns: int = COLUMNS) -> dict[str, bytes]:
    """Called by the host-less launcher in its current CUDA context.

    ProducerParams is 256 bytes aligned to 128: CUtensorMap at zero, int32
    columns at 128, then tail padding. The launcher aligns the aggregate blob.
    """
    driver = ctypes.CDLL("libcuda.so.1")
    encode = driver.cuTensorMapEncodeTiled
    encode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                       ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    encode.restype = ctypes.c_int
    storage = ctypes.create_string_buffer(256 + 127)
    aligned = (ctypes.addressof(storage) + 127) & ~127
    dims = (ctypes.c_uint64 * 2)(columns, rows)
    strides = (ctypes.c_uint64 * 1)(columns * 2)
    box = (ctypes.c_uint32 * 2)(64, 128)
    element_strides = (ctypes.c_uint32 * 2)(1, 1)
    # cuda.h: BFLOAT16=9, INTERLEAVE_NONE=0, SWIZZLE_128B=3,
    # L2_PROMOTION_L2_128B=2, FLOAT_OOB_FILL_NONE=0.
    status = encode(aligned, 9, 2, addresses["input"], dims, strides, box,
                    element_strides, 0, 3, 2, 0)
    if status:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: CUDA driver error {status}")
    ctypes.c_int.from_address(aligned + 128).value = columns
    return {
        "org.ring.producer": ctypes.string_at(aligned, 256),
        "org.ring.consumer": bytes(ConsumerParams(addresses["output"], columns, SCALE, BIAS)),
    }
