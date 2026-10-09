"""Hostless owner params for the Hopper GEMM stages; no CUDA work at import."""
from __future__ import annotations

import ctypes as C
from typing import Mapping

from tools.tensormap import encode_tiled


class GemmLoadParams(C.Structure):
    # Explicit tail padding mirrors alignas(128).
    _fields_ = [
        ("a", C.c_uint64 * 16),
        ("b", C.c_uint64 * 16),
        ("K", C.c_int32),
        ("_padding", C.c_ubyte * 124),
    ]


class GemmMmaParams(C.Structure):
    _fields_ = [("K", C.c_int32)]


class GemmEpilogueParams(C.Structure):
    _fields_ = [
        ("output", C.c_void_p),
        ("bias", C.c_void_p),
        ("scale", C.c_void_p),
        ("M", C.c_int32),
        ("N", C.c_int32),
    ]


def build_load(addresses: Mapping[str, int], *, M: int, N: int, K: int) -> bytes:
    """Encode both rank-two BF16/SW128 maps with the actual current CUDA driver.

    Called only after open_hostless has selected the device, made its context
    current, and allocated A[M,K] and B[N,K].
    """
    params = GemmLoadParams()
    for name, rows in (("a", M), ("b", N)):
        image = encode_tiled(addresses[name], dtype=9, dims=(K, rows),
                             strides_bytes=(K * 2,), box=(64, 128),
                             swizzle=3, l2_promotion=2)
        C.memmove(C.addressof(params) + getattr(GemmLoadParams, name).offset,
                  image, len(image))
    params.K = K
    return bytes(params)


def build_mma(*, K: int) -> bytes:
    return bytes(GemmMmaParams(K))


def build_epilogue(addresses: Mapping[str, int], *, M: int, N: int,
                   epilogue: str) -> bytes:
    return bytes(GemmEpilogueParams(
        addresses["output"],
        addresses["bias"] if epilogue == "bias_relu" else 0,
        addresses["scale"] if epilogue == "rowscale" else 0,
        M, N,
    ))


def build(addresses: Mapping[str, int], *, M: int, N: int, K: int,
          epilogue: str) -> dict[str, bytes]:
    """Bind independent load, MMA, and epilogue owner blobs after allocation."""
    return {
        "gemm.load": build_load(addresses, M=M, N=N, K=K),
        "gemm.mma": build_mma(K=K),
        f"gemm.epilogue.{epilogue.replace('_', '-')}": build_epilogue(
            addresses, M=M, N=N, epilogue=epilogue,
        ),
    }
