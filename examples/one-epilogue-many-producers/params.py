"""Inputs, f64 oracle, and the consumer/params.h stage blob."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import struct

import numpy as np

EPS = np.float32(1e-5)


def inputs(m: int, n: int, k: int) -> dict[str, np.ndarray]:
    if min(m, n, k) <= 0 or m % 128 or n % 128 or k % 64:
        raise ValueError("Shape must contain full 128x128x64 tiles")
    # Match std::mt19937 + libstdc++ uniform_real_distribution<float>(-1, 1),
    # including its float32 rounding before the fp16 input conversion.
    random = np.random.RandomState(20260925)

    def uniform(shape):
        unit = random.randint(0, 2**32, size=shape, dtype=np.uint32).astype(np.float32)
        unit *= np.float32(2**-32)
        np.minimum(unit, np.nextafter(np.float32(1), np.float32(0)), out=unit)
        return unit * np.float32(2) - np.float32(1)

    return {
        "a": uniform((m, k)).astype(np.float16),
        "b": uniform((n, k)).astype(np.float16),
        "bias": uniform((m,)),
        "output": np.full((m, n), 0x7fffffff, dtype=np.uint32).view(np.float32),
    }


def oracle(buffers: Mapping[str, np.ndarray]) -> np.ndarray:
    a = buffers["a"].astype(np.float64)
    b = buffers["b"].astype(np.float64)
    scale = 1 / np.sqrt(np.sum(a * a, axis=1) / a.shape[1] + float(EPS))
    return np.maximum((a @ b.T) * scale[:, None] + buffers["bias"].astype(np.float64)[:, None], 0)


def _tensor_maps(addresses: Mapping[str, int], shape: tuple[int, int, int]) -> bytes:
    """A[M,K] and B[N,K] fp16 K-major maps, box 64x128, 128-byte swizzle (provider contract)."""
    from tools.tensormap import encode_tiled

    m, n, k = shape
    return b"".join(encode_tiled(addresses[name], dtype=6, dims=(k, rows), strides_bytes=(k * 2,),
                                 box=(64, 128), swizzle=3)
                    for name, rows in (("a", m), ("b", n)))


# Per profile: the hand accumulator provider (source, pack defines, exported placements)
# and the hand-fused native control. The keys are the example's supported profiles;
# sm_100a and sm_121a reuse their family sibling's sources.
_TCGEN05 = {"provider": "providers/accumulator/producer_source.cu",
            "defines": {"PL_TILE_N": "128", "PL_MMA_M": "128"}, "placements": ("tmem", "smem"),
            "native": "native/fused_sm100.cu"}
_MMA_SYNC = {"provider": "providers/accumulator/producer_sm120.cu",
             "defines": {"PL_TILE_N": "128"}, "placements": ("reg", "smem"),
             "native": "native/fused_sm120.cu"}
PROFILE_SOURCES = {
    "sm_103a": _TCGEN05,
    "sm_100a": _TCGEN05,
    "sm_90a": {"provider": "providers/accumulator/producer_sm90.cu",
               "defines": {"PL_TILE_N": "128"}, "placements": ("reg", "smem"),
               "native": "native/fused_sm90.cu"},
    "sm_120a": _MMA_SYNC,
    "sm_121a": _MMA_SYNC,
}

CONSUMERS = {"rowscale-relu-epilogue", "tensor.consumer.tilelang-epilogue",
             *(f"native.tilelang.gemm.{profile.replace('_', '')}" for profile in PROFILE_SOURCES)}
TMA_PROVIDERS = {"tensor.producer.m128n128", "tensor.producer.smem128n128",
                 "tensor.producer.sm90-reg128n128", "tensor.producer.sm90-smem128n128",
                 "tensor.producer.sm120-reg128n128", "tensor.producer.sm120-smem128n128"}
TMA_NATIVES = {"native.fused.sm100", "native.fused.sm90", "native.fused.sm120"}


def build_params(addresses: Mapping[str, int], *, entries: Sequence[Sequence],
                 shape: tuple[int, int, int]) -> dict[str, bytes]:
    # Consumer blob: four 64-bit pointers, three int32 dimensions, float32 epsilon (48 bytes).
    # TMA providers: two 128-byte tensor maps. TMA natives: maps, then the consumer blob.
    blob = struct.pack("<QQQQiiif", *(addresses[name] for name in ("a", "b", "bias", "output")),
                       *shape, EPS)
    blobs = {}
    for stage, _offset, size, _align in entries:
        if not size:
            blobs[stage] = b""
            continue
        if stage in CONSUMERS:
            value = blob
        elif stage in TMA_PROVIDERS:
            value = _tensor_maps(addresses, shape)
        elif stage in TMA_NATIVES:
            value = _tensor_maps(addresses, shape) + blob
        else:
            raise ValueError(f"No params builder for stage {stage!r}")
        blobs[stage] = value.ljust(size, b"\0")
    return blobs
