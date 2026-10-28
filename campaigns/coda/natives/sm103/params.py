"""Owner-side native argument encoding; caller must make its CUDA context current."""
import ctypes as C
import struct
from tools.tensormap import encode_tiled


def arguments(kind, addresses, M, N, K, stagger_ns=None):
    if kind not in {"fused", "unfused_gemm"}:
        raise ValueError(kind)
    driver = C.CDLL("libcuda.so.1")
    device, sms = C.c_int(), C.c_int()
    if driver.cuCtxGetDevice(C.byref(device)) or driver.cuDeviceGetAttribute(C.byref(sms), 16, device):
        raise RuntimeError("Cannot query the current device's SM count")
    image = bytearray(512)
    for offset, name, rows in ((0, "x", M), (128, "w", N)):
        image[offset:offset+128] = encode_tiled(addresses[name], dtype=9, dims=(K, rows),
            strides_bytes=(K*2,), box=(64,128), swizzle=3, l2_promotion=0)
    struct.pack_into("iI", image, 256, sms.value, 22500 if stagger_ns is None else stagger_ns)
    struct.pack_into("Qiii", image, 384, addresses["d" if kind == "unfused_gemm" else "out"], M, N, K)
    if kind == "unfused_gemm":
        image.extend(encode_tiled(addresses["d"], dtype=9, dims=(N, M),
            strides_bytes=(N*2,), box=(32,128), swizzle=2, l2_promotion=0))
    return [bytes(image)]
