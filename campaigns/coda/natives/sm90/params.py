"""Producer-owned native argument images; caller makes its CUDA context current."""
import ctypes as C
from tools.tensormap import encode_tiled


def arguments(kind, addresses, M, N, K, stagger_ns=None):
    descriptors = b"".join(
        encode_tiled(addresses[key], dtype=9, dims=(K, rows), strides_bytes=(K * 2,),
                     box=(64, 64), swizzle=3, l2_promotion=0)
        for key, rows in (("x", M), ("w", N))
    )
    columns = N if kind == "unfused_gemm" else N // 2
    output = encode_tiled(addresses["d" if kind == "unfused_gemm" else "out"],
                          dtype=9, dims=(columns, M), strides_bytes=(columns * 2,),
                          box=(64, 64), swizzle=3, l2_promotion=0)
    return [C.create_string_buffer(descriptors, 256), C.create_string_buffer(output, 128),
            C.c_int(M), C.c_int(N), C.c_int(K)]
