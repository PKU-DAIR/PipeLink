"""Owner-native launch arguments; the cp.async mainloop uses pointers, not TMA."""
import ctypes as ct


def arguments(kind, addresses, M, N, K, stagger_ns=None):
    output = addresses["d" if kind == "unfused_gemm" else "out"]
    return [ct.c_void_p(addresses["x"]), ct.c_void_p(addresses["w"]), ct.c_void_p(output),
            ct.c_int32(M), ct.c_int32(N), ct.c_int32(K)]
