"""Public host contract inventories and non-gating toolchain provenance."""

from __future__ import annotations

import ctypes as ct
import sys


from ..canonjson import canonical_digest


HOST_ABI = "linux-lp64-le-v2"
DTYPES = {
    "u8": {"id": 1, "bits": 8, "category": "unsigned"},
    "i8": {"id": 2, "bits": 8, "category": "signed"},
    "u16": {"id": 3, "bits": 16, "category": "unsigned"},
    "i16": {"id": 4, "bits": 16, "category": "signed"},
    "u32": {"id": 5, "bits": 32, "category": "unsigned"},
    "i32": {"id": 6, "bits": 32, "category": "signed"},
    "bf16": {"id": 7, "bits": 16, "category": "bfloat"},
    "f16": {"id": 8, "bits": 16, "category": "ieee-float"},
    "f32": {"id": 9, "bits": 32, "category": "ieee-float"},
    "f64": {"id": 10, "bits": 64, "category": "ieee-float"},
    "u64": {"id": 11, "bits": 64, "category": "unsigned"},
    "i64": {"id": 12, "bits": 64, "category": "signed"},
    "s4": {"id": 13, "bits": 4, "category": "signed"},
    "u4": {"id": 14, "bits": 4, "category": "unsigned"},
}
TOOLCHAIN_PROVENANCE = {
    "host_compiler": "GCC-11.4.0", "cuda_toolkit": "13.3",
    "cutlass_commit": "dcf215af68a2d08d305076c152a06f201728cd53",
}
HARDWARE_SCALARS = ["device_ordinal", "observed_sm_count", "max_active_clusters", "cluster_x", "cluster_y", "cluster_z"]

__all__ = [
    "HOST_ABI", "DTYPES", "HARDWARE_SCALARS", "TOOLCHAIN_PROVENANCE",
    "dtypes_digest", "validate_host_abi",
]


def dtypes_digest(profile_name: str) -> str:
    """Digest of the dtype inventory shared by every supported host profile."""
    return canonical_digest(DTYPES)


def validate_host_abi() -> None:
    if sys.platform != "linux" or sys.byteorder != "little" or any(ct.sizeof(typ) != 8 for typ in (ct.c_void_p, ct.c_size_t, ct.c_long, ct.c_double)):
        raise ValueError(f"unsupported host ABI: requires {HOST_ABI}")


