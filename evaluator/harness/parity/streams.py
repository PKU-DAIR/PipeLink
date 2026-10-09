"""Exact M3 §2.1 Philox streams, sharing classes, and stream-map writer."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Any, Iterable

from .constants import (
    ALPHA,
    AUTHORING_CORRECTNESS_ALPHA,
    BETA,
    F1_AUX_SENTINEL,
    INT4_CARDINALITY,
    INT4_SIGN_CUTOFF,
    NAMESPACE_IDS,
    PHILOX_LANES,
    PHILOX_M0,
    PHILOX_M1,
    PHILOX_ROUNDS,
    PHILOX_SEED,
    PHILOX_W0,
    PHILOX_W1,
    REGISTER_OUTPUT_INITIAL,
    SCHEMA_VERSIONS,
    UINT32_MASK,
    UNIFORM_SCALE,
    UNIFORM_WORD_OFFSET,
)
from .util import canonical_json_bytes, sha256_bytes, write_json_atomic


@dataclass(frozen=True)
class RoleSpec:
    name: str
    dtype: str
    distribution: str
    lower: float | None = None
    upper: float | None = None


_COMMON_ABC = {
    1: RoleSpec("A", "BF16", "U[-2,2)", -2.0, 2.0),
    2: RoleSpec("B", "BF16", "U[-2,2)", -2.0, 2.0),
    3: RoleSpec("C", "BF16", "U[-2,2)", -2.0, 2.0),
}

# M3 plan §2.1, lines 305-315: complete frozen role table.  F1 roles 4/5
# deliberately remain separate objects/classes (V3 columns versus V4 rows).
ROLE_TABLES: dict[str, dict[int, RoleSpec]] = {
    "F1": {
        **_COMMON_ABC,
        4: RoleSpec("V3-per-column-bias", "FP32", "U[-2,2)", -2.0, 2.0),
        5: RoleSpec("V4-per-row-bias", "FP32", "U[-2,2)", -2.0, 2.0),
    },
    "F2": {
        1: RoleSpec("Q", "FP16", "U[-2,2)", -2.0, 2.0),
        2: RoleSpec("K", "FP16", "U[-2,2)", -2.0, 2.0),
        3: RoleSpec("V", "FP16", "U[-2,2)", -2.0, 2.0),
    },
    "F3": {
        1: RoleSpec("activation", "BF16", "U[-2,2)", -2.0, 2.0),
        2: RoleSpec("signed-int4", "INT4", "signed nibble", None, None),
        3: RoleSpec("scale", "BF16", "U[0.5,1.5)", 0.5, 1.5),
        4: RoleSpec("C", "BF16", "U[-2,2)", -2.0, 2.0),
        5: RoleSpec("zero", "BF16", "U[-1,1)", -1.0, 1.0),
    },
    "A-specialization": dict(_COMMON_ABC),
    "A-solver": dict(_COMMON_ABC),
    "A-authoring": dict(_COMMON_ABC),
    "A-F1-continuity": dict(_COMMON_ABC),
    "N-register-coupling": {
        1: RoleSpec("score-coefficients", "FP32", "U[-1,1)", -1.0, 1.0),
        2: RoleSpec("local-P", "FP32", "U[-1,1)", -1.0, 1.0),
        3: RoleSpec("remote-P", "FP32", "U[-1,1)", -1.0, 1.0),
    },
    "N-tiny-jit": dict(_COMMON_ABC),
}

# M3 plan §2.1, lines 305-315: numeric-case encodings.
CASE_ENCODINGS: dict[str, dict[str, Any]] = {
    "F1": {"case_ids": [f"F1-{index:02d}" for index in range(13)], "numeric_cases": list(range(13))},
    "F2": {"case_ids": [f"F2-{index:02d}" for index in range(18)], "numeric_cases": list(range(18))},
    "F3": {"case_ids": [f"F3-{index:02d}" for index in range(7)], "numeric_cases": list(range(7))},
    "A-specialization": {
        "case_ids": ["timing-4096^3", "timing-8192^3", "timing-tail", "correctness-256^3"],
        "numeric_cases": [0, 1, 2, 3],
    },
    "A-solver": {
        "timed_case_ids": ["D4/S", "D2/S", "R-id/S", "R-rev/S"],
        "timed_numeric_cases": [0, 1, 2, 3],
        "correctness_formula": "table row j: 4096^3=16+2j, 256^3=17+2j",
    },
    "A-authoring": {
        "case_ids": ["timing-and-4096^3-correctness", "256^3-correctness"],
        "numeric_cases": [0, 1],
    },
    "A-F1-continuity": {"case_ids": ["shape-0", "shape-1", "shape-2"], "numeric_cases": [0, 1, 2]},
    "N-register-coupling": {
        "case_ids": ["M", "L-same", "L-cross"],
        "numeric_cases": [0, 0, 0],
    },
    "N-tiny-jit": {"case_ids": ["64^3", "128^3", "256^3"], "numeric_cases": [0, 1, 2]},
}

# M3 plan §2.1, lines 317-331: exact consumer equivalence classes.
SHARING_EQUIVALENCE_CLASSES: dict[str, str] = {
    "F1.roles-1-3": "all V0-V4 linked images and their five exact one-TU controls",
    "F1.role-4": "V3 linked and V3 exact control only; exactly N column-bias elements",
    "F1.role-5": "V4 linked and V4 exact control only; exactly M row-bias elements",
    "F2.roles-1-3": "linked CUTLASS, same-source control, and public FA4 in every comparison series",
    "F3.roles-1-5": "linked, both fused-control labels, and both unfused-composite kernels",
    "A-specialization.roles-1-3": "on/off comparison members at the same numeric case",
    "A-solver.roles-1-3": "each numerator/S comparison pair at the same numeric case",
    "A-authoring.roles-1-3": "wrapped/native; case 0 also shares timing and 4096^3 correctness",
    "A-F1-continuity.roles-1-3": "linked/control pair at the same numeric case",
    "N-register-coupling.roles-1-3": "M/L-same/L-cross all use numeric case 0",
    "N-tiny-jit.roles-1-3": "linked/control pair at the same numeric case",
}


def _u32(value: int) -> int:
    return value & UINT32_MASK


def philox4x32_10(counter: tuple[int, int, int, int], key: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return Random123-compatible Philox4x32-10 words."""
    values = tuple(_u32(value) for value in counter)
    key0, key1 = (_u32(value) for value in key)
    for _ in range(PHILOX_ROUNDS):
        product0 = PHILOX_M0 * values[0]
        product1 = PHILOX_M1 * values[2]
        hi0, lo0 = _u32(product0 >> 32), _u32(product0)
        hi1, lo1 = _u32(product1 >> 32), _u32(product1)
        values = (
            _u32(hi1 ^ values[1] ^ key0),
            lo1,
            _u32(hi0 ^ values[3] ^ key1),
            lo0,
        )
        key0 = _u32(key0 + PHILOX_W0)
        key1 = _u32(key1 + PHILOX_W1)
    return values


def resolve_namespace(namespace: str | int) -> tuple[str, int]:
    if isinstance(namespace, str):
        if namespace not in NAMESPACE_IDS:
            raise ValueError(f"unknown M3 stream namespace: {namespace}")
        return namespace, NAMESPACE_IDS[namespace]
    matches = [name for name, value in NAMESPACE_IDS.items() if value == namespace]
    if len(matches) != 1:
        raise ValueError(f"unregistered M3 stream namespace: 0x{namespace:08x}")
    return matches[0], namespace


def stream_address(namespace: str | int, numeric_case: int, role_id: int, element_index: int) -> dict[str, Any]:
    """Resolve the exact §2.1 key/counter/lane address for one element."""
    name, ns = resolve_namespace(namespace)
    if not 0 <= numeric_case <= UINT32_MASK:
        raise ValueError("numeric case must fit uint32")
    if role_id not in ROLE_TABLES[name]:
        raise ValueError(f"role {role_id} is not registered for {name}")
    if not 0 <= element_index < (1 << 66):
        raise ValueError("element index must be non-negative and its block index fit uint64")
    block, lane = divmod(element_index, PHILOX_LANES)
    return {
        "namespace": name,
        "namespace_id": ns,
        "numeric_case": numeric_case,
        "role_id": role_id,
        "element_index": element_index,
        "key": (PHILOX_SEED, ns),
        "counter": (block & UINT32_MASK, block >> 32, numeric_case, role_id),
        "lane": lane,
        "class_id": f"ns=0x{ns:04x}/case={numeric_case}/role={role_id}",
    }


def word_at(namespace: str | int, numeric_case: int, role_id: int, element_index: int) -> int:
    address = stream_address(namespace, numeric_case, role_id, element_index)
    words = philox4x32_10(address["counter"], address["key"])
    return words[address["lane"]]


def uniform01(word: int) -> float:
    """§2.1 binary64 ``(word + 0.5) * 2^-32``."""
    if not 0 <= word <= UINT32_MASK:
        raise ValueError("Philox word must fit uint32")
    return (word + UNIFORM_WORD_OFFSET) * UNIFORM_SCALE


def uniform_at(namespace: str | int, numeric_case: int, role_id: int, element_index: int) -> float:
    role = ROLE_TABLES[resolve_namespace(namespace)[0]][role_id]
    if role.lower is None or role.upper is None:
        raise ValueError(f"role {role_id} is not a uniform role")
    u = uniform01(word_at(namespace, numeric_case, role_id, element_index))
    return role.lower + (role.upper - role.lower) * u


def signed_int4_at(namespace: str | int, numeric_case: int, role_id: int, element_index: int) -> int:
    role = ROLE_TABLES[resolve_namespace(namespace)[0]][role_id]
    if role.dtype != "INT4":
        raise ValueError(f"role {role_id} is not signed INT4")
    nibble = word_at(namespace, numeric_case, role_id, element_index) & (INT4_CARDINALITY - 1)
    return nibble if nibble < INT4_SIGN_CUTOFF else nibble - INT4_CARDINALITY


def _bfloat16_bits_rne(value: float) -> int:
    """Round a binary64 value directly once to BF16, ties to even."""
    if math.isnan(value):
        return 0x7FC0
    sign = 0x8000 if math.copysign(1.0, value) < 0 else 0
    absolute = abs(value)
    if math.isinf(absolute):
        return sign | 0x7F80
    if absolute == 0.0:
        return sign
    significand, exponent2 = math.frexp(absolute)
    unbiased = exponent2 - 1
    if unbiased < -126:
        fraction = round(math.ldexp(absolute, 133))
        if fraction <= 0:
            return sign
        if fraction >= 128:
            return sign | 0x0080
        return sign | int(fraction)
    exponent = unbiased + 127
    fraction = round((significand * 2.0 - 1.0) * 128.0)
    if fraction == 128:
        fraction = 0
        exponent += 1
    if exponent >= 255:
        return sign | 0x7F80
    return sign | (exponent << 7) | int(fraction)


def storage_bytes(namespace: str | int, numeric_case: int, role_id: int, count: int) -> bytes:
    """Generate canonical little-endian packed bytes for one sharing class."""
    if count < 0:
        raise ValueError("count must be non-negative")
    name, _ = resolve_namespace(namespace)
    role = ROLE_TABLES[name][role_id]
    if role.dtype == "INT4":
        result = bytearray((count + 1) // 2)
        for index in range(count):
            nibble = signed_int4_at(name, numeric_case, role_id, index) & 0xF
            result[index // 2] |= nibble << (4 * (index & 1))
        return bytes(result)
    output = bytearray()
    for index in range(count):
        value = uniform_at(name, numeric_case, role_id, index)
        if role.dtype == "BF16":
            output.extend(struct.pack("<H", _bfloat16_bits_rne(value)))
        elif role.dtype == "FP16":
            output.extend(struct.pack("<e", value))
        elif role.dtype == "FP32":
            output.extend(struct.pack("<f", value))
        else:
            raise AssertionError(f"unhandled storage dtype: {role.dtype}")
    return bytes(output)


def values_match(actual: float, expected: float, *, rtol: float, atol: float) -> bool:
    """M3 §2.1 non-finite policy plus ordinary rtol/atol comparison."""
    if math.isnan(actual) or math.isnan(expected):
        return math.isnan(actual) and math.isnan(expected)
    if math.isinf(actual) or math.isinf(expected):
        return math.isinf(actual) and math.isinf(expected) and (actual > 0) == (expected > 0)
    return abs(actual - expected) <= atol + rtol * abs(expected)


def _valid_numeric_case(namespace: str, numeric_case: int) -> bool:
    if namespace == "A-solver":
        return numeric_case in range(4) or numeric_case >= 16
    return numeric_case in CASE_ENCODINGS[namespace]["numeric_cases"]


def stream_catalog() -> dict[str, Any]:
    roles = {
        namespace: {
            str(role_id): {
                "name": role.name,
                "dtype": role.dtype,
                "distribution": role.distribution,
            }
            for role_id, role in sorted(table.items())
        }
        for namespace, table in ROLE_TABLES.items()
    }
    return {
        "namespaces": dict(NAMESPACE_IDS),
        "case_encodings": CASE_ENCODINGS,
        "role_tables": roles,
        "sharing_equivalence_classes": SHARING_EQUIVALENCE_CLASSES,
        "fixed_values": {
            "seed": f"0x{PHILOX_SEED:08x}",
            "alpha": ALPHA,
            "beta": BETA,
            "authoring_correctness_alpha": AUTHORING_CORRECTNESS_ALPHA,
            "f1_aux_sentinel": f"0x{F1_AUX_SENTINEL:02x}",
            "register_output_initial": REGISTER_OUTPUT_INITIAL,
        },
    }


def build_stream_map(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a resolved family config and return canonical stream-map content."""
    namespace, namespace_id = resolve_namespace(config.get("namespace"))
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("stream-map config requires a non-empty cases list")
    resolved_cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each stream-map case must be an object")
        case_id = case.get("case_id")
        numeric_case = case.get("numeric_case")
        if not isinstance(case_id, str) or not case_id or case_id in seen_case_ids:
            raise ValueError("case_id must be a unique non-empty string")
        if not isinstance(numeric_case, int) or not _valid_numeric_case(namespace, numeric_case):
            raise ValueError(f"numeric case {numeric_case!r} is not registered for {namespace}")
        seen_case_ids.add(case_id)
        dimensions = case.get("dimensions", {})
        tensors = case.get("tensors")
        if not isinstance(dimensions, dict) or not isinstance(tensors, list) or not tensors:
            raise ValueError(f"{case_id}: dimensions must be an object and tensors must be non-empty")
        resolved_tensors: list[dict[str, Any]] = []
        seen_roles: set[int] = set()
        for tensor in tensors:
            if not isinstance(tensor, dict):
                raise ValueError(f"{case_id}: tensor must be an object")
            role_id = tensor.get("role_id")
            if not isinstance(role_id, int) or role_id not in ROLE_TABLES[namespace] or role_id in seen_roles:
                raise ValueError(f"{case_id}: role_id must be registered and unique")
            seen_roles.add(role_id)
            spec = ROLE_TABLES[namespace][role_id]
            shape = tensor.get("shape")
            strides = tensor.get("strides")
            consumers = tensor.get("consumers")
            if not isinstance(shape, list) or not shape or any(not isinstance(x, int) or x <= 0 for x in shape):
                raise ValueError(f"{case_id} role {role_id}: shape must contain positive integers")
            if not isinstance(strides, list) or len(strides) != len(shape) or any(not isinstance(x, int) or x < 0 for x in strides):
                raise ValueError(f"{case_id} role {role_id}: strides must match shape rank")
            if not isinstance(consumers, list) or not consumers or any(not isinstance(x, str) or not x for x in consumers):
                raise ValueError(f"{case_id} role {role_id}: consumers must be non-empty strings")
            if len(set(consumers)) != len(consumers):
                raise ValueError(f"{case_id} role {role_id}: consumers must be unique")
            if tensor.get("dtype", spec.dtype) != spec.dtype or tensor.get("distribution", spec.distribution) != spec.distribution:
                raise ValueError(f"{case_id} role {role_id}: dtype/distribution drift from §2.1")
            element_count = math.prod(shape)
            if namespace == "F1" and role_id == 4 and dimensions.get("N") != element_count:
                raise ValueError(f"{case_id}: F1 role 4 must contain exactly N elements")
            if namespace == "F1" and role_id == 5 and dimensions.get("M") != element_count:
                raise ValueError(f"{case_id}: F1 role 5 must contain exactly M elements")
            address = stream_address(namespace, numeric_case, role_id, 0)
            resolved_tensors.append({
                "role_id": role_id,
                "role_name": spec.name,
                "class_id": address["class_id"],
                "shape": shape,
                "strides": strides,
                "element_count": element_count,
                "dtype": spec.dtype,
                "distribution": spec.distribution,
                "consumers": sorted(consumers),
            })
        resolved_cases.append({
            "case_id": case_id,
            "numeric_case": numeric_case,
            "dimensions": dimensions,
            "tensors": sorted(resolved_tensors, key=lambda item: item["role_id"]),
        })
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSIONS["stream_map"],
        "derivation": {
            "algorithm": "Philox4x32-10",
            "seed": f"0x{PHILOX_SEED:08x}",
            "key": "(0x50495045, ns)",
            "block": "floor(j/4)",
            "counter": "(block & 0xffffffff, block >> 32, numeric_case, role_id)",
            "lane": "j mod 4",
            "uniform_binary64": "(word + 0.5) * 2^-32",
            "rounding": "one round-to-nearest-even conversion to declared storage dtype",
            "signed_int4": "n=word&15; q=n when n<8 else n-16",
        },
        "namespace": namespace,
        "namespace_id": namespace_id,
        "case_encoding": CASE_ENCODINGS[namespace],
        "sharing_policy": SHARING_EQUIVALENCE_CLASSES,
        "cases": sorted(resolved_cases, key=lambda item: item["case_id"]),
    }
    payload["sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def write_stream_map(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    result = build_stream_map(config)
    write_json_atomic(path, result)
    return result


def class_bytes_identical(
    namespace: str | int, numeric_case: int, role_id: int, consumers: Iterable[str], count: int
) -> bool:
    """Audit helper: every named consumer of one class gets the exact same bytes."""
    names = list(consumers)
    if not names:
        raise ValueError("equivalence class must name at least one consumer")
    payloads = [storage_bytes(namespace, numeric_case, role_id, count) for _ in names]
    return all(payload == payloads[0] for payload in payloads[1:])
