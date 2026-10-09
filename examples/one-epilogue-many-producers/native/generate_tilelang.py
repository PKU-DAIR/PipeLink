#!/usr/bin/env python3
"""Build a native TileLang stage from its generated source and packed placement.

Run with the pipec environment. The generated source needs the same TileLang and
CUTLASS include directories as its provider. Params are consumer EpilogueParams:
a, b, bias, output (four pointers), M, N, K (int32), eps (float); 48 bytes, align 8.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipec2.container import read as read_package
from pipec2.tensor.expr import Domain, evaluate, parse


def generate(provider_source: Path, provider_package: Path, out: Path) -> Path:
    manifest = read_package(provider_package)[0]
    profile = manifest["pipelink_abi"]["target"]["profile"]
    export, = manifest["tensor"]["exports"]
    placement, = export["placements"]
    if placement["space"] != "reg" or placement["guard"] is not None:
        raise ValueError("Native TileLang requires an unconditional register placement")
    domains = {item["name"]: Domain.from_json_obj(item) for item in export["identity"]["coords"]}
    expressions = {key: parse(value, domains)
                   for key, value in placement["components"].items()}
    coords = {}
    for m in range(128):
        for n in range(128):
            env = {"m": m, "n": n}
            address = tuple(evaluate(expressions[key], env) for key in ("participant", "position"))
            if address in coords:
                raise ValueError("Provider placement aliases logical elements")
            coords[address] = (m, n)
    if set(coords) != {(t, p) for t in range(128) for p in range(128)}:
        raise ValueError("Provider must fill a 128-word frame in each of 128 threads")

    # Factor the captured bijection rather than hardcoding either architecture's
    # fragment map. Compile-time P makes position offsets immediate; thread bits
    # avoid runtime constant-memory table lookups in the epilogue.
    functions = []
    for axis, name in enumerate(("row", "col")):
        offsets = [coords[0, p][axis] for p in range(128)]
        bits = [coords[1 << bit, 0][axis] - coords[0, 0][axis] for bit in range(7)]
        if any(coords[t, p][axis] != offsets[p] + sum(((t >> bit) & 1) * v
                                                    for bit, v in enumerate(bits))
               for t in range(128) for p in range(128)):
            raise ValueError("Provider coordinate map is not additive in participant bits")
        terms = "".join(f" + ((t >> {bit}) & 1) * {v}" for bit, v in enumerate(bits) if v)
        functions.append(f"template <int P> __device__ __forceinline__ int pl_tl_{name}(int t) {{\n"
                         f"  constexpr int offsets[128] = {{{', '.join(map(str, offsets))}}};\n"
                         f"  return offsets[P]{terms};\n}}")
    for t in range(128):
        for p in range(0, 128, 2):
            m, n = coords[t, p]
            if n % 2 or coords[t, p + 1] != (m, n + 1):
                raise ValueError("Provider fragment does not support aligned adjacent float2 stores")
    pairs = " \\\n".join(f"  pl_tl_store_pair<{p}>(params, frame, scale);" for p in range(0, 128, 2))
    layout = "#pragma once\n" + "\n\n".join(functions)
    layout += "\n#define PL_TILELANG_STORE_PAIRS(params, frame, scale) \\\n" + pairs + "\n"

    claims = []
    for claim in manifest["claims"]:
        if claim["extent"]["per"] or claim["domain"] not in ("smem", "named_barrier"):
            raise ValueError("Unexpected provider claim: native generator needs an explicit implementation")
        claims.append(f'PL_CLAIM("{claim["handle"]}", "{claim["domain"]}", '
                      f'{claim["extent"]["const"]}, {claim["align"]}, "kernel");')
    source = (f'#define PL_TILELANG_PROVIDER_SOURCE "{provider_source.resolve()}"\n'
              f'#define PL_TILELANG_LAYOUT "{(out / "tilelang_layout.cuh").resolve()}"\n'
              f'#define plx_produce_source native_tilelang_gemm_{profile.replace("_", "")}_produce_source\n'
              f'#define plx_release native_tilelang_gemm_{profile.replace("_", "")}_release\n'
              f'#include "{Path(__file__).with_name("tilelang.cuh").resolve()}"\n'
              f'PL_STAGE("native.tilelang.gemm.{profile.replace("_", "")}", "1.0.0");\n'
              'PL_PARAMS(EpilogueParams, a, b, bias, output, M, N, K, eps);\n'
              'PL_ROLE(native, plx_role_native_tilelang, 4);\n'
              + "\n".join(claims) + '\n'
              'PL_CLAIM("pipelink_native_tilelang_scale", "smem", 512, 128, "kernel");\n')
    out.mkdir(parents=True, exist_ok=True)
    (out / "tilelang_layout.cuh").write_text(layout)
    destination = out / "native_tilelang.cu"
    destination.write_text(source)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-source", type=Path, required=True)
    parser.add_argument("--provider-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.provider_source, args.provider_package, args.out))


if __name__ == "__main__":
    main()
