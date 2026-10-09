"""Build a native provider and its ABI report from the same compiler inventory."""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from .abi import ABI_TYPES, NEUTRAL_TYPE_ID_MIN, NEUTRAL_TYPE_ID_MAX, header_path, probe_layouts
from .profile import dtypes_digest as _dtypes_digest


def build_provider(sources: Sequence[str | Path], output: str | Path, *, owner: str,
                   owner_contract_digest: str, profile: str, dtypes_digest: str,
                   type_ids: Mapping[str, int],
                   extra_headers: Sequence[str | Path] = (),
                   extra_types: Mapping[str, Sequence[str]] | None = None,
                   compiler: str = "g++", compile_flags: Sequence[str] = ()) -> list[dict]:
    """Compile query/build/release sources plus a generated ``abi`` export.

    The returned exact ABI-type inventory goes into the owner manifest. The
    generated report TU and layout probe use identical headers, compiler and
    flags. No source-name hash or serialized object padding is treated as layout.
    Caller sources must not define the generated ABI export themselves.
    type_ids covers only owner-local types; pipec supplies the neutral inventory.
    profile is the owner's pipec2 target profile name; dtypes_digest binds
    the owner's declared dtype table on that profile.
    """
    if not re.fullmatch(r"[A-Za-z_]\w*", owner):
        raise ValueError("provider owner export suffix must be a C identifier")
    if not re.fullmatch(r"[0-9a-f]{64}", owner_contract_digest):
        raise ValueError("owner contract must be a SHA-256 digest")
    if dtypes_digest != _dtypes_digest(profile):
        raise ValueError("provider requires the current digest for its declared dtype table")
    layouts = probe_layouts(extra_headers, extra_types, compiler, compile_flags=compile_flags)
    if set(type_ids) != set(layouts) - ABI_TYPES.keys() or len(set(type_ids.values())) != len(type_ids) or any(
            type(value) is not int or value <= 0 or value > 0xffffffff
            or NEUTRAL_TYPE_ID_MIN <= value <= NEUTRAL_TYPE_ID_MAX for value in type_ids.values()):
        raise ValueError("owner type IDs must exactly cover the local compiler-probed inventory outside the neutral reserved range")
    all_type_ids = {name: row["id"] for name, row in layouts.items() if name in ABI_TYPES}
    all_type_ids.update(type_ids)
    inventory = [{"id": all_type_ids[name], "name": name, "bytes": row["bytes"],
                  "align": row["align"], "layout_digest": row["layout_digest"],
                  "fields": row["fields"]} for name, row in layouts.items()]
    inventory.sort(key=lambda item: item["id"])
    digest = lambda text: "{{" + ",".join(str(b) for b in bytes.fromhex(text)) + "}}"
    lines = [f'#include "{header_path()}"', "#include <string.h>"]
    for header in extra_headers:
        path = str(Path(header).resolve())
        if any(ch in path for ch in '\n\r"'):
            raise ValueError("invalid provider header path")
        lines.append(f'#include "{path}"')
    lines.append("static const plk_layout_entry_v2 layouts[] = {")
    for row in inventory:
        lines.append(f'  {{{row["id"]},0,sizeof({row["name"]}),alignof({row["name"]}),{digest(row["layout_digest"])}}},')
    lines += ["};", f'extern "C" int pipelink_params_abi_{owner}(plk_abi_report_v2* out) {{',
              "  if (!out || out->header.abi_version != PLK_HOST_ABI_V2 ||",
              "      out->header.struct_bytes != sizeof(*out)) return PLK_E_VERSION;",
              f"  if (out->entry_capacity != {len(inventory)} || !out->entries) return PLK_E_CAPACITY;",
              f"  for (unsigned i=0; i<{len(inventory)}; ++i) {{",
              "    if (out->entries[i].abi_type_id != layouts[i].abi_type_id) return PLK_E_CONTRACT;",
              "  }", f"  memcpy(out->entries, layouts, sizeof(layouts));",
              f"  out->entry_count = {len(inventory)};",
              f"  const plk_profile_digest_v2 profile = {{{digest(dtypes_digest)}}};",
              f"  const plk_digest_v2 contract = {digest(owner_contract_digest)};",
              "  out->profile = profile; out->owner_contract = contract;",
              "  return PLK_OK;", "}", ""]
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plk-provider-build-") as temporary:
        report = Path(temporary) / "abi_report.cpp"
        report.write_text("\n".join(lines))
        built = Path(temporary) / "provider.so"
        pic_flag = "-Xcompiler=-fPIC" if Path(compiler).name == "nvcc" else "-fPIC"
        command = [compiler, "-std=c++17", "-shared", pic_flag,
                   *compile_flags, *(str(Path(source).resolve()) for source in sources),
                   str(report), "-o", str(built)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        output.write_bytes(built.read_bytes())
    return inventory
