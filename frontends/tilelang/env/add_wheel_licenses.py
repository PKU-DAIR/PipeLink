#!/usr/bin/env python3
"""Add the bundled-code notices for PipeLink's TileLang 0.1.14 CUDA wheel."""

import argparse
import base64
import copy
import csv
import hashlib
import io
import os
from pathlib import Path
import tempfile
import zipfile


DIST_INFO = "tilelang-0.1.14.dist-info"
TVM = "3rdparty/tvm"
FFI = f"{TVM}/3rdparty/tvm-ffi"
# Audited against v0.1.14's CMake targets and wheel.packages mapping. These
# components contribute bundled binaries or headers; this is not a recursive
# collection of every submodule's license (most optional TVM backends are off).
LICENSE_FILES = {
    "tvm/LICENSE": f"{TVM}/LICENSE",
    "tvm/NOTICE": f"{TVM}/NOTICE",
    "cutlass/LICENSE.txt": "3rdparty/cutlass/LICENSE.txt",
    "tvm-ffi/LICENSE": f"{FFI}/LICENSE",
    "tvm-ffi/NOTICE": f"{FFI}/NOTICE",
    "dlpack/LICENSE": f"{FFI}/3rdparty/dlpack/LICENSE",
    "compiler-rt/LICENSE.builtin_fp16.txt": f"{TVM}/licenses/LICENSE.builtin_fp16.txt",
    "nvbench/LICENSE.l2_cache_flush.txt": f"{TVM}/licenses/LICENSE.l2_cache_flush.txt",
    "vllm/LICENSE.vllm.txt": f"{TVM}/licenses/LICENSE.vllm.txt",
}
# These two headers are compiled into TVM but are not included in the wheel.
# Preserve their actual copyright/permission comments as well as the full
# upstream license texts, without adding executable source to licenses/.
HEADER_NOTICES = {
    "compiler-rt/NOTICE": f"{TVM}/3rdparty/compiler-rt/builtin_fp16.h",
    "nvbench/NOTICE": f"{TVM}/3rdparty/nvbench/l2_cache_flush.h",
}


def add_wheel_licenses(wheel: Path, source_tree: Path) -> None:
    additions = {
        f"{DIST_INFO}/licenses/{name}": (source_tree / source).read_bytes()
        for name, source in LICENSE_FILES.items()
    }
    for name, source in HEADER_NOTICES.items():
        header = (source_tree / source).read_bytes()
        end = header.index(b"*/") + 2
        additions[f"{DIST_INFO}/licenses/{name}"] = header[:end] + b"\n"

    record_name = f"{DIST_INFO}/RECORD"
    record = io.StringIO(newline="")
    rows = csv.writer(record, lineterminator="\n")
    mode = wheel.stat().st_mode
    with tempfile.NamedTemporaryFile(
        dir=wheel.parent, prefix=f".{wheel.name}.", suffix=".tmp", delete=False
    ) as temporary:
        replacement = Path(temporary.name)
    try:
        with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(replacement, "w") as updated:
            record_info = original.getinfo(record_name)
            updated.comment = original.comment
            for info in original.infolist():
                if info.filename == record_name or info.filename in additions:
                    continue
                digest = hashlib.sha256()
                size = 0
                with original.open(info) as source, updated.open(copy.copy(info), "w") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if not info.is_dir():
                    encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
                    rows.writerow((info.filename, f"sha256={encoded}", size))

            for name, contents in sorted(additions.items()):
                info = zipfile.ZipInfo(name, date_time=record_info.date_time)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                updated.writestr(info, contents)
                encoded = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode("ascii")
                rows.writerow((name, f"sha256={encoded}", len(contents)))
            rows.writerow((record_name, "", ""))
            updated.writestr(copy.copy(record_info), record.getvalue().encode("utf-8"))
        os.chmod(replacement, mode)
        os.replace(replacement, wheel)
    finally:
        replacement.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, metavar="WHEEL")
    parser.add_argument("source_tree", type=Path, metavar="SOURCE_TREE")
    args = parser.parse_args()
    add_wheel_licenses(args.wheel, args.source_tree)


if __name__ == "__main__":
    main()
