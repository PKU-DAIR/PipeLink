#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "pipec" / "src"))

from build import PROFILE, SM90_FLAGS
from pipec2.link import _NvJitLink, _link_options
from pipec2.pack.emit import math_mode
from pipec2.profile import get_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("object", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maxrregcount", type=int, required=True)
    args = parser.parse_args()
    cuda_text = os.environ.get("PIPELINK_CUDA_HOME")
    if not cuda_text:
        parser.error("PIPELINK_CUDA_HOME is not set; source tools/env.sh")
    library = Path(cuda_text) / "lib64" / "libnvJitLink.so.13"
    # The same options the linker uses for the packed stages, including their math mode.
    cubin, info, error = _NvJitLink(library).link(
        [(args.object.name, args.object.read_bytes())],
        _link_options(get_profile(PROFILE), args.maxrregcount, math_mode(SM90_FLAGS)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(cubin)
    if info:
        print(info, end="")
    if error:
        print(error, end="", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
