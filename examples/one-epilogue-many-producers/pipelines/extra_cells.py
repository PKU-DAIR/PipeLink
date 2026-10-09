#!/usr/bin/env python3
"""Write the TileLang cell recipes for build.py --extra-cells (provider + contract-authored consumer)."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from params import PROFILE_SOURCES


def main(profile: str) -> None:
    out = HERE / "out" / profile
    tilelang = Path(os.environ["TILELANG_INCLUDE_DIR"]).parent  # <site-packages>/tilelang
    tl_includes = [str(REPO / "third_party/cutlass/include"), str(tilelang / "src"), str(REPO / "pipec/src/pipec2/pack/include")]
    hand_includes = [str(REPO / "third_party/cutlass/include"), str(REPO / "pipec/src/pipec2/pack/include")]
    provider = {"package": str(out / "packed/tl-reg.plink"), "source": str(out / "tilelang/provider/roles.cu"),
                "includes": tl_includes, "defines": {}}
    consumer = {"package": str(out / "packed/tl-epilogue.plink"),
                "source": str(out / "tilelang/consumer/generation/tilelang-epilogue/roles.cu"),
                "includes": tl_includes, "defines": {}}
    hand = {"package": str(out / "packed/epilogue.plink"), "source": str(HERE / "consumer/epilogue.cu"),
            "includes": hand_includes, "defines": {}}
    sources = PROFILE_SOURCES[profile]
    cells = {"tl-reg": {"acc": provider, "consumer": hand}, "tl-reg-tlc": {"acc": provider, "consumer": consumer}}
    for placement in sources["placements"]:
        acc = {"package": str(out / f"packed/{placement}.plink"), "source": str(HERE / sources["provider"]),
               "includes": hand_includes, "defines": {**sources["defines"], f"PL_EXPORT_{placement.upper()}": "1"}}
        cells[f"{placement}-tlc"] = {"acc": acc, "consumer": consumer}
    target = out / "extra-cells.json"
    target.write_text(json.dumps(cells, indent=1) + "\n")
    print(target)


if __name__ == "__main__":
    main(sys.argv[1])
