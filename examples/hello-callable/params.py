"""Host mirror of caller.cu's PL_PARAMS, also usable by pipec2 launch."""
from __future__ import annotations

import ctypes
from typing import Mapping

COUNT = 1 << 24


class CallerParams(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("count", ctypes.c_uint32),
    ]


def build(addresses: Mapping[str, int], *, count: int = COUNT) -> dict[str, bytes]:
    """Called after device allocation; return one blob per params-bearing stage.

    The CLI uses COUNT elements. Python callers may bind a different count with
    functools.partial(build, count=...). The provider declares PL_NO_PARAMS.
    """
    return {"org.hello.caller": bytes(CallerParams(addresses["input"], addresses["output"], count))}
