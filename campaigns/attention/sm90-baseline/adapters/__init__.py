"""Lazy public-kernel registry; importing this module never imports a backend.

Adapters own all compilation, plans and input-layout conversion in prepare.
run is one complete product iteration, including resets/merges, without a host
synchronization. outputs reads the most recent result and performs any untimed
layout or log-base conversion; it must not launch attention again.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Literal, Protocol

Role = Literal["modular-lower", "modular-flash", "expert", "production", "ceiling", "equivalence"]


class Prepared(Protocol):
    prep_ms: float
    config: dict[str, Any]

    def run(self) -> tuple[Any, Any | None]:
        """Enqueue full attention product; return native output and optional LSE."""
        ...

    def outputs(self) -> tuple[Any, Any | None]:
        """Untimed O bf16 [B,S,Hq,D], natural-log LSE f32 [B,Hq,S]."""
        ...


@dataclass(frozen=True)
class Impl:
    name: str
    role: Role
    supports: Callable[[dict], str | None]
    prepare: Callable[[dict, Any, Any, Any, Any | None], Prepared]


def load_registry() -> tuple[list[Impl], dict[str, str]]:
    """Load available modules and report every import failure, without hiding it."""
    impls, errors = [], {}
    for name in ("fa3", "fa4", "flashinfer", "flex", "production"):
        try:
            module = import_module(f"{__name__}.{name}")
            candidates = module.IMPLS
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        impls.extend(candidates)
    names = [impl.name for impl in impls]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate implementation names in adapter registry")
    return impls, errors
