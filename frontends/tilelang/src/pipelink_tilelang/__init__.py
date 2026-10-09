"""TileLang frontend for PipeLink stage declarations and device code generation.

``pipelink_tilelang.context`` is the recipe-visible build context: the pipec2
profile the roles are lowered for and the composition demo whose entry pins
contract partners. ``python -m pipelink_tilelang`` publishes it before the
recipe module executes; the default is a standalone sm103 profile.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pipec2.profile import get_profile

from .annotate import AnnotationError

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROFILE = "sm_103a"


@dataclass(frozen=True)
class Context:
    profile: str
    demo: str | None
    composition: dict = field(repr=False)


def load_composition(path: Path) -> dict:
    return json.loads(path.read_text())


def demo_profile(composition: dict, demo: str) -> str:
    """A demo targets its own composition target, else the composition's global stamp."""
    entry = composition["demos"][demo]
    return entry.get("target", composition["pipelink_abi"]["target"])["profile"]


def set_context(profile: str | None, demo: str | None, composition: dict | None = None) -> Context:
    """Publish a standalone profile or a demo from an explicitly supplied registry."""
    if demo is not None and composition is None:
        raise AnnotationError("demo", "a composition registry is required with a demo")
    composition = composition if composition is not None else {}
    if demo is not None and demo not in composition.get("demos", {}):
        raise AnnotationError("demo", f"unknown composition demo {demo!r}")
    if profile is None:
        profile = demo_profile(composition, demo) if demo is not None else DEFAULT_PROFILE
    get_profile(profile)
    published = Context(profile, demo, composition)
    globals()["context"] = published
    return published


def __getattr__(name):
    if name == "context":
        return set_context(DEFAULT_PROFILE, None)
    if name in ("TensorExportStage", "export_fragment"):
        from . import tensor_export
        return getattr(tensor_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
