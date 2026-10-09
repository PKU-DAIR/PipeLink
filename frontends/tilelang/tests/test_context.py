"""Explicit registry selection and standalone recipe contexts."""
import json
import sys

import pytest

import pipelink_tilelang
from pipelink_tilelang import DEFAULT_PROFILE, set_context
from pipelink_tilelang.__main__ import main
from pipelink_tilelang.annotate import AnnotationError


@pytest.fixture(autouse=True)
def restore_context(monkeypatch):
    monkeypatch.setattr(pipelink_tilelang, "context", pipelink_tilelang.context)


def test_lazy_context_is_standalone(monkeypatch):
    monkeypatch.delattr(pipelink_tilelang, "context")
    context = pipelink_tilelang.context
    assert (context.profile, context.demo, context.composition) == (DEFAULT_PROFILE, None, {})
    assert set_context("sm_90a", None).composition == {}


def test_demo_requires_explicit_registry():
    with pytest.raises(AnnotationError, match="composition registry is required"):
        set_context(None, "consumer")
    registry = {"pipelink_abi": {"target": {"profile": "sm_103a"}},
                "demos": {"consumer": {"target": {"profile": "sm_90a"}}}}
    assert set_context(None, "consumer", registry).profile == "sm_90a"
    with pytest.raises(AnnotationError, match="unknown composition demo"):
        set_context(None, "missing", registry)


def test_cli_requires_composition(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["pipelink_tilelang", "--demo", "consumer",
                                      "--kernel", "unused.py", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--demo requires --composition" in capsys.readouterr().err


def test_cli_rejects_disagreeing_profile(monkeypatch, tmp_path):
    from tilelang.env import CUDA_HOME
    monkeypatch.setenv("PIPELINK_CUDA_HOME", CUDA_HOME)
    registry = tmp_path / "composition.json"
    registry.write_text(json.dumps({"pipelink_abi": {"target": {"profile": "sm_103a"}},
                                    "demos": {"consumer": {}}}))
    monkeypatch.setattr(sys, "argv", ["pipelink_tilelang", "--demo", "consumer",
                                      "--composition", str(registry), "--profile", "sm_90a",
                                      "--kernel", "unused.py", "--out", str(tmp_path)])
    with pytest.raises(AnnotationError, match="sm_90a disagrees with demo consumer target sm_103a"):
        main()
