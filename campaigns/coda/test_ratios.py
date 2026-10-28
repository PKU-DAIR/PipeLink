"""GPU-free coverage of reference/candidate orientation in CODA summaries."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parents[1]))
    monkeypatch.syspath_prepend(str(HERE.parents[1] / "pipec/src"))
    monkeypatch.setitem(sys.modules, "params", load_module("coda_params", HERE / "params.py"))
    return load_module("coda_run", HERE / "run.py")


@pytest.fixture
def pair():
    # An even count distinguishes median(B/A)=2.5 from 1/median(A/B)=1.6.
    return {
        "repetitions": 1,
        "blocks": [
            {"order": "ABBA", "milliseconds": [2, 2, 2, 2], "ratio_onetu_over_linked": 1},
            {"order": "BAAB", "milliseconds": [8, 2, 2, 8], "ratio_onetu_over_linked": 4},
        ],
        "ratio_onetu_over_linked": statistics.median([1, 4]),
        "telemetry": {"A": {}, "B": {}},
        "status": "PASS",
    }


def test_native_placement_uses_paired_speedup(harness, monkeypatch, pair):
    candidate = SimpleNamespace(backend=SimpleNamespace(stream_token=0))
    baseline = SimpleNamespace(launch=lambda: None, read=lambda _: None)
    monkeypatch.setattr(harness, "native_session", lambda *a, **kw: nullcontext(baseline))
    monkeypatch.setattr(harness, "check", lambda *a: {"status": "PASS"})

    def measure(a, b, **kwargs):
        assert a is candidate and b is baseline
        return pair

    monkeypatch.setattr(harness, "time_pair", measure)
    result = harness.time_placement(candidate, {}, "fused", {}, [1, 2, 3],
                                    {"fused": None}, None, 0, stagger_ns=0)
    assert result["speedup_over_baseline"] == 2.5
    assert result["milliseconds"] == 5


def test_locked_public_summaries_use_reference_over_candidate(harness):
    timing = {"shape": [1, 2, 3], "linked_ms": 2, "producer_gemm": {"milliseconds": 4}}
    results = {
        "mode": "locked", "cells": {"hand": {"timings": [timing]}},
        "public": {"cases": [{"shape": [1, 2, 3], "milliseconds": {"compile": 8, "liger": 10},
                               "gemm_only_ms": {"interleaved": 6},
                               "pair": {"telemetry": {"A": {}, "B": {}}}}]},
    }
    harness.attach_public_placement(results)
    assert timing["public_placement"]["compile"]["speedup_over_baseline"] == 4
    assert timing["public_placement"]["liger"]["speedup_over_baseline"] == 5
    assert timing["producer_gemm"]["speedup_over_cublas"] == 1.5


def test_public_pairs_use_candidate_arm_a(harness, monkeypatch, pair):
    public = load_module("coda_public_run", HERE / "public/run.py")
    candidate = SimpleNamespace(backend=SimpleNamespace(stream_token=0), poison=lambda: None,
                                launch=lambda: None, read=lambda: None)
    producer = SimpleNamespace(launch=lambda: None, read=lambda _: None)
    monkeypatch.setitem(sys.modules, "run", harness)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        stream=lambda _: nullcontext(), ExternalStream=lambda *a, **kw: None,
        current_stream=lambda: SimpleNamespace(cuda_stream=0))))
    monkeypatch.setattr(harness, "linked_images", lambda *a, **kw: nullcontext(({"linked": candidate}, None)))
    monkeypatch.setattr(harness, "native_session", lambda *a, **kw: nullcontext(producer))
    monkeypatch.setattr(harness, "check", lambda *a: {"status": "PASS"})
    monkeypatch.setattr(public, "check_unfused", lambda *a: {"status": "PASS"})
    candidates = []

    def measure(a, b, **kwargs):
        candidates.append(a)
        assert isinstance(b, public.Sequence)
        return pair

    monkeypatch.setattr(public, "time_pair", measure)
    result = public.paired_placement({"cells": {"hand": {}}}, {}, [1, 2, 3],
                                    {"fused": None, "unfused": None}, None,
                                    {name: lambda: None for name in ("compile", "liger", "gemm_interleaved")},
                                    0, stagger_ns=0)
    assert candidates == [candidate, candidate, producer]
    for record in result["cells"]["hand"]["public"].values():
        assert record["speedup_over_baseline"] == 2.5
    assert result["producer_gemm"]["speedup_over_cublas"] == 2.5
    assert result["producer_gemm"]["milliseconds"] == 2
    assert result["producer_gemm"]["cublas_ms"] == 5


def test_direct_public_attachment_preserves_paired_speedup(harness, pair):
    comparison = {"pair": pair, "speedup_over_baseline": 2.5}
    producer = {"pair": pair, "speedup_over_cublas": 2.5}
    timing = {"shape": [1, 2, 3], "producer_gemm": {"milliseconds": 4, "speedup_over_cublas": 1.5}}
    results = {
        "mode": "unlocked", "cells": {"hand": {"timings": [timing]}},
        "public": {"cases": [{"shape": [1, 2, 3], "paired_placement": {
            "cells": {"hand": {"public": {"compile": comparison}}}, "producer_gemm": producer}}]},
    }
    harness.attach_public_placement(results)
    assert timing["public_placement"]["compile"]["speedup_over_baseline"] == 2.5
    assert timing["producer_gemm"]["speedup_over_cublas"] == 2.5
    assert timing["producer_gemm"]["timing_arm"] == "A"
    assert timing["producer_gemm_linked_diagnostic"] == {"milliseconds": 4}


def test_report_displays_speedups_without_inverting(pair):
    report = load_module("coda_report", HERE / "report.py")
    timing = {"shape": [1, 2, 3], "linked_ms": 2, "one_tu_ms": 5,
              "pair": pair, "status": "PASS",
              "placements": {"fused": {"milliseconds": 5, "pair": pair, "speedup_over_baseline": 2.5}},
              "producer_gemm": {"milliseconds": 2, "pair": pair, "speedup_over_cublas": 3.5}}
    results = {"profile": "sm103", "status": "PASS", "cells": {"hand": {
        "timings": [timing], "correctness": [], "sanitizers": [],
        "sass": {"checks": {"instruction_counts_equal": True}}, "source_withheld": {"status": "PASS"}}}}
    rows = [[cell.strip() for cell in line.strip("|").split("|")]
            for line in report.render(results).splitlines() if line.startswith("| hand |")]
    assert float(rows[1][5]) == 2.5
    assert float(rows[2][4]) == 3.5
