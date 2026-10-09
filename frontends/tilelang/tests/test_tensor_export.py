"""CPU-only fragment ownership and caller-frame export contracts."""
from dataclasses import replace
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from tilelang import tvm
from pipec2.pack.assemble import assemble
from pipec2.schema import load_stage_manifest

from pipelink_tilelang.annotate import AnnotationError
from pipelink_tilelang.capture import capture
from pipelink_tilelang.externalize import _nodes
from pipelink_tilelang.tensor_export import _annotations, fragment_placement, lower_fragment


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "examples/one-epilogue-many-producers/providers/tilelang/gemm_provider.py"
PACK = ROOT / "pipec/src/pipec2/pack"


@pytest.fixture(scope="module")
def recipe():
    spec = importlib.util.spec_from_file_location("tensor_export_recipe", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=("sm_90a", "sm_103a", "sm_120a"))
def lowered(recipe, request):
    target = tvm.target.Target({"kind": "cuda", "arch": request.param})
    result = capture(recipe.program(), target=target, pass_configs=recipe.PASS_CONFIGS)
    generated = {"modules": result.modules, "facts": result.facts, "layout_objects": result.layout_objects}
    return target, result, generated


def test_fragment_owns_every_caller_frame_slot(recipe, lowered):
    target, result, _ = lowered
    name = recipe.TENSOR_EXPORT.fragment
    rows = fragment_placement(result.layout_objects[name], result.facts["layouts"][name])
    assert set(rows) == {(thread, position) for thread in range(128) for position in range(128)}
    # Hopper wgmma places adjacent pairs in one thread, while rows eight apart
    # occupy successive pairs of that thread's fragment.
    if str(target.attrs["arch"]) == "sm_90a":
        assert [rows[i] for i in (0, 1, 2, 8, 8 * 128, 64 * 128)] == [
            (0, 0), (0, 1), (1, 0), (0, 4), (0, 2), (0, 64)]


def test_frame_copy_preserves_physical_fragment_positions(recipe, lowered):
    target, _, generated = lowered
    _, _, measured, function = lower_fragment(generated, recipe.TENSOR_EXPORT, target)
    tir = tvm.tirx
    frame = function.params[-1]
    stores = [node for node in _nodes(function.body, tir.BufferStore) if node.buffer.data.same_as(frame)]
    assert len(stores) == 1
    store = stores[0]
    assert isinstance(store.value, tir.BufferLoad)
    assert store.value.buffer.name == recipe.TENSOR_EXPORT.fragment
    assert tvm.ir.structural_equal(store.indices[0], store.value.indices[0])
    loop = next(loop for loop in _nodes(function.body, tir.For) if loop.body.same_as(store))
    assert int(loop.min) == 0
    assert int(loop.extent) == measured["frame_words"] == 128
    assert measured["threads"] == 128
    assert measured["smem_bytes"] == 65536


def test_missing_fragment_rejected_at_export_boundary(recipe, lowered):
    target, _, generated = lowered
    with pytest.raises(AnnotationError) as caught:
        lower_fragment(generated, replace(recipe.TENSOR_EXPORT, fragment="absent_fragment"), target)
    assert caught.value.field == "tensor_export.fragment"


def test_replication_cannot_silently_choose_one_owner(recipe, lowered):
    _, result, _ = lowered
    name = recipe.TENSOR_EXPORT.fragment
    record = {**result.facts["layouts"][name], "replication": 2}
    with pytest.raises(AnnotationError) as caught:
        fragment_placement(result.layout_objects[name], record)
    assert caught.value.field == "tensor_export.placement"


def test_exported_calls_have_no_caller_role(recipe, lowered, tmp_path):
    target, result, generated = lowered
    stage = recipe.TENSOR_EXPORT
    record = result.facts["layouts"][stage.fragment]
    rows = fragment_placement(result.layout_objects[stage.fragment], record)
    _, _, measured, _ = lower_fragment(generated, stage, target)
    annotations = _annotations(stage, record, rows, measured)
    source = tmp_path / "annotations.cpp"
    source.write_text(
        '#include <pipelink/stage.hpp>\n'
        + f'#include "{stage.args_header}"\n'
        + f'extern "C" __device__ pipelink_tensor_source {stage.produce_symbol}(const {stage.args.name}*, void*);\n'
        + f'extern "C" __device__ void {stage.release_symbol}(const {stage.args.name}*);\n'
        + annotations
    )
    binary = tmp_path / "annotations"
    compiled = subprocess.run(
        ["g++", "-std=c++17", "-DPIPELINK_EMIT", "-D__device__=", f"-I{PACK / 'include'}",
         f"-I{PACK.parent / 'host/include'}",
         str(source), str(PACK / "runner.cpp"), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)
    calls = {item["name"]: item for item in records["records"] if item["kind"] == "call"}
    assert {(name, item["direction"], item["role"], item["export"]) for name, item in calls.items()} == {
        (stage.produce, "consume", None, "produce"),
        (stage.release, "consume", None, "release"),
    }
    manifest = load_stage_manifest(assemble(records, profile=str(target.attrs["arch"])))
    assert {(endpoint.name, endpoint.owner_role) for endpoint in manifest.endpoints} == {
        (stage.produce, None), (stage.release, None),
    }
