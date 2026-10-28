"""Fragment ownership and caller-frame export contracts on the TileLang tensor-export GEMM fixture."""
from dataclasses import replace
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from tilelang import tvm
from pipec2.pack.assemble import assemble
from pipec2.schema import load_stage_manifest

from pipelink_tilelang import set_context
from pipelink_tilelang.annotate import AnnotationError
from pipelink_tilelang.capture import capture
from pipelink_tilelang.externalize import _nodes, _op_name
from pipelink_tilelang.tensor_export import _annotations, fragment_placement, lower_fragment


ROOT = Path(__file__).resolve().parents[3]
RECIPE = Path(__file__).resolve().parent / "fixtures/tensor_export_gemm.py"
PACK = ROOT / "pipec/src/pipec2/pack"


@pytest.fixture(scope="module", params=("sm_90a", "sm_103a", "sm_120a"))
def recipe(request):
    set_context(request.param, None)
    spec = importlib.util.spec_from_file_location("tensor_export_recipe", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lowered(recipe):
    target = tvm.target.Target({"kind": "cuda", "arch": recipe.context.profile})
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
    elif str(target.attrs["arch"]) == "sm_103a":
        assert rows == [(m, n) for m in range(128) for n in range(128)]
    else:
        assert [rows[i] for i in (0, 1, 2, 8, 8 * 128, 64 * 128)] == [
            (0, 0), (0, 1), (1, 0), (0, 4), (0, 2), (32, 0)]


def test_frame_copy_preserves_physical_fragment_positions(recipe, lowered):
    target, _, generated = lowered
    _, _, measured, module = lower_fragment(generated, recipe.TENSOR_EXPORT, target)
    function = next(func for func in module.functions.values() if len(func.params))
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
    claims = {claim.handle: claim for claim in manifest.claims}
    prefix = "pipelink_tilelang_gemm_reg_"
    if str(target.attrs["arch"]) == "sm_103a":
        expected = {
            "C_tmem_columns": ("tmem", 128, 128), "C_tmem": ("smem", 4, 4),
            "mma_barrier": ("smem", 8, 8),
        }
        for name, (domain, extent, align) in expected.items():
            claim = claims[prefix + name]
            assert (claim.domain, claim.extent.const, claim.align) == (domain, extent, align)
            assert claim.lifetime == "kernel"
    else:
        assert {(claim.domain, claim.extent.const, claim.align) for claim in claims.values()} == {
            ("smem", 65536, 1024), ("named_barrier", 1, 1)}


def test_owned_tmem_moves_cleanup_to_release(recipe, lowered):
    target, _, generated = lowered
    if str(target.attrs["arch"]) != "sm_103a":
        pytest.skip("TMEM ownership is available on sm103")
    _, _, measured, module = lower_fragment(generated, recipe.TENSOR_EXPORT, target)
    produce = next(func for func in module.functions.values() if len(func.params))
    release = next(func for func in module.functions.values() if not len(func.params))
    produce_calls = _nodes(produce.body, tvm.tirx.Call)
    assert not any(_op_name(call) in ("tl.ptx_init_tensor_memory", "tl.ptx_deallocate_tensor_memory")
                   for call in produce_calls)
    release_calls = _nodes(release.body, tvm.tirx.Call)
    free, = [call for call in release_calls if _op_name(call) == "tl.ptx_deallocate_tensor_memory"]
    assert int(free.args[1]) == 128
    sync, = [call for call in release_calls if _op_name(call) == "tirx.tvm_storage_sync"]
    assert int(sync.args[2]) == measured["threads"]
    assert not any(node.buffer.scope() in ("shared", "shared.barrier")
                   for node in _nodes(produce.body, tvm.tirx.AllocBuffer))
    assert any(_op_name(call) == "tl.tcgen05_ld" for call in produce_calls)


def _mutate_generated(generated, rewrite):
    module = generated["modules"]["pre_codegen"]
    variable, function = next(iter(module.functions.items()))
    body = tvm.tirx.stmt_functor.ir_transform(function.body, None, rewrite)
    replacement = tvm.tirx.PrimFunc(function.params, body, ret_type=function.ret_type,
                                  buffer_map=function.buffer_map, attrs=function.attrs)
    return {**generated, "modules": {**generated["modules"],
                                    "pre_codegen": tvm.IRModule({variable: replacement})}}


@pytest.mark.parametrize("case", ("multiple", "missing_allocate", "missing_free", "conditional",
                                  "looped", "while", "width", "cohort"))
def test_invalid_tmem_ownership_is_rejected(recipe, lowered, case):
    target, _, generated = lowered
    if str(target.attrs["arch"]) != "sm_103a":
        pytest.skip("TMEM ownership is available on sm103")
    tir = tvm.tirx

    def rewrite(node):
        if case == "cohort" and isinstance(node, tir.IfThenElse):
            if any(_op_name(call) == "tl.ptx_init_tensor_memory" for call in _nodes(node.then_case, tir.Call)):
                return tir.IfThenElse(tir.const(True), node.then_case, node.else_case)
        if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
            return None
        call = node.value
        op = _op_name(call)
        if op == "tl.ptx_deallocate_tensor_memory":
            if case == "missing_free":
                return tir.Evaluate(0)
            if case == "width":
                return tir.Evaluate(tir.Call(call.dtype, call.op, [call.args[0], tir.const(64)]))
        if op != "tl.ptx_init_tensor_memory":
            return None
        if case == "missing_allocate":
            return tir.Evaluate(0)
        if case == "multiple":
            return tir.SeqStmt([node, node])
        if case == "conditional":
            return tir.IfThenElse(tir.Var("allocate_enabled", "bool"), node, None)
        if case == "looped":
            return tir.For(tir.Var("allocation_index", "int32"), 0, 2, tir.ForKind.SERIAL, node)
        if case == "while":
            return tir.While(tir.Var("allocate_again", "bool"), node)
        return None

    with pytest.raises(AnnotationError) as caught:
        lower_fragment(_mutate_generated(generated, rewrite), recipe.TENSOR_EXPORT, target)
    assert caught.value.field == "tensor_export.tmem"


def test_special_storage_requires_declared_ownership(recipe, lowered):
    target, _, generated = lowered
    if str(target.attrs["arch"]) != "sm_103a":
        pytest.skip("TMEM ownership is available on sm103")
    with pytest.raises(AnnotationError) as caught:
        lower_fragment(generated, replace(recipe.TENSOR_EXPORT, internal_storage=()), target)
    assert caught.value.field == "tensor_export.storage"


def test_tmem_claim_and_free_width_come_from_capture(recipe, lowered):
    target, _, generated = lowered
    if str(target.attrs["arch"]) != "sm_103a":
        pytest.skip("TMEM ownership is available on sm103")
    tir = tvm.tirx

    def rewrite(node):
        if isinstance(node, tir.Call) and _op_name(node) in (
                "tl.ptx_init_tensor_memory", "tl.ptx_deallocate_tensor_memory"):
            return tir.Call(node.dtype, node.op, [node.args[0], tir.const(64)])
        return None

    _, _, measured, module = lower_fragment(
        _mutate_generated(generated, rewrite), recipe.TENSOR_EXPORT, target)
    claim, = [item for item in measured["internal_resources"] if item["domain"] == "tmem"]
    assert (claim["extent"], claim["align"]) == (64, 64)
    release = next(func for func in module.functions.values() if not len(func.params))
    free, = [call for call in _nodes(release.body, tir.Call)
             if _op_name(call) == "tl.ptx_deallocate_tensor_memory"]
    assert int(free.args[1]) == 64


def test_release_participation_comes_from_capture(recipe, lowered):
    target, _, generated = lowered
    if str(target.attrs["arch"]) != "sm_103a":
        pytest.skip("TMEM ownership is available on sm103")
    tir = tvm.tirx

    def rewrite(node):
        if (isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent"
                and node.node.thread_tag == "threadIdx.x"):
            return tir.AttrStmt(node.node, node.attr_key, tir.const(64), node.body)
        return None

    changed = _mutate_generated(generated, rewrite)
    facts = generated["facts"]
    pre_codegen = facts["captures"]["pre_codegen"]
    changed["facts"] = {**facts, "captures": {**facts["captures"], "pre_codegen": {
        **pre_codegen, "attrs": {**pre_codegen["attrs"], "thread_extent": {
            **pre_codegen["attrs"]["thread_extent"], "threadIdx.x": 64}}}}}
    _, _, measured, module = lower_fragment(changed, recipe.TENSOR_EXPORT, target)
    release = next(func for func in module.functions.values() if not len(func.params))
    sync, = [call for call in _nodes(release.body, tir.Call)
             if _op_name(call) == "tirx.tvm_storage_sync"]
    assert int(sync.args[2]) == measured["threads"] == 64
