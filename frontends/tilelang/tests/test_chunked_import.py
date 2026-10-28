"""Chunk scopes, physical subset ownership and mutable per-thread arguments."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

import pipelink_tilelang as pl
from pipelink_tilelang.explicit import _demands, emit
from test_contract_tensor_import import ROOT, _partner, _lower, _registry, TMEM_ROWS
from pipec2.pack.assemble import assemble
from pipec2.schema import load_stage_manifest
from pipec2.tensor.expr import evaluate

BANDED_REGISTERS = dict(guard=None, space="reg", units="scalar", components={
    "participant": ["add", ["add", ["mulc", 64, ["divc", ["coord", "n"], 64]],
                                    ["mulc", 32, ["divc", ["coord", "m"], 64]]],
                    ["add", ["mulc", 4, ["modc", ["coord", "m"], 8]],
                            ["divc", ["modc", ["coord", "n"], 8], 2]]],
    "position": ["add", ["mulc", 16, ["divc", ["modc", ["coord", "n"], 64], 8]],
                 ["add", ["mulc", 4, ["divc", ["modc", ["coord", "m"], 64], 16]],
                  ["add", ["mulc", 2, ["divc", ["modc", ["coord", "m"], 16], 8]],
                   ["modc", ["coord", "n"], 2]]]]})

WGMMA_SUBSET = dict(guard=None, space="reg", units="scalar", components={
    "participant": ["add", ["const", 128],
                    ["add", ["mulc", 32, ["divc", ["coord", "m"], 16]],
                     ["add", ["mulc", 4, ["modc", ["coord", "m"], 8]],
                      ["divc", ["modc", ["coord", "n"], 8], 2]]]],
    "position": ["add", ["mulc", 4, ["divc", ["coord", "n"], 8]],
                 ["add", ["mulc", 2, ["divc", ["modc", ["coord", "m"], 16], 8]],
                  ["modc", ["coord", "n"], 2]]]})


def _assert_converged_shared_barriers(function):
    from tilelang import tvm
    from pipelink_tilelang.capture import _nodes
    tir = tvm.tirx
    threads = [node.node.var for node in _nodes(function.body, tir.AttrStmt)
               if node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x"]
    for branch in _nodes(function.body, tir.IfThenElse):
        if any(var.same_as(thread) for var in _nodes(branch.condition, tir.Var) for thread in threads):
            assert not any(getattr(call.op, "name", None) == "tirx.tvm_storage_sync"
                           for call in _nodes(branch, tir.Call))

def _assert_chunk_callable(imported):
    from pipec2.tensor.lower import _declaration
    assert [(p.name, p.ctype, p.meaning) for p in imported.callable.params] == [
        ("source", "pipelink_tensor_source", "source_base"),
        ("chunk", "uint32_t", "binding"),
        ("fragment", "float*", "dest_fragment"),
        ("scratch", "void*", "scratch"),
    ]
    _declaration(imported)



@pytest.mark.parametrize("profile,first", [("sm_103a", 0), ("sm_90a", 128), ("sm_120a", 0)])
def test_chunked_recipe_captures_physical_subset_and_outputs(tmp_path, monkeypatch, profile, first):
    placement = dict(guard=None, space="reg", units="scalar", components={
        "participant": ["add", ["const", first], ["add", ["mulc", 2, ["coord", "m"]], ["modc", ["divc", ["coord", "n"], 16], 2]]],
        "position": ["add", ["mulc", 16, ["divc", ["coord", "n"], 32]], ["modc", ["coord", "n"], 16]]})
    if profile == "sm_103a":
        placement = TMEM_ROWS
    elif profile == "sm_120a":
        placement = BANDED_REGISTERS
    elif profile == "sm_90a":
        placement = WGMMA_SUBSET
    partner = _partner(tmp_path, "chunk", "accumulator", [("m", 128), ("n", 256)], "chunk_args",
                       [("const void*", "a"), ("const void*", "b"), ("int32_t", "M"), ("int32_t", "N"),
                        ("int32_t", "K"), ("int32_t", "m0"), ("int32_t", "n0")],
                       profile=profile, placement=placement, mutable=True, view_call=True)
    monkeypatch.setenv("PIPELINK_ACC_PARTNER", str(partner.path))
    monkeypatch.setattr(pl, "context", pl.context)
    pl.set_context(profile, None)
    spec = importlib.util.spec_from_file_location("chunk_recipe", ROOT / "campaigns/coda/consumer-tilelang/swiglu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = _lower(module, profile)
    from tilelang import tvm
    from pipelink_tilelang.capture import _nodes
    function = next(iter(generated[None]["modules"]["pre_codegen"].functions.values()))
    _assert_converged_shared_barriers(function)
    intrinsics = {call.args[0].value for call in _nodes(function.body, tvm.tirx.Call)
                  if getattr(call.op, "name", None) == "tirx.call_pure_extern"}
    assert {"__fdividef", "__expf"} <= intrinsics
    record, rows = _demands(module.STAGE, {}, generated)[module.acc.name]
    assert record["logical_shape"] == [128, 32]
    owners, positions = (64, 64) if profile == "sm_120a" else (256, 16)
    assert {p for _, p, _ in rows} == set(range(first, first + owners))
    assert {p for _, _, p in rows} == set(range(positions))
    fields = {f.name for f in module.STAGE.params.fields}
    assert {"a", "b", "M", "N", "K"} <= fields
    assert not {"m0", "n0"} & fields
    assert "single_completion" in module.acc.demand.guarantees
    assert next(lend["extent"] for lend in module.acc.lends if lend["kind"] == "frame") == 128
    assert not any(lend["kind"] == "smem" for lend in module.acc.lends)
    assert generated[None]["facts"]["layouts"]["staging"]["physical_shape"] == [128, 128]
    projection = emit(tmp_path, module, generated)[module.STAGE.stage_id]
    assert projection["source"]
    consumer = load_stage_manifest(assemble({"records": _registry(tmp_path, module, generated)}, profile=profile))
    imported, = consumer.tensor.imports
    _assert_chunk_callable(imported)
    demand, = imported.demands
    for chunk in range(8):
        env = {"i0": 127, "i1": 31, "chunk": chunk}
        assert evaluate(demand.identity_coords["n"], env) == 32 * chunk + 31
        if profile != "sm_103a":
            source = partner.manifest.tensor.exports[0].placements[0]
            coords = {"m": 127, "n": 32 * chunk + 31}
            assert evaluate(demand.participant, env) == evaluate(source.components["participant"], coords)
            assert evaluate(demand.position, env) == evaluate(source.components["position"], coords) % positions
    if profile == "sm_120a":
        assert generated[None]["facts"]["chunk_layouts"][module.acc.symbol]["thread_offsets"] == [0, 0, 64, 64, 128, 128, 192, 192]
    assert consumer.tensor.bindings[0].domain.hi == 8


@pytest.mark.parametrize("profile,first", [("sm_103a", 0), ("sm_90a", 128), ("sm_120a", 0)])
def test_compiled_fixture_producer_and_chunk_consumer(tmp_path, monkeypatch, profile, first):
    from tilelang.env import TILELANG_TEMPLATE_PATH
    monkeypatch.syspath_prepend(str(ROOT))
    from tools.pipelink_env import load
    monkeypatch.setenv("PIPELINK_CUDA_HOME", load(ROOT)["PIPELINK_CUDA_HOME"])
    producer = tmp_path / "producer.plink"
    fixture = Path(__file__).parent / "fixtures/chunk_producer.cu"

    def pack(source, destination, *flags):
        completed = subprocess.run(
            [str(Path(sys.executable).with_name("pipec2")), "pack", str(source), "--profile", profile,
             "-o", str(destination), *flags], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    define = {"sm_90a": "WGMMA_PLACEMENT=1", "sm_120a": "BANDED_PLACEMENT=1"}.get(profile)
    flags = ["-D", define] if define else []
    pack(fixture, producer, "-D", f"FIRST_THREAD={first}", *flags)
    monkeypatch.setenv("PIPELINK_ACC_PARTNER", str(producer))
    monkeypatch.setattr(pl, "context", pl.context)
    pl.set_context(profile, None)
    spec = importlib.util.spec_from_file_location("packed_chunk_recipe", ROOT / "campaigns/coda/consumer-tilelang/swiglu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = _lower(module, profile)
    _assert_converged_shared_barriers(next(iter(generated[None]["modules"]["pre_codegen"].functions.values())))
    projection = emit(tmp_path, module, generated)[module.STAGE.stage_id]
    consumer = tmp_path / "consumer.plink"
    pack(projection["source"], consumer, "-I", TILELANG_TEMPLATE_PATH,
         "-I", str(ROOT / "third_party/cutlass/include"))
    with zipfile.ZipFile(consumer) as archive:
        manifest = load_stage_manifest(json.loads(archive.read("manifest.json")))
    assert manifest.params.bytes == 0
    assert module.STAGE.host.residue_type == "CodaEpilogueParams"
    assert set(module.acc.outputs) == {"m0", "n0"}
    assert len(manifest.tensor.imports) == 1
    _assert_chunk_callable(manifest.tensor.imports[0])


def test_chunk_thread_translation_preserves_lane_and_bijection():
    from tilelang import tvm
    from pipelink_tilelang.capture import _chunk_thread_remap, _integer, _nodes, CaptureError
    tir = tvm.tirx
    tx, chunk = tir.Var("tx", "int32"), tir.Var("chunk", "int32")
    axis = tir.IterVar(tvm.ir.Range(0, 256), tx, tir.IterVar.ThreadIndex, "threadIdx.x")
    buffer = tir.decl_buffer((256,), "int32")
    body = tir.For(chunk, 0, 8, tir.ForKind.SERIAL, tir.SeqStmt([
        tir.Evaluate(tir.call_extern("int32", "import_chunk", chunk)),
        tir.BufferStore(buffer, tx, [tx])]))
    function = tir.PrimFunc([], tir.AttrStmt(axis, "thread_extent", 256, body))
    module = tvm.IRModule.from_expr(function)
    offsets = {"import_chunk": [0, 0, 64, 64, 128, 128, 192, 192]}
    translated = _chunk_thread_remap(module, offsets)
    store, = _nodes(next(iter(translated.functions.values())).body, tir.BufferStore)
    for c, offset in enumerate(offsets["import_chunk"]):
        permutation = [_integer(store.value, {tx: t, chunk: c}, "translation") for t in range(256)]
        assert set(permutation) == set(range(256))
        assert permutation == [(t - offset) % 256 for t in range(256)]
        assert all(value % 32 == t % 32 for t, value in enumerate(permutation))
    with pytest.raises(CaptureError):
        _chunk_thread_remap(module, offsets, initial=True)


@pytest.mark.parametrize("count", [1, 2])
def test_shared_barrier_convergence_preserves_work_and_generations(count):
    from tilelang import tvm
    from pipelink_tilelang.capture import _converge_shared_barriers
    tir = tvm.tirx
    tx = tir.Var("tx", "int32")
    axis = tir.IterVar(tvm.ir.Range(0, 384), tx, tir.IterVar.ThreadIndex, "threadIdx.x")
    buffer = tir.decl_buffer((384,), "int32")
    sync = tir.Evaluate(tir.call_intrin("int32", "tirx.tvm_storage_sync", "shared"))
    statements = [tir.BufferStore(buffer, tir.const(0), [tx])]
    for index in range(count):
        statements.extend([sync, tir.BufferStore(buffer, tir.const(index + 1), [tx])])
    function = tir.PrimFunc([], tir.AttrStmt(axis, "thread_extent", 384,
        tir.IfThenElse(tx >= 128, tir.SeqStmt(statements), None)))
    transformed, splits = _converge_shared_barriers(tvm.IRModule.from_expr(function))
    output = next(iter(transformed.functions.values()))
    assert splits == count
    _assert_converged_shared_barriers(output)

    def trace(statement, thread):
        if isinstance(statement, tir.AttrStmt):
            return trace(statement.body, thread)
        if isinstance(statement, tir.SeqStmt):
            return [event for child in statement.seq for event in trace(child, thread)]
        if isinstance(statement, tir.IfThenElse):
            predicate = tvm.arith.Analyzer().simplify(
                tir.stmt_functor.substitute(statement.condition, {tx: tir.const(thread)}))
            return trace(statement.then_case, thread) if int(predicate) else []
        if isinstance(statement, tir.BufferStore):
            return [("write", int(statement.value))]
        assert isinstance(statement, tir.Evaluate)
        return ["barrier"]

    for thread in range(384):
        events = trace(output.body, thread)
        assert events.count("barrier") == count
        if thread < 128:
            assert events == ["barrier"] * count
        else:
            assert events == trace(function.body, thread)


@pytest.mark.parametrize("case", ["else", "nested_loop", "mutable_predicate"])
def test_shared_barrier_convergence_rejects_unsupported_control(case):
    from tilelang import tvm
    from pipelink_tilelang.capture import _converge_shared_barriers, CaptureError
    tir = tvm.tirx
    tx = tir.Var("tx", "int32")
    axis = tir.IterVar(tvm.ir.Range(0, 384), tx, tir.IterVar.ThreadIndex, "threadIdx.x")
    sync = tir.Evaluate(tir.call_intrin("int32", "tirx.tvm_storage_sync", "shared"))
    condition, body, otherwise = tx >= 128, sync, None
    if case == "else":
        otherwise = tir.Evaluate(0)
    elif case == "nested_loop":
        body = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, sync)
    else:
        predicate = tir.decl_buffer((384,), "int32")
        condition = tir.And(condition, predicate[tx] != 0)
    function = tir.PrimFunc([], tir.AttrStmt(axis, "thread_extent", 384,
        tir.IfThenElse(condition, body, otherwise)))
    with pytest.raises(CaptureError) as caught:
        _converge_shared_barriers(tvm.IRModule.from_expr(function))
    assert caught.value.fact == "barriers.converged"
