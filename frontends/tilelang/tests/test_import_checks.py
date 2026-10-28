"""GPU-free whole-kernel schedule and event-lived staging rejection fixtures."""
from types import SimpleNamespace

import pytest
import tilelang.language as T
from tilelang import tvm

from pipelink_tilelang.annotate import AnnotationError
from pipelink_tilelang.contract import import_tensor
from pipelink_tilelang.explicit import RoleSpec
from pipelink_tilelang.import_checks import validate_imports
from test_contract_tensor_import import _partner


tir = tvm.tirx


@pytest.fixture
def kernel(tmp_path):
    partner = _partner(tmp_path, "acc", "foreign_export", [("m", 32)], "pl_args", [("int32_t", "M")])
    channel = import_tensor(partner, "foreign_export", symbol="local_import")
    channel.role = "worker"
    channel._fragment = tir.decl_buffer((32,), "float32", name="fragment", scope="local")
    scratch = tir.decl_buffer((128,), "uint8", name="scratch", scope="shared")
    staging = tir.decl_buffer((32,), "float32", name="staging", scope="shared")
    channel._buffers = {"scratch": (scratch, "pipelink_scratch")}
    channel._staging = [(staging, "pipelink_staging")]
    stage = SimpleNamespace(channels=[channel], roles={"worker": RoleSpec((0, 4), ())})
    return SimpleNamespace(channel=channel, stage=stage, fragment=channel._fragment,
                           scratch=scratch, staging=staging)


def _call(symbol, *args):
    return tir.call_extern("int32", symbol, *args)


def _sequence(kernel, middle=()):
    channel = kernel.channel
    produce = _call(channel.produce_symbol)
    ready = _call(channel.symbol + "_ready", produce)
    imported = tir.Evaluate(_call(channel.symbol, ready, kernel.fragment.access_ptr("w"),
                                 kernel.scratch.access_ptr("rw")))
    return [imported, *middle, tir.Evaluate(_call(channel.symbol + "_done")),
            tir.Evaluate(_call(channel.release_symbol))]


def _module(statements):
    body = tir.SeqStmt(statements) if isinstance(statements, list) else statements
    return tvm.IRModule.from_expr(tir.PrimFunc([], body))


def _check(kernel, statements, *, initial=None, threads=128, other_extents=None):
    final = _module(statements)
    generated = {None: {"modules": {"initial": final if initial is None else _module(initial), "pre_codegen": final},
                       "facts": {"captures": {"pre_codegen": {"attrs": {
                           "thread_extent": {"threadIdx.x": threads, **(other_extents or {})}}}}}}}
    validate_imports(kernel.stage, generated)


def _reject(kernel, statements, suffix, **kwargs):
    with pytest.raises(AnnotationError) as caught:
        _check(kernel, statements, **kwargs)
    assert caught.value.field == f"tensor.imports.local_import.{suffix}"


def _use(buffer):
    return tir.BufferStore(buffer, tir.const(1, buffer.dtype), [0])


def test_valid_nested_sequence_and_looped_synchronous_accesses(kernel):
    i = tir.Var("i", "int32")
    body = tir.BufferStore(kernel.staging, kernel.fragment[i], [i])
    loop = tir.For(i, 0, 32, tir.ForKind.SERIAL, body)
    _check(kernel, _sequence(kernel, [loop, _use(kernel.scratch)]))


@pytest.mark.parametrize("case,guarantee", [
    ("successive_scopes", "enter_once"), ("missing_produce", "enter_once"),
    ("missing_ready", "enter_once"), ("missing_import", "enter_once"),
    ("missing_completion", "single_completion"), ("duplicate_completion", "single_completion"),
    ("early_completion", "single_completion"), ("missing_release", "single_completion"),
    ("duplicate_release", "single_completion"), ("early_release", "single_completion"),
    ("import_after_completion", "enter_once"), ("fragment_after_completion", "single_completion"),
    ("scratch_after_completion", "single_completion"), ("fragment_before_import", "enter_once"),
])
def test_whole_kernel_occurrence_and_last_use_rejections(kernel, case, guarantee):
    statements = _sequence(kernel)
    c = kernel.channel
    if case == "successive_scopes":
        statements += _sequence(kernel)
    elif case in ("missing_produce", "missing_ready"):
        source = _call(c.symbol + "_ready") if case == "missing_produce" else _call(c.produce_symbol)
        statements[0] = tir.Evaluate(_call(c.symbol, source))
    elif case == "missing_import":
        statements[0] = tir.Evaluate(_call(c.symbol + "_ready", _call(c.produce_symbol)))
    elif case == "missing_completion":
        del statements[1]
    elif case == "duplicate_completion":
        statements.insert(2, statements[1])
    elif case == "early_completion":
        statements[0], statements[1] = statements[1], statements[0]
    elif case == "missing_release":
        statements.pop()
    elif case == "duplicate_release":
        statements.append(statements[-1])
    elif case == "early_release":
        statements[1], statements[2] = statements[2], statements[1]
    elif case == "import_after_completion":
        statements.append(tir.Evaluate(_call(c.symbol)))
    elif case == "fragment_before_import":
        statements.insert(0, _use(kernel.fragment))
    else:
        statements.append(_use(kernel.fragment if case == "fragment_after_completion" else kernel.scratch))
    _reject(kernel, statements, "guarantees." + guarantee)


@pytest.mark.parametrize("case", ["serial_scope", "single_iteration_scope", "thread_branch", "while_scope",
                                   "branched_completion", "expression_branch"])
def test_non_top_level_import_rejections(kernel, case):
    statements = _sequence(kernel)
    thread = tir.Var("threadIdx.x", "int32")
    if case in ("serial_scope", "single_iteration_scope"):
        statements = tir.For(tir.Var("i", "int32"), 0, 1 if case == "single_iteration_scope" else 2,
                             tir.ForKind.SERIAL, tir.SeqStmt(statements))
    elif case == "thread_branch":
        statements = tir.IfThenElse(thread < 32, tir.SeqStmt(statements), None)
    elif case == "while_scope":
        statements = tir.While(thread < 32, tir.SeqStmt(statements))
    elif case == "branched_completion":
        statements[1] = tir.IfThenElse(thread < 32, statements[1], None)
    else:
        statements[0] = tir.Evaluate(tir.if_then_else(thread < 32, statements[0].value, 0))
    _reject(kernel, statements, "guarantees.converged")


@pytest.mark.parametrize("position", [0, 1, 3])
def test_no_early_return_anywhere_in_kernel(kernel, position):
    statements = _sequence(kernel)
    statements.insert(position, tir.Evaluate(tir.ret(0)))
    _reject(kernel, statements, "guarantees.no_early_return")


@pytest.mark.parametrize("threads,role,other", [(33, (0, 1), {}), (128, (1, 5), {}),
                                                (128, (0, 2), {}), (128, (0, 4), {"threadIdx.y": 2})])
def test_full_kernel_complete_warps_required(kernel, threads, role, other):
    kernel.stage.roles["worker"] = RoleSpec(role, ())
    _reject(kernel, _sequence(kernel), "guarantees.full_warps", threads=threads, other_extents=other)


@pytest.mark.parametrize("position", [0, 2, 3])
def test_staging_before_import_or_after_done_rejected(kernel, position):
    statements = _sequence(kernel)
    statements.insert(position, _use(kernel.staging))
    _reject(kernel, statements, "staging.staging.lifetime")


@pytest.mark.parametrize("operation", ["async_copy", "tma_copy", "cp_async", "bulk", "opaque_pointer", "opaque_scalar"])
def test_asynchronous_and_opaque_staging_rejected(kernel, operation):
    global_buffer = tir.decl_buffer((32,), "float32", name="global")
    if operation == "async_copy":
        access = T.async_copy(global_buffer, kernel.staging)
    elif operation == "tma_copy":
        access = T.tma_copy(kernel.staging, global_buffer)
    elif operation == "cp_async":
        access = tir.call_intrin("handle", tvm.ir.Op.get("tl.ptx_cp_async"), kernel.staging.access_ptr("w"),
                                 global_buffer.access_ptr("r"), 4)
    elif operation == "bulk":
        access = tir.call_intrin("handle", tvm.ir.Op.get("tl.tma_store"), global_buffer.access_ptr("w"),
                                 kernel.staging.access_ptr("r"), 128, 0, 0)
    elif operation == "opaque_pointer":
        access = _call("owner_opaque", kernel.staging.access_ptr("rw"))
    else:
        access = _call("owner_opaque", kernel.staging[0])
    _reject(kernel, _sequence(kernel, [tir.Evaluate(access)]), "staging.staging.quiescent")


@pytest.mark.parametrize("alias", ["pointer", "bind", "view", "match_buffer"])
def test_staging_address_and_view_escapes_rejected(kernel, alias):
    if alias == "pointer":
        use = tir.Evaluate(kernel.staging.access_ptr("r"))
    elif alias == "bind":
        use = tir.Bind(tir.Var("escaped", "handle"), kernel.staging.data)
    else:
        view = tir.decl_buffer((16,), "float32", name="view", data=kernel.staging.data, scope="shared")
        if alias == "view":
            use = tir.DeclBuffer(view)
        else:
            region = tir.BufferRegion(kernel.staging, [tvm.ir.Range(0, 16)])
            use = tir.SBlock([], [], [], "alias", _use(view), match_buffers=[tir.MatchBufferRegion(view, region)])
    _reject(kernel, _sequence(kernel, [use]), "staging.staging.no_alias")


def test_initial_async_provenance_cannot_be_erased_by_lowering(kernel):
    source = tir.decl_buffer((32,), "float32", name="global")
    initial = _sequence(kernel, [tir.Evaluate(T.async_copy(source, kernel.staging))])
    _reject(kernel, _sequence(kernel, [_use(kernel.staging)]), "staging.staging.quiescent", initial=initial)


def test_synchronous_copy_pointer_lowering_is_not_an_escape(kernel):
    source = tir.decl_buffer((32,), "float32", name="global")
    initial = _sequence(kernel, [tir.Evaluate(T.copy(source, kernel.staging))])
    lowered = tir.call_intrin("uint32", tvm.ir.Op.get("tl.lds32"), kernel.staging.access_ptr("r"))
    _check(kernel, _sequence(kernel, [tir.Evaluate(lowered)]), initial=initial)


def test_compiler_waited_async_lowering_requires_plain_copy_provenance(kernel):
    source = tir.decl_buffer((32,), "float32", name="global")
    initial = _sequence(kernel, [tir.Evaluate(T.copy(source, kernel.staging))])
    copy = tir.call_intrin("handle", tvm.ir.Op.get("tl.ptx_cp_async"), kernel.staging.access_ptr("w"), source.access_ptr("r"), 4)
    wait = tir.call_intrin("int32", "tirx.ptx_wait_group", 0)
    lowered = _sequence(kernel, [tir.Evaluate(copy), tir.Evaluate(wait)])
    _check(kernel, lowered, initial=initial)
    _reject(kernel, lowered, "staging.staging.quiescent", initial=_sequence(kernel, [_use(kernel.staging)]))


def test_final_tir_order_is_checked_independently_of_source(kernel):
    initial = _sequence(kernel, [_use(kernel.fragment)])
    final = _sequence(kernel)
    final.append(_use(kernel.fragment))
    _reject(kernel, final, "guarantees.single_completion", initial=initial)


def test_unrelated_async_buffer_is_not_rejected(kernel):
    unrelated = tir.decl_buffer((32,), "float32", name="unrelated", scope="shared")
    source = tir.decl_buffer((32,), "float32", name="global")
    _check(kernel, _sequence(kernel, [tir.Evaluate(T.async_copy(source, unrelated))]))


def test_explicit_async_with_wait_is_still_outside_the_admitted_source_forms(kernel):
    source = tir.decl_buffer((32,), "float32", name="global")
    copy = tir.Evaluate(T.async_copy(source, kernel.staging))
    wait = tir.Evaluate(tir.call_intrin("int32", "tirx.ptx_wait_group", 0))
    _reject(kernel, _sequence(kernel, [copy, wait]), "staging.staging.quiescent")


def test_final_staging_access_cannot_move_past_completion(kernel):
    initial = _sequence(kernel, [_use(kernel.staging)])
    final = _sequence(kernel)
    final.append(_use(kernel.staging))
    _reject(kernel, final, "staging.staging.lifetime", initial=initial)


def test_single_iteration_loop_cannot_disappear_into_a_valid_schedule(kernel):
    initial = tir.For(tir.Var("i", "int32"), 0, 1, tir.ForKind.SERIAL,
                      tir.SeqStmt(_sequence(kernel)))
    _reject(kernel, _sequence(kernel), "guarantees.converged", initial=initial)


def test_import_without_fragment_has_a_named_guarantee_error(kernel):
    kernel.channel._fragment = None
    _reject(kernel, _sequence(kernel), "guarantees.enter_once")


def test_partner_without_release_requires_only_done(kernel):
    statements = _sequence(kernel)[:-1]
    kernel.channel.release_name = None
    _check(kernel, statements)


def _chunks(kernel, *, count=8, binding=None, body_tail=()):
    channel = kernel.channel
    channel.chunk = (0, 4, 8)
    chunk = tir.Var("chunk", "int32")
    imported = tir.Evaluate(_call(channel.symbol, tir.const(0, "uint64"),
                                 chunk if binding is None else binding,
                                 kernel.fragment.access_ptr("w"), kernel.scratch.access_ptr("rw")))
    loop = tir.For(chunk, 0, count, tir.ForKind.SERIAL, tir.SeqStmt([
        imported, _use(kernel.fragment), _use(kernel.staging),
        tir.Evaluate(_call(channel.symbol + "_done")), *body_tail]))
    return [tir.Evaluate(_call(channel.produce_symbol)), tir.Evaluate(_call(channel.symbol + "_ready")),
            loop, tir.Evaluate(_call(channel.release_symbol))]


def test_static_chunk_loop_has_one_ready_and_per_chunk_completion(kernel):
    _check(kernel, _chunks(kernel))


def test_chunk_owner_staging_can_drain_after_last_completion(kernel):
    statements = _chunks(kernel)
    statements.insert(-1, _use(kernel.staging))
    _check(kernel, statements)
    _reject(kernel, [*statements, _use(kernel.staging)], "staging.staging.lifetime")
    _reject(kernel, [_use(kernel.staging), *statements], "staging.staging.lifetime")


@pytest.mark.parametrize("case", ["short", "dynamic", "wrong_binding", "branched", "outside_done",
                                  "two_loops", "after_done", "escape"])
def test_chunk_loop_rejects_unproved_schedule(kernel, case):
    statements = _chunks(kernel)
    guarantee = "converged"
    if case == "short":
        statements = _chunks(kernel, count=7)
    elif case == "dynamic":
        statements = _chunks(kernel, count=tir.Var("count", "int32"))
    elif case == "wrong_binding":
        statements = _chunks(kernel, binding=tir.const(0, "int32"))
    elif case == "branched":
        statements[2] = tir.IfThenElse(tir.Var("condition", "bool"), statements[2], None)
    elif case == "outside_done":
        loop = statements[2]
        statements[2] = tir.For(loop.loop_var, loop.min, loop.extent, loop.kind,
                                tir.SeqStmt(list(loop.body.seq)[:-1]))
        statements.insert(3, loop.body.seq[-1])
    elif case == "two_loops":
        statements.insert(3, statements[2])
        guarantee = "enter_once"
    elif case == "after_done":
        statements = _chunks(kernel, body_tail=[_use(kernel.fragment)])
        guarantee = "single_completion"
    elif case == "escape":
        statements = _chunks(kernel, body_tail=[
            tir.Evaluate(_call("opaque", kernel.fragment.access_ptr("r")))])
        guarantee = "single_completion"
    _reject(kernel, statements, "guarantees." + guarantee)


def test_producer_output_reads_follow_produce(kernel):
    kernel.channel.outputs = ("origin",)
    read = tir.Evaluate(_call(kernel.channel.symbol + "_args_origin"))
    _check(kernel, _sequence(kernel, [read]))
    _reject(kernel, [read, *_sequence(kernel)], "guarantees.enter_once")


def _final_chunks(kernel):
    statements = _chunks(kernel)
    kernel.channel.completion = "final"
    kernel.channel._scratch_bytes = 0
    kernel.channel._buffers = {}
    loop = statements[2]
    body = list(loop.body.seq)
    body[0] = tir.Evaluate(_call(kernel.channel.symbol, tir.const(0, "uint64"), loop.loop_var,
                                 kernel.fragment.access_ptr("w"), tir.const(0, "uint64")))
    done = body.pop()
    statements[2] = tir.For(loop.loop_var, loop.min, loop.extent, loop.kind, tir.SeqStmt(body))
    statements.insert(3, done)
    return statements


def test_zero_scratch_chunks_can_complete_once_after_final_drain(kernel):
    statements = _final_chunks(kernel)
    statements.insert(3, _use(kernel.staging))
    _check(kernel, statements)


@pytest.mark.parametrize("case", ["scratch_capacity", "scratch_storage", "fragment_escape", "done_in_loop"])
def test_final_completion_rejects_unsafe_reuse(kernel, case):
    statements = _final_chunks(kernel)
    guarantee = "single_completion"
    if case == "scratch_capacity":
        kernel.channel._scratch_bytes = 128
    elif case == "scratch_storage":
        kernel.channel._buffers = {"scratch": (kernel.scratch, "scratch")}
    elif case == "fragment_escape":
        statements.insert(3, _use(kernel.fragment))
    else:
        loop = statements[2]
        statements[2] = tir.For(loop.loop_var, loop.min, loop.extent, loop.kind,
                               tir.SeqStmt([*loop.body.seq, statements.pop(3)]))
        guarantee = "converged"
    _reject(kernel, statements, "guarantees." + guarantee)
