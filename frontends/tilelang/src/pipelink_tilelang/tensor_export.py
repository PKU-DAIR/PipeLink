"""Export a captured TileLang fragment through a caller-owned per-thread frame.

Recipes mark the completed fragment with ``T.evaluate(export_fragment(fragment))``.
Only that marker and owned resource addresses are rewritten in final TIR; arithmetic,
pipeline scheduling and the physical fragment layout remain TileLang's output.
"""
from dataclasses import dataclass
from itertools import product
import json
from math import prod
from pathlib import Path

import tilelang
from tilelang import tvm

from .annotate import AnnotationError, symbol_prefix
from .capture import _integer
from .declarations import ParamsSpec
from .externalize import _codegen, _nodes, _op_name


tir = tvm.tirx
_MARKER = "pipelink_export_fragment"
# Pinned codegen_cuda.cc PrintStorageScope(shared.dyn), not Buffer.data_alignment.
_SHARED_ALIGNMENT = 1024


@dataclass(frozen=True)
class TensorExportStage:
    stage_id: str
    version: str
    args: ParamsSpec
    args_header: str
    arguments: dict[str, str]
    fragment: str
    name: str
    work: str
    snapshot: str
    numerical: str
    axes: tuple[str, ...]
    ready: str
    done: str
    owner: str
    produce: str
    release: str
    produce_symbol: str
    release_symbol: str
    capabilities: tuple[str, ...]


def export_fragment(buffer):
    """Mark a completed register fragment, without imposing a consumer layout."""
    return tir.call_extern("int32", _MARKER, buffer.data)


def _require(condition, field, detail):
    if not condition:
        raise AnnotationError("tensor_export." + field, detail)


def fragment_placement(layout, record):
    """Enumerate logical coordinates using captured ASTs, never expression text."""
    _require(record.get("replication") == 1, "placement", "export requires an unreplicated fragment")
    variables = layout.get_forward_vars()
    indices = layout.get_forward_index()
    physical = record["physical_shape"]
    rows = []
    for coordinate in product(*(range(size) for size in record["logical_shape"])):
        values = dict(zip(variables, coordinate))
        participant = _integer(layout.thread, values, "tensor_export.placement") + record["thread_min"]
        position = 0
        for index, extent in zip(indices, physical):
            position = position * extent + _integer(index, values, "tensor_export.placement")
        rows.append((participant, position))
    _require(len(set(rows)) == len(rows), "placement", "fragment ownership is not one-to-one")
    _require(all(record["thread_min"] <= thread < record["thread_min"] + record["thread_extent"]
                 and 0 <= position < prod(physical) for thread, position in rows),
             "placement", "fragment address exceeds captured bounds")
    return rows


def lower_fragment(generated, stage, target):
    """Bind the merged shared arena and substitute the terminal frame-copy marker."""
    funcs = list(generated["modules"]["pre_codegen"].functions.values())
    _require(len(funcs) == 1, "module", "expected one single-role device function")
    func = funcs[0]
    allocations = [node.buffer for node in _nodes(func.body, tir.AllocBuffer)]
    fragments = [buffer for buffer in allocations if buffer.name == stage.fragment and buffer.scope() == "local"]
    _require(len(fragments) == 1, "fragment", "expected one flattened local fragment")
    fragment = fragments[0]
    _require(fragment.dtype == "float32" and len(fragment.shape) == 1,
             "fragment", "register export requires a flattened f32 fragment")
    arenas = [buffer for buffer in allocations if buffer.scope() == "shared.dyn"]
    _require(len(arenas) == 1, "storage", "expected one merged shared arena")
    arena = arenas[0]
    _require(not any(buffer.scope() in ("shared.barrier", "shared.cluster_barrier", "shared.tmem")
                     for buffer in allocations), "storage", "unbound special storage")
    facts = generated["facts"]["captures"]["pre_codegen"]
    threads = facts["attrs"]["thread_extent"]["threadIdx.x"]
    _require(not facts["warp_scope"] and all(facts["attrs"]["thread_extent"].get(axis, 1) == 1
                 for axis in ("threadIdx.y", "threadIdx.z")), "role", "expected a single one-dimensional thread cohort")
    prefix = "pipelink_" + symbol_prefix(stage)
    frame = tir.Var(prefix + "frame", tvm.ir.PointerType(tvm.ir.PrimType(fragment.dtype), "global"))
    destination = tir.decl_buffer(fragment.shape, fragment.dtype, name=prefix + "frame_buffer", data=frame)
    markers, syncs, storage_binds = 0, 0, 0

    def replace(node):
        nonlocal markers, syncs, storage_binds
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            call = node.value
            if _op_name(call) == "tirx.call_extern" and call.args[0].value == _MARKER:
                _require(len(call.args) == 2 and call.args[1].same_as(fragment.data),
                         "fragment", "marker must reference the declared fragment")
                markers += 1
                index = tir.Var(prefix + "frame_i", "int32")
                return tir.For(index, 0, fragment.shape[0], tir.ForKind.UNROLLED,
                               tir.BufferStore(destination, tir.BufferLoad(fragment, [index]), [index]))
            if _op_name(call) == "tirx.tvm_storage_sync" and call.args[0].value.startswith("shared"):
                _require(len(call.args) == 1, "barrier", "explicit partial barriers need their own contract")
                syncs += 1
                return tir.Evaluate(tir.Call(call.dtype, call.op, [call.args[0],
                    tir.call_extern("int32", prefix + "barrier"), tir.const(threads)]))
        if isinstance(node, tir.Bind) and isinstance(node.value, tir.Call):
            call = node.value
            if _op_name(call) == "tirx.handle_add_byte_offset" and call.args[0].same_as(arena.data):
                storage_binds += 1
                return tir.Bind(node.var, tir.Call(call.dtype, call.op,
                    [call.args[0], call.args[1] + tir.call_extern("int32", prefix + "smem")]))
        return None

    body = tir.stmt_functor.ir_transform(func.body, None, replace, ["tirx.Evaluate", "tirx.Bind"])
    _require(markers == 1, "fragment", "expected exactly one completed-fragment marker")
    _require(storage_binds > 0, "storage", "merged arena has no operand bindings")
    attrs = dict(func.attrs.items())
    name = prefix + "body"
    attrs.update({"global_symbol": name, "tl.device_function": True})
    lowered = tir.PrimFunc([*func.params, frame], body, ret_type=func.ret_type,
                          buffer_map=func.buffer_map).with_attrs(attrs)
    preamble, source = _codegen({name: lowered}, target)
    arguments = []
    _require(set(stage.arguments) == {param.name for param in func.params},
             "params", "parameter projection must bind every lowered parameter")
    for param in func.params:
        argument = "args->" + stage.arguments[param.name]
        annotation = param.type_annotation
        if isinstance(annotation, tvm.ir.PointerType):
            dtype = str(annotation.element_type.dtype)
            ctype = {"float16": "half_t", "bfloat16": "bfloat16_t", "float32": "float"}.get(dtype)
            _require(ctype is not None, "params", "unsupported pointer element " + dtype)
            argument = f"static_cast<const {ctype}*>({argument})"
        arguments.append(argument)
    arguments.append("static_cast<float*>(frame)")
    declarations = f'extern "C" __device__ int {prefix}smem();\n'
    if syncs:
        declarations += f'extern "C" __device__ int {prefix}barrier();\n'
    wrapper = f'''extern "C" __device__ pipelink_tensor_source {stage.produce_symbol}(const {stage.args.name}* args, void* frame) {{
  {name}({", ".join(arguments)});
  return pl::source_of(frame);
}}
extern "C" __device__ void {stage.release_symbol}(const {stage.args.name}*) {{}}
'''
    measured = {"threads": threads, "frame_words": int(fragment.shape[0]),
                "smem_bytes": prod(int(x) for x in arena.shape) * tvm.DataType(arena.dtype).bits // 8,
                "smem_align": _SHARED_ALIGNMENT, "named_barrier": bool(syncs)}
    return preamble, declarations + source + wrapper, measured, lowered


def _annotations(stage, record, rows, measured):
    q = json.dumps
    prefix = "pipelink_" + symbol_prefix(stage)
    shape = record["logical_shape"]
    _require(len(stage.axes) == len(shape), "axes", "axis names must match the captured logical rank")
    axes = ", ".join(f'{{{q(axis)}, 0, {size}}}' for axis, size in zip(stage.axes, shape))
    coordinates = ", ".join(f'pl::coord{{{q(axis)}, 0, {size}}}' for axis, size in zip(stage.axes, shape))
    linear = "0"
    for axis, size in zip(stage.axes, shape):
        linear = f"({linear} * {size} + {axis})"
    table = ",\n".join(f"      {{{thread}, {position}}}" for thread, position in rows)
    # Register exports leave only the caller frame live: the single operand
    # arena has no mbarriers, and release never accesses it.
    storage = ""
    claims = (f'PL_CLAIM("{prefix}smem", "smem", {measured["smem_bytes"]}, {measured["smem_align"]}, '
              f'pl::during{{{q(stage.name)}, {q(stage.work)}, {q(stage.ready)}}});\n')
    if measured["named_barrier"]:
        claims += f'PL_CLAIM("{prefix}barrier", "named_barrier", 1, 1, "kernel");\n'
        storage = f'''{{"producer_rendezvous", "barrier", 1, 1, "cta", "relocatable",
        "{prefix}barrier", std::nullopt, {q(stage.owner)}, {q(stage.work)}, {q(stage.done)}, pl::sync{{2, 1}}}}'''
    return f'''PL_STAGE({q(stage.stage_id)}, {q(stage.version)});
PL_NO_PARAMS();
PL_TYPE({stage.args.name}, {" ".join(field.cpp_type + " " + field.name + ";" for field in stage.args.fields)});
PL_EXPORT(produce, {stage.produce_symbol});
PL_CALL({stage.produce}, {stage.produce_symbol}, pl::consume, "produce");
PL_EXPORT(release, {stage.release_symbol});
PL_CALL({stage.release}, {stage.release_symbol}, pl::consume, "release");
{claims}PL_EXPORT_TENSOR({stage.name},
  pl::identity{{{q(stage.name)}, {q(stage.work)}, {q(stage.snapshot)}, "f32", {q(stage.numerical)}, {{{axes}}}}},
  pl::placement{{"reg", "scalar", {{"participant", "position"}}, []({", ".join("int " + axis for axis in stage.axes)}) {{
    static constexpr std::array<int, 2> addresses[] = {{
{table}
    }};
    return addresses[{linear}];
  }}, {coordinates}}},
  pl::strings{{{", ".join(q(item) for item in stage.capabilities)}}},
  pl::readiness{{{q(stage.ready)}, {q(stage.done)}}}, pl::fixed_bindings{{}},
  pl::lends{{{storage}}}, pl::calls{{{q(stage.produce)}, {q(stage.release)}}});
'''


def emit(out, module, generated):
    import pipelink_tilelang
    stage = module.TENSOR_EXPORT
    target = tvm.target.Target({"kind": "cuda", "arch": pipelink_tilelang.context.profile})
    record = generated["facts"]["layouts"][stage.fragment]
    rows = fragment_placement(generated["layout_objects"][stage.fragment], record)
    preamble, body, measured, _ = lower_fragment(generated, stage, target)
    _require(record["physical_shape"] == [measured["frame_words"]], "fragment", "layout differs from final allocation")
    annotations = _annotations(stage, record, rows, measured)
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    (path / "roles.cu").write_text('#include <pipelink/stage.hpp>\n' + f'#include "{stage.args_header}"\n'
        + "#if defined(__CUDACC__)\n" + preamble + body + "#endif\n"
        + "#ifdef PIPELINK_EMIT\n" + annotations + "#endif\n")
    (path / "facts.json").write_text(json.dumps(generated["facts"], indent=2) + "\n")
    root = Path(tilelang.__file__).resolve().parent
    build = {"stage_id": stage.stage_id, "profile": pipelink_tilelang.context.profile,
             "sources": [str((path / "roles.cu").resolve())],
             "includes": [str(root / "src"), str(root / "3rdparty/cutlass/include")],
             "defines": [], "args_type": stage.args.name, **measured}
    (path / "build.json").write_text(json.dumps(build, indent=2) + "\n")
    return build
