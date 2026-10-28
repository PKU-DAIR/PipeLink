"""Export a captured TileLang fragment through a caller-owned per-thread frame.

Recipes mark the completed fragment with ``T.evaluate(export_fragment(fragment))``.
Only that marker and owned resource addresses are rewritten in final TIR; arithmetic,
pipeline scheduling and the physical fragment layout remain TileLang's output.
"""
from dataclasses import dataclass
import json
from math import prod
from pathlib import Path

import tilelang
from tilelang import tvm

from .annotate import AnnotationError, symbol_prefix
from .capture import CaptureError, fragment_rows
from .declarations import ParamsSpec
from .externalize import _barrier_operands, _bind_smem, _codegen, _local_barrier, _nodes, _op_name


tir = tvm.tirx
_MARKER = "pipelink_export_fragment"
# From the pinned codegen_cuda.cc PrintStorageScope(shared.dyn); Buffer.data_alignment differs.
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
    internal_storage: tuple[str, ...] = ()


def export_fragment(buffer):
    """Mark a completed register fragment for export in its own layout."""
    return tir.call_extern("int32", _MARKER, buffer.data)


def _require(condition, field, detail):
    if not condition:
        raise AnnotationError("tensor_export." + field, detail)


def fragment_placement(layout, record):
    """Producer placement in logical row-major order; an export owns each value once."""
    _require(record.get("replication") == 1, "placement", "export requires an unreplicated fragment")
    try:
        rows = fragment_rows(layout, record, "tensor_export.placement")
    except CaptureError as error:
        raise AnnotationError("tensor_export.placement", str(error)) from error
    return [(participant, position) for _, participant, position in rows]


@dataclass(frozen=True)
class _OwnedTmem:
    buffer: tir.Buffer
    columns: int
    deallocate: tir.Call
    guard: tir.PrimExpr


def _owned_storage(func, names, prefix, threads):
    """Validate ownership against final TIR before changing allocation or cleanup."""
    allocations = [node.buffer for node in _nodes(func.body, tir.AllocBuffer)]
    owned = {buffer.name: buffer for buffer in allocations if buffer.name in names}
    _require(len(set(names)) == len(names) and set(owned) == set(names),
             "storage", "internal storage must name distinct captured allocations")
    _require(all(buffer.name in owned for buffer in allocations
                 if buffer.scope().startswith("shared") and buffer.scope() != "shared.dyn"),
             "storage", "unbound special storage")
    thread = next(node for node in _nodes(func.body, tir.AttrStmt)
                  if node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x")
    calls = _nodes(func.body, tir.Call)
    statements = _nodes(func.body, tir.Evaluate)
    initializers = [call for call in calls if _op_name(call) == "tirx.ptx_init_barrier_thread_count"]
    barriers = {name: _local_barrier(buffer, initializers, prefix + name)
                for name, buffer in owned.items() if buffer.scope() == "shared.barrier"}
    sites, scopes, markers, captured = {}, [], [], []
    order = 0
    analyzer = tvm.arith.Analyzer()

    def validate(node, order, scopes):
        op = _op_name(node)
        _require(any(statement.value.same_as(node) for statement in statements),
                 "tmem", "allocation and free must be standalone statements")
        _require(len(node.args) == 2 and isinstance(node.args[0], tir.Call)
                 and _op_name(node.args[0]) == "tirx.address_of"
                 and isinstance(node.args[0].args[0], tir.BufferLoad),
                 "tmem", "expected a scalar TMEM base-cell address and allocation width")
        load = node.args[0].args[0]
        cell = owned.get(load.buffer.data.name)
        _require(cell is not None and cell.scope() == "shared" and cell.dtype == "uint32"
                 and list(map(int, cell.shape)) == [1]
                 and len(load.indices) == 1 and analyzer.can_prove(load.indices[0] == 0),
                 "tmem", "TMEM allocation requires an owned uint32 base cell")
        width = node.args[1]
        _require(isinstance(width, tir.IntImm) and 32 <= int(width) <= 512
                 and int(width) & (int(width) - 1) == 0,
                 "tmem", "TMEM width must be a supported constant power of two")
        _require(threads >= 32 and threads % 32 == 0 and len(scopes) == 1
                 and isinstance(scopes[0], tir.IfThenElse) and scopes[0].else_case is None,
                 "tmem", "allocation and free must execute once in the first warp")
        guard = scopes[0].condition
        for participant in range(threads):
            selected = analyzer.simplify(tir.stmt_functor.substitute(
                guard, {thread.node.var: tir.const(participant, thread.node.var.dtype)}))
            _require(isinstance(selected, tir.IntImm) and bool(int(selected)) == (participant < 32),
                     "tmem", "allocation and free require the captured first-warp cohort")
        sites.setdefault(cell.name, {}).setdefault(op, []).append((node, order, guard))
        return None

    def enter(node):
        nonlocal order
        if isinstance(node, (tir.For, tir.While, tir.IfThenElse)):
            scopes.append(node)
        if isinstance(node, tir.Call):
            order += 1
            op = _op_name(node)
            if op == "tirx.call_extern" and node.args[0].value == _MARKER:
                markers.append(order)
            if op in ("tl.ptx_init_tensor_memory", "tl.ptx_deallocate_tensor_memory"):
                captured.append((node, order, tuple(scopes)))
        return None

    def leave(node):
        if isinstance(node, (tir.For, tir.While, tir.IfThenElse)):
            scopes.pop()
        return None

    tir.stmt_functor.ir_transform(func.body, enter, leave,
                                 ["tirx.For", "tirx.While", "tirx.IfThenElse", "tirx.Call"])
    for site in captured:
        validate(*site)
    _require(set(owned) == set(barriers) | set(sites), "storage", "unsupported internal storage")
    tmem = []
    for name, operations in sites.items():
        allocate = operations.get("tl.ptx_init_tensor_memory", [])
        deallocate = operations.get("tl.ptx_deallocate_tensor_memory", [])
        _require(len(allocate) == len(deallocate) == 1, "tmem",
                 "each owned TMEM cell needs exactly one allocation and one free")
        alloc, begin, guard = allocate[0]
        free, end, _ = deallocate[0]
        _require(int(alloc.args[1]) == int(free.args[1]), "tmem", "allocation and free widths differ")
        _require(len(markers) == 1 and begin < markers[0] < end, "tmem",
                 "allocation and free must enclose the completed-fragment export")
        tmem.append(_OwnedTmem(owned[name], int(alloc.args[1]), free, guard))
    resources = [{"handle": record.handle, "domain": "smem", "extent": record.depth * 8, "align": 8}
                 for record in barriers.values()]
    for item in tmem:
        handle = prefix + item.buffer.name
        resources.extend(({"handle": handle, "domain": "smem", "extent": 4, "align": 4},
                          {"handle": handle + "_columns", "domain": "tmem",
                           "extent": item.columns, "align": item.columns}))
    return owned, barriers, tmem, resources, thread


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
    owned, barriers, tmem, resources, thread = _owned_storage(
        func, stage.internal_storage, "pipelink_" + symbol_prefix(stage),
        generated["facts"]["captures"]["pre_codegen"]["attrs"]["thread_extent"]["threadIdx.x"])
    facts = generated["facts"]["captures"]["pre_codegen"]
    threads = facts["attrs"]["thread_extent"]["threadIdx.x"]
    _require(not facts["warp_scope"] and all(facts["attrs"]["thread_extent"].get(axis, 1) == 1
                 for axis in ("threadIdx.y", "threadIdx.z")), "role", "expected a single one-dimensional thread cohort")
    prefix = "pipelink_" + symbol_prefix(stage)
    frame = tir.Var(prefix + "frame", tvm.ir.PointerType(tvm.ir.PrimType(fragment.dtype), "global"))
    destination = tir.decl_buffer(fragment.shape, fragment.dtype, name=prefix + "frame_buffer", data=frame)
    markers, syncs, storage_binds = 0, 0, 0

    def address(load):
        record = barriers.get(load.buffer.data.name)
        if record is None:
            return None
        return tir.call_extern("handle", prefix + "barrier_slot", arena.data,
                               tir.call_extern("int32", record.handle), load.indices[0])

    def bind_owned(node):
        if isinstance(node, tir.Call):
            if _op_name(node) == "tl.ptx_deallocate_tensor_memory":
                return tir.const(0)
            if _op_name(node) == "tl.ptx_init_tensor_memory":
                return tir.call_extern("int32", prefix + "allocate", *node.args)
            value, count = _barrier_operands(node, address, pointer=prefix + "barrier_address")
            return value if count else None
        if isinstance(node, (tir.AllocBuffer, tir.DeclBuffer)) and node.buffer.data.name in owned:
            if isinstance(node, tir.DeclBuffer) or node.buffer.data.name in barriers:
                return tir.Evaluate(0)
            return _bind_smem(node.buffer.data, arena, prefix + node.buffer.data.name)
        return None

    prepared = tir.stmt_functor.ir_transform(func.body, None, bind_owned,
                                            ["tirx.Call", "tirx.AllocBuffer", "tirx.DeclBuffer"])

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
            if (_op_name(call) == "tirx.handle_add_byte_offset" and call.args[0].same_as(arena.data)
                    and node.var.name not in owned):
                storage_binds += 1
                return tir.Bind(node.var, tir.Call(call.dtype, call.op,
                    [call.args[0], call.args[1] + tir.call_extern("int32", prefix + "smem")]))
        return None

    body = tir.stmt_functor.ir_transform(prepared, None, replace, ["tirx.Evaluate", "tirx.Bind"])
    _require(markers == 1, "fragment", "expected exactly one completed-fragment marker")
    _require(storage_binds > 0, "storage", "merged arena has no operand bindings")
    attrs = dict(func.attrs.items())
    name = prefix + "body"
    attrs.update({"global_symbol": name, "tl.device_function": True})
    lowered = tir.PrimFunc([*func.params, frame], body, ret_type=func.ret_type,
                          buffer_map=func.buffer_map).with_attrs(attrs)
    functions = {name: lowered}
    release_call = ""
    if tmem:
        _require(syncs > 0, "tmem", "owned TMEM requires a captured CTA rendezvous")
        release_name = prefix + "release_body"
        cleanup = [tir.AllocBuffer(arena)]
        for item in tmem:
            cleanup.extend((_bind_smem(item.buffer.data, arena, prefix + item.buffer.name),
                            tir.IfThenElse(item.guard, tir.Evaluate(item.deallocate), None)))
        cleanup.append(tir.Evaluate(tir.call_intrin(
            "int32", "tirx.tvm_storage_sync", "shared",
            tir.call_extern("int32", prefix + "barrier"), tir.const(threads))))
        release_body = tir.AttrStmt(thread.node, thread.attr_key, thread.value, tir.SeqStmt(cleanup))
        functions[release_name] = tir.PrimFunc([], release_body).with_attrs({
            "global_symbol": release_name, "tl.device_function": True, "calling_conv": 2,
            "target": target, "thread_extent": {"threadIdx.x": threads}})
        release_call = release_name + "();"
    preamble, source = _codegen(functions, target)
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
    for resource in resources:
        if resource["domain"] == "smem":
            declarations += f'extern "C" __device__ int {resource["handle"]}();\n'
    if barriers:
        declarations += f'''__device__ __forceinline__ Barrier& {prefix}barrier_slot(uchar* smem, int base, int slot) {{
  return *reinterpret_cast<Barrier*>(smem + base + slot * 8);
}}
__device__ __forceinline__ Barrier* {prefix}barrier_address(Barrier& barrier) {{ return &barrier; }}
'''
    if tmem:
        preamble += '#include <cute/arch/tmem_allocator_sm100.hpp>\n'
        declarations += f'''__device__ __forceinline__ int {prefix}allocate(void* cell, int columns) {{
  cute::TMEM::Allocator1Sm allocator;
  allocator.allocate(columns, static_cast<uint32_t*>(cell));
  allocator.release_allocation_lock();
  return 0;
}}
'''
    wrapper = f'''extern "C" __device__ pipelink_tensor_source {stage.produce_symbol}(const {stage.args.name}* args, void* frame) {{
  {name}({", ".join(arguments)});
  return pl::source_of(frame);
}}
extern "C" __device__ void {stage.release_symbol}(const {stage.args.name}*) {{ {release_call} }}
'''
    measured = {"threads": threads, "frame_words": int(fragment.shape[0]),
                "smem_bytes": prod(int(x) for x in arena.shape) * tvm.DataType(arena.dtype).bits // 8,
                "smem_align": _SHARED_ALIGNMENT, "named_barrier": bool(syncs),
                "internal_resources": resources}
    return preamble, declarations + source + wrapper, measured, tvm.IRModule(functions)


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
    # Operand storage expires at ready; owned special storage survives through release.
    storage = ""
    claims = (f'PL_CLAIM("{prefix}smem", "smem", {measured["smem_bytes"]}, {measured["smem_align"]}, '
              f'pl::during{{{q(stage.name)}, {q(stage.work)}, {q(stage.ready)}}});\n')
    for resource in measured["internal_resources"]:
        claims += (f'PL_CLAIM("{resource["handle"]}", "{resource["domain"]}", '
                   f'{resource["extent"]}, {resource["align"]}, "kernel");\n')
    if measured["named_barrier"]:
        claims += f'PL_CLAIM("{prefix}barrier", "named_barrier", 1, 1, "kernel");\n'
        storage = f'''{{"producer_rendezvous", "barrier", 1, 1, "cta", "relocatable",
        "{prefix}barrier", std::nullopt, {q(stage.owner)}, {q(stage.work)}, {q(stage.done)}, pl::sync{{2, 1}}}}'''
    return f'''PL_STAGE({q(stage.stage_id)}, {q(stage.version)});
PL_NO_PARAMS();
PL_TYPE({stage.args.name}, {", ".join(field.name for field in stage.args.fields)});
PL_CALL({stage.produce}, {stage.produce_symbol}, pl::consume, "produce");
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
