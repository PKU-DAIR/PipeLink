"""Structural role selection and contract-driven TileLang CUDA externalization."""
from dataclasses import dataclass
import hashlib

from tilelang import tvm
from tilelang.cuda.codegen import build_cuda_without_compile

from .annotate import AnnotationError
from .external import ExternalBarrier, ExternalStorage, ExternalThreadBase


tir = tvm.tirx


@dataclass(frozen=True)
class RoleFunction:
    depth: int
    name: str
    params: tuple[str, ...]


@dataclass(frozen=True)
class Split:
    preamble: str
    body: str
    functions: tuple[RoleFunction, ...]
    record: dict


def _require(condition, field, detail):
    if not condition:
        raise AnnotationError(field, detail)


def _nodes(body, kind):
    nodes = []
    tir.stmt_functor.post_order_visit(body, lambda node: nodes.append(node) if isinstance(node, kind) else None)
    return nodes


def _variables(body):
    # tirx post_order_visit skips buffer fields, including their data Vars.
    return [node if isinstance(node, tir.Var) else node.buffer.data
            for node in _nodes(body, (tir.Var, tir.BufferLoad, tir.BufferStore))]


def _uses(var, variables):
    return any(var.same_as(other) for other in variables)


def _op_name(call):
    return call.op.name if isinstance(call.op, tvm.ir.Op) else str(call.op)


def _arm(node, branch):
    body = node.body
    _require(isinstance(body, tir.IfThenElse) and body.else_case is not None,
             "split.branch", "warp-specialization scope must contain both IfThenElse arms")
    return body.then_case if branch == "then" else body.else_case


def _barrier_operands(call, address, *, pointer=None):
    """Rewrite barrier loads even when nested in an intrinsic's address operand."""
    rewritten = 0

    def address_of(node):
        nonlocal rewritten
        if isinstance(node, tir.Call) and _op_name(node) == "tirx.address_of" \
                and isinstance(node.args[0], tir.BufferLoad):
            result = address(node.args[0])
            if result is not None:
                _require(pointer is not None, "split.barrier", "pointer operand needs a barrier-address helper")
                rewritten += 1
                return tir.call_extern("handle", pointer, result)
        return None

    def replace(node):
        nonlocal rewritten
        result = address(node) if isinstance(node, tir.BufferLoad) else None
        if result is not None:
            rewritten += 1
        return result

    value = tir.stmt_functor.ir_transform(tir.Evaluate(call), address_of, replace, ["tirx.Call", "tirx.BufferLoad"]).value
    return value, rewritten


def _storage_bind(node, handles, offsets, dyn=None):
    if node.var.name not in handles:
        return None
    value = node.value
    _require(isinstance(value, tir.Call) and _op_name(value) == "tirx.handle_add_byte_offset"
             and isinstance(value.args[1], tir.IntImm), "split.smem", node.var.name)
    offsets[node.var.name] += 1
    return tir.Bind(node.var, tir.Call(value.dtype, value.op,
                    [value.args[0] if dyn is None else dyn.data,
                     tir.call_extern("int32", handles[node.var.name])]))


def _prune(body):
    used, locals_, guards = _variables(body), [], 0
    nop = tir.Evaluate(tir.const(0, "int32"))

    def noop(stmt):
        if isinstance(stmt, tir.Evaluate):
            return isinstance(stmt.value, tir.IntImm)
        return isinstance(stmt, tir.SeqStmt) and all(noop(item) for item in stmt.seq)

    def sweep(node):
        nonlocal guards
        if isinstance(node, (tir.AllocBuffer, tir.DeclBuffer)) and node.buffer.scope().startswith("local") \
                and not _uses(node.buffer.data, used):
            locals_.append(node.buffer.name)
            return nop
        if isinstance(node, tir.IfThenElse) and node.else_case is None and noop(node.then_case):
            guards += 1
            return nop
        return None

    body = tir.stmt_functor.ir_transform(body, None, sweep, ["tirx.AllocBuffer", "tirx.DeclBuffer", "tirx.IfThenElse"])
    return body, locals_, guards


def _role_function(func, body, name, declared_params):
    replacements, params = [], []
    for param in func.params:
        annotation = param.type_annotation
        dtype = annotation.element_type.dtype if isinstance(annotation, tvm.ir.PointerType) else param.dtype
        if dtype == "uint8x128":
            pointer = tir.Var(param.name, tvm.ir.PointerType(tvm.ir.PrimType("uint8x128"), "global"))
            buffer = tir.decl_buffer((1,), "uint8x128", name=param.name + "_buf", data=pointer, scope="global")
            replacements.append((param, pointer, tir.BufferLoad(buffer, [0])))
            params.append(pointer)
        else:
            params.append(param)

    def descriptors(node):
        args = [next((load for old, _, load in replacements if arg.same_as(old)), arg)
                if isinstance(arg, tir.Var) else arg for arg in node.args]
        return tir.Call(node.dtype, node.op, args) if any(not a.same_as(b) for a, b in zip(args, node.args)) else None

    body = tir.stmt_functor.ir_transform(body, None, descriptors, ["tirx.Call"])
    variables = _variables(body)
    pruned = [param.name for param in params if not _uses(param, variables)]
    params = [param for param in params if _uses(param, variables)]
    names = tuple(param.name for param in params)
    _require(set(names) == set(declared_params), "split.params", f"{names} != {declared_params}")
    descriptor_indices = [i for i, param in enumerate(params) if _uses(param, [ptr for _, ptr, _ in replacements])]
    attrs = dict(func.attrs.items())
    attrs.update({"global_symbol": name, "tl.readonly_param_indices": descriptor_indices,
                  "tl.device_function": True})
    buffer_map = {param: buffer for param, buffer in func.buffer_map.items() if _uses(param, params)}
    role = tir.PrimFunc(params, body, ret_type=func.ret_type, buffer_map=buffer_map).with_attrs(attrs)
    role = tir.transform.Simplify()(tvm.IRModule({name: role}))[name]
    return role, names, pruned, [names[i] for i in descriptor_indices]


def _codegen(roles, target):
    source = build_cuda_without_compile(tvm.IRModule(roles), target).inspect_source()
    _require("__global__" not in source and "__launch_bounds__" not in source, "split.prefix",
             "codegen emitted a kernel entry; tl.device_function needs the patched TileLang build")
    debug_include = "#include <tl_templates/cuda/debug.h>\n"
    _require(source.count(debug_include) == 1, "split.preamble", "expected one debug.h include")
    source = source.replace(debug_include, "", 1)
    preamble, marker, body = source.partition('extern "C" __device__')
    body = marker + body
    _require("debug_print" not in body and "PrintTraits" not in body, "split.preamble", "debug.h is used by a role")
    return preamble, body


def _slot_helper(prefix):
    return f"__device__ __forceinline__ Barrier& {prefix}slot(uchar* smem, int base, int slot) {{ return *reinterpret_cast<Barrier*>(smem + base * 16 + slot * 16); }}\n"


def split_roles(modules: dict[int, "tvm.IRModule"], *, branch: str, depth_of: None = None,
                prefix: str, smem_handles: dict[str, str], full: str, empty: str,
                declared_params: tuple[str, ...], target) -> Split:
    roles, functions, depths = {}, [], {}
    dropped_ops = {"tl.set_max_nreg", "tirx.ptx_init_barrier_thread_count",
                   "tl.ptx_fence_barrier_init", "tirx.tvm_storage_sync"}
    nop = tir.Evaluate(tir.const(0, "int32"))
    for key in sorted(modules):
        func = next(iter(modules[key].functions.values()))
        before = hashlib.sha256(tvm.ir.save_json(func).encode()).hexdigest()
        nodes = _nodes(func.body, (tir.DeclBuffer, tir.AllocBuffer, tir.AttrStmt))
        bar = next(node.buffer for node in nodes if isinstance(node, tir.DeclBuffer) and node.buffer.scope() == "shared.barrier")
        dyn = next(node.buffer for node in nodes if isinstance(node, tir.AllocBuffer) and node.buffer.scope() == "shared.dyn")
        depth = int(bar.shape[0]) // 2
        scope = next(node for node in nodes if isinstance(node, tir.AttrStmt) and node.attr_key == "kWarpSpecializationScope")
        arm_vars = _variables(_arm(scope, branch))
        analyzer = tvm.arith.Analyzer()
        dropped, offsets, rewritten = {}, dict.fromkeys(smem_handles, 0), 0
        prepare_nodes = ["tirx.AttrStmt", "tirx.Evaluate", "tirx.AllocBuffer", "tirx.DeclBuffer"]

        def prepare(node):
            if isinstance(node, tir.AttrStmt) and node.attr_key == "kWarpSpecializationScope":
                return tir.stmt_functor.ir_transform(_arm(node, branch), prepare, None, prepare_nodes)
            if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
                op = _op_name(node.value)
                unused_prefetch = op == "tl.prefetch_tma_descriptor" and not _uses(node.value.args[0], arm_vars)
                if op in dropped_ops or unused_prefetch:
                    dropped[op] = dropped.get(op, 0) + 1
                    return nop
            if isinstance(node, (tir.AllocBuffer, tir.DeclBuffer)) and node.buffer.scope() == "shared.barrier":
                return nop
            return None

        body = tir.stmt_functor.ir_transform(func.body, prepare, None, prepare_nodes)
        body, pruned_locals, dropped_guards = _prune(body)

        def address(load):
            if not load.buffer.same_as(bar):
                return None
            idx = load.indices[0]
            if analyzer.can_prove(idx >= depth):
                handle, slot = empty, analyzer.simplify(idx - depth)
            else:
                _require(analyzer.can_prove(idx < depth), "split.barrier", str(idx))
                handle, slot = full, idx
            return tir.call_extern("handle", prefix + "slot", dyn.data,
                                   tir.call_extern("int32", handle), slot)

        def post(node):
            nonlocal rewritten
            if isinstance(node, tir.Call):
                value, count = _barrier_operands(node, address)
                rewritten += count
                return value if count else None
            if isinstance(node, tir.Bind):
                return _storage_bind(node, smem_handles, offsets)
            return None

        body = tir.stmt_functor.ir_transform(body, None, post, ["tirx.Call", "tirx.Bind"])
        _require(all(count == 1 for count in offsets.values()), "split.smem", str(offsets))
        name = f"{prefix}run_{depth}"
        roles[name], names, pruned, descriptor_names = _role_function(func, body, name, declared_params)
        functions.append(RoleFunction(depth, name, names))
        depths[str(depth)] = {"function": name, "params": list(names), "dropped": dropped,
                              "barrier_operands_rewritten": rewritten, "descriptor_params": descriptor_names,
                              "pruned_params": pruned, "pruned_locals": pruned_locals, "dropped_guards": dropped_guards,
                              "simplified": True, "tir_sha256_before": before,
                              "tir_sha256_after": hashlib.sha256(tvm.ir.save_json(roles[name]).encode()).hexdigest()}

    preamble, body = _codegen(roles, target)
    handles = "".join(f'extern "C" __device__ int {handle}();\n' for handle in (*smem_handles.values(), full, empty))
    helper = _slot_helper(prefix)
    record = {"kind": "structural_tir_ws_role_split", "device_function_attr": "tl.device_function",
              "preamble_changes": ["remove unused tl_templates/cuda/debug.h"], "depths": depths}
    return Split(preamble, handles + helper + body, tuple(functions), record)


def _thread_interval(condition, thread, lo, hi):
    if isinstance(condition, tir.And):
        left = _thread_interval(condition.a, thread, lo, hi)
        return None if left is None else _thread_interval(condition.b, thread, *left)
    if not isinstance(condition, (tir.LT, tir.LE, tir.GT, tir.GE)):
        return None
    a, b = condition.a, condition.b
    if isinstance(a, tir.Var) and a.same_as(thread) and isinstance(b, tir.IntImm):
        bound = int(b)
        if isinstance(condition, (tir.LT, tir.LE)):
            return lo, min(hi, bound + isinstance(condition, tir.LE))
        return max(lo, bound + isinstance(condition, tir.GT)), hi
    if isinstance(b, tir.Var) and b.same_as(thread) and isinstance(a, tir.IntImm):
        reverse = {tir.LT: tir.GT, tir.LE: tir.GE, tir.GT: tir.LT, tir.GE: tir.LE}
        return _thread_interval(reverse[type(condition)](b, a), thread, lo, hi)
    return None


def _select_role(body, branch, warp_range):
    if branch is not None:
        _require(branch in ("then", "else") and warp_range is None, "split.branch",
                 "select either a WS arm or a warp range")
        scopes = [node for node in _nodes(body, tir.AttrStmt) if node.attr_key == "kWarpSpecializationScope"]
        _require(len(scopes) == 1, "split.branch", "expected one warp-specialization scope")
        return scopes[0], _arm(scopes[0], branch)
    _require(warp_range is not None and len(warp_range) == 2 and 0 <= warp_range[0] < warp_range[1],
             "split.warp_range", "expected a nonempty half-open warp interval")
    threads = [node for node in _nodes(body, tir.AttrStmt)
               if node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x"]
    _require(len(threads) == 1, "split.warp_range", "expected one threadIdx.x extent")
    thread, extent = threads[0].node.var, int(threads[0].value)
    wanted = tuple(warp * 32 for warp in warp_range)
    if wanted == (0, extent):
        return body, body
    matches = []

    def visit(stmt):
        if isinstance(stmt, tir.AttrStmt):
            visit(stmt.body)
        elif isinstance(stmt, tir.SeqStmt):
            for item in stmt.seq:
                visit(item)
        elif isinstance(stmt, tir.IfThenElse):
            root, lo, hi = stmt, 0, extent
            while isinstance(stmt, tir.IfThenElse):
                if not stmt.same_as(root) and (lo, hi) == wanted:
                    matches.append((root, stmt))
                    return
                interval = _thread_interval(stmt.condition, thread, lo, hi)
                if interval is None:
                    return
                if interval == wanted:
                    matches.append((root, stmt.then_case))
                    return
                if interval[0] == lo:
                    lo = interval[1]
                elif interval[1] == hi:
                    hi = interval[0]
                else:
                    return
                stmt = stmt.else_case
            if stmt is not None and (lo, hi) == wanted:
                matches.append((root, stmt))

    visit(body)
    _require(len(matches) == 1, "split.warp_range", f"expected one branch for warps {warp_range}, found {len(matches)}")
    return matches[0]


def externalize_role(module, *, role: str, warp_range: tuple[int, int] | None,
                     branch: str | None,
                     external: list[ExternalBarrier | ExternalStorage | ExternalThreadBase],
                     prefix: str, declared_params: tuple[str, ...], target) -> Split:
    """Extract one lowered role and bind its borrowed resources to solved handles.

    ``module`` is the final pre-codegen TIR capture. Select exactly one of a WS
    ``branch`` or ``warp_range``; the latter is a half-open interval in warps.
    TMEM storage names the lowered arena word and the partner's published smem
    scalar, with an optional separately solved TMEM column offset.
    A role compiled at warp ``lo`` and linked at solved warp ``base()`` keeps
    its compiled thread coordinates: ``tx -> tx - base()*32 + lo*32``.

    Whole-CTA syncs and TileLang's reserved id-3 partial syncs are rebound;
    explicit partner named barriers retain their id/count. This pin allocates
    partial ids starting at kFirstUsedBarrier (thread_storage_sync.cc:349-351);
    id 3 is the generated first role-local synchronization domain.
    """
    funcs = list(module.functions.values())
    _require(len(funcs) == 1 and isinstance(funcs[0], tir.PrimFunc), "split.module", "expected one PrimFunc")
    func = funcs[0]
    root, selected = _select_role(func.body, branch, warp_range)
    all_nodes = _nodes(func.body, (tir.AllocBuffer, tir.DeclBuffer, tir.AttrStmt))
    thread_scope = next(node for node in all_nodes
                        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent"
                        and node.node.thread_tag == "threadIdx.x")
    thread, extent = thread_scope.node.var, int(thread_scope.value)
    if branch is not None:
        interval = _thread_interval(root.body.condition, thread, 0, extent)
        _require(interval is not None, "split.branch", "WS arm must have a thread-range predicate")
        selected_interval = interval if branch == "then" else ((interval[1], extent) if interval[0] == 0 else (0, interval[0]))
    else:
        selected_interval = tuple(warp * 32 for warp in warp_range)
    barriers = {item.buffer: item for item in external if isinstance(item, ExternalBarrier)}
    storage = {item.buffer: item for item in external if isinstance(item, ExternalStorage)}
    bases = [item for item in external if isinstance(item, ExternalThreadBase)]
    _require(len(bases) <= 1, "split.thread_base", "expected at most one thread-base handle")
    thread_base = bases[0] if bases else None
    warps = (selected_interval[1] - selected_interval[0]) // 32
    _require(warps > 0, "split.thread_base", "role must contain at least one warp")
    if thread_base:
        _require(thread_base.warps == warps, "split.thread_base", "role width differs from selected branch")
    nop = tir.Evaluate(tir.const(0, "int32"))
    dropped, rewritten = {}, 0
    offsets = dict.fromkeys(storage, 0)
    arm_vars = _variables(selected)
    dynamic = [node.buffer for node in all_nodes if isinstance(node, tir.AllocBuffer)
               and node.buffer.scope() == "shared.dyn"]
    _require(len(dynamic) <= 1, "split.smem", "shared allocations must be merged before externalization")
    reuse_dyn = bool(dynamic and dynamic[0].dtype == "uint8" and dynamic[0].name not in storage)
    dyn = dynamic[0] if reuse_dyn else tir.decl_buffer((1,), "uint8", name=prefix + "smem", scope="shared.dyn")

    def barrier_for(load):
        return barriers.get(load.buffer.name) or barriers.get(load.buffer.data.name)

    init_calls = [call for call in _nodes(func.body, tir.Call) if _op_name(call) == "tirx.ptx_init_barrier_thread_count"]
    keep_init = any(barrier_for(call.args[0]) is None or barrier_for(call.args[0]).init == "keep"
                    for call in init_calls)
    initializers = []
    initialized = {}
    physical_thread = (thread - selected_interval[0] + tir.call_extern("int32", thread_base.handle) * 32
                       if thread_base else thread)
    for call in init_calls:
        record = barrier_for(call.args[0])
        if record is None or record.init != "keep":
            continue
        key = (record.handle, record.native, record.side)
        if key in initialized:
            prior = initialized[key]
            _require((prior.count, prior.depth, prior.init_warp) == (record.count, record.depth, record.init_warp),
                     "split.barrier.init", "aliased barrier initialization contracts differ")
            continue
        _require(isinstance(record.count, int) and record.count > 0,
                 "split.barrier.count", f"{record.buffer}: unresolved count {record.count!r}")
        initialized[key] = record
        predicate = physical_thread == record.init_warp * 32 if record.init_warp is not None else thread == selected_interval[0]
        loads = [tir.BufferLoad(call.args[0].buffer, [slot]) for slot in range(record.depth)]
        stores = [tir.Evaluate(tir.Call(call.dtype, call.op, [load, tir.const(record.count)])) for load in loads]
        initializers.append(tir.IfThenElse(predicate, stores[0] if len(stores) == 1 else tir.SeqStmt(stores), None))
    fence_kept = False

    def drop(op):
        dropped[op] = dropped.get(op, 0) + 1
        return nop

    def named_sync(call):
        return tir.Call(call.dtype, call.op, [call.args[0],
                        tir.call_extern("int32", prefix + "named_barrier_id"), tir.const(warps * 32)])

    def prepare_body(body, inside):
        def prepare(node):
            nonlocal fence_kept
            if not inside and node.same_as(root):
                return prepare_body(selected, True)
            if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
                call, op = node.value, _op_name(node.value)
                if op == "tl.set_max_nreg" or (op == "tl.prefetch_tma_descriptor" and not _uses(call.args[0], arm_vars)):
                    return drop(op)
                if op == "tirx.ptx_init_barrier_thread_count":
                    record = barrier_for(call.args[0])
                    if record:
                        if record.init == "drop":
                            return drop(op)
                        return nop
                if op in ("tl.ptx_init_tensor_memory", "tl.ptx_deallocate_tensor_memory"):
                    if any(var.name in storage and storage[var.name].space == "tmem" for var in _variables(node)):
                        return drop(op)
                if op == "tl.ptx_fence_barrier_init":
                    if not keep_init or fence_kept:
                        return drop(op)
                    fence_kept = True
                    if initializers:
                        return tir.SeqStmt([*initializers, node])
                if op == "tirx.tvm_storage_sync" and call.args[0].value.startswith("shared"):
                    if len(call.args) > 1 and not (isinstance(call.args[1], tir.IntImm) and int(call.args[1]) == 3):
                        return None
                    if not inside and not keep_init:
                        return drop(op)
                    return tir.Evaluate(named_sync(call))
            if isinstance(node, (tir.AllocBuffer, tir.DeclBuffer)) and node.buffer.scope() in ("shared.barrier", "shared.cluster_barrier"):
                if node.buffer.name in barriers or node.buffer.data.name in barriers:
                    return nop
            return None

        return tir.stmt_functor.ir_transform(body, prepare, None,
                                            ["tirx.AttrStmt", "tirx.IfThenElse", "tirx.Evaluate",
                                             "tirx.AllocBuffer", "tirx.DeclBuffer"])

    body = prepare_body(func.body, False)

    def address(load):
        record = barrier_for(load)
        if record is None:
            return None
        _require(len(load.indices) == 1, "split.barrier", "expected a flattened barrier array")
        slot = load.indices[0]
        if record.native is None:
            return tir.call_extern("handle", prefix + "slot", dyn.data,
                                   tir.call_extern("int32", record.handle), slot)
        native = record.native
        _require(native.unit_bytes in (1, 16), "split.barrier.native", "base unit must be 1 or 16 bytes")
        return tir.call_extern("handle", prefix + "native", dyn.data,
                               tir.call_extern("int32", native.base), native.unit_bytes,
                               native.offset_bytes, native.stride_bytes, slot)

    handles = {name: item.handle for name, item in storage.items()}

    def bind_storage(var, record):
        offsets[record.buffer] += 1
        pointer = tir.call_intrin("handle", "tirx.handle_add_byte_offset", dyn.data,
                                 tir.call_extern("int32", record.handle))
        return tir.Bind(var, pointer)

    columns = {name: item.column_handle for name, item in storage.items()
               if item.space == "tmem" and item.column_handle is not None}

    def post(node):
        nonlocal rewritten
        if isinstance(node, tir.Call):
            if _op_name(node) == "tl.tl_shuffle_elect" and int(node.args[0]) != 32:
                local = thread - selected_interval[0]
                group = int(node.args[0])
                first_warp = local < 32 if group == 0 else tir.truncmod(local, group) < 32
                return tir.And(first_warp, tir.Call(node.dtype, node.op, [tir.const(32)]))
            value, count = _barrier_operands(node, address, pointer=prefix + "address")
            rewritten += count
            args = list(value.args)
            # LowerSharedTmem leaves MMA operands as (arena pointer, offset).
            for index, arg in enumerate(args):
                if isinstance(arg, tir.Var) and arg.name in columns:
                    _require(index + 1 < len(args) and args[index + 1].dtype.startswith(("int", "uint")),
                             "split.tmem", f"{_op_name(node)} has no column offset after {arg.name}")
                    args[index + 1] = args[index + 1] + tir.Cast(args[index + 1].dtype,
                                                               tir.call_extern("int32", columns[arg.name]))
                    count += 1
            return tir.Call(value.dtype, value.op, args) if count else None
        if isinstance(node, tir.BufferLoad) and node.buffer.data.name in columns:
            # Load at the original use, after the caller's publish handshake;
            # eagerly copying the word at allocation time can read it too soon.
            return node + tir.Cast(node.dtype, tir.call_extern("int32", columns[node.buffer.data.name]))
        if isinstance(node, tir.Bind) and node.var.name in storage:
            record = storage[node.var.name]
            if record.space == "tmem":
                return bind_storage(node.var, record)
            return _storage_bind(node, handles, offsets, dyn)
        if isinstance(node, tir.AllocBuffer):
            buffer = node.buffer
            record = storage.get(buffer.name) or storage.get(buffer.data.name)
            if record:
                return bind_storage(buffer.data, record)
            if buffer.scope() == "shared.dyn" and not reuse_dyn:
                return tir.Bind(buffer.data, tir.call_intrin("handle", "tirx.handle_add_byte_offset", dyn.data, 0))
        return None

    body = tir.stmt_functor.ir_transform(body, None, post, ["tirx.Call", "tirx.Bind", "tirx.AllocBuffer", "tirx.BufferLoad"])
    _require(all(count == 1 for count in offsets.values()), "split.storage", str(offsets))
    if thread_base:
        body = tir.stmt_functor.substitute(body, {thread: thread - tir.call_extern("int32", thread_base.handle) * 32 + selected_interval[0]})
    body, pruned_locals, dropped_guards = _prune(body)
    if not reuse_dyn:
        body = tir.SeqStmt([tir.AllocBuffer(dyn), body])
    name = prefix + "run_" + role
    lowered, names, pruned, descriptors = _role_function(func, body, name, declared_params)
    preamble, source = _codegen({name: lowered}, target)
    handle_names = []
    for item in external:
        handle_names.append(item.handle)
        if isinstance(item, ExternalBarrier) and item.native is not None:
            handle_names.append(item.native.base)
        if isinstance(item, ExternalStorage) and item.column_handle is not None:
            handle_names.append(item.column_handle)
    handle_names.append("pipelink_" + prefix + "named_barrier")
    declarations = "".join(f'extern "C" __device__ int {handle}();\n' for handle in dict.fromkeys(handle_names))
    helpers = _slot_helper(prefix) if any(item.native is None for item in barriers.values()) else ""
    if any(item.native is not None for item in barriers.values()):
        helpers += f"__device__ __forceinline__ Barrier& {prefix}native(uchar* smem, int base, int unit, int offset, int stride, int slot) {{ return *reinterpret_cast<Barrier*>(smem + base * unit + offset + slot * stride); }}\n"
    if barriers:
        helpers += f"__device__ __forceinline__ Barrier* {prefix}address(Barrier& barrier) {{ return &barrier; }}\n"
    helpers += f"__device__ __forceinline__ int {prefix}named_barrier_id() {{ return pipelink_{prefix}named_barrier(); }}\n"
    record = {"kind": "structural_tir_role_externalization", "role": role, "branch": branch,
              "warp_range": warp_range, "function": name, "params": list(names),
              "barrier_operands_rewritten": rewritten, "storage_rebound": offsets,
              "initializers": [{"handle": item.handle, "side": item.side, "count": item.count,
                                "depth": item.depth, "warp": item.init_warp}
                               for item in initialized.values()],
              "dropped": dropped, "descriptor_params": descriptors, "pruned_params": pruned,
              "pruned_locals": pruned_locals, "dropped_guards": dropped_guards,
              "tir_sha256_before": hashlib.sha256(tvm.ir.save_json(func).encode()).hexdigest(),
              "tir_sha256_after": hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest()}
    depth = max((item.depth for item in barriers.values()), default=1)
    return Split(preamble, declarations + helpers + source, (RoleFunction(depth, name, names),), record)
