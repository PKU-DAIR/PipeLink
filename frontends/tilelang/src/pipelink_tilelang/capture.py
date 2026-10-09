"""CPU-only compiler facts from stock TileLang 0.1.14 pass instrumentation.

The JSON contract is the probe's ``passes`` and ``facts`` subtree, unchanged.
Its C++ compiler runs and source-comparison rows are probe evidence, not facts.
Set CUDA_HOME before importing TileLang; this module does not configure the process.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import tilelang
from tilelang import tvm


tir = tvm.tirx
DEFAULT_WINDOWS = MappingProxyType({
    "tl.LayoutInference": "post_layout",
    "tl.MergeSharedMemoryAllocations": "post_shared_merge",
    "tl.AnnotateWarpGroupRegAlloc": "post_register_assignment",
})


class CaptureError(ValueError):
    """A named compiler fact could not be measured or failed its consistency check."""

    def __init__(self, fact: str, detail: str):
        self.fact = fact
        super().__init__(f"{fact}: {detail}")


def _require(condition, fact, detail):
    if not condition:
        raise CaptureError(fact, detail)


@dataclass(frozen=True)
class Capture:
    passes: list[str]
    facts: dict[str, Any]
    snapshots: dict[str, str]
    # Retain the original AST objects for downstream projection/map printing;
    # never reconstruct executable expressions by parsing the JSON strings.
    modules: dict[str, Any]
    layout_objects: dict[str, Any]
    kernel_source: str

    def to_json(self) -> str:
        return json.dumps({"passes": self.passes, "facts": self.facts}, indent=2) + "\n"

    def write(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "facts.json").write_text(self.to_json())
        for window in ("post_layout", "pre_codegen"):
            (directory / f"{window}.tir").write_text(self.snapshots[window])


def _nodes(body, kind):
    found = []
    tir.stmt_functor.post_order_visit(body, lambda node: found.append(node) if isinstance(node, kind) else None)
    return found


def _op_name(call):
    return call.op.name if isinstance(call.op, tvm.ir.Op) else str(call.op)


def _plain(value):
    if isinstance(value, tir.IntImm):
        return int(value)
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or type(value).__name__ == "Array":
        return [_plain(item) for item in value]
    return str(value)


def _integer(expr, values, fact):
    if isinstance(expr, tir.IntImm):
        return int(expr)
    if isinstance(expr, tir.Var):
        _require(expr in values, fact, f"unbound variable: {expr}")
        return values[expr]
    if isinstance(expr, tir.Cast):
        return _integer(expr.value, values, fact)
    if isinstance(expr, (tir.Div, tir.Mod, tir.FloorDiv, tir.FloorMod)):
        a, b = _integer(expr.a, values, fact), _integer(expr.b, values, fact)
        _require(b > 0, fact, "integer evaluator requires a positive divisor")
        if isinstance(expr, (tir.Div, tir.Mod)):
            _require(a >= 0, fact, "truncating division requires a nonnegative domain")
        return a // b if isinstance(expr, (tir.Div, tir.FloorDiv)) else a % b
    for cls, fn in ((tir.Add, lambda a, b: a + b), (tir.Sub, lambda a, b: a - b),
                    (tir.Mul, lambda a, b: a * b), (tir.LT, lambda a, b: a < b)):
        if isinstance(expr, cls):
            return fn(_integer(expr.a, values, fact), _integer(expr.b, values, fact))
    if isinstance(expr, tir.Call):
        functions = {"tirx.bitwise_xor": lambda a, b: a ^ b,
                     "tirx.bitwise_and": lambda a, b: a & b,
                     "tirx.bitwise_or": lambda a, b: a | b,
                     "tirx.shift_left": lambda a, b: a << b,
                     "tirx.shift_right": lambda a, b: a >> b}
        if _op_name(expr) in functions:
            return functions[_op_name(expr)](*(_integer(x, values, fact) for x in expr.args))
    raise CaptureError(fact, f"unmeasured integer expression: {type(expr).__name__}: {expr}")


def _layout_capture(mod):
    records, objects = {}, {}

    def add(name, source, shape, layout):
        if name in records:
            return
        rec = {"source": source, "logical_shape": [int(x) for x in shape],
               "physical_shape": [int(x) for x in layout.get_output_shape()],
               "forward_vars": [str(x) for x in layout.get_forward_vars()],
               "forward_index": [str(x) for x in layout.get_forward_index()]}
        if isinstance(layout, tilelang.layout.Fragment):
            _require(layout.thread_range is not None, f"layouts.{name}.thread_min", "fragment has no measured thread range")
            # thread is role-local. Keep thread_min separate; CTA = thread + min.
            rec.update(thread=str(layout.thread), thread_min=int(layout.thread_range.min),
                       thread_extent=int(layout.thread_range.extent),
                       thread_size=int(layout.get_thread_size()), replication=int(layout.replicate_size))
        records[name] = rec
        objects[name] = layout

    for func in mod.functions.values():
        if not isinstance(func, tir.PrimFunc):
            continue
        for block in _nodes(func.body, tir.SBlock):
            for buffer, layout in block.annotations.get("layout_map", {}).items():
                add(buffer.name, "SBlock.annotations.layout_map", buffer.shape, layout)
        for loop in _nodes(func.body, tir.For):
            layout = loop.annotations.get("parallel_loop_layout")
            if layout is not None:
                add("parallel_" + loop.loop_var.name, "For.annotations.parallel_loop_layout", [loop.extent], layout)
    return records, objects


def _allocated_buffers(body):
    buffers = [node.buffer for node in _nodes(body, tir.AllocBuffer)]
    for block in _nodes(body, tir.SBlock):
        buffers.extend(block.alloc_buffers)
    return buffers


def _barrier_offsets(buffers):
    """Number distinct barrier buffers in allocation order, not annotation-map order."""
    offsets, offset = {}, 0
    for buffer in buffers:
        if buffer.scope() in ("shared.barrier", "shared.cluster_barrier") and buffer.data not in offsets:
            size = math.prod(int(x) for x in buffer.shape)
            offsets[buffer.data] = (offset, size)
            offset += size
    return offsets


def _snapshot(mod):
    result = {"attrs": {}, "allocations": [], "barrier_init": [], "barrier_init_annotation": {},
              "registers": [], "warp_scope": [], "offsets": {}, "calls": [], "bindings": []}
    for func in mod.functions.values():
        if not isinstance(func, tir.PrimFunc):
            continue
        attrs = {str(k): _plain(v) for k, v in (func.attrs.items() if func.attrs else [])}
        result["attrs"].update({k: v for k, v in attrs.items() if k in
                                ("thread_extent", "dyn_shared_memory_buf", "tl.smem_alignment_map")})
        buffers = _allocated_buffers(func.body)
        barrier_offsets = _barrier_offsets(buffers)
        explicit_init = {}
        automatic = any(node.attr_key == "kWarpSpecializationScope" for node in _nodes(func.body, tir.AttrStmt))
        for block in _nodes(func.body, tir.SBlock):
            if not automatic:
                for data, counts in block.annotations.get("barrier_init", {}).items():
                    base, _ = barrier_offsets[data]
                    explicit_init.update((base + i, int(count)) for i, count in enumerate(counts))
            result["barrier_init_annotation"].update(_plain(block.annotations.get("barrier_init", {})))
        for buffer in buffers:
            shape = [int(x) for x in buffer.shape]
            dtype = tvm.DataType(buffer.dtype)
            result["allocations"].append({"name": buffer.name, "scope": buffer.scope(), "shape": shape,
                                          "dtype": str(buffer.dtype), "bytes": math.prod(shape) * dtype.bits * dtype.lanes // 8})
        for node in _nodes(func.body, tir.AttrStmt):
            if node.attr_key == "thread_extent":
                result["attrs"].setdefault("thread_extent", {})[node.node.thread_tag] = _plain(node.value)
            if node.attr_key == "kWarpSpecializationScope":
                result["warp_scope"].append(_plain(node.node))
        for call in _nodes(func.body, tir.Call):
            op = _op_name(call)
            if any(part in op for part in ("barrier", "set_max_nreg", "tma_load")):
                result["calls"].append({"op": op, "args": [_plain(x) for x in call.args]})
            if op.endswith("ptx_init_barrier_thread_count"):
                if automatic:
                    result["barrier_init"].append([int(call.args[0].indices[0]), int(call.args[1])])
                else:
                    base, _ = barrier_offsets[call.args[0].buffer.data]
                    slot, count = base + int(call.args[0].indices[0]), int(call.args[1])
                    _require(slot not in explicit_init or explicit_init[slot] == count,
                             "barrier_init", "annotation and initialization call disagree")
                    explicit_init[slot] = count
            if op.endswith("set_max_nreg"):
                result["registers"].append([int(x) for x in call.args])
        if not automatic:
            result["barrier_init"].extend([slot, count] for slot, count in sorted(explicit_init.items()))
        # Bind is the pinned tirx carrier for flattened shared pointers, not LetStmt.
        for node in _nodes(func.body, tir.Stmt):
            if hasattr(node, "var") and hasattr(node, "value"):
                value = node.value
                if isinstance(value, tir.Call) and _op_name(value).endswith("handle_add_byte_offset"):
                    result["offsets"][node.var.name] = int(value.args[1])
                    result["bindings"].append({"carrier": type(node).__name__, "name": node.var.name,
                                                "base": str(value.args[0]), "offset": int(value.args[1])})
    return result


def _register_branches(mod, final):
    funcs = [func for func in mod.functions.values() if isinstance(func, tir.PrimFunc)]
    _require(len(funcs) == 1, "branches", "expected one device PrimFunc")
    func = funcs[0]
    extents = final["attrs"].get("thread_extent", {})
    threads = [node.node.var for node in _nodes(func.body, tir.AttrStmt)
               if node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x"]
    _require(len(threads) == 1 and "threadIdx.x" in extents, "thread_extent", "expected one threadIdx.x extent")
    thread = threads[0]
    if not final["warp_scope"]:
        return _explicit_branches(func.body, thread, extents["threadIdx.x"])
    candidates = []
    for branch in _nodes(func.body, tir.IfThenElse):
        if branch.else_case is None:
            continue
        arms = [branch.then_case, branch.else_case]
        regs = [[call for call in _nodes(arm, tir.Call) if _op_name(call).endswith("set_max_nreg")] for arm in arms]
        if not all(regs):
            continue
        _require(all(len(reg) == 1 for reg in regs), "branches.set_max_nreg", "ambiguous register assignments")
        records = []
        for is_then, arm, reg in zip((True, False), arms, regs):
            lanes = [t for t in range(extents["threadIdx.x"]) if bool(_integer(branch.condition, {thread: t}, "branches.predicate")) == is_then]
            _require(bool(lanes) and lanes == list(range(lanes[0], lanes[-1] + 1))
                     and lanes[0] % 32 == 0 and len(lanes) % 32 == 0,
                     "branches.warps", "branch must own a contiguous whole-warp interval")
            calls = _nodes(arm, tir.Call)
            records.append({"predicate": str(branch.condition) if is_then else f"not ({branch.condition})",
                            "thread_min": min(lanes), "thread_count": len(lanes), "warp_base": min(lanes) // 32,
                            "warps": len(lanes) // 32, "set_max_nreg": [int(x) for x in reg[0].args],
                            "calls": [{"op": _op_name(c), "args": [_plain(x) for x in c.args]} for c in calls
                                      if "barrier" in _op_name(c) or "tma_load" in _op_name(c)]})
        candidates.append((records, arms))
    _require(len(candidates) <= 1, "branches", "ambiguous warp-specialized branch")
    return candidates[0] if candidates else ([], [])


def _explicit_branches(body, thread, extent):
    """Capture outer thread-range arms; lane elections inside a role stay inside it."""
    analyzer = tvm.arith.Analyzer()
    groups = {}

    def is_range(condition):
        if isinstance(condition, tir.And):
            return is_range(condition.a) and is_range(condition.b)
        if not isinstance(condition, (tir.LT, tir.LE, tir.GT, tir.GE)):
            return False
        return ((condition.a.same_as(thread) and isinstance(condition.b, tir.IntImm))
                or (condition.b.same_as(thread) and isinstance(condition.a, tir.IntImm)))

    def add(arm, lanes, predicate):
        if not lanes:
            return
        _require(lanes == list(range(lanes[0], lanes[-1] + 1))
                 and lanes[0] % 32 == 0 and len(lanes) % 32 == 0,
                 "branches.warps", "branch must own a contiguous whole-warp interval")
        key = (lanes[0], len(lanes))
        if key not in groups:
            groups[key] = (predicate, [])
        groups[key][1].append(arm)

    def visit(node, lanes):
        if isinstance(node, tir.IfThenElse):
            if not is_range(node.condition):
                return
            yes = [t for t in lanes if int(analyzer.simplify(
                tir.stmt_functor.substitute(node.condition, {thread: tir.IntImm(thread.dtype, t)})))]
            no = [t for t in lanes if t not in yes]
            add(node.then_case, yes, str(node.condition))
            if node.else_case is not None:
                other = node.else_case
                if isinstance(other, tir.SeqStmt) and len(other.seq) == 1:
                    other = other.seq[0]
                if isinstance(other, tir.IfThenElse) and is_range(other.condition):
                    visit(other, no)
                else:
                    add(other, no, f"not ({node.condition})")
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child, lanes)
        elif isinstance(node, tir.SBlockRealize):
            visit(node.block, lanes)
        elif isinstance(node, (tir.AttrStmt, tir.SBlock)):
            visit(node.body, lanes)

    visit(body, list(range(extent)))
    records, arms = [], []
    for (start, count), (predicate, statements) in groups.items():
        arm = statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
        calls = _nodes(arm, tir.Call)
        registers = [[int(x) for x in c.args] for c in calls if _op_name(c).endswith("set_max_nreg")]
        _require(len(registers) <= 1, "branches.set_max_nreg", "ambiguous register assignments")
        record = {"predicate": predicate, "thread_min": start, "thread_count": count,
                  "warp_base": start // 32, "warps": count // 32,
                  "calls": [{"op": _op_name(c), "args": [_plain(x) for x in c.args]} for c in calls
                            if "barrier" in _op_name(c) or "tma_load" in _op_name(c)]}
        if registers:
            record["set_max_nreg"] = registers[0]
        records.append(record)
        arms.append(arm)
    return records, arms


def _barrier_schedule(branches, arms, *, explicit_mod=None):
    if explicit_mod is not None:
        return _explicit_barrier_schedule(explicit_mod, branches, arms)
    schedule = []
    # Preserve the probe's positional schedule labels; Declarations separately
    # bind semantic role names to the then/else branch selectors.
    for role, branch, arm in zip(("producer", "consumer"), branches, arms):
        for loop in _nodes(arm, tir.For):
            arrivals = [call for call in _nodes(loop.body, tir.Call) if _op_name(call) in
                        ("tirx.ptx_arrive_barrier_expect_tx", "tirx.ptx_arrive_barrier")]
            if not arrivals:
                continue
            _require(not any(any(c.same_as(a) for c in _nodes(child.body, tir.Call) for a in arrivals)
                             for child in _nodes(loop.body, tir.For)), "barrier_schedule", "arrival nested in an unmeasured inner loop")
            for arrival in arrivals:
                guards = []
                for guard in _nodes(loop.body, tir.IfThenElse):
                    _require(guard.else_case is None or not any(c.same_as(arrival) for c in _nodes(guard.else_case, tir.Call)),
                             "barrier_schedule.guard", "arrival under an unsupported else guard")
                    if any(c.same_as(arrival) for c in _nodes(guard.then_case, tir.Call)):
                        guards.append(guard)
                if guards:
                    _require(len(guards) == 1 and guards[0].else_case is None, "barrier_schedule.guard", "expected a single election guard")
                    cond = guards[0].condition
                    _require(isinstance(cond, tir.Call) and _op_name(cond) == "tl.tl_shuffle_elect",
                             "barrier_schedule.guard", "unsupported arrival election")
                    _require(int(cond.args[0]) == branch["thread_count"], "barrier_schedule.arrivals", "election extent differs from branch")
                    count = 1
                else:
                    count = branch["thread_count"]
                for k in range(int(loop.min), int(loop.min + loop.extent)):
                    barrier = _integer(arrival.args[0].indices[0], {loop.loop_var: k}, "barrier_schedule.barrier")
                    tx = [call for call in _nodes(loop.body, tir.Call)
                          if _op_name(call) in ("tl.mbarrier_expect_tx", "tirx.ptx_arrive_barrier_expect_tx")]
                    schedule.append({"role": role, "iteration": k, "barrier": barrier, "arrivals": count,
                                     "transaction_bytes": sum(int(call.args[1]) for call in tx),
                                     "guard": str(guards[0].condition) if guards else "all branch threads"})
    return schedule


def _explicit_barrier_schedule(mod, branches, arms):
    operations = {"mbarrier_wait_parity", "ptx_arrive_barrier", "ptx_arrive_cluster_barrier",
                  "mbarrier_expect_tx", "ptx_arrive_barrier_expect_tx"}
    func = next(func for func in mod.functions.values() if isinstance(func, tir.PrimFunc))
    offsets = _barrier_offsets(_allocated_buffers(func.body))
    thread = next(node.node.var for node in _nodes(func.body, tir.AttrStmt)
                  if node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x")
    schedule = []

    def relevant(call):
        return _op_name(call).split(".")[-1] in operations

    # A whole-CTA explicit program can use barriers without declaring role arms.
    work = list(zip(branches, arms)) or [(
        {"thread_min": 0, "thread_count": int(func.attrs["thread_extent"]["threadIdx.x"])}, func.body)]
    for branch_index, (branch, arm) in enumerate(work):
        analyzer = tvm.arith.Analyzer()
        analyzer.bind(thread, tvm.ir.Range.from_min_extent(branch["thread_min"], branch["thread_count"]))

        def constant(expr, values):
            simplified = analyzer.simplify(tir.stmt_functor.substitute(expr, values))
            try:
                # The pinned analyzer does not fold constant bitwise intrinsics.
                return int(_integer(simplified, {}, "barrier_schedule.expression"))
            except CaptureError:
                pass
            bound = analyzer.const_int_bound(simplified)
            return int(bound.min_value) if bound.min_value == bound.max_value else None

        def visit(node, values, iterations, guards):
            if not any(relevant(c) for c in _nodes(node, tir.Call)):
                return
            if isinstance(node, tir.SeqStmt):
                values = dict(values)
                for child in node.seq:
                    if isinstance(child, tir.Bind):
                        values[child.var] = tir.stmt_functor.substitute(child.value, values)
                    visit(child, values, iterations, guards)
            elif isinstance(node, tir.For):
                start, size = constant(node.min, values), constant(node.extent, values)
                if start is None or size is None:
                    visit(node.body, values, [*iterations, None], guards)
                else:
                    for k in range(start, start + size):
                        visit(node.body, {**values, node.loop_var: tir.IntImm(node.loop_var.dtype, k)},
                              [*iterations, k], guards)
            elif isinstance(node, tir.IfThenElse):
                known = constant(node.condition, values)
                if known != 0:
                    visit(node.then_case, values, iterations, [*guards, node.condition])
                if node.else_case is not None and known != 1:
                    visit(node.else_case, values, iterations, [*guards, tir.Not(node.condition)])
            elif isinstance(node, tir.SBlockRealize):
                visit(node.block, values, iterations, guards)
            elif isinstance(node, (tir.AttrStmt, tir.SBlock, tir.While)):
                visit(node.body, values, iterations, guards)
            else:
                for call in _nodes(node, tir.Call):
                    if not relevant(call):
                        continue
                    op = _op_name(call)
                    load = call.args[0]
                    _require(isinstance(load, tir.BufferLoad) and len(load.indices) == 1,
                             "barrier_schedule.barrier", "expected a flattened barrier buffer load")
                    slot = constant(load.indices[0], values)
                    _require(load.buffer.data in offsets, "barrier_schedule.barrier", "barrier allocation is missing")
                    base, size = offsets[load.buffer.data]
                    if slot is not None:
                        _require(0 <= slot < size, "barrier_schedule.barrier", "slot exceeds its barrier allocation")
                    arrivals = branch["thread_count"] if "arrive" in op else 0
                    if guards and arrivals:
                        arrivals = None
                        if len(guards) == 1 and isinstance(guards[0], tir.Call):
                            guard = guards[0]
                            if _op_name(guard).endswith("tl_shuffle_elect") and int(guard.args[0]) == branch["thread_count"]:
                                arrivals = 1
                    schedule.append({
                        "role": f"branch_{branch_index}" if branches else "body",
                        "iteration": iterations[0] if len(iterations) == 1 else (iterations or None),
                        "barrier": base + slot if slot is not None else None,
                        "arrivals": arrivals,
                        "transaction_bytes": constant(call.args[1], values) if op.endswith("expect_tx") else 0,
                        "guard": " and ".join(str(g) for g in guards) if guards else "all branch threads",
                        "op": op, "buffer": load.buffer.name, "slot": slot,
                    })

        visit(arm, {}, [], [])
    return schedule


def _validate(facts):
    windows, layouts = facts["captures"], facts["layouts"]
    final, early = windows["pre_codegen"], windows["post_layout"]
    branches, schedule = facts["branches"], facts["barrier_schedule"]
    if not any(window["warp_scope"] for window in windows.values()):
        initialized = {slot for slot, _ in final["barrier_init"]}
        for row in schedule:
            if row["barrier"] is not None:
                _require(row["barrier"] in initialized, "barrier_schedule.barrier",
                         f"slot {row['barrier']} has no initialization count")
        return

    counts = [branch["thread_count"] for branch in branches]
    _require(sum(counts) == final["attrs"]["thread_extent"]["threadIdx.x"], "branches.thread_count", "partitions do not cover CTA")
    _require(final["registers"] == [branch["set_max_nreg"] for branch in branches], "branches.set_max_nreg", "register calls differ from branches")
    for name, window in windows.items():
        _require(all(partition == counts for partition in window["warp_scope"]), f"captures.{name}.warp_scope", "partition differs from branch intervals")
    _require(bool(final["warp_scope"]), "captures.pre_codegen.warp_scope", "missing warp specialization annotation")
    init = final["barrier_init"]
    _require(bool(init) and [slot for slot, _ in init] == list(range(len(init))), "barrier_init", "noncontiguous or duplicate initialization slots")
    expected = [count for _, count in init]
    _require(len(early["barrier_init_annotation"]) == 1, "barrier_init_annotation", "expected one barrier allocation")
    for name, window in windows.items():
        if window["barrier_init"]:
            _require(window["barrier_init"] == init, f"captures.{name}.barrier_init", "initialization counts differ across windows")
        if window["barrier_init_annotation"]:
            _require(list(window["barrier_init_annotation"].values()) == [expected],
                     f"captures.{name}.barrier_init_annotation", "initialization counts differ across windows")
    _require(set(s["barrier"] for s in schedule) == set(range(len(init))), "barrier_schedule.barrier", "arrival coverage differs from initialization")
    for slot, count in init:
        observed = {s["arrivals"] for s in schedule if s["barrier"] == slot}
        _require(observed == {count}, f"barrier_schedule.arrivals[{slot}]", "per-use arrivals differ from initialization")
    role_slots = [{s["barrier"] for s in schedule if s["role"] == role} for role in ("producer", "consumer")]
    depth = len(role_slots[0])
    _require(depth > 0 and role_slots == [set(range(depth)), set(range(depth, 2 * depth))], "barrier_schedule.barrier", "full/empty rings disagree on depth")
    allocations = {buffer["name"]: buffer for buffer in final["allocations"]}
    _require("mbarrier" in allocations and allocations["mbarrier"]["shape"] == [len(init)], "allocations.mbarrier", "storage differs from initialized slots")
    _require("buf_dyn_shmem" in allocations, "allocations.buf_dyn_shmem", "missing merged shared allocation")
    merged = allocations["buf_dyn_shmem"]["bytes"]
    _require(merged == final["attrs"].get("dyn_shared_memory_buf"),
             "dyn_shared_memory_buf", "merged allocation disagrees with ring storage")
    # Ring planes exist before MergeSharedMemoryAllocations; their byte offsets are final-window Binds.
    rings = {buffer["name"]: buffer["bytes"] for buffer in early["allocations"]
             if buffer["scope"] == "shared.dyn" and buffer["name"] in final["offsets"]}
    _require(set(rings) == set(final["offsets"]), "allocations", "shared ring allocations differ from bound offsets")
    for name, size in rings.items():
        _require(size % depth == 0, f"allocations.{name}", "ring allocation is not divisible by depth")
    regions = sorted((final["offsets"][name], final["offsets"][name] + size) for name, size in rings.items())
    _require(all(0 <= start and end <= merged for start, end in regions)
             and all(left[1] <= right[0] for left, right in zip(regions, regions[1:])),
             "offsets", "shared ring regions overlap or exceed merged allocation")
    _require({s["transaction_bytes"] for s in schedule if s["role"] == "producer"}
             == {sum(size // depth for size in rings.values())},
             "barrier_schedule.transaction_bytes", "full-barrier transactions differ from the ring planes")
    for name, layout in layouts.items():
        if name in allocations and "thread" in layout:
            _require(layout["physical_shape"] == allocations[name]["shape"], f"layouts.{name}.physical_shape", "fragment slots differ from local allocation")
            _require(any(layout["thread_min"] == branch["thread_min"] and layout["thread_extent"] == branch["thread_count"]
                         for branch in branches), f"layouts.{name}.thread_min", "fragment range differs from branch interval")


@tvm.instrument.pass_instrument
class _PassCapture:
    def __init__(self, windows):
        self.windows = windows
        self.passes, self.mods = [], {}
        self.last_mod = None

    def run_after_pass(self, mod, info):
        name = str(info.name)
        self.passes.append(name)
        self.last_mod = mod
        if name in self.windows:
            self.mods[self.windows[name]] = mod


def capture(program, *, target, pass_configs: Mapping[str, Any],
            windows: Mapping[str, str] = DEFAULT_WINDOWS) -> Capture:
    """Lower one recipe without GPU/device compilation and capture measured facts.

    ``windows`` maps pass names to snapshot names and must retain post_layout.
    pre_codegen always comes from the verified final callback, not device_mod.
    Unsupported/missing compiler facts fail closed with ``CaptureError.fact``.

    When automatic WS is disabled, schedule-bearing functions first run
    BindTarget, MaterializeKernelLaunch and MaterializeWSSchedule; the pin's
    cuda/pipeline.py:105-107 otherwise disables both WS paths together.
    These explicit passes are included in ``Capture.passes``.
    """
    _require(tilelang.__version__ == "0.1.14", "tilelang.version", f"unsupported compiler: {tilelang.__version__}")
    _require(windows.get("tl.LayoutInference") == "post_layout", "windows.post_layout", "LayoutInference window is required")
    _require("pre_codegen" not in windows.values() and len(set(windows.values())) == len(windows), "windows", "duplicate or reserved snapshot name")
    target = tvm.target.Target(target)
    trace = _PassCapture(dict(windows))
    with target, tvm.transform.PassContext(opt_level=3, config=dict(pass_configs), instruments=[trace]):
        if pass_configs.get("tl.disable_warp_specialized") and any(
                "tl.ws_schedule" in block.annotations for block in _nodes(program.body, tir.SBlock)):
            # The pin gates both passes together (cuda/pipeline.py:105-107).
            # Materialize the authored schedule without enabling automatic WS.
            scheduled = tvm.IRModule.from_expr(program)
            scheduled = tir.transform.BindTarget(target)(scheduled)
            scheduled = tilelang.transform.MaterializeKernelLaunch()(scheduled)
            scheduled = tilelang.cuda.transform.MaterializeWSSchedule()(scheduled)
            program = next(iter(scheduled.functions.values()))
        artifact = tilelang.lower(program, target=target, enable_host_codegen=False, enable_device_compile=False)
    _require(trace.passes[-3:] == ["tl.LowerIntrin", "tirx.Simplify", "pass_fn"],
             "passes.pre_codegen", f"unexpected final callbacks: {trace.passes[-5:]}")
    # In this pin pass_fn is HoistBroadcastValues, the actual codegen input.
    trace.mods["pre_codegen"] = trace.last_mod
    _require("post_layout" in trace.mods, "windows.post_layout", "capture callback did not run")
    captured = {}
    for name, mod in trace.mods.items():
        try:
            captured[name] = _snapshot(mod)
        except (TypeError, ValueError, AttributeError, IndexError) as error:
            raise CaptureError(f"captures.{name}", str(error)) from error
    try:
        layouts, objects = _layout_capture(trace.mods["post_layout"])
    except CaptureError:
        raise
    except (TypeError, ValueError, AttributeError) as error:
        raise CaptureError("layouts", str(error)) from error
    try:
        branches, arms = _register_branches(trace.last_mod, captured["pre_codegen"])
    except CaptureError:
        raise
    except (TypeError, ValueError, AttributeError, IndexError, KeyError) as error:
        raise CaptureError("branches", str(error)) from error
    try:
        schedule = _barrier_schedule(branches, arms, explicit_mod=trace.last_mod
                                     if not captured["pre_codegen"]["warp_scope"] else None)
    except CaptureError:
        raise
    except (TypeError, ValueError, AttributeError, IndexError, KeyError) as error:
        raise CaptureError("barrier_schedule", str(error)) from error
    facts = {"captures": captured, "layouts": layouts, "branches": branches, "barrier_schedule": schedule}
    _validate(facts)
    return Capture(trace.passes, facts, {name: mod.script() for name, mod in trace.mods.items()},
                   trace.mods, objects, artifact.kernel_source)
