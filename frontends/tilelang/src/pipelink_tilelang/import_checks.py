"""Prove the bounded, once-only schedule emitted by generated tensor imports.

Ordering is checked on the final codegen TIR. Provenance is also checked on the
original TIR, so that optimization cannot erase an explicit asynchronous
operation or buffer view and make an unsupported program look safe.
The checks cover only the generated import pattern; they are not an effect or
alias analysis of arbitrary owner code.
"""
from tilelang import tvm

from .annotate import AnnotationError
from .contract import TensorImportChannel


tir = tvm.tirx
_POINTERS = {"tirx.tvm_access_ptr", "tirx.address_of", "tl.access_ptr"}
# Instructions that finish their memory access synchronously. cp.async and
# TMA/bulk operations are excluded even when a wait follows them.
_SYNC_POINTER_OPS = {
    "tl.ptx_ldmatrix", "tl.ptx_stmatrix", "tl.__ldg",
    *(f"tl.{operation}{width}" for operation in ("ldg", "stg", "lds", "sts")
      for width in (32, 64, 128, 256)),
}
# Accepted only as lowerings of a plain T.copy in the original program, whose
# compiler-inserted completion is trusted. Explicit async/TMA operations in the
# original program are rejected.
_COPY_ASYNC_OPS = {"tl.ptx_cp_async", "tirx.ptx_cp_async", "tl.tma_load", "tl.tma_store"}
_SCALAR_OPS = {
    f"tirx.{name}" for name in (
        "reinterpret", "if_then_else", "exp", "exp2", "log", "log2", "sqrt",
        "rsqrt", "abs", "fabs", "floor", "ceil", "round", "trunc", "pow",
        "sin", "cos", "tanh", "isnan", "isinf", "isfinite", "fma",
        "min", "max", "shift_left", "shift_right", "bitwise_and",
        "bitwise_or", "bitwise_xor", "bitwise_not", "likely",
    )
}


def _op(call):
    return call.op.name if isinstance(call.op, tvm.ir.Op) else str(call.op)


def _symbol(call):
    if _op(call) in ("tirx.call_extern", "tirx.call_pure_extern") and call.args:
        if isinstance(call.args[0], tir.StringImm):
            return call.args[0].value
    return None


def _true(value):
    return isinstance(value, tir.IntImm) and bool(value.value)


def _launch_loop(node):
    return (isinstance(node, tir.For) and node.thread_binding is not None
            and str(node.thread_binding.thread_tag).startswith(("threadIdx.", "blockIdx.")))


def _conditional(node):
    return (isinstance(node, (tir.IfThenElse, tir.While))
            or isinstance(node, tir.For) and not _launch_loop(node)
            or isinstance(node, tir.SBlockRealize) and not _true(node.predicate)
            or isinstance(node, tir.Call) and _op(node) == "tirx.if_then_else")


class _Schedule:
    def __init__(self, channel, func, *, initial, synchronous_copies=None):
        self.channel, self.initial = channel, initial
        self.prefix = f"tensor.imports.{channel.name}"
        self.symbols = {
            channel.produce_symbol: "produce", channel.symbol + "_ready": "ready",
            channel.symbol: "import", channel.symbol + "_done": "done",
        }
        if channel.release_symbol is not None:
            self.symbols[channel.release_symbol] = "release"
        self.events = {event: [] for event in self.symbols.values()}
        self.chunk_loops = {}
        self.output_reads = []
        self.fragment = channel._fragment
        self.scratch = [buffer for buffer, _ in channel._buffers.values()]
        self.staging = [buffer for buffer, _ in channel._staging]
        self.synchronous_copies = set() if synchronous_copies is None else synchronous_copies
        self.uses = []
        self.stack = []
        self.clock = 0
        # TVM's FFI rethrows Python callback exceptions as RuntimeError; collect
        # AnnotationErrors here and re-raise the first one after the transform.
        errors = []

        def enter(node):
            if errors:
                return node
            try:
                return self._enter(node)
            except AnnotationError as error:
                errors.append(error)
                return node

        def leave(node):
            if not errors:
                try:
                    return self._leave(node)
                except AnnotationError as error:
                    errors.append(error)
            return None

        # IRTransform visits occurrences, including shared Call objects.
        tir.stmt_functor.ir_transform(func.body, enter, leave)
        if errors:
            raise errors[0]

    def fail(self, guarantee, detail):
        raise AnnotationError(f"{self.prefix}.guarantees.{guarantee}", detail)

    def staging_fail(self, buffer, guarantee, detail):
        raise AnnotationError(f"{self.prefix}.staging.{buffer.name}.{guarantee}", detail)

    @staticmethod
    def matches(buffer, candidate):
        return (candidate.same_as(buffer) or candidate.data.same_as(buffer.data)
                or candidate.name == buffer.name or candidate.data.name == buffer.data.name)

    @staticmethod
    def pointer_matches(buffer, variable):
        return variable.same_as(buffer.data) or variable.name == buffer.data.name

    def _touch(self, buffer, *, pointer=False):
        calls = [node for node in self.stack if isinstance(node, tir.Call)]
        events = [self.symbols.get(_symbol(call)) for call in calls]
        self.uses.append((buffer, self.clock, events,
                          tuple(node for node in self.stack if isinstance(node, tir.For))))
        if buffer not in self.staging:
            if self.initial and (pointer or any(_op(call) in _POINTERS for call in calls)) and not any(events) and not any(
                    _op(call) == "tl.tileop.copy" for call in calls):
                self.fail("single_completion", f"{buffer.name} pointer escapes the checked schedule")
            return
        # Source TIR may not introduce owner views or aliases. Flattened views the
        # compiler creates in final TIR are allowed once the source check passes.
        for call in calls:
            op = _op(call)
            if op in _POINTERS or op == "tl.region":
                continue
            if op == "tl.tileop.copy" and self.initial:
                if any(key in call.annotations for key in ("barrier", "dst_block", "cluster_mask")):
                    self.staging_fail(buffer, "quiescent", "cluster/split-phase copies are not synchronous owner staging")
                self.synchronous_copies.add(buffer.name)
                continue
            if not self.initial and (op in _SYNC_POINTER_OPS or
                                     op in _COPY_ASYNC_OPS and buffer.name in self.synchronous_copies):
                continue
            if op in _SCALAR_OPS and str(call.dtype) != "handle":
                continue
            self.staging_fail(buffer, "quiescent", f"unsupported staging operation {op}")
        if pointer or any(_op(call) in _POINTERS or _op(call) == "tl.region" for call in calls):
            allowed = (any(_op(call) == "tl.tileop.copy" for call in calls) if self.initial
                       else any(_op(call) in _SYNC_POINTER_OPS or
                                _op(call) in _COPY_ASYNC_OPS and buffer.name in self.synchronous_copies
                                for call in calls))
            if not allowed:
                self.staging_fail(buffer, "no_alias", "staging address escapes a synchronous copy")

    def _enter(self, node):
        self.clock += 1
        self.stack.append(node)
        if isinstance(node, tir.AssertStmt):
            self.fail("no_early_return", "assertions can exit before the completion")
        if isinstance(node, tir.Call):
            op = _op(node)
            if op in ("tirx.ret", "tirx.tvm_throw_last_error") or _symbol(node) in ("return", "__trap", "trap"):
                self.fail("no_early_return", "kernel contains an early exit")
        if self.initial and isinstance(node, tir.SBlock):
            for match in node.match_buffers:
                for buffer in self.staging:
                    if self.matches(buffer, match.source.buffer) or self.matches(buffer, match.buffer):
                        self.staging_fail(buffer, "no_alias", "staging match_buffer view is not admitted")
        if isinstance(node, (tir.AllocBuffer, tir.DeclBuffer)):
            if self.initial:
                for buffer in self.staging:
                    if self.matches(buffer, node.buffer) and not node.buffer.same_as(buffer):
                        self.staging_fail(buffer, "no_alias", "staging buffer alias is not admitted")
            # Allocation and shape metadata do not access memory.
            self.stack.pop()
            return node
        buffers = [self.fragment, *self.scratch, *self.staging]
        if isinstance(node, (tir.BufferLoad, tir.BufferStore)):
            for buffer in buffers:
                if self.matches(buffer, node.buffer):
                    if self.initial and buffer in self.staging and not node.buffer.same_as(buffer):
                        self.staging_fail(buffer, "no_alias", "staging buffer view is not admitted")
                    self._touch(buffer)
        if isinstance(node, tir.Var):
            # Bind.var declares a variable. Address-taking shows up in Bind.value,
            # which the transform visits separately.
            parent = self.stack[-2] if len(self.stack) > 1 else None
            if isinstance(parent, tir.Bind) and node.same_as(parent.var):
                return None
            for buffer in buffers:
                if self.pointer_matches(buffer, node):
                    self._touch(buffer, pointer=True)
        return None

    def _leave(self, node):
        self.clock += 1
        if isinstance(node, tir.Call):
            if _symbol(node) in {f"{self.channel.symbol}_args_{name}" for name in self.channel.outputs}:
                self.output_reads.append(self.clock)
            event = self.symbols.get(_symbol(node))
            if event is not None:
                controls = [parent for parent in self.stack[:-1] if _conditional(parent)]
                if self.channel.chunk and (event == "import" or event == "done" and self.channel.completion == "each"):
                    if len(controls) != 1:
                        self.fail("converged", "chunk import and completion require one uniform loop")
                    loop = controls[0]
                    if (not isinstance(loop, tir.For) or loop.kind != tir.ForKind.SERIAL
                            or not isinstance(loop.min, tir.IntImm) or int(loop.min) != 0
                            or not isinstance(loop.extent, tir.IntImm)
                            or int(loop.extent) != self.channel.chunk[2]):
                        self.fail("converged", "chunk loop must cover its static binding domain")
                    self.chunk_loops[event] = loop
                    if event == "import":
                        binding = node.args[2]
                        if isinstance(binding, tir.Cast):
                            binding = binding.value
                        if not binding.same_as(loop.loop_var):
                            self.fail("converged", "import binding must be the chunk loop index")
                elif controls:
                    self.fail("converged", f"{event} is enclosed by a loop or branch")
                self.events[event].append(self.clock)
        self.stack.pop()
        return None

    def validate(self):
        for event, occurrences in self.events.items():
            if len(occurrences) != 1:
                guarantee = "single_completion" if event in ("done", "release") else "enter_once"
                self.fail(guarantee, f"expected one {event}, found {len(occurrences)}")
        if self.channel.chunk and self.channel.completion == "each":
            if not self.chunk_loops["import"].same_as(self.chunk_loops["done"]):
                self.fail("single_completion", "chunk import and completion must share one loop")
        elif self.channel.completion == "final" and (self.channel._scratch_bytes != 0 or self.scratch):
            self.fail("single_completion", "final completion requires a zero-scratch contract")
        order = [self.events[event][0] for event in ("produce", "ready", "import", "done")]
        if "release" in self.events:
            order.append(self.events["release"][0])
        if order != sorted(order):
            self.fail("single_completion", "expected produce, ready, import, done, release order")
        if any(at < self.events["produce"][0] for at in self.output_reads):
            self.fail("enter_once", "producer-written argument member is read before produce")
        imported, done = self.events["import"][0], self.events["done"][0]
        staging_end = self.events.get("release", [self.clock])[0] if self.channel.chunk else done
        for buffer, at, events, loops in self.uses:
            if buffer in self.staging:
                if not imported < at < staging_end:
                    self.staging_fail(buffer, "lifetime", "staging access is outside the checked owner lifetime")
            elif at > done:
                self.fail("single_completion", f"{buffer.name} is used after completion")
            elif buffer.same_as(self.fragment) and at < imported and "import" not in events:
                self.fail("enter_once", "destination fragment is used before import")
            if (self.channel.chunk and buffer.same_as(self.fragment)
                    and not any(loop.same_as(self.chunk_loops["import"]) for loop in loops)):
                self.fail("single_completion", "chunk fragment escapes its import iteration")


def validate_imports(stage, generated):
    """Reject unsupported generated imports before recording captured guarantees.

    Explicit ContractStage channels keep their declared promises. Every generated
    variant must provide ``modules.initial`` and ``modules.pre_codegen``; the
    checks walk these IR modules, never source text or snapshot strings.
    """
    channels = [channel for channel in stage.channels
                if isinstance(channel, TensorImportChannel) and channel.generated]
    if not channels:
        return
    for channel in channels:
        if channel._fragment is None or not generated:
            raise AnnotationError(f"tensor.imports.{channel.name}.guarantees.enter_once",
                                  "missing generated fragment or kernel")
    for generation in generated.values():
        for channel in channels:
            prefix = f"tensor.imports.{channel.name}.guarantees"
            modules = generation["modules"]
            synchronous_copies = set()
            for window in ("initial", "pre_codegen"):
                if window not in modules:
                    raise AnnotationError(prefix + ".enter_once", f"missing {window} TIR provenance")
                funcs = [func for func in modules[window].functions.values() if isinstance(func, tir.PrimFunc)]
                if len(funcs) != 1:
                    raise AnnotationError(prefix + ".enter_once", "expected exactly one kernel")
                if window == "pre_codegen":
                    extents = generation["facts"]["captures"][window]["attrs"].get("thread_extent", {})
                    width = extents.get("threadIdx.x")
                    role = stage.roles[channel.role].warp_range
                    if (not isinstance(width, int) or width <= 0 or width % 32
                            or role != (0, width // 32)
                            or any(extents.get(axis, 1) != 1 for axis in ("threadIdx.y", "threadIdx.z"))):
                        raise AnnotationError(prefix + ".full_warps", "import role must span all complete kernel warps")
                _Schedule(channel, funcs[0], initial=window == "initial",
                          synchronous_copies=synchronous_copies).validate()
