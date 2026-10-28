"""Generate an explicit TileLang role from a packed partner, without its headers.

A Channel owns allocations in the current PrimFunc. Handle-sized rings require
``buffer(depth=...)`` before synchronization; fixed rings need no depth argument.
Shared-completion declarations name the complete group. TMEM requires the
explicitly consumed publication endpoint as ``base=...``.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from contextvars import ContextVar
from itertools import product
from dataclasses import dataclass
from pathlib import Path

from pipec2.model import BarrierSide, SharedMember, StageManifest, TxSum
from pipec2.schema import load_stage_manifest
from pipec2.stubgen import _declarator, _split_parameters, declaration_parts

from .declarations import Endpoint, InitContract, Role, Sharing, StageDeclaration, Variants
from .external import ExternalBarrier, ExternalStorage, NativeStorage


@dataclass(frozen=True)
class Partner:
    path: Path
    raw: dict
    manifest: StageManifest
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> Partner:
        path = Path(path)
        package = path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            raw = json.loads(archive.read("manifest.json"))
        return cls(path, raw, load_stage_manifest(raw), hashlib.sha256(package).hexdigest())

    @property
    def endpoints(self):
        return {endpoint.name: endpoint for endpoint in self.manifest.endpoints}

    @property
    def profile(self) -> str:
        """The pipec2 profile the partner was packed for; contract roles target it too."""
        return self.manifest.target.profile

    @property
    def scheduler_strategy(self):
        topology = self.manifest.topology
        return topology.scheduler_strategy if topology is not None else None

    @property
    def variants(self):
        return tuple(variant.key for variant in self.manifest.variants)

    @property
    def roles(self):
        """Role widths in warps, as the partner manifest declares them."""
        return {name: role.warps for name, role in self.manifest.roles.items()}


_COMPLEMENT = {"produce": "consume", "consume": "produce", "provide": "require", "require": "provide"}
_DTYPES = {"bf16": "bfloat16", **{f"{short}{bits}": f"{long}{bits}"
           for short, long, widths in (("f", "float", (16, 32, 64)),
                                       ("i", "int", (8, 16, 32, 64)),
                                       ("s", "int", (8, 16, 32, 64)),
                                       ("u", "uint", (8, 16, 32, 64))) for bits in widths}}
_ARRIVE_TERM = re.compile(r"\s*(?:([1-9][0-9]*)\s*\*\s*)?(threads|warps)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*")


def _arrive_count(count, roles):
    if isinstance(count, int) or count == "1":
        return int(count)
    known, unknown = 0, []
    for term in count.split("+"):
        coefficient, unit, role = _ARRIVE_TERM.fullmatch(term).groups()
        coefficient = int(coefficient or 1)
        if role in roles:
            known += coefficient * roles[role] * (32 if unit == "threads" else 1)
        else:
            unknown.append((f"{coefficient}*" if coefficient != 1 else "") + f"{unit}({role})")
    if not unknown:
        return known
    return " + ".join(([str(known)] if known else []) + unknown)


def _channel(partner, endpoint, role, directions, roles):
    if not isinstance(role, str) or not role:
        raise TypeError("role must be a nonempty string")
    if not isinstance(partner, Partner):
        raise TypeError("partner must be loaded with Partner.load")
    definition = partner.endpoints[endpoint]
    if definition.direction not in directions:
        raise ValueError(f"{endpoint}: partner direction {definition.direction!r} is not complementary")
    return Channel(partner, definition, role, roles)


def consume(partner: Partner, endpoint: str, *, role: str, roles: dict[str, int] | None = None) -> Channel:
    """Consume a partner endpoint, resolving this stage's role widths in warps."""
    return _channel(partner, endpoint, role, ("produce", "provide"), roles)


def produce(partner: Partner, endpoint: str, *, role: str, roles: dict[str, int] | None = None) -> Channel:
    """Produce for a partner endpoint, resolving this stage's role widths in warps."""
    return _channel(partner, endpoint, role, ("consume", "require"), roles)


@dataclass(frozen=True)
class _FragmentDemand:
    """Demand the captured lowered layout of the named destination fragment.

    Each held element copy (every replica of a replicated fragment) demands its
    logical coordinate at the holding participant and fragment position. The
    recipe chooses the layout (``T.annotate_layout``, e.g. with
    ``TensorImportChannel.delivery_layout()``) or leaves it to TileLang inference.
    """
    fragment: str
    site: str = "tensor_import_site"
    participants: str = "pl::all_threads"
    guarantees: tuple[str, ...] = ("enter_once", "converged", "full_warps", "no_early_return")


@dataclass(frozen=True)
class _TensorRendezvous:
    event: str
    barrier: str
    cohort: str = "pl::all_threads"


_import_capture = ContextVar("pipelink_import_capture", default=None)


def import_tensor(partner: Partner, export_name: str, *, symbol=None, arguments=None,
                  frame_words=None, scratch_bytes=None, metadata=1, outputs=()) -> TensorImportChannel:
    """Import a packed tensor; this frontend generates its storage and schedule.

    ``arguments`` overrides the default binding by published member name.
    Capacity overrides still pass the linker's envelope checks.
    The generated single-site catalogues keep only static guard metadata and
    at most one warp-scoped ``receipt`` handle (machine.action_domain /
    construct.receipt), so one metadata slot suffices on every supported profile.
    """
    return TensorImportChannel(partner, export_name, symbol or f"pl_import_{export_name}",
                               arguments, frame_words, scratch_bytes, metadata, outputs)


def _placement_expr(node, env):
    """A parsed pipec2 placement expression over TileLang index variables, with its exact semantics.

    Returns ``None`` for operations TileLang fragment layouts cannot express
    (bitwise mixing, predicates); shifts become exact integer arithmetic.
    """
    from tilelang import tvm
    tir = tvm.tirx
    args = []
    for arg in node.args:
        if hasattr(arg, "op"):
            arg = _placement_expr(arg, env)
            if arg is None:
                return None
        args.append(arg)
    match node.op:
        case "const":
            return tir.IntImm("int32", args[0])
        case "cbits":
            return tir.IntImm("int32", args[1])
        case "coord":
            return env[args[0]]
        case "tobits":
            return args[1]
        case "toint":
            return args[0]
        case "add":
            return args[0] + args[1]
        case "sub":
            return args[0] - args[1]
        case "mulc":
            return args[0] * args[1]
        case "divc":
            return tir.floordiv(args[0], args[1])
        case "modc":
            return tir.floormod(args[0], args[1])
        case "shlc" | "shrc":
            width, shift = int(node.sort[5:-1]), args[1]
            if shift >= width:
                return tir.IntImm("int32", 0)
            if node.op == "shrc":
                return tir.floordiv(args[0], 1 << shift)
            return tir.floormod(args[0] * (1 << shift), 1 << width)
    return None


class TensorImportChannel:
    def __init__(self, partner, export_name, symbol, arguments, frame_words, scratch_bytes, metadata, outputs=()):
        self.partner, self.role, self.symbol = partner, None, symbol
        self.generated = True
        self.export_index, self.export = next((index, item) for index, item in enumerate(partner.raw["tensor"]["exports"])
                                              if item["name"] == export_name)
        self.name = symbol
        self._frame_words, self._scratch_bytes, self._metadata = frame_words, scratch_bytes, metadata
        self.demand = None
        self.lends = ()
        calls = self.export["calls"]
        self.produce_endpoint = partner.endpoints[calls["produce"]]
        self.release_endpoint = partner.endpoints[calls["release"]] if calls["release"] else None
        self.produce_name = self.produce_endpoint.name
        self.release_name = self.release_endpoint.name if self.release_endpoint is not None else None
        self.ready = self.export["readiness"]["after"]
        self.done = self.export["readiness"]["reuse_after"]
        self.rendezvous = tuple(_TensorRendezvous(event, "outer") for event in (self.ready, self.done))
        self._buffers, self._staging = {}, []
        self._fragment = None
        self.chunk = None
        self.completion = "each"
        self.outputs = tuple(outputs)
        self._output_reads = set()
        self._builder = None
        self.arguments = arguments
        self._arguments_explicit = arguments is not None
        self.args_param = f"{symbol}_args"
        endpoint = self.produce_endpoint
        declaration = endpoint.protocol.call.decl
        exported = partner.raw["stage"]["exports"][endpoint.protocol.call.export]["symbol"]
        _, parameters, _ = declaration_parts(declaration, exported)
        parameters = _split_parameters(parameters)
        self.view_call = len(parameters) == 3
        parameter = parameters[1 if self.view_call else 0].strip()
        self.argument_type = parameter.removeprefix("const ").removesuffix("*").strip()
        self.argument_decl = next((item for item in partner.manifest.types or ()
                                   if item.name == self.argument_type), None)
        if self.argument_decl is None:
            raise ValueError(f"{export_name}: produce arguments need a published PL_TYPE {self.argument_type}")
        names = {member.name for member in self.argument_decl.members}
        if not set(self.outputs) <= names or (self.outputs and parameter.startswith("const ")):
            raise ValueError(f"{export_name}: outputs require published members and mutable produce arguments")
        if arguments is None:
            self.arguments = {member.name: member.name for member in self.argument_decl.members
                              if member.name not in self.outputs}
        elif set(arguments) & set(self.outputs):
            raise ValueError(f"{export_name}: output members cannot also be input arguments")

    @property
    def argument_members(self) -> tuple[str, ...]:
        """Member names of the published argument type, in declaration order."""
        return tuple(member.name for member in self.argument_decl.members)

    def call_declaration(self, endpoint, symbol):
        """Rename the callable and keep its published type prelude."""
        declaration = endpoint.protocol.call.decl
        exported = self.partner.raw["stage"]["exports"][endpoint.protocol.call.export]["symbol"]
        _, match = _declarator(declaration, exported)
        return declaration[:match.start()] + symbol + declaration[match.end():]

    @property
    def produce_symbol(self):
        return f"pipelink_call_{self.produce_name}"

    @property
    def release_symbol(self):
        return f"pipelink_call_{self.release_name}" if self.release_name is not None else None

    def _context(self):
        from tilelang import tvm
        builder = tvm.script.ir_builder.IRBuilder.current()
        if self._builder is None or not builder.same_as(self._builder):
            self._builder = builder
            self._buffers, self._staging = {}, []
            self._fragment = None
            self._args_var = tvm.tirx.Var(self.args_param, "handle")
            self._view_var = tvm.tirx.Var(f"{self.symbol}_view", "handle")
        capture = _import_capture.get()
        if capture is not None and self not in capture:
            capture.append(self)

    def fragment(self, shape=None, dtype=None, *, specialize=False, chunk=None):
        """Allocate the destination, optionally adopting the export's delivery layout."""
        import tilelang.language as T
        from tilelang.language.kernel import KernelLaunchFrame
        from tilelang.layout import Fragment
        self._context()
        identity = self.export["identity"]
        shape = tuple(shape or (item["hi"] - item["lo"] for item in identity["coords"]))
        self.chunk = None
        self._chunk_offsets = ()
        if chunk is not None:
            axis, width = chunk
            index = next(i for i, coord in enumerate(identity["coords"]) if coord["name"] == axis)
            if width <= 0 or shape[index] % width:
                raise ValueError("chunk width must divide the exported axis")
            self.chunk = (index, width, shape[index] // width)
            self._chunk_offsets = (0,) * self.chunk[2]
            shape = tuple(width if i == index else extent for i, extent in enumerate(shape))
        dtype = dtype or _DTYPES[identity["dtype"]]
        if dtype != _DTYPES[identity["dtype"]]:
            from .annotate import AnnotationError
            raise AnnotationError(f"tensor.imports.{self.name}.demand", "fragment dtype differs from export identity")
        fragment = T.alloc_fragment(shape, dtype)
        self._fragment = fragment
        layout = self.delivery_layout() if specialize else None
        self.specialization = {"requested": bool(specialize), "adopted": layout is not None}
        threads = int(KernelLaunchFrame.Current().get_num_threads())
        if layout is None and len(shape) == 2:
            rows, columns = shape
            vector = self._coalesced_width
            if columns % vector == 0 and threads % (columns // vector) == 0:
                row_group = threads // (columns // vector)
                if rows % row_group == 0:
                    layout = Fragment(shape,
                        forward_thread_fn=lambda m, n: (m % row_group) * (columns // vector) + n // vector,
                        forward_index_fn=lambda m, n: (m // row_group) * vector + n % vector)
        if layout is not None:
            T.annotate_layout({fragment: layout})
        return fragment

    def _frame_extent(self, threads):
        from pipec2.tensor.expr import evaluate
        export = self.partner.manifest.tensor.exports[self.export_index]
        extent = self._frame_words
        if extent is not None:
            return extent
        positions = [p for p in export.placements if p.space == "reg"]
        if positions:
            env = dict(export.fixed_bindings)
            try:
                return 1 + max(evaluate(p.components["position"], {**env, **dict(zip(
                    (c.name for c in export.identity.coords), point))})
                    for point in product(*(c.values if c.values is not None else range(c.lo, c.hi)
                                           for c in export.identity.coords))
                    for p in positions)
            except KeyError:
                # Binding-dependent extents use the even split below; T-PLAN-FRAME still checks them.
                pass
        return math.ceil(math.prod(c.hi - c.lo for c in export.identity.coords) / threads)

    def _storage(self):
        from tilelang import tvm
        from tilelang.language.kernel import KernelLaunchFrame
        words = self._frame_extent(int(KernelLaunchFrame.Current().get_num_threads()))
        identity = self.export["identity"]
        size = self._scratch_bytes
        if size is None:
            size = math.prod(c["hi"] - c["lo"] for c in identity["coords"]) * tvm.DataType(_DTYPES[identity["dtype"]]).bits // 8
            if self.chunk:
                size //= self.chunk[2]
        scratch = (dict(kind="smem", name="scratch", extent=size, align=128,
                        handle=f"pipelink_{self.symbol}_scratch"),) if size else ()
        self.lends = (*scratch,
            dict(kind="regs", name="destination", handle="fragment"),
            dict(kind="cta_barrier", name="outer", base=0, sync=(2, 1)),
            dict(kind="frame", name="caller_frame", extent=words, handle=f"{self.symbol}_frame"),
            dict(kind="metadata", name="metadata", extent=self._metadata, handle=f"{self.symbol}_metadata"))
        return words, size

    def produced(self, *, completion="each"):
        """Produce one generation and rendezvous once, enclosing the chunk loop."""
        import tilelang.language as T
        from tilelang import tvm
        from tvm.script.ir_builder.base import IRBuilderFrame
        self._context()
        if completion not in ("each", "final"):
            raise ValueError("completion must be 'each' or 'final'")
        if completion == "final" and (self.chunk is None or self._scratch_bytes != 0):
            raise ValueError("final completion requires chunked imports with explicit scratch_bytes=0")
        self.completion = completion
        words, size = self._storage()
        channel = self

        class Produced(IRBuilderFrame):
            def __enter__(self):
                frame = T.alloc_local((words,), _DTYPES[channel.export["identity"]["dtype"]])
                tvm.script.ir_builder.IRBuilder.name(channel.symbol + "_frame", frame)
                if size:
                    scratch = T.alloc_shared((size,), "uint8")
                    tvm.script.ir_builder.IRBuilder.name(channel.symbol + "_scratch", scratch)
                    channel._buffers["scratch"] = (scratch, f"pipelink_{channel.symbol}_scratch")
                    channel._scratch_pointer = T.address_of(scratch[0])
                else:
                    channel._scratch_pointer = T.reinterpret(T.uint64(0), "handle")
                source = T.alloc_local((1,), "uint64")
                channel._source = source
                T.buffer_store(source, T.call_extern("uint64", "pipelink_tilelang_source_word",
                    T.call_extern("handle", channel.produce_symbol, *channel._call_args(),
                                  T.address_of(frame[0]))), [0])
                T.evaluate(T.call_extern("void", channel.symbol + "_ready"))

            def __exit__(self, *exc):
                if channel.completion == "final":
                    T.evaluate(T.call_extern("void", channel.symbol + "_done"))
                if channel.release_symbol is not None:
                    T.evaluate(T.call_extern("void", channel.release_symbol, *channel._call_args()))

        return Produced()

    def imported(self, fragment, *, chunk=None):
        """Import/complete one chunk, or produce/import/complete/release a full tile."""
        import tilelang.language as T
        from tvm.script.ir_builder.base import IRBuilderFrame
        self._context()
        if (chunk is None) != (self.chunk is None):
            raise ValueError("chunked fragments require an explicit chunk index")
        self._fragment = fragment
        self.demand = _FragmentDemand(fragment.name, guarantees=(
            "enter_once", *(("single_completion",) if chunk is None or self.completion == "final" else ()),
            "converged", "full_warps", "no_early_return"))
        channel = self
        produced = channel.produced() if chunk is None else None

        class Imported(IRBuilderFrame):
            def __enter__(self):
                if produced is not None:
                    produced.__enter__()
                bindings = [] if chunk is None else [T.cast(chunk, "uint32")]
                destination = fragment.data
                if str(fragment.dtype) in ("float16", "bfloat16"):
                    destination = T.call_extern("handle", "pipelink_tilelang_bits16", destination)
                T.evaluate(T.call_extern("void", channel.symbol,
                    T.call_extern("handle", "pipelink_tilelang_source_from_word", channel._source[0]),
                    *bindings, destination, channel._scratch_pointer))

            def __exit__(self, *exc):
                if channel.completion == "each":
                    T.evaluate(T.call_extern("void", channel.symbol + "_done"))
                if produced is not None:
                    produced.__exit__(*exc)

        return Imported()

    @property
    def args(self):
        """Read the per-thread argument record after the producer returns."""
        import tilelang.language as T
        channel = self

        class Members:
            def __getattr__(self, name):
                member = next((m for m in channel.argument_decl.members if m.name == name), None)
                if member is None:
                    raise AttributeError(name)
                dtype = {"int32_t": "int32", "uint32_t": "uint32", "float": "float32",
                         "int64_t": "int64", "uint64_t": "uint64", "double": "float64"}.get(member.ctype)
                if dtype is None or member.extent:
                    raise ValueError(f"argument member {name} is not a supported scalar")
                channel._output_reads.add(name)
                return T.call_extern(dtype, f"{channel.symbol}_args_{name}", channel._args_var)

        return Members()

    def _args(self):
        import tilelang.language as T
        return T.call_extern("handle", f"{self.symbol}_args_cast", self._args_var)

    def _call_args(self):
        return [self._view_var, self._args()] if self.view_call else [self._args()]

    def staging(self, shape, dtype):
        """Allocate synchronous owner storage live between this import's ready/done."""
        import tilelang.language as T
        self._context()
        buffer = T.alloc_shared(shape, dtype)
        self._staging.append((buffer, f"pipelink_{self.symbol}_staging_{len(self._staging)}"))
        return buffer

    def _delivery_components(self):
        from pipec2.profile import get_profile
        from pipec2.tensor import delivery_layout
        from tilelang.language.kernel import KernelLaunchFrame
        frame = KernelLaunchFrame.Current()
        if frame is None:
            raise RuntimeError("delivery layout queries require an active T.Kernel")
        tensor = self.partner.manifest.tensor
        return delivery_layout(tensor.exports[self.export_index], tensor.bindings,
                               get_profile(self.partner.profile), range(int(frame.get_num_threads())))

    def delivery_layout(self):
        """The export's register or TMEM receipt layout as a fragment, else ``None``.

        Valid only inside ``T.Kernel``: its active threads determine receipt
        ownership. Pipec derives the map; this frontend renders expressible
        operations. Correctness never depends on specialization: the linker
        still plans and checks the connection.
        """
        from tilelang import tvm
        from tilelang.layout import Fragment
        components = self._delivery_components()
        if components is None:
            return None
        coords = self.partner.manifest.tensor.exports[self.export_index].identity.coords
        names, shape = [item.name for item in coords], [item.hi for item in coords]
        probe = {name: tvm.tirx.Var(name, "int32") for name in names}
        if any(_placement_expr(component, probe) is None for component in components):
            return None
        compact = None
        if self.chunk:
            from pipec2.tensor.expr import evaluate
            axis, width, count = self.chunk
            shape[axis] = width
            points = list(product(*(range(size) for size in shape)))
            holders = [(evaluate(components[0], dict(zip(names, point))),
                        evaluate(components[1], dict(zip(names, point)))) for point in points]
            owners = {thread for thread, _ in holders}
            compact = len(points) // len(owners)
            expected = [(thread, position % compact) for thread, position in holders]
            if (len(set(expected)) != len(points)
                    or any(sum(thread == owner for thread, _ in expected) != compact for owner in owners)):
                return None
            offsets = [0]
            for chunk in range(1, count):
                offset = None
                for point, holder in zip(points, expected):
                    env = dict(zip(names, point))
                    env[names[axis]] += chunk * width
                    owner = evaluate(components[0], env)
                    if offset is None:
                        offset = owner - holder[0]
                    if (owner - offset, evaluate(components[1], env) % compact) != holder:
                        return None
                if offset % 32:
                    return None
                offsets.append(offset)
            self._chunk_offsets = tuple(offsets)

        def at(component):
            return lambda *variables: _placement_expr(component, dict(zip(names, variables)))
        return Fragment(shape, forward_thread_fn=at(components[0]),
                        forward_index_fn=lambda *variables: (
                            at(components[1])(*variables) % compact if compact is not None
                            else at(components[1])(*variables)))

    @property
    def _coalesced_width(self):
        return 8 if self.partner.profile == "sm_90a" else 16

    def delivery_layout_is_narrow(self, access_width=None):
        """Whether delivery has shorter fastest-axis runs than the access width.

        Valid only inside ``T.Kernel``. A run stays on one participant and
        increments its private position by one. The default is the frontend's
        coalesced access width, not a prediction that staging is profitable.
        Unsupported delivery layouts use the fallback and return False.
        """
        from pipec2.tensor.expr import evaluate
        if self.delivery_layout() is None:
            return False
        width = self._coalesced_width if access_width is None else access_width
        export = self.partner.manifest.tensor.exports[self.export_index]
        participant, position = self._delivery_components()
        coords = export.identity.coords
        for prefix in product(*(range(c.lo, c.hi) for c in coords[:-1])):
            env = dict(zip((c.name for c in coords[:-1]), prefix))
            previous, run = None, 0
            for column in range(coords[-1].lo, coords[-1].hi):
                env[coords[-1].name] = column
                current = (evaluate(participant, env), evaluate(position, env))
                if previous is not None and current != (previous[0], previous[1] + 1):
                    if run < width:
                        return True
                    run = 0
                run += 1
                previous = current
            if run < width:
                return True
        return False

    @property
    def external(self):
        return [ExternalStorage(buffer.name, handle, "smem")
                for buffer, handle in [*self._buffers.values(), *self._staging]]


class Channel:
    def __init__(self, partner, endpoint, role, roles=None):
        self.partner, self.endpoint, self.role = partner, endpoint, role
        widths = partner.roles.copy()
        for name, width in (roles or {}).items():
            if not isinstance(name, str) or not name or type(width) is not int or width <= 0:
                raise ValueError("roles must map nonempty role names to positive warp widths")
            if name in widths and widths[name] != width:
                raise ValueError(f"{name}: own role width conflicts with the partner role")
            widths[name] = width
        self.direction = _COMPLEMENT[endpoint.direction]
        self.tile, self.layout, self.space = endpoint.tile, endpoint.layout, endpoint.space
        # Completion-only (space none) endpoints have no data: wait/arrive, never buffer().
        self.dtype = endpoint.tile.dtype if endpoint.space != "none" else None
        self.protocol, self.binds = endpoint.protocol, endpoint.binds
        self.depth = endpoint.ring.depth.fixed if endpoint.ring else 1
        self.depth_handle = endpoint.ring.depth.handle if endpoint.ring else None
        topology = partner.manifest.topology
        # A role-less partner may omit its topology; it then imposes no CTA pairing.
        self.cta_group = topology.cta_group if topology is not None else 1
        self.cluster = topology.cluster_compiled if topology is not None else (1, 1, 1)
        self.cluster_release = endpoint.scope == "cluster" or self.cta_group > 1
        self.full = self.protocol.full
        if isinstance(self.full, SharedMember):
            self.full = partner.endpoints[self.full.member_of].protocol.full
        self.empty = self.protocol.empty
        self.counts = {side: _arrive_count(value.arrive.count, widths)
                       for side, value in (("full", self.full), ("empty", self.empty))
                       if value is not None}
        self._builder = None
        self._buffers = {}
        self._records = []
        self._allocated_depth = self.depth

    def _context(self):
        from tilelang import tvm
        builder = tvm.script.ir_builder.IRBuilder.current()
        if self._builder is None or not builder.same_as(self._builder):
            self._builder = builder
            self._buffers = {}
            self._records = []
            self._allocated_depth = self.depth

    @property
    def external(self) -> list[ExternalBarrier | ExternalStorage]:
        # Script assignment may rename buffers after buffer() returns.
        return [record(buffer=buffer if isinstance(buffer, str) else buffer.name, **fields)
                for buffer, record, fields in self._records]

    def buffer(self, *, depth: int | None = None, base: Channel | None = None):
        """Allocate contract storage; omit a handle-ring's depth for WS versioning.

        Explicit synchronization uses ``buffer(depth=...)``. A scheduled pipeline
        instead versions the single logical plane returned by ``buffer()``.
        TMEM base names its publication channel.

        Shared planes allocate TileLang's 2-D view of the contract plane: a rank-3
        tile ``(R, C, H)`` becomes ``(R, H*C)`` with ``(r, c, h) -> (r, h*C + c)``
        (``layout.plane_shape``), behind an optional leading ring dimension. The
        storage is bound to ``solved(binds.base) + layout.origin_bits / 8``.
        """
        import tilelang.language as T
        from .layout import origin_bytes, plane_shape, tmem_shape, to_tilelang
        if self.space == "none":
            raise ValueError(f"{self.endpoint.name}: completion-only endpoint has no storage")
        self._context()
        selected = self.depth if depth is None else depth
        if selected is not None or self.space != "smem":
            self._check_depth(selected)
        if self._buffers and selected != self._allocated_depth:
            raise ValueError("a channel cannot change depth within one PrimFunc")
        self._allocated_depth = selected
        if "storage" in self._buffers:
            return self._buffers["storage"]
        dtype = _DTYPES.get(self.dtype)
        if dtype is None:
            raise ValueError(f"unsupported TileLang dtype: {self.dtype}")
        handle = self.binds.get("base")
        if handle is None:
            raise ValueError(f"{self.endpoint.name}: protocol has no storage base")
        column_handle, offset = None, 0
        if self.space == "smem":
            plane = plane_shape(self.layout, profile=self.partner.profile)
            ring = self.endpoint.ring
            if (ring is not None and selected != 1
                    and ring.stage_stride_bytes * 8 != math.prod(plane) * self.layout.elem_bits):
                raise ValueError(f"{self.endpoint.name}: ring stride differs from TileLang's compact plane footprint")
            claim = next(claim for claim in self.partner.manifest.claims if claim.handle == handle)
            offset = origin_bytes(self.layout, profile=self.partner.profile, base_align=claim.align)
            buffer = T.alloc_shared(plane if selected is None else (selected, *plane), dtype)
        elif self.space == "tmem":
            if (not isinstance(base, Channel) or base.partner.sha256 != self.partner.sha256
                    or base.space != "smem" or base.dtype != "u32" or base.tile.shape != (1,)
                    or base.protocol.kind != "named_barrier" or base.direction not in ("consume", "require")):
                raise ValueError("TMEM buffer requires base=<consumed u32 publication channel from this partner>")
            column_handle, handle = handle, base.binds["base"]
            lanes, columns = tmem_shape(self.layout, profile=self.partner.profile)
            stride_bits = self.endpoint.ring.stage_stride_bytes * 8 if self.endpoint.ring else columns * self.layout.elem_bits
            if stride_bits % self.layout.elem_bits:
                raise ValueError(f"{self.endpoint.name}: TMEM ring stride must be element-integral")
            stride = stride_bits // self.layout.elem_bits
            claim = next(claim for claim in self.partner.manifest.claims if claim.handle == column_handle)
            if not claim.extent.per and (selected - 1) * stride + columns > claim.extent.const:
                raise ValueError(f"{self.endpoint.name}: TMEM ring exceeds its bound allocation")
            buffer = T.alloc_tmem((selected, lanes, columns), dtype)
            T.annotate_layout({buffer: T.Layout((selected, lanes, columns),
                                               lambda slot, lane, col: (lane, slot * stride + col))})
        else:
            raise ValueError(f"{self.endpoint.name}: cannot allocate {self.space} storage")
        if self.space == "smem":
            T.annotate_layout({buffer: to_tilelang(self.layout, buffer, profile=self.partner.profile)})
        self._buffers["storage"] = buffer
        self._records.append((buffer, ExternalStorage, dict(handle=handle, space=self.space,
                                                          column_handle=column_handle, offset_bytes=offset)))
        return buffer

    def _check_depth(self, depth):
        if type(depth) is not int or depth <= 0:
            raise ValueError(f"{self.endpoint.name}: pass depth explicitly for {self.depth_handle}")
        if self.depth is not None and depth != self.depth:
            raise ValueError(f"{self.endpoint.name}: fixed ring depth is {self.depth}")
        if self.depth_handle and depth not in self._variant_values():
            raise ValueError(f"{self.endpoint.name}: depth {depth} is not a partner variant")

    def _barrier_fields(self, side, depth):
        if self.protocol.kind != "mbarrier":
            raise ValueError(f"{self.endpoint.name}: requires mbarrier protocol, got {self.protocol.kind}")
        definition = self.full if side == "full" else self.empty
        if definition is None:
            raise ValueError(f"{self.endpoint.name}: release={self.protocol.release!r} has no {side} barrier")
        init = self.protocol.init
        if init.phase not in (0, 1):
            raise ValueError(f"{self.endpoint.name}: unsupported symbolic init phase {init.phase!r}")
        native = definition.native
        if native is not None:
            claim = next(claim for claim in self.partner.manifest.claims if claim.handle == native.base)
            native = NativeStorage(native.base, 16 if claim.domain == "mbarrier" else 1,
                                   native.offset_bytes, native.stride_bytes)
        # A physical warp alone does not identify a role; never guess it from widths.
        keep = init.materializer == "stage" and init.owner == self.role
        return dict(side=side, handle=definition.slots, native=native, depth=depth,
                    count=self.counts[side], init="keep" if keep else "drop",
                    phase=init.phase, init_warp=init.warp)

    def ws_pipeline(self, name: str, buffers, *, depth: int | None = None):
        """Version logical planes and bind the scheduled barriers to this contract.

        Only the partner-signaled side supplies an explicit arrival count; the
        scheduled role's own side is derived from its signaling operations.
        """
        import tilelang.language as T
        self._context()
        selected = self.depth if depth is None else depth
        self._check_depth(selected)
        fields = {side: self._barrier_fields(side, selected) for side in ("full", "empty")}
        partner_side = "full" if self.direction in ("consume", "require") else "empty"
        count = fields[partner_side]["count"]
        if not isinstance(count, int):
            raise ValueError(f"{self.endpoint.name}.{partner_side}: unresolved arrive count {count}; supply missing stage role widths with roles=")
        if self.cluster_release and math.prod(self.cluster) > self.cta_group:
            raise ValueError("WSPipeline requires a constant cluster-leader rank; multiple CTA groups are unsupported")
        pipeline = T.WSPipeline(name, buffers, selected, **{partner_side + "_count": count},
                                arrive_cta=0 if self.cluster_release else -1,
                                init_phase=self.protocol.init.phase)
        self._allocated_depth = selected
        # PlanVersionedBuffers creates these names, not endpoint_full/empty.
        self._records.extend((f"{name}_{side}", ExternalBarrier, fields[side])
                             for side in ("full", "empty"))
        return pipeline

    def _barrier(self, side):
        import tilelang.language as T
        self._context()
        fields = self._barrier_fields(side, self._allocated_depth)
        if side in self._buffers:
            return self._buffers[side]
        count = self.counts[side]
        if not isinstance(count, int):
            raise ValueError(f"{self.endpoint.name}.{side}: unresolved arrive count {count}; supply missing stage role widths with roles=")
        depth = self._allocated_depth
        if depth is None:
            raise ValueError(f"{self.endpoint.name}: call buffer(depth=...) before synchronization for {self.depth_handle}")
        allocate = T.alloc_cluster_barrier if self.cluster_release else T.alloc_barrier
        buffer = allocate([count] * depth)
        self._builder.name(f"{self.endpoint.name}_{side}", buffer)
        self._buffers[side] = buffer
        self._records.append((buffer, ExternalBarrier, fields))
        return buffer

    def _named_id(self):
        import tilelang.language as T
        return T.call_extern("int32", self.protocol.named_barrier.id_handle)

    def wait(self, slot, phase):
        import tilelang.language as T
        if self.protocol.kind == "named_barrier":
            return T.sync_threads(self._named_id(), self.protocol.named_barrier.count)
        side = "full" if self.direction in ("consume", "require") else "empty"
        barrier = self._barrier(side)
        inversion = self.protocol.init.phase ^ int(side == "empty")
        parity = phase ^ 1 if inversion else phase
        return T.mbarrier_wait_parity(barrier[slot], parity)

    def release(self, slot):
        import tilelang.language as T
        if self.direction not in ("consume", "require"):
            raise ValueError("release is a consumer operation; producers use arrive")
        barrier = self._barrier("empty")
        if self.cluster_release:
            rank = T.block_rank_in_cluster()
            return T.mbarrier_arrive(barrier[slot], rank - rank % self.cta_group)
        return T.mbarrier_arrive(barrier[slot])

    def arrive(self, slot):
        import tilelang.language as T
        if self.direction not in ("produce", "provide"):
            raise ValueError("arrive is a producer operation; consumers use release")
        if self.protocol.kind == "named_barrier":
            return T.named_barrier_arrive(self._named_id(), self.protocol.named_barrier.count)
        return T.mbarrier_arrive(self._barrier("full")[slot])

    @property
    def tx_bytes(self):
        if not isinstance(self.full, BarrierSide):
            raise ValueError(f"{self.endpoint.name}: no full transaction contract")
        value = self.full.tx_bytes
        if isinstance(value, int):
            return value
        endpoints = self.partner.endpoints
        names = value.sum if isinstance(value, TxSum) else (self.endpoint.name,)
        total = 0
        for name in names:
            member = endpoints[name]
            contribution = member.protocol.full.tx_bytes
            if isinstance(contribution, int):
                total += contribution
            else:
                bits = math.prod(member.tile.shape) * member.layout.elem_bits
                if bits % 8:
                    raise ValueError(f"{name}: transaction tile is not byte-integral")
                total += bits // 8
        return total

    def expect_tx(self, slot):
        import tilelang.language as T
        if self.direction not in ("produce", "provide"):
            raise ValueError("expect_tx is a producer operation")
        return T.mbarrier_arrive_expect_tx(self._barrier("full")[slot], self.tx_bytes)

    def _group(self):
        full = self.full
        names = full.tx_bytes.sum if isinstance(full, BarrierSide) and isinstance(full.tx_bytes, TxSum) else (self.endpoint.name,)
        return [endpoint for endpoint in self.partner.manifest.endpoints if endpoint.name in names]

    def _variant_values(self):
        return tuple(dict.fromkeys(key[self.depth_handle] for key in self.partner.variants))

    def declaration(self, stage_id, version, params, launch_register_cap):
        endpoints = []
        for endpoint in self._group():
            protocol = endpoint.protocol
            sharing, init, full_by, empty_by = None, None, None, None
            if protocol.kind == "mbarrier":
                full = protocol.full
                if isinstance(full, SharedMember):
                    sharing = Sharing(member_of=full.member_of)
                    full = self.partner.endpoints[full.member_of].protocol.full
                else:
                    sharing = Sharing(tx_bytes=tuple(full.tx_bytes.sum)) if isinstance(full.tx_bytes, TxSum) else Sharing()
                init = InitContract(protocol.init.materializer, protocol.init.phase,
                                    protocol.init.owner if protocol.init.materializer == "dispatch" else None)
                full_by = full.arrive.by
                empty_by = protocol.empty.arrive.by if protocol.empty else None
            endpoints.append(Endpoint(endpoint.name, _COMPLEMENT[endpoint.direction], self.role,
                sharing=sharing, init=init, full_arrive_by=full_by, empty_arrive_by=empty_by,
                cluster_leader_policy="mma_pair_leader" if self.cta_group == 2 else None))
        variants = Variants(self.depth_handle, self._variant_values()) if self.depth_handle else None
        return StageDeclaration(stage_id, version, {self.role: Role("explicit")}, tuple(endpoints),
                                launch_register_cap, params, variants)

    def ledger_entries(self):
        """Leaf-level package evidence; authored identity/role/params are not contract facts."""
        entries = []
        def source(field, pointer):
            entries.append((field, f"{self.partner.path}#sha256={self.partner.sha256}:{pointer}", "contract"))
        def leaves(value, field, pointer):
            if isinstance(value, dict):
                for key, item in value.items():
                    escaped = key.replace("~", "~0").replace("/", "~1")
                    leaves(item, f"{field}.{key}", f"{pointer}/{escaped}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    leaves(item, f"{field}.{index}", f"{pointer}/{index}")
            else:
                source(field, pointer)
        names = {endpoint.name for endpoint in self._group()}
        indices = {endpoint["name"]: index for index, endpoint in enumerate(self.partner.raw["endpoints"])}
        for index, endpoint in enumerate(self.partner.raw["endpoints"]):
            if endpoint["name"] not in names:
                continue
            field, pointer = f"endpoints.{endpoint['name']}", f"/endpoints/{index}"
            for key, value in endpoint.items():
                # The emitted owner is the caller's role, not the partner's role.
                name = "partner_owner_role" if key == "owner_role" else key
                leaves(value, f"{field}.{name}", f"{pointer}/{key}")
            protocol = endpoint["protocol"]
            if protocol["kind"] == "mbarrier":
                full_pointer = pointer + "/protocol/full"
                if "member_of" in protocol["full"]:
                    source(f"{field}.sharing.member_of", full_pointer + "/member_of")
                    full_pointer = f"/endpoints/{indices[protocol['full']['member_of']]}/protocol/full"
                else:
                    source(f"{field}.sharing.tx_bytes", full_pointer + "/tx_bytes")
                source(f"{field}.full_arrive_by", full_pointer + "/arrive/by")
                for key in ("materializer", "phase"):
                    source(f"{field}.init.{key}", f"{pointer}/protocol/init/{key}")
                if protocol["init"]["materializer"] == "dispatch":
                    source(f"{field}.init.owner", pointer + "/protocol/init/owner")
                if "empty" in protocol:
                    source(f"{field}.empty_arrive_by", pointer + "/protocol/empty/arrive/by")
            if self.cta_group == 2:
                source(f"{field}.cluster_leader_policy", "/topology/cta_group/fixed")
        for key in ("roles", "variants"):
            leaves(self.partner.raw["stage"][key], f"partner.{key}", f"/stage/{key}")
        if self.depth_handle:
            source("variants.depth_handle", f"/endpoints/{indices[self.endpoint.name]}/ring/depth/handle")
            seen = set()
            for index, key in enumerate(self.partner.variants):
                value = key[self.depth_handle]
                if value not in seen:
                    source(f"variants.values.{len(seen)}", f"/stage/variants/{index}/key/{self.depth_handle}")
                    seen.add(value)
        if "topology" in self.partner.raw:
            leaves(self.partner.raw["topology"], "topology", "/topology")
        return entries
