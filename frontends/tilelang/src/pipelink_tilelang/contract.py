"""Generate an explicit TileLang role from a packed partner, without its headers.

A Channel owns allocations in the current PrimFunc. Handle-sized rings require
``buffer(depth=...)`` before synchronization; fixed rings need no depth argument.
Shared-completion declarations include the complete group, not an invalid partial
sum. TMEM requires the explicitly consumed publication endpoint as ``base=...``.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from pipec2.model import BarrierSide, SharedMember, StageManifest, TxSum
from pipec2.schema import load_stage_manifest
from pipec2.stubgen import _code_mask, _declarator, _split_parameters, declaration_parts

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
        return self.manifest.topology.scheduler_strategy

    @property
    def variants(self):
        return tuple(variant.key for variant in self.manifest.variants)

    @property
    def roles(self):
        """Partner-owned role widths, in warps (never guessed peer widths)."""
        return {name: role.warps for name, role in self.manifest.roles.items()}


_COMPLEMENT = {"produce": "consume", "consume": "produce", "provide": "require", "require": "provide"}
_DTYPES = {"bf16": "bfloat16", **{f"{short}{bits}": f"{long}{bits}"
           for short, long, widths in (("f", "float", (16, 32, 64)),
                                       ("i", "int", (8, 16, 32, 64)),
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
class TensorDemand:
    """Position map: logical coordinates followed by participant and position.

    ``domain`` contains (name, lower, upper) triples; ``mapping`` contains C++
    integer expressions in those names. The frontend emits the sampling lambda.
    """
    domain: tuple[tuple[str, int, int], ...]
    mapping: tuple[str | int, ...]
    site: str = "tensor_import_site"
    participants: str = "pl::all_threads"
    guarantees: tuple[str, ...] = ("enter_once", "converged", "full_warps", "no_early_return")


@dataclass(frozen=True)
class TensorRendezvous:
    event: str
    barrier: str
    cohort: str = "pl::all_threads"


def import_tensor(partner: Partner, export_name: str, *, role: str, symbol: str,
                  demand: TensorDemand, lends, rendezvous, arguments=None) -> TensorImportChannel:
    """Import a named tensor using only its packed identity and call endpoints.

    Lends are dictionaries selecting a stage.hpp helper with ``kind`` and its
    resource facts. Their owner/lifetime defaults are the role and seam events.
    """
    return TensorImportChannel(partner, export_name, role, symbol, demand, lends, rendezvous, arguments)


class TensorImportChannel:
    def __init__(self, partner, export_name, role, symbol, demand, lends, rendezvous, arguments):
        self.partner, self.role, self.symbol = partner, role, symbol
        self.export_index, self.export = next((index, item) for index, item in enumerate(partner.raw["tensor"]["exports"])
                                              if item["name"] == export_name)
        self.demand, self.lends, self.rendezvous = demand, tuple(lends), tuple(rendezvous)
        for lend in self.lends:
            # Link-resolved handle accessors live in pipec's namespace; pack admits imports by that prefix.
            if lend["kind"] == "smem" and not lend["handle"].startswith("pipelink_"):
                raise ValueError(f"{symbol}: lend {lend['name']!r} handle {lend['handle']!r} must start with 'pipelink_'")
        self.name = symbol
        calls = self.export["calls"]
        self.produce_endpoint = partner.endpoints[calls["produce"]]
        self.release_endpoint = partner.endpoints[calls["release"]] if calls["release"] else None
        self.produce_name = self.produce_endpoint.name
        self.release_name = self.release_endpoint.name if self.release_endpoint is not None else None
        self.ready = self.export["readiness"]["after"]
        self.done = self.export["readiness"]["reuse_after"]
        self._buffers = {}
        self._builder = None
        self.arguments = arguments
        self.args_param = f"{symbol}_args"
        self.argument_type = None
        self.type_members = None
        if arguments is not None:
            endpoint = self.produce_endpoint
            declaration = endpoint.protocol.call.decl
            exported = partner.raw["stage"]["exports"][endpoint.protocol.call.export]["symbol"]
            _, parameters, _ = declaration_parts(declaration, exported)
            parameter = _split_parameters(parameters)[0].strip()
            self.argument_type = parameter.removeprefix("const ").removesuffix("*").strip()
            # PL_TYPE publishes a guarded struct definition in the call
            # declaration. Walk its balanced body, ignoring comments/literals.
            mask = _code_mask(declaration)
            tokens = list(re.finditer(r"[A-Za-z_][A-Za-z0-9_]*|\S", mask))
            short_name = self.argument_type.rsplit("::", 1)[-1]
            for index, token in enumerate(tokens[:-2]):
                if (token.group() == "struct" and tokens[index + 1].group() == short_name
                        and tokens[index + 2].group() == "{"):
                    start = tokens[index + 2].end()
                    depth = 1
                    for closing in tokens[index + 3:]:
                        depth += (closing.group() == "{") - (closing.group() == "}")
                        if depth == 0:
                            self.type_members = declaration[start:closing.start()].strip()
                            break
                    break
            if self.type_members is None:
                raise ValueError(f"{export_name}: produce arguments need a published PL_TYPE struct")

    def call_declaration(self, endpoint, symbol):
        """Rename only the callable, retaining its published type prelude."""
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

    def args(self, pointer):
        """Give a TIR handle its manifest-published C++ argument pointer type."""
        import tilelang.language as T
        return T.call_extern("handle", f"{self.symbol}_args_cast", pointer)

    def scratch(self, name):
        """Allocate the caller's declared smem lend, addressed by its link handle."""
        import tilelang.language as T
        from tilelang import tvm
        builder = tvm.script.ir_builder.IRBuilder.current()
        if self._builder is None or not builder.same_as(self._builder):
            self._builder, self._buffers = builder, {}
        if name not in self._buffers:
            lend = next(item for item in self.lends if item["name"] == name and item["kind"] == "smem")
            self._buffers[name] = (T.alloc_shared((lend["extent"],), "uint8"), lend["handle"])
        return self._buffers[name][0]

    @property
    def external(self):
        return [ExternalStorage(buffer.name, handle, "smem") for buffer, handle in self._buffers.values()]


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
        self.dtype = endpoint.tile.dtype
        self.protocol, self.binds = endpoint.protocol, endpoint.binds
        self.depth = endpoint.ring.depth.fixed if endpoint.ring else 1
        self.depth_handle = endpoint.ring.depth.handle if endpoint.ring else None
        self.cta_group = partner.manifest.topology.cta_group
        self.cluster = partner.manifest.topology.cluster_compiled
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
        """
        import tilelang.language as T
        from .layout import tmem_shape, to_tilelang
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
        column_handle = None
        if self.space == "smem":
            shape = self.tile.shape if selected is None else (selected, *self.tile.shape)
            buffer = T.alloc_shared(shape, dtype)
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
                                                          column_handle=column_handle)))
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
        leaves(self.partner.raw["topology"], "topology", "/topology")
        return entries
