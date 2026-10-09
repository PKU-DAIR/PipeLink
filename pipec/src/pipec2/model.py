"""Shared contract types for PipeLink ABI v2.

W2 entry points::

    protocol.expand(channel_name, producer: Endpoint, consumer: Endpoint,
                    stages, pipeline, canon: dict[str, CanonEndpoint], profile)
        -> ProtocolExpansion
    solve.solve(pipeline: Pipeline, stages: list[StageManifest], canon,
                expansions: list[ProtocolExpansion], profile) -> Solution
    stubgen.generate_stub(solution: Solution, pipeline, stages, expansions) -> str
    dispatchgen.generate_dispatch(solution, pipeline, stages, expansions,
                                  params: ParamsLayout) -> str
    dispatchgen.params_layout(stages, profile) -> ParamsLayout
    link.link(pipeline_path, out_dir) -> LinkResult

The launcher consumes only ``LinkResult`` and blob files.

Serialization is deterministic. Mapping keys are lexical; stages sort by id,
channels/endpoints and protocol expansions by name, handles by name, claims by
``(domain, handle)``, and failures by ``(clause_id, path)``. Other set-like
collections are sorted by their complete canonical keys. The solver optimizes
lexicographic tiers (depths, registers, per-physical-domain high-water marks,
variants, finite values) and publishes the resulting model's bases; each tier's
status is recorded in ``Solution.objective``. Input manifest, stage, and
mapping order must not affect output.
``canonical_dumps`` additionally sorts all mapping keys, uses compact separators,
and appends one LF.

``SymbolTable`` stores three sorted tuples of names and an ``info`` mapping whose
keys exactly cover their union. Each ``SymbolInfo`` records the symbol kind and
its strong, weak, or undefined binding.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import re
from types import UnionType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from .layout import Layout


CLAUSES: dict[str, str] = {}
CLAUSE_PHASES: dict[str, str] = {}
CLAUSE_IDS = CLAUSES.keys()
_CLAUSE_ID_RE = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\Z")
_CLAUSE_PHASES = frozenset({"load", "pack", "layout", "protocol", "solve", "link", "launch"})


def register_clause(clause_id: str, *, phase: str, description: str = "") -> None:
    """Register a clause and its primary validation phase at module import time."""
    if not isinstance(clause_id, str) or _CLAUSE_ID_RE.fullmatch(clause_id) is None:
        raise ValueError(f"invalid clause ID {clause_id!r}")
    if phase not in _CLAUSE_PHASES:
        raise ValueError(f"unknown clause phase {phase!r}")
    if not isinstance(description, str):
        raise TypeError("clause description must be a string")
    if clause_id in CLAUSES:
        raise ValueError(f"duplicate clause ID {clause_id!r}")
    CLAUSES[clause_id] = description
    CLAUSE_PHASES[clause_id] = phase


for _clause_id, _phase, _description in (
    ("L-ALIGN", "solve", "claim bindings satisfy native and physical alignment"),
    ("L-BUDGET", "solve", "solver finds a feasible layout within its deterministic work limit"),
    ("L8", "layout", "paired endpoints have equal canonical physical layout identity"),
    ("L-ENTRYABI", "load", "stage entry ABI is supported by the live linker"),
    ("L-HANDLE", "load", "handle declarations and references are complete and consistent"),
    ("L-HOSTCHAN", "link", "host declarations and selected resource contracts agree"),
    ("L-HOSTABI", "launch", "native host provider ABI agrees with the declared inventory"),
    ("L-LINK", "link", "artifact linking and dispatch generation complete successfully"),
    ("L-OVERLAP", "solve", "physical intervals do not overlap without valid alias or lifetime exclusion"),
    ("L-OWNER", "solve", "every allocation handle has exactly one owning claim"),
    ("L-PAIR", "link", "each channel has one compatible producer and consumer"),
    ("L-PARAMS", "link", "stage and aggregate parameter layouts satisfy the target ABI"),
    ("L-PHASE", "solve", "claim phases and pairwise exclusions are finite and valid"),
    ("L-PROTO", "protocol", "paired endpoint protocols expand consistently"),
    ("L-REGS", "solve", "register groups and the register pool are legal"),
    ("L-SYM", "link", "artifact symbols and normalized prefixes are unique"),
    ("L-TOPO", "solve", "stage topology constraints have a legal intersection"),
    ("P-IMPORT", "pack", "artifact imports are declared PipeLink symbols"),
    ("T-CONTRACT", "load", "tensor contracts and explicit connection references are valid"),
    ("T-FIT", "pack", "enumerated maps fit exact named-coordinate expressions"),
    ("T-MODEL", "layout", "finite tensor execution models admit the declared contracts"),
    ("T-PLAN", "solve", "bounded tensor synthesis records exact outcomes and limits"),
    ("T-LOWER", "link", "tensor lowering preserves the plan and frozen import declaration"),
    ("T-NATIVE", "link", "tensor adapter compilation and native resources satisfy the target"),
):
    register_clause(_clause_id, phase=_phase, description=_description)
del _clause_id, _phase, _description


_T = TypeVar("_T", bound="_JsonMixin")


class _JsonMixin:
    """Strict dataclass JSON round-trip shared by the model types."""

    def to_json_obj(self) -> dict[str, Any]:
        return _canonicalize_model(self, _encode_dataclass(self))

    @classmethod
    def from_json_obj(cls: type[_T], obj: dict[str, Any]) -> _T:
        return _decode_dataclass(cls, obj)


def _encode_dataclass(value: Any) -> dict[str, Any]:
    return {
        field.name: _encode_value(getattr(value, field.name))
        for field in fields(value)
        if not (field.metadata.get("omit_none") and getattr(value, field.name) is None)
    }


def _encode_value(value: Any) -> Any:
    if is_dataclass(value):
        if isinstance(value, _JsonMixin):
            return value.to_json_obj()
        from .layout import Layout

        if isinstance(value, Layout):
            return value.to_json_obj()
        return _encode_dataclass(value)
    if isinstance(value, dict):
        return {key: _encode_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_encode_value(item) for item in sorted(value)]
    return value


def _decode_dataclass(cls: type[_T], obj: Any) -> _T:
    if not isinstance(obj, dict):
        raise TypeError(f"{cls.__name__} JSON value must be an object")
    model_fields = fields(cls)
    expected = {field.name for field in model_fields}
    optional = {field.name for field in model_fields if field.metadata.get("omit_none")}
    signal = cls is Endpoint and obj.get("space") == "none"
    if signal:
        expected -= {"tile", "layout"}
    actual = set(obj)
    if expected - actual - optional or actual - expected:
        missing = sorted(expected - actual - optional)
        extra = sorted(actual - expected)
        raise ValueError(f"{cls.__name__} field mismatch: missing={missing}, extra={extra}")
    if cls is Endpoint:
        from .layout import Layout, load_layout

        hints = get_type_hints(cls, localns={"Layout": Layout})
    else:
        hints = get_type_hints(cls)
    return cls(**{
        field.name: (
            Layout(space="none") if signal and field.name == "layout"
            else Tile("none", ()) if signal and field.name == "tile"
            else load_layout(obj[field.name], obj["space"], "$.layout")
            if cls is Endpoint and field.name == "layout"
            else _decode_value(hints[field.name], obj.get(field.name))
        )
        for field in model_fields
    })


def _decode_value(annotation: Any, value: Any) -> Any:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is UnionType or origin is Union:
        last_error: Exception | None = None
        for choice in args:
            if choice is type(None) and value is None:
                return None
            if choice is type(None):
                continue
            try:
                return _decode_value(choice, value)
            except (TypeError, ValueError, KeyError) as exc:
                last_error = exc
        raise TypeError(f"value {value!r} does not match {annotation!r}") from last_error

    if origin is Literal:
        if value not in args:
            raise ValueError(f"expected one of {args!r}, got {value!r}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise TypeError("expected JSON array")
        return [_decode_value(args[0], item) for item in value]
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError("expected JSON array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode_value(args[0], item) for item in value)
        if len(value) != len(args):
            raise ValueError(f"expected tuple of length {len(args)}, got {len(value)}")
        return tuple(_decode_value(item_type, item) for item_type, item in zip(args, value))
    if origin in (set, frozenset):
        if not isinstance(value, list):
            raise TypeError("expected JSON array")
        return origin(_decode_value(args[0], item) for item in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError("expected JSON object")
        key_type, value_type = args
        return {
            _decode_value(key_type, key): _decode_value(value_type, item)
            for key, item in value.items()
        }
    if isinstance(annotation, type) and is_dataclass(annotation):
        if issubclass(annotation, _JsonMixin):
            return annotation.from_json_obj(value)
        return _decode_dataclass(annotation, value)
    if annotation in (str, int, bool, float):
        if type(value) is not annotation:
            raise TypeError(f"expected {annotation.__name__}, got {type(value).__name__}")
        return value
    if annotation is type(None):
        if value is not None:
            raise TypeError("expected null")
        return None
    return value


@dataclass(frozen=True)
class ClauseFailure(_JsonMixin):
    clause_id: str
    path: str
    message: str
    detail: dict[str, Any]

    def __post_init__(self) -> None:
        if self.clause_id not in CLAUSES:
            raise ValueError(f"unknown clause ID {self.clause_id!r}")


class PipeLinkError(RuntimeError):
    """Base for deterministic failures produced by ABI v2 phases."""

    failures: list[ClauseFailure]

    def __init__(self, failures: list[ClauseFailure]) -> None:
        self.failures = sorted(failures, key=lambda item: (item.clause_id, item.path))
        text = "; ".join(
            f"[{failure.clause_id}] {failure.path}: {failure.message}" for failure in self.failures
        )
        super().__init__(text or self.__class__.__name__)


class SchemaError(PipeLinkError):
    pass


class ProfileError(PipeLinkError):
    pass


class LayoutError(PipeLinkError):
    pass


class PackError(PipeLinkError):
    pass


class SolveError(PipeLinkError):
    pass


class LinkError(PipeLinkError):
    pass


@dataclass(frozen=True)
class Handle(_JsonMixin):
    name: str
    cls: str
    binding: str
    domain: str | None
    values: list[int] | None
    owner: str


@dataclass(frozen=True)
class Role(_JsonMixin):
    entry: str
    warps: int


@dataclass(frozen=True)
class Export(_JsonMixin):
    symbol: str
    decl: str
    decl_sha256: str


@dataclass(frozen=True)
class Variant(_JsonMixin):
    key: dict[str, int]
    index: int
    entry: str | None


@dataclass(frozen=True)
class MathMode(_JsonMixin):
    """nvcc floating-point codegen switches a stage was compiled with. LTO code generation
    happens at the final link, which must apply them; nvJitLink does not take them from the IR."""
    ftz: bool
    prec_div: bool
    prec_sqrt: bool
    fmad: bool


# nvcc's defaults: what a stage without recorded switches was compiled with.
NVCC_DEFAULT_MATH = MathMode(ftz=False, prec_div=True, prec_sqrt=True, fmad=True)


@dataclass(frozen=True)
class TargetStamp(_JsonMixin):
    schema: str
    profile: str
    lto_arch: str
    toolkit: str
    host_abi: str | None = field(default=None, metadata={"omit_none": True})
    math: MathMode | None = field(default=None, metadata={"omit_none": True})


@dataclass(frozen=True)
class ClusterConstraints(_JsonMixin):
    x_multiple_of: int
    max: tuple[int, int, int]


@dataclass(frozen=True)
class SchedulerStrategy(_JsonMixin):
    id: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class StageTopology(_JsonMixin):
    cta_group: int
    cluster_compiled: tuple[int, int, int] | Literal["dynamic"]
    cluster_launch: ClusterConstraints
    multicast: list[str]
    persistent: bool
    scheduler: str
    scheduler_strategy: SchedulerStrategy | None = field(default=None, metadata={"omit_none": True})


@dataclass(frozen=True)
class ParamsDecl(_JsonMixin):
    bytes: int
    align: int
    type_name: str
    layout_probe_sha256: str


@dataclass(frozen=True)
class DescriptorDecl(_JsonMixin):
    bytes: int
    align: int


@dataclass(frozen=True)
class Tile(_JsonMixin):
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Depth(_JsonMixin):
    fixed: int | None
    handle: str | None

    def __post_init__(self) -> None:
        if (self.fixed is None) == (self.handle is None):
            raise ValueError("Depth requires exactly one of fixed or handle")


@dataclass(frozen=True)
class Ring(_JsonMixin):
    depth: Depth
    stage_stride_bytes: int


@dataclass(frozen=True)
class Arrive(_JsonMixin):
    count: int | str
    by: str


@dataclass(frozen=True)
class TxSum(_JsonMixin):
    sum: list[str]


@dataclass(frozen=True)
class BarrierStorage(_JsonMixin):
    base: str
    offset_bytes: int
    stride_bytes: int


@dataclass(frozen=True)
class BarrierSide(_JsonMixin):
    slots: str
    arrive: Arrive
    tx_bytes: int | Literal["tile"] | TxSum
    native: BarrierStorage | None = field(default=None, metadata={"omit_none": True})


@dataclass(frozen=True)
class SharedMember(_JsonMixin):
    member_of: str
    tx_bytes: int | Literal["tile"]


@dataclass(frozen=True)
class Init(_JsonMixin):
    owner: str | None = field(metadata={"omit_none": True})
    phase: int | str
    materializer: Literal["stage", "dispatch"]
    warp: int | None = field(default=None, metadata={"omit_none": True})


@dataclass(frozen=True)
class CallDecl(_JsonMixin):
    export: str
    signature: str
    decl: str
    decl_sha256: str


@dataclass(frozen=True)
class NamedBarrierDecl(_JsonMixin):
    id_handle: str
    count: int


@dataclass(frozen=True)
class CounterDecl(_JsonMixin):
    slot_handle: str
    release_op: str
    acquire_op: str


@dataclass(frozen=True)
class Protocol(_JsonMixin):
    kind: str
    full: BarrierSide | SharedMember | None
    empty: BarrierSide | None
    init: Init | None
    release: str | None
    call: CallDecl | None
    named_barrier: NamedBarrierDecl | None
    counter: CounterDecl | None


@dataclass(frozen=True)
class Endpoint(_JsonMixin):
    name: str
    direction: str
    owner_role: str | None
    space: str
    scope: str
    proxy: str
    tile: Tile
    layout: Layout
    ring: Ring | None
    protocol: Protocol
    binds: dict[str, str]
    # Provided-call exports of the same stage whose bodies use this endpoint; when every one of
    # them is unreferenced in a pipeline, the endpoint is dropped with them (None: always live).
    exports: list[str] | None = field(default=None, metadata={"omit_none": True})


@dataclass(frozen=True)
class Extent(_JsonMixin):
    const: int
    per: dict[str, int]
    table: dict[str, int]


_INTERVAL_DOMAINS = frozenset(
    {
        "smem",
        "mbarrier",
        "tmem",
        "named_barrier",
        "warp",
        "gmem_counter",
        "smem_page",
        "instr_page",
        "instr_semaphore",
        "instr_scratch",
        "opcode",
    }
)


@dataclass(frozen=True)
class TensorLifetime(_JsonMixin):
    tensor: str
    from_: str
    to: str | None = field(default=None, metadata={"omit_none": True})

    def label(self) -> str:
        return f"tensor:{self.tensor}:{self.from_}..{self.to or 'end'}"

    def to_json_obj(self) -> dict[str, Any]:
        obj = {"tensor": self.tensor, "from": self.from_}
        if self.to is not None:
            obj["to"] = self.to
        return obj

    @classmethod
    def from_json_obj(cls, obj: dict[str, Any]) -> TensorLifetime:
        if not isinstance(obj, dict):
            raise TypeError("TensorLifetime JSON value must be an object")
        if not {"tensor", "from"} <= obj.keys() or obj.keys() - {"tensor", "from", "to"}:
            raise ValueError("TensorLifetime requires tensor/from and optional to")
        return cls(
            _decode_value(str, obj["tensor"]),
            _decode_value(str, obj["from"]),
            _decode_value(str, obj["to"]) if "to" in obj else None,
        )


@dataclass(frozen=True)
class Claim(_JsonMixin):
    domain: str
    handle: str
    lifetime: str | TensorLifetime
    extent: Extent | None
    align: int | None
    offset: int | None
    roles: list[str] | None
    reg_group: str | None
    values: list[int] | None
    absolute_base: int | None
    alias: str | None

    def __post_init__(self) -> None:
        if self.domain == "regs":
            forbidden = {
                "extent": self.extent,
                "align": self.align,
                "offset": self.offset,
                "absolute_base": self.absolute_base,
                "alias": self.alias,
            }
            present = sorted(name for name, value in forbidden.items() if value is not None)
            if present:
                raise ValueError(f"regs claim forbids fields: {', '.join(present)}")
            if not self.roles or not self.values:
                raise ValueError("regs claim requires non-empty roles and values")
        elif self.domain in _INTERVAL_DOMAINS and (self.extent is None or self.align is None):
            raise ValueError(f"{self.domain} claim requires extent and align")


@dataclass(frozen=True)
class StageManifest(_JsonMixin):
    id: str
    version: str
    symbol_prefix: str
    roles: dict[str, Role]
    exports: dict[str, Export]
    entry_signature: str
    entry_abi: Literal["v2"]
    variants: list[Variant]
    endpoints: list[Endpoint]
    claims: list[Claim]
    topology: StageTopology
    params: ParamsDecl
    descriptor: DescriptorDecl | None
    hints: dict[str, Any]
    target: TargetStamp
    symbols: SymbolTable | None = None
    host: dict[str, Any] | None = field(default=None, metadata={"omit_none": True})
    tensor: Any | None = field(default=None, metadata={"omit_none": True})
    # Informational evidence, including optional sections[path] source labels;
    # all present provenance entries participate in canonical serialized digests.
    provenance: dict[str, Any] | None = field(default=None, metadata={"omit_none": True})

    @classmethod
    def from_json_obj(cls, obj: dict[str, Any]) -> StageManifest:
        if isinstance(obj, dict) and "tensor" in obj:
            from pipec2.tensor import contracts

            obj = {**obj, "tensor": contracts.load_stage_tensor(obj["tensor"], "$.tensor")}
        return _decode_dataclass(cls, obj)


@dataclass(frozen=True)
class StageRef(_JsonMixin):
    artifact_path: str
    stage_id: str


@dataclass(frozen=True)
class PipelineDomain(_JsonMixin):
    capacity: int
    page_bytes: int | None


@dataclass(frozen=True)
class PipelineTopology(_JsonMixin):
    persistent: bool
    scheduler: str


@dataclass(frozen=True)
class TensorConnection(_JsonMixin):
    name: str
    export: str
    import_: str

    def to_json_obj(self) -> dict[str, Any]:
        return {"export": self.export, "import": self.import_, "name": self.name}

    @classmethod
    def from_json_obj(cls, obj: dict[str, Any]) -> TensorConnection:
        if not isinstance(obj, dict):
            raise TypeError("TensorConnection JSON value must be an object")
        expected = {"name", "export", "import"}
        actual = set(obj)
        if actual != expected:
            raise ValueError(
                f"TensorConnection field mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        return cls(
            _decode_value(str, obj["name"]),
            _decode_value(str, obj["export"]),
            _decode_value(str, obj["import"]),
        )


@dataclass(frozen=True)
class Pipeline(_JsonMixin):
    name: str
    stages: list[StageRef]
    handles: dict[str, Handle]
    phases: list[str]
    exclusive: list[tuple[str, str]]
    domains: dict[str, PipelineDomain]
    depth_priority: list[str]
    regs_priority: list[str]
    forced_bindings: dict[str, int]
    topology: PipelineTopology
    target: TargetStamp
    tensor_connections: tuple[TensorConnection, ...] = ()
    # Final-link floating-point codegen mode chosen by the integrator; None = stages must agree.
    math: MathMode | None = field(default=None, metadata={"omit_none": True})


@dataclass(frozen=True)
class CanonEndpoint(_JsonMixin):
    addr_fn: tuple[Any, ...]
    elem_bits: int
    swizzle_bits: tuple[int, ...] | None
    align_bits: int | None
    footprint_bits: int
    ring_geom: tuple[str, int] | None
    space: str
    scope: str
    proxy: str
    dtype: str
    tile_shape: tuple[int, ...]
    tv_role: str | None


@dataclass(frozen=True)
class ProtocolExpansion(_JsonMixin):
    channel: str
    kind: str
    tx_bytes_producer: int | None
    tx_bytes_consumer: int | None
    arrive_full: int | None
    arrive_empty: int | None
    init_owner: str | None
    init_materializer: str | None
    init_slots: tuple[str, str | None] | None
    shared_group: tuple[str, tuple[str, ...], dict[str, int]] | None
    callable: tuple[str, str, str] | None
    named_barrier: tuple[str, int] | None
    # (slot handle, release operation, sorted acquire patterns of the channel's consumers)
    counter: tuple[str, str, tuple[str, ...]] | None
    native: dict[str, dict[str, int]] | None = field(default=None, metadata={"omit_none": True})


@dataclass(frozen=True)
class Interval(_JsonMixin):
    handle: str
    lo: int
    hi: int
    phase: str


@dataclass(frozen=True)
class Launch(_JsonMixin):
    threads: int
    regs_cap: int
    dyn_smem_bytes: int
    cluster: tuple[int, int, int] | Literal["dynamic"]


@dataclass(frozen=True)
class RegGroup(_JsonMixin):
    roles: list[str]
    warp_lo: int
    warp_hi: int
    handle: str
    value: int
    op: Literal["inc", "dec", "none"]


@dataclass(frozen=True)
class ObjectiveTier(_JsonMixin):
    """One solver objective tier: its published value and the best value it
    could take given earlier tiers (``bound``). ``proved``: searched to the
    optimum; ``bound_met``: the incumbent already equals the bound;
    ``budget_gap``: the allowance ran out, the incumbent is published."""

    name: str
    value: int
    bound: int
    status: Literal["proved", "bound_met", "budget_gap"]


@dataclass(frozen=True)
class Solution(_JsonMixin):
    depths: dict[str, int]
    regs: dict[str, int]
    native_bases: dict[str, int]
    intervals: dict[str, list[Interval]]
    variants: dict[str, dict[str, int]]
    reg_groups: list[RegGroup]
    warp_map: dict[str, tuple[int, int]]
    launch: Launch
    objective: list[ObjectiveTier]
    rlimit_used: int


ParamsEntry = tuple[str, int, int, int]


@dataclass(frozen=True)
class ParamsLayout(_JsonMixin):
    total_bytes: int
    align: int
    entries: list[ParamsEntry]


SYMBOL_MECHANISM = "nm-external-text+nvjitlink-seeded-v2"


@dataclass(frozen=True)
class SymbolInfo(_JsonMixin):
    kind: str
    binding: str


@dataclass(frozen=True)
class SymbolTable(_JsonMixin):
    defined_strong: tuple[str, ...]
    defined_weak: tuple[str, ...]
    undefined: tuple[str, ...]
    info: dict[str, SymbolInfo]
    mechanism: str = SYMBOL_MECHANISM

    def __post_init__(self) -> None:
        object.__setattr__(self, "defined_strong", tuple(sorted(self.defined_strong)))
        object.__setattr__(self, "defined_weak", tuple(sorted(self.defined_weak)))
        object.__setattr__(self, "undefined", tuple(sorted(self.undefined)))
        names = set(self.defined_strong) | set(self.defined_weak) | set(self.undefined)
        if set(self.info) != names:
            raise ValueError("SymbolTable.info keys must cover exactly all symbol sets")

    def to_manifest(self) -> dict[str, Any]:
        """Encode the externally defined text and unresolved symbol surface."""
        return {
            "mechanism": self.mechanism,
            "defined_strong": list(self.defined_strong),
            "defined_weak": list(self.defined_weak),
            "undefined": {
                name: "variable" if self.info[name].kind == "object" else "function"
                for name in self.undefined
            },
        }

    @classmethod
    def from_manifest(cls, section: Any) -> SymbolTable:
        """Load a packed symbol table; the mechanism records provenance only."""
        if not isinstance(section, dict) or set(section) != {
            "mechanism", "defined_strong", "defined_weak", "undefined"
        }:
            raise ValueError("symbols requires mechanism, defined_strong, defined_weak and undefined")
        if not isinstance(section["mechanism"], str) or not section["mechanism"]:
            raise ValueError("symbols.mechanism must be a non-empty string")
        collections: list[tuple[str, ...]] = []
        info: dict[str, SymbolInfo] = {}
        for field, binding in (("defined_strong", "strong"), ("defined_weak", "weak")):
            names = section[field]
            if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
                raise ValueError(f"symbols.{field} must be an array of non-empty strings")
            if len(names) != len(set(names)):
                raise ValueError(f"symbols.{field} must not contain duplicates")
            collections.append(tuple(names))
            info.update({name: SymbolInfo("func", binding) for name in names})
        undefined = section["undefined"]
        if not isinstance(undefined, dict) or any(
            not isinstance(name, str) or not name or kind not in ("function", "variable")
            for name, kind in undefined.items()
        ):
            raise ValueError("symbols.undefined must map symbol names to function or variable")
        collections.append(tuple(undefined))
        if sum(map(len, collections)) != len(set().union(*collections)):
            raise ValueError("symbols definition and import sets must be disjoint")
        info.update({
            name: SymbolInfo("object" if kind == "variable" else "func", "undefined")
            for name, kind in undefined.items()
        })
        return cls(*collections, info, mechanism=section["mechanism"])


@dataclass(frozen=True)
class LegalitySet(_JsonMixin):
    x_multiple_of: int
    max: tuple[int, int, int]


@dataclass(frozen=True)
class LinkResult(_JsonMixin):
    kernel_symbol: str
    threads: int
    regs_cap: int
    dyn_smem_bytes: int
    cluster: tuple[int, int, int] | LegalitySet
    params: ParamsLayout
    bindings: dict[str, tuple[int, Literal["solved", "forced"]]]
    variants: dict[str, dict[str, int]]
    intervals: dict[str, list[Interval]]
    protocol_expansions: list[ProtocolExpansion]
    observed: tuple[int, int, int]
    input_hashes: dict[str, str]
    generated_hashes: dict[str, str]
    sass_report: dict[str, Any] | None
    host: dict[str, Any] | None = field(default=None, metadata={"omit_none": True})
    solver: dict[str, Any] | None = field(default=None, metadata={"omit_none": True})
    tensor: Any | None = field(default=None, metadata={"omit_none": True})
    # Unreferenced provided exports and the endpoints they own, dropped before pairing.
    dropped_endpoints: list[dict[str, str]] | None = field(default=None, metadata={"omit_none": True})
    # The pipeline's `target.math` codegen mode and the packed modes of stages it overrides.
    math_override: dict[str, Any] | None = field(default=None, metadata={"omit_none": True})


def _canonicalize_model(instance: Any, obj: dict[str, Any]) -> dict[str, Any]:
    if isinstance(instance, Handle) and obj["values"] is not None:
        obj["values"] = sorted(obj["values"])
    elif isinstance(instance, StageTopology):
        obj["multicast"] = sorted(obj["multicast"])
    elif isinstance(instance, TxSum):
        obj["sum"] = sorted(obj["sum"])
    elif isinstance(instance, Endpoint) and instance.space == "none":
        del obj["tile"], obj["layout"]
    elif isinstance(instance, Claim):
        if obj["roles"] is not None:
            obj["roles"] = sorted(obj["roles"])
        if obj["values"] is not None:
            obj["values"] = sorted(obj["values"])
    elif isinstance(instance, StageManifest):
        obj["variants"] = sorted(
            obj["variants"],
            key=lambda item: (tuple(item["key"].items()), item["index"], item["entry"] or ""),
        )
        obj["endpoints"] = sorted(obj["endpoints"], key=lambda item: item["name"])
        obj["claims"] = sorted(obj["claims"], key=lambda item: (item["domain"], item["handle"]))
    elif isinstance(instance, Pipeline):
        obj["stages"] = sorted(obj["stages"], key=lambda item: item["stage_id"])
        obj["phases"] = sorted(obj["phases"])
        obj["exclusive"] = sorted([sorted(pair) for pair in obj["exclusive"]])
        obj["tensor_connections"] = sorted(obj["tensor_connections"], key=lambda item: item["name"])
    elif isinstance(instance, RegGroup):
        obj["roles"] = sorted(obj["roles"])
    elif isinstance(instance, Solution):
        for domain in obj["intervals"]:
            obj["intervals"][domain] = sorted(
                obj["intervals"][domain],
                key=lambda item: (item["handle"], item["lo"], item["hi"], item["phase"]),
            )
        obj["reg_groups"] = sorted(
            obj["reg_groups"],
            key=lambda item: (item["warp_lo"], item["warp_hi"], item["handle"], item["roles"]),
        )
    elif isinstance(instance, ParamsLayout):
        obj["entries"] = sorted(obj["entries"], key=lambda item: item[0])
    elif isinstance(instance, LinkResult):
        for domain in obj["intervals"]:
            obj["intervals"][domain] = sorted(
                obj["intervals"][domain],
                key=lambda item: (item["handle"], item["lo"], item["hi"], item["phase"]),
            )
        obj["protocol_expansions"] = sorted(obj["protocol_expansions"], key=lambda item: item["channel"])
    return {key: obj[key] for key in sorted(obj)}


__all__ = [
    "CLAUSES",
    "CLAUSE_IDS",
    "CLAUSE_PHASES",
    "Arrive",
    "BarrierSide",
    "BarrierStorage",
    "CallDecl",
    "CanonEndpoint",
    "Claim",
    "ClauseFailure",
    "ClusterConstraints",
    "CounterDecl",
    "Depth",
    "DescriptorDecl",
    "Endpoint",
    "Export",
    "Extent",
    "Handle",
    "Init",
    "Interval",
    "Launch",
    "LayoutError",
    "LegalitySet",
    "LinkError",
    "LinkResult",
    "NamedBarrierDecl",
    "PackError",
    "ParamsDecl",
    "ParamsLayout",
    "PipeLinkError",
    "Pipeline",
    "PipelineDomain",
    "PipelineTopology",
    "ProfileError",
    "Protocol",
    "ProtocolExpansion",
    "RegGroup",
    "Ring",
    "Role",
    "SchemaError",
    "SchedulerStrategy",
    "SharedMember",
    "Solution",
    "SolveError",
    "StageManifest",
    "StageRef",
    "StageTopology",
    "SymbolInfo",
    "SymbolTable",
    "TargetStamp",
    "TensorConnection",
    "TensorLifetime",
    "Tile",
    "TxSum",
    "Variant",
    "register_clause",
]
