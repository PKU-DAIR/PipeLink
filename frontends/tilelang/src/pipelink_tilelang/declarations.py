"""Author promises only; compiler geometry belongs to capture, never declarations."""
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from collections.abc import Mapping
from typing import Literal


# Numeric facts and their containing records from the frozen probe .facts contract.
# Names/role labels are shared vocabulary, not compiler measurements.
FORBIDDEN_CAPTURE_KEYS = frozenset({
    "captures", "layouts", "branches", "barrier_schedule", "attrs", "allocations",
    "barrier_init", "barrier_init_annotation", "registers", "warp_scope", "offsets",
    "calls", "bindings", "thread_extent", "dyn_shared_memory_buf", "tl.smem_alignment_map",
    "shape", "dtype", "bytes", "scope", "op", "args", "carrier", "base", "offset",
    "source", "logical_shape", "physical_shape", "forward_vars", "forward_index",
    "thread", "thread_min", "thread_count", "thread_size", "replication", "predicate",
    "warp_base", "warps", "set_max_nreg", "iteration", "barrier", "arrivals",
    "transaction_bytes", "guard",
})
# Family C facts are not present in arithmetic TIR, but are equally forbidden.
FORBIDDEN_FAMILY_KEYS = frozenset({
    "warp", "initializer_warp", "physical_initializer_warp", "initializer_physical_warp",
    "extent", "align", "absolute_base", "stage_stride_bytes", "slots", "count",
    "cluster_shape", "cta_group", "sizeof", "alignof", "offsetof", "storage_bytes",
    "tma_bytes", "load_stages", "store_stages", "accumulator_stages", "tile_shape",
    "reuse_smem_c", "overlapping_accum", "unacquired_store_stages",
})


class DeclarationError(ValueError):
    """An author declaration contains a compiler fact or an invalid promise."""


def _require(condition, field, message):
    if not condition:
        raise DeclarationError(f"{field}: {message}")


def _text(value, field):
    _require(isinstance(value, str) and bool(value), field, "expected nonempty text")


def _reject_compiler_facts(value, path="declarations"):
    if is_dataclass(value) and not isinstance(value, type):
        items = ((field.name, getattr(value, field.name)) for field in fields(value))
    elif isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_compiler_facts(item, f"{path}[{index}]")
        return
    else:
        return
    for key, item in items:
        _require(key not in FORBIDDEN_CAPTURE_KEYS | FORBIDDEN_FAMILY_KEYS,
                 f"{path}.{key}", "compiler-supplied fact is forbidden")
        _reject_compiler_facts(item, f"{path}.{key}")


@dataclass(frozen=True, slots=True)
class Role:
    # then/else select the measured WS branch. body is unspecialized arithmetic;
    # family selects an authored family role absent from the arithmetic program.
    branch: Literal["then", "else", "body", "family", "explicit"]

    def __post_init__(self):
        _require(self.branch in ("then", "else", "body", "family", "explicit"), "branch", "unknown selector")


@dataclass(frozen=True, slots=True)
class Sharing:
    member_of: str | None = None
    tx_bytes: Literal["tile"] | tuple[str, ...] = "tile"

    def __post_init__(self):
        if self.member_of is not None:
            _text(self.member_of, "member_of")
            _require(self.tx_bytes == "tile", "tx_bytes", "a shared member contributes its tile")
        elif self.tx_bytes != "tile":
            _require(isinstance(self.tx_bytes, tuple) and bool(self.tx_bytes), "tx_bytes", "expected endpoint-name sum")
            for name in self.tx_bytes:
                _text(name, "tx_bytes")
            _require(len(set(self.tx_bytes)) == len(self.tx_bytes), "tx_bytes", "duplicate sum member")


@dataclass(frozen=True, slots=True)
class InitContract:
    materializer: Literal["dispatch", "stage"]
    phase: int
    owner: str | None = None

    def __post_init__(self):
        _require(self.materializer in ("dispatch", "stage"), "materializer", "unknown materializer")
        _require(type(self.phase) is int and self.phase in (0, 1), "phase", "expected protocol phase bit")
        if self.owner is not None:
            _text(self.owner, "owner")
        _require(self.materializer != "stage" or self.owner is None, "owner", "stage initializer owner is linker-derived")


@dataclass(frozen=True, slots=True)
class Endpoint:
    name: str
    direction: Literal["produce", "consume", "provide", "require"]
    owner_role: str
    sharing: Sharing | None = None
    init: InitContract | None = None
    cluster_leader_policy: Literal["mma_pair_leader"] | None = None
    full_arrive_by: str | None = None
    empty_arrive_by: str | None = None

    def __post_init__(self):
        _text(self.name, "name")
        _text(self.owner_role, "owner_role")
        _require(self.direction in ("produce", "consume", "provide", "require"), "direction", "unknown endpoint direction")
        _require(self.sharing is None or type(self.sharing) is Sharing, "sharing", "expected Sharing")
        _require(self.init is None or type(self.init) is InitContract, "init", "expected InitContract")
        _require(self.cluster_leader_policy in (None, "mma_pair_leader"), "cluster_leader_policy", "unknown policy")
        if self.full_arrive_by is not None:
            _text(self.full_arrive_by, "full_arrive_by")
        if self.empty_arrive_by is not None:
            _text(self.empty_arrive_by, "empty_arrive_by")


@dataclass(frozen=True, slots=True)
class ParamField:
    name: str
    cpp_type: str

    def __post_init__(self):
        _text(self.name, "name")
        _text(self.cpp_type, "cpp_type")


@dataclass(frozen=True, slots=True)
class ParamsSpec:
    name: str
    fields: tuple[ParamField, ...]
    # Authored alignas, not a measured sizeof/alignof/offsetof result.
    alignas: int

    def __post_init__(self):
        _text(self.name, "params.name")
        _require(type(self.alignas) is int and self.alignas > 0 and self.alignas & (self.alignas - 1) == 0,
                 "alignas", "expected positive power of two")
        _require(isinstance(self.fields, tuple) and all(type(f) is ParamField for f in self.fields), "fields", "expected ParamField tuple")
        _require(len({f.name for f in self.fields}) == len(self.fields), "fields", "duplicate parameter")


@dataclass(frozen=True, slots=True)
class Variants:
    depth_handle: str
    values: tuple[int, ...]

    def __post_init__(self):
        _text(self.depth_handle, "depth_handle")
        _require(isinstance(self.values, tuple) and bool(self.values) and all(type(v) is int and v > 0 for v in self.values),
                 "values", "expected declared positive depth choices")
        _require(len(set(self.values)) == len(self.values), "values", "duplicate variant")


@dataclass(frozen=True, slots=True)
class StageDeclaration:
    stage_id: str
    version: str
    roles: Mapping[str, Role]
    endpoints: tuple[Endpoint, ...]
    launch_register_cap: int
    params: ParamsSpec
    variants: Variants | None = None

    def __post_init__(self):
        _reject_compiler_facts(self)
        _text(self.stage_id, "stage_id")
        _text(self.version, "version")
        _require(isinstance(self.roles, Mapping) and bool(self.roles), "roles", "expected named roles")
        for name, role in self.roles.items():
            _text(name, "roles.name")
            _require(type(role) is Role, f"roles.{name}", "expected Role")
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))
        _require(isinstance(self.endpoints, tuple) and all(type(e) is Endpoint for e in self.endpoints), "endpoints", "expected Endpoint tuple")
        names = {e.name for e in self.endpoints}
        _require(len(names) == len(self.endpoints), "endpoints", "duplicate endpoint")
        for endpoint in self.endpoints:
            # A consume-side contract may name a role exported by the peer stage.
            _text(endpoint.owner_role, "owner_role")
            if endpoint.sharing is not None:
                sharing = endpoint.sharing
                if sharing.member_of is not None:
                    _require(sharing.member_of in names and sharing.member_of != endpoint.name, "member_of", "unknown or self member")
                    leader = next(e for e in self.endpoints if e.name == sharing.member_of)
                    _require(leader.sharing is not None and isinstance(leader.sharing.tx_bytes, tuple)
                             and endpoint.name in leader.sharing.tx_bytes, "member_of", "leader must sum this member")
                elif isinstance(sharing.tx_bytes, tuple):
                    _require(endpoint.name in sharing.tx_bytes and set(sharing.tx_bytes) <= names,
                             "tx_bytes", "sum must include this endpoint and known members")
        _require(type(self.launch_register_cap) is int and 0 < self.launch_register_cap <= 255, "launch_register_cap", "invalid launch cap")
        _require(type(self.params) is ParamsSpec, "params", "expected ParamsSpec")
        _require(self.variants is None or type(self.variants) is Variants, "variants", "expected Variants or no keyed variants")


@dataclass(frozen=True, slots=True)
class Declarations:
    """One recipe's independently exported stages; validated at construction."""
    stages: tuple[StageDeclaration, ...]

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        _reject_compiler_facts(self)
        _require(isinstance(self.stages, tuple) and bool(self.stages)
                 and all(type(stage) is StageDeclaration for stage in self.stages), "stages", "expected StageDeclaration tuple")
        _require(len({stage.stage_id for stage in self.stages}) == len(self.stages), "stage_id", "duplicate stage")
