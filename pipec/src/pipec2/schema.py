from __future__ import annotations

import itertools
import re
from typing import Any, NoReturn

from .decls import declaration_digest
from .layout import Layout, load_layout
from .model import (
    Arrive,
    BarrierSide,
    BarrierStorage,
    CallDecl,
    Claim,
    ClauseFailure,
    ClusterConstraints,
    CounterDecl,
    Depth,
    DescriptorDecl,
    Endpoint,
    Export,
    Extent,
    Handle,
    Init,
    NamedBarrierDecl,
    ParamsDecl,
    Pipeline,
    PipelineDomain,
    PipelineTopology,
    Protocol,
    Ring,
    Role,
    SchemaError,
    SchedulerStrategy,
    SharedMember,
    StageManifest,
    StageRef,
    StageTopology,
    SymbolTable,
    MathMode,
    TargetStamp,
    TensorConnection,
    TensorLifetime,
    Tile,
    TxSum,
    Variant,
)
from .profile import Profile, ProfileError, get_profile


_SCHEMA = "2.0-draft"
_HANDLE_RE = re.compile(r"pipelink_[A-Za-z_][A-Za-z0-9_]*\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_STAGE_ID_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)+\Z")
_PHASE_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ARRIVE_TERM = r"(?:[1-9][0-9]*\s*\*\s*)?(?:threads|warps)\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)"
_ARRIVE_EXPR_RE = re.compile(rf"\s*{_ARRIVE_TERM}(?:\s*\+\s*{_ARRIVE_TERM})*\s*\Z")
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
_HANDLE_CLASSES = frozenset({"depth", "regs", "base", "variant"})
_BINDINGS = frozenset({"bind", "fixed", "launch"})
# Scheduler admission is per profile (Profile.schedulers / scheduler_strategies);
# this registry only pins each strategy's Params inputs.
# CUTLASS PersistentTileSchedulerSm100::swizzle_and_rasterize consumes these
# Params values; gridDim.x/y and the CLC response coordinates are device inputs.
# log_swizzle_size_ is not used by this pinned decode; divmod_swizzle_size_.divisor is.
_SCHEDULER_STRATEGY_INPUTS = {
    "clc-sm100-v1": ("swizzle_size", "raster_order", "cluster_shape_m", "cluster_shape_n"),
}
_SPACES = frozenset({"smem", "tmem", "reg", "gmem", "none"})
_SCOPES = frozenset({"cta", "cluster", "grid", "system"})
# Counter endpoints: the release operation fixes the endpoint scope. Each scope admits both
# PTX acquire patterns: an acquiring load, or a relaxed poll followed by an acquire fence.
_COUNTER_RELEASE_SCOPES = {"red.release.gpu": "grid", "red.release.sys": "system"}
_COUNTER_ACQUIRES = {
    "grid": ("ld.acquire.gpu", "ld.relaxed.gpu+fence.acquire.gpu"),
    "system": ("ld.acquire.sys", "ld.relaxed.sys+fence.acquire.sys"),
}
_PROXIES = frozenset({"async", "generic"})


def _reject(clause_id: str, path: str, message: str, **detail: Any) -> NoReturn:
    raise SchemaError([ClauseFailure(clause_id, path, message, detail)])


def _object(
    value: Any,
    path: str,
    clause_id: str,
    *,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _reject(clause_id, path, "expected an object with string keys")
    required_keys = set(required)
    allowed = required_keys | set(optional)
    missing = sorted(required_keys - value.keys())
    if missing:
        _reject(clause_id, path, f"missing required key {missing[0]!r}")
    unknown = sorted(set(value) - allowed)
    if unknown:
        key = unknown[0]
        _reject(clause_id, f"{path}.{key}", f"unknown key {key!r}")
    return value


def _array(value: Any, path: str, clause_id: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        _reject(clause_id, path, "expected an array")
    if nonempty and not value:
        _reject(clause_id, path, "array must not be empty")
    return value


def _string(value: Any, path: str, clause_id: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        _reject(clause_id, path, "expected a non-empty string")
    return value


def _identifier(value: Any, path: str, clause_id: str) -> str:
    result = _string(value, path, clause_id)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        _reject(clause_id, path, "expected a C-linkage identifier")
    return result


def _handle_name(value: Any, path: str) -> str:
    result = _string(value, path, "L-HANDLE")
    if _HANDLE_RE.fullmatch(result) is None:
        _reject("L-HANDLE", path, "expected a canonical pipelink_* handle")
    return result


def _integer(
    value: Any,
    path: str,
    clause_id: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if type(value) is not int:
        _reject(clause_id, path, "expected an integer")
    if positive and value <= 0:
        _reject(clause_id, path, "value must be strictly positive")
    if nonnegative and value < 0:
        _reject(clause_id, path, "value must be non-negative")
    return value


def _boolean(value: Any, path: str, clause_id: str) -> bool:
    if type(value) is not bool:
        _reject(clause_id, path, "expected a boolean")
    return value


def _power_of_two(value: Any, path: str, clause_id: str) -> int:
    result = _integer(value, path, clause_id, positive=True)
    if result & (result - 1):
        _reject(clause_id, path, "value must be a power of two")
    return result


def _unique_strings(
    value: Any,
    path: str,
    clause_id: str,
    *,
    nonempty: bool = False,
    identifiers: bool = False,
) -> list[str]:
    items = _array(value, path, clause_id, nonempty=nonempty)
    result = [
        (_identifier(item, f"{path}[{index}]", clause_id) if identifiers else _string(item, f"{path}[{index}]", clause_id))
        for index, item in enumerate(items)
    ]
    if len(set(result)) != len(result):
        _reject(clause_id, path, "array entries must be unique")
    return result


def _unique_ints(value: Any, path: str, clause_id: str, *, positive: bool = False) -> list[int]:
    items = _array(value, path, clause_id, nonempty=True)
    result = [
        _integer(item, f"{path}[{index}]", clause_id, positive=positive)
        for index, item in enumerate(items)
    ]
    if len(set(result)) != len(result):
        _reject(clause_id, path, "array entries must be unique")
    return result


def _sha256(value: Any, path: str, clause_id: str) -> str:
    result = _string(value, path, clause_id)
    if _SHA256_RE.fullmatch(result) is None:
        _reject(clause_id, path, "expected a lowercase SHA-256 hex digest")
    return result


def _shape3(value: Any, path: str) -> tuple[int, int, int]:
    items = _array(value, path, "L-TOPO")
    if len(items) != 3:
        _reject("L-TOPO", path, "expected a three-element shape")
    return tuple(
        _integer(item, f"{path}[{index}]", "L-TOPO", positive=True)
        for index, item in enumerate(items)
    )  # type: ignore[return-value]


def _target_stamp(value: Any, path: str = "$.pipelink_abi") -> tuple[TargetStamp, Profile]:
    abi = _object(value, path, "L-TOPO", required=("schema", "target"))
    schema = _string(abi["schema"], f"{path}.schema", "L-TOPO")
    if schema not in {_SCHEMA, "2.2", "3.0"}:
        _reject("L-TOPO", f"{path}.schema", "unsupported artifact schema", actual=schema)
    target = _object(
        abi["target"],
        f"{path}.target",
        "L-TOPO",
        required=("profile", "lto_arch", "toolkit"),
        optional=("host_abi", "math"),
    )
    profile_name = _string(target["profile"], f"{path}.target.profile", "L-TOPO")
    try:
        profile = get_profile(profile_name)
    except ProfileError as exc:
        failures = [
            ClauseFailure(failure.clause_id, f"{path}.target.profile", failure.message, failure.detail)
            for failure in exc.failures
        ]
        raise SchemaError(failures) from exc
    lto_arch = _string(target["lto_arch"], f"{path}.target.lto_arch", "L-TOPO")
    toolkit = _string(target["toolkit"], f"{path}.target.toolkit", "L-TOPO")
    # D-ARCH: stages built on the explicit compute_*a/lto_*a adapter route carry
    # that arch honestly next to the neutral lto_*; composition compatibility is
    # checked at link (T-NATIVE), never silently substituted.
    admitted_lto = {profile.arch.lto_base, profile.arch.lto}
    if lto_arch not in admitted_lto:
        _reject(
            "L-TOPO",
            f"{path}.target.lto_arch",
            "target lto_arch does not match profile",
            expected=sorted(admitted_lto),
            actual=lto_arch,
        )
    if toolkit != profile.toolkit:
        _reject(
            "L-TOPO",
            f"{path}.target.toolkit",
            "target toolkit does not match profile",
            expected=profile.toolkit,
            actual=toolkit,
        )
    host_abi = target.get("host_abi")
    if schema != "3.0" and host_abi is not None:
        _reject("L-HOSTCHAN", f"{path}.target", "host declarations require schema 3.0", subreason="inventory")
    math = _math_mode(target["math"], f"{path}.target.math", "L-TOPO") if "math" in target else None
    return TargetStamp(schema, profile_name, lto_arch, toolkit, host_abi, math), profile


def _math_mode(value: Any, path: str, clause_id: str) -> MathMode:
    fields = _object(value, path, clause_id, required=("ftz", "prec_div", "prec_sqrt", "fmad"))
    return MathMode(**{name: _boolean(fields[name], f"{path}.{name}", clause_id) for name in fields})


def _normalized_prefix(stage_id: str) -> str:
    return re.sub(r"[.-]", "_", stage_id.lower()) + "_"


def _roles(value: Any, path: str) -> dict[str, Role]:
    obj = _object_map(value, path, "L-SYM")
    result: dict[str, Role] = {}
    for name in sorted(obj):
        _identifier(name, f"{path}.{name}", "L-SYM")
        role_path = f"{path}.{name}"
        role = _object(obj[name], role_path, "L-SYM", required=("entry", "warps"))
        result[name] = Role(
            _identifier(role["entry"], f"{role_path}.entry", "L-SYM"),
            _integer(role["warps"], f"{role_path}.warps", "L-TOPO", positive=True),
        )
    return result


def _object_map(value: Any, path: str, clause_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) or not key for key in value):
        _reject(clause_id, path, "expected an object with non-empty string keys")
    return value


def _exports(value: Any, path: str) -> dict[str, Export]:
    obj = _object_map(value, path, "L-SYM")
    result: dict[str, Export] = {}
    for name in sorted(obj):
        _identifier(name, f"{path}.{name}", "L-SYM")
        export_path = f"{path}.{name}"
        export = _object(
            obj[name],
            export_path,
            "L-SYM",
            required=("symbol", "decl", "decl_sha256"),
        )
        decl = _string(export["decl"], f"{export_path}.decl", "L-SYM")
        digest = _sha256(export["decl_sha256"], f"{export_path}.decl_sha256", "L-SYM")
        actual = declaration_digest(decl)
        if digest != actual:
            _reject(
                "L-SYM",
                f"{export_path}.decl_sha256",
                "declaration hash does not match normalized UTF-8 text",
                expected=actual,
                actual=digest,
            )
        result[name] = Export(
            _identifier(export["symbol"], f"{export_path}.symbol", "L-SYM"),
            decl,
            digest,
        )
    return result


def _variants(value: Any, path: str, symbol_prefix: str) -> list[Variant]:
    items = _array(value, path, "L-HANDLE", nonempty=True)
    result: list[Variant] = []
    complete_keys: tuple[str, ...] | None = None
    tuples_seen: set[tuple[tuple[str, int], ...]] = set()
    indices_seen: set[int] = set()
    entries_seen: set[str] = set()
    values_by_handle: dict[str, set[int]] = {}

    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        raw = _object(item, item_path, "L-HANDLE", required=("key", "index"), optional=("entry",))
        raw_key = _object_map(raw["key"], f"{item_path}.key", "L-HANDLE")
        key: dict[str, int] = {}
        for name in sorted(raw_key):
            _handle_name(name, f"{item_path}.key.{name}")
            key[name] = _integer(raw_key[name], f"{item_path}.key.{name}", "L-HANDLE")
            values_by_handle.setdefault(name, set()).add(key[name])
        key_names = tuple(sorted(key))
        if complete_keys is None:
            complete_keys = key_names
        elif key_names != complete_keys:
            _reject(
                "L-HANDLE",
                f"{item_path}.key",
                "variant key is incomplete or names a different handle set",
                expected=list(complete_keys),
                actual=list(key_names),
            )
        key_tuple = tuple(sorted(key.items()))
        if key_tuple in tuples_seen:
            _reject("L-HANDLE", f"{item_path}.key", "duplicate complete variant key tuple")
        tuples_seen.add(key_tuple)

        variant_index = _integer(raw["index"], f"{item_path}.index", "L-HANDLE", nonnegative=True)
        if variant_index in indices_seen:
            _reject("L-HANDLE", f"{item_path}.index", "variant indices must be injective")
        indices_seen.add(variant_index)

        entry_value = raw.get("entry")
        entry: str | None
        if entry_value is None:
            entry = None
        else:
            entry = _identifier(entry_value, f"{item_path}.entry", "L-HANDLE")
            if not entry.startswith(symbol_prefix):
                _reject("L-HANDLE", f"{item_path}.entry", "variant entry must carry the stage symbol prefix")
            if entry in entries_seen:
                _reject("L-HANDLE", f"{item_path}.entry", "variant entry symbols must be injective")
            entries_seen.add(entry)
        result.append(Variant(key, variant_index, entry))

    if complete_keys:
        expected_count = 1
        for name in complete_keys:
            expected_count *= len(values_by_handle[name])
        if len(result) != expected_count:
            _reject(
                "L-HANDLE",
                path,
                "variant map is not total over the complete key tuples",
                expected=expected_count,
                actual=len(result),
            )
    return result


def _tile(value: Any, path: str) -> Tile:
    obj = _object(value, path, "L-PAIR", required=("dtype", "shape"))
    shape_items = _array(obj["shape"], f"{path}.shape", "L-PAIR", nonempty=True)
    shape = tuple(
        _integer(item, f"{path}.shape[{index}]", "L-PAIR", positive=True)
        for index, item in enumerate(shape_items)
    )
    return Tile(_string(obj["dtype"], f"{path}.dtype", "L-PAIR"), shape)


def _depth(value: Any, path: str) -> Depth:
    obj = _object(value, path, "L-HANDLE", optional=("fixed", "handle"))
    if ("fixed" in obj) == ("handle" in obj):
        _reject("L-HANDLE", path, "depth requires exactly one of 'fixed' or 'handle'")
    if "fixed" in obj:
        return Depth(_integer(obj["fixed"], f"{path}.fixed", "L-HANDLE", positive=True), None)
    return Depth(None, _handle_name(obj["handle"], f"{path}.handle"))


def _ring(value: Any, path: str, *, signal: bool = False) -> Ring:
    obj = _object(value, path, "L-HANDLE", required=("depth", "stage_stride_bytes"))
    return Ring(
        _depth(obj["depth"], f"{path}.depth"),
        _integer(obj["stage_stride_bytes"], f"{path}.stage_stride_bytes", "L-HANDLE",
                 positive=not signal, nonnegative=signal),
    )


def _arrive(value: Any, path: str) -> Arrive:
    obj = _object(value, path, "L-PROTO", required=("count", "by"))
    count = obj["count"]
    if type(count) is int:
        count = _integer(count, f"{path}.count", "L-PROTO", positive=True)
    elif isinstance(count, str):
        if count != "1" and _ARRIVE_EXPR_RE.fullmatch(count) is None:
            _reject("L-PROTO", f"{path}.count", "invalid arrive count expression")
    else:
        _reject("L-PROTO", f"{path}.count", "arrive count must be an integer or role expression")
    return Arrive(count, _identifier(obj["by"], f"{path}.by", "L-PROTO"))


def _tx_bytes(value: Any, path: str, *, allow_sum: bool) -> int | str | TxSum:
    if type(value) is int:
        return _integer(value, path, "L-PROTO", nonnegative=True)
    if isinstance(value, str):
        if value != "tile":
            _reject("L-PROTO", path, "tx_bytes string must be 'tile'")
        return value
    if allow_sum:
        obj = _object(value, path, "L-PROTO", required=("sum",))
        return TxSum(_unique_strings(obj["sum"], f"{path}.sum", "L-PROTO", nonempty=True))
    _reject("L-PROTO", path, "shared-member tx_bytes must be an integer or 'tile'")


def _barrier_storage(value: Any, path: str) -> BarrierStorage:
    obj = _object(value, path, "L-PROTO", required=("base", "offset_bytes", "stride_bytes"))
    base = _string(obj["base"], f"{path}.base", "L-PROTO")
    if _HANDLE_RE.fullmatch(base) is None:
        _reject("L-PROTO", f"{path}.base", "native base must be a claim handle")
    return BarrierStorage(
        base,
        _integer(obj["offset_bytes"], f"{path}.offset_bytes", "L-PROTO", nonnegative=True),
        _integer(obj["stride_bytes"], f"{path}.stride_bytes", "L-PROTO", positive=True),
    )


def _barrier_full(value: Any, path: str) -> BarrierSide | SharedMember:
    if not isinstance(value, dict):
        _reject("L-PROTO", path, "expected an object")
    member_fields = "member_of" in value
    full_fields = bool({"slots", "arrive"} & set(value))
    if member_fields and full_fields:
        _reject("L-PROTO", path, "member_of is mutually exclusive with full-side slots and arrive")
    if member_fields:
        obj = _object(value, path, "L-PROTO", required=("member_of", "tx_bytes"))
        return SharedMember(
            _string(obj["member_of"], f"{path}.member_of", "L-PROTO"),
            _tx_bytes(obj["tx_bytes"], f"{path}.tx_bytes", allow_sum=False),  # type: ignore[arg-type]
        )
    obj = _object(value, path, "L-PROTO", required=("slots", "arrive", "tx_bytes"), optional=("native",))
    return BarrierSide(
        _handle_name(obj["slots"], f"{path}.slots"),
        _arrive(obj["arrive"], f"{path}.arrive"),
        _tx_bytes(obj["tx_bytes"], f"{path}.tx_bytes", allow_sum=True),  # type: ignore[arg-type]
        _barrier_storage(obj["native"], f"{path}.native") if "native" in obj else None,
    )


def _barrier_side(value: Any, path: str) -> BarrierSide:
    result = _barrier_full(value, path)
    if isinstance(result, SharedMember):
        _reject("L-PROTO", path, "empty barrier side cannot be a shared-completion member")
    return result


def _init(value: Any, path: str) -> Init:
    obj = _object(value, path, "L-PROTO", required=("materializer", "phase"), optional=("owner", "warp"))
    materializer = _string(obj["materializer"], f"{path}.materializer", "L-PROTO")
    if materializer not in {"stage", "dispatch"}:
        _reject("L-PROTO", f"{path}.materializer", "materializer must be 'stage' or 'dispatch'")
    phase = obj["phase"]
    if type(phase) is int:
        phase = _integer(phase, f"{path}.phase", "L-PROTO", nonnegative=True)
    else:
        phase = _string(phase, f"{path}.phase", "L-PROTO")
    warp = _integer(obj["warp"], f"{path}.warp", "L-PROTO", nonnegative=True) if "warp" in obj else None
    owner = _identifier(obj["owner"], f"{path}.owner", "L-PROTO") if "owner" in obj else None
    if owner is None and warp is None:
        _reject("L-PROTO", path, "init requires an owner or a physical initializing warp")
    return Init(owner, phase, materializer, warp)  # type: ignore[arg-type]


def _protocol(value: Any, path: str, *, signal: bool = False) -> Protocol:
    if not isinstance(value, dict):
        _reject("L-PROTO", path, "expected an object")
    kind = _string(value.get("kind"), f"{path}.kind", "L-PROTO")
    if kind == "mbarrier":
        obj = _object(
            value,
            path,
            "L-PROTO",
            required=("kind", "full", "init", "release"),
            optional=("empty",),
        )
        release = _string(obj["release"], f"{path}.release", "L-PROTO")
        if release not in ({"never", "external"} if signal else {"once-per-acquire", "never"}):
            _reject("L-PROTO", f"{path}.release", "unknown mbarrier release policy")
        if ("empty" in obj) != (release == "once-per-acquire"):
            _reject("L-PROTO", f"{path}.empty", "empty side is required exactly for once-per-acquire release")
        full = _barrier_full(obj["full"], f"{path}.full")
        init = _init(obj["init"], f"{path}.init")
        if release == "never" and (
            not isinstance(full, BarrierSide) or isinstance(full.tx_bytes, TxSum)
            or (not signal and init.materializer != "stage")
        ):
            _reject("L-PROTO", path, "one-shot barriers require a complete full side and stage initialization")
        return Protocol(
            kind,
            full,
            _barrier_side(obj["empty"], f"{path}.empty") if "empty" in obj else None,
            init,
            release,
            None,
            None,
            None,
        )
    if kind == "call":
        obj = _object(
            value,
            path,
            "L-PROTO",
            required=("kind", "export", "signature", "decl", "decl_sha256"),
        )
        decl = _string(obj["decl"], f"{path}.decl", "L-PROTO")
        digest = _sha256(obj["decl_sha256"], f"{path}.decl_sha256", "L-PROTO")
        actual = declaration_digest(decl)
        if digest != actual:
            _reject(
                "L-PROTO",
                f"{path}.decl_sha256",
                "declaration hash does not match normalized UTF-8 text",
                expected=actual,
                actual=digest,
            )
        call = CallDecl(
            _identifier(obj["export"], f"{path}.export", "L-PROTO"),
            _string(obj["signature"], f"{path}.signature", "L-PROTO"),
            decl,
            digest,
        )
        return Protocol(kind, None, None, None, None, call, None, None)
    if kind == "named_barrier":
        obj = _object(value, path, "L-PROTO", required=("kind", "id", "count"))
        named = NamedBarrierDecl(
            _handle_name(obj["id"], f"{path}.id"),
            _integer(obj["count"], f"{path}.count", "L-PROTO", positive=True),
        )
        return Protocol(kind, None, None, None, None, None, named, None)
    if kind == "counter":
        obj = _object(value, path, "L-PROTO", required=("kind", "slot", "release", "acquire"))
        release = _string(obj["release"], f"{path}.release", "L-PROTO")
        acquire = _string(obj["acquire"], f"{path}.acquire", "L-PROTO")
        scope = _COUNTER_RELEASE_SCOPES.get(release)
        if scope is None or acquire not in _COUNTER_ACQUIRES[scope]:
            _reject(
                "L-PROTO",
                path,
                "counter release/acquire must be red.release.gpu with a .gpu acquire pattern "
                "or red.release.sys with a .sys acquire pattern",
                release=release,
                acquire=acquire,
                admitted={key: list(value) for key, value in _COUNTER_ACQUIRES.items()},
            )
        counter = CounterDecl(_handle_name(obj["slot"], f"{path}.slot"), release, acquire)
        return Protocol(kind, None, None, None, None, None, None, counter)
    _reject("L-PROTO", f"{path}.kind", "unknown protocol kind", actual=kind)


def _binds(value: Any, path: str) -> dict[str, str]:
    obj = _object_map(value, path, "L-HANDLE")
    result: dict[str, str] = {}
    for role in sorted(obj):
        _identifier(role, f"{path}.{role}", "L-HANDLE")
        result[role] = _handle_name(obj[role], f"{path}.{role}")
    return result


def _endpoint(value: Any, path: str) -> Endpoint:
    obj = _object(
        value,
        path,
        "L-PAIR",
        required=(
            "name",
            "direction",
            "owner_role",
            "space",
            "scope",
            "proxy",
            "protocol",
            "binds",
        ),
        optional=("ring", "tile", "layout", "exports"),
    )
    direction = _string(obj["direction"], f"{path}.direction", "L-PAIR")
    if direction not in {"produce", "consume"}:
        _reject("L-PAIR", f"{path}.direction", "direction must be 'produce' or 'consume'")
    space = _string(obj["space"], f"{path}.space", "L-PAIR")
    if space not in _SPACES:
        _reject("L-PAIR", f"{path}.space", "unknown endpoint space")
    scope = _string(obj["scope"], f"{path}.scope", "L-PAIR")
    if scope not in _SCOPES:
        _reject("L-PAIR", f"{path}.scope", "unknown endpoint scope")
    proxy = _string(obj["proxy"], f"{path}.proxy", "L-PAIR")
    if proxy not in _PROXIES:
        _reject("L-PAIR", f"{path}.proxy", "unknown endpoint proxy")
    signal = space == "none"
    if signal:
        if "tile" in obj or "layout" in obj:
            _reject("L-PAIR", path, "completion-only endpoints must omit tile and layout")
        if scope not in {"cta", "cluster"} or proxy != "generic":
            _reject("L-PAIR", path, "completion-only endpoints require cta/cluster scope and generic proxy")
        tile, layout = Tile("none", ()), Layout(space="none")
    else:
        for key in ("tile", "layout"):
            if key not in obj:
                _reject("L-PAIR", f"{path}.{key}", f"tiled endpoints require {key}")
        tile = _tile(obj["tile"], f"{path}.tile")
        layout = load_layout(obj["layout"], space, f"{path}.layout")
    ring = _ring(obj["ring"], f"{path}.ring", signal=signal) if "ring" in obj else (
        Ring(Depth(1, None), 0) if signal else None
    )
    protocol = _protocol(obj["protocol"], f"{path}.protocol", signal=signal)
    if protocol.counter is not None:
        required = _COUNTER_RELEASE_SCOPES[protocol.counter.release_op]
        if scope != required:
            _reject("L-PAIR", f"{path}.scope", f"this counter release/acquire pattern requires scope {required!r}")
    elif scope == "system":
        _reject("L-PAIR", f"{path}.scope", "system scope is admitted only for counter endpoints")
    binds = _binds(obj["binds"], f"{path}.binds")
    if signal:
        if protocol.kind != "mbarrier" or not isinstance(protocol.full, BarrierSide):
            _reject("L-PROTO", f"{path}.protocol", "completion-only endpoints require a complete mbarrier full side")
        if protocol.full.tx_bytes != 0:
            _reject("L-PROTO", f"{path}.protocol.full.tx_bytes", "completion-only endpoints require tx_bytes 0")
        if binds != {"full": protocol.full.slots}:
            _reject("L-PAIR", f"{path}.binds", "completion-only endpoints require only the full slot binding")
    if protocol.kind == "call" and direction == "consume":
        if obj["owner_role"] is not None:
            _reject("L-PAIR", f"{path}.owner_role", "provided call endpoints must not name a caller role")
        owner_role = None
    elif protocol.kind != "call" and space != "reg" and obj["owner_role"] is None:
        # An endpoint signalled or awaited inside an exported callable runs in its caller's role.
        # Only register tiles (thread-value layout) and host transfers depend on that role.
        owner_role = None
    else:
        owner_role = _identifier(obj["owner_role"], f"{path}.owner_role", "L-PAIR")
    exports = None
    if "exports" in obj:
        if protocol.kind == "call":
            _reject("L-PAIR", f"{path}.exports", "call endpoints are themselves exports or callsites")
        exports = _unique_strings(obj["exports"], f"{path}.exports", "L-PAIR")
        if not exports:
            _reject("L-PAIR", f"{path}.exports", "an export association names at least one export")
        for index, name in enumerate(exports):
            _identifier(name, f"{path}.exports[{index}]", "L-PAIR")
    return Endpoint(
        _string(obj["name"], f"{path}.name", "L-PAIR"),
        direction,
        owner_role,
        space,
        scope,
        proxy,
        tile,
        layout,
        ring,
        protocol,
        binds,
        exports,
    )


def _canonical_table_key(value: Any, path: str) -> tuple[str, ...]:
    key = _string(value, path, "L-HANDLE")
    parts = key.split(",")
    if not parts or any(not part for part in parts):
        _reject("L-HANDLE", path, "extent table key must be a non-empty handle tuple")
    names: list[str] = []
    rebuilt: list[str] = []
    for part in parts:
        if part.count("=") != 1:
            _reject("L-HANDLE", path, "extent table key components must be 'handle=value'")
        name, raw_value = part.split("=", 1)
        _handle_name(name, path)
        if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", raw_value):
            _reject("L-HANDLE", path, "extent table values must be canonical decimal integers")
        names.append(name)
        rebuilt.append(f"{name}={int(raw_value)}")
    if len(set(names)) != len(names) or names != sorted(names) or key != ",".join(rebuilt):
        _reject("L-HANDLE", path, "extent table key must name unique handles in lexical order")
    return tuple(names)


def _extent(value: Any, path: str) -> Extent:
    if not isinstance(value, dict):
        _reject("L-HANDLE", path, "extent must be an object")
    if "table" in value:
        obj = _object(value, path, "L-HANDLE", required=("table",))
        table_obj = _object_map(obj["table"], f"{path}.table", "L-HANDLE")
        if not table_obj:
            _reject("L-HANDLE", f"{path}.table", "extent table must not be empty")
        table: dict[str, int] = {}
        complete_names: tuple[str, ...] | None = None
        for key in sorted(table_obj):
            names = _canonical_table_key(key, f"{path}.table.{key}")
            if complete_names is None:
                complete_names = names
            elif names != complete_names:
                _reject(
                    "L-HANDLE",
                    f"{path}.table.{key}",
                    "extent table keys must all be complete over the same handles",
                    expected=list(complete_names),
                    actual=list(names),
                )
            table[key] = _integer(table_obj[key], f"{path}.table.{key}", "L-HANDLE", nonnegative=True)
        return Extent(0, {}, table)
    obj = _object(value, path, "L-HANDLE", required=("const", "per"))
    per_obj = _object_map(obj["per"], f"{path}.per", "L-HANDLE")
    per: dict[str, int] = {}
    for name in sorted(per_obj):
        _handle_name(name, f"{path}.per.{name}")
        per[name] = _integer(per_obj[name], f"{path}.per.{name}", "L-HANDLE", nonnegative=True)
    return Extent(
        _integer(obj["const"], f"{path}.const", "L-HANDLE", nonnegative=True),
        per,
        {},
    )


def _claim(value: Any, path: str, profile: Profile) -> Claim:
    obj = _object(
        value,
        path,
        "L-HANDLE",
        required=("domain", "handle"),
        optional=(
            "lifetime",
            "extent",
            "align",
            "offset",
            "roles",
            "reg_group",
            "values",
            "absolute_base",
            "alias",
        ),
    )
    domain = _string(obj["domain"], f"{path}.domain", "L-HANDLE")
    if domain not in profile.domains:
        _reject("L-HANDLE", f"{path}.domain", "claim domain is absent from the target profile", domain=domain)
    handle = _handle_name(obj["handle"], f"{path}.handle")
    raw_lifetime = obj.get("lifetime", "kernel")
    if isinstance(raw_lifetime, dict):
        bound = _object(raw_lifetime, f"{path}.lifetime", "L-PHASE",
                        required=("tensor", "from"), optional=("to",))
        lifetime = TensorLifetime(
            _string(bound["tensor"], f"{path}.lifetime.tensor", "L-PHASE"),
            _string(bound["from"], f"{path}.lifetime.from", "L-PHASE"),
            _string(bound["to"], f"{path}.lifetime.to", "L-PHASE") if "to" in bound else None,
        )
    else:
        lifetime = _string(raw_lifetime, f"{path}.lifetime", "L-PHASE")
        if lifetime != "kernel":
            if not lifetime.startswith("phase:") or not lifetime.removeprefix("phase:"):
                _reject("L-PHASE", f"{path}.lifetime", "expected kernel, phase:<name>, or tensor event lifetime")
            phase = lifetime.removeprefix("phase:")
            if _PHASE_RE.fullmatch(phase) is None or "*" in phase:
                _reject("L-PHASE", f"{path}.lifetime", "invalid literal phase name")

    if domain == "regs":
        forbidden = [name for name in ("extent", "align", "offset", "absolute_base", "alias") if name in obj]
        if forbidden:
            _reject("L-REGS", f"{path}.{sorted(forbidden)[0]}", "regs claims omit interval and alias fields")
        if "roles" not in obj or "values" not in obj:
            _reject("L-REGS", path, "regs claims require roles and values")
        roles = _unique_strings(obj["roles"], f"{path}.roles", "L-REGS", nonempty=True, identifiers=True)
        values = _unique_ints(obj["values"], f"{path}.values", "L-REGS", positive=True)
        for index, item in enumerate(values):
            if not profile.caps.regs_min <= item <= profile.caps.regs_max or item % profile.caps.regs_step:
                _reject(
                    "L-REGS",
                    f"{path}.values[{index}]",
                    "register value is outside the target range or granularity",
                )
        reg_group = None
        if "reg_group" in obj:
            reg_group = _identifier(obj["reg_group"], f"{path}.reg_group", "L-REGS")
        return Claim(domain, handle, lifetime, None, None, None, roles, reg_group, values, None, None)

    if domain not in _INTERVAL_DOMAINS:
        _reject("L-HANDLE", f"{path}.domain", "profile domain has no schema claim form", domain=domain)
    if "extent" not in obj or "align" not in obj:
        _reject("L-ALIGN", path, "interval claims require extent and align")
    extent = _extent(obj["extent"], f"{path}.extent")
    align = _power_of_two(obj["align"], f"{path}.align", "L-ALIGN")
    offset = (
        _integer(obj["offset"], f"{path}.offset", "L-ALIGN", nonnegative=True)
        if "offset" in obj
        else None
    )
    absolute_base = (
        _integer(obj["absolute_base"], f"{path}.absolute_base", "L-ALIGN", nonnegative=True)
        if "absolute_base" in obj
        else None
    )
    if absolute_base is not None and absolute_base % align:
        _reject("L-ALIGN", f"{path}.absolute_base", "absolute base violates claim alignment")
    alias = _identifier(obj["alias"], f"{path}.alias", "L-OWNER") if "alias" in obj else None
    roles = (
        _unique_strings(obj["roles"], f"{path}.roles", "L-OWNER", nonempty=True, identifiers=True)
        if "roles" in obj
        else None
    )
    values = (
        _unique_ints(obj["values"], f"{path}.values", "L-HANDLE") if "values" in obj else None
    )
    reg_group = (
        _identifier(obj["reg_group"], f"{path}.reg_group", "L-REGS") if "reg_group" in obj else None
    )
    return Claim(domain, handle, lifetime, extent, align, offset, roles, reg_group, values, absolute_base, alias)


def _stage_topology(value: Any, path: str, profile: Profile) -> StageTopology:
    obj = _object(
        value,
        path,
        "L-TOPO",
        required=("cta_group", "cluster", "multicast", "persistent", "scheduler"),
        optional=("scheduler_strategy",),
    )
    cta_obj = _object(obj["cta_group"], f"{path}.cta_group", "L-TOPO", required=("fixed",))
    cta_group = _integer(cta_obj["fixed"], f"{path}.cta_group.fixed", "L-TOPO", positive=True)
    if cta_group not in profile.caps.cta_group_legal:
        _reject("L-TOPO", f"{path}.cta_group.fixed", "cta_group is not legal for the target profile")
    cluster = _object(obj["cluster"], f"{path}.cluster", "L-TOPO", required=("compiled", "launch"))
    compiled_raw = cluster["compiled"]
    compiled: tuple[int, int, int] | str
    if compiled_raw == "dynamic":
        compiled = "dynamic"
    else:
        compiled = _shape3(compiled_raw, f"{path}.cluster.compiled")
    launch_obj = _object(
        cluster["launch"],
        f"{path}.cluster.launch",
        "L-TOPO",
        required=("x_multiple_of", "max"),
    )
    x_multiple = _integer(
        launch_obj["x_multiple_of"],
        f"{path}.cluster.launch.x_multiple_of",
        "L-TOPO",
        positive=True,
    )
    maximum = _shape3(launch_obj["max"], f"{path}.cluster.launch.max")
    if any(value > cap for value, cap in zip(maximum, profile.caps.max_cluster)):
        _reject("L-TOPO", f"{path}.cluster.launch.max", "cluster maximum exceeds the target profile")
    if x_multiple > maximum[0]:
        _reject("L-TOPO", f"{path}.cluster.launch.x_multiple_of", "x multiple exceeds launch maximum")
    if compiled != "dynamic" and (
        any(value > limit for value, limit in zip(compiled, maximum)) or compiled[0] % x_multiple != 0
    ):
        _reject("L-TOPO", f"{path}.cluster.compiled", "compiled cluster violates its launch constraints")
    scheduler = _string(obj["scheduler"], f"{path}.scheduler", "L-TOPO")
    if scheduler not in profile.schedulers:
        _reject("L-TOPO", f"{path}.scheduler", "scheduler is not admitted by the target profile",
                admitted=sorted(profile.schedulers))
    strategy = None
    if "scheduler_strategy" in obj:
        strategy_path = f"{path}.scheduler_strategy"
        declaration = _object(obj["scheduler_strategy"], strategy_path, "L-TOPO", required=("id", "inputs"))
        strategy_id = _string(declaration["id"], f"{strategy_path}.id", "L-TOPO")
        if strategy_id not in profile.scheduler_strategies:
            _reject("L-TOPO", f"{strategy_path}.id", "scheduler strategy is not admitted by the target profile",
                    admitted=sorted(profile.scheduler_strategies))
        inputs = tuple(_unique_strings(declaration["inputs"], f"{strategy_path}.inputs", "L-TOPO"))
        if inputs != _SCHEDULER_STRATEGY_INPUTS[strategy_id]:
            _reject("L-TOPO", f"{strategy_path}.inputs", "scheduler strategy inputs disagree with its registry",
                    expected=list(_SCHEDULER_STRATEGY_INPUTS[strategy_id]))
        strategy = SchedulerStrategy(strategy_id, inputs)
    return StageTopology(
        cta_group,
        compiled,  # type: ignore[arg-type]
        ClusterConstraints(x_multiple, maximum),
        _unique_strings(obj["multicast"], f"{path}.multicast", "L-TOPO"),
        _boolean(obj["persistent"], f"{path}.persistent", "L-TOPO"),
        scheduler,
        strategy,
    )


def _params(value: Any, path: str, profile: Profile) -> ParamsDecl:
    obj = _object(
        value,
        path,
        "L-PARAMS",
        required=("bytes", "align", "type", "sha256_of_layout_probe"),
    )
    byte_count = _integer(obj["bytes"], f"{path}.bytes", "L-PARAMS", nonnegative=True)
    if byte_count > profile.caps.param_limit_bytes:
        _reject("L-PARAMS", f"{path}.bytes", "parameter blob exceeds the target parameter limit")
    return ParamsDecl(
        byte_count,
        _power_of_two(obj["align"], f"{path}.align", "L-PARAMS"),
        _string(obj["type"], f"{path}.type", "L-PARAMS"),
        _sha256(obj["sha256_of_layout_probe"], f"{path}.sha256_of_layout_probe", "L-PARAMS"),
    )


def _descriptor(value: Any, path: str) -> DescriptorDecl:
    obj = _object(value, path, "L-PARAMS", required=("bytes", "align"))
    return DescriptorDecl(
        _integer(obj["bytes"], f"{path}.bytes", "L-PARAMS", positive=True),
        _power_of_two(obj["align"], f"{path}.align", "L-PARAMS"),
    )


def _validate_stage_handle_uses(stage: StageManifest) -> None:
    uses: dict[str, tuple[set[str], set[str]]] = {}
    constraints: dict[str, list[set[int]]] = {}

    def use(name: str, classes: set[str], domains: set[str] | None = None) -> None:
        domain_set = domains or set()
        if name not in uses:
            uses[name] = (set(classes), set(domain_set))
            return
        old_classes, old_domains = uses[name]
        common_classes = old_classes & classes
        common_domains = old_domains & domain_set if old_domains and domain_set else old_domains | domain_set
        if not common_classes or (old_domains and domain_set and not common_domains):
            _reject(
                "L-HANDLE",
                f"$.handles.{name}",
                "stage references the same handle with incompatible class or domain requirements",
            )
        uses[name] = (common_classes, common_domains)

    for variant in stage.variants:
        for name in variant.key:
            use(name, set(_HANDLE_CLASSES))
    for endpoint in stage.endpoints:
        if endpoint.ring is not None and endpoint.ring.depth.handle is not None:
            use(endpoint.ring.depth.handle, {"depth"})
        for bind_role, name in endpoint.binds.items():
            if bind_role in {"full", "empty"}:
                use(name, {"base"}, {"mbarrier"})
            else:
                use(name, {"base"})
        protocol = endpoint.protocol
        if isinstance(protocol.full, BarrierSide):
            use(protocol.full.slots, {"base"}, {"mbarrier"})
        if protocol.empty is not None:
            use(protocol.empty.slots, {"base"}, {"mbarrier"})
        if protocol.named_barrier is not None:
            use(protocol.named_barrier.id_handle, {"base"}, {"named_barrier"})
        if protocol.counter is not None:
            use(protocol.counter.slot_handle, {"base"}, {"gmem_counter"})
    for claim in stage.claims:
        if claim.domain == "regs":
            use(claim.handle, {"regs"}, {"regs"})
            assert claim.values is not None
            constraints.setdefault(claim.handle, []).append(set(claim.values))
        else:
            use(claim.handle, {"base"}, {claim.domain})
        if claim.extent is not None:
            for name in claim.extent.per:
                use(name, {"depth", "variant"})
            for table_key in claim.extent.table:
                for part in table_key.split(","):
                    use(part.split("=", 1)[0], {"depth", "variant"})
    for name, value_sets in constraints.items():
        if value_sets and not set.intersection(*value_sets):
            _reject("L-HANDLE", f"$.handles.{name}", "shared handle value constraints have an empty intersection")


def _symbols(value: Any) -> SymbolTable:
    try:
        return SymbolTable.from_manifest(value)
    except ValueError as exc:
        _reject("L-SYM", "$.symbols", str(exc))


def _provenance(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path, "L-SYM", required=("mechanism", "inputs"), optional=("sections",))
    result = {
        "mechanism": _string(obj["mechanism"], f"{path}.mechanism", "L-SYM"),
        "inputs": {
            name: _sha256(digest, f"{path}.inputs.{name}", "L-SYM")
            for name, digest in _object_map(obj["inputs"], f"{path}.inputs", "L-SYM").items()
        },
    }
    if "sections" in obj:
        sections = _object_map(obj["sections"], f"{path}.sections", "L-SYM")
        for name, source in sections.items():
            if source not in ("declared", "compiled", "observed", "fitted"):
                _reject("L-SYM", f"{path}.sections.{name}", "unknown provenance source", actual=source)
        result["sections"] = dict(sections)
    return result


def load_stage_manifest(obj: dict[str, Any]) -> StageManifest:
    """Load one strict ABI-v2 stage manifest into the frozen contract model."""

    root = _object(
        obj,
        "$",
        "L-SYM",
        required=("pipelink_abi", "stage", "endpoints", "claims", "topology", "params"),
        optional=("hints", "payload", "symbols", "host", "tensor", "provenance"),
    )
    target, profile = _target_stamp(root["pipelink_abi"])
    if "tensor" in root and target.schema not in {"2.2", "3.0"}:
        _reject("T-CONTRACT", "$.tensor", "tensor contracts require schema 2.2 or 3.0")
    stage_obj = _object(
        root["stage"],
        "$.stage",
        "L-SYM",
        required=(
            "id",
            "version",
            "symbol_prefix",
            "entry_abi",
            "roles",
            "exports",
            "entry_signature",
            "variants",
        ),
        optional=("descriptor",),
    )
    stage_id = _string(stage_obj["id"], "$.stage.id", "L-SYM")
    if _STAGE_ID_RE.fullmatch(stage_id) is None:
        _reject("L-SYM", "$.stage.id", "stage id must be a lowercase reverse-DNS identifier")
    symbol_prefix = _string(stage_obj["symbol_prefix"], "$.stage.symbol_prefix", "L-SYM")
    expected_prefix = _normalized_prefix(stage_id)
    if symbol_prefix != expected_prefix:
        _reject(
            "L-SYM",
            "$.stage.symbol_prefix",
            "symbol_prefix does not match normalized stage id",
            expected=expected_prefix,
            actual=symbol_prefix,
        )
    roles = _roles(stage_obj["roles"], "$.stage.roles")
    exports = _exports(stage_obj["exports"], "$.stage.exports")
    overlap = sorted(set(roles) & set(exports))
    if overlap:
        _reject("L-SYM", f"$.stage.roles.{overlap[0]}", "roles and exports must be disjoint")
    entry_abi = _string(stage_obj["entry_abi"], "$.stage.entry_abi", "L-ENTRYABI")
    if entry_abi != "v2":
        _reject("L-ENTRYABI", "$.stage.entry_abi", "pipec2 requires entry_abi 'v2'; replay historical artifacts from the pre-retirement archive (see pipec/README.md)")
    entry_signature = _string(stage_obj["entry_signature"], "$.stage.entry_signature", "L-SYM")
    if entry_signature != "void(const void* params, unsigned variant)":
        _reject("L-SYM", "$.stage.entry_signature", "v2 roles require the fixed entry signature")

    endpoint_items = _array(root["endpoints"], "$.endpoints", "L-PAIR")
    endpoints = [_endpoint(item, f"$.endpoints[{index}]") for index, item in enumerate(endpoint_items)]
    # Channel cardinality is a link-assembly check (L-PAIR), not a per-stage schema rule.
    for index, endpoint in enumerate(endpoints):
        unknown = sorted(set(endpoint.exports or ()) - set(exports))
        if unknown:
            _reject("L-PAIR", f"$.endpoints[{index}].exports", "endpoint names an export this stage does not provide",
                    export=unknown[0])
        if endpoint.protocol.kind == "call" or endpoint.owner_role is None:
            # A callsite carries its own role; the provided endpoint is roleless.
            # Unlike register tiles, call endpoints do not bind a shared role view.
            # Role-less non-register endpoints execute inside exported callables.
            continue
        if endpoint.owner_role not in roles:
            _reject("L-PAIR", f"$.endpoints[{index}].owner_role", "endpoint owner_role is not a declared role")

    claim_items = _array(root["claims"], "$.claims", "L-HANDLE")
    claims = [_claim(item, f"$.claims[{index}]", profile) for index, item in enumerate(claim_items)]
    claim_keys: set[tuple[str, str]] = set()
    for index, claim in enumerate(claims):
        key = (claim.domain, claim.handle)
        if key in claim_keys:
            _reject("L-OWNER", f"$.claims[{index}]", "duplicate claim for domain and handle")
        claim_keys.add(key)
        if claim.roles is not None:
            unknown_roles = sorted(set(claim.roles) - set(roles))
            if unknown_roles:
                _reject("L-OWNER", f"$.claims[{index}].roles", "claim names an undeclared role", role=unknown_roles[0])

    for index, endpoint in enumerate(endpoints):
        protocol_path = f"$.endpoints[{index}].protocol"
        if endpoint.protocol.release == "never" and (
            endpoint.ring is None or endpoint.ring.depth.fixed != 1
        ):
            _reject("L-PROTO", protocol_path, "one-shot barriers require a fixed single-slot ring")
        for label in ("full", "empty"):
            side = getattr(endpoint.protocol, label)
            if not isinstance(side, BarrierSide) or side.native is None:
                continue
            path = f"{protocol_path}.{label}.native"
            if endpoint.protocol.init.materializer != "stage":
                _reject("L-PROTO", path, "native barrier storage requires stage initialization")
            if endpoint.ring is None:
                _reject("L-PROTO", path, "native barrier storage requires a ring")
            backing = [claim for claim in claims if claim.handle == side.native.base]
            if len(backing) != 1 or backing[0].domain not in {"smem", "mbarrier"}:
                _reject("L-PROTO", f"{path}.base", "native base must name this stage's smem or mbarrier claim")
            claim = backing[0]
            assert claim.extent is not None
            unit_bytes = profile.domains[claim.domain].unit_scale
            claim_start = (claim.offset or 0) * unit_bytes
            if side.native.offset_bytes < claim_start:
                _reject("L-PROTO", path, "native barrier slots precede the backing claim extent")
            # Per-handle extents and depths are checked after solving. Even without
            # a fixed depth, a constant allocation must contain the first barrier.
            if not claim.extent.per and not claim.extent.table:
                depth = endpoint.ring.depth.fixed if endpoint.ring is not None else None
                required = side.native.offset_bytes + ((depth or 1) - 1) * side.native.stride_bytes + 8
                if required > claim_start + claim.extent.const * unit_bytes:
                    _reject("L-PROTO", path, "native barrier slots exceed the backing claim extent")

    descriptor = (
        _descriptor(stage_obj["descriptor"], "$.stage.descriptor") if "descriptor" in stage_obj else None
    )
    hints = root.get("hints", {})
    if not isinstance(hints, dict) or any(not isinstance(key, str) for key in hints):
        _reject("L-PARAMS", "$.hints", "hints must be an object with string keys")

    params = _params(root["params"], "$.params", profile)
    host = None
    if "host" in root:
        if target.schema != "3.0":
            _reject("L-HOSTCHAN", "$.host", "host declarations require schema 3.0", subreason="inventory")
        from .host.contracts import parse_host
        host = parse_host(root["host"], stage_id, roles, target)
        if params.bytes != 0:
            _reject("L-HOSTCHAN", "$.params", "host declarations forbid stage parameter blobs", subreason="inventory")
    elif target.host_abi is not None:
        _reject("L-HOSTCHAN", "$.pipelink_abi.target", "host ABI requires an explicit host declaration", subreason="inventory")
    if params.bytes == 0 and host is None:
        if roles:
            _reject(
                "L-PARAMS",
                "$.params.bytes",
                "zero-byte params are allowed only for stages with no roles",
            )

    tensor = None
    if "tensor" in root:
        from pipec2.tensor import contracts

        tensor = contracts.load_stage_tensor(root["tensor"], "$.tensor")

    stage = StageManifest(
        stage_id,
        _string(stage_obj["version"], "$.stage.version", "L-SYM"),
        symbol_prefix,
        roles,
        exports,
        entry_signature,
        entry_abi,  # type: ignore[arg-type]
        _variants(stage_obj["variants"], "$.stage.variants", symbol_prefix),
        endpoints,
        claims,
        _stage_topology(root["topology"], "$.topology", profile),
        params,
        descriptor,
        hints,
        target,
        _symbols(root["symbols"]) if "symbols" in root else None,
        host,
        tensor,
        _provenance(root["provenance"], "$.provenance") if "provenance" in root else None,
    )
    _validate_stage_handle_uses(stage)
    return stage


def _handle(value: Any, name: str, path: str, profile: Profile) -> Handle:
    obj = _object(
        value,
        path,
        "L-HANDLE",
        required=("class", "binding", "owner"),
        optional=("domain", "values"),
    )
    cls = _string(obj["class"], f"{path}.class", "L-HANDLE")
    if cls not in _HANDLE_CLASSES:
        _reject("L-HANDLE", f"{path}.class", "unknown handle class")
    binding = _string(obj["binding"], f"{path}.binding", "L-HANDLE")
    if binding not in _BINDINGS:
        _reject("L-HANDLE", f"{path}.binding", "unknown handle binding")
    domain = None
    if cls in {"base", "regs"}:
        if "domain" not in obj:
            _reject("L-HANDLE", path, f"{cls} handle requires domain")
        domain = _string(obj["domain"], f"{path}.domain", "L-HANDLE")
        if domain not in profile.domains:
            _reject("L-HANDLE", f"{path}.domain", "handle domain is absent from the target profile")
        if cls == "regs" and domain != "regs":
            _reject("L-HANDLE", f"{path}.domain", "regs handles require the regs domain")
    elif "domain" in obj:
        _reject("L-HANDLE", f"{path}.domain", f"{cls} handles omit domain")

    values = None
    if cls in {"depth", "regs", "variant"}:
        if "values" not in obj:
            _reject("L-HANDLE", path, f"{cls} handle requires finite values")
        values = _unique_ints(obj["values"], f"{path}.values", "L-HANDLE", positive=cls != "variant")
        if cls == "regs":
            launch_cap = name == "pipelink_launch_regs" and obj["owner"] == "reg_group:launch"
            maximum = profile.caps.launch_regs_max if launch_cap else profile.caps.regs_max
            step = 1 if launch_cap else profile.caps.regs_step
            for index, item in enumerate(values):
                if not profile.caps.regs_min <= item <= maximum or item % step:
                    _reject(
                        "L-HANDLE",
                        f"{path}.values[{index}]",
                        "launch register cap is outside the target range"
                        if launch_cap
                        else "register value is outside the target range or granularity",
                    )
    elif "values" in obj:
        _reject("L-HANDLE", f"{path}.values", "base handles omit values")

    owner = _string(obj["owner"], f"{path}.owner", "L-HANDLE")
    owner_prefix = {"depth": "channel:", "regs": "reg_group:", "base": "claim:", "variant": "stage:"}[cls]
    if not owner.startswith(owner_prefix) or not owner.removeprefix(owner_prefix):
        _reject("L-HANDLE", f"{path}.owner", f"{cls} handle owner must start with {owner_prefix!r}")
    return Handle(name, cls, binding, domain, values, owner)


def _stage_ref(value: Any, path: str) -> StageRef:
    obj = _object(value, path, "L-SYM", required=("artifact_path", "stage_id"))
    stage_id = _string(obj["stage_id"], f"{path}.stage_id", "L-SYM")
    if _STAGE_ID_RE.fullmatch(stage_id) is None:
        _reject("L-SYM", f"{path}.stage_id", "stage_id must be a lowercase reverse-DNS identifier")
    return StageRef(_string(obj["artifact_path"], f"{path}.artifact_path", "L-SYM"), stage_id)


def _pipeline_domain(value: Any, name: str, path: str) -> PipelineDomain:
    if name == "smem_page":
        obj = _object(value, path, "L-HANDLE", required=("capacity", "page_bytes"))
        return PipelineDomain(
            _integer(obj["capacity"], f"{path}.capacity", "L-HANDLE", positive=True),
            _integer(obj["page_bytes"], f"{path}.page_bytes", "L-HANDLE", positive=True),
        )
    obj = _object(value, path, "L-HANDLE", required=("capacity",))
    return PipelineDomain(_integer(obj["capacity"], f"{path}.capacity", "L-HANDLE", positive=True), None)


def _phase(value: Any, path: str) -> str:
    result = _string(value, path, "L-PHASE")
    if _PHASE_RE.fullmatch(result) is None or "*" in result:
        _reject("L-PHASE", path, "phase must be a literal finite name")
    return result


def _exclusive(value: Any, phases: list[str], path: str) -> list[tuple[str, str]]:
    entries = _array(value, path, "L-PHASE")
    phase_set = set(phases)
    instruction_phases = sorted(phase for phase in phases if phase.startswith("instr_"))
    result: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        pair_path = f"{path}[{index}]"
        pair = _array(entry, pair_path, "L-PHASE")
        if len(pair) != 2:
            _reject("L-PHASE", pair_path, "exclusive entry must contain exactly two phase selectors")
        selectors = [_string(item, f"{pair_path}[{side}]", "L-PHASE") for side, item in enumerate(pair)]
        matches: list[list[str]] = []
        for side, selector in enumerate(selectors):
            if selector == "instr_*":
                if not instruction_phases:
                    _reject("L-PHASE", f"{pair_path}[{side}]", "instr_* matches no declared phase")
                matches.append(instruction_phases)
            elif "*" in selector:
                _reject("L-PHASE", f"{pair_path}[{side}]", "only the instr_* phase pattern is supported")
            elif selector not in phase_set:
                _reject("L-PHASE", f"{pair_path}[{side}]", "unknown phase")
            else:
                matches.append([selector])

        expanded: set[tuple[str, str]] = set()
        if selectors[0] == selectors[1] == "instr_*":
            expanded.update(itertools.combinations(instruction_phases, 2))
        else:
            for left, right in itertools.product(matches[0], matches[1]):
                if left == right:
                    _reject("L-PHASE", pair_path, "exclusive self-pair is invalid", phase=left)
                expanded.add(tuple(sorted((left, right))))  # type: ignore[arg-type]
        duplicate = sorted(expanded & result)
        if duplicate:
            _reject(
                "L-PHASE",
                pair_path,
                "duplicate exclusive pair after canonical expansion",
                pair=list(duplicate[0]),
            )
        result.update(expanded)
    return sorted(result)


def _pipeline_topology(value: Any, path: str, profile: Profile) -> PipelineTopology:
    obj = _object(value, path, "L-TOPO", required=("persistent", "scheduler"))
    scheduler = _string(obj["scheduler"], f"{path}.scheduler", "L-TOPO")
    if scheduler not in profile.schedulers:
        _reject("L-TOPO", f"{path}.scheduler", "scheduler is not admitted by the target profile",
                admitted=sorted(profile.schedulers))
    return PipelineTopology(_boolean(obj["persistent"], f"{path}.persistent", "L-TOPO"), scheduler)


def _handle_list(
    value: Any,
    path: str,
    handles: dict[str, Handle],
    expected_class: str,
) -> list[str]:
    result = _unique_strings(value, path, "L-HANDLE")
    for index, name in enumerate(result):
        _handle_name(name, f"{path}[{index}]")
        handle = handles.get(name)
        if handle is None:
            _reject("L-HANDLE", f"{path}[{index}]", "referenced handle is not declared")
        if handle.cls != expected_class:
            _reject(
                "L-HANDLE",
                f"{path}[{index}]",
                f"priority entry requires handle class {expected_class!r}",
                actual=handle.cls,
            )
        if handle.binding != "bind":
            _reject("L-HANDLE", f"{path}[{index}]", "priority entries require bind handles")
    return result


def _tensor_connections(value: Any, path: str) -> tuple[TensorConnection, ...]:
    items = _array(value, path, "T-CONTRACT")
    result: list[TensorConnection] = []
    names: set[str] = set()
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        obj = _object(item, item_path, "T-CONTRACT", required=("name", "export", "import"))
        name = _string(obj["name"], f"{item_path}.name", "T-CONTRACT")
        if name in names:
            _reject("T-CONTRACT", f"{item_path}.name", "tensor connection names must be unique")
        names.add(name)
        references: list[str] = []
        for kind in ("export", "import"):
            ref_path = f"{item_path}.{kind}"
            ref = _string(obj[kind], ref_path, "T-CONTRACT")
            stage_id, separator, record_name = ref.partition(":")
            if (
                not separator
                or _STAGE_ID_RE.fullmatch(stage_id) is None
                or not record_name
                or ":" in record_name
            ):
                _reject("T-CONTRACT", ref_path, "expected '<stage_id>:<record_name>'")
            references.append(ref)
        result.append(TensorConnection(name, references[0], references[1]))
    return tuple(sorted(result, key=lambda item: item.name))


def load_pipeline(obj: dict[str, Any]) -> Pipeline:
    """Load one strict ABI-v2 pipeline declaration into the frozen contract model."""

    root = _object(obj, "$", "L-HANDLE", required=("pipelink_abi", "pipeline"))
    target, profile = _target_stamp(root["pipelink_abi"])
    pipeline_obj = _object(
        root["pipeline"],
        "$.pipeline",
        "L-HANDLE",
        required=(
            "name",
            "stages",
            "handles",
            "phases",
            "exclusive",
            "domains",
            "depth_priority",
            "regs_priority",
            "forced_bindings",
            "topology",
        ),
        optional=("tensor_connections", "math"),
    )
    stage_items = _array(pipeline_obj["stages"], "$.pipeline.stages", "L-SYM", nonempty=True)
    stages = [_stage_ref(item, f"$.pipeline.stages[{index}]") for index, item in enumerate(stage_items)]
    stage_ids = [stage.stage_id for stage in stages]
    if len(set(stage_ids)) != len(stage_ids):
        _reject("L-SYM", "$.pipeline.stages", "stage_id entries must be unique")

    raw_handles = _object_map(pipeline_obj["handles"], "$.pipeline.handles", "L-HANDLE")
    handles: dict[str, Handle] = {}
    for name in sorted(raw_handles):
        _handle_name(name, f"$.pipeline.handles.{name}")
        handles[name] = _handle(raw_handles[name], name, f"$.pipeline.handles.{name}", profile)

    phase_items = _array(pipeline_obj["phases"], "$.pipeline.phases", "L-PHASE", nonempty=True)
    phases = [_phase(item, f"$.pipeline.phases[{index}]") for index, item in enumerate(phase_items)]
    if len(set(phases)) != len(phases):
        _reject("L-PHASE", "$.pipeline.phases", "declared phases must be unique")
    exclusive = _exclusive(pipeline_obj["exclusive"], phases, "$.pipeline.exclusive")

    raw_domains = _object_map(pipeline_obj["domains"], "$.pipeline.domains", "L-HANDLE")
    expected_domains = {
        name for name, domain in profile.domains.items() if domain.capacity == "pipeline"
    }
    actual_domains = set(raw_domains)
    unknown_domains = sorted(actual_domains - expected_domains)
    if unknown_domains:
        _reject(
            "L-HANDLE",
            "$.pipeline.domains",
            "declared domains must be a subset of the profile's pipeline-capacity domains",
            expected=sorted(expected_domains),
            actual=sorted(actual_domains),
            unknown=unknown_domains,
        )
    domains = {
        name: _pipeline_domain(raw_domains[name], name, f"$.pipeline.domains.{name}")
        for name in sorted(actual_domains)
    }

    depth_priority = _handle_list(
        pipeline_obj["depth_priority"], "$.pipeline.depth_priority", handles, "depth"
    )
    regs_priority = _handle_list(
        pipeline_obj["regs_priority"], "$.pipeline.regs_priority", handles, "regs"
    )
    raw_forced = _object_map(pipeline_obj["forced_bindings"], "$.pipeline.forced_bindings", "L-HANDLE")
    forced: dict[str, int] = {}
    for name in sorted(raw_forced):
        _handle_name(name, f"$.pipeline.forced_bindings.{name}")
        handle = handles.get(name)
        if handle is None:
            _reject("L-HANDLE", f"$.pipeline.forced_bindings.{name}", "forced binding handle is not declared")
        selected = _integer(raw_forced[name], f"$.pipeline.forced_bindings.{name}", "L-HANDLE")
        if handle.values is not None and selected not in handle.values:
            _reject("L-HANDLE", f"$.pipeline.forced_bindings.{name}", "forced value is not allowed by handle")
        forced[name] = selected

    return Pipeline(
        _string(pipeline_obj["name"], "$.pipeline.name", "L-SYM"),
        stages,
        handles,
        phases,
        exclusive,
        domains,
        depth_priority,
        regs_priority,
        forced,
        _pipeline_topology(pipeline_obj["topology"], "$.pipeline.topology", profile),
        target,
        (
            _tensor_connections(pipeline_obj["tensor_connections"], "$.pipeline.tensor_connections")
            if "tensor_connections" in pipeline_obj else ()
        ),
        _math_mode(pipeline_obj["math"], "$.pipeline.math", "L-LINK") if "math" in pipeline_obj else None,
    )


def validate_assembly(pipeline: Pipeline, stages: list[StageManifest]) -> list[ClauseFailure]:
    """Return deterministic cross-object schema failures for one link assembly."""

    failures: dict[tuple[str, str, str], ClauseFailure] = {}

    def fail(path: str, message: str, *, clause_id: str = "L-HANDLE", **detail: Any) -> None:
        key = (clause_id, path, message)
        failures[key] = ClauseFailure(clause_id, path, message, detail)
    profile = get_profile(pipeline.target.profile)


    declarations: dict[str, Handle] = {}
    declaration_keys: dict[str, str] = {}
    for key in sorted(pipeline.handles):
        handle = pipeline.handles[key]
        path = f"pipeline.handles.{key}"
        if key != handle.name:
            fail(path, "handle map key does not match the declared handle name", declared=handle.name)
        previous_key = declaration_keys.get(handle.name)
        if previous_key is not None:
            fail(
                path,
                "handle is declared more than once in pipeline scope",
                first_key=previous_key,
                name=handle.name,
            )
        else:
            declarations[handle.name] = handle
            declaration_keys[handle.name] = key

    stage_by_id: dict[str, StageManifest] = {}
    for stage in sorted(stages, key=lambda item: item.id):
        if stage.id in stage_by_id:
            fail(f"stages.{stage.id}", "stage manifest is present more than once")
        else:
            stage_by_id[stage.id] = stage
    referenced_stage_ids = {ref.stage_id for ref in pipeline.stages}
    for stage_id in sorted(referenced_stage_ids - stage_by_id.keys()):
        fail(f"pipeline.stages.{stage_id}", "pipeline stage has no loaded manifest")
    for stage_id in sorted(stage_by_id.keys() - referenced_stage_ids):
        fail(f"stages.{stage_id}", "loaded stage is not declared by the pipeline")

    for connection in sorted(pipeline.tensor_connections, key=lambda item: item.name):
        for kind, reference in (("export", connection.export), ("import", connection.import_)):
            path = f"pipeline.tensor_connections.{connection.name}.{kind}"
            stage_id, record_name = reference.split(":", 1)
            stage = stage_by_id.get(stage_id)
            if stage_id not in referenced_stage_ids or stage is None:
                fail(
                    path,
                    "tensor connection references an unknown stage",
                    clause_id="T-CONTRACT",
                    stage_id=stage_id,
                )
                continue
            records = getattr(stage.tensor, f"{kind}s") if stage.tensor is not None else ()
            if not any(record.name == record_name for record in records):
                fail(
                    path,
                    f"tensor {kind} is not declared by the stage",
                    clause_id="T-CONTRACT",
                    stage_id=stage_id,
                    name=record_name,
                )

    uses: dict[str, list[tuple[str, set[str], set[str], set[str]]]] = {}
    value_constraints: dict[str, list[tuple[str, set[int]]]] = {}

    def use(
        name: str,
        path: str,
        *,
        classes: set[str],
        bindings: set[str],
        domains: set[str] | None = None,
    ) -> None:
        uses.setdefault(name, []).append((path, classes, bindings, domains or set()))

    def constrain(name: str, path: str, values: set[int]) -> None:
        value_constraints.setdefault(name, []).append((path, values))

    for index, name in enumerate(pipeline.depth_priority):
        use(
            name,
            f"pipeline.depth_priority[{index}]",
            classes={"depth"},
            bindings={"bind"},
        )
    for index, name in enumerate(pipeline.regs_priority):
        use(
            name,
            f"pipeline.regs_priority[{index}]",
            classes={"regs"},
            bindings={"bind"},
            domains={"regs"},
        )
    for name, value in sorted(pipeline.forced_bindings.items()):
        use(
            name,
            f"pipeline.forced_bindings.{name}",
            classes=set(_HANDLE_CLASSES),
            bindings=set(_BINDINGS),
        )
        declaration = declarations.get(name)
        if declaration is not None and declaration.values is not None:
            constrain(name, f"pipeline.forced_bindings.{name}", {value})

    for stage_id in sorted(stage_by_id):
        stage = stage_by_id[stage_id]
        stage_path = f"stages.{stage_id}"
        variant_values: dict[str, set[int]] = {}
        for variant_index, variant in enumerate(
            sorted(stage.variants, key=lambda item: (tuple(sorted(item.key.items())), item.index))
        ):
            for name, value in sorted(variant.key.items()):
                path = f"{stage_path}.variants[{variant_index}].key.{name}"
                use(
                    name,
                    path,
                    classes={"depth", "regs", "variant"},
                    bindings={"bind", "fixed"},
                )
                variant_values.setdefault(name, set()).add(value)
        for name, values in sorted(variant_values.items()):
            constrain(name, f"{stage_path}.variants.{name}", values)

        for endpoint in sorted(stage.endpoints, key=lambda item: item.name):
            endpoint_path = f"{stage_path}.endpoints.{endpoint.name}"
            if endpoint.ring is not None and endpoint.ring.depth.handle is not None:
                use(
                    endpoint.ring.depth.handle,
                    f"{endpoint_path}.ring.depth.handle",
                    classes={"depth"},
                    bindings={"bind"},
                )
            for bind_role, name in sorted(endpoint.binds.items()):
                domains = {"mbarrier"} if bind_role in {"full", "empty"} else None
                use(
                    name,
                    f"{endpoint_path}.binds.{bind_role}",
                    classes={"base"},
                    bindings={"bind"},
                    domains=domains,
                )
            protocol = endpoint.protocol
            if isinstance(protocol.full, BarrierSide):
                use(
                    protocol.full.slots,
                    f"{endpoint_path}.protocol.full.slots",
                    classes={"base"},
                    bindings={"bind"},
                    domains={"mbarrier"},
                )
            if protocol.empty is not None:
                use(
                    protocol.empty.slots,
                    f"{endpoint_path}.protocol.empty.slots",
                    classes={"base"},
                    bindings={"bind"},
                    domains={"mbarrier"},
                )
            if protocol.named_barrier is not None:
                use(
                    protocol.named_barrier.id_handle,
                    f"{endpoint_path}.protocol.id",
                    classes={"base"},
                    bindings={"bind"},
                    domains={"named_barrier"},
                )
            if protocol.counter is not None:
                use(
                    protocol.counter.slot_handle,
                    f"{endpoint_path}.protocol.slot",
                    classes={"base"},
                    bindings={"bind"},
                    domains={"gmem_counter"},
                )

        for claim in sorted(stage.claims, key=lambda item: (item.domain, item.handle)):
            claim_path = f"{stage_path}.claims.{claim.domain}.{claim.handle}"
            domain = profile.domains.get(claim.domain)
            if (
                domain is not None
                and domain.capacity == "pipeline"
                and claim.domain not in pipeline.domains
            ):
                fail(
                    f"{claim_path}.domain",
                    "claim uses a pipeline-capacity domain omitted from pipeline.domains",
                    domain=claim.domain,
                )
            if claim.domain == "regs":
                use(
                    claim.handle,
                    f"{claim_path}.handle",
                    classes={"regs"},
                    bindings={"bind"},
                    domains={"regs"},
                )
                if claim.values is not None:
                    constrain(claim.handle, f"{claim_path}.values", set(claim.values))
            else:
                use(
                    claim.handle,
                    f"{claim_path}.handle",
                    classes={"base"},
                    bindings={"bind"},
                    domains={claim.domain},
                )
            if isinstance(claim.lifetime, TensorLifetime):
                lifetime = claim.lifetime
                tensor = stage.tensor
                exports = [item for item in tensor.exports if item.name == lifetime.tensor] if tensor else []
                imports = [item for item in tensor.imports if item.name == lifetime.tensor] if tensor else []
                if len(exports) + len(imports) != 1:
                    fail(f"{claim_path}.lifetime.tensor", "lifetime must name one own-stage tensor endpoint",
                         clause_id="L-PHASE", tensor=lifetime.tensor)
                else:
                    if exports:
                        endpoint = exports[0]
                        events = {endpoint.identity.work: 0, endpoint.readiness.after: 1,
                                  endpoint.readiness.reuse_after: 2}
                    else:
                        endpoint = imports[0]
                        events = {site.invocation: 0 for site in tensor.sites if site.name in endpoint.sites}
                        events.update({item.event: 1 for item in endpoint.rendezvous
                                       if item.event != endpoint.completion})
                        events[endpoint.completion] = 2
                    for field_name, event in (("from", lifetime.from_), ("to", lifetime.to)):
                        if event is not None and event not in events:
                            fail(f"{claim_path}.lifetime.{field_name}", "unknown tensor endpoint event",
                                 clause_id="L-PHASE", event=event)
                    if lifetime.from_ in events and lifetime.to in events and events[lifetime.from_] >= events[lifetime.to]:
                        fail(f"{claim_path}.lifetime", "tensor lifetime must start before it ends",
                             clause_id="L-PHASE")
            elif claim.lifetime.startswith("phase:"):
                phase = claim.lifetime.removeprefix("phase:")
                if phase not in pipeline.phases:
                    fail(
                        f"{claim_path}.lifetime",
                        "claim lifetime names an undeclared phase",
                        clause_id="L-PHASE",
                        phase=phase,
                    )
            if claim.extent is None:
                continue
            for name in sorted(claim.extent.per):
                use(
                    name,
                    f"{claim_path}.extent.per.{name}",
                    classes={"depth", "variant"},
                    bindings={"bind", "fixed"},
                )
            table_values: dict[str, set[int]] = {}
            for key in sorted(claim.extent.table):
                for component in key.split(","):
                    name, raw_value = component.split("=", 1)
                    use(
                        name,
                        f"{claim_path}.extent.table.{key}",
                        classes={"depth", "variant"},
                        bindings={"bind", "fixed"},
                    )
                    table_values.setdefault(name, set()).add(int(raw_value))
            for name, values in sorted(table_values.items()):
                constrain(name, f"{claim_path}.extent.table.{name}", values)

    for name in sorted(uses):
        declaration = declarations.get(name)
        if declaration is None:
            for path, _classes, _bindings, _domains in uses[name]:
                fail(path, "referenced pipelink_* handle is not declared in pipeline scope", handle=name)
            continue
        required_classes: set[str] | None = None
        required_bindings: set[str] | None = None
        required_domains: set[str] | None = None
        for path, classes, bindings, domains in uses[name]:
            required_classes = set(classes) if required_classes is None else required_classes & classes
            required_bindings = set(bindings) if required_bindings is None else required_bindings & bindings
            if domains:
                required_domains = set(domains) if required_domains is None else required_domains & domains
            if declaration.cls not in classes:
                fail(
                    path,
                    "handle class disagrees with its reference",
                    handle=name,
                    declared=declaration.cls,
                    expected=sorted(classes),
                )
            if declaration.binding not in bindings:
                fail(
                    path,
                    "handle binding disagrees with its reference",
                    handle=name,
                    declared=declaration.binding,
                    expected=sorted(bindings),
                )
            if domains and declaration.domain not in domains:
                fail(
                    path,
                    "handle domain disagrees with its reference",
                    handle=name,
                    declared=declaration.domain,
                    expected=sorted(domains),
                )
        if required_classes == set():
            fail(f"pipeline.handles.{name}", "shared references disagree on handle class")
        if required_bindings == set():
            fail(f"pipeline.handles.{name}", "shared references disagree on handle binding")
        if required_domains == set():
            fail(f"pipeline.handles.{name}", "shared references disagree on handle domain")

    for name in sorted(value_constraints):
        declaration = declarations.get(name)
        if declaration is None:
            continue
        declared_values = set(declaration.values or [])
        for path, values in value_constraints[name]:
            unsupported = sorted(values - declared_values)
            if unsupported:
                fail(
                    path,
                    "handle reference contains a value absent from its declaration",
                    handle=name,
                    values=unsupported,
                )
        sets = [declared_values, *(values for _path, values in value_constraints[name])]
        if not declared_values or not set.intersection(*sets):
            fail(
                f"pipeline.handles.{name}.values",
                "shared handle value constraints have an empty intersection",
            )

    from .host.contracts import validate_host_assembly
    for failure in validate_host_assembly(pipeline, stages):
        failures[(failure.clause_id, failure.path, failure.message)] = failure

    return sorted(failures.values(), key=lambda item: (item.clause_id, item.path, item.message))


__all__ = ["load_pipeline", "load_stage_manifest", "validate_assembly"]
