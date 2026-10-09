from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .model import ClauseFailure, ProfileError, _JsonMixin


@dataclass(frozen=True)
class Domain(_JsonMixin):
    units: str
    capacity: int | Literal["pipeline"]
    lower_bound: int
    granularity: int
    binding_class: str
    physical_domain: str | None
    unit_scale: int | None


@dataclass(frozen=True)
class TargetCaps(_JsonMixin):
    param_limit_bytes: int
    param_align: int
    max_threads: int
    max_cluster: tuple[int, int, int]
    cta_group_legal: set[int]
    regs_min: int = field(default=24, kw_only=True)
    regs_max: int = field(default=256, kw_only=True)
    regs_step: int = field(default=8, kw_only=True)
    launch_regs_max: int = field(default=255, kw_only=True)
    reg_alloc_granularity: int
    smem_capacity_dynamic: int
    warpgroup_warps: int = field(default=4, kw_only=True)


@dataclass(frozen=True)
class AdapterCompile(_JsonMixin):
    """Adapter architecture route and environment variables naming include directories."""

    gencode: tuple[str, str]
    forced_macros: tuple[str, ...]
    include_roots_env: tuple[str, ...]
    toolkit: str


@dataclass(frozen=True)
class Arch:
    """Toolchain spellings of one profile's architecture, derived from the profile only."""

    sm: str  # nvcc code=sm_*, nvJitLink -arch (sm_103a)
    compute: str  # nvcc arch=compute_* (compute_103a)
    lto: str  # arch-conditional LTO payload/adapter route and target stamp (lto_103a)
    lto_base: str  # architecture-neutral LTO for probes and stubs (lto_103)


@dataclass(frozen=True)
class Profile(_JsonMixin):
    name: str
    lto_arch: str
    toolkit: str
    tmem_lanes: int | None
    domains: dict[str, Domain]
    atoms: dict[str, dict[str, Any]]
    caps: TargetCaps
    adapter_compile: AdapterCompile
    catalogue: str
    schedulers: frozenset[str]
    scheduler_strategies: frozenset[str]
    fairness: tuple[str, ...] = (
        "owner_eventually_ready", "cta_eventually_scheduled", "target_eventually_completes",
    )

    @property
    def arch(self) -> Arch:
        compute, lto = self.adapter_compile.gencode
        return Arch(self.name, compute, lto, self.lto_arch)


_NOT_IMPLEMENTED_ATOM: dict[str, Any] = {
    "encoding": "pl-layout-v2",
    "expansion": "NotImplemented",
}


# CuTe's flagged atoms use byte Swizzle<B,4,3>. These bf16 expansions use the
# carrier's default element units: M = 4 - log2(2). The same Swizzle<3,3,3> /
# Swizzle<2,3,3> element-unit atoms appear on both Blackwell (sm103) and Hopper
# (sm90) under wgmma-neutral names, so the atom maps are shared by reference.
_SW_BF16_ATOMS: dict[str, dict[str, Any]] = {
    "sw128_k_bf16": {
        "encoding": "pl-layout-v2",
        "elem_bits": 16,
        "shape": [[8], [64]],
        "stride": [[64], [1]],
        "swizzle": {"B": 3, "M": 3, "S": 3},
    },
    "sw64_mn_bf16": {
        "encoding": "pl-layout-v2",
        "elem_bits": 16,
        "shape": [[32], [8]],
        "stride": [[1], [32]],
        "swizzle": {"B": 2, "M": 3, "S": 3},
    },
}


SM_103A = Profile(
    name="sm_103a",
    lto_arch="lto_103",
    toolkit="cuda-13.3.0",
    tmem_lanes=128,
    domains={
        "smem": Domain("bytes", 232448, 0, 1, "bind", "smem", 1),
        "mbarrier": Domain("16-byte slots", 232448, 0, 1, "bind", "smem", 16),
        "tmem": Domain("columns", 512, 0, 32, "bind", "tmem", 1),
        "named_barrier": Domain("IDs", 16, 1, 1, "bind", "named_barrier", 1),
        "warp": Domain("launch warps", 32, 0, 1, "bind", "warp", 1),
        "regs": Domain("per-thread registers", 256, 24, 8, "bind", "regs", 1),
        "gmem_counter": Domain("4-byte slots", "pipeline", 0, 1, "bind", "gmem_counter", 1),
        "instr_page": Domain("pages", "pipeline", 0, 1, "bind", "instr_page", 1),
        "instr_semaphore": Domain(
            "8-byte semaphores", "pipeline", 0, 1, "bind", "instr_semaphore", 1
        ),
        "instr_scratch": Domain("bytes", "pipeline", 0, 1, "bind", "instr_scratch", 1),
        "opcode": Domain("opcodes", "pipeline", 0, 1, "bind", "opcode", 1),
        "smem_page": Domain("pages", "pipeline", 0, 1, "bind", "smem", None),
    },
    atoms={
        **_SW_BF16_ATOMS,
        "tma_interleave_bf16": dict(_NOT_IMPLEMENTED_ATOM),
        "umma_acc_m128n256": dict(_NOT_IMPLEMENTED_ATOM),
    },
    caps=TargetCaps(
        param_limit_bytes=32764,
        param_align=64,
        max_threads=1024,
        max_cluster=(8, 8, 8),
        cta_group_legal={1, 2},
        reg_alloc_granularity=8,
        smem_capacity_dynamic=232448,
    ),
    adapter_compile=AdapterCompile(
        gencode=("compute_103a", "lto_103a"),
        forced_macros=(),
        include_roots_env=("CUTLASS_INCLUDE_DIR", "TILELANG_INCLUDE_DIR"),
        toolkit="cuda-13.3.0",
    ),
    catalogue="tcgen05-readonly-v1",
    schedulers=frozenset({"clc", "static", "stream_k"}),
    scheduler_strategies=frozenset({"clc-sm100-v1"}),
)


# Hopper (CC 9.0): no TMEM, no CLC, no 2-SM cta_group. Same dynamic smem, register
# and cluster limits as sm103. CuTe's Layout_K_SW128_Atom<bf16> / MN_SW64 are the same
# Swizzle<3,3,3> / Swizzle<2,3,3> element-unit atoms on Hopper, so the shared
# _SW_BF16_ATOMS entries are referenced (not copied) under the wgmma-neutral
# sm90 atoms.
SM_90A = Profile(
    name="sm_90a",
    lto_arch="lto_90",
    toolkit="cuda-13.3.0",
    tmem_lanes=None,
    domains={
        name: domain for name, domain in SM_103A.domains.items() if name != "tmem"
    },
    atoms=dict(_SW_BF16_ATOMS),
    caps=TargetCaps(
        param_limit_bytes=32764,
        param_align=64,
        max_threads=1024,
        max_cluster=(8, 8, 8),
        cta_group_legal={1},
        reg_alloc_granularity=8,
        smem_capacity_dynamic=232448,
    ),
    adapter_compile=AdapterCompile(
        gencode=("compute_90a", "lto_90a"),
        forced_macros=(),
        include_roots_env=("CUTLASS_INCLUDE_DIR", "TILELANG_INCLUDE_DIR"),
        toolkit="cuda-13.3.0",
    ),
    catalogue="warp-readonly-v1",
    schedulers=frozenset({"static", "stream_k"}),
    scheduler_strategies=frozenset(),
)


# Consumer Blackwell (CC 12.0): no TMEM/tcgen05, no wgmma (warp-level mma.sync), no 2-SM
# cta_group; 101376 B opt-in dynamic smem per block, shared by the smem and mbarrier domains.
# Thread block clusters are a CC 12.x feature (CUDA Programming Guide, compute capabilities
# table); the portable limit is 8, as on sm90. CLC stays unadmitted: ptxas accepts it, but the
# only CLC strategy is an sm100 host schema.
_SM120_SMEM = 101376
SM_120A = Profile(
    name="sm_120a",
    lto_arch="lto_120",
    toolkit="cuda-13.3.0",
    tmem_lanes=None,
    domains={
        **SM_90A.domains,
        "smem": Domain("bytes", _SM120_SMEM, 0, 1, "bind", "smem", 1),
        "mbarrier": Domain("16-byte slots", _SM120_SMEM, 0, 1, "bind", "smem", 16),
    },
    atoms=dict(_SW_BF16_ATOMS),
    caps=TargetCaps(
        param_limit_bytes=32764,
        param_align=64,
        max_threads=1024,
        max_cluster=(8, 8, 8),
        cta_group_legal={1},
        reg_alloc_granularity=8,
        smem_capacity_dynamic=_SM120_SMEM,
    ),
    adapter_compile=AdapterCompile(
        gencode=("compute_120a", "lto_120a"),
        forced_macros=(),
        include_roots_env=("CUTLASS_INCLUDE_DIR", "TILELANG_INCLUDE_DIR"),
        toolkit="cuda-13.3.0",
    ),
    catalogue="warp-readonly-v1",
    schedulers=frozenset({"static", "stream_k"}),
    scheduler_strategies=frozenset(),
)


# Siblings within a family. CC 10.0 and 12.1 share their sibling's limits (CUDA Programming Guide,
# compute-capabilities tables) and instruction vocabulary; an `a` image runs only on its exact CC,
# so each needs its own profile. Registered for pack/link, not evidenced on hardware.
def _sibling(profile: Profile, number: str) -> Profile:
    return replace(profile, name=f"sm_{number}a", lto_arch=f"lto_{number}", adapter_compile=replace(
        profile.adapter_compile, gencode=(f"compute_{number}a", f"lto_{number}a")))


SM_100A = _sibling(SM_103A, "100")
SM_121A = _sibling(SM_120A, "121")


PROFILES: dict[str, Profile] = {
    profile.name: profile for profile in (SM_103A, SM_100A, SM_90A, SM_120A, SM_121A)
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        failure = ClauseFailure(
            clause_id="L-TOPO",
            path="target.profile",
            message=f"unknown target profile {name!r}",
            detail={"available": sorted(PROFILES)},
        )
        raise ProfileError([failure]) from exc


__all__ = ["AdapterCompile", "Arch", "Domain", "Profile", "ProfileError", "TargetCaps", "get_profile"]
