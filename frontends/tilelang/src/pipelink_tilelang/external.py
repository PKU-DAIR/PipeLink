"""Records handed from contract helpers to externalization: what a role borrows from the link.

Every address a role uses for a partner-owned resource comes from a solved handle, never
from partner source. Barrier addressing is either the linker's 16-byte mbarrier slot
convention (``native is None``) or the partner's declared physical storage
(``protocol.<side>.native``): ``dyn_smem + solved(base) * unit(base.domain) + offset_bytes
+ slot * stride_bytes``.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class NativeStorage:
    base: str            # claim handle of the partner stage (domain smem or mbarrier)
    unit_bytes: int      # 16 for mbarrier-domain bases, 1 for smem-domain bases
    offset_bytes: int
    stride_bytes: int


@dataclass(frozen=True, slots=True)
class ExternalBarrier:
    """One TileLang barrier buffer standing in for a partner-owned barrier array."""
    buffer: str                      # TileLang buffer name allocated by the channel
    side: Literal["full", "empty"]
    handle: str                      # partner ``slots`` handle (16-byte slot convention)
    native: NativeStorage | None     # physical addressing when declared
    depth: int
    count: int | str                 # arrive count from the contract (int or role expression)
    init: Literal["drop", "keep"]    # keep = this role materializes init with ``count``
    phase: int                       # protocol init phase bit
    init_warp: int | None = None     # physical contract warp; None = role's first thread


@dataclass(frozen=True, slots=True)
class ExternalStorage:
    """A ring/plane/scalar buffer whose base is a solved partner handle.

    smem: ``buffer`` lives at ``dyn_smem + solved(handle)``.
    tmem: ``handle`` is the smem scalar endpoint publishing the runtime TMEM base pointer
    (e.g. the partner's ``tmem_base`` endpoint) and ``column_handle`` the solved tmem-domain
    column base (the accumulator endpoint's ``binds.base``); the arena word becomes
    ``*(u32*)(dyn_smem + solved(handle)) + solved(column_handle)``.
    """
    buffer: str
    handle: str
    space: Literal["smem", "tmem"]
    column_handle: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalThreadBase:
    """The role's first warp is a solved handle rather than warp 0."""
    handle: str
    warps: int
