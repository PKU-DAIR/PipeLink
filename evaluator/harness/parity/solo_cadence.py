from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

FAST_SOLO_REFRESH_SECONDS = 30.0
FAST_SOLO_REFRESH_NS = 30_000_000_000


class SoloCadenceError(RuntimeError):
    """Raised when a refreshed solo block violates its initial preassignment."""


@dataclass
class SoloRefreshCadence:
    """Monotonic, schedule-only FAST solo refresh state.

    Eligibility and throughput are intentionally absent from this interface: a caller
    can refresh only because elapsed monotonic time reached the fixed cadence.
    """

    initial_branches: dict[str, str]
    current_block_id: str
    current_block_sha256: str
    sealed_monotonic_ns: int
    refresh_index: int = 0

    @classmethod
    def begin(
        cls,
        branches: Mapping[str, str],
        block_id: str,
        block_sha256: str,
        sealed_monotonic_ns: int,
    ) -> "SoloRefreshCadence":
        if set(branches) != {"reference", "linked"}:
            raise ValueError("solo cadence requires exactly reference and linked branches")
        if not isinstance(sealed_monotonic_ns, int) or isinstance(sealed_monotonic_ns, bool):
            raise ValueError("solo cadence seal must be an integer monotonic timestamp")
        return cls(dict(branches), block_id, block_sha256, sealed_monotonic_ns)

    def refresh_due(self, monotonic_ns: int) -> bool:
        if not isinstance(monotonic_ns, int) or isinstance(monotonic_ns, bool):
            raise ValueError("solo cadence timestamp must be an integer")
        if monotonic_ns < self.sealed_monotonic_ns:
            raise ValueError("monotonic clock moved backwards")
        return monotonic_ns - self.sealed_monotonic_ns >= FAST_SOLO_REFRESH_NS

    def seal_refresh(
        self,
        branches: Mapping[str, str],
        block_id: str,
        block_sha256: str,
        sealed_monotonic_ns: int,
    ) -> None:
        if dict(branches) != self.initial_branches:
            raise SoloCadenceError("refreshed solo branch differs from initial preassignment")
        if sealed_monotonic_ns < self.sealed_monotonic_ns:
            raise ValueError("refreshed solo seal predates current solo block")
        self.refresh_index += 1
        self.current_block_id = block_id
        self.current_block_sha256 = block_sha256
        self.sealed_monotonic_ns = sealed_monotonic_ns

    def binding(self) -> dict[str, object]:
        return {
            "solo_block_id": self.current_block_id,
            "solo_block_sha256": self.current_block_sha256,
            "solo_refresh_index": self.refresh_index,
            "solo_refresh_cadence_seconds": FAST_SOLO_REFRESH_SECONDS,
        }
