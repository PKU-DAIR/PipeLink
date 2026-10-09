"""Caller-owned tensor loans and runner-side fixture fabrication.

Providers receive only metadata and bounded addresses. Python ownership, input
fabrication, launch-stream readiness and output readback stay on this side of
the C seam and outside the session's launch path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple, Sequence


class Domain(NamedTuple):
    """This rank's place in one caller-named communication domain (``open_host(domains=)``)."""
    rank: int
    world: int


@dataclass(frozen=True)
class Resource:
    address: int
    allocation_bytes: int
    extents: tuple[int, ...]
    bit_strides: tuple[int, ...]
    dtype_id: int
    alignment: int
    origin_byte_offset: int = 0
    origin_bit_offset: int = 0
    # Symmetric loans: `domain` names the caller domain this allocation is symmetric over
    # (one allocation per domain rank, same bytes and origin). `peers` is () or one base per
    # domain rank addressable from this rank's context (peers[rank] == address); `multicast`
    # is 0 or this allocation's base in the domain's multicast VA space. Local loans keep
    # all three empty. Keyword-only so positional construction cannot shift a lender.
    domain: str | None = field(default=None, kw_only=True)
    peers: tuple[int, ...] = field(default=(), kw_only=True)
    multicast: int = field(default=0, kw_only=True)
    lender: Any = field(default=None, repr=False, compare=False, kw_only=True)


class Fixture:
    """Own fabricated caller buffers until the enclosing host session has closed.

    Data generation is supplied by the experiment/caller, never a provider DSO.
    A Fixture is scoped inside its CUDA backend and outside open_host(). Inputs
    are uploaded synchronously at fabrication, so they are ready on whichever
    stream the caller launches on; outputs are read back by the caller through
    readback() after it synchronized.
    """

    def __init__(self, backend: Any):
        self.backend = backend
        self.resources: dict[str, Resource] = {}
        self.outputs: dict[str, bytes] = {}
        self._outputs: list[str] = []
        self._allocations: list[int] = []

    def tensor(self, name: str, *, data: bytes | None, allocation_bytes: int,
               extents: tuple[int, ...], bit_strides: tuple[int, ...],
               dtype_id: int, alignment: int,
               output: bool = False, origin_byte_offset: int = 0,
               origin_bit_offset: int = 0) -> Resource:
        if name in self.resources:
            raise ValueError(f"duplicate fixture resource {name!r}")
        if allocation_bytes < 0 or (data is not None and len(data) != allocation_bytes):
            raise ValueError("fixture data must cover the declared allocation")
        address = self.backend.allocate(allocation_bytes, alignment)
        self._allocations.append(address)
        if data:
            self.backend.upload(address, data)
        resource = Resource(
            address, allocation_bytes, extents, bit_strides, dtype_id, alignment,
            origin_byte_offset, origin_bit_offset, lender=self,
        )
        self.resources[name] = resource
        if output:
            self._outputs.append(name)
        return resource

    def readback(self) -> dict[str, bytes]:
        """Copy every output allocation to host memory; the caller synchronized first."""
        for name in self._outputs:
            resource = self.resources[name]
            self.outputs[name] = self.backend.readback(resource.address, resource.allocation_bytes)
        return self.outputs

    def __enter__(self) -> Fixture:
        return self

    def __exit__(self, *_: Any) -> None:
        # A session drains before close returns, including its failure paths. An
        # undrainable context is torn down, never treated as reusable allocation.
        if not getattr(self.backend, "lost", False):
            self.backend.synchronize()
            for address in reversed(self._allocations):
                self.backend.free(address)
        self._allocations.clear()


def symmetric(fixtures: Sequence[Fixture], name: str, *, domain: str, data: Sequence[bytes | None],
              allocation_bytes: int, extents: tuple[int, ...], bit_strides: tuple[int, ...], dtype_id: int,
              alignment: int, output: bool = False) -> list[Resource]:
    """Fabricate one allocation per rank of `domain` (domain rank r = fixtures[r]) with a peer table.

    Each rank's ``Fixture`` allocates and uploads its own buffer; the returned
    per-rank resources (also recorded in each fixture) share one peer table of
    the ranks' addresses. In-process callers must have enabled peer access
    (``CudaBackend.enable_peer_access``); the table is unified virtual addresses.
    """
    if len(data) != len(fixtures):
        raise ValueError("symmetric fixtures need one data entry per rank")
    local = [fixture.tensor(name, data=item, allocation_bytes=allocation_bytes, extents=extents,
                            bit_strides=bit_strides, dtype_id=dtype_id, alignment=alignment, output=output)
             for fixture, item in zip(fixtures, data, strict=True)]
    peers = tuple(resource.address for resource in local)
    result = [replace(resource, domain=domain, peers=peers) for resource in local]
    for fixture, resource in zip(fixtures, result, strict=True):
        fixture.resources[name] = resource
    return result


def load_inputs(path: Path, fixture: Fixture) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load explicit caller fixtures for ``pipec2 launch --host-inputs``.

    Paths are relative to the input JSON. Every tensor declares its complete
    allocation metadata; absent input data is an output allocation, not an
    implicit random fixture. Returns the scalars and, for every resource that
    names an ``output`` file, the resolved path the caller writes from
    ``Fixture.readback()`` once its launches have drained.
    """
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict) or set(obj) != {"scalars", "resources"}:
        raise ValueError("host inputs require exactly scalars and resources objects")
    if not isinstance(obj["scalars"], dict) or not isinstance(obj["resources"], dict):
        raise ValueError("host scalars/resources must be objects")
    required = {"data", "allocation_bytes", "extents", "bit_strides", "dtype_id",
                "alignment", "origin_byte_offset", "origin_bit_offset"}
    output_paths: dict[str, Path] = {}
    for name, item in obj["resources"].items():
        if not isinstance(item, dict) or not required <= set(item) or set(item) - required - {"output"}:
            raise ValueError(f"invalid fixture declaration for {name}")
        data = None if item["data"] is None else (path.parent / item["data"]).read_bytes()
        output = item.get("output")
        fixture.tensor(
            name, data=data, allocation_bytes=item["allocation_bytes"],
            extents=tuple(item["extents"]), bit_strides=tuple(item["bit_strides"]),
            dtype_id=item["dtype_id"], alignment=item["alignment"],
            origin_byte_offset=item["origin_byte_offset"],
            origin_bit_offset=item["origin_bit_offset"], output=output is not None,
        )
        if output is not None:
            output_paths[name] = path.parent / output
    return obj["scalars"], output_paths
