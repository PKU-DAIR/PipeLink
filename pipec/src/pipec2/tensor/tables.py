"""Author tensor expression JSON from complete, ordered finite coordinate tables."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from itertools import product
from math import prod
from typing import Any

from .expr import SortEnv


def coordinate_points(domains: SortEnv) -> Iterator[tuple[int, ...]]:
    """Enumerate lexical axis names, preserving each domain's declared value order."""
    return product(*(_values(domains, name) for name in sorted(domains)))


def _values(domains: SortEnv, name: str) -> range | tuple[int, ...]:
    domain = domains[name]
    return domain.values if domain.values is not None else range(domain.lo, domain.hi)


def _encode_axis(values: Sequence[int], coordinate: str, indices: Sequence[int]) -> tuple[list[Any], str]:
    """Infer contiguous bit-permutation runs, otherwise encode the observed table."""
    coord = ["coord", coordinate]
    if all(value == index for index, value in zip(indices, values)):
        return coord, "executed identity"
    if len(set(values)) == 1:
        return ["const", values[0]], "executed constant"
    width = (len(values) - 1).bit_length()
    if len(values) == 1 << width and all(index == value for index, value in enumerate(indices)):
        basis = [values[1 << bit] for bit in range(width)]
        if (values[0] == 0 and
                all(value > 0 and value & (value - 1) == 0 for value in basis) and
                len(set(basis)) == width and
                all(value == sum(basis[bit] for bit in range(width) if index & (1 << bit))
                    for index, value in enumerate(values))):
            destinations = [value.bit_length() - 1 for value in basis]
            terms = []
            start = 0
            while start < width:
                stop = start + 1
                while stop < width and destinations[stop] == destinations[start] + stop - start:
                    stop += 1
                term = coord if start == 0 else ["divc", coord, 1 << start]
                if stop < width:
                    term = ["modc", term, 1 << (stop - start)]
                if destinations[start]:
                    term = ["mulc", 1 << destinations[start], term]
                terms.append(term)
                start = stop
            result = terms[0]
            for term in terms[1:]:
                result = ["add", result, term]
            return result, f"executed bit permutation {destinations}"
    result = ["const", values[-1]]
    for offset in range(len(indices) - 2, -1, -1):
        result = ["select", ["eq", coord, ["const", indices[offset]]], ["const", values[offset]], result]
    return result, "executed piecewise table"


def encode_component(
    values_by_coord: Mapping[tuple[int, ...], int], domains: SortEnv,
) -> tuple[list[Any], str]:
    """Encode a complete integer-valued table, returning expression JSON and its derivation.

    Keys follow lexical domain-name order, as in ``coordinate_points``; mapping
    insertion order is irrelevant. Domains are already checked ``Domain`` values
    of sort int, with at least one value per axis. Coordinate values need not be
    contiguous or zero-based. A table with the wrong cardinality is rejected.

    Prefer a single-axis identity, constant, or bit permutation. Otherwise build
    nested piecewise selections in declared value order, using each final value
    as the fallback anchor. No owner address interpretation belongs here.
    """
    names = tuple(sorted(domains))
    axes = tuple(_values(domains, name) for name in names)
    if any(domains[name].sort != "int" for name in names) or any(not axis for axis in axes):
        raise ValueError("table encoding requires nonempty integer coordinate domains")
    expected_size = prod(len(axis) for axis in axes)
    if len(values_by_coord) != expected_size:
        raise ValueError(f"table must contain exactly {expected_size} coordinate entries, got {len(values_by_coord)}")

    def encode(prefix: tuple[int, ...], depth: int) -> tuple[list[Any], str]:
        if depth == len(names):
            return ["const", values_by_coord[prefix]], "executed constant"
        remaining = axes[depth:]
        anchor = tuple(axis[0] for axis in remaining)
        for axis_index, axis in enumerate(remaining):
            values = {
                value: values_by_coord[prefix + anchor[:axis_index] + (value,) + anchor[axis_index + 1:]]
                for value in axis
            }
            if all(values_by_coord[prefix + point] == values[point[axis_index]] for point in product(*remaining)):
                return _encode_axis([values[value] for value in axis], names[depth + axis_index], axis)
        # Nonseparable maps remain expressible through nested finite selections.
        axis = axes[depth]
        rows = [encode(prefix + (value,), depth + 1)[0] for value in axis]
        result = rows[-1]
        for offset in range(len(axis) - 2, -1, -1):
            result = ["select", ["eq", ["coord", names[depth]], ["const", axis[offset]]], rows[offset], result]
        rank = "two" if len(remaining) == 2 else str(len(remaining))
        return result, f"executed {rank}-coordinate piecewise table"

    return encode((), 0)


__all__ = ["coordinate_points", "encode_component"]
