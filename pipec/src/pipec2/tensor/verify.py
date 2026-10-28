"""Exact comparison of placement annotations with the owner's layout (no GPU instructions are checked)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Literal

from pipec2.model import _JsonMixin

from . import expr


@dataclass(frozen=True)
class MapMismatch(_JsonMixin):
    coordinates: dict[str, int]
    bindings: dict[str, int]
    expected: dict[str, int] | None
    declared: dict[str, int]
    reason: Literal["value", "missing_observation"]


@dataclass(frozen=True)
class VerifyResult(_JsonMixin):
    ok: bool
    checked_points: int
    ground_entries: dict[str, int]
    mismatch_count: int
    mismatches: tuple[MapMismatch, ...]


def _references(expression: expr.Expr) -> set[str]:
    if expression.op in {"coord", "bind"}:
        return {expression.args[0]}
    return set().union(*(_references(arg) for arg in expression.args if isinstance(arg, expr.Expr)))


def verify_export_map(
    placements: Mapping[str, expr.Expr],
    domains: expr.SortEnv,
    bindings: expr.SortEnv,
    observations: Mapping[tuple[int, ...], Mapping[str, int]],
    *,
    max_mismatches: int = 8,
) -> VerifyResult:
    """Compare one export placement's components with an independent owner table.

    ``placements`` is a parsed ``PlacementView.components`` mapping. Coordinate
    ``domains`` and ``bindings`` are the declared, disjoint finite domains (not
    inferred from the observations). Observation keys follow lexical name order
    across BOTH mappings; values contain the observed physical components. The
    caller decodes and relocates owner-specific addresses, and must derive the
    observations from the owner, never by evaluating the declared expressions.

    Grounding follows C1's lexical referenced-domain order; comparison covers the
    full declared Cartesian product, including unreferenced dimensions and every
    binding value. A missing observation counts as a failure. The whole product
    is checked and counted, but only the first ``max_mismatches`` witnesses are
    kept. ``expected`` is the owner value and ``declared`` the grounded annotation
    value; a missing row has ``expected=None`` and names its exact coordinate and
    binding values.

    Inputs must already be sort/domain checked; contract and relation validation
    is the caller's job. Undefined expressions raise ``expr.ground``'s errors.
    """
    if type(max_mismatches) is not int or max_mismatches < 0:
        raise ValueError("max_mismatches must be a nonnegative integer")
    all_domains = {**domains, **bindings}
    names = sorted(all_domains)
    indices = {name: index for index, name in enumerate(names)}
    values = [
        domain.values if domain.values is not None else range(domain.lo, domain.hi)
        for name in names
        for domain in (all_domains[name],)
    ]
    grounded = {
        name: (tuple(indices[ref] for ref in sorted(_references(expression))),
               expr.ground(expression, all_domains))
        for name, expression in placements.items()
    }
    mismatches = []
    checked_points = mismatch_count = 0
    for point in product(*values):
        checked_points += 1
        declared = {
            name: table[tuple(point[index] for index in refs)]
            for name, (refs, table) in grounded.items()
        }
        expected = observations.get(point)
        if expected == declared:
            continue
        mismatch_count += 1
        if len(mismatches) < max_mismatches:
            mismatches.append(MapMismatch(
                coordinates={name: point[indices[name]] for name in sorted(domains)},
                bindings={name: point[indices[name]] for name in sorted(bindings)},
                expected=dict(expected) if expected is not None else None,
                declared=declared,
                reason="value" if expected is not None else "missing_observation",
            ))
    return VerifyResult(
        ok=mismatch_count == 0,
        checked_points=checked_points,
        ground_entries={name: len(table) for name, (_, table) in grounded.items()},
        mismatch_count=mismatch_count,
        mismatches=tuple(mismatches),
    )


__all__ = ["MapMismatch", "VerifyResult", "verify_export_map"]
