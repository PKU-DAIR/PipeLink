"""Exact, deterministic finite-table fitting; fitting is not certification."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Literal

from pipec2.model import ClauseFailure, PipeLinkError

from . import expr


def _node(op, *args, sort="int"):
    return expr.Expr(op, args, sort)


def _reference(name):
    return _node("coord" if name in {"participant", "slot"} else "bind", name)


def _affine(rows, names):
    basis = {}
    mismatch = None
    for point, value in rows:
        row = [Fraction(1), *(Fraction(v) for v in point), Fraction(value)]
        for pivot, previous in sorted(basis.items()):
            if row[pivot]:
                factor = row[pivot]
                row = [a - factor * b for a, b in zip(row, previous)]
        pivot = next((i for i, v in enumerate(row[:-1]) if v), None)
        if pivot is None:
            if row[-1]:
                mismatch = {"inputs": list(point), "expected": value, "residual": str(row[-1])}
                break
        else:
            factor = row[pivot]
            basis[pivot] = [v / factor for v in row]
    if mismatch:
        return None, mismatch
    solution = [Fraction(0)] * (len(names) + 1)
    for pivot, row in sorted(basis.items(), reverse=True):
        solution[pivot] = row[-1] - sum(row[i] * solution[i] for i in range(pivot + 1, len(solution)))
    if any(v.denominator != 1 for v in solution):
        return None, {"reason": "noninteger_coefficients", "coefficients": [str(v) for v in solution]}
    terms = [_node("mulc", int(coefficient), _reference(name)) if coefficient != 1 else _reference(name)
             for name, coefficient in zip(names, solution[1:]) if coefficient]
    if solution[0] or not terms:
        terms.append(_node("const", int(solution[0])))
    result = terms[0]
    for term in terms[1:]:
        result = _node("add", result, term)
    return result, None


def _integer_bitfields(solution, inputs):
    """Render nonmixing GF(2) output bits as maximal integer fields."""
    sources = {}
    offset = 1
    for name, input_width in inputs:
        for source in range(input_width):
            output = solution[offset + source]
            while output:
                target = (output & -output).bit_length() - 1
                if target in sources:
                    return None  # This output bit XORs multiple input bits.
                sources[target] = (name, source, input_width)
                output &= output - 1
        offset += input_width
    constant = solution[0]
    terms = []
    end = 0
    for target in sorted(sources):
        if target < end:
            continue
        name, source, input_width = sources[target]
        inverted = (constant >> target) & 1
        width = 1
        while (following := sources.get(target + width)) is not None:
            if (following[0] != name or following[1] != source + width
                    or ((constant >> (target + width)) & 1) != inverted):
                break
            width += 1
        end = target + width
        term = _reference(name)
        if source:
            term = _node("divc", term, 1 << source)
        if source + width < input_width:
            term = _node("modc", term, 1 << width)
        # A constant on a copied bit complements it: c XOR x = c - x,
        # not c + x. Split runs at changes of polarity to preserve this.
        coefficient = -(1 << target) if inverted else 1 << target
        if coefficient != 1:
            term = _node("mulc", coefficient, term)
        terms.append(term)
    if constant or not terms:
        terms.append(_node("const", constant))
    result = terms[0]
    for term in terms[1:]:
        result = _node("add", result, term)
    return result


def _linear(rows, inputs):
    # The RHS packs all output-bit equations. Row operations therefore solve
    # each independent output bit simultaneously, without a matrix per bit.
    basis = {}
    for point, value in rows:
        bits, offset = 1, 1
        for coordinate, (_, width) in zip(point, inputs):
            if coordinate < 0 or coordinate.bit_length() > width or value < 0:
                return None, {"inputs": list(point), "expected": value, "reason": "outside_unsigned_template"}
            bits |= coordinate << offset
            offset += width
        rhs = value
        while bits:
            pivot = (bits & -bits).bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (bits, rhs)
                break
            previous, target = basis[pivot]
            bits ^= previous
            rhs ^= target
        if not bits and rhs:
            return None, {"inputs": list(point), "expected": value, "actual": value ^ rhs}
    count = 1 + sum(width for _, width in inputs)
    solution = [0] * count
    for pivot, (bits, rhs) in sorted(basis.items(), reverse=True):
        bits ^= 1 << pivot
        while bits:
            bit = (bits & -bits).bit_length() - 1
            rhs ^= solution[bit]
            bits &= bits - 1
        solution[pivot] = rhs
    result = _integer_bitfields(solution, inputs)
    if result is not None:
        return result, None
    width = max(1, *(w for _, w in inputs), *(value.bit_length() for _, value in rows))
    sort = f"bits[{width}]"
    terms = []
    if solution[0]:
        terms.append(_node("cbits", width, solution[0], sort=sort))
    offset = 1
    for name, input_width in inputs:
        shifts = {}
        for bit in range(input_width):
            output = solution[offset + bit]
            while output:
                destination = (output & -output).bit_length() - 1
                shift = destination - bit
                shifts[shift] = shifts.get(shift, 0) | (1 << bit)
                output &= output - 1
        reference = _node("tobits", width, _reference(name), sort=sort)
        for shift, mask in sorted(shifts.items()):
            term = _node("band", reference, _node("cbits", width, mask, sort=sort), sort=sort)
            if shift:
                term = _node("shlc" if shift > 0 else "shrc", term, abs(shift), sort=sort)
            terms.append(term)
        offset += input_width
    if not terms:
        return _node("const", 0), None
    result = terms[0]
    for term in terms[1:]:
        result = _node("bxor", result, term, sort=sort)
    return _node("toint", result), None


def _candidates(rows, inputs):
    # Plain integer forms are cheaper to lower; reserve bit operations for XOR mixing.
    yield "integer-affine", _affine(rows, tuple(name for name, _ in inputs))
    yield "gf2-linear", _linear(rows, inputs)


def fit_expression(rows: Mapping[tuple[int, ...], int], inputs: Sequence[tuple[str, int]],
                   *, failures: list[dict] | None = None) -> expr.Expr | None:
    """Fit integer-affine, then GF(2)-linear, and verify every supplied row.

    Input tuple positions follow ``inputs``; elimination order is lexical by
    input name and row. ``participant``/``slot`` are coordinates, other names
    are bindings. Optional diagnostics retain the first inconsistent row of
    each failed template, without mutable global fitter state.
    """
    if not rows:
        return None
    order = sorted(range(len(inputs)), key=lambda i: inputs[i][0])
    ordered_inputs = tuple(inputs[i] for i in order)
    if len({name for name, _ in inputs}) != len(inputs) or any(width <= 0 for _, width in inputs):
        raise ValueError("fitter inputs require unique names and positive bit widths")
    table = sorted((tuple(point[i] for i in order), value) for point, value in rows.items())
    names = tuple(name for name, _ in ordered_inputs)
    for template, (result, mismatch) in _candidates(table, ordered_inputs):
        if result is not None:
            for point, expected in table:
                actual = expr.evaluate(result, dict(zip(names, point)))
                if actual != expected:
                    mismatch = {"inputs": list(point), "expected": expected, "actual": actual}
                    break
            if mismatch is None:
                return result
        if failures is not None:
            failures.append({"template": template, "input_order": list(names), "first_mismatch": mismatch})
    return None


class FitError(PipeLinkError):
    """A named map has no exact expression in the supported fitting families."""


def _named_tree(node: expr.Expr, references: Mapping[str, list]) -> list:
    if node.op in {"coord", "bind"}:
        return references[node.args[0]]
    return [node.op, *(_named_tree(arg, references) if isinstance(arg, expr.Expr) else arg
                       for arg in node.args)]


def _fit_named(
    rows: Sequence[Sequence[int]],
    inputs: Sequence[tuple[expr.Domain, Literal["coord", "bind"]]],
    output_index: int,
    sort_env: expr.SortEnv,
) -> tuple[expr.Expr | None, list[dict]]:
    if not rows:
        return None, []
    if type(output_index) is not int or output_index < 0:
        raise ValueError("output_index must be a nonnegative output-column index")
    names = tuple(domain.name for domain, _ in inputs)
    if len(set(names)) != len(names) or any(role not in {"coord", "bind"} for _, role in inputs):
        raise ValueError("fitter inputs require unique names and coord or bind roles")
    references = {}
    widths = []
    for domain, role in inputs:
        reference = [role, domain.name]
        references[domain.name] = reference if domain.sort == "int" else ["toint", reference]
        maximum = max(domain.values, default=0) if domain.values is not None else domain.hi - 1
        widths.append((domain.name, max(1, maximum.bit_length())))
    column = len(inputs) + output_index
    table = [(tuple(row[:len(inputs)]), row[column]) for row in rows]
    failures = []
    for template, (result, mismatch) in _candidates(table, widths):
        if result is not None:
            result = expr.parse(_named_tree(result, references), sort_env)
            for point, expected in table:
                actual = expr.evaluate(result, dict(zip(names, point)))
                if actual != expected:
                    mismatch = {"inputs": list(point), "expected": expected, "actual": actual}
                    break
            if mismatch is None:
                return result, []
        row_index = None
        if mismatch is not None and "inputs" in mismatch:
            point = tuple(mismatch["inputs"])
            row_index = next(index for index, (values, expected) in enumerate(table)
                             if values == point and expected == mismatch["expected"])
        failures.append({"template": template, "input_order": list(names),
                         "row_index": row_index, "first_mismatch": mismatch})
    return None, failures


def fit_named(
    rows: Sequence[Sequence[int]],
    inputs: Sequence[tuple[expr.Domain, Literal["coord", "bind"]]],
    output_index: int,
    *,
    sort_env: expr.SortEnv,
) -> expr.Expr | None:
    """Fit a named map's zero-based output column, exactly over every row.

    Rows contain input columns followed by output columns. Input roles, never
    their spellings, determine reference namespaces. Templates use the supplied
    row order and try integer-affine before GF(2)-linear. Nonmixing bit fields
    use integer operations; XOR mixing keeps bit operations. The returned carrier
    is sort-checked and bits-valued references are explicitly converted to integers.
    """
    result, _ = _fit_named(rows, inputs, output_index, sort_env)
    return result


def fit_named_or_raise(
    rows: Sequence[Sequence[int]],
    inputs: Sequence[tuple[expr.Domain, Literal["coord", "bind"]]],
    output_index: int,
    *,
    sort_env: expr.SortEnv,
    output_name: str,
) -> expr.Expr:
    """Fit once or report the first failure of the longest-consistent-prefix template."""
    result, failures = _fit_named(rows, inputs, output_index, sort_env)
    if result is not None:
        return result
    detail = {"output": output_name, "template_failures": failures}
    if failures:
        best = max(failures, key=lambda failure: failure["row_index"]
                   if failure["row_index"] is not None else -1)
        detail.update(best)
        index = best["row_index"]
        reason = (f"{best['template']} first fails at row {index}: {list(rows[index])}"
                  if index is not None else f"{best['template']}: {best['first_mismatch']}")
    else:
        reason = "table is empty"
    raise FitError([ClauseFailure("T-FIT", "$", f"cannot fit output {output_name!r}: {reason}", detail)])
