"""Internal scalar staging layouts, ranked on physical 32-lane bank accesses."""
from collections import Counter, defaultdict
from itertools import permutations

from . import expr, fit


def _cost(node):
    return 1 + sum(_cost(arg) for arg in node.args if isinstance(arg, expr.Expr))


def select(tables, scalar_bytes, extent, bindings):
    """Return fitted addresses for per-round (participant, slot, world...) tables.

    Shears preserve the byte-within-word and all word bits above bit four.
    Complete 128-byte blocks therefore retain their extent and bijection. The
    finite family is identity plus at most eight windows x 120 permutations.
    Scalar alignment bits are excluded from the bank-bit permutation; partial
    blocks retain identity. No candidate changes scalar alignment or extent.
    Each instruction's distinct words are counted, so broadcasts cost one.
    Bank-optimal candidates are tried by shear structural cost (XOR entries
    plus distinct shifts), then signature. Stop at the first exact fit rather
    than fitting every tied layout; expression cost is reported, not minimized.
    """
    byte_tables = [{point: identity * scalar_bytes for point, identity in table.items()} for table in tables]
    groups = []
    for table in byte_tables:
        lanes = defaultdict(set)
        for (participant, slot, *world), address in table.items():
            lanes[participant // 32, slot, *world].update(range(address // 4, (address + scalar_bytes - 1) // 4 + 1))
        groups.extend(lanes.values())

    def score(signatures, permutation):
        worst = total = 0
        shuffled = [sum(((value >> bit) & 1) << target for bit, target in enumerate(permutation)) for value in range(32)]
        for signature, count in signatures.items():
            banks = [0] * 32
            for low, high in signature:
                banks[low ^ shuffled[high]] += 1
            degree = max(banks)
            worst = max(worst, degree)
            total += count * degree
        return worst, total

    baseline = (max((max(Counter(word & 31 for word in group).values()) for group in groups), default=0),
                sum(max(Counter(word & 31 for word in group).values()) for group in groups))
    candidates = [(baseline, -1, ())]
    low_bit = max(0, scalar_bytes.bit_length() - 3)
    bank_bits = 5 - low_bit
    if scalar_bytes & (scalar_bytes - 1) == 0 and bank_bits > 0 and extent % 128 == 0:
        for shift in range(5, min(13, ((extent - 1) // 4).bit_length() - bank_bits + 1)):
            # Deduplicate instruction signatures, retaining multiplicities and
            # distinct words even when their projected bank bits coincide.
            mask = (1 << bank_bits) - 1
            signatures = Counter(tuple(sorted((word & 31, (word >> shift) & mask) for word in group)) for group in groups)
            for permutation in permutations(range(low_bit, 5)):
                candidates.append((score(signatures, permutation), shift, permutation))
    def structural_cost(candidate):
        _, shift, permutation = candidate
        return len(permutation) + len({shift + bit - target for bit, target in enumerate(permutation)})

    candidates.sort(key=lambda candidate: (candidate[0], structural_cost(candidate), candidate[1:]))
    names = ("participant", "slot", *bindings)
    inputs = [[(name, max(1, max(point[i] for point in table).bit_length())) for i, name in enumerate(names)] for table in byte_tables]
    for bank_score, shift, permutation in candidates:
        if shift < 0:
            mapped = byte_tables
        else:
            shuffled = [sum(((value >> bit) & 1) << target for bit, target in enumerate(permutation)) << 2 for value in range(32)]
            mapped = [{point: address ^ shuffled[(address >> (shift + 2)) & 31] for point, address in table.items()} for table in byte_tables]
        expressions = [fit.fit_expression(table, domain) for table, domain in zip(mapped, inputs)]
        if any(value is None for value in expressions):
            continue
        return mapped, expressions, {"word_bit_start": shift, "bank_permutation": list(permutation),
                        "max_conflict_degree": bank_score[0], "total_conflict_cost": bank_score[1],
                        "expression_cost": sum(_cost(value) for value in expressions),
                        "structural_cost": structural_cost((bank_score, shift, permutation)),
                        "candidates": len(candidates), "baseline_max_conflict_degree": baseline[0]}
    return None, None, {"reason": "no_fitted_staging_layout"}
