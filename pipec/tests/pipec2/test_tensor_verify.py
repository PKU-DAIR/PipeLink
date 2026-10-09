from __future__ import annotations

from pipec2.tensor.expr import Domain, parse, relation_check
from pipec2.tensor.verify import verify_export_map


def _export_map():
    domains = {"m": Domain("m", "int", 0, 2), "n": Domain("n", "int", 0, 3)}
    bindings = {"tmem_base": Domain("tmem_base", "bits[32]", values=(0, 3, 6))}
    sort_env = {**domains, **bindings}
    placements = {
        "dp": parse(["coord", "m"], sort_env),
        "column": parse(["add", ["coord", "n"], ["toint", ["bind", "tmem_base"]]], sort_env),
    }
    # Independent owner facts: never evaluate the declared expressions to make
    # this table. The verifier must reject another bijection of these same cells.
    observations = {
        (m, n, base): {"dp": m, "column": n + base}
        for m in range(2)
        for n in range(3)
        for base in (0, 3, 6)
    }
    return placements, domains, bindings, observations


def test_export_map_matches_owner_at_every_coordinate_and_binding() -> None:
    placements, domains, bindings, observations = _export_map()

    result = verify_export_map(placements, domains, bindings, observations)

    assert result.ok
    assert result.mismatches == ()
    assert result.mismatch_count == 0
    assert result.checked_points == 18
    # C1 grounding projects to each expression's referenced dimensions, while
    # equality above still covers the whole 2 x 3 x 3 declared product.
    assert result.ground_entries == {"dp": 2, "column": 9}


def test_export_map_rejects_wrong_total_in_bounds_bijection() -> None:
    placements, domains, bindings, observations = _export_map()
    placements["dp"] = parse(["sub", ["const", 1], ["coord", "m"]], domains)
    relation = relation_check(
        placements, {**domains, **bindings}, {"dp": (0, 2), "column": (0, 9)}, cap=18,
    )
    assert relation.total and relation.in_bounds and relation.injective and relation.covering
    assert not relation.capped

    result = verify_export_map(placements, domains, bindings, observations, max_mismatches=2)

    assert not result.ok
    assert result.checked_points == 18
    assert result.mismatch_count == 18  # Witness truncation must not truncate the gate.
    assert len(result.mismatches) == 2
    first, second = result.mismatches
    assert first.coordinates == {"m": 0, "n": 0}
    assert first.bindings == {"tmem_base": 0}
    assert first.reason == "value"
    assert first.expected == {"dp": 0, "column": 0}
    assert first.declared == {"dp": 1, "column": 0}
    assert second.coordinates == {"m": 0, "n": 0}
    assert second.bindings == {"tmem_base": 3}
    assert second.expected == {"dp": 0, "column": 3}
    assert second.declared == {"dp": 1, "column": 3}


def test_export_map_rejects_missing_declared_binding_value() -> None:
    placements, domains, bindings, observations = _export_map()
    partial = {point: observed for point, observed in observations.items() if point[2] != 6}

    result = verify_export_map(placements, domains, bindings, partial)

    assert not result.ok
    assert result.checked_points == 18
    assert result.mismatch_count == 6
    assert {tuple(mismatch.coordinates.values()) for mismatch in result.mismatches} == {
        (m, n) for m in range(2) for n in range(3)
    }
    assert all(mismatch.reason == "missing_observation" for mismatch in result.mismatches)
    assert all(mismatch.bindings == {"tmem_base": 6} for mismatch in result.mismatches)
    assert all(mismatch.expected is None for mismatch in result.mismatches)
    first = result.mismatches[0]
    assert first.coordinates == {"m": 0, "n": 0}
    assert first.declared == {"dp": 0, "column": 6}
