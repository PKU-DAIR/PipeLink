from __future__ import annotations

from itertools import product

import pytest

from pipec2.tensor.expr import Domain, evaluate, parse
from pipec2.tensor.tables import coordinate_points, encode_component


def assert_table_evaluates(cells, domains, expected_operator) -> None:
    tree, _ = encode_component(cells, domains)
    assert tree[0] == expected_operator
    expression = parse(tree, domains)
    names = sorted(domains)
    for point, expected in cells.items():
        assert evaluate(expression, dict(zip(names, point))) == expected


@pytest.mark.parametrize(
    ("value", "operator"),
    [
        pytest.param(lambda row, column: row, "coord", id="row-identity"),
        pytest.param(lambda row, column: column, "coord", id="column-identity"),
        pytest.param(lambda row, column: -7, "const", id="constant"),
        pytest.param(lambda row, column: (2, 9, -3, 5, 4, 1)[row], "select", id="axis-piecewise"),
        pytest.param(lambda row, column: row * column + column, "select", id="nonseparable-piecewise"),
    ],
)
def test_encodes_complete_non_power_of_two_table(value, operator) -> None:
    domains = {"row": Domain("row", "int", 0, 6), "column": Domain("column", "int", 0, 5)}
    # Keys are lexical (column, row), not the insertion order of the declarations.
    cells = {(column, row): value(row, column) for column, row in product(range(5), range(6))}
    assert_table_evaluates(cells, domains, operator)


def test_encodes_bit_permutation_runs() -> None:
    domains = {"row": Domain("row", "int", 0, 8), "column": Domain("column", "int", 0, 5)}
    # Move row bit zero above the contiguous pair of bits one and two.
    permuted_rows = (0, 4, 1, 5, 2, 6, 3, 7)
    cells = {(column, row): permuted_rows[row] for column, row in product(range(5), range(8))}
    assert_table_evaluates(cells, domains, "add")


@pytest.mark.parametrize("identity", [False, True])
def test_preserves_explicit_coordinate_values_and_order(identity) -> None:
    rows = (7, 1, -4, 10, 2, 6)
    columns = (9, -2, 5, 8, 3)
    domains = {
        "z_column": Domain("z_column", "int", values=columns),
        "a_row": Domain("a_row", "int", values=rows),
    }
    expected_points = list(product(rows, columns))
    assert list(coordinate_points(domains)) == expected_points
    cells = {point: point[0] if identity else point[0] * point[1] for point in expected_points}
    # Encoding must not depend on the observation mapping's insertion order.
    assert_table_evaluates(dict(reversed(tuple(cells.items()))), domains, "coord" if identity else "select")


def test_rejects_wrong_size_table() -> None:
    domains = {"row": Domain("row", "int", 0, 6), "column": Domain("column", "int", 0, 5)}
    cells = {(column, row): row for column, row in product(range(5), range(6))}
    del cells[4, 5]
    with pytest.raises(ValueError, match="exactly 30 coordinate entries, got 29"):
        encode_component(cells, domains)
