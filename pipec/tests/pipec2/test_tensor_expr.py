from __future__ import annotations

import json

import pytest

from pipec2.canonjson import canonical_bytes
from pipec2.tensor.expr import (
    Domain,
    ExprError,
    domain_check,
    evaluate,
    ground,
    parse,
    relation_check,
    to_json_obj,
)


@pytest.mark.parametrize(
    ("obj", "expected", "sort"),
    [
        (["const", -(1 << 100)], -(1 << 100), "int"),
        (["cbits", 4, 15], 15, "bits[4]"),
        (["coord", "m"], -3, "int"),
        (["bind", "base"], 9, "bits[4]"),
        (["add", ["coord", "m"], ["const", 5]], 2, "int"),
        (["sub", ["coord", "m"], ["const", 5]], -8, "int"),
        (["mulc", -4, ["coord", "m"]], 12, "int"),
        (["divc", ["coord", "m"], 2], -2, "int"),
        (["modc", ["coord", "m"], 2], 1, "int"),
        (["band", ["bind", "base"], ["cbits", 4, 3]], 1, "bits[4]"),
        (["bor", ["bind", "base"], ["cbits", 4, 3]], 11, "bits[4]"),
        (["bxor", ["bind", "base"], ["cbits", 4, 3]], 10, "bits[4]"),
        (["shlc", ["bind", "base"], 1], 2, "bits[4]"),
        (["shrc", ["bind", "base"], 1], 4, "bits[4]"),
        (["shlc", ["bind", "base"], 4], 0, "bits[4]"),
        (["shrc", ["bind", "base"], 100000000000000000000], 0, "bits[4]"),
        (["shlc", ["bind", "base"], 0], 9, "bits[4]"),
        (["tobits", 4, ["add", ["coord", "m"], ["const", 3]]], 0, "bits[4]"),
        (["toint", ["bind", "base"]], 9, "int"),
        (["eq", ["coord", "m"], ["const", -3]], 1, "pred"),
        (["lt", ["coord", "m"], ["const", -3]], 0, "pred"),
        (["le", ["coord", "m"], ["const", -3]], 1, "pred"),
        (["not", ["eq", ["coord", "m"], ["const", -3]]], 0, "pred"),
        (["pand", ["le", ["coord", "m"], ["const", -3]],
          ["lt", ["coord", "m"], ["const", -3]]], 0, "pred"),
        (["por", ["lt", ["coord", "m"], ["const", -3]],
          ["eq", ["coord", "m"], ["const", -3]]], 1, "pred"),
        (["select", ["lt", ["coord", "m"], ["const", 0]],
          ["const", 5], ["const", 6]], 5, "int"),
        (["select", ["eq", ["coord", "m"], ["const", 0]],
          ["cbits", 4, 5], ["cbits", 4, 6]], 6, "bits[4]"),
    ],
)
def test_expression_grammar_semantics(obj, expected, sort) -> None:
    domains = {
        "m": Domain("m", "int", -3, 4),
        "base": Domain("base", "bits[4]", values=(9,)),
    }
    expression = parse(obj, domains)
    assert expression.sort == sort
    assert evaluate(expression, {"m": -3, "base": 9}) == expected


def test_expression_canonical_round_trip() -> None:
    obj = ["add", ["coord", "m"], ["toint", ["bind", "base"]]]
    domains = {
        "m": Domain("m", "int", -3, 4),
        "base": Domain("base", "bits[4]", values=(9,)),
    }
    expression = parse(obj, domains)
    assert to_json_obj(expression) == obj
    encoded = canonical_bytes(to_json_obj(expression))
    restored = parse(json.loads(encoded), dict(reversed(tuple(domains.items()))))
    assert restored == expression


@pytest.mark.parametrize(
    "obj",
    [
        None, [], {}, ("const", 1), ["const"], ["const", 1, 2],
        ["mystery", ["const", 1]], ["const", True], ["const", 1.5],
        ["cbits", 0, 0], ["cbits", -1, 0], ["cbits", True, 0],
        ["cbits", 4, -1], ["cbits", 4, 16], ["cbits", 4, False],
        ["coord", "unknown"], ["bind", "unknown"], ["coord", 4],
        ["add", ["cbits", 4, 1], ["const", 2]],
        ["sub", ["const", 2], ["cbits", 4, 1]],
        ["mulc", 2, ["cbits", 4, 1]],
        ["mulc", ["const", 2], ["const", 1]],
        ["divc", ["cbits", 4, 1], 2],
        ["modc", ["cbits", 4, 1], 2],
        ["divc", ["const", 1], 0], ["divc", ["const", 1], -1],
        ["modc", ["const", 1], 0], ["modc", ["const", 1], -1],
        ["divc", ["const", 1], True], ["modc", ["const", 1], 1.5],
        ["shlc", ["const", 1], 1], ["shrc", ["const", 1], 1],
        ["shlc", ["cbits", 4, 1], -1], ["shrc", ["cbits", 4, 1], True],
        ["tobits", 4, ["cbits", 4, 1]], ["tobits", 0, ["const", 1]],
        ["toint", ["const", 1]],
        ["select", ["const", 1], ["const", 2], ["const", 3]],
        ["select", ["eq", ["const", 1], ["const", 1]], ["const", 2], ["cbits", 4, 3]],
        ["select", ["eq", ["const", 1], ["const", 1]], ["cbits", 4, 2], ["cbits", 5, 3]],
        ["select", ["eq", ["const", 1], ["const", 1]],
         ["eq", ["const", 2], ["const", 2]], ["eq", ["const", 3], ["const", 3]]],
        ["not", ["const", 1]],
        ["pand", ["const", 1], ["eq", ["const", 1], ["const", 1]]],
        ["por", ["eq", ["const", 1], ["const", 1]], ["const", 0]],
    ],
)
def test_rejects_malformed_or_ill_sorted_expressions(obj) -> None:
    with pytest.raises(ExprError):
        parse(obj, {})


@pytest.mark.parametrize("op", ["band", "bor", "bxor"])
@pytest.mark.parametrize("left", [["const", 1], ["cbits", 5, 1]])
def test_bitwise_operators_require_equal_bit_widths(op, left) -> None:
    with pytest.raises(ExprError):
        parse([op, left, ["cbits", 4, 1]], {})


@pytest.mark.parametrize("op", ["eq", "lt", "le"])
@pytest.mark.parametrize("position", [0, 1])
def test_comparison_operands_must_both_be_int(op, position) -> None:
    args = [["const", 1], ["const", 2]]
    args[position] = ["cbits", 4, 1]
    with pytest.raises(ExprError):
        parse([op, *args], {})


def test_parse_reports_all_independent_errors_in_stable_order() -> None:
    obj = ["select", ["pand", ["coord", "missing"], ["const", 1]],
           ["divc", ["cbits", 4, 16], 0], ["add", ["cbits", 4, 2], ["bind", "absent"]]]
    with pytest.raises(ExprError) as caught:
        parse(obj, {})
    issues = caught.value.issues
    assert len(issues) == 6
    assert issues == sorted(issues)
    assert {failure.path for failure in caught.value.failures} == {
        "$[1][1][1]", "$[1][2]", "$[2][1][2]", "$[2][2]", "$[3][1]", "$[3][2][1]",
    }
    assert {failure.clause_id for failure in caught.value.failures} == {"T-CONTRACT"}


@pytest.mark.parametrize(
    "obj",
    [
        {"name": "m", "sort": "int", "lo": -3, "hi": 10},
        {"name": "lane", "sort": "bits[5]", "lo": 0, "hi": 32},
        {"name": "m", "sort": "int", "values": [5, -2, 9]},
        {"name": "mask", "sort": "bits[3]", "values": [7, 0, 3]},
        {"name": "empty", "sort": "int", "values": []},
    ],
)
def test_domain_json_round_trip(obj) -> None:
    domain = Domain.from_json_obj(obj)
    assert domain.to_json_obj() == obj
    assert Domain.from_json_obj(json.loads(canonical_bytes(domain.to_json_obj()))) == domain


@pytest.mark.parametrize(
    "obj",
    [
        {"name": "m", "sort": "int", "lo": 2, "hi": 1},
        {"name": "m", "sort": "int", "lo": False, "hi": 2},
        {"name": "m", "sort": "int", "lo": 0, "hi": 2, "values": [1]},
        {"name": "m", "sort": "int", "lo": None, "hi": None, "values": [1]},
        {"name": "m", "sort": "int", "values": [1, 1]},
        {"name": "m", "sort": "int", "values": [True]},
        {"name": "m", "sort": "int", "values": (1,)},
        {"name": "m", "sort": "bits[0]", "values": [0]},
        {"name": "m", "sort": "bits[04]", "values": [0]},
        {"name": "m", "sort": "bits[4]", "values": [16]},
        {"name": "m", "sort": "bits[4]", "lo": -1, "hi": 1},
        {"name": "m", "sort": "bits[4]", "lo": 0, "hi": 17},
        {"name": "m", "sort": "float", "lo": 0, "hi": 1},
        {"name": "", "sort": "int", "lo": 0, "hi": 1},
        {"name": "m", "sort": "int"},
    ],
)
def test_domain_rejects_ambiguous_or_invalid_finite_sets(obj) -> None:
    with pytest.raises(ExprError):
        Domain.from_json_obj(obj)


def test_domain_check_proves_cast_bounds_and_reports_both_edges() -> None:
    domains = {"m": Domain("m", "int", -1, 17)}
    expression = parse(["tobits", 4, ["coord", "m"]], domains)
    report = domain_check(expression, domains, cap=18)
    assert not report.ok and not report.capped
    assert report.enumerated == 18
    assert len(report.issues) == 2
    assert any("(-1,)" in issue for issue in report.issues)
    assert any("(16,)" in issue for issue in report.issues)
    valid_domains = {"m": Domain("m", "int", 0, 16)}
    valid = domain_check(expression, valid_domains, cap=16)
    assert valid.ok and not valid.capped and valid.enumerated == 16
    assert valid.issues == ()
    for value in (-1, 16):
        with pytest.raises(ExprError):
            evaluate(expression, {"m": value})


def test_domain_check_checks_all_casts_including_unselected_arms() -> None:
    expression = parse(
        ["select", ["eq", ["const", 0], ["const", 0]],
         ["tobits", 3, ["const", -1]], ["tobits", 3, ["const", 8]]], {},
    )
    report = domain_check(expression, {}, cap=1)
    assert not report.ok and not report.capped and report.enumerated == 1
    assert len(report.issues) == 2
    assert any("$[2]:" in issue for issue in report.issues)
    assert any("$[3]:" in issue for issue in report.issues)


def test_domain_check_caps_product_without_sampling_or_unused_domains() -> None:
    domains = {
        "m": Domain("m", "int", 0, 4),
        "offset": Domain("offset", "int", values=(0, 4)),
        "unused": Domain("unused", "int", 0, 1 << 100),
    }
    expression = parse(["tobits", 3, ["add", ["coord", "m"], ["bind", "offset"]]], domains)
    report = domain_check(expression, domains, cap=7)
    assert report.capped and not report.ok and report.enumerated == 0
    complete = domain_check(expression, domains, cap=8)
    assert complete.ok and not complete.capped and complete.enumerated == 8
    huge = parse(["coord", "unused"], domains)
    assert domain_check(huge, domains, cap=8).capped


def test_ground_uses_lexical_referenced_names_and_explicit_values() -> None:
    domains = {
        "z": Domain("z", "int", values=(4, 2)),
        "unused": Domain("unused", "int", 0, 1 << 100),
        "a": Domain("a", "int", 0, 2),
    }
    expression = parse(["add", ["mulc", 10, ["coord", "a"]], ["bind", "z"]], domains)
    assert ground(expression, domains) == {(0, 4): 4, (0, 2): 2, (1, 4): 14, (1, 2): 12}
    assert ground(parse(["const", 7], domains), domains) == {(): 7}
    empty = {"a": Domain("a", "int", values=())}
    assert ground(parse(["coord", "a"], empty), empty) == {}
    assert domain_check(parse(["coord", "a"], empty), empty, cap=0).ok
    assert domain_check(parse(["const", 1], {}), {}, cap=0).capped


def test_relation_checks_small_bijection_and_component_order() -> None:
    domains = {"j": Domain("j", "int", 0, 3), "i": Domain("i", "int", 0, 2)}
    components = {
        "y": parse(["sub", ["const", 2], ["coord", "j"]], domains),
        "x": parse(["coord", "i"], domains),
    }
    report = relation_check(components, domains, {"x": (0, 2), "y": (0, 3)}, cap=6)
    assert report.total and report.in_bounds and report.injective and report.covering
    assert report.witnesses == () and not report.capped
    reordered = relation_check(dict(reversed(tuple(components.items()))),
                               dict(reversed(tuple(domains.items()))),
                               {"y": (0, 3), "x": (0, 2)}, cap=6)
    assert reordered == report


def test_relation_replication_and_omitted_source_dimensions() -> None:
    domains = {"m": Domain("m", "int", 0, 4)}
    report = relation_check({"x": parse(["modc", ["coord", "m"], 2], domains)},
                            domains, {"x": (0, 2)}, cap=4)
    assert report.total and report.in_bounds and report.covering and not report.injective
    assert any("(0,) and (2,)" in witness for witness in report.witnesses)
    constant = relation_check({"x": parse(["const", 0], domains)}, domains, {"x": (0, 1)}, cap=4)
    assert constant.total and constant.in_bounds and constant.covering and not constant.injective


def test_relation_out_of_bounds_and_missing_coverage_have_concrete_witnesses() -> None:
    domains = {"m": Domain("m", "int", 0, 3)}
    report = relation_check({"x": parse(["add", ["coord", "m"], ["const", 1]], domains)},
                            domains, {"x": (0, 3)}, cap=3)
    assert report.total and report.injective and not report.in_bounds and not report.covering
    assert any("(2,)" in witness and "(3,)" in witness for witness in report.witnesses)
    assert any("not covering" in witness and "(0,)" in witness for witness in report.witnesses)


def test_relation_padded_injection_does_not_invent_coverage() -> None:
    domains = {"m": Domain("m", "int", 0, 4)}
    report = relation_check({"x": parse(["mulc", 2, ["coord", "m"]], domains)},
                            domains, {"x": (0, 8)}, cap=4)
    assert report.total and report.in_bounds and report.injective and not report.covering
    assert any("(1,)" in witness for witness in report.witnesses)


def test_relation_cap_and_huge_target_box_are_bounded() -> None:
    domains = {"m": Domain("m", "int", 0, 3)}
    components = {"x": parse(["coord", "m"], domains)}
    capped = relation_check(components, domains, {"x": (0, 3)}, cap=2)
    assert capped.capped
    assert not any((capped.total, capped.in_bounds, capped.injective, capped.covering))
    huge = relation_check(components, domains, {"x": (0, 1 << 100)}, cap=3)
    assert huge.total and huge.in_bounds and huge.injective and not huge.covering
    assert not huge.capped
    assert any("not covering" in witness and "(3,)" in witness for witness in huge.witnesses)


def test_relation_undefined_cast_is_non_total_not_a_wrapped_map() -> None:
    domains = {"m": Domain("m", "int", 0, 3)}
    report = relation_check({"x": parse(["toint", ["tobits", 1, ["coord", "m"]]], domains)},
                            domains, {"x": (0, 3)}, cap=3)
    assert not report.total and not report.in_bounds and not report.injective and not report.covering
    assert any("non-total" in witness and "(2,)" in witness for witness in report.witnesses)


@pytest.mark.parametrize("bounds", [{}, {"x": (2, 1)}, {"x": (0, True)}, {"other": (0, 2)}])
def test_relation_rejects_invalid_bounds(bounds) -> None:
    with pytest.raises(ExprError):
        relation_check({"x": parse(["const", 0], {})}, {}, bounds, cap=1)


def test_worked_m64_datapath_map_is_a_permutation() -> None:
    domains = {"m": Domain("m", "int", 0, 128)}
    # d(m) = m%16 + 32*((m%64)//16) + 16*(m//64), design-experiments.
    expression = parse(
        ["add", ["add", ["modc", ["coord", "m"], 16],
                 ["mulc", 32, ["divc", ["modc", ["coord", "m"], 64], 16]]],
         ["mulc", 16, ["divc", ["coord", "m"], 64]]], domains,
    )
    report = relation_check({"d": expression}, domains, {"d": (0, 128)}, cap=128)
    assert report.total and report.in_bounds and report.injective and report.covering
    assert not report.capped and report.witnesses == ()
