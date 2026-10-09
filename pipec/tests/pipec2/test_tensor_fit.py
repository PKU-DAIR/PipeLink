import json

import pytest

from pipec2.model import PipeLinkError
from pipec2.tensor import expr
from pipec2.tensor.contracts import TensorExport
from pipec2.tensor.fit import FitError, fit_expression, fit_named, fit_named_or_raise


def _assert_fit(rows, inputs):
    fitted = fit_expression(rows, inputs)
    assert fitted is not None
    for point, expected in rows.items():
        assert expr.evaluate(fitted, dict(zip((name for name, _ in inputs), point))) == expected
    return fitted


def test_bit_permutation_and_canonical_input_order():
    rows = {(p, s): ((p & 3) << 5) | ((p >> 2) << 2) | s for p in range(16) for s in range(4)}
    fitted = _assert_fit(rows, [("participant", 4), ("slot", 2)])
    reordered = fit_expression({(s, p): value for (p, s), value in reversed(list(rows.items()))},
                               [("slot", 2), ("participant", 4)])
    assert fitted == reordered
    assert not (_ops(fitted) & {"band", "bxor", "tobits", "cbits"})


def test_xor_swizzle_generated_from_atom_geometry():
    rows, columns, swizzle_bits, shift = 8, 64, 3, 3
    mask = ((1 << swizzle_bits) - 1) << (columns.bit_length() - 1)
    table = {(p, s): 2 * ((p * columns + s) ^ (((p * columns + s) & mask) >> shift))
             for p in range(rows) for s in range(columns)}
    fitted = _assert_fit(table, [("participant", 3), ("slot", 6)])
    assert "bxor" in str(fitted.to_json_obj())


def test_padded_stride_uses_integer_affine_template():
    table = {(p, s): p * 144 + s * 4 for p in range(16) for s in range(32)}
    fitted = _assert_fit(table, [("participant", 4), ("slot", 5)])
    assert "mulc" in str(fitted.to_json_obj())


def test_affine_form_wins_when_bitwise_template_also_fits():
    table = {(p, s): 16 * p + s for p in range(8) for s in range(16)}
    fitted = _assert_fit(table, [("participant", 3), ("slot", 4)])
    assert fitted.to_json_obj() == [
        "add", ["mulc", 16, ["coord", "participant"]], ["coord", "slot"],
    ]


def test_nonfittable_table_reports_first_inconsistent_rows():
    failures = []
    table = {(p,): p * p for p in range(8)}
    assert fit_expression(table, [("participant", 3)], failures=failures) is None
    assert [failure["template"] for failure in failures] == ["integer-affine", "gf2-linear"]
    assert failures[0]["first_mismatch"]["inputs"] == [2]
    assert failures[1]["first_mismatch"] == {"inputs": [3], "expected": 9, "actual": 5}


def _assert_named_fit(rows, inputs, output_index):
    sort_env = {domain.name: domain for domain, _ in inputs}
    fitted = fit_named(rows, inputs, output_index, sort_env=sort_env)
    assert fitted is not None
    parsed = expr.parse(json.loads(json.dumps(fitted.to_json_obj())), sort_env)
    names = tuple(domain.name for domain, _ in inputs)
    for row in rows:
        assert expr.evaluate(parsed, dict(zip(names, row))) == row[len(inputs) + output_index]
    return parsed


def _ops(node):
    return {node.op}.union(*(_ops(arg) for arg in node.args if isinstance(arg, expr.Expr)))


def test_named_m128_tmem_placement_with_bits_binding():
    m = expr.Domain("m", "int", 0, 128)
    n = expr.Domain("n", "int", 0, 128)
    base = expr.Domain("tmem_base", "bits[32]", values=(0, 128, 256, 384))
    inputs = [(m, "coord"), (n, "coord"), (base, "bind")]
    # producer_m128 components from phase2/TIProducers/old-manifests.json.
    old_components = {
        "dp": expr.parse(["coord", "m"], {domain.name: domain for domain, _ in inputs}),
        "column": expr.parse(["add", ["coord", "n"], ["toint", ["bind", "tmem_base"]]],
                             {domain.name: domain for domain, _ in inputs}),
    }
    rows = [[i, j, b, *(expr.evaluate(component, {"m": i, "n": j, "tmem_base": b})
                       for component in old_components.values())]
            for i in range(128) for j in range(128) for b in base.values]
    components = {name: _assert_named_fit(rows, inputs, index)
                  for index, name in enumerate(("dp", "column"))}
    assert len(rows) == 65_536
    for component in components.values():
        assert not (_ops(component) & {"band", "bxor", "tobits", "cbits"})
    exported = TensorExport.from_json_obj({
        "name": "accumulator",
        "identity": {
            "result": "accumulator", "work": "tensor_work", "version": "completed_k",
            "dtype": "f32", "interpretation": "fp16_dot_fp32_completed",
            "coords": [m.to_json_obj(), n.to_json_obj()],
        },
        "placements": [{
            "guard": None, "space": "tmem", "units": "dp/column",
            "components": {name: component.to_json_obj() for name, component in components.items()},
        }],
        "read_capabilities": ["tmem_packet_receipt"],
        "readiness": {"after": "tensor_ready", "reuse_after": "import_done"},
        "fixed_bindings": {}, "backing": [],
        "calls": {"produce": "produce", "release": None},
    }, sort_env={base.name: base})
    assert exported.placements[0].components == components


def test_named_padded_smem_address_requires_integer_affine():
    inputs = [(expr.Domain("m", "int", 0, 128), "coord"),
              (expr.Domain("n", "int", 0, 32), "coord")]
    rows = [[m, n, 144 * m + 4 * n] for m in range(128) for n in range(32)]
    fitted = _assert_named_fit(rows, inputs, 0)
    # The GF(2) family cannot express the carry at (m,n)=(1,4).
    assert fitted.op == "add"


def test_named_sw128_operand_view_address():
    inputs = [(expr.Domain("m", "int", 0, 128), "coord"),
              (expr.Domain("n", "int", 0, 128), "coord")]
    rows = []
    for m in range(128):
        for n in range(128):
            # annotations_operand_view.hpp:38-65, SW128 K-major bf16.
            local_elements = 64 * (m % 8) + n % 64
            swizzle = local_elements ^ ((local_elements & 448) >> 3)
            blocks = (128 // 8) * (n // 64) + m // 8
            rows.append([m, n, 1024 * blocks + 2 * swizzle])
    fitted = _assert_named_fit(rows, inputs, 0)
    assert fitted.op == "toint"
    assert {"band", "bxor", "tobits", "cbits"} <= _ops(fitted)


@pytest.mark.parametrize("sort,role", [("int", "coord"), ("bits[32]", "bind")])
def test_named_bit_permutation_uses_maximal_integer_fields(sort, role):
    domain = expr.Domain("x", sort, 0, 64)
    rows = [[x, ((x >> 3) & 7) | ((x & 7) << 3)] for x in range(64)]
    fitted = _assert_named_fit(rows, [(domain, role)], 0)
    reference = [role, "x"] if sort == "int" else ["toint", [role, "x"]]
    assert fitted.to_json_obj() == [
        "add", ["divc", reference, 8], ["mulc", 8, ["modc", reference, 8]],
    ]


def test_named_bit_fields_with_inverted_and_constant_output_bits():
    domain = expr.Domain("x", "int", 0, 64)
    rows = [[x, (((x >> 3) | ((x & 7) << 3)) ^ 0b101101) | 128] for x in range(64)]
    fitted = _assert_named_fit(rows, [(domain, "coord")], 0)
    assert not (_ops(fitted) & {"band", "bxor", "tobits", "cbits"})


def test_named_bit_field_can_copy_one_input_bit_to_multiple_outputs():
    domain = expr.Domain("x", "int", 0, 64)
    rows = [[x, (x >> 3) | ((x & 7) << 3) | ((x & 7) << 6)] for x in range(64)]
    fitted = _assert_named_fit(rows, [(domain, "coord")], 0)
    assert not (_ops(fitted) & {"band", "bxor", "tobits", "cbits"})


def test_named_nonlinear_map_reports_first_failing_row():
    inputs = [(expr.Domain("m", "int", 0, 8), "coord"),
              (expr.Domain("n", "int", 0, 8), "coord")]
    rows = [[m, n, m * n] for m in range(8) for n in range(8)]
    sort_env = {domain.name: domain for domain, _ in inputs}
    assert fit_named(rows, inputs, 0, sort_env=sort_env) is None
    with pytest.raises(FitError) as caught:
        fit_named_or_raise(rows, inputs, 0, sort_env=sort_env, output_name="nonlinear.address")
    assert isinstance(caught.value, PipeLinkError)
    failure, = caught.value.failures
    assert failure.clause_id == "T-FIT"
    assert "nonlinear.address" in str(caught.value)
    assert str(rows[9]) in str(caught.value)
    assert failure.detail["row_index"] == 9
    assert failure.detail["first_mismatch"]["inputs"] == [1, 1]
    assert failure.detail["first_mismatch"]["expected"] == 1


def test_named_demand_map_fits_each_output_column():
    inputs = [(expr.Domain("t", "int", 0, 128), "coord"),
              (expr.Domain("p", "int", 0, 128), "coord")]
    rows = [[t, p, t, p, t, p] for t in range(128) for p in range(128)]
    for index in range(4):
        _assert_named_fit(rows, inputs, index)


def test_named_signed_affine_with_binding_named_like_legacy_coordinate():
    inputs = [(expr.Domain("offset", "int", -2, 3), "coord"),
              (expr.Domain("participant", "bits[32]", 0, 8), "bind")]
    rows = [[offset, binding, 144 * offset + 4 * binding]
            for offset in range(-2, 3) for binding in range(8)]
    fitted = _assert_named_fit(rows, inputs, 0)
    assert fitted.op == "add"
    assert ["mulc", 4, ["toint", ["bind", "participant"]]] in fitted.to_json_obj()


def test_named_duplicate_inputs_do_not_drop_a_conflicting_row():
    domain = expr.Domain("m", "int", 0, 2)
    rows = [[0, 0], [1, 1], [1, 2]]
    assert fit_named(rows, [(domain, "coord")], 0, sort_env={"m": domain}) is None


def test_named_failure_uses_template_with_longest_consistent_prefix():
    domain = expr.Domain("m", "int", 0, 8)
    rows = [[m, m * m] for m in range(8)]
    with pytest.raises(FitError) as caught:
        fit_named_or_raise(rows, [(domain, "coord")], 0, sort_env={"m": domain},
                           output_name="square.address")
    detail = caught.value.failures[0].detail
    assert detail["template"] == "gf2-linear"
    assert detail["row_index"] == 3
    assert detail["first_mismatch"] == {"inputs": [3], "expected": 9, "actual": 5}
