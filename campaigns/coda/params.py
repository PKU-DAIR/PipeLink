"""CODA fixtures and independent f64 fused/unfused references."""
from __future__ import annotations

import numpy as np
import ml_dtypes

BF16 = ml_dtypes.bfloat16
PROFILES = {"sm103": "sm_103a", "sm90": "sm_90a", "sm120": "sm_120a"}
SMALL_SHAPES = ((256, 512, 64), (256, 512, 320), (256, 1024, 1024), (512, 1024, 5120))
FULL_SHAPES = ((16384, 51200, 5120), (4096, 51200, 5120))
SHAPES = SMALL_SHAPES + FULL_SHAPES
SEEDS = {"x": 20261015, "gate": 20261016, "up": 20261017, "sample": 20261018}
TOLERANCE = {"rtol": 1e-2, "atol": 1e-2}


def inputs(M, N, K):
    if M % 256 or N % 512 or K % 64 or min(M, N, K) <= 0:
        raise ValueError("CODA fixtures require complete 256x512x64 tiles")
    x, w = np.empty((M, K), dtype=BF16), np.empty((N, K), dtype=BF16)
    for array, seed, scale in ((x, SEEDS["x"], 1), (w[0::2], SEEDS["gate"], .02),
                               (w[1::2], SEEDS["up"], .02)):
        rng = np.random.default_rng(seed)
        for start in range(0, len(array), 256):
            part = array[start:start + 256]
            part[:] = rng.standard_normal(part.shape, dtype=np.float32) * np.float32(scale)
    return {"x": x, "w": w, "out": np.full((M, N // 2), np.nan, dtype=BF16)}


def sample_coordinates(M, N):
    """65,536 unique outputs; all 128x128 local residues in each of four strata.

    Four consecutive row tiles per local coordinate include both local halves of
    a two-CTA pair. Seeded tile origins/column tiles disperse coverage grid-wide.
    The output-column residue also covers all eight 16-output chunks.
    """
    if M < 512 or M % 128 or N % 256:
        raise ValueError("stratification needs at least four full row tiles")
    rng = np.random.default_rng(SEEDS["sample"])
    local_r, local_c = np.indices((128, 128))
    base = 2 * rng.integers(0, M // 256, size=(128, 128))
    row_tiles = (base[None, :, :] + np.arange(4)[:, None, None]) % (M // 128)
    col_tiles = rng.integers(0, N // 256, size=(4, 128, 128))
    return ((row_tiles * 128 + local_r).ravel(), (col_tiles * 128 + local_c).ravel())


def round_bf16(value):
    """Round f64 directly to bf16, avoiding ml_dtypes' intermediate f32 tie."""
    value = np.asarray(value, dtype=np.float64)
    rounded = value.astype(BF16)
    best = np.abs(value - rounded.astype(np.float64))
    for direction in (-np.inf, np.inf):
        neighbor = np.nextafter(rounded, np.asarray(direction, dtype=BF16))
        distance = np.abs(value - neighbor.astype(np.float64))
        take = (distance < best) | ((distance == best) & ((neighbor.view(np.uint16) & 1) == 0))
        rounded = np.where(take, neighbor, rounded).astype(BF16)
        best = np.minimum(best, distance)
    return rounded


def references(buffers, coordinates=None):
    """bf16_rn(silu64(g)*u), with optional bf16 intermediate for unfused.

    No emulation of a tensor-core reduction order: the dot is f64 on bf16-exact
    inputs; f32 accumulation/expf differences are covered by the fixed tolerance.
    """
    x, w = buffers["x"], buffers["w"]
    if coordinates is None:
        d = x.astype(np.float64) @ w.astype(np.float64).T
        gate, up = d[:, 0::2], d[:, 1::2]
    else:
        rows, cols = coordinates
        gate, up = np.empty(len(rows), np.float64), np.empty(len(rows), np.float64)
        for start in range(0, len(rows), 256):
            end = start + 256
            a = x[rows[start:end]].astype(np.float64)
            for dest, offset in ((gate, 0), (up, 1)):
                b = w[2 * cols[start:end] + offset].astype(np.float64)
                dest[start:end] = np.einsum("ij,ij->i", a, b)
    def activation(g, u):
        return round_bf16(g / (1 + np.exp(-g)) * u)
    fused = activation(gate, up)
    unfused = activation(round_bf16(gate).astype(np.float64), round_bf16(up).astype(np.float64))
    return {"fused": fused, "unfused": unfused}


def check(actual, expected, coordinates=None):
    finite = bool(np.isfinite(actual).all())
    selected = actual if coordinates is None else actual[coordinates]
    a, b = selected.astype(np.float64), expected.astype(np.float64)
    close = np.isclose(a, b, **TOLERANCE)
    return {"status": "PASS" if finite and close.all() else "FAIL", "finite": finite,
            "checked": int(b.size), "mismatches": int((~close).sum()),
            "max_abs": float(np.max(np.abs(a - b)))}


def check_unfused(actual, expected, buffers, coordinates=None):
    """Check nominal f64 oracle, then bounded intermediate-rounding ambiguity.

    The ambiguity check runs only on nominal failures. An alternative must be an
    adjacent bf16 value enclosing the exact dot, whose midpoint lies within the
    standard f32 forward-error bound gamma_K * sum(abs(x*w)), gamma_K=Ku/(1-Ku).
    The final-output tolerance is the same as for the nominal check. Every
    accepted case is recorded.
    """
    result = check(actual, expected, coordinates)
    result["nominal_mismatches"] = result["mismatches"]
    result["boundary_rule_used"] = 0
    result["boundary_cases"] = []
    selected = actual if coordinates is None else actual[coordinates]
    bad = np.argwhere(~np.isclose(selected.astype(np.float64), expected.astype(np.float64), **TOLERANCE))
    K = buffers["x"].shape[1]
    ku = K * 2.0**-24
    gamma = ku / (1 - ku)
    for index in bad:
        location = tuple(index)
        row, col = location if coordinates is None else (int(coordinates[0][location]), int(coordinates[1][location]))
        a = buffers["x"][row].astype(np.float64)
        dots, bounds, choices = [], [], []
        for offset in (0, 1):
            products = a * buffers["w"][2 * col + offset].astype(np.float64)
            exact, bound = float(products.sum()), float(gamma * np.abs(products).sum())
            rounded = round_bf16(exact)
            nominal = float(rounded)
            alternatives = [nominal]
            if nominal != exact:
                neighbor = float(np.nextafter(rounded, np.asarray(np.inf if exact > nominal else -np.inf, dtype=BF16)))
                midpoint = (nominal + neighbor) / 2
                if abs(exact - midpoint) <= bound:
                    alternatives.append(neighbor)
            dots.append(exact)
            bounds.append(bound)
            choices.append(alternatives)
        candidates = [float(round_bf16(g / (1 + np.exp(-g)) * u))
                      for g in choices[0] for u in choices[1]]
        if any(np.isclose(float(selected[location]), value, **TOLERANCE) for value in candidates):
            result["boundary_rule_used"] += 1
            result["boundary_cases"].append({"row": int(row), "column": int(col), "dots_f64": dots,
                "f32_error_bounds": bounds, "candidates": candidates, "actual": float(selected[location])})
    result["mismatches"] -= result["boundary_rule_used"]
    result["status"] = "PASS" if result["finite"] and result["mismatches"] == 0 else "FAIL"
    return result


def semantic_difference(refs):
    a, b = (refs[key].astype(np.float64) for key in ("fused", "unfused"))
    return {"different": int(np.count_nonzero(a != b)), "outputs": int(a.size),
            "max_abs": float(np.max(np.abs(a - b))), "diagnostic_only": True}


def self_check():
    for M, N, _ in FULL_SHAPES:
        r, c = sample_coordinates(M, N)
        assert len(r) == 65536 and len(np.unique(r * (N // 2) + c)) == 65536
        assert len(np.unique((r % 128) * 128 + c % 128)) == 16384
        assert set((c % 128 // 16).tolist()) == set(range(8))
        assert set((r // 128 % 2).tolist()) == {0, 1}
        assert r.min() < 128 and r.max() >= M - 128
    fixture = {"x": np.array([[1, 2]], dtype=BF16),
               "w": np.array([[1, 0], [0, 1]], dtype=BF16)}
    ref = references(fixture)
    expected = np.array([[2 / (1 + np.exp(-1))]], dtype=BF16)
    assert np.array_equal(ref["fused"], expected)
    assert np.array_equal(ref["unfused"], expected)
    sampled = references(fixture, (np.array([0]), np.array([0])))
    assert sampled["fused"][0] == expected[0, 0]
    assert check(np.full((1, 1), np.nan, dtype=BF16), expected)["status"] == "FAIL"
    boundary = {"x": np.ones((1, 3), dtype=BF16),
                "w": np.array([[1.359375, .00390625, 2**-24], [-4.75, 0, 0]], dtype=BF16)}
    reference = references(boundary)["unfused"]
    alternative = np.array([[-5.125]], dtype=BF16)
    assert check(alternative, reference)["status"] == "FAIL"
    bounded = check_unfused(alternative, reference, boundary)
    assert bounded["status"] == "PASS" and bounded["boundary_rule_used"] == 1
    assert check_unfused(np.array([[10]], dtype=BF16), reference, boundary)["status"] == "FAIL"
    return {"sampler": "PASS", "f64_oracles": "PASS", "nan_poison": "PASS", "rounding_boundary": "PASS"}
