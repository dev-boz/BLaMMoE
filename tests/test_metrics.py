"""WP5 acceptance: the battery separates known-good from known-bad pairs, and
diverges from CKA on Davari et al.'s single-outlier case (spec 1.3 item 5)."""
import numpy as np
import pytest

from blammoe.metrics import (
    linear_cka,
    mutual_knn,
    procrustes_residual,
    run_battery,
    rung1_gate,
)
from blammoe.metrics.knn import chance_level

N, D, RANK = 1000, 32, 6


def structured(rng: np.random.Generator, n: int = N) -> np.ndarray:
    """Low-rank-plus-noise activations: real states have far fewer effective dims than d."""
    return rng.standard_normal((n, RANK)) @ rng.standard_normal((RANK, D)) + 0.1 * rng.standard_normal(
        (n, D)
    )


@pytest.fixture
def pairs():
    rng = np.random.default_rng(0)
    X = structured(rng)
    R, _ = np.linalg.qr(rng.standard_normal((D, D)))
    good = X @ R + 0.05 * rng.standard_normal((N, D))  # known-good: rotation + noise
    bad = structured(rng)  # known-bad: an independent representation
    return X, good, bad, rng


def test_cka_invariances_and_bounds():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 16))
    R, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    assert linear_cka(X, X) == pytest.approx(1.0)
    assert linear_cka(X, 5.0 * X @ R + 3.0) == pytest.approx(1.0)  # rotation/scale/shift
    assert 0.0 <= linear_cka(X, rng.standard_normal((200, 8))) < 0.2  # also d_A != d_B


def test_mutual_knn_identity_and_chance():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((500, 16))
    assert mutual_knn(X, X, k=10) == pytest.approx(1.0)
    unrelated = mutual_knn(X, rng.standard_normal((500, 16)), k=10)
    assert unrelated < 3 * chance_level(500, 10)
    with pytest.raises(ValueError):
        mutual_knn(X, X, k=500)


def test_procrustes_residual_handles_rectangular_pairs():
    rng = np.random.default_rng(3)
    X = structured(rng)
    Y = X @ np.linalg.qr(rng.standard_normal((D, 48)).T)[0].T  # isometric embed 32 -> 48
    assert procrustes_residual(X, Y) < 1e-6


def test_battery_scores_known_good_high_and_known_bad_low(pairs):
    X, good, bad, _ = pairs
    g, b = run_battery(X, good), run_battery(X, bad)

    assert g.cka > 0.95 and g.mutual_knn > 0.9 and g.procrustes_residual < 0.01
    assert b.cka < 0.1
    assert b.mutual_knn < 3 * b.knn_chance
    assert b.procrustes_residual > 0.9  # no better than predicting the target mean


def test_rung1_gate_passes_on_good_vs_bad_and_fails_on_bad_vs_bad(pairs):
    X, good, bad, rng = pairs
    gate = rung1_gate(run_battery(X, good), run_battery(X, bad))
    assert gate.passed
    assert set(gate.separates) == {"cka", "mutual_knn", "procrustes_residual"}

    # a "known-good" that is actually unrelated must not pass the gate
    fake = rung1_gate(run_battery(X, structured(rng)), run_battery(X, bad))
    assert not fake.passed
    assert not any(fake.separates.values())


def test_single_outlier_breaks_cka_but_not_the_battery(pairs):
    """Davari et al. 2210.16156 Cor. 3: translate ONE point far away and linear CKA
    collapses, though the two sets are otherwise identical. The other two metrics
    must keep calling the pair similar — that disagreement is why CKA never runs alone."""
    X, good, _, rng = pairs
    v = rng.standard_normal(D)
    Y = good.copy()
    Y[0] += 1000.0 * v / np.linalg.norm(v)

    before, after = run_battery(X, good), run_battery(X, Y)

    assert before.cka > 0.95 and after.cka < 0.2
    assert after.mutual_knn > 0.9
    for seed in (0, 2):  # outlier lands in the fit split (seed 0) and the eval split (seed 2)
        assert procrustes_residual(X, Y, seed=seed) < 0.01


def test_run_battery_subsamples_consistently(pairs):
    X, good, _, _ = pairs
    res = run_battery(X, good, max_n=300)
    assert res.n == 300
    assert res.knn_chance == pytest.approx(chance_level(300, 10))
    assert res.mutual_knn > 0.8  # rows stayed paired through the subsample
