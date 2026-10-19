"""WP4 acceptance: the fitters recover known maps on synthetic data (spec 1.4)."""
from pathlib import Path

import numpy as np
import pytest

from blammoe.align import (
    LinearMap,
    apply,
    fit_from_config,
    fit_linear,
    fit_orthogonal,
    residual,
)
from blammoe.config import AlignConfig, ExperimentConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))  # Haar-uniform, includes reflections


def test_procrustes_recovers_known_rotation_to_machine_precision():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((500, 32))
    R = random_orthogonal(32, rng)
    offset = rng.standard_normal(32)
    B = A @ R + offset

    m = fit_orthogonal(A, B)

    assert np.allclose(m.W, R, atol=1e-10)
    assert np.allclose(m.W.T @ m.W, np.eye(32), atol=1e-10)
    assert np.allclose(apply(m, A), B, atol=1e-10)
    assert residual(m, A, B) < 1e-20


def test_procrustes_scale_absorbs_global_norm_difference():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((400, 16))
    R = random_orthogonal(16, rng)
    B = 3.0 * A @ R

    rot_only = fit_orthogonal(A, B)
    scaled = fit_orthogonal(A, B, scale=True)

    assert np.allclose(scaled.W, 3.0 * R, atol=1e-10)
    assert residual(scaled, A, B) < 1e-20
    assert residual(rot_only, A, B) > 0.1  # a pure rotation can't fix the norm gap


def test_procrustes_rectangular_is_semi_orthogonal():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((300, 32))
    B = rng.standard_normal((300, 16))
    m = fit_orthogonal(A, B)
    assert m.W.shape == (32, 16)
    assert np.allclose(m.W.T @ m.W, np.eye(16), atol=1e-10)


def test_linear_recovers_rectangular_map():
    rng = np.random.default_rng(3)
    d_a, d_b = 20, 12
    A = rng.standard_normal((200, d_a))
    M = rng.standard_normal((d_a, d_b))
    B = A @ M + rng.standard_normal(d_b)

    m = fit_linear(A, B)

    assert (m.d_in, m.d_out) == (d_a, d_b)
    assert m.method == "linear_lstsq"
    assert np.allclose(m.W, M, atol=1e-10)
    assert residual(m, A, B) < 1e-20


def test_ridge_shrinks_toward_zero_as_lambda_grows():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((100, 10))
    B = A @ rng.standard_normal((10, 6)) + 0.1 * rng.standard_normal((100, 6))

    norms = [np.linalg.norm(fit_linear(A, B, ridge=lam).W) for lam in (0.0, 1.0, 1e2, 1e4, 1e12)]

    assert all(a > b for a, b in zip(norms, norms[1:]))  # strictly monotone
    assert norms[-1] < 1e-8


def test_ridge_rejects_negative_lambda():
    A = np.zeros((4, 2))
    with pytest.raises(ValueError):
        fit_linear(A, A, ridge=-1.0)


def test_heldout_residual_separates_signal_from_target_mean():
    """Transmission-Illusion baseline: predicting the target mean scores exactly 1;
    a real map on noisy data scores near its noise floor; an unrelated pair ~>= 1."""
    rng = np.random.default_rng(5)
    n, d = 2000, 24
    A = rng.standard_normal((n, d))
    R = random_orthogonal(d, rng)
    B = A @ R + 0.1 * rng.standard_normal((n, d))
    tr, te = slice(0, n // 2), slice(n // 2, n)

    good = fit_orthogonal(A[tr], B[tr])
    unrelated = fit_linear(A[tr], rng.standard_normal((n // 2, d)))
    mean_only = LinearMap(W=np.zeros((d, d)), mu_a=np.zeros(d), mu_b=B[te].mean(axis=0))

    assert residual(mean_only, A[te], B[te]) == pytest.approx(1.0)
    assert residual(good, A[te], B[te]) < 0.02  # noise variance 0.01 / signal 1.0
    assert residual(unrelated, A[te], B[te]) > 0.95


def test_fit_from_config_dispatches_on_method():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((50, 8))
    B = A @ random_orthogonal(8, rng)

    rung1 = ExperimentConfig.from_yaml(CONFIGS / "rung1.yaml")
    assert fit_from_config(rung1.align, A, B).method == "orthogonal_procrustes"
    assert fit_from_config(AlignConfig(method="linear_lstsq"), A, B).method == "linear_lstsq"
    ridge = fit_from_config(AlignConfig(method="linear_ridge", ridge_lambda=0.5), A, B)
    assert ridge.method == "linear_ridge"

    with pytest.raises(ValueError):
        fit_from_config(AlignConfig(method="linear_ridge", ridge_lambda=0.0), A, B)
    with pytest.raises(ValueError):
        fit_from_config(AlignConfig(method="vec2vec"), A, B)


def test_mismatched_row_counts_rejected():
    with pytest.raises(ValueError):
        fit_orthogonal(np.zeros((5, 3)), np.zeros((4, 3)))
