"""Held-out Procrustes residual (WP5; spec 1.3 item 5).

Fit similarity Procrustes (rotation + isotropic scale, WP4) on one split of the
pairs and report the relative residual (1 - R^2) on the other: ~0 means a
rotation explains B from A, ~1 means it does no better than predicting B's mean.

Rows are L2-normalized first by default — the Atlas-Alignment setup
(arXiv 2510.27413), where Procrustes on row-normalized activations amounts to
minimising cosine distance. Besides matching that paper, it caps any single
sample's leverage on the fit, so this metric is not hijacked by the norm
outliers that break linear CKA (Davari et al., arXiv 2210.16156).

Accepts d_A != d_B (semi-orthogonal map), so the same metric runs at Rung 3.
"""
from __future__ import annotations

import numpy as np

from blammoe.align import fit_orthogonal, residual


def _row_normalize(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(norms == 0.0, 1.0, norms)


def procrustes_residual(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    test_frac: float = 0.5,
    normalize: bool = True,
    seed: int = 0,
) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"row counts differ: {n} vs {Y.shape[0]}")
    if normalize:
        X, Y = _row_normalize(X), _row_normalize(Y)
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_frac))
    if not 1 < n_test < n - 1:
        raise ValueError(f"test_frac={test_frac} leaves too few rows on one side (n={n})")
    te, tr = perm[:n_test], perm[n_test:]
    m = fit_orthogonal(X[tr], Y[tr], scale=True)
    return residual(m, X[te], Y[te])
