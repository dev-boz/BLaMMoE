"""Mutual k-nearest-neighbour alignment (WP5; spec 1.3 item 5).

For each paired sample i, take its k nearest neighbours (cosine, self excluded)
within X and, separately, within Y; score the overlap |N_X(i) ∩ N_Y(i)| / k and
average over i. This is the metric of Huh et al. 2024, "The Platonic
Representation Hypothesis" (arXiv 2405.07987). It needs no fitted map and only
looks at local neighbourhood structure, so — unlike linear CKA — one far-away
point cannot dominate it. Chance level for unrelated spaces is ~k / (n - 1).

Memory is O(n^2); subsample rows (``battery.run_battery(max_n=...)``) for big n.
"""
from __future__ import annotations

import numpy as np


def _knn_indices(Z: np.ndarray, k: int) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / np.where(norms == 0.0, 1.0, norms)
    sim = Zn @ Zn.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, k, axis=1)[:, :k]


def mutual_knn(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"row counts differ: {n} vs {Y.shape[0]}")
    if not 0 < k < n:
        raise ValueError(f"need 0 < k < n, got k={k}, n={n}")
    nx, ny = _knn_indices(X, k), _knn_indices(Y, k)
    overlap = [np.intersect1d(nx[i], ny[i], assume_unique=True).size for i in range(n)]
    return float(np.mean(overlap) / k)


def chance_level(n: int, k: int) -> float:
    """Expected mutual-kNN score for two unrelated representations of n points."""
    return k / (n - 1)
