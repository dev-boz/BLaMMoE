"""Linear CKA (WP5; spec 1.3 item 5).

Kornblith et al. 2019 linear centered kernel alignment, in feature-space form:

    CKA(X, Y) = ||Y0^T X0||_F^2 / (||X0^T X0||_F * ||Y0^T Y0||_F)

with X0, Y0 column-centered. Invariant to rotation and isotropic scale.

Never use it alone: Davari et al. (arXiv 2210.16156, Cor. 3) prove that
translating a *single* point far enough drives linear CKA toward 0 between two
otherwise-identical representation sets. That is why this sits inside a battery.
"""
from __future__ import annotations

import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"row counts differ: {X.shape[0]} vs {Y.shape[0]}")
    X0 = X - X.mean(axis=0)
    Y0 = Y - Y.mean(axis=0)
    cross = np.sum((Y0.T @ X0) ** 2)
    denom = np.linalg.norm(X0.T @ X0) * np.linalg.norm(Y0.T @ Y0)
    if denom == 0.0:
        raise ValueError("a representation has zero variance; CKA is undefined")
    return float(cross / denom)
