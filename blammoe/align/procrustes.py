"""Orthogonal Procrustes fitter (WP4; spec 1.4, Rungs 1-2).

Solves ``min_R ||A R - B||_F`` subject to ``R^T R = I`` via the SVD of the
cross-covariance ``A^T B = U S V^T``, giving ``R = U V^T`` (Schönemann 1966).
With ``scale=True`` an isotropic scale ``s = tr(S) / ||A||_F^2`` is also fitted
(similarity Procrustes), which absorbs a global norm difference between the two
checkpoints' expert outputs without giving up the rotation-only constraint.

When ``d_a != d_b`` the same solve returns a *semi*-orthogonal ``R`` (orthonormal
columns if ``d_a >= d_b``, rows otherwise). The spec reserves Procrustes for the
matched-dim rungs, but this variant is cheap to have as a Rung-3 baseline:
Atlas-Alignment (arXiv 2510.27413) maps between spaces of different width this
way and finds it beats unconstrained linear regression for retrieval/steering.
"""
from __future__ import annotations

import numpy as np

from blammoe.align.base import LinearMap, as_pair, centered


def fit_orthogonal(
    H_A: np.ndarray, H_B: np.ndarray, *, center: bool = True, scale: bool = False
) -> LinearMap:
    A, B = as_pair(H_A, H_B)
    A0, B0, mu_a, mu_b = centered(A, B, center)
    U, S, Vt = np.linalg.svd(A0.T @ B0, full_matrices=False)
    R = U @ Vt
    if scale:
        norm_a = np.sum(A0**2)
        R = R * (S.sum() / norm_a if norm_a > 0 else 0.0)
    return LinearMap(W=R, mu_a=mu_a, mu_b=mu_b, method="orthogonal_procrustes")
