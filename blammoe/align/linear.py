"""General linear (rectangular) least-squares fitter (WP4; spec 1.4, Rung 3).

``ridge == 0`` -> unconstrained least squares, ``min_W ||A W - B||_F`` (minimum-norm
solution via ``lstsq``, so it is defined even when n < d_a).
``ridge > 0``  -> ``W = (A^T A + ridge * I)^{-1} A^T B``.

Data are centered first (by default), so the intercept lives in the stored means
and is never penalized. ``ridge`` is in units of summed squared activations — it
is not rescaled by n.

Per the spec, ridge is a hedge against small-n/large-p overfitting, *not* a
literature-backed default: try both variants empirically and compare held-out
residuals. (Atlas-Alignment, arXiv 2510.27413, saw plain linear regression
degrade with more data where Procrustes kept improving — another reason to always
report a Procrustes baseline alongside this.)
"""
from __future__ import annotations

import numpy as np

from blammoe.align.base import LinearMap, as_pair, centered


def fit_linear(
    H_A: np.ndarray, H_B: np.ndarray, ridge: float = 0.0, *, center: bool = True
) -> LinearMap:
    if ridge < 0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    A, B = as_pair(H_A, H_B)
    A0, B0, mu_a, mu_b = centered(A, B, center)
    if ridge == 0.0:
        W, *_ = np.linalg.lstsq(A0, B0, rcond=None)
        method = "linear_lstsq"
    else:
        gram = A0.T @ A0
        gram[np.diag_indices_from(gram)] += ridge
        W = np.linalg.solve(gram, A0.T @ B0)
        method = "linear_ridge"
    return LinearMap(W=W, mu_a=mu_a, mu_b=mu_b, method=method)
