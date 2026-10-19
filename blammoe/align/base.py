"""Shared map type + apply/residual helpers for the alignment fitters (WP4, spec 1.4).

Every fitter returns a :class:`LinearMap` — an affine map ``H_A -> H_B`` stored
as means plus a matrix, so centering is undone on the way out:

    apply(map, H) = (H - mu_a) @ W + mu_b

``W`` is ``[d_a, d_b]`` (row-vector convention, rows of H are samples), so the
same type covers the square Procrustes case (Rungs 1-2) and the rectangular
least-squares case (Rung 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearMap:
    W: np.ndarray  # [d_a, d_b]
    mu_a: np.ndarray  # [d_a]; zeros when fitted with center=False
    mu_b: np.ndarray  # [d_b]
    method: str = ""

    @property
    def d_in(self) -> int:
        return self.W.shape[0]

    @property
    def d_out(self) -> int:
        return self.W.shape[1]


def as_pair(H_A: np.ndarray, H_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a paired activation set and promote to float64 for the solve."""
    A = np.asarray(H_A, dtype=np.float64)
    B = np.asarray(H_B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"expected 2-D [n, d] arrays, got {A.shape} and {B.shape}")
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"row counts differ: {A.shape[0]} vs {B.shape[0]} (pairs must align)")
    return A, B


def centered(
    A: np.ndarray, B: np.ndarray, center: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu_a = A.mean(axis=0) if center else np.zeros(A.shape[1])
    mu_b = B.mean(axis=0) if center else np.zeros(B.shape[1])
    return A - mu_a, B - mu_b, mu_a, mu_b


def apply(m: LinearMap, H: np.ndarray) -> np.ndarray:
    """Translate source states ``H`` ([n, d_a]) into target space ([n, d_b])."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape[-1] != m.d_in:
        raise ValueError(f"map expects d_in={m.d_in}, got {H.shape[-1]}")
    return (H - m.mu_a) @ m.W + m.mu_b


def residual(m: LinearMap, H_A: np.ndarray, H_B: np.ndarray) -> float:
    """Relative residual ``||apply(H_A) - H_B||^2 / ||H_B - mean(H_B)||^2``  (= 1 - R^2).

    0 is a perfect map. 1 is exactly what "predict the target's mean" scores —
    i.e. the Transmission Illusion baseline (architecture.md) — so a map is only
    carrying information if it lands clearly below 1 on *held-out* pairs.
    """
    A, B = as_pair(H_A, H_B)
    err = np.sum((apply(m, A) - B) ** 2)
    total = np.sum((B - B.mean(axis=0)) ** 2)
    if total == 0.0:
        raise ValueError("H_B has zero variance; relative residual is undefined")
    return float(err / total)
