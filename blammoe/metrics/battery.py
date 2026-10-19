"""Metric battery + validate-the-validators gate (WP5; spec 1.3 item 5).

``run_battery`` scores one paired set (H_A, H_B) on three metrics that fail in
different ways — linear CKA (global, outlier-sensitive), mutual k-NN (local,
map-free) and held-out Procrustes residual (does a rotation actually transport
A to B). Never trust CKA alone (Davari et al., arXiv 2210.16156).

``rung1_gate`` is the "validate the validators" step: before the battery is
trusted on a real pair, every metric must clearly separate a known-good pair
(same run, different checkpoint) from a known-bad pair (trained vs random init).
If any metric fails to separate them, that metric is not usable on this data and
the gate fails.

The default gaps are a starting judgment call, not literature values; revisit
them once real Rung-1 numbers exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from blammoe.metrics.cka import linear_cka
from blammoe.metrics.knn import chance_level, mutual_knn
from blammoe.metrics.procrustes_residual import procrustes_residual

DEFAULT_MIN_GAP = {"cka": 0.1, "mutual_knn": 0.1, "procrustes_residual": 0.1}


@dataclass(frozen=True)
class BatteryResult:
    cka: float  # higher = more similar, in [0, 1]
    mutual_knn: float  # higher = more similar, in [0, 1]
    procrustes_residual: float  # LOWER = more similar; ~1 = no better than the mean
    knn_chance: float  # mutual_knn expected for unrelated spaces
    n: int


def run_battery(
    H_A: np.ndarray, H_B: np.ndarray, *, k: int = 10, max_n: int = 5000, seed: int = 0
) -> BatteryResult:
    """Score a paired set; rows beyond ``max_n`` are subsampled (same rows for A and B)."""
    A = np.asarray(H_A)
    B = np.asarray(H_B)
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"row counts differ: {A.shape[0]} vs {B.shape[0]}")
    n = A.shape[0]
    if n > max_n:
        rows = np.random.default_rng(seed).choice(n, size=max_n, replace=False)
        A, B, n = A[rows], B[rows], max_n
    return BatteryResult(
        cka=linear_cka(A, B),
        mutual_knn=mutual_knn(A, B, k=k),
        procrustes_residual=procrustes_residual(A, B, seed=seed),
        knn_chance=chance_level(n, k),
        n=n,
    )


@dataclass(frozen=True)
class GateResult:
    good: BatteryResult
    bad: BatteryResult
    separates: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.separates) and all(self.separates.values())


def rung1_gate(
    good: BatteryResult, bad: BatteryResult, min_gap: dict[str, float] | None = None
) -> GateResult:
    gap = {**DEFAULT_MIN_GAP, **(min_gap or {})}
    separates = {
        "cka": good.cka - bad.cka >= gap["cka"],
        "mutual_knn": good.mutual_knn - bad.mutual_knn >= gap["mutual_knn"],
        # residual is lower-is-better, so the bad pair must sit *above* the good one
        "procrustes_residual": bad.procrustes_residual - good.procrustes_residual
        >= gap["procrustes_residual"],
    }
    return GateResult(good=good, bad=bad, separates=separates)
