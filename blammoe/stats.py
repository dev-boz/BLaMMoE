"""Uncertainty for per-document results: bootstrap CIs and seed summaries.

Accuracy here is pooled: sum(correct) / sum(total) over documents, so a document with
more answers weighs more. Resampling documents (not answers) respects that answers in
one document are correlated. ``paired_bootstrap_diff`` resamples the SAME documents for
both conditions, which is what makes "fused minus wrong-partner" or "experts minus
residual" comparisons tight. A resample that happens to draw only answer-less documents has no
accuracy (0/0): it is dropped from the CI, not counted as 0, and ``n_dropped`` says how many.
"""
from __future__ import annotations

import numpy as np


def _resampled(rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Pooled accuracy of each bootstrap resample; NaN where it drew no answers."""
    total = rows[idx, 1].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, rows[idx, 0].sum(1) / total, np.nan)


def _ci(stats: np.ndarray, alpha: float) -> tuple[float, float, int]:
    ok = stats[~np.isnan(stats)]
    if not len(ok):
        return float("nan"), float("nan"), len(stats)
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), len(stats) - len(ok)


def _pooled(rows: np.ndarray) -> float:
    total = rows[:, 1].sum()
    return float(rows[:, 0].sum() / total) if total else float("nan")


def bootstrap_ci(per_doc: list[list[int]], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """per_doc: [[correct, total], ...] -> pooled accuracy with a percentile CI."""
    rows = np.asarray(per_doc, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    lo, hi, dropped = _ci(_resampled(rows, idx), alpha)
    return {"acc": _pooled(rows), "lo": lo, "hi": hi, "n_docs": len(rows), "n_dropped": dropped}


def paired_bootstrap_diff(a: list[list[int]], b: list[list[int]], n_boot: int = 2000, alpha: float = 0.05,
                          seed: int = 0) -> dict:
    """Pooled accuracy of a minus b over the same documents, with a paired percentile CI."""
    A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if A.shape != B.shape:
        raise ValueError("paired bootstrap needs the same documents in both conditions")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), size=(n_boot, len(A)))
    lo, hi, dropped = _ci(_resampled(A, idx) - _resampled(B, idx), alpha)
    return {"diff": _pooled(A) - _pooled(B), "lo": lo, "hi": hi, "n_docs": len(A), "n_dropped": dropped}


def seed_summary(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n_seeds": len(v)}
