"""Supervised alignment fitters H_A -> H_B (WP4, spec 1.4).

``fit_from_config`` is the config-driven entry point: it dispatches on
``AlignConfig.method`` so a rung YAML alone decides which fitter runs.
"""
from __future__ import annotations

import numpy as np

from blammoe.align.base import LinearMap, apply, residual
from blammoe.align.linear import fit_linear
from blammoe.align.procrustes import fit_orthogonal
from blammoe.config import AlignConfig

__all__ = ["LinearMap", "apply", "residual", "fit_linear", "fit_orthogonal", "fit_from_config"]


def fit_from_config(cfg: AlignConfig, H_A: np.ndarray, H_B: np.ndarray) -> LinearMap:
    if cfg.method == "orthogonal_procrustes":
        return fit_orthogonal(H_A, H_B)
    if cfg.method == "linear_lstsq":
        return fit_linear(H_A, H_B, ridge=0.0)
    if cfg.method == "linear_ridge":
        if cfg.ridge_lambda <= 0:
            raise ValueError("method 'linear_ridge' needs ridge_lambda > 0")
        return fit_linear(H_A, H_B, ridge=cfg.ridge_lambda)
    raise ValueError(f"unknown align.method {cfg.method!r}")
