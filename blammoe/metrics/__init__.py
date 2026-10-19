"""Representational-similarity metric battery (WP5, spec 1.3 item 5)."""
from blammoe.metrics.battery import BatteryResult, GateResult, run_battery, rung1_gate
from blammoe.metrics.cka import linear_cka
from blammoe.metrics.knn import mutual_knn
from blammoe.metrics.procrustes_residual import procrustes_residual

__all__ = [
    "BatteryResult",
    "GateResult",
    "run_battery",
    "rung1_gate",
    "linear_cka",
    "mutual_knn",
    "procrustes_residual",
]
