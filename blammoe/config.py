"""Experiment configuration (WP0).

A rung is fully described by one YAML file in configs/. Load with
``ExperimentConfig.from_yaml(path)``. Nested sections (pooling, align) are parsed
into their own dataclasses so downstream code gets typed access, not raw dicts.
"""
# NOTE: intentionally no `from __future__ import annotations` — _from_dict()
# introspects real field types (is_dataclass), which PEP 563 would stringify.
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PoolingConfig:
    """Character-span pooling for tokenizer mismatch (spec 1.3)."""

    method: str = "char_span"
    span_chars: int = 20  # width of the character window, e.g. chars 100-120
    boundary_margin: int = 3  # discard spans within N chars of a token boundary


@dataclass
class AlignConfig:
    """Supervised alignment fitter (spec 1.4)."""

    method: str = "orthogonal_procrustes"  # | "linear_lstsq" | "linear_ridge"
    ridge_lambda: float = 0.0  # only used when method == "linear_ridge"


@dataclass
class ExperimentConfig:
    rung: int
    model_a: str  # key into blammoe.models.registry.REGISTRY
    model_b: str
    layers: list[int] = field(default_factory=list)  # depth sweep (spec 1.3)
    expert_a: int | None = None
    expert_b: int | None = None
    corpus: list[str] = field(default_factory=lambda: ["wikipedia"])
    max_pairs: int = 100_000
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return _from_dict(cls, data)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Shallow, one-level-nested dataclass builder (no external deps)."""
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(f.type, value)  # type: ignore[arg-type]
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
