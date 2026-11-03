"""Toy-POC config (WP-T): one YAML (configs/toy_poc.yaml) drives the whole run."""
# NOTE: no `from __future__ import annotations` — blammoe.config._from_dict needs real types.
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from blammoe.config import _from_dict


@dataclass
class ToyModelSpec:
    tokenizer: str = "char"  # "char" | "bpe"
    bpe_vocab: int = 1024
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_experts: int = 8
    top_k: int = 2
    d_expert: int = 256
    seed: int = 0
    p_prose: float = 0.5  # prose share of the non-recall pretraining windows (and of the BPE fit); 0.5 = every run before it


@dataclass
class BridgeSpec:
    layer_a: int = 3  # A layer whose expert is transmitted
    layer_b: int = 1  # inject into B's residual stream after this block
    d_bus: int = 64  # Universal Latent Bus width
    n_heads: int = 4
    steps: int = 600
    batch_size: int = 16
    lr: float = 2e-3
    answer_weight: float = 1.0  # extra loss weight on recall-answer tokens


@dataclass
class FusedSpec:
    """Lockstep fusion (WP-T2). ``variants`` maps a name to {pair: ab|bb|aa, links: [...], signal}."""

    steps: int = 1500
    batch_size: int = 16
    lr: float = 2e-3
    answer_weight: float = 1.0
    d_bus: int = 64
    n_heads: int = 4
    p_split: float = 0.5  # share of split-view recall samples (rest: prose/code, both see it); a variant may override
    seed: int = 0  # bridge init and training-draw order (0 = every run before the sweep's r1 retry)
    variants: dict = field(default_factory=dict)


@dataclass
class Stage2Spec:
    """WP-T3 translator fine-tuning on ordinary text (scripts/toy_fused_reward.py)."""

    steps: int = 1000
    batch_size: int = 16
    lr: float = 2e-4
    lam: float = 2.0  # extra CE weight on partner-dependent win tokens (reward arm only)


@dataclass
class HideSpec:
    """Attribution controls for an expert-bus fusion (audit M3, scripts/toy_fused_hide.py):
    the recovery fine-tune that retrains the links with the chosen experts hidden."""

    finetune_steps: int = 300  # 0 = skip the fine-tunes
    finetune_warmup: int = 50  # the links are already trained: a fresh Adam needs warmup
    lr: float = 2e-3


@dataclass
class ToyPOCConfig:
    out_dir: str = "results/toy_poc"
    seq_chars: int = 256
    p_recall: float = 0.5
    val_share: float = 0.1  # file-level corpus split (audit M6): val = selection, test = reported
    test_share: float = 0.1
    steps: int = 4000
    batch_size: int = 32
    lr: float = 3e-3
    model_a: ToyModelSpec = field(default_factory=ToyModelSpec)
    model_b: ToyModelSpec = field(default_factory=lambda: ToyModelSpec(tokenizer="bpe", seed=1))
    bridge: BridgeSpec = field(default_factory=BridgeSpec)
    fused: FusedSpec = field(default_factory=FusedSpec)
    stage2: Stage2Spec = field(default_factory=Stage2Spec)
    hide: HideSpec = field(default_factory=HideSpec)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ToyPOCConfig":
        return _from_dict(cls, yaml.safe_load(Path(path).read_text()) or {})
