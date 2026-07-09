"""Model registry — concrete checkpoint specs per rung.

Fields tagged VERIFY are best-effort values that MUST be confirmed against the
real checkpoint before any code relies on them:
  - `expert_module_template` — confirm against `model.named_modules()`.
  - dimension fields          — confirm against the HF `config.json`.
Do not trust them until WP1 has checked them on the actual download. They are
here so an agent starts from a candidate, not a blank page.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str | None = None  # HF branch/tag; None == default (main)
    family: str = ""
    # VERIFY against named_modules() on the real checkpoint. `{layer}` and
    # `{expert}` are .format() placeholders resolved by blammoe.hooks.
    expert_module_template: str | None = None
    hidden_size: int | None = None  # VERIFY from config.json
    num_layers: int | None = None  # VERIFY
    num_experts: int | None = None  # VERIFY (routed experts, excl. shared)
    num_active: int | None = None  # VERIFY (top-k)
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    REGISTRY[spec.key] = spec
    return spec


def get(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]


# --- OLMoE family (spec: fine-grained, domain-specialized; Part 2 §C) ----------
# Numbers below are the published OLMoE-1B-7B config; VERIFY on download.
register(ModelSpec(
    key="olmoe-base",
    hf_id="allenai/OLMoE-1B-7B-0924",
    family="olmoe",
    expert_module_template="model.layers.{layer}.mlp.experts.{expert}",  # VERIFY
    hidden_size=2048, num_layers=16, num_experts=64, num_active=8,  # VERIFY
    notes="Rung 2/3 model A. Base pretrained checkpoint.",
))
register(ModelSpec(
    key="olmoe-instruct",
    hf_id="allenai/OLMoE-1B-7B-0924-Instruct",
    family="olmoe",
    expert_module_template="model.layers.{layer}.mlp.experts.{expert}",  # VERIFY
    hidden_size=2048, num_layers=16, num_experts=64, num_active=8,  # VERIFY
    notes="Rung 2 model B. Same family/arch, different fine-tune from olmoe-base.",
))
register(ModelSpec(
    key="olmoe-base-ckpt-early",
    hf_id="allenai/OLMoE-1B-7B-0924",
    revision="VERIFY-pick-an-intermediate-branch",  # HF has intermediate steps
    family="olmoe",
    expert_module_template="model.layers.{layer}.mlp.experts.{expert}",  # VERIFY
    hidden_size=2048, num_layers=16, num_experts=64, num_active=8,  # VERIFY
    notes="Rung 1: SAME run as olmoe-base, EARLIER checkpoint. Pick a real "
          "revision from the model's 'Files and versions' branches.",
))

# --- Qwen-MoE family (Rung 3 cross-family target) ------------------------------
register(ModelSpec(
    key="qwen-moe",
    hf_id="Qwen/Qwen1.5-MoE-A2.7B",
    family="qwen_moe",
    expert_module_template="model.layers.{layer}.mlp.experts.{expert}",  # VERIFY
    hidden_size=2048, num_layers=24, num_experts=60, num_active=4,  # VERIFY
    notes="Rung 3 model B. Has shared experts too (excluded from count above); "
          "confirm whether dims match OLMoE — decides Procrustes vs lstsq.",
))
