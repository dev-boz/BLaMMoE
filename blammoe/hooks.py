"""Expert-FFN output capture (WP1 reference primitive).

Captures a single MoE expert's own FFN output *before* router weighting and
*before* summation into the residual stream. This is the hook point that makes
BLaMMoE a test of expert-level transmissibility rather than whole-model
alignment (spec 1.1).

Why a plain forward hook is sufficient: a `register_forward_hook` on the expert
submodule sees exactly that module's return value. The router weight multiply
(`gate * expert_out`) and the residual add both happen in the *enclosing* MoE
block, after the expert has returned — so both are excluded by construction, with
no need to understand the block's internal arithmetic.

Resolving the expert submodule for a real checkpoint uses
``ModelSpec.expert_module_template`` (e.g. "model.layers.{layer}.mlp.experts.
{expert}"). Those templates are VERIFY-tagged in the registry and must be checked
against ``model.named_modules()`` on the real download (WP1 acceptance).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Walk a dotted path (integer segments index into ModuleList/Sequential)."""
    mod: nn.Module = root
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def expert_path(template: str, layer: int, expert: int) -> str:
    return template.format(layer=layer, expert=expert)


class ExpertCapture:
    """Context manager capturing one expert FFN's output via a forward hook.

    Usage::

        path = expert_path(spec.expert_module_template, layer=6, expert=12)
        with ExpertCapture(model, path) as cap:
            model(**inputs)
        h = cap.output  # [*, hidden] pre-router-weight, pre-residual
    """

    def __init__(self, model: nn.Module, module_path: str):
        self.module = resolve_module(model, module_path)
        self.module_path = module_path
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self.output: torch.Tensor | None = None

    def _hook(self, module: nn.Module, inputs, output) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        self.output = tensor.detach()

    def __enter__(self) -> "ExpertCapture":
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False
