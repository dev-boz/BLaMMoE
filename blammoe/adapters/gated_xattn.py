"""Tx/Rx adapters with tanh-gated cross-attention injection (WP6; spec 1.1).

Hub-and-spoke: each model gets one ``TxAdapter`` (its space -> the shared bus)
and one ``RxAdapter`` (bus -> its residual stream). The Rx side is the Flamingo
(arXiv 2204.14198) / CALM (arXiv 2401.02412) pattern: the target's residual
stream cross-attends over the foreign bus sequence and the result is added back
through ``tanh(gate)``, with ``gate`` initialised to 0. At init the target model
is therefore *exactly* unchanged; training opens the gate only as far as the
foreign signal helps — the anti-interference property the spec relies on.

Cross-attention (rather than a pointwise add) also absorbs tokenizer mismatch:
each target position may read any source position the mask allows, so the
caller only has to supply a causal mask (which source positions have read no
more text than the target position has).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TxAdapter(nn.Module):
    """Model space (d_model) -> bus (d_bus)."""

    def __init__(self, d_model: int, d_bus: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_bus)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))


class RxAdapter(nn.Module):
    """Bus -> target residual stream via tanh-gated cross-attention (gate init 0)."""

    def __init__(self, d_model: int, d_bus: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_bus)
        self.attn = nn.MultiheadAttention(d_bus, n_heads, batch_first=True)
        self.out = nn.Linear(d_bus, d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, bus: torch.Tensor, block_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: [B, T, d_model]; bus: [B, S, d_bus]; block_mask: [B, T, S] bool, True = may not attend."""
        mask = None
        if block_mask is not None:
            mask = block_mask.repeat_interleave(self.n_heads, dim=0)
        read, _ = self.attn(self.q(self.norm(x)), bus, bus, attn_mask=mask, need_weights=False)
        return x + torch.tanh(self.gate) * self.out(read)
