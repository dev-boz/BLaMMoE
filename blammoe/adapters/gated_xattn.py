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
    """Bus -> target residual stream via tanh-gated cross-attention (gate init 0).

    ``n_rel_buckets > 0`` adds a learned per-head attention bias indexed by a
    caller-supplied bucket for each (target, bus entry) pair — e.g. bucketed
    *character* distance, the one coordinate two models with different
    tokenizers and position embeddings share. Zero-initialised, so it changes
    nothing until trained.
    """

    def __init__(self, d_model: int, d_bus: int, n_heads: int = 4, n_rel_buckets: int = 0):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_bus)
        self.attn = nn.MultiheadAttention(d_bus, n_heads, batch_first=True)
        self.out = nn.Linear(d_bus, d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.rel = None
        if n_rel_buckets:
            self.rel = nn.Embedding(n_rel_buckets, n_heads)
            nn.init.zeros_(self.rel.weight)

    def forward(
        self, x: torch.Tensor, bus: torch.Tensor, block_mask: torch.Tensor | None = None,
        return_weights: bool = False, rel_buckets: torch.Tensor | None = None,
    ):
        """x: [B, T, d_model]; bus: [B, S, d_bus]; block_mask: [B, T, S] bool, True = may not attend;
        rel_buckets: [B, T, S] long (only used if the adapter has ``n_rel_buckets``).

        With ``return_weights`` also returns head-averaged attention [B, T, S] — i.e.
        which bus entries the target chose to read.
        """
        mask = None
        if block_mask is not None:
            mask = block_mask.repeat_interleave(self.n_heads, dim=0)  # [B*H, T, S], batch-major
        if self.rel is not None and rel_buckets is not None:
            B, T, S = rel_buckets.shape
            bias = self.rel(rel_buckets).permute(0, 3, 1, 2).reshape(B * self.n_heads, T, S)
            mask = bias if mask is None else bias.masked_fill(mask, float("-inf"))
        read, w = self.attn(self.q(self.norm(x)), bus, bus, attn_mask=mask, need_weights=return_weights)
        y = x + torch.tanh(self.gate) * self.out(read)
        return (y, w) if return_weights else y
