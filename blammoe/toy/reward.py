"""Stage-2 translator fine-tuning: reinforce what the fusion gets right *because of
the partner* (WP-T3).

Stage 1 taught the links to pass expert traces and choose partner experts.
Stage 2 keeps the models frozen and nudges the links toward fusion "wins". A win
at a token of model X is

    fused X top-1 is correct
    AND X alone (all links off) is wrong
    AND the same fusion fed a *wrong-document* partner is wrong

The last condition is the important one. Measured on stage-1 links, "novel"
predictions occur as often with a wrong partner as with the right one, so a
reward for novelty or correctness alone would credit noise and extra capacity.
Requiring the wrong-partner run to fail credits only wins that depend on what
the partner actually said.

The loss is ordinary next-token CE plus ``lam`` x CE on the win tokens (token-
level rejection-sampling fine-tuning on teacher-forced text).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def partner_win_mask(
    fused: torch.Tensor, solo: torch.Tensor, wrong_partner: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """[B, T] bool: tokens the fusion gets right that neither X alone nor X with a
    wrong partner gets right. Ignored targets (-100) are never wins."""
    valid = targets != -100
    hit = lambda logits: logits.argmax(-1) == targets  # noqa: E731
    return valid & hit(fused) & ~hit(solo) & ~hit(wrong_partner)


def win_weighted_loss(logits: torch.Tensor, targets: torch.Tensor, wins: torch.Tensor, lam: float) -> torch.Tensor:
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
    if lam and wins.any():
        loss = loss + lam * F.cross_entropy(logits[wins], targets[wins])
    return loss
