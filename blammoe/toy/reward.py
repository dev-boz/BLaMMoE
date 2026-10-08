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

The loss is ordinary next-token CE with win tokens weighted ``1 + lam`` (token-
level rejection-sampling fine-tuning on teacher-forced text). Both terms are
normalised by the number of valid tokens. An earlier version averaged the win
term over the wins alone. With wins at ~2% of tokens that gave each win ~100x the
weight of an ordinary token, and the links learned to push predictions away from
the solo model everywhere: more novelty, much worse accuracy.
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
    """Mean CE over valid tokens, with win tokens counted ``1 + lam`` times."""
    ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="none")
    valid = (targets != -100).flatten()
    w = 1.0 + lam * wins.flatten().float()
    return (ce * w)[valid].sum() / valid.sum()
