"""Prefix-honest evaluation of a fused pair: a cross-tokenizer causality audit.

The leak this guards against: in normal evaluation the BPE partner (B) is
tokenized on the WHOLE text. Where a BPE token boundary falls depends on the
characters after it: "the" is one token, so no B token ends after "th", but
"thx" may split there. So when the char-level model (A) has read
``text[:i+1]`` and reads B's states for tokens ending at or before char i+1, the
mere existence and identity of those tokens can carry information about char
i+1 and beyond. The chars-read mask cannot see this, and neither can the
wrong-document, wrong-position or copy-of-itself controls (a same-tokenizer copy
has no boundary to leak across).

Prefix-honest evaluation removes it: to score A's prediction of char i+1, the
partner is re-tokenized on ``text[:i+1]`` alone and the fused pair is re-run on
that prefix. The last B token is then cut at the prefix end, so nothing about
later chars can reach A. Comparing A's honest score with its normal fused score
measures the leak directly.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from blammoe.toy.fused import FusedPair, NEVER


def _pad(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    T = max(len(s) for s in seqs)
    out = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def _ends(offsets: list[list[tuple[int, int]]], T: int) -> torch.Tensor:
    ends = torch.full((len(offsets), T), NEVER, dtype=torch.long)
    for i, off in enumerate(offsets):
        ends[i, : len(off)] = torch.tensor([e for _, e in off])
    return ends


def prefix_inputs(text: str, positions: list[int], tok_a, tok_b):
    """For each A input position i (A has read text[:i+1] and predicts text[i+1]):
    A gets chars 0..i, B gets tokenize(text[:i+1]) - every token as input, none
    peeking past char i+1. Returns (ids_a, ends_a, ids_b, ends_b, target_a, last_a,
    closed_a, closed_b); the last two are writer ends for a safe-mask fusion."""
    a_seqs, a_offs, b_seqs, b_offs, tgt, a_cl, b_cl = [], [], [], [], [], [], []
    for i in positions:
        prefix = text[: i + 1]
        ia, oa = tok_a.encode(prefix)
        ib, ob = tok_b.encode(prefix)
        assert ob and ob[-1][1] <= i + 1, "B prefix tokens must not pass the prefix end"
        a_seqs.append(ia)
        a_offs.append(oa)
        b_seqs.append(ib)
        b_offs.append(ob)
        a_cl.append([(0, c) for c in tok_a.closed_ends(prefix, oa)])
        b_cl.append([(0, c) for c in tok_b.closed_ends(prefix, ob)])
        tgt.append(tok_a.encode(text[i + 1])[0][0])
    ids_a, ids_b = _pad(a_seqs, tok_a.pad_id), _pad(b_seqs, tok_b.pad_id)
    return (ids_a, _ends(a_offs, ids_a.shape[1]), ids_b, _ends(b_offs, ids_b.shape[1]),
            torch.tensor(tgt), torch.tensor([len(s) - 1 for s in a_seqs]),
            _ends(a_cl, ids_a.shape[1]), _ends(b_cl, ids_b.shape[1]))


@torch.no_grad()
def honest_a_nll(
    pair: FusedPair, text: str, positions: list[int], tok_a, tok_b, batch: int = 64, safe: bool = False,
    **fwd_kw,
) -> torch.Tensor:
    """A's NLL (nats) of text[i+1] at each position i, partner re-run on the prefix only.
    ``safe``: score a fusion that uses closed-chunk writer ends (as it was trained)."""
    out = []
    for k in range(0, len(positions), batch):
        ids_a, ends_a, ids_b, ends_b, tgt, last, cl_a, cl_b = prefix_inputs(
            text, positions[k : k + batch], tok_a, tok_b)
        if safe:
            fwd_kw = {**fwd_kw, "write_ends_a": cl_a, "write_ends_b": cl_b}
        la, _ = pair(ids_a, ids_b, ends_a, ends_b, **fwd_kw)
        lp = la[torch.arange(len(last)), last].log_softmax(-1)
        out.append(-lp[torch.arange(len(tgt)), tgt])
    return torch.cat(out)


@torch.no_grad()
def full_text_a_nll(pair: FusedPair, a_batch, b_batch, ends_a, ends_b, **fwd_kw) -> torch.Tensor:
    """A's per-position NLL [B, T] (nats) with the partner tokenized on the full text, the
    standard way. -inf-free: ignored targets give 0."""
    la, _ = pair(a_batch.ids, b_batch.ids, ends_a, ends_b, **fwd_kw)
    nll = F.cross_entropy(la.flatten(0, 1), a_batch.targets.flatten(), ignore_index=-100, reduction="none")
    return nll.view_as(a_batch.targets)


def to_bits(nll_nats: torch.Tensor) -> float:
    return float(nll_nats.mean() / math.log(2))
