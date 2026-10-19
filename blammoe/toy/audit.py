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

from blammoe.toy.batching import make_batch
from blammoe.toy.corpus import Sample
from blammoe.toy.fused import NEVER, FusedPair, closed_token_ends, run_pair, token_ends


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
        dev = next(pair.parameters()).device
        if safe:
            fwd_kw = {**fwd_kw, "write_ends_a": cl_a.to(dev), "write_ends_b": cl_b.to(dev)}
        la, _ = pair(ids_a.to(dev), ids_b.to(dev), ends_a.to(dev), ends_b.to(dev), **fwd_kw)
        rows = torch.arange(len(last), device=dev)
        lp = la[rows, last.to(dev)].log_softmax(-1)
        out.append(-lp[rows, tgt.to(dev)].cpu())
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


# --------------------------------------------------------------------------- per-position gate
def _fill(rows: list[list[int]], fill: int) -> torch.Tensor:
    out = torch.full((len(rows), max(len(r) for r in rows)), fill, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r)
    return out


@torch.no_grad()
def honest_logprobs(
    pair: FusedPair, side: str, text_a: str, text_b: str, positions: list[int], tok_a, tok_b, L: int,
    batch: int = 32, **fwd_kw,
) -> torch.Tensor:
    """[P, V] log-probs of model ``side`` ("a" | "b") at its input ``positions``, each re-run on
    only what that position has read. The reader keeps its own full-text tokens 0..t (with
    their full-text closed ends); the partner is re-tokenized on ``partner_text[:e_t]``, e_t =
    chars read by position t. Works in both directions and for any pair of tokenizers."""
    own_tok, own_text, p_tok, p_text = (tok_a, text_a, tok_b, text_b) if side == "a" else (tok_b, text_b, tok_a, text_a)
    full = make_batch([Sample(text=own_text, blind=own_text)], own_tok, L)
    own_ends, own_closed = token_ends(full)[0].tolist(), closed_token_ends(full)[0].tolist()
    dev = next(pair.parameters()).device
    out = []
    for k in range(0, len(positions), batch):
        chunk = positions[k : k + batch]
        o_ids, o_end, o_cl, p_ids, p_end, p_cl = [], [], [], [], [], []
        for t in chunk:
            e = own_ends[t]
            o_ids.append(full.ids[0, : t + 1].tolist())
            o_end.append(own_ends[: t + 1])
            o_cl.append(own_closed[: t + 1])
            ids, off = p_tok.encode(p_text[:e])
            p_ids.append(ids)
            p_end.append([x for _, x in off])
            p_cl.append(p_tok.closed_ends(p_text[:e], off))
        own = (_fill(o_ids, own_tok.pad_id), _fill(o_end, NEVER), _fill(o_cl, NEVER))
        par = (_fill(p_ids, p_tok.pad_id), _fill(p_end, NEVER), _fill(p_cl, NEVER))
        (ia, ea, ca), (ib, eb, cb) = (own, par) if side == "a" else (par, own)
        if not pair.safe_mask:
            ca, cb = ea, eb
        la, lb = pair(ia.to(dev), ib.to(dev), ea.to(dev), eb.to(dev),
                      write_ends_a=ca.to(dev), write_ends_b=cb.to(dev), **fwd_kw)
        lo = la if side == "a" else lb
        out.append(lo[torch.arange(len(chunk), device=dev), torch.tensor(chunk, device=dev)].log_softmax(-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def causality_gate(
    pair: FusedPair, docs: list[tuple[str, str]], tok_a, tok_b, L: int, tol: float = 1e-4,
    batch: int = 32, **fwd_kw,
) -> dict:
    """Full-text vs prefix-honest log-probs at EVERY scored position of every (text_a, text_b)
    doc, both models. ``ok`` is False if any position differs by more than ``tol``: then some
    model's prediction depended on chars it had not read."""
    dev = next(pair.parameters()).device
    res = {s: {"max_diff": 0.0, "n_positions": 0, "n_bad": 0, "worst": None} for s in ("a", "b")}
    for d, (text_a, text_b) in enumerate(docs):
        a = make_batch([Sample(text=text_a, blind=text_a)], tok_a, L, device=dev)
        b = make_batch([Sample(text=text_b, blind=text_b)], tok_b, L, device=dev)
        la, lb = run_pair(pair, a, b, **fwd_kw)
        for side, logits, bt in (("a", la, a), ("b", lb, b)):
            pos = [t for t in range(bt.ids.shape[1]) if bt.targets[0, t] != -100]
            full = logits[0, pos].log_softmax(-1).cpu()
            honest = honest_logprobs(pair, side, text_a, text_b, pos, tok_a, tok_b, L, batch, **fwd_kw)
            diff = (full - honest).abs().amax(-1)
            r = res[side]
            r["n_positions"] += len(pos)
            r["n_bad"] += int((diff > tol).sum())
            if float(diff.max()) > r["max_diff"]:
                j = int(diff.argmax())
                r["max_diff"], r["worst"] = float(diff[j]), {"doc": d, "position": pos[j]}
    res["tol"] = tol
    res["ok"] = res["a"]["n_bad"] == 0 and res["b"]["n_bad"] == 0
    return res
