"""Train and evaluate a FusedPair (WP-T2), device-agnostic and importable.

Shared by scripts/toy_fused.py (one variant, CPU box) and the Colab sweep
(scripts/toy_sweep.py: many seeds/pairs on a GPU). Batches are built on the CPU and
moved to the pair's device.
"""
from __future__ import annotations

import random
from collections.abc import Callable

import torch

from blammoe.toy.batching import Batch, make_batch
from blammoe.toy.corpus import MixedSampler, Sample, split_recall_sample
from blammoe.toy.fused import (
    FusedPair, bits_per_char, char_predictions, first_char_matrix, pair_loss, run_pair, subset_accuracy,
)
from blammoe.toy.tokenizers import CharTokenizer


def device_of(pair: FusedPair) -> torch.device:
    return next(pair.parameters()).device


def mixed_draw(prose: str, code: str, L: int, p_split: float, seed: int):
    """Training items: split-view recall pairs (each model sees its own view) or ordinary
    text (both see it), mixed p_split : 1 - p_split."""
    rng = random.Random(seed)
    general = MixedSampler(prose, code, L, p_recall=0.0, seed=seed + 1)

    def draw(n: int) -> list[tuple[Sample, Sample]]:
        items = []
        for _ in range(n):
            if rng.random() < p_split:
                s = split_recall_sample(rng, L)
                items.append((s.view("a"), s.view("b")))
            else:
                g = general.sample()
                items.append((g, g))
        return items

    return draw


def train_fused(
    pair: FusedPair, tok1, tok2, draw: Callable, L: int, *, steps: int, batch_size: int, lr: float,
    answer_weight: float, log: Callable[[str], None] = print, log_every: int = 100,
) -> list[dict]:
    dev = device_of(pair)
    opt = torch.optim.AdamW(pair.bridge_parameters(), lr=lr, weight_decay=0.0)
    history = []
    for step in range(steps):
        items = draw(batch_size)
        b1 = make_batch([x for x, _ in items], tok1, L, device=dev)
        b2 = make_batch([y for _, y in items], tok2, L, device=dev)
        l1, l2 = run_pair(pair, b1, b2)
        loss = pair_loss(l1, b1, answer_weight) + pair_loss(l2, b2, answer_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            gates = {k: [round(float(torch.tanh(lk.rx_a.gate.detach())), 3),
                         round(float(torch.tanh(lk.rx_b.gate.detach())), 3)] for k, lk in pair.links.items()}
            rec = {"step": step, "loss": round(float(loss.detach()), 4), "gates": gates}
            history.append(rec)
            log(str(rec))
    pair.eval()
    return history


def split_eval_set(L: int, n: int = 96, seed: int = 2025):
    rng = random.Random(seed)
    return [split_recall_sample(rng, L) for _ in range(n)]


def split_scores(pair: FusedPair, splits, tok1, tok2, L: int, per_doc: bool = False, **kw) -> dict:
    """Accuracy on answers each model needs its partner for / can read itself.
    ``per_doc``: also return per-document (correct, total) for bootstrapping."""
    dev = device_of(pair)
    s1 = make_batch([s.view("a") for s in splits], tok1, L, device=dev)
    s2 = make_batch([s.view("b") for s in splits], tok2, L, device=dev)
    with torch.no_grad():
        o1, o2 = run_pair(pair, s1, s2, **kw)
    out = {
        "m1_needs_partner": subset_accuracy(o1, s1, [s.partner_answers("a") for s in splits]),
        "m1_own": subset_accuracy(o1, s1, [s.own_answers("a") for s in splits]),
        "m2_needs_partner": subset_accuracy(o2, s2, [s.partner_answers("b") for s in splits]),
        "m2_own": subset_accuracy(o2, s2, [s.own_answers("b") for s in splits]),
    }
    if per_doc:
        out["per_doc"] = {
            "m1_needs_partner": _per_doc(o1, s1, [s.partner_answers("a") for s in splits]),
            "m2_needs_partner": _per_doc(o2, s2, [s.partner_answers("b") for s in splits]),
        }
    return out


def _per_doc(logits: torch.Tensor, batch: Batch, chars) -> list[list[int]]:
    rows = []
    for i, c in enumerate(chars):
        r = subset_accuracy(logits[i : i + 1], _row(batch, i), [c])
        rows.append([round(r["acc"] * r["n"]) if r["n"] else 0, r["n"]])
    return rows


def _row(batch: Batch, i: int) -> Batch:
    return Batch(ids=batch.ids[i : i + 1], targets=batch.targets[i : i + 1], offsets=batch.offsets[i : i + 1],
                 answer_pos=batch.answer_pos[i : i + 1], full_offsets=batch.full_offsets[i : i + 1],
                 closed=batch.closed[i : i + 1] if batch.closed else None)


def general_eval_set(prose: str, code: str, L: int, n: int = 48, seed: int = 777) -> list[Sample]:
    gs = MixedSampler(prose, code, L, p_recall=0.0, seed=seed)
    return [gs.sample() for _ in range(n)]


def general_bpc(pair: FusedPair, docs, tok1, tok2, L: int, **kw) -> dict:
    dev = device_of(pair)
    g1, g2 = make_batch(docs, tok1, L, device=dev), make_batch(docs, tok2, L, device=dev)
    with torch.no_grad():
        o1, o2 = run_pair(pair, g1, g2, **kw)
    return {"m1_bpc": bits_per_char(o1, g1), "m2_bpc": bits_per_char(o2, g2)}


def ensemble_baseline(pair: FusedPair, docs, splits, tok1, tok2, L: int) -> dict:
    """No-communication baseline: average the two SOLO models' next-character
    distributions at positions where both tokenizers have a boundary.

    Reports bits/char for model 1 alone, model 2 alone and the 50/50 mixture on ordinary
    text, and next-char accuracy of each and of the mixture on split-recall answers.
    The mixture combines OUTPUTS; the fusion claim is about information entering a
    model's own computation, so this is a reference point, not a competitor on recall."""
    dev = device_of(pair)
    chars = CharTokenizer().itos[1:]
    col = {c: j for j, c in enumerate(chars)}
    M1, M2 = first_char_matrix(tok1, chars), first_char_matrix(tok2, chars)
    off = frozenset(int(k) for k in pair.links)

    def solo_preds(samples_1, samples_2):
        b1 = make_batch(samples_1, tok1, L, device=dev)
        b2 = make_batch(samples_2, tok2, L, device=dev)
        with torch.no_grad():
            o1, o2 = run_pair(pair, b1, b2, disabled=off)
        return char_predictions(o1, b1, M1), char_predictions(o2, b2, M2)

    p1, p2 = solo_preds(docs, docs)
    texts = [d.text for d in docs]
    keys = [k for k in p1 if k in p2 and k[1] < len(texts[k[0]]) and texts[k[0]][k[1]] in col]
    y = torch.tensor([col[texts[i][e]] for i, e in keys], device=dev)
    P1 = torch.stack([p1[k] for k in keys])
    P2 = torch.stack([p2[k] for k in keys])
    bits = lambda P: float(-(P.gather(1, y[:, None]).clamp_min(1e-12)).log2().mean())  # noqa: E731
    res = {"general_shared_positions": len(keys), "general_bits": {
        "m1_solo": bits(P1), "m2_solo": bits(P2), "mixture": bits(0.5 * (P1 + P2))}}

    q1, q2 = solo_preds([s.view("a") for s in splits], [s.view("b") for s in splits])
    need = [(i, c) for i, s in enumerate(splits) for c in s.partner_answers("a")]
    # model-1's answer char c is predicted at chars_read == c
    keys = [(i, c) for i, c in need if (i, c) in q1 and (i, c) in q2]
    y = torch.tensor([col[splits[i].text[c]] for i, c in keys], device=dev)
    Q1 = torch.stack([q1[k] for k in keys])
    Q2 = torch.stack([q2[k] for k in keys])
    acc = lambda Q: float((Q.argmax(-1) == y).float().mean())  # noqa: E731
    res["split_m1_needs_partner_acc"] = {"m1_solo": acc(Q1), "m2_solo": acc(Q2), "mixture": acc(0.5 * (Q1 + Q2)),
                                        "n": len(keys)}
    return res


def hide_ablation(pair: FusedPair, splits, tok1, tok2, L: int) -> dict:
    """Hide one partner expert at a time (expert-token signal only)."""
    n1, n2 = pair.A.cfg.n_experts, pair.B.cfg.n_experts
    out = {"hide_m2_expert_from_m1": {}, "hide_m1_expert_from_m2": {}}
    for e in range(n2):
        r = split_scores(pair, splits, tok1, tok2, L, hide_experts=(frozenset(), frozenset({e})))
        out["hide_m2_expert_from_m1"][e] = r["m1_needs_partner"]["acc"]
    for e in range(n1):
        r = split_scores(pair, splits, tok1, tok2, L, hide_experts=(frozenset({e}), frozenset()))
        out["hide_m1_expert_from_m2"][e] = r["m2_needs_partner"]["acc"]
    return out
