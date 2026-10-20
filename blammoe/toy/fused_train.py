"""Train and evaluate a FusedPair (WP-T2), device-agnostic and importable.

Shared by scripts/toy_fused.py (one variant, CPU box) and the Colab sweep
(scripts/toy_sweep.py: many seeds/pairs on a GPU). Batches are built on the CPU and
moved to the pair's device.
"""
from __future__ import annotations

import copy
import random
from collections.abc import Callable

import torch

from blammoe.toy.batching import Batch, make_batch
from blammoe.toy.corpus import MixedSampler, Sample, split_recall_sample
from blammoe.toy.fused import (
    NO_ABLATION, BusAblation, FusedPair, bits_per_char, char_predictions, first_char_matrix, pair_loss, run_pair,
    subset_accuracy,
)
from blammoe.toy.moe import lr_at
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
    answer_weight: float, log: Callable[[str], None] = print, log_every: int = 100, warmup: int = 0,
    **run_kw,
) -> list[dict]:
    """``warmup`` > 0: linear warmup then cosine decay (for links that are already trained:
    a fresh Adam's first full-size steps knock them off). ``run_kw`` go to every forward,
    e.g. ``ablate`` to train with an expert hidden."""
    dev = device_of(pair)
    opt = torch.optim.AdamW(pair.bridge_parameters(), lr=lr, weight_decay=0.0)
    history = []
    for step in range(steps):
        if warmup:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps, lr, warmup)
        items = draw(batch_size)
        b1 = make_batch([x for x, _ in items], tok1, L, device=dev)
        b2 = make_batch([y for _, y in items], tok2, L, device=dev)
        l1, l2 = run_pair(pair, b1, b2, **run_kw)
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
    Each entry has digit accuracy (``acc``) and exact whole-answer accuracy (``exact``).
    ``per_doc``: also return per-document (correct, total) digits, and the same for whole
    answers under ``*_exact``, for bootstrapping."""
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
        out["per_doc"] = {}
        for k, (o, b, w) in {"m1_needs_partner": (o1, s1, "a"), "m2_needs_partner": (o2, s2, "b")}.items():
            out["per_doc"][k], out["per_doc"][f"{k}_exact"] = _per_doc(o, b, [s.partner_answers(w) for s in splits])
    return out


def _per_doc(logits: torch.Tensor, batch: Batch, chars) -> tuple[list[list[int]], list[list[int]]]:
    """-> per-document [correct, total] for digits, and for whole answers."""
    digits, answers = [], []
    for i, c in enumerate(chars):
        r = subset_accuracy(logits[i : i + 1], _row(batch, i), [c])
        digits.append([round(r["acc"] * r["n"]) if r["n"] else 0, r["n"]])
        answers.append([round(r["exact"] * r["n_answers"]) if r["n_answers"] else 0, r["n_answers"]])
    return digits, answers


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


def _label(experts) -> str:
    return "+".join(str(e) for e in sorted(experts))


def hide_ablation(
    pair: FusedPair, splits, tok1, tok2, L: int, *, joint_m1: tuple[frozenset[int], ...] = (),
    joint_m2: tuple[frozenset[int], ...] = (), blank_values: tuple[dict, dict] | None = None,
) -> dict:
    """Attribution controls on the expert bus (audit M3), each in ONE direction at a time
    (model 1's answers that need model 2 are scored with model 2's bus ablated, and vice versa):

    hide_*  : one partner expert at a time made unreadable (the original ablation).
    blank_* : one partner expert at a time keeps its tag, slot and mask but sends a constant
              (``blank_values`` = (m1's, m2's) per-link means from ``expert_output_means``;
              zeros if None). Collapse here = the information is in the expert's OUTPUT.
    tags_*  : every partner tag zeroed / shifted to the wrong expert, outputs kept.
              No collapse here = the reader does not need the tag.
    joint_* : the given partner expert sets hidden together (``joint_m1``: sets of model 1's
              experts, hidden from model 2)."""
    n1, n2 = pair.A.cfg.n_experts, pair.B.cfg.n_experts
    bv1, bv2 = blank_values or (None, None)

    def m1_acc(abl_m2: BusAblation) -> float:
        return split_scores(pair, splits, tok1, tok2, L, ablate=(NO_ABLATION, abl_m2))["m1_needs_partner"]["acc"]

    def m2_acc(abl_m1: BusAblation) -> float:
        return split_scores(pair, splits, tok1, tok2, L, ablate=(abl_m1, NO_ABLATION))["m2_needs_partner"]["acc"]

    return {
        "none": {"m1_needs_partner": m1_acc(NO_ABLATION), "m2_needs_partner": m2_acc(NO_ABLATION)},
        "hide_m2_expert_from_m1": {e: m1_acc(BusAblation(hide=frozenset({e}))) for e in range(n2)},
        "hide_m1_expert_from_m2": {e: m2_acc(BusAblation(hide=frozenset({e}))) for e in range(n1)},
        "blank_m2_expert_for_m1": {e: m1_acc(BusAblation(blank=frozenset({e}), blank_value=bv2)) for e in range(n2)},
        "blank_m1_expert_for_m2": {e: m2_acc(BusAblation(blank=frozenset({e}), blank_value=bv1)) for e in range(n1)},
        "tags_m2_for_m1": {t: m1_acc(BusAblation(tags=t)) for t in ("zero", "shift")},
        "tags_m1_for_m2": {t: m2_acc(BusAblation(tags=t)) for t in ("zero", "shift")},
        "joint_hide_m2_from_m1": {_label(s): m1_acc(BusAblation(hide=frozenset(s))) for s in joint_m2},
        "joint_hide_m1_from_m2": {_label(s): m2_acc(BusAblation(hide=frozenset(s))) for s in joint_m1},
    }


def recovery_finetune(
    pair: FusedPair, ablate: tuple[BusAblation, BusAblation], tok1, tok2, draw: Callable, splits, L: int, *,
    steps: int, batch_size: int, lr: float, answer_weight: float, warmup: int = 100,
    log: Callable[[str], None] = print,
) -> tuple[FusedPair, dict]:
    """Fine-tune a COPY of the pair's links with ``ablate`` in force during training and eval.

    The reader never saw the expert missing in training, so a collapse under hiding may only
    mean its attention broke. If accuracy recovers here, the information also reaches it
    through other experts (not exclusive to the hidden one). Run the same with nothing
    ablated as the control for the fine-tuning itself. Returns (fine-tuned copy, scores)."""
    ft = copy.deepcopy(pair)
    before = split_scores(ft, splits, tok1, tok2, L, ablate=ablate)
    history = train_fused(ft, tok1, tok2, draw, L, steps=steps, batch_size=batch_size, lr=lr,
                          answer_weight=answer_weight, log=log, warmup=warmup, ablate=ablate)
    after = split_scores(ft, splits, tok1, tok2, L, ablate=ablate)
    acc = lambda r: {k: r[k]["acc"] for k in ("m1_needs_partner", "m2_needs_partner")}  # noqa: E731
    return ft, {"before": acc(before), "after": acc(after), "train": history}


def chosen_experts(per_link: dict, direction: str) -> dict:
    """The partner experts a translator chose to read, from ``toy_fused_experts.py``'s
    ``split_needs_partner`` section ({link: {direction: {"share": [...]}}}), ``direction``
    "m1_reads_m2" or "m2_reads_m1". Returns ``main`` (largest attention share summed over
    links), ``per_link`` top-2 by share, and joint sets: the union of each link's top-1 and of
    its top-2 (only sets of 2+ experts; single experts are already in the hide sweep)."""
    share = {int(l): r[direction]["share"] for l, r in per_link.items()}
    n = len(next(iter(share.values())))
    top = {l: sorted(range(n), key=lambda e: -s[e])[:2] for l, s in sorted(share.items())}
    total = [sum(s[e] for s in share.values()) for e in range(n)]
    joint = []
    for k in (1, 2):
        u = frozenset(e for t in top.values() for e in t[:k])
        if len(u) > 1 and u not in joint:
            joint.append(u)
    return {"main": max(range(n), key=lambda e: total[e]), "per_link": top, "joint": joint}
