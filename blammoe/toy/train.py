"""Pretrain a tiny MoE LM from scratch on the mixed toy corpus (WP-T)."""
from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from blammoe.toy.batching import answer_mask, make_batch
from blammoe.toy.corpus import MixedSampler
from blammoe.toy.moe import ToyMoELM, lr_at


@torch.no_grad()
def recall_accuracy(model: ToyMoELM, tokenizer, samples, max_len: int, blind: bool = False) -> float:
    """Top-1 accuracy on the ask-section answer digits."""
    b = make_batch(samples, tokenizer, max_len, blind=blind, device=next(model.parameters()).device)
    m = answer_mask(b)
    pred = model(b.ids).argmax(-1)
    return float((pred[m] == b.targets[m]).float().mean())


def train_lm(
    model: ToyMoELM,
    tokenizer,
    sampler: MixedSampler,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    eval_samples=None,
    log: Callable[[str], None] = print,
    log_every: int = 100,
) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    history, t0 = [], time.time()
    for step in range(steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        b = make_batch([sampler.sample() for _ in range(batch_size)], tokenizer, model.cfg.max_len,
                       device=next(model.parameters()).device)
        logits = model(b.ids)
        lm = F.cross_entropy(logits.flatten(0, 1), b.targets.flatten(), ignore_index=-100)
        loss = lm + model.cfg.aux_loss_coef * model.aux_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            model.eval()
            rec = {"step": step, "lm_loss": float(lm.detach()), "aux": float(model.aux_loss().detach()),
                   "sec": round(time.time() - t0, 1)}
            if eval_samples:
                rec["recall_acc"] = recall_accuracy(model, tokenizer, eval_samples, model.cfg.max_len)
            history.append(rec)
            log(str(rec))
    model.eval()
    return history
