"""Fused pair: two frozen MoE LMs run in lockstep and talk at every chosen depth (WP-T2).

The "slap two models together" experiment. A and B advance one block at a time.
After each block listed in ``links``, each model's residual stream cross-attends
over the other model's signal at that depth (Tx -> bus -> tanh-gated Rx, WP6),
in both directions. Gates start at 0, so an untrained FusedPair is exactly the
two models run separately.

What is exchanged (``signal``):

* ``"moe"`` — each layer's MoE output, the router-weighted sum of the experts
  that actually fired, pre-residual. The *sender's* router decides.
* ``"experts"`` — every routed expert's own output as a separate bus entry tagged
  with a learned expert-identity embedding. The *receiver's* cross-attention
  decides which foreign experts to use: the translator does the routing.
* ``"resid"`` — the whole residual stream (whole-model baseline).

Causality across tokenizers: a reader position that has read ``text[:e_r]`` may
read a writer position that has read ``text[:e_w]`` only if ``e_w <= e_r``. Every
reader can also attend to a zero "null" slot, so it is never forced to read
anything (and no softmax row is ever empty).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from blammoe.adapters import RxAdapter, TxAdapter
from blammoe.toy.batching import Batch, answer_positions
from blammoe.toy.moe import ToyMoELM

NEVER = 10**9  # end-char for padding: unreadable as a writer


def token_ends(batch: Batch) -> torch.Tensor:
    """[B, T] chars read by each input position (pads -> NEVER)."""
    ends = torch.full_like(batch.ids, NEVER)
    for i, off in enumerate(batch.offsets):
        ends[i, : len(off)] = torch.tensor([e for _, e in off])
    return ends


def read_mask(reader_ends: torch.Tensor, writer_ends: torch.Tensor) -> torch.Tensor:
    """[B, T_r, 1 + T_w] True = blocked. Column 0 is the always-open null slot."""
    blocked = writer_ends[:, None, :] > reader_ends[:, :, None]
    null = torch.zeros_like(blocked[..., :1])
    return torch.cat([null, blocked], dim=-1)


def _causal(T: int, device) -> torch.Tensor:
    return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


class Link(nn.Module):
    """Bidirectional exchange at one depth.

    With expert tagging (``n_exp_a``/``n_exp_b`` > 0) the bus carries one entry per
    (position, routed expert): the expert's own output projected to the bus plus a
    learned embedding of *which* expert produced it. The receiver's cross-attention
    then chooses which foreign experts to read — the translator does the routing.
    """

    def __init__(self, d_a: int, d_b: int, d_bus: int, n_heads: int, n_exp_a: int = 0, n_exp_b: int = 0):
        super().__init__()
        self.tx_a, self.tx_b = TxAdapter(d_a, d_bus), TxAdapter(d_b, d_bus)
        self.rx_a, self.rx_b = RxAdapter(d_a, d_bus, n_heads), RxAdapter(d_b, d_bus, n_heads)
        self.eid_a = nn.Embedding(n_exp_a, d_bus) if n_exp_a else None  # tags A's experts
        self.eid_b = nn.Embedding(n_exp_b, d_bus) if n_exp_b else None

    @staticmethod
    def _bus(tx: TxAdapter, h: torch.Tensor, eid: nn.Embedding | None, idx: torch.Tensor | None):
        z = tx(h)  # [B, T, d_bus] or, for expert tokens, [B, T, k, d_bus]
        if idx is not None:
            z = (z + eid(idx)).flatten(1, 2)  # position-major: (t0,slot0), (t0,slot1), ...
        return torch.cat([torch.zeros_like(z[:, :1]), z], dim=1)  # prepend null slot

    def forward(self, xa, xb, sa, sb, mask_a, mask_b, ia=None, ib=None, want_weights=False):
        ra = self.rx_a(xa, self._bus(self.tx_b, sb, self.eid_b, ib), mask_a, return_weights=want_weights)
        rb = self.rx_b(xb, self._bus(self.tx_a, sa, self.eid_a, ia), mask_b, return_weights=want_weights)
        return ra, rb


SIGNALS = ("moe", "resid", "experts")


class FusedPair(nn.Module):
    """``signal``: "moe" = each layer's router-weighted MoE output (sender's router
    decides, summed); "experts" = every routed expert's own output as a separate,
    expert-tagged bus entry (receiver decides which to read); "resid" = residual."""

    def __init__(self, A: ToyMoELM, B: ToyMoELM, links: list[int], d_bus: int = 64, n_heads: int = 4,
                 signal: str = "moe"):
        super().__init__()
        if A.cfg.n_layers != B.cfg.n_layers:
            raise ValueError("lockstep fusion needs equal depth")
        if signal not in SIGNALS:
            raise ValueError(signal)
        self.A, self.B, self.signal = A, B, signal
        for p in list(A.parameters()) + list(B.parameters()):
            p.requires_grad_(False)
        ea, eb = (A.cfg.n_experts, B.cfg.n_experts) if signal == "experts" else (0, 0)
        self.links = nn.ModuleDict(
            {str(l): Link(A.cfg.d_model, B.cfg.d_model, d_bus, n_heads, ea, eb) for l in links}
        )

    def bridge_parameters(self):
        return self.links.parameters()

    def forward(
        self, ids_a: torch.Tensor, ids_b: torch.Tensor, ends_a: torch.Tensor, ends_b: torch.Tensor,
        *, disabled: frozenset[int] = frozenset(), partner_roll: int = 0, attn: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``disabled``: link depths switched off. ``partner_roll``: each model hears the
        partner of a *different* document in the batch (shuffled-partner control).
        ``attn``: if a dict is given, filled with {layer: (A-reads-B, B-reads-A)} attention
        [B, T_reader, 1 + S] (column 0 = null slot) plus the expert ids on the bus."""
        A, B = self.A, self.B
        xa = A.tok(ids_a) + A.pos(torch.arange(ids_a.shape[1], device=ids_a.device))
        xb = B.tok(ids_b) + B.pos(torch.arange(ids_b.shape[1], device=ids_b.device))
        ca, cb = _causal(ids_a.shape[1], ids_a.device), _causal(ids_b.shape[1], ids_b.device)
        wa, wb = ends_a.roll(partner_roll, 0), ends_b.roll(partner_roll, 0)  # partner ends as writers
        experts = self.signal == "experts"
        if experts:
            wa, wb = wa.repeat_interleave(A.cfg.top_k, 1), wb.repeat_interleave(B.cfg.top_k, 1)
        mask_a, mask_b = read_mask(ends_a, wb), read_mask(ends_b, wa)
        for l, (blk_a, blk_b) in enumerate(zip(A.layers, B.layers)):
            active = str(l) in self.links and l not in disabled
            cap: dict[str, torch.Tensor] = {}
            ha = blk_a.moe.register_forward_hook(lambda m, i, o: cap.__setitem__("a", o))
            hb = blk_b.moe.register_forward_hook(lambda m, i, o: cap.__setitem__("b", o))
            blk_a.moe.record = blk_b.moe.record = active and experts
            try:
                xa, xb = blk_a(xa, ca), blk_b(xb, cb)
            finally:
                ha.remove()
                hb.remove()
                blk_a.moe.record = blk_b.moe.record = False
            if not active:
                continue
            ia = ib = None
            if experts:
                sa, sb = blk_a.moe.last_slots, blk_b.moe.last_slots
                ia, ib = blk_a.moe.last_idx.roll(partner_roll, 0), blk_b.moe.last_idx.roll(partner_roll, 0)
            elif self.signal == "moe":
                sa, sb = cap["a"], cap["b"]
            else:
                sa, sb = xa, xb
            ra, rb = self.links[str(l)](
                xa, xb, sa.roll(partner_roll, 0), sb.roll(partner_roll, 0), mask_a, mask_b, ia, ib,
                want_weights=attn is not None,
            )
            if attn is not None:
                (xa, w_ab), (xb, w_ba) = ra, rb
                attn[l] = {"a_reads_b": w_ab, "b_reads_a": w_ba, "b_expert_ids": ib, "a_expert_ids": ia}
            else:
                xa, xb = ra, rb
        return A.head(A.ln_f(xa)), B.head(B.ln_f(xb))


# --------------------------------------------------------------------------- loss / eval
def pair_loss(logits: torch.Tensor, batch: Batch, answer_weight: float = 0.0) -> torch.Tensor:
    loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100)
    if answer_weight:
        m = torch.zeros_like(batch.targets, dtype=torch.bool)
        for i, pos in enumerate(batch.answer_pos):
            m[i, pos] = True
        if m.any():
            loss = loss + answer_weight * F.cross_entropy(logits[m], batch.targets[m])
    return loss


@torch.no_grad()
def subset_accuracy(logits: torch.Tensor, batch: Batch, chars_per_sample: list[tuple[int, ...]]) -> dict:
    """Top-1 accuracy on the answer tokens whose char is in each sample's given set."""
    rows, cols = [], []
    for i, (full, chars) in enumerate(zip(batch.full_offsets, chars_per_sample)):
        for p in answer_positions(full, chars):
            rows.append(i)
            cols.append(p)
    if not rows:
        return {"acc": float("nan"), "n": 0}
    r, c = torch.tensor(rows), torch.tensor(cols)
    pred = logits[r, c].argmax(-1)
    return {"acc": float((pred == batch.targets[r, c]).float().mean()), "n": len(rows)}


@torch.no_grad()
def bits_per_char(logits: torch.Tensor, batch: Batch) -> float:
    """Total next-token NLL in bits / chars covered by the predicted (target) tokens.

    Comparable across tokenizers: targets are tokens 1..n, which tile the text
    from the end of token 0 to the end of token n.
    """
    nll = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100, reduction="sum")
    chars = sum(full[-1][1] - full[0][1] for full in batch.full_offsets)
    return float(nll / chars / torch.log(torch.tensor(2.0)))


# --------------------------------------------------------------------------- diversity of thinking
def first_char_matrix(tokenizer, chars: list[str]) -> torch.Tensor:
    """[V, C] maps each token to its first character, so any tokenizer's next-token
    distribution can be read as a next-*character* distribution (pad -> zero row)."""
    col = {c: j for j, c in enumerate(chars)}
    vocab = tokenizer.itos if hasattr(tokenizer, "itos") else [
        tokenizer.tok.id_to_token(i) for i in range(tokenizer.vocab_size)]
    M = torch.zeros(len(vocab), len(chars))
    for i, t in enumerate(vocab):
        if i != tokenizer.pad_id and t and t[0] in col:
            M[i, col[t[0]]] = 1.0
    return M


@torch.no_grad()
def char_predictions(logits: torch.Tensor, batch: Batch, M: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
    """{(sample, chars_read): next-char distribution} at every valid position."""
    p = logits.softmax(-1) @ M
    out = {}
    for i, full in enumerate(batch.full_offsets):
        for t in range(len(full) - 1):
            if batch.targets[i, t] != -100:
                out[(i, full[t][1])] = p[i, t]
    return out


def _js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * ((a + 1e-12) / (b + 1e-12)).log()).sum(-1)  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def diversity(solo: dict, fused: dict, partner_solo: dict, texts: list[str], chars: list[str]) -> dict:
    """How the fused model's next-char guesses differ from its own solo self and its
    partner's solo self, at positions where both tokenizers have a boundary.

    novel    : fused top-1 is neither model's solo top-1
    emergent : fused is right where BOTH solo models were wrong
    js_*     : mean Jensen-Shannon divergence (nats) between distributions
    """
    col = {c: j for j, c in enumerate(chars)}
    keys = [k for k in solo if k in partner_solo and k[1] < len(texts[k[0]]) and texts[k[0]][k[1]] in col]
    S = torch.stack([solo[k] for k in keys])
    F_ = torch.stack([fused[k] for k in keys])
    P = torch.stack([partner_solo[k] for k in keys])
    y = torch.tensor([col[texts[i][e]] for i, e in keys])
    s, f, pp = S.argmax(-1), F_.argmax(-1), P.argmax(-1)
    both_wrong = (s != y) & (pp != y)
    novel = (f != s) & (f != pp)
    ent = lambda d: -(d * (d + 1e-12).log()).sum(-1).mean()  # noqa: E731
    return {
        "n_positions": len(keys),
        "acc_solo": float((s == y).float().mean()),
        "acc_fused": float((f == y).float().mean()),
        "acc_partner_solo": float((pp == y).float().mean()),
        "acc_oracle_either_solo": float(((s == y) | (pp == y)).float().mean()),
        "changed_vs_solo": float((f != s).float().mean()),
        "novel": float(novel.float().mean()),
        "novel_and_correct": float((novel & (f == y)).float().mean()),
        "emergent_of_both_wrong": float(((f == y) & both_wrong).float().sum() / both_wrong.float().sum().clamp_min(1)),
        "js_fused_vs_solo": float(_js(F_, S).mean()),
        "js_fused_vs_partner": float(_js(F_, P).mean()),
        "js_solo_vs_partner": float(_js(S, P).mean()),
        "entropy_solo": float(ent(S)),
        "entropy_fused": float(ent(F_)),
    }
