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

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from blammoe.adapters import RxAdapter, TxAdapter
from blammoe.toy.batching import Batch, answer_positions, answer_runs
from blammoe.toy.moe import ToyMoELM

NEVER = 10**9  # end-char for padding: unreadable as a writer


def token_ends(batch: Batch) -> torch.Tensor:
    """[B, T] chars read by each input position (pads -> NEVER)."""
    ends = torch.full(batch.ids.shape, NEVER, dtype=torch.long)
    for i, off in enumerate(batch.offsets):
        ends[i, : len(off)] = torch.tensor([e for _, e in off])
    return ends.to(batch.ids.device)


def closed_token_ends(batch: Batch) -> torch.Tensor:
    """[B, T] chars to read before each input token is final (pads -> NEVER). Pass these as
    a model's WRITER ends to make the fusion leak-proof across tokenizers."""
    ends = torch.full(batch.ids.shape, NEVER, dtype=torch.long)
    for i, c in enumerate(batch.closed):
        ends[i, : len(c)] = torch.tensor(c)
    return ends.to(batch.ids.device)


def read_mask(reader_ends: torch.Tensor, writer_ends: torch.Tensor) -> torch.Tensor:
    """[B, T_r, 1 + T_w] True = blocked. Column 0 is the always-open null slot."""
    blocked = writer_ends[:, None, :] > reader_ends[:, :, None]
    null = torch.zeros_like(blocked[..., :1])
    return torch.cat([null, blocked], dim=-1)


REL_BUCKETS = 12  # 0: future (masked) | 1-5: distance 0-4 | 6..10: 5-7, 8-15, 16-31, 32-63, 64+ | 11: null


def char_distance_buckets(reader_ends: torch.Tensor, writer_ends: torch.Tensor) -> torch.Tensor:
    """[B, T_r, 1 + T_w] bucket of (chars read by reader - chars read by writer); col 0 = null slot."""
    d = reader_ends[:, :, None] - writer_ends[:, None, :]
    b = torch.zeros_like(d)
    b = torch.where((d >= 0) & (d <= 4), d + 1, b)
    for i, (lo, hi) in enumerate(((5, 7), (8, 15), (16, 31), (32, 63))):
        b = torch.where((d >= lo) & (d <= hi), torch.full_like(d, 6 + i), b)
    b = torch.where(d >= 64, torch.full_like(d, 10), b)
    null = torch.full_like(b[..., :1], REL_BUCKETS - 1)
    return torch.cat([null, b], dim=-1)


def _causal(T: int, device) -> torch.Tensor:
    """True = blocked. Boolean, so it stays causal in fp64/bf16 (a float32 -inf mask does not)."""
    return torch.ones(T, T, dtype=torch.bool, device=device).triu(1)


class Link(nn.Module):
    """Bidirectional exchange at one depth.

    With expert tagging (``n_exp_a``/``n_exp_b`` > 0) the bus carries one entry per
    (position, routed expert): the expert's own output projected to the bus plus a
    learned embedding of *which* expert produced it. The receiver's cross-attention
    then chooses which foreign experts to read — the translator does the routing.
    """

    def __init__(self, d_a: int, d_b: int, d_bus: int, n_heads: int, n_exp_a: int = 0, n_exp_b: int = 0,
                 n_rel_buckets: int = 0):
        super().__init__()
        self.tx_a, self.tx_b = TxAdapter(d_a, d_bus), TxAdapter(d_b, d_bus)
        self.rx_a = RxAdapter(d_a, d_bus, n_heads, n_rel_buckets)
        self.rx_b = RxAdapter(d_b, d_bus, n_heads, n_rel_buckets)
        self.eid_a = nn.Embedding(n_exp_a, d_bus) if n_exp_a else None  # tags A's experts
        self.eid_b = nn.Embedding(n_exp_b, d_bus) if n_exp_b else None

    @staticmethod
    def _bus(tx: TxAdapter, h: torch.Tensor, eid: nn.Embedding | None, idx: torch.Tensor | None,
             tags: str = "keep"):
        z = tx(h)  # [B, T, d_bus] or, for expert tokens, [B, T, k, d_bus]
        if idx is not None:
            if tags == "shift":  # every entry tagged as the wrong expert
                idx = (idx + 1) % eid.num_embeddings
            if tags != "zero":
                z = z + eid(idx)
            z = z.flatten(1, 2)  # position-major: (t0,slot0), (t0,slot1), ...
        return torch.cat([torch.zeros_like(z[:, :1]), z], dim=1)  # prepend null slot

    def forward(self, xa, xb, sa, sb, mask_a, mask_b, ia=None, ib=None, want_weights=False,
                rel_a=None, rel_b=None, tags_a="keep", tags_b="keep"):
        ra = self.rx_a(xa, self._bus(self.tx_b, sb, self.eid_b, ib, tags_b), mask_a, want_weights, rel_a)
        rb = self.rx_b(xb, self._bus(self.tx_a, sa, self.eid_a, ia, tags_a), mask_b, want_weights, rel_b)
        return ra, rb


SIGNALS = ("moe", "resid", "experts")
TAGS = ("keep", "zero", "shift")


@dataclass(frozen=True)
class BusAblation:
    """Attribution controls on one model's expert bus, as its partner reads it (signal
    "experts" only; audit M3). Applied at every link.

    ``hide``: these experts' entries are masked out, so the reader cannot attend to them.
    ``blank``: these experts' entries keep their slot, mask and expert-ID tag, but the expert's
    output is replaced by a constant: ``blank_value[layer][e]`` (e.g. its mean output on a
    selection set, see ``expert_output_means``), zeros if not given. Content-only ablation.
    ``tags``: "zero" drops every entry's expert-ID tag, "shift" tags expert e as e+1 (mod E);
    the outputs are untouched. ID-only ablation.
    """
    hide: frozenset[int] = frozenset()
    blank: frozenset[int] = frozenset()
    tags: str = "keep"
    blank_value: dict[int, torch.Tensor] | None = field(default=None, compare=False)

    def __post_init__(self):
        if self.tags not in TAGS:
            raise ValueError(self.tags)

    def apply(self, layer: int, slots: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor):
        """slots [B, T, k, d], idx [B, T, k], reader mask [B, T_r, 1 + T*k] -> (slots, mask)."""
        if self.blank:
            on = torch.isin(idx, torch.tensor(sorted(self.blank), device=idx.device))[..., None]
            if self.blank_value is None:
                const = torch.zeros_like(slots)
            else:
                const = self.blank_value[layer].to(slots)[idx]
            slots = torch.where(on, const, slots)
        if self.hide:
            gone = torch.isin(idx.flatten(1), torch.tensor(sorted(self.hide), device=idx.device))
            mask = mask | torch.cat([torch.zeros_like(gone[:, :1]), gone], 1)[:, None, :]
        return slots, mask


NO_ABLATION = BusAblation()


class FusedPair(nn.Module):
    """``signal``: "moe" = each layer's router-weighted MoE output (sender's router
    decides, summed); "experts" = every routed expert's own output as a separate,
    expert-tagged bus entry (receiver decides which to read); "resid" = residual."""

    def __init__(self, A: ToyMoELM, B: ToyMoELM, links: list[int], d_bus: int = 64, n_heads: int = 4,
                 signal: str = "moe", rel_pos: bool = False, safe_mask: bool = False):
        super().__init__()
        if A.cfg.n_layers != B.cfg.n_layers:
            raise ValueError("lockstep fusion needs equal depth")
        if signal not in SIGNALS:
            raise ValueError(signal)
        self.A, self.B, self.signal, self.rel_pos = A, B, signal, rel_pos
        self.safe_mask = safe_mask  # callers use run_pair, which then passes closed-chunk writer ends
        for p in list(A.parameters()) + list(B.parameters()):
            p.requires_grad_(False)
        ea, eb = (A.cfg.n_experts, B.cfg.n_experts) if signal == "experts" else (0, 0)
        self.links = nn.ModuleDict(
            {str(l): Link(A.cfg.d_model, B.cfg.d_model, d_bus, n_heads, ea, eb, REL_BUCKETS if rel_pos else 0)
             for l in links}
        )

    def bridge_parameters(self):
        return self.links.parameters()

    def forward(
        self, ids_a: torch.Tensor, ids_b: torch.Tensor, ends_a: torch.Tensor, ends_b: torch.Tensor,
        *, disabled: frozenset[int] = frozenset(), partner_roll: int = 0, attn: dict | None = None,
        write_ends_a: torch.Tensor | None = None, write_ends_b: torch.Tensor | None = None,
        ablate: tuple[BusAblation, BusAblation] = (NO_ABLATION, NO_ABLATION),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``disabled``: link depths switched off. ``partner_roll``: each model hears the
        partner of a *different* document in the batch (shuffled-partner control).
        ``attn``: if a dict is given, filled with {layer: (A-reads-B, B-reads-A)} attention
        [B, T_reader, 1 + S] (column 0 = null slot), the read masks used (``*_blocked``, True =
        blocked, after any ``ablate``), plus the expert ids and outputs on the bus (true ids and
        unablated outputs, before any ``ablate``).
        ``write_ends_*``: when each position may be READ by the partner (default: its own
        ends). Pass ``closed_token_ends`` to stop a partner reading a BPE token before its
        boundaries are final - otherwise those boundaries leak upcoming chars.
        ``ablate``: (A's, B's) ``BusAblation`` - what the partner may read of each model's
        expert bus (expert-token signal only): the controls for "does the chosen expert carry it?"."""
        if self.safe_mask and (write_ends_a is None or write_ends_b is None):
            raise ValueError("safe_mask pair needs closed-chunk writer ends: call run_pair, or pass write_ends_*")
        A, B = self.A, self.B
        xa = A.tok(ids_a) + A.pos(torch.arange(ids_a.shape[1], device=ids_a.device))
        xb = B.tok(ids_b) + B.pos(torch.arange(ids_b.shape[1], device=ids_b.device))
        ca, cb = _causal(ids_a.shape[1], ids_a.device), _causal(ids_b.shape[1], ids_b.device)
        wa = (ends_a if write_ends_a is None else write_ends_a).roll(partner_roll, 0)  # partner as writer
        wb = (ends_b if write_ends_b is None else write_ends_b).roll(partner_roll, 0)
        experts = self.signal == "experts"
        if not experts and ablate != (NO_ABLATION, NO_ABLATION):
            raise ValueError("bus ablations need signal='experts'")
        if experts:
            wa, wb = wa.repeat_interleave(A.cfg.top_k, 1), wb.repeat_interleave(B.cfg.top_k, 1)
        mask_a, mask_b = read_mask(ends_a, wb), read_mask(ends_b, wa)
        rel_a = rel_b = None
        if self.rel_pos:
            rel_a, rel_b = char_distance_buckets(ends_a, wb), char_distance_buckets(ends_b, wa)
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
            la_mask, lb_mask = mask_a, mask_b
            if experts:
                sa, sb = blk_a.moe.last_slots.roll(partner_roll, 0), blk_b.moe.last_slots.roll(partner_roll, 0)
                ia, ib = blk_a.moe.last_idx.roll(partner_roll, 0), blk_b.moe.last_idx.roll(partner_roll, 0)
                raw = (sa, sb)
                abl_a, abl_b = ablate
                sa, lb_mask = abl_a.apply(l, sa, ia, mask_b)  # what B may read of A
                sb, la_mask = abl_b.apply(l, sb, ib, mask_a)  # what A may read of B
            elif self.signal == "moe":
                sa, sb = cap["a"].roll(partner_roll, 0), cap["b"].roll(partner_roll, 0)
            else:
                sa, sb = xa.roll(partner_roll, 0), xb.roll(partner_roll, 0)
            ra, rb = self.links[str(l)](
                xa, xb, sa, sb, la_mask, lb_mask, ia, ib,
                want_weights=attn is not None, rel_a=rel_a, rel_b=rel_b,
                tags_a=ablate[0].tags, tags_b=ablate[1].tags,
            )
            if attn is not None:
                (xa, w_ab), (xb, w_ba) = ra, rb
                attn[l] = {"a_reads_b": w_ab, "b_reads_a": w_ba, "b_expert_ids": ib, "a_expert_ids": ia,
                           "a_reads_b_blocked": la_mask, "b_reads_a_blocked": lb_mask}
                if experts:
                    attn[l]["a_expert_out"], attn[l]["b_expert_out"] = raw
            else:
                xa, xb = ra, rb
        return A.head(A.ln_f(xa)), B.head(B.ln_f(xb))


def run_pair(pair: "FusedPair", batch_a: Batch, batch_b: Batch, **kw):
    """Run a fused pair on two batches of the same texts, with the right read/write ends:
    if the pair was built with ``safe_mask``, partners may only read tokens whose
    boundaries are final (leak-proof across tokenizers)."""
    if pair.safe_mask:
        kw = {"write_ends_a": closed_token_ends(batch_a), "write_ends_b": closed_token_ends(batch_b), **kw}
    return pair(batch_a.ids, batch_b.ids, token_ends(batch_a), token_ends(batch_b), **kw)


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
    """Teacher-forced top-1 accuracy on the answer tokens whose char is in each sample's given set:
    per digit (``acc`` over ``n`` digits) and per whole answer (``exact``: all its digits right,
    over ``n_answers``). An answer only counts for ``exact`` if every one of its digits is scored."""
    rows, cols, groups = [], [], []
    for i, (full, chars) in enumerate(zip(batch.full_offsets, chars_per_sample)):
        for run in answer_runs(chars):
            pos = answer_positions(full, run)
            if len(pos) == len(run):
                groups.append(range(len(rows), len(rows) + len(pos)))
            rows += [i] * len(pos)
            cols += pos
    if not rows:
        return {"acc": float("nan"), "n": 0, "exact": float("nan"), "n_answers": 0}
    r, c = torch.tensor(rows), torch.tensor(cols)
    ok = (logits[r, c].argmax(-1) == batch.targets[r, c]).tolist()
    exact = [all(ok[j] for j in g) for g in groups]
    return {"acc": sum(ok) / len(ok), "n": len(rows),
            "exact": sum(exact) / len(exact) if exact else float("nan"), "n_answers": len(exact)}


@torch.no_grad()
def bits_per_char(logits: torch.Tensor, batch: Batch) -> float:
    """Total next-token NLL in bits / chars covered by the predicted (target) tokens.

    Comparable across tokenizers: targets are tokens 1..n, which tile the text
    from the end of token 0 to the end of token n.
    """
    nll = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100, reduction="sum")
    chars = sum(full[-1][1] - full[0][1] for full in batch.full_offsets)
    return float(nll) / chars / math.log(2)


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
    p = logits.softmax(-1) @ M.to(logits.device)
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
    y = torch.tensor([col[texts[i][e]] for i, e in keys], device=S.device)
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


@torch.no_grad()
def expert_attention_mass(weights: torch.Tensor, expert_ids: torch.Tensor, n_experts: int,
                          reader_mask: torch.Tensor, blocked: torch.Tensor) -> dict:
    """Which partner experts did the reader choose to read?

    weights: [B, T_r, 1 + T_w*k] attention (col 0 = null slot); expert_ids: [B, T_w, k];
    reader_mask: [B, T_r] bool, the reader positions to aggregate over; blocked: [B, T_r, 1 + T_w*k]
    the read mask the weights were computed under (True = blocked; ``attn[l]["*_blocked"]``).
    Returns the null-slot share, each expert's share of attention mass, its baseline share
    and their ratio (>1 = sought out). The baseline is the share each expert would get if every
    reader spread its bus mass uniformly over the entries it may legally read (not future, not
    padding, not hidden), so uniform attention gives a ratio of exactly 1.
    """
    w = weights[reader_mask]  # [N, 1 + S]
    legal = (~blocked[reader_mask][:, 1:]).float()  # [N, S]
    ids = expert_ids.flatten(1)  # [B, S]
    ids_r = ids[:, None, :].expand(-1, reader_mask.shape[1], -1)[reader_mask]  # [N, S]
    onehot = torch.nn.functional.one_hot(ids_r, n_experts).float()  # [N, S, E]
    mass = torch.einsum("ns,nse->e", w[:, 1:], onehot)
    total = w.sum()
    share = mass / mass.sum().clamp_min(1e-12)
    uniform = legal / legal.sum(1, keepdim=True).clamp_min(1)  # [N, S]
    base = torch.einsum("n,ns,nse->e", w[:, 1:].sum(1), uniform, onehot)
    base = base / base.sum().clamp_min(1e-12)
    return {"null_share": float(w[:, 0].sum() / total), "share": share.tolist(),
            "baseline": base.tolist(), "ratio": (share / base.clamp_min(1e-12)).tolist()}


@torch.no_grad()
def expert_output_means(pair: FusedPair, batch_a: Batch, batch_b: Batch) -> tuple[dict, dict]:
    """(A's, B's) {link: [E, d_model]}: each expert's mean routed output over the real (non-pad)
    positions of a batch, run fused with nothing ablated. Experts never routed there get zeros.
    Use as ``BusAblation.blank_value``; compute it on a selection set, not the reported one,
    so the constant carries nothing about the documents being scored."""
    attn: dict = {}
    run_pair(pair, batch_a, batch_b, attn=attn)
    means: tuple[dict, dict] = ({}, {})
    for side, (batch, n) in enumerate(((batch_a, pair.A.cfg.n_experts), (batch_b, pair.B.cfg.n_experts))):
        real = token_ends(batch) != NEVER
        s = "ab"[side]
        for l, a in attn.items():
            out, ids = a[f"{s}_expert_out"][real], a[f"{s}_expert_ids"][real]  # [N, k, d], [N, k]
            onehot = F.one_hot(ids, n).to(out)  # [N, k, E]
            total = torch.einsum("nkd,nke->ed", out, onehot)
            means[side][l] = total / onehot.sum((0, 1)).clamp_min(1)[:, None]
    return means
