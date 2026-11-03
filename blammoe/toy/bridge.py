"""Expert-level transmission between two toy MoE LMs (WP-T; spec 1.1, 1.4, 1.5).

Model A (informed) reads the clean text; model B (receiver) reads the *blind*
text with stored recall values masked. A signal taken from ONE of A's experts is
carried into B, and we ask whether B can then produce the answer digits it has
no way of knowing on its own. Two carriers:

* **Linear transplant** (zero-shot w.r.t. the task): fit a least-squares map
  (WP4) from A-expert outputs to a B-expert's outputs on *clean* paired
  positions, then at test time swap that B expert's output for the translated
  A output. The map never sees a blind example.
* **Learned bridge** (the "LaM"): Tx (A -> bus) + Rx (tanh-gated cross-attention
  into B's residual stream, WP6), trained on B's LM loss with both models frozen.

Positions are aligned causally across tokenizers: B token [s, e) pairs with A's
char e-1 (A is char-level), so both have read exactly text[:e].
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blammoe.adapters import RxAdapter, TxAdapter
from blammoe.toy.batching import Batch, answer_mask, answer_runs, make_batch
from blammoe.toy.corpus import Sample
from blammoe.toy.moe import ToyMoELM

SIGNALS = ("expert_dense", "expert_routed", "resid")
CONTROLS = ("matched", "shuffled_doc", "blind_a", "random")


# --------------------------------------------------------------------------- alignment
def b_end_chars(batch: Batch) -> torch.Tensor:
    """[B, T] end char of each B input token (pads -> 1 so their mask row is non-empty)."""
    ends = torch.ones(batch.ids.shape, dtype=torch.long)
    for i, off in enumerate(batch.offsets):
        ends[i, : len(off)] = torch.tensor([e for _, e in off])
    return ends.to(batch.ids.device)


def causal_block_mask(ends: torch.Tensor, n_a: int) -> torch.Tensor:
    """[B, T_b, T_a] True where A char position > (B token end - 1): not yet read by B."""
    return torch.arange(n_a, device=ends.device)[None, None, :] > (ends[:, :, None] - 1)


def gather_aligned(h_a: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Pointwise alignment: A state at char e-1 for each B token. [B, T_b, d_a]."""
    idx = (ends - 1).clamp(min=0, max=h_a.shape[1] - 1)
    return torch.gather(h_a, 1, idx[..., None].expand(-1, -1, h_a.shape[-1]))


# --------------------------------------------------------------------------- A-side signal
@dataclass
class Source:
    layer: int
    expert: int
    signal: str = "expert_dense"  # | "expert_routed" | "resid"

    def __call__(self, model_a: ToyMoELM, ids_a: torch.Tensor) -> torch.Tensor:
        cap = model_a.expert_outputs(ids_a, self.layer, self.expert)
        if self.signal == "expert_dense":
            return cap["out"]
        if self.signal == "expert_routed":  # native: zero wherever A's router did not pick it
            return cap["out"] * cap["routed"][..., None]
        if self.signal == "resid":  # whole-model baseline (what CALM/MoT already do)
            return cap["resid"]
        raise ValueError(self.signal)


def control_signal(
    source: Source, model_a: ToyMoELM, a_clean: Batch, a_blind: Batch, control: str, gen: torch.Generator
) -> torch.Tensor:
    if control == "matched":
        return source(model_a, a_clean.ids)
    if control == "shuffled_doc":  # right kind of signal, wrong document
        return source(model_a, a_clean.ids).roll(1, dims=0)
    if control == "blind_a":  # A is blindfolded too: no extra information exists
        return source(model_a, a_blind.ids)
    if control == "random":  # matched per-position norm, no content
        h = source(model_a, a_clean.ids)
        r = torch.randn(h.shape, generator=gen).to(h.device)  # CPU generator: same draws on any device
        return r * h.norm(dim=-1, keepdim=True) / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    raise ValueError(control)


# --------------------------------------------------------------------------- expert selection
@torch.no_grad()
def routing_at_answers(model: ToyMoELM, batch: Batch, layer: int) -> np.ndarray:
    """Fraction of answer-predicting positions on which each expert is in the top-k."""
    moe = model.layers[layer].moe
    cap = {}
    h = moe.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0]))
    try:
        model(batch.ids)
    finally:
        h.remove()
    probs = moe.router(cap["x"]).softmax(-1)
    top = probs.topk(moe.top_k, -1).indices  # [B, T, k]
    m = answer_mask(batch)
    hits = F.one_hot(top[m], probs.shape[-1]).sum(1).float()  # [n_ans, E]
    return hits.mean(0).numpy()


# --------------------------------------------------------------------------- B-side scoring
@torch.no_grad()
def score_answers(logits: torch.Tensor, batch: Batch) -> dict[str, float]:
    """Teacher-forced answer scores: per digit (``acc``, ``nll`` over ``n``) and per whole
    answer (``exact``: all its scored digits right, over ``n_answers``)."""
    m = answer_mask(batch)
    lp = logits.log_softmax(-1)
    ok = lp.argmax(-1) == batch.targets
    exact = []
    for i, (full, pos) in enumerate(zip(batch.full_offsets, batch.answer_pos)):
        at = {full[p + 1][0]: p for p in pos}  # answer char -> position predicting it
        exact += [bool(ok[i, [at[c] for c in run]].all()) for run in answer_runs(at)]
    tgt = batch.targets[m]
    return {
        "acc": float(ok[m].float().mean()),
        "nll": float(-lp[m].gather(1, tgt[:, None]).mean()),
        "n": int(m.sum()),
        "exact": sum(exact) / len(exact) if exact else float("nan"),
        "n_answers": len(exact),
    }


def lm_loss(logits: torch.Tensor, batch: Batch) -> float:
    return float(F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100))


class _Inject:
    """Context manager: register a forward hook, remove it on exit."""

    def __init__(self, module: nn.Module, fn: Callable):
        self.module, self.fn, self.handle = module, fn, None

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self.fn)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        return False


# --------------------------------------------------------------------------- linear transplant
@torch.no_grad()
def paired_expert_states(
    model_a: ToyMoELM, model_b: ToyMoELM, src: Source, layer_b: int, expert_b: int,
    a_batch: Batch, b_batch: Batch,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean paired (h_A, h_B) at causally aligned positions, B pads dropped."""
    h_a = gather_aligned(src(model_a, a_batch.ids), b_end_chars(b_batch))
    h_b = model_b.expert_outputs(b_batch.ids, layer_b, expert_b)["out"]
    keep = b_batch.targets != -100
    return h_a[keep].numpy(), h_b[keep].numpy()


def native_transplant_hook(model_b: ToyMoELM, layer_b: int, expert_b: int, translated: torch.Tensor,
                           alpha: float = 1.0):
    """Swap B expert ``expert_b`` for ``translated`` the way B's MoE uses it: only where B's router
    picked that expert, at its router weight w. moe_out += alpha * w * routed * (translated - e_j(x)).
    alpha=1 is an exact replacement; tokens not routed to e_j are untouched. This is the
    expert-level test (spec 1.1, CLAUDE.md rule 8)."""
    expert = model_b.layers[layer_b].moe.experts[expert_b]

    def fn(module, inputs, output):
        x = inputs[0]
        w, idx = module.router(x).softmax(-1).topk(module.top_k, dim=-1)
        gate = (w * (idx == expert_b)).sum(-1, keepdim=True)  # router weight if routed, else 0
        return output + alpha * gate * (translated - expert(x))

    return _Inject(model_b.layers[layer_b].moe, fn)


def transplant_hook(model_b: ToyMoELM, layer_b: int, expert_b: int, translated: torch.Tensor, alpha: float):
    """CONTROL, not an expert swap: moe_out += alpha * (translated - e_j(x)) at EVERY token,
    ignoring B's routing and router weight (unrouted tokens get -e_j(x) injected). Kept so the
    original "dense additive" numbers stay comparable; use ``native_transplant_hook`` for claims."""
    expert = model_b.layers[layer_b].moe.experts[expert_b]

    def fn(module, inputs, output):
        return output + alpha * (translated - expert(inputs[0]))

    return _Inject(model_b.layers[layer_b].moe, fn)


# --------------------------------------------------------------------------- learned bridge
class Bridge(nn.Module):
    def __init__(self, d_a: int, d_b: int, d_bus: int, n_heads: int):
        super().__init__()
        self.tx = TxAdapter(d_a, d_bus)
        self.rx = RxAdapter(d_b, d_bus, n_heads)

    def hook(self, model_b: ToyMoELM, layer_b: int, h_a: torch.Tensor, block_mask: torch.Tensor):
        bus = self.tx(h_a)

        def fn(module, inputs, output):
            return self.rx(output, bus, block_mask)

        return _Inject(model_b.layers[layer_b], fn)


def encode_pair(samples: list[Sample], tok_a, tok_b, max_len: int) -> tuple[Batch, Batch, Batch, Batch]:
    """(A clean, A blind, B clean, B blind) batches for the same samples."""
    return (make_batch(samples, tok_a, max_len), make_batch(samples, tok_a, max_len, blind=True),
            make_batch(samples, tok_b, max_len), make_batch(samples, tok_b, max_len, blind=True))


def train_bridge(
    bridge: Bridge, model_a: ToyMoELM, model_b: ToyMoELM, src: Source, layer_b: int,
    draw: Callable[[int], list[Sample]], tok_a, tok_b, max_len: int, *,
    steps: int, batch_size: int, lr: float, answer_weight: float = 0.0,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Both LMs frozen; only the bridge learns, from B's LM loss on blind text.

    ``answer_weight`` adds that multiple of the mean loss on recall-answer tokens
    (the divergence positions of spec Test B), which are otherwise ~5% of tokens.
    The plain LM term stays in, so the bridge is still penalised for damaging
    ordinary text.
    """
    for p in list(model_a.parameters()) + list(model_b.parameters()):
        p.requires_grad_(False)
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.0)
    hist = []
    for step in range(steps):
        a_clean, _, _, b_blind = encode_pair(draw(batch_size), tok_a, tok_b, max_len)
        h_a = src(model_a, a_clean.ids)
        mask = causal_block_mask(b_end_chars(b_blind), h_a.shape[1])
        with bridge.hook(model_b, layer_b, h_a, mask):
            logits = model_b(b_blind.ids)
        loss = F.cross_entropy(logits.flatten(0, 1), b_blind.targets.flatten(), ignore_index=-100)
        m = answer_mask(b_blind)
        if answer_weight and m.any():
            loss = loss + answer_weight * F.cross_entropy(logits[m], b_blind.targets[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            rec = {"step": step, "loss": float(loss.detach()), "gate": float(bridge.rx.gate.detach())}
            hist.append(rec)
            log(str(rec))
    return hist


@torch.no_grad()
def eval_bridge(
    bridge: Bridge, model_a: ToyMoELM, model_b: ToyMoELM, src: Source, layer_b: int,
    batches: tuple[Batch, Batch, Batch, Batch], seed: int = 0,
) -> dict[str, dict[str, float]]:
    """B's answer accuracy/NLL on blind text with the bridge fed each control signal."""
    a_clean, a_blind, _, b_blind = batches
    gen = torch.Generator().manual_seed(seed)
    mask = causal_block_mask(b_end_chars(b_blind), a_clean.ids.shape[1])
    out = {}
    for control in CONTROLS:
        h_a = control_signal(src, model_a, a_clean, a_blind, control, gen)
        with bridge.hook(model_b, layer_b, h_a, mask):
            out[control] = score_answers(model_b(b_blind.ids), b_blind)
    return out
