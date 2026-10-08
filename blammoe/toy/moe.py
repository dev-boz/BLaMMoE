"""Tiny MoE decoder LM for the toy POC (WP-T).

Structure mirrors OLMoE closely enough for the real pipeline to carry over:
``layers.{layer}.moe.experts.{expert}`` is an nn.ModuleList of SwiGLU FFNs, the
router is a linear layer + softmax with top-k selection (no renormalisation,
OLMoE-style), and each expert is called only on the tokens routed to it. So a
forward hook on an expert (``blammoe.hooks.ExpertCapture``) sees exactly
[n_routed, d], pre-router-weight and pre-residual — the spec 1.1 hook point.

``ToyMoELM.expert_outputs`` additionally evaluates one expert *densely* on every
token's MoE input (plus its router probability and a routed mask), which is what
the translator experiments consume.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyMoEConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_experts: int = 8
    top_k: int = 2
    d_expert: int = 256
    max_len: int = 256
    aux_loss_coef: float = 0.01


class Expert(nn.Module):
    def __init__(self, d: int, f: int):
        super().__init__()
        self.w1 = nn.Linear(d, f, bias=False)
        self.w3 = nn.Linear(d, f, bias=False)
        self.w2 = nn.Linear(f, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, cfg: ToyMoEConfig):
        super().__init__()
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(cfg.d_model, cfg.d_expert) for _ in range(cfg.n_experts))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probs = self.router(flat).softmax(dim=-1)  # [N, E]
        weights, idx = probs.topk(self.top_k, dim=-1)  # [N, k]
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (idx == e).nonzero(as_tuple=True)
            if rows.numel():
                out.index_add_(0, rows, weights[rows, slot, None] * expert(flat[rows]))
        # Switch-style load balancing: E * sum_e frac_tokens(e) * mean_prob(e)
        frac = F.one_hot(idx, probs.shape[-1]).sum(1).float().mean(0) / self.top_k
        self.aux_loss = probs.shape[-1] * (frac * probs.mean(0)).sum()
        return out.reshape(shape)


class Block(nn.Module):
    def __init__(self, cfg: ToyMoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        return x + self.moe(self.ln2(x))


class ToyMoELM(nn.Module):
    def __init__(self, cfg: ToyMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        causal = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), diagonal=1)
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.layers:
            x = blk(x, causal)
        return self.head(self.ln_f(x))

    def aux_loss(self) -> torch.Tensor:
        return sum(blk.moe.aux_loss for blk in self.layers) / len(self.layers)

    @torch.no_grad()
    def expert_outputs(self, ids: torch.Tensor, layer: int, expert: int) -> dict[str, torch.Tensor]:
        """Dense output of one expert on every token at ``layer`` (pre-weight, pre-residual).

        Returns ``out`` [B, T, d], router ``prob`` [B, T] for that expert, ``routed``
        [B, T] bool (expert in that token's top-k), and the layer ``resid`` stream
        after the block [B, T, d] (a whole-model baseline signal).
        """
        moe = self.layers[layer].moe
        cap: dict[str, torch.Tensor] = {}
        h1 = moe.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0]))
        h2 = self.layers[layer].register_forward_hook(lambda m, i, o: cap.__setitem__("resid", o))
        try:
            self(ids)
        finally:
            h1.remove()
            h2.remove()
        x = cap["x"]
        probs = moe.router(x).softmax(-1)
        routed = (probs.topk(moe.top_k, dim=-1).indices == expert).any(-1)
        return {"out": moe.experts[expert](x), "prob": probs[..., expert], "routed": routed,
                "resid": cap["resid"]}


def save(model: ToyMoELM, path, **extra) -> None:
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(), **extra}, path)


def load(path, device: torch.device | str = "cpu") -> tuple[ToyMoELM, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ToyMoELM(ToyMoEConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    return model.eval(), ck


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def lr_at(step: int, total: int, peak: float, warmup: int = 100) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return 0.5 * peak * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
