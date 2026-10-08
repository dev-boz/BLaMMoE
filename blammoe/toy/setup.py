"""Build the corpus sampler and both tokenizers for a ToyPOCConfig (WP-T)."""
from __future__ import annotations

import random
from pathlib import Path

from blammoe.toy.config import ToyModelSpec, ToyPOCConfig
from blammoe.toy.corpus import MixedSampler, load_code, load_prose, recall_sample
from blammoe.toy.moe import ToyMoEConfig, ToyMoELM
from blammoe.toy.tokenizers import BPETokenizer, CharTokenizer

REPO = Path(__file__).resolve().parents[2]


def sampler(cfg: ToyPOCConfig, seed: int) -> MixedSampler:
    return MixedSampler(load_prose(REPO / "papers"), load_code(), cfg.seq_chars, cfg.p_recall, seed=seed)


def tokenizer(spec: ToyModelSpec, cfg: ToyPOCConfig):
    if spec.tokenizer == "char":
        return CharTokenizer()
    path = Path(cfg.out_dir) / f"bpe{spec.bpe_vocab}.json"
    if path.exists():
        return BPETokenizer.load(path)
    s = sampler(cfg, seed=12345)
    tok = BPETokenizer.train([s.sample().text for _ in range(4000)], spec.bpe_vocab)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(path)
    return tok


def build_model(spec: ToyModelSpec, vocab_size: int, cfg: ToyPOCConfig) -> ToyMoELM:
    return ToyMoELM(ToyMoEConfig(
        vocab_size=vocab_size, d_model=spec.d_model, n_layers=spec.n_layers, n_heads=spec.n_heads,
        n_experts=spec.n_experts, top_k=spec.top_k, d_expert=spec.d_expert, max_len=cfg.seq_chars,
    ))


def eval_recall_samples(cfg: ToyPOCConfig, n: int = 64, seed: int = 999):
    rng = random.Random(seed)
    return [recall_sample(rng, cfg.seq_chars) for _ in range(n)]
