"""fused_train: training moves the links, evals return well-formed results (CPU, tiny)."""
import random

import torch

from blammoe.toy.corpus import recall_sample, split_recall_sample
from blammoe.toy.fused import FusedPair
from blammoe.toy.fused_train import (
    ensemble_baseline, general_bpc, hide_ablation, mixed_draw, split_eval_set, split_scores, train_fused,
)
from blammoe.toy.moe import ToyMoEConfig, ToyMoELM
from blammoe.toy.tokenizers import BPETokenizer, CharTokenizer

L = 100
PROSE = "the model reads text and predicts the next character. " * 40
CODE = "def f(x):\n    return x + 1\n" * 60


def tiny(vocab, d, experts, seed):
    torch.manual_seed(seed)
    return ToyMoELM(ToyMoEConfig(vocab_size=vocab, d_model=d, n_layers=2, n_heads=2, n_experts=experts,
                                 top_k=2, d_expert=16, max_len=128)).eval()


def pair_and_toks():
    ta = CharTokenizer()
    tb = BPETokenizer.train([recall_sample(random.Random(i), 200).text for i in range(30)] + [PROSE, CODE],
                            vocab_size=200)
    pair = FusedPair(tiny(ta.vocab_size, 16, 4, 0), tiny(tb.vocab_size, 12, 6, 1), [0, 1], d_bus=8, n_heads=2,
                     signal="experts", rel_pos=True, safe_mask=True)
    return pair, ta, tb


def test_train_fused_opens_gates_and_evals_are_well_formed():
    pair, ta, tb = pair_and_toks()
    hist = train_fused(pair, ta, tb, mixed_draw(PROSE, CODE, L, 0.5, seed=1), L, steps=4, batch_size=2,
                       lr=1e-2, answer_weight=1.0, log=lambda m: None, log_every=2)
    assert hist and all(abs(g) > 0 for lk in hist[-1]["gates"].values() for g in lk)
    splits = split_eval_set(L, n=3, seed=5)
    r = split_scores(pair, splits, ta, tb, L, per_doc=True)
    assert 0 <= r["m1_needs_partner"]["acc"] <= 1
    per = r["per_doc"]["m1_needs_partner"]
    assert len(per) == 3 and sum(t for _, t in per) == r["m1_needs_partner"]["n"]
    assert sum(c for c, _ in per) == round(r["m1_needs_partner"]["acc"] * r["m1_needs_partner"]["n"])
    docs = [split_recall_sample(random.Random(9), L).view("a")]
    bpc = general_bpc(pair, docs, ta, tb, L)
    assert bpc["m1_bpc"] > 0 and bpc["m2_bpc"] > 0


def test_ensemble_baseline_and_hide_ablation_shapes():
    pair, ta, tb = pair_and_toks()
    splits = split_eval_set(L, n=3, seed=6)
    docs = [s.view("a") for s in split_eval_set(L, n=2, seed=7)]
    ens = ensemble_baseline(pair, docs, splits, ta, tb, L)
    g = ens["general_bits"]
    assert ens["general_shared_positions"] > 0 and g["mixture"] > 0
    # a 50/50 mixture can never be worse than the worse of its two parts (log is concave)
    assert g["mixture"] <= max(g["m1_solo"], g["m2_solo"]) + 1e-6
    assert 0 <= ens["split_m1_needs_partner_acc"]["mixture"] <= 1
    hid = hide_ablation(pair, splits, ta, tb, L)
    assert len(hid["hide_m2_expert_from_m1"]) == pair.B.cfg.n_experts
    assert len(hid["hide_m1_expert_from_m2"]) == pair.A.cfg.n_experts
