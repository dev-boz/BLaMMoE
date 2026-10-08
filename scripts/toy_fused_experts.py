"""WP-T2: which partner experts did each translator choose to read?

For a trained expert-token fused pair, report per link and direction the
attention share on each partner expert vs its share of the bus (ratio > 1 =
sought out), at (a) split-recall answer positions that need the partner and
(b) all positions of ordinary text. Writes <out_dir>/fused_<variant>_experts.json.

    PYTHONPATH=. python scripts/toy_fused_experts.py configs/toy_poc.yaml <variant> [threads]
"""
import copy
import json
import random
import sys
from pathlib import Path

import torch

from blammoe.toy import setup
from blammoe.toy.batching import answer_positions, make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import MixedSampler, split_recall_sample
from blammoe.toy.fused import FusedPair, expert_attention_mass, token_ends
from blammoe.toy.moe import load


def answer_mask_for(batch, chars_per_sample):
    m = torch.zeros_like(batch.ids, dtype=torch.bool)
    for i, (full, chars) in enumerate(zip(batch.full_offsets, chars_per_sample)):
        m[i, answer_positions(full, chars)] = True
    return m


def main(cfg_path: str, variant: str, threads: int = 2) -> None:
    torch.set_num_threads(threads)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    fs, out, L = cfg.fused, Path(cfg.out_dir), cfg.seq_chars
    v = fs.variants[variant]
    assert v.get("signal") == "experts", "only meaningful for signal: experts"
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt")
    B, _ = load(out / "model_b.pt")
    models = {"a": (A, tok_a), "b": (B, tok_b)}
    (m1, t1), (m2, t2) = models[v["pair"][0]], models[v["pair"][1]]
    if m2 is m1:
        m2 = copy.deepcopy(m1)
    pair = FusedPair(m1, m2, v["links"], fs.d_bus, fs.n_heads, "experts", v.get("rel_pos", False))
    pair.links.load_state_dict(torch.load(out / f"fused_{variant}.pt"))
    pair.eval()

    rng = random.Random(2025)
    splits = [split_recall_sample(rng, L) for _ in range(48)]
    s1 = make_batch([s.view("a") for s in splits], t1, L)
    s2 = make_batch([s.view("b") for s in splits], t2, L)
    base = setup.sampler(cfg, seed=0)
    gs = MixedSampler(base.prose, base.code, L, p_recall=0.0, seed=777)
    gen = [gs.sample() for _ in range(32)]
    g1, g2 = make_batch(gen, t1, L), make_batch(gen, t2, L)

    res = {"variant": variant}
    for name, (b1, b2, r1, r2) in {
        "split_needs_partner": (s1, s2, answer_mask_for(s1, [s.partner_answers("a") for s in splits]),
                                answer_mask_for(s2, [s.partner_answers("b") for s in splits])),
        "general_all": (g1, g2, g1.targets != -100, g2.targets != -100),
    }.items():
        attn = {}
        with torch.no_grad():
            pair(b1.ids, b2.ids, token_ends(b1), token_ends(b2), attn=attn)
        res[name] = {}
        for l, a in attn.items():
            res[name][l] = {
                "m1_reads_m2": expert_attention_mass(a["a_reads_b"], a["b_expert_ids"], m2.cfg.n_experts, r1),
                "m2_reads_m1": expert_attention_mass(a["b_reads_a"], a["a_expert_ids"], m1.cfg.n_experts, r2),
            }
            for d, r in res[name][l].items():
                top = sorted(range(len(r["ratio"])), key=lambda e: -r["share"][e])[:3]
                print(f"{name} link {l} {d}: null {r['null_share']:.2f} | top experts by share: "
                      + ", ".join(f"e{e} {r['share'][e]:.2f} (x{r['ratio'][e]:.1f})" for e in top), flush=True)
    (out / f"fused_{variant}_experts.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
