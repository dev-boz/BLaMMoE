"""Expert ablation for a translator-routed fusion: hide one partner expert at a time.

For every expert of B: hide its bus entries from A (all links) and measure A's
split-recall accuracy on answers that need B; and vice versa. If the experts the
translator chose are the ones it cannot do without, while others of similar traffic
are dispensable, the signal really is carried at the expert level.

    PYTHONPATH=. python scripts/toy_fused_hide.py configs/toy_poc.yaml <variant> [threads]
"""
import json
import random
import sys
from pathlib import Path

import torch

from blammoe.toy import setup
from blammoe.toy.batching import make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import split_recall_sample
from blammoe.toy.fused import FusedPair, run_pair, subset_accuracy
from blammoe.toy.moe import load


def main(cfg_path: str, variant: str, threads: int = 4) -> None:
    torch.set_num_threads(threads)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    fs, out, L = cfg.fused, Path(cfg.out_dir), cfg.seq_chars
    v = fs.variants[variant]
    assert v["pair"] == "ab" and v["signal"] == "experts"
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt")
    B, _ = load(out / "model_b.pt")
    pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, "experts", v.get("rel_pos", False),
                     v.get("safe_mask", False))
    pair.links.load_state_dict(torch.load(out / f"fused_{variant}.pt"))
    pair.eval()
    rng = random.Random(2025)
    splits = [split_recall_sample(rng, L) for _ in range(96)]
    s1 = make_batch([s.view("a") for s in splits], tok_a, L)
    s2 = make_batch([s.view("b") for s in splits], tok_b, L)
    need_a = [s.partner_answers("a") for s in splits]
    need_b = [s.partner_answers("b") for s in splits]

    def score(hide=(frozenset(), frozenset())):
        with torch.no_grad():
            o1, o2 = run_pair(pair, s1, s2, hide_experts=hide)
        return subset_accuracy(o1, s1, need_a)["acc"], subset_accuracy(o2, s2, need_b)["acc"]

    base_a, base_b = score()
    res = {"variant": variant, "none_hidden": {"A_needs_B": base_a, "B_needs_A": base_b},
           "hide_B_expert_from_A": {}, "hide_A_expert_from_B": {}}
    print(f"nothing hidden: A needs B {base_a:.3f} | B needs A {base_b:.3f}", flush=True)
    for e in range(B.cfg.n_experts):
        a, _ = score((frozenset(), frozenset({e})))
        res["hide_B_expert_from_A"][e] = a
        print(f"hide B e{e:<2} from A: A needs B {a:.3f}", flush=True)
    for e in range(A.cfg.n_experts):
        _, b = score((frozenset({e}), frozenset()))
        res["hide_A_expert_from_B"][e] = b
        print(f"hide A e{e} from B: B needs A {b:.3f}", flush=True)
    (out / f"fused_{variant}_hide.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
