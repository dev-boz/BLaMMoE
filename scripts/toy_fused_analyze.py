"""WP-T2: diversity-of-thinking analysis for a trained fused pair.

Loads <out_dir>/fused_<variant>.pt (bridges) and compares each model's next-char
predictions on ordinary held-out text: solo vs fused vs fused-with-wrong-partner.
Writes <out_dir>/fused_<variant>_diversity.json.

    PYTHONPATH=. python scripts/toy_fused_analyze.py configs/toy_poc.yaml <variant> [threads]
"""
import copy
import json
import sys
from pathlib import Path

import torch

from blammoe.toy import setup
from blammoe.toy.batching import make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import MixedSampler
from blammoe.toy.fused import run_pair, FusedPair, char_predictions, diversity, first_char_matrix, token_ends
from blammoe.toy.moe import load
from blammoe.toy.tokenizers import CharTokenizer


def main(cfg_path: str, variant: str, threads: int = 2) -> None:
    torch.set_num_threads(threads)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    fs, out, L = cfg.fused, Path(cfg.out_dir), cfg.seq_chars
    v = fs.variants[variant]
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt")
    B, _ = load(out / "model_b.pt")
    models = {"a": (A, tok_a), "b": (B, tok_b)}
    (m1, t1), (m2, t2) = models[v["pair"][0]], models[v["pair"][1]]
    if m2 is m1:
        m2 = copy.deepcopy(m1)
    pair = FusedPair(m1, m2, v["links"], fs.d_bus, fs.n_heads, v.get("signal", "moe"), v.get("rel_pos", False), v.get("safe_mask", False))
    pair.links.load_state_dict(torch.load(out / f"fused_{variant}.pt"))
    pair.eval()

    gs = MixedSampler(*setup.texts(cfg, "test"), L, p_recall=0.0, seed=777)
    gen = [gs.sample() for _ in range(48)]
    texts = [g.text for g in gen]
    b1, b2 = make_batch(gen, t1, L), make_batch(gen, t2, L)
    e1, e2 = token_ends(b1), token_ends(b2)
    chars = CharTokenizer().itos[1:]
    M1, M2 = first_char_matrix(t1, chars), first_char_matrix(t2, chars)
    links = frozenset(int(k) for k in pair.links)

    def preds(**kw):
        with torch.no_grad():
            o1, o2 = run_pair(pair, b1, b2, **kw)
        return char_predictions(o1, b1, M1), char_predictions(o2, b2, M2)

    solo1, solo2 = preds(disabled=links)
    fus1, fus2 = preds()
    shf1, shf2 = preds(partner_roll=1)
    res = {
        "variant": variant, "spec": v,
        "m1": {"fused": diversity(solo1, fus1, solo2, texts, chars),
               "wrong_partner": diversity(solo1, shf1, solo2, texts, chars)},
        "m2": {"fused": diversity(solo2, fus2, solo1, texts, chars),
               "wrong_partner": diversity(solo2, shf2, solo1, texts, chars)},
    }
    for m in ("m1", "m2"):
        for cond, d in res[m].items():
            print(f"{variant} {m} {cond}: " + " ".join(
                f"{k}={x:.4f}" if isinstance(x, float) else f"{k}={x}" for k, x in d.items()), flush=True)
    (out / f"fused_{variant}_diversity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
