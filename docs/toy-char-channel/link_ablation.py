"""One-off CPU check for docs/toy-char-channel-results.md: per-link eval-time ablation of the e units
(A prose, A' code) plus strong-domain costs. Run from the repo root with PYTHONPATH=. (~15 min).
Args: [results root] [variant ...]; defaults are the channel run's."""
import sys
from pathlib import Path
import torch
from blammoe.toy import setup
from blammoe.toy.batching import make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.fused import FusedPair, bits_per_doc, run_pair
from blammoe.toy.fused_train import domain_eval_sets
from blammoe.toy.moe import load
torch.set_num_threads(8)
root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/toy_char_channel")
for var in sys.argv[2:] or ["ab_moe_all_rel_safe_text", "ab_resid_012_rel_safe_text"]:
    for u in ["e0", "e1", "e2"]:
        unit = root / u
        cfg = ToyPOCConfig.from_yaml(unit / "config.yaml"); fs, L = cfg.fused, cfg.seq_chars; v = fs.variants[var]
        tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
        A, _ = load(unit / "model_a.pt"); B, _ = load(unit / "model_b.pt")
        pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, v.get("signal", "moe"), v.get("rel_pos", False), v.get("safe_mask", False))
        pair.links.load_state_dict(torch.load(unit / f"fused_{var}.pt", map_location="cpu")); pair.eval()
        links = sorted(int(k) for k in pair.links)
        conds = {"solo": frozenset(links), "all": frozenset()}
        conds.update({f"only{l}": frozenset(links) - {l} for l in links})
        conds.update({f"no{l}": frozenset({l}) for l in links})
        out = {}
        with torch.no_grad():
            for d, docs in domain_eval_sets(*setup.texts(cfg, "test"), L).items():
                b = make_batch(docs, tok_a, L), make_batch(docs, tok_b, L)
                for c, dis in conds.items():
                    o = run_pair(pair, *b, disabled=dis)
                    for m, name in ((0, "A"), (1, "A'")):
                        x = bits_per_doc(o[m], b[m]); out[(name, d, c)] = sum(r[0] for r in x) / sum(r[1] for r in x)
        for name, d in (("A", "prose"), ("A'", "code"), ("A", "code"), ("A'", "prose")):
            s = out[(name, d, "solo")]
            print(var, u, f"{name:2s} {d:5s} solo {s:.3f} | " + " ".join(f"{c} {out[(name, d, c)] - s:+.3f}" for c in conds if c != "solo"), flush=True)
