"""One-off CPU check for docs/toy-char-gate-results.md: which of A''s experts does A read at the last link
once its gate starts open, and which ones carry A's prose gain?

Per e unit, on the prose test windows: (1) A''s routing share per expert on prose vs code (its prose
specialists), (2) A's attention share per A' expert at link 3 against the uniform-legal baseline
(expert_attention_mass), (3) A's prose bits with each A' expert hidden from A (every link), and with its
content blanked to its mean output on the code test windows (a different set from the scored one).
Args: [results root] [variant]. Run from the repo root with PYTHONPATH=. (~10 min)."""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from blammoe.toy import setup
from blammoe.toy.batching import make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.fused import (NEVER, NO_ABLATION, BusAblation, FusedPair, bits_per_doc, expert_attention_mass,
                               expert_output_means, run_pair, token_ends)
from blammoe.toy.fused_train import domain_eval_sets
from blammoe.toy.moe import load

torch.set_num_threads(8)
root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/toy_char_gate")
var = sys.argv[2] if len(sys.argv) > 2 else "ab_experts_all_rel_safe_text_ga3"
pooled = lambda rows: sum(r[0] for r in rows) / sum(r[1] for r in rows)  # noqa: E731
res = {}
for unit in sorted(p for p in root.iterdir() if p.name.startswith("e") and (p / f"fused_{var}.pt").exists()):
    cfg = ToyPOCConfig.from_yaml(unit / "config.yaml"); fs, L = cfg.fused, cfg.seq_chars; v = fs.variants[var]
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(unit / "model_a.pt"); B, _ = load(unit / "model_b.pt")
    pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, "experts", v.get("rel_pos", False), v.get("safe_mask", False))
    pair.links.load_state_dict(torch.load(unit / f"fused_{var}.pt", map_location="cpu")); pair.eval()
    last, E = max(v["links"]), B.cfg.n_experts
    dom = {d: (make_batch(docs, tok_a, L), make_batch(docs, tok_b, L))
           for d, docs in domain_eval_sets(*setup.texts(cfg, "test"), L).items()}
    out = {}
    with torch.no_grad():
        route = {}
        for d, (ba, bb) in dom.items():
            attn = {}
            run_pair(pair, ba, bb, attn=attn)
            real_b = token_ends(bb) != NEVER
            ids = attn[last]["b_expert_ids"][real_b]  # [N, k]
            route[d] = (F.one_hot(ids, E).float().sum((0, 1)) / ids.numel()).tolist()
            if d == "prose":
                real_a = token_ends(ba) != NEVER
                out["read"] = expert_attention_mass(attn[last]["a_reads_b"], attn[last]["b_expert_ids"], E, real_a,
                                                    attn[last]["a_reads_b_blocked"])
        out["route_prose"], out["route_code"] = route["prose"], route["code"]
        means = expert_output_means(pair, *dom["code"])[1]  # blank values from the code windows
        ba, bb = dom["prose"]
        score = lambda abl: pooled(bits_per_doc(run_pair(pair, ba, bb, ablate=(NO_ABLATION, abl))[0], ba))  # noqa: E731
        solo = pooled(bits_per_doc(run_pair(pair, ba, bb, disabled=frozenset(v["links"]))[0], ba))
        fused = score(NO_ABLATION)
        out["solo"], out["fused"] = solo, fused
        out["hide"] = [score(BusAblation(hide=frozenset({e}))) - fused for e in range(E)]
        out["blank"] = [score(BusAblation(blank=frozenset({e}), blank_value=means)) - fused for e in range(E)]
        order = sorted(range(E), key=lambda e: -out["read"]["share"][e])
        out["hide_top"] = {n: score(BusAblation(hide=frozenset(order[:n]))) - fused for n in (1, 2, 4, E - 1)}
        out["blank_all"] = score(BusAblation(blank=frozenset(range(E)), blank_value=means)) - fused
    res[unit.name] = out
    r = out["read"]
    print(f"{unit.name}: solo {solo:.3f} fused {fused:.3f} (gain {fused - solo:+.3f}) null share {r['null_share']:.2f}")
    print("  expert      " + " ".join(f"{e:>6d}" for e in range(E)))
    print("  route prose " + " ".join(f"{x:6.3f}" for x in out["route_prose"]))
    print("  route code  " + " ".join(f"{x:6.3f}" for x in out["route_code"]))
    print("  read share  " + " ".join(f"{x:6.3f}" for x in r["share"]))
    print("  read ratio  " + " ".join(f"{x:6.2f}" for x in r["ratio"]))
    print("  hide +bits  " + " ".join(f"{x:+6.3f}" for x in out["hide"]))
    print("  blank +bits " + " ".join(f"{x:+6.3f}" for x in out["blank"]))
    print(f"  hide most-read top-n: {', '.join(f'{n}: {x:+.3f}' for n, x in out['hide_top'].items())} | blank all: {out['blank_all']:+.3f}", flush=True)
(root / f"which_experts_{var}.json").write_text(json.dumps(res, indent=1))
