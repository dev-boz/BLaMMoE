"""Cross-tokenizer causality audit of trained fused pairs (prefix-honest A scoring).

For each variant: A's bits/char on ordinary text and A's accuracy on split-recall
answers that need B, scored three ways on the SAME positions:
  solo         - A alone (links off)
  fused_full   - fused, partner tokenized on the whole text (how results were reported)
  fused_honest - fused, partner re-run on the prefix only (no future-dependent boundaries)
fused_full - fused_honest = what A got from B's token boundaries peeking ahead.

    PYTHONPATH=. python scripts/toy_fused_audit.py configs/toy_poc.yaml [threads] [n_docs] [variant ...]
"""
import json
import random
import sys
from pathlib import Path

import torch

from blammoe.toy import setup
from blammoe.toy.audit import full_text_a_nll, honest_a_nll, prefix_inputs, to_bits
from blammoe.toy.batching import make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import MixedSampler, Sample, split_recall_sample
from blammoe.toy.fused import FusedPair, token_ends
from blammoe.toy.moe import load

DEFAULT = ["ab_experts_all_rel", "ab_experts_all_rel_plain", "ab_all_rel"]


def main(cfg_path: str, threads: int = 4, n_docs: int = 16, *variants: str) -> None:
    torch.set_num_threads(threads)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    fs, out, L = cfg.fused, Path(cfg.out_dir), cfg.seq_chars
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt")
    B, _ = load(out / "model_b.pt")
    base = setup.sampler(cfg, seed=0)
    gs = MixedSampler(base.prose, base.code, L, p_recall=0.0, seed=777)
    docs = [gs.sample().text for _ in range(48)][:n_docs]
    rng = random.Random(2025)
    splits = [split_recall_sample(rng, L) for _ in range(96)][:48]
    report = {}
    for name in variants or DEFAULT:
        v = fs.variants[name]
        pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, v.get("signal", "moe"), v.get("rel_pos", False))
        pair.links.load_state_dict(torch.load(out / f"fused_{name}.pt"))
        pair.eval()
        off = frozenset(int(k) for k in pair.links)
        nll = {"solo": [], "fused_full": [], "fused_honest": []}
        is_letter = []
        for text in docs:
            doc = Sample(text=text, blind=text, kind="general")
            a, b = make_batch([doc], tok_a, L), make_batch([doc], tok_b, L)
            ea, eb = token_ends(a), token_ends(b)
            pos = [i for i in range(a.ids.shape[1]) if a.targets[0, i] != -100]
            with torch.no_grad():
                nll["solo"].append(full_text_a_nll(pair, a, b, ea, eb, disabled=off)[0, pos])
                nll["fused_full"].append(full_text_a_nll(pair, a, b, ea, eb)[0, pos])
            nll["fused_honest"].append(honest_a_nll(pair, text, pos, tok_a, tok_b))
            is_letter += [text[i + 1].isalpha() for i in pos]
        letter = torch.tensor(is_letter)
        cat = {k: torch.cat(x) for k, x in nll.items()}
        general = {k: {"all": to_bits(x), "next_is_letter": to_bits(x[letter]),
                       "next_not_letter": to_bits(x[~letter])} for k, x in cat.items()}
        general["n_positions"] = int(letter.numel())

        # split recall: A's answers that only B can supply
        rec = {"fused_full": [], "fused_honest": [], "solo": []}
        for s in splits:
            sa, sb = s.view("a"), s.view("b")
            a, b = make_batch([sa], tok_a, L), make_batch([sb], tok_b, L)
            ea, eb = token_ends(a), token_ends(b)
            pos = [c - 1 for c in s.partner_answers("a") if 0 <= c - 1 < a.ids.shape[1]]
            if not pos:
                continue
            tgt = a.targets[0, pos]
            with torch.no_grad():
                la, _ = pair(a.ids, b.ids, ea, eb)
                ls, _ = pair(a.ids, b.ids, ea, eb, disabled=off)
            rec["fused_full"].append(la[0, pos].argmax(-1) == tgt)
            rec["solo"].append(ls[0, pos].argmax(-1) == tgt)
            # honest: B gets ITS OWN view, re-tokenized on the prefix; A its own view
            hon = []  # one prefix per answer position
            for p in pos:
                ia, ea2, _, _, t, last = prefix_inputs(sa.text, [p], tok_a, tok_b)
                _, _, ib, eb2, _, _ = prefix_inputs(sb.text, [p], tok_a, tok_b)
                with torch.no_grad():
                    lh, _ = pair(ia, ib, ea2, eb2)
                hon.append(lh[0, last[0]].argmax(-1) == t[0])
            rec["fused_honest"].append(torch.stack(hon))
        recall = {k: float(torch.cat(x).float().mean()) for k, x in rec.items()}
        recall["n_answers"] = int(torch.cat(rec["solo"]).numel())
        report[name] = {"general_bits_per_char": general, "split_recall_A_needs_B_acc": recall}
        print(f"== {name}", flush=True)
        print("  general bpc  " + " | ".join(f"{k}: {v['all']:.4f} (letter {v['next_is_letter']:.3f}, "
              f"non-letter {v['next_not_letter']:.3f})" for k, v in general.items() if isinstance(v, dict)), flush=True)
        print("  recall acc   " + " | ".join(f"{k}: {v:.3f}" for k, v in recall.items()), flush=True)
    (out / "fused_audit.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], *(int(a) for a in args[1:3]), *args[3:])
