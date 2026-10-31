"""Is a paired model DIFFERENT because of what its partner knows? (not only: is it better)

The owner (2026-10-30): the aim is a different model, not necessarily a better one. Being different
alone is easy, though: dial 2's reward at lam 50 moved every prediction and made both models worse,
just as much with a wrong partner. So this asks for difference that depends on the partner's content
and leans towards it. Per trained translator, model and domain (``domain_eval_sets``, test split), on
the positions where both tokenizers have a boundary, with next-token distributions read as
next-character ones (``fused.diversity``'s view):

  shift        JS(paired, alone): how different the paired model is
  shift_wrong  JS(paired with a wrong-document partner, alone)
  dependence   JS(paired, paired with a wrong partner): how much the change depends on what the
               partner is reading (0 = the links ignore the partner's content)
  toward       JS(paired with a wrong partner, partner alone) - JS(paired, partner alone): how much
               closer the real partner pulls the model to what the partner itself predicts at that
               spot than a wrong partner does (positive = towards the partner's own view)
               = attraction + repulsion, split exactly (the Codex review, 2026-10-31):
  attraction   JS(alone, partner alone) - JS(paired, partner alone): how far the real partner
               actually moves the model towards the partner's view (the transfer part)
  repulsion    JS(paired with a wrong partner, partner alone) - JS(alone, partner alone): how far a
               wrong partner pushes it away (damage; it inflates toward when the links lean hard)
  bits         bits/char paired minus alone, and paired minus wrong partner (the "better" view)

JS in nats, means per position. Each has per-document rows so differences get paired bootstrap CIs.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/toy_partner_shift.py results/toy_complement [threads]

Every unit dir under the root with a config.yaml, every fused_<variant>.pt in it. Writes
<root>/partner_shift.json. CPU only, no training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from blammoe.stats import paired_bootstrap_diff
from blammoe.toy import setup
from blammoe.toy.batching import make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.fused import FusedPair, _js, bits_per_doc, char_predictions, first_char_matrix, run_pair
from blammoe.toy.fused_train import domain_eval_sets
from blammoe.toy.moe import load
from blammoe.toy.tokenizers import CharTokenizer

CONDS = ("solo", "fused", "wrong")


def doc_rows(values: torch.Tensor, docs: list[int], n_docs: int) -> list[list[float]]:
    """Per-position values -> per-document [sum, count] rows (pooled mean = sum / count)."""
    s = torch.zeros(n_docs, dtype=torch.float64).index_add_(0, torch.tensor(docs), values.double())
    c = torch.zeros(n_docs, dtype=torch.float64).index_add_(0, torch.tensor(docs), torch.ones(len(docs), dtype=torch.float64))
    return [[float(a), float(b)] for a, b in zip(s, c)]


def shift_rows(p: dict, partner_solo: dict, texts: list[str], chars: list[str]) -> dict:
    """``p``: {cond: {(doc, chars read): next-char probs}} for one model; ``partner_solo`` the same
    for the partner alone. -> per-document rows of each JS quantity."""
    col = {c: j for j, c in enumerate(chars)}
    keys = [k for k in p["solo"] if k in partner_solo and k[1] < len(texts[k[0]]) and texts[k[0]][k[1]] in col]
    S, Fu, W, P = (torch.stack([d[k] for k in keys]) for d in (p["solo"], p["fused"], p["wrong"], partner_solo))
    docs = [k[0] for k in keys]
    js = {"shift": _js(Fu, S), "shift_wrong": _js(W, S), "dependence": _js(Fu, W),
          "fused_to_partner": _js(Fu, P), "wrong_to_partner": _js(W, P), "solo_to_partner": _js(S, P)}
    return {k: doc_rows(v, docs, len(texts)) for k, v in js.items()}


def pooled(rows) -> float:
    r = np.asarray(rows)
    return float(r[:, 0].sum() / r[:, 1].sum())


@torch.no_grad()
def analyse(unit: Path, variant: str) -> dict:
    cfg = ToyPOCConfig.from_yaml(unit / "config.yaml")
    fs, L = cfg.fused, cfg.seq_chars
    v = fs.variants[variant]
    if v["pair"] != "ab":
        raise ValueError(f"{variant}: only A-B pairs")
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(unit / "model_a.pt")
    B, _ = load(unit / "model_b.pt")
    pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, v.get("signal", "moe"), v.get("rel_pos", False),
                     v.get("safe_mask", False))
    pair.links.load_state_dict(torch.load(unit / f"fused_{variant}.pt", map_location="cpu"))
    pair.eval()
    links = frozenset(int(k) for k in pair.links)
    kw = {"solo": {"disabled": links}, "fused": {}, "wrong": {"partner_roll": 1}}
    chars = CharTokenizer().itos[1:]
    M = first_char_matrix(tok_a, chars), first_char_matrix(tok_b, chars)
    out = {}
    for d, docs in domain_eval_sets(*setup.texts(cfg, "test"), L).items():
        b = make_batch(docs, tok_a, L), make_batch(docs, tok_b, L)
        p, bits = ({}, {}), ({}, {})
        for c in CONDS:
            o = run_pair(pair, *b, **kw[c])
            for m in (0, 1):
                p[m][c] = char_predictions(o[m], b[m], M[m])
                bits[m][c] = bits_per_doc(o[m], b[m])
        texts = [s.text for s in docs]
        for m, name in ((0, "m1"), (1, "m2")):
            rows = shift_rows(p[m], p[1 - m]["solo"], texts, chars)
            r = {k: pooled(x) for k, x in rows.items()}
            r["toward"] = paired_bootstrap_diff(rows["wrong_to_partner"], rows["fused_to_partner"])
            r["attraction"] = paired_bootstrap_diff(rows["solo_to_partner"], rows["fused_to_partner"])
            r["repulsion"] = paired_bootstrap_diff(rows["wrong_to_partner"], rows["solo_to_partner"])
            r["dependence_minus_shift_wrong"] = paired_bootstrap_diff(rows["dependence"], rows["shift_wrong"])
            r["bits"] = {c: pooled(bits[m][c]) for c in CONDS}
            r["bits_fused_minus_solo"] = paired_bootstrap_diff(bits[m]["fused"], bits[m]["solo"])
            r["bits_fused_minus_wrong"] = paired_bootstrap_diff(bits[m]["fused"], bits[m]["wrong"])
            out[f"{name}_{d}"] = r
    return out


def main(root: str, threads: int = 8) -> None:
    torch.set_num_threads(threads)
    root = Path(root)
    res = {}
    for unit in sorted(p for p in root.iterdir() if (p / "config.yaml").exists()):
        for pt in sorted(unit.glob("fused_*.pt")):
            variant = pt.stem.removeprefix("fused_")
            r = res.setdefault(variant, {})[unit.name] = analyse(unit, variant)
            for k in ("m1_prose", "m2_code"):  # each model's weak domain
                x = r[k]
                print(f"{variant} {unit.name} {k}: shift {x['shift']:.4f} wrong {x['shift_wrong']:.4f} "
                      f"dep {x['dependence']:.4f} toward {x['toward']['diff']:+.4f} "
                      f"[{x['toward']['lo']:+.4f}, {x['toward']['hi']:+.4f}] "
                      f"attr {x['attraction']['diff']:+.4f} [{x['attraction']['lo']:+.4f}, "
                      f"{x['attraction']['hi']:+.4f}] rep {x['repulsion']['diff']:+.4f} "
                      f"bits {x['bits_fused_minus_solo']['diff']:+.3f}", flush=True)
    (root / "partner_shift.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
