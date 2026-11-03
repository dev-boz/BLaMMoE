"""Where does each toy model keep its domain knowledge: in its experts or its attention?

The A-side gap (docs/toy-complement-results.md): A (char, code-heavy) gives B code through the
expert bus, but B (BPE, prose-heavy) never gives A prose. One explanation: B's prose skill is not in
its experts' outputs, so the bus has little to carry. This measures, on the held-out prose and code
windows (``domain_eval_sets``), for the four model kinds of a complement run (s_i's A and B, c_i's
code-heavy B', d_i's prose-heavy A'):

  routing : each expert's share of top-k slots per domain, and per layer the JS divergence (nats)
            between prose and code routing (0 = the router ignores the domain)
  ablate  : bits/char per domain with one part replaced by its mean output over a balanced
            train-split reference set: a layer's attention, a layer's whole MoE, the MoE at every
            layer, or one expert (only where routed, at its router weight, as
            ``bridge.native_transplant_hook`` swaps an expert)

A part holds domain knowledge if ablating it costs the model's strong domain more than its weak one.
Comparing A with A' (and B with B') separates the domain from the architecture and tokenizer, and
gives the headline: how much of a model's advantage on its strong domain over the same-architecture
model with the other mix is left once the MoE at every layer is mean-ablated in both.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/toy_where_knowledge.py results/toy_complement [threads]

Writes <root>/where_knowledge.json. CPU only, no training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from blammoe.stats import paired_bootstrap_diff
from blammoe.toy import setup
from blammoe.toy.batching import make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import MixedSampler
from blammoe.toy.fused import _js, bits_per_doc
from blammoe.toy.fused_train import domain_eval_sets
from blammoe.toy.moe import load

DOMAINS = ("prose", "code")
KINDS = {  # label: (unit prefix, model); A/B as in the s units, A'/B' = the other mix (d / c controls)
    "A_code_char": ("s", "a"), "Ap_prose_char": ("d", "a"),
    "B_prose_bpe": ("s", "b"), "Bp_code_bpe": ("c", "b"),
}
PAIRS = (("A_code_char", "Ap_prose_char"), ("B_prose_bpe", "Bp_code_bpe"))  # (model, same arch other mix)


class Replace:
    """Forward hooks that swap module outputs: ``fns`` = [(module, hook)], removed on exit."""

    def __init__(self, fns):
        self.fns, self.handles = fns, []

    def __enter__(self):
        self.handles = [m.register_forward_hook(f) for m, f in self.fns]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False


def _top(moe, x):
    return moe.router(x).softmax(-1).topk(moe.top_k, -1)


@torch.no_grad()
def reference_means(model, batch) -> dict:
    """Mean output of each layer's attention and MoE, and each expert's mean routed output, over
    the batch's real positions."""
    valid = batch.targets != -100
    cap = {}
    fns = []
    for l, blk in enumerate(model.layers):
        fns.append((blk.attn, lambda m, i, o, l=l: cap.__setitem__(("attn", l), o[0])))
        fns.append((blk.moe, lambda m, i, o, l=l: cap.__setitem__(("moe", l), (i[0], o))))
    with Replace(fns):
        model(batch.ids)
    means = {}
    for l, blk in enumerate(model.layers):
        means["attn", l] = cap["attn", l][valid].mean(0)
        x, out = cap["moe", l]
        means["moe", l] = out[valid].mean(0)
        x = x[valid]
        idx = _top(blk.moe, x).indices
        for e, ex in enumerate(blk.moe.experts):
            r = (idx == e).any(-1)
            means["expert", l, e] = ex(x[r]).mean(0) if r.any() else x.new_zeros(x.shape[-1])
    return means


def ablation(model, means: dict, what: tuple) -> list:
    """Hooks for ``what`` = ("attn", l) | ("moe", l) | ("moe_all",) | ("expert", l, e)."""
    if what[0] == "moe_all":
        return [f for l in range(len(model.layers)) for f in ablation(model, means, ("moe", l))]
    m = means[what]
    if what[0] == "attn":
        return [(model.layers[what[1]].attn, lambda mod, i, o: (m.expand_as(o[0]), o[1]))]
    if what[0] == "moe":
        return [(model.layers[what[1]].moe, lambda mod, i, o: m.expand_as(o))]
    _, l, e = what

    def fn(module, inputs, output):
        x = inputs[0]
        w, idx = _top(module, x)
        gate = (w * (idx == e)).sum(-1, keepdim=True)  # router weight if routed, else 0
        return output + gate * (m - module.experts[e](x))

    return [(model.layers[l].moe, fn)]


@torch.no_grad()
def routing(model, batch) -> list[list[float]]:
    """Per layer, each expert's share of the top-k slots over the batch's real positions."""
    valid = batch.targets != -100
    cap = {}
    with Replace([(blk.moe, lambda m, i, o, l=l: cap.__setitem__(l, i[0])) for l, blk in enumerate(model.layers)]):
        model(batch.ids)
    out = []
    for l, blk in enumerate(model.layers):
        idx = _top(blk.moe, cap[l][valid]).indices
        out.append((F.one_hot(idx, blk.moe.router.out_features).sum(1).float().mean(0) / blk.moe.top_k).tolist())
    return out


def conditions(model) -> list[tuple]:
    n, E = len(model.layers), model.cfg.n_experts
    return ([("moe_all",)] + [(k, l) for l in range(n) for k in ("attn", "moe")]
            + [("expert", l, e) for l in range(n) for e in range(E)])


def name(what: tuple) -> str:
    return "_".join(str(x) for x in what)


def pooled(rows) -> float:
    r = np.asarray(rows)
    return float(r[:, 0].sum() / r[:, 1].sum())


@torch.no_grad()
def analyse(unit: Path, which: str) -> dict:
    cfg = ToyPOCConfig.from_yaml(unit / "config.yaml")
    spec = cfg.model_a if which == "a" else cfg.model_b
    L = cfg.seq_chars
    tok = setup.tokenizer(spec, cfg)
    model, _ = load(unit / f"model_{which}.pt")
    docs = domain_eval_sets(*setup.texts(cfg, "test"), L)
    batches = {d: make_batch(docs[d], tok, L) for d in DOMAINS}
    ref = MixedSampler(*setup.texts(cfg, "train"), L, p_recall=0.0, seed=4321)
    means = reference_means(model, make_batch([ref.sample(d) for d in DOMAINS for _ in range(48)], tok, L))
    base = {d: bits_per_doc(model(b.ids), b) for d, b in batches.items()}
    res = {"unit": str(unit), "model": which, "p_prose": spec.p_prose,
           "strong": "prose" if spec.p_prose > 0.5 else "code",
           "solo": {d: pooled(r) for d, r in base.items()},
           "routing": {d: routing(model, b) for d, b in batches.items()}, "ablate": {}}
    res["routing_js"] = [float(_js(torch.tensor(p), torch.tensor(c)))
                         for p, c in zip(res["routing"]["prose"], res["routing"]["code"])]
    for what in conditions(model):
        r = {}
        with Replace(ablation(model, means, what)):
            for d, b in batches.items():
                rows = bits_per_doc(model(b.ids), b)
                r[d] = {"bpc": pooled(rows), "delta": paired_bootstrap_diff(rows, base[d])}
        res["ablate"][name(what)] = r
    return res


def summary(res: dict) -> dict:
    """Per kind, mean over seeds, of each part's cost on the strong and the weak domain; and per pair
    the strong-domain advantage over the other-mix model, whole and with every MoE layer ablated."""
    out = {"kinds": {}, "advantage": {}}
    for kind, per_seed in res.items():
        seeds = list(per_seed.values())
        strong = seeds[0]["strong"]
        weak = "code" if strong == "prose" else "prose"
        n = len(seeds[0]["routing_js"])
        cost = lambda c, d: float(np.mean([s["ablate"][c][d]["delta"]["diff"] for s in seeds]))  # noqa: E731
        k = {"strong": strong, "solo": {d: float(np.mean([s["solo"][d] for s in seeds])) for d in DOMAINS},
             "routing_js": np.mean([s["routing_js"] for s in seeds], 0).tolist(),
             "parts": {c: {"strong": cost(c, strong), "weak": cost(c, weak)}
                       for c in ["moe_all"] + [f"{p}_{l}" for l in range(n) for p in ("attn", "moe")]}}
        # single experts: the domain-selective cost (strong minus weak), and how concentrated it is
        sel = []
        for s in seeds:
            e = {c: v[strong]["delta"]["diff"] - v[weak]["delta"]["diff"]
                 for c, v in s["ablate"].items() if c.startswith("expert_")}
            v = np.array(sorted(e.values(), reverse=True))
            pos = v[v > 0]
            sel.append({"max": float(v[0]), "sum_pos": float(pos.sum()),
                        "top2_share": float(pos[:2].sum() / pos.sum()) if pos.sum() > 0 else float("nan"),
                        "mean_strong": float(np.mean([vv[strong]["delta"]["diff"] for c, vv in s["ablate"].items()
                                                      if c.startswith("expert_")])),
                        "top": sorted(e, key=lambda c: -e[c])[:3]})
        k["single_expert_selective"] = {key: (float(np.nanmean([x[key] for x in sel])) if key != "top"
                                              else [x["top"] for x in sel]) for key in sel[0]}
        out["kinds"][kind] = k
    for m, other in PAIRS:
        strong = out["kinds"][m]["strong"]
        rows = []
        for i in res[m]:
            a, o = res[m][i], res[other][i]
            whole = o["solo"][strong] - a["solo"][strong]
            parts = {c: o["ablate"][c][strong]["bpc"] - a["ablate"][c][strong]["bpc"] for c in a["ablate"]
                     if not c.startswith("expert_")}
            rows.append({"whole": whole, **parts})
        out["advantage"][m] = {"vs": other, "domain": strong,
                               "per_seed": rows, "mean": {c: float(np.mean([r[c] for r in rows])) for c in rows[0]}}
    return out


def main(root: str, threads: int = 8) -> None:
    torch.set_num_threads(threads)
    root = Path(root)
    seeds = sorted(int(p.name[1:]) for p in root.glob("s[0-9]*") if (p / "config.yaml").exists())
    res = {}
    for kind, (prefix, which) in KINDS.items():
        res[kind] = {}
        for i in seeds:
            res[kind][i] = analyse(root / f"{prefix}{i}", which)
            r = res[kind][i]
            print(f"{kind} seed set {i}: solo " + " ".join(f"{d}={r['solo'][d]:.3f}" for d in DOMAINS)
                  + "  moe_all +" + " +".join(f"{d}={r['ablate']['moe_all'][d]['delta']['diff']:.3f}" for d in DOMAINS),
                  flush=True)
    s = summary(res)
    (root / "where_knowledge.json").write_text(json.dumps({"per_seed": res, "summary": s}, indent=1))
    for kind, k in s["kinds"].items():
        print(f"\n{kind} (strong {k['strong']}): solo " + " ".join(f"{d}={k['solo'][d]:.3f}" for d in DOMAINS)
              + "  routing JS by layer " + " ".join(f"{x:.3f}" for x in k["routing_js"]))
        for c, v in k["parts"].items():
            print(f"  {c:8s} cost strong {v['strong']:+.3f}  weak {v['weak']:+.3f}  selective {v['strong'] - v['weak']:+.3f}")
        print("  single experts: " + " ".join(f"{a}={b}" if a == "top" else f"{a}={b:.3f}"
                                              for a, b in k["single_expert_selective"].items()))
    for m, a in s["advantage"].items():
        print(f"\n{m} advantage on {a['domain']} over {a['vs']} (bits/char, mean of seeds): "
              + " ".join(f"{c}={v:.3f}" for c, v in a["mean"].items()))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
