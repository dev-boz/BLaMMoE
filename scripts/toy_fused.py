"""WP-T2: fuse two frozen toy MoEs in lockstep and let them talk at every depth.

    PYTHONPATH=. python scripts/toy_fused.py configs/toy_poc.yaml <variant> [threads]

Variants are defined under ``fused.variants`` in the YAML. Writes
<out_dir>/fused_<variant>.json (+ .pt with the bridge weights).
"""
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch

from blammoe.toy import setup
from blammoe.toy.batching import make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import MixedSampler, split_recall_sample
from blammoe.toy.fused import FusedPair, bits_per_char, pair_loss, subset_accuracy, token_ends
from blammoe.toy.moe import load


def main(cfg_path: str, variant: str, threads: int = 4) -> None:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    fs, out, L = cfg.fused, Path(cfg.out_dir), cfg.seq_chars
    v = fs.variants[variant]
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt")
    B, _ = load(out / "model_b.pt")
    models = {"a": (A, tok_a), "b": (B, tok_b)}
    m1, t1 = models[v["pair"][0]]
    m2, t2 = models[v["pair"][1]]
    if m2 is m1:
        m2 = copy.deepcopy(m1)  # self-fusion control: two copies of the same model
    pair = FusedPair(m1, m2, v["links"], fs.d_bus, fs.n_heads, v.get("signal", "moe"))
    t0 = time.time()
    logf = open(out / f"fused_{variant}.log", "w")

    def log(msg):
        line = f"[{time.time() - t0:6.0f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    base = setup.sampler(cfg, seed=0)
    general = MixedSampler(base.prose, base.code, L, p_recall=0.0, seed=4243)
    rng = random.Random(4242)

    def draw(n: int):
        items = []
        for _ in range(n):
            if rng.random() < fs.p_split:
                s = split_recall_sample(rng, L)
                items.append((s.view("a"), s.view("b")))
            else:
                g = general.sample()
                items.append((g, g))
        return items

    def encode(items):
        b1 = make_batch([x for x, _ in items], t1, L)
        b2 = make_batch([y for _, y in items], t2, L)
        return b1, b2, token_ends(b1), token_ends(b2)

    opt = torch.optim.AdamW(pair.bridge_parameters(), lr=fs.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in pair.bridge_parameters())
    log(f"variant {variant} {v}: {n_params:,} bridge params")
    history = []
    for step in range(fs.steps):
        b1, b2, e1, e2 = encode(draw(fs.batch_size))
        l1, l2 = pair(b1.ids, b2.ids, e1, e2)
        loss = pair_loss(l1, b1, fs.answer_weight) + pair_loss(l2, b2, fs.answer_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == fs.steps - 1:
            gates = {k: [round(float(torch.tanh(lk.rx_a.gate.detach())), 3),
                         round(float(torch.tanh(lk.rx_b.gate.detach())), 3)] for k, lk in pair.links.items()}
            rec = {"step": step, "loss": round(float(loss.detach()), 4), "gates": gates}
            history.append(rec)
            log(str(rec))

    # ---------------- evaluation (held-out seeds)
    pair.eval()
    erng = random.Random(2025)
    splits = [split_recall_sample(erng, L) for _ in range(96)]
    s1 = make_batch([s.view("a") for s in splits], t1, L)
    s2 = make_batch([s.view("b") for s in splits], t2, L)
    se1, se2 = token_ends(s1), token_ends(s2)
    gsamp = MixedSampler(base.prose, base.code, L, p_recall=0.0, seed=777)
    gen = [gsamp.sample() for _ in range(48)]
    g1, g2 = make_batch(gen, t1, L), make_batch(gen, t2, L)
    ge1, ge2 = token_ends(g1), token_ends(g2)
    all_links = frozenset(int(k) for k in pair.links)

    def split_eval(**kw):
        with torch.no_grad():
            o1, o2 = pair(s1.ids, s2.ids, se1, se2, **kw)
        return {
            "m1_needs_partner": subset_accuracy(o1, s1, [s.partner_answers("a") for s in splits]),
            "m1_own": subset_accuracy(o1, s1, [s.own_answers("a") for s in splits]),
            "m2_needs_partner": subset_accuracy(o2, s2, [s.partner_answers("b") for s in splits]),
            "m2_own": subset_accuracy(o2, s2, [s.own_answers("b") for s in splits]),
        }

    def gen_eval(**kw):
        with torch.no_grad():
            o1, o2 = pair(g1.ids, g2.ids, ge1, ge2, **kw)
        return {"m1_bpc": bits_per_char(o1, g1), "m2_bpc": bits_per_char(o2, g2)}

    res = {"variant": variant, "spec": v, "bridge_params": n_params, "train": history,
           "split": {"fused": split_eval(), "shuffled_partner": split_eval(partner_roll=1),
                     "solo": split_eval(disabled=all_links)},
           "general": {"fused": gen_eval(), "shuffled_partner": gen_eval(partner_roll=1),
                       "solo": gen_eval(disabled=all_links)}}
    if len(all_links) > 1:
        res["split"]["only_link"] = {l: split_eval(disabled=all_links - {l}) for l in sorted(all_links)}
        res["split"]["without_link"] = {l: split_eval(disabled=frozenset({l})) for l in sorted(all_links)}
    for k, r in res["split"].items():
        if k in ("only_link", "without_link"):
            for l, rr in r.items():
                log(f"split {k} {l}: " + " ".join(f"{n}={x['acc']:.3f}" for n, x in rr.items()))
        else:
            log(f"split {k}: " + " ".join(f"{n}={x['acc']:.3f}" for n, x in r.items()))
    for k, r in res["general"].items():
        log(f"general {k}: " + " ".join(f"{n}={x:.4f}" for n, x in r.items()))
    (out / f"fused_{variant}.json").write_text(json.dumps(res, indent=1, default=str))
    torch.save(pair.links.state_dict(), out / f"fused_{variant}.pt")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
