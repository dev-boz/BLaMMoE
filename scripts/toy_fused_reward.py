"""WP-T3: stage-2 fine-tuning of a trained fusion's links on ordinary text.

    PYTHONPATH=. python scripts/toy_fused_reward.py configs/toy_poc.yaml <variant> [threads]

<variant> names an entry under fused.variants with ``init_from`` (stage-1 links to
start from) and ``stage2: plain|reward``. Both arms see the same data in the same
order. Writes fused_<variant>.{pt,json,log}; run toy_fused_analyze.py on it after.
"""
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
from blammoe.toy.fused import FusedPair, bits_per_char, subset_accuracy, token_ends
from blammoe.toy.moe import load, lr_at
from blammoe.toy.reward import partner_win_mask, win_weighted_loss


def main(cfg_path: str, variant: str, threads: int = 2) -> None:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    fs, st, out, L = cfg.fused, cfg.stage2, Path(cfg.out_dir), cfg.seq_chars
    v = fs.variants[variant]
    if v["pair"] != "ab":
        raise ValueError("stage 2 is defined for the A+B pair")
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt")
    B, _ = load(out / "model_b.pt")
    pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, v["signal"], v.get("rel_pos", False))
    pair.links.load_state_dict(torch.load(out / f"fused_{v['init_from']}.pt"))
    lam = st.lam if v["stage2"] == "reward" else 0.0
    links = frozenset(int(k) for k in pair.links)
    t0 = time.time()
    logf = open(out / f"fused_{variant}.log", "w")

    def log(msg):
        line = f"[{time.time() - t0:6.0f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    base = setup.sampler(cfg, seed=0)
    train = MixedSampler(base.prose, base.code, L, p_recall=0.0, seed=5151)  # same stream for both arms
    opt = torch.optim.AdamW(pair.bridge_parameters(), lr=st.lr, weight_decay=0.0)
    log(f"{variant}: stage2={v['stage2']} lam={lam} from {v['init_from']}")
    hist = []
    for step in range(st.steps):
        for g in opt.param_groups:  # warmup: a fresh Adam's first steps would knock stage-1 links off
            g["lr"] = lr_at(step, st.steps, st.lr, warmup=100)
        samples = [train.sample() for _ in range(st.batch_size)]
        b1, b2 = make_batch(samples, tok_a, L), make_batch(samples, tok_b, L)
        e1, e2 = token_ends(b1), token_ends(b2)
        pair.train()
        l1, l2 = pair(b1.ids, b2.ids, e1, e2)
        with torch.no_grad():  # reward bookkeeping is done for both arms, so logs are comparable
            s1, s2 = pair(b1.ids, b2.ids, e1, e2, disabled=links)
            w1, w2 = pair(b1.ids, b2.ids, e1, e2, partner_roll=1)
        win1 = partner_win_mask(l1, s1, w1, b1.targets)
        win2 = partner_win_mask(l2, s2, w2, b2.targets)
        loss = win_weighted_loss(l1, b1.targets, win1, lam) + win_weighted_loss(l2, b2.targets, win2, lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == st.steps - 1:
            n1, n2 = int((b1.targets != -100).sum()), int((b2.targets != -100).sum())
            rec = {"step": step, "loss": round(float(loss.detach()), 4),
                   "win_rate_a": round(int(win1.sum()) / n1, 4), "win_rate_b": round(int(win2.sum()) / n2, 4)}
            hist.append(rec)
            log(str(rec))

    # ---------------- evaluation on held-out text (same sets as stage 1)
    pair.eval()
    gs = MixedSampler(base.prose, base.code, L, p_recall=0.0, seed=777)
    gen = [gs.sample() for _ in range(48)]
    g1, g2 = make_batch(gen, tok_a, L), make_batch(gen, tok_b, L)
    ge1, ge2 = token_ends(g1), token_ends(g2)
    with torch.no_grad():
        f = pair(g1.ids, g2.ids, ge1, ge2)
        s = pair(g1.ids, g2.ids, ge1, ge2, disabled=links)
        w = pair(g1.ids, g2.ids, ge1, ge2, partner_roll=1)
    res = {"variant": variant, "spec": v, "stage2": vars(st), "lam": lam, "train": hist, "general": {
        cond: {"m1_bpc": bits_per_char(x[0], g1), "m2_bpc": bits_per_char(x[1], g2)}
        for cond, x in (("fused", f), ("shuffled_partner", w), ("solo", s))}}
    res["general"]["win_rate"] = {
        "m1": float(partner_win_mask(f[0], s[0], w[0], g1.targets).sum() / (g1.targets != -100).sum()),
        "m2": float(partner_win_mask(f[1], s[1], w[1], g2.targets).sum() / (g2.targets != -100).sum()),
    }
    erng = random.Random(2025)
    splits = [split_recall_sample(erng, L) for _ in range(96)]
    p1 = make_batch([x.view("a") for x in splits], tok_a, L)
    p2 = make_batch([x.view("b") for x in splits], tok_b, L)
    with torch.no_grad():
        o1, o2 = pair(p1.ids, p2.ids, token_ends(p1), token_ends(p2))
    res["split_fused"] = {
        "m1_needs_partner": subset_accuracy(o1, p1, [x.partner_answers("a") for x in splits]),
        "m2_needs_partner": subset_accuracy(o2, p2, [x.partner_answers("b") for x in splits]),
    }
    for k, r in res["general"].items():
        log(f"general {k}: " + " ".join(f"{n}={x:.4f}" for n, x in r.items()))
    log("split fused: " + " ".join(f"{n}={x['acc']:.3f}" for n, x in res["split_fused"].items()))
    (out / f"fused_{variant}.json").write_text(json.dumps(res, indent=1, default=str))
    torch.save(pair.links.state_dict(), out / f"fused_{variant}.pt")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
