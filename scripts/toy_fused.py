"""WP-T2: fuse two frozen toy MoEs in lockstep and let them talk at every depth.

    PYTHONPATH=. python scripts/toy_fused.py configs/toy_poc.yaml <variant> [threads]

Variants are defined under ``fused.variants`` in the YAML. Writes
<out_dir>/fused_<variant>.json (+ .pt with the bridge weights). Runs on a GPU if one is
available (``BLAMMOE_DEVICE`` overrides). Training/eval logic: blammoe/toy/fused_train.py.
"""
import copy
import json
import sys
import time
from pathlib import Path

import torch

from blammoe.device import pick_device
from blammoe.toy import setup
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.fused import FusedPair
from blammoe.toy.fused_train import (
    general_bpc, general_eval_set, mixed_draw, split_eval_set, split_scores, train_fused,
)
from blammoe.toy.moe import load


def main(cfg_path: str, variant: str, threads: int = 4) -> None:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    device = pick_device()
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    fs, out, L = cfg.fused, Path(cfg.out_dir), cfg.seq_chars
    v = fs.variants[variant]
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt", device)
    B, _ = load(out / "model_b.pt", device)
    models = {"a": (A, tok_a), "b": (B, tok_b)}
    m1, t1 = models[v["pair"][0]]
    m2, t2 = models[v["pair"][1]]
    if m2 is m1:
        m2 = copy.deepcopy(m1)  # self-fusion control: two copies of the same model
    pair = FusedPair(m1, m2, v["links"], fs.d_bus, fs.n_heads, v.get("signal", "moe"), v.get("rel_pos", False),
                     v.get("safe_mask", False)).to(device)
    t0 = time.time()
    logf = open(out / f"fused_{variant}.log", "w")

    def log(msg):
        line = f"[{time.time() - t0:6.0f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    n_params = sum(p.numel() for p in pair.bridge_parameters())
    log(f"variant {variant} {v}: {n_params:,} bridge params on {device}")
    history = train_fused(pair, t1, t2, mixed_draw(*setup.texts(cfg, "train"), L, fs.p_split, seed=4242), L,
                          steps=v.get("steps", fs.steps), batch_size=fs.batch_size, lr=fs.lr,
                          answer_weight=fs.answer_weight, log=log)

    # ---------------- evaluation (held-out seeds)
    splits = split_eval_set(L)
    gen = general_eval_set(*setup.texts(cfg, "test"), L)
    all_links = frozenset(int(k) for k in pair.links)
    split_eval = lambda **kw: split_scores(pair, splits, t1, t2, L, **kw)  # noqa: E731
    gen_eval = lambda **kw: general_bpc(pair, gen, t1, t2, L, **kw)  # noqa: E731
    res = {"variant": variant, "spec": v, "bridge_params": n_params, "train": history,
           "split": {"fused": split_eval(), "shuffled_partner": split_eval(partner_roll=1),
                     "solo": split_eval(disabled=all_links)},
           "general": {"fused": gen_eval(), "shuffled_partner": gen_eval(partner_roll=1),
                       "solo": gen_eval(disabled=all_links)}}
    if len(all_links) > 1:
        res["split"]["only_link"] = {l: split_eval(disabled=all_links - {l}) for l in sorted(all_links)}
        res["split"]["without_link"] = {l: split_eval(disabled=frozenset({l})) for l in sorted(all_links)}
    fmt = lambda r: " ".join(f"{n}={x['acc']:.3f} (exact {x['exact']:.3f})" for n, x in r.items())  # noqa: E731
    for k, r in res["split"].items():
        if k in ("only_link", "without_link"):
            for l, rr in r.items():
                log(f"split {k} {l}: {fmt(rr)}")
        else:
            log(f"split {k}: {fmt(r)}")
    for k, r in res["general"].items():
        log(f"general {k}: " + " ".join(f"{n}={x:.4f}" for n, x in r.items()))
    (out / f"fused_{variant}.json").write_text(json.dumps(res, indent=1, default=str))
    torch.save(pair.links.state_dict(), out / f"fused_{variant}.pt")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
