"""Toy POC follow-ups: what is the linear expert transplant actually relying on?

Runs after scripts/toy_poc.py (reuses its trained models). Transplants into the
B layer/expert where toy_poc found transmission, and asks:

  1. which A expert / depth / signal works (specialist vs unrelated, dense vs
     routed-only, whole residual as a baseline);
  2. does a map fitted on prose+code ONLY (zero recall docs) still transmit —
     i.e. is the alignment general, or did the map learn the task?
  3. how many paired positions does the map need;
  4. what the transplant does to B's ordinary-text loss (interference);
  5. a longer learned-bridge run, from a routed-only AND a dense A source (rule 8: only the
     routed one can support an expert-level claim).

Transplant scores are reported for the ``native`` expert swap (only where B routed its target
expert, at its router weight) and the ``dense_additive`` control (the original hook). All
choices (target B layer, A specialist/unrelated experts) are made on the seed-2023 selection
set; only the seed-2024 set is reported.

    PYTHONPATH=. python scripts/toy_followup.py configs/toy_poc.yaml [threads] [bridge_steps]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from blammoe.align import apply, fit_linear
from blammoe.toy import setup
from blammoe.toy.bridge import (
    Bridge, Source, b_end_chars, causal_block_mask, encode_pair, eval_bridge, gather_aligned, lm_loss,
    native_transplant_hook, paired_expert_states, routing_at_answers, score_answers, train_bridge,
    transplant_hook,
)
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import MixedSampler
from blammoe.toy.moe import load


def translate(W, h: torch.Tensor) -> torch.Tensor:
    flat = apply(W, h.reshape(-1, h.shape[-1]).numpy()).astype(np.float32)
    return torch.from_numpy(flat).reshape(*h.shape[:-1], W.d_out)


def main(cfg_path: str, threads: int = 4, bridge_steps: int = 3000) -> None:
    torch.set_num_threads(threads)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    out, L = Path(cfg.out_dir), cfg.seq_chars
    poc = json.loads((out / "poc_results.json").read_text())
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt")
    B, _ = load(out / "model_b.pt")
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    base = setup.sampler(cfg, seed=0)
    texts = (base.prose, base.code)
    ev = encode_pair(setup.eval_recall_samples(cfg, n=96, seed=2024), tok_a, tok_b, L)
    a_clean, a_blind, _, b_blind = ev
    s_clean_a = encode_pair(setup.eval_recall_samples(cfg, n=96, seed=2023), tok_a, tok_b, L)[0]  # selection only
    ends = b_end_chars(b_blind)
    gen = MixedSampler(*texts, L, p_recall=0.0, seed=777)
    g_a, _, g_b, _ = encode_pair([gen.sample() for _ in range(48)], tok_a, tok_b, L)
    g_ends = b_end_chars(g_b)
    with torch.no_grad():
        g_base = lm_loss(B(g_b.ids), g_b)

    # target: the B layer/expert where toy_poc's native transplant worked best ON THE SELECTION SET
    lb = max(poc["transplant"], key=lambda k: poc["transplant"][k]["selection_native_acc"])
    lb, eb = int(lb), poc["transplant"][lb]["B_expert"]
    log(f"target B layer {lb} expert {eb}; B general LM loss {g_base:.3f}")

    def fit_set(p_recall: float, n: int, seed: int):
        s = MixedSampler(*texts, L, p_recall=p_recall, seed=seed)
        return encode_pair([s.sample() for _ in range(n)], tok_a, tok_b, L)

    fit_mixed = fit_set(cfg.p_recall, 160, 31337)
    fit_general = fit_set(0.0, 160, 31338)

    def run(src: Source, fit, ridge=1.0, max_pairs=None) -> dict:
        fa, _, fb, _ = fit
        HA, HB = paired_expert_states(A, B, src, lb, eb, fa, fb)
        if max_pairs:
            rows = np.random.default_rng(0).choice(len(HA), size=min(max_pairs, len(HA)), replace=False)
            HA, HB = HA[rows], HB[rows]
        W = fit_linear(HA, HB, ridge=ridge)
        sig = {"matched": src(A, a_clean.ids), "shuffled_doc": src(A, a_clean.ids).roll(1, dims=0),
               "blind_a": src(A, a_blind.ids)}
        row = {"n_pairs": len(HA)}
        for mode, hook in (("native", native_transplant_hook), ("dense_additive", transplant_hook)):
            row[mode] = {}
            for name, h in sig.items():
                with hook(B, lb, eb, translate(W, gather_aligned(h, ends)), 1.0), torch.no_grad():
                    row[mode][name] = score_answers(B(b_blind.ids), b_blind)["acc"]
            with hook(B, lb, eb, translate(W, gather_aligned(src(A, g_a.ids), g_ends)), 1.0), torch.no_grad():
                row[mode]["general_lm_loss"] = lm_loss(B(g_b.ids), g_b)
        return row

    res = {"target": {"B_layer": lb, "B_expert": eb}, "B_general_lm_loss": g_base}

    # 1. which A expert / depth / signal
    grid = {}
    for la in range(A.cfg.n_layers):
        route = routing_at_answers(A, s_clean_a, la)
        spec, unrel = int(route.argmax()), int(route.argmin())
        for tag, e in (("specialist", spec), ("unrelated", unrel)):
            for signal in ("expert_dense", "expert_routed"):
                key = f"A{la}_{tag}{e}_{signal}"
                grid[key] = {"route_at_answers": float(route[e]), **run(Source(la, e, signal), fit_mixed)}
                log(f"{key}: {grid[key]}")
        key = f"A{la}_resid"
        grid[key] = run(Source(la, spec, "resid"), fit_mixed)
        log(f"{key}: {grid[key]}")
    res["grid"] = grid

    # 2. map fitted with NO recall docs at all; 3. data-size sweep (both on A3 specialist, routed-only)
    la = A.cfg.n_layers - 1
    spec = int(routing_at_answers(A, s_clean_a, la).argmax())
    src = Source(la, spec, "expert_routed")
    res["fit_general_only"] = run(src, fit_general)
    log(f"fit on prose+code only: {res['fit_general_only']}")
    res["pairs_sweep"] = {}
    for n in (250, 1000, 4000, 16000):
        res["pairs_sweep"][n] = run(src, fit_mixed, max_pairs=n)
        log(f"pairs={n}: {res['pairs_sweep'][n]}")

    # 5. longer learned bridge (specialist, routed-only and dense), at the configured mid layer and B's last layer
    bs = cfg.bridge
    res["long_bridge"] = {}
    for kind, layer_b in [(k, lb_) for k in ("expert_routed", "expert_dense") for lb_ in sorted({bs.layer_b, B.cfg.n_layers - 1})]:
        src = Source(la, spec, kind)
        torch.manual_seed(0)
        bridge = Bridge(A.cfg.d_model, B.cfg.d_model, bs.d_bus, bs.n_heads)
        sampler = MixedSampler(*texts, L, p_recall=cfg.p_recall, seed=4242)
        hist = train_bridge(bridge, A, B, src, layer_b, lambda n: [sampler.sample() for _ in range(n)],
                            tok_a, tok_b, L, steps=bridge_steps, batch_size=bs.batch_size, lr=bs.lr,
                            answer_weight=bs.answer_weight, log=lambda m: log(f"  bridge {kind}@{layer_b} {m}"))
        scores = eval_bridge(bridge, A, B, src, layer_b, ev)
        with torch.no_grad(), bridge.hook(B, layer_b, src(A, g_a.ids), causal_block_mask(g_ends, g_a.ids.shape[1])):
            g_loss = lm_loss(B(g_b.ids), g_b)
        res["long_bridge"][f"{kind}@B{layer_b}"] = {"steps": bridge_steps, "eval": scores, "train": hist,
                                       "general_lm_loss": g_loss,
                                       "final_gate": float(torch.tanh(bridge.rx.gate.detach()))}
        log(f"long bridge {kind}@B{layer_b}: " + " ".join(f"{c}={s['acc']:.3f}" for c, s in scores.items())
            + f" | general LM loss {g_loss:.3f}")

    (out / "followup_results.json").write_text(json.dumps(res, indent=1))
    log(f"wrote {out / 'followup_results.json'}")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
