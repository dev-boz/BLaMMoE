"""Toy POC: can one MoE expert's output be carried into a different MoE and used?

Needs model_a.pt / model_b.pt from scripts/toy_train.py. Writes
<out_dir>/poc_results.json and prints a summary.

    PYTHONPATH=. python scripts/toy_poc.py configs/toy_poc.yaml [threads]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from blammoe.align import apply, fit_linear, residual
from blammoe.metrics import run_battery
from blammoe.toy import setup
from blammoe.toy.bridge import (
    Bridge, Source, b_end_chars, causal_block_mask, encode_pair, gather_aligned, lm_loss, paired_expert_states,
    routing_at_answers, score_answers, train_bridge, eval_bridge, transplant_hook,
)
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import MixedSampler
from blammoe.toy.moe import load


def main(cfg_path: str, threads: int = 4) -> None:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    out = Path(cfg.out_dir)
    L = cfg.seq_chars
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt")
    B, _ = load(out / "model_b.pt")
    res: dict = {"config": cfg_path}
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    # ---- held-out eval data (seeds disjoint from pretraining / bridge training)
    recall = setup.eval_recall_samples(cfg, n=96, seed=2024)
    ev = encode_pair(recall, tok_a, tok_b, L)
    a_clean, a_blind, b_clean, b_blind = ev
    general = MixedSampler(*_texts(cfg), L, p_recall=0.0, seed=777)
    gen_samples = [general.sample() for _ in range(48)]
    g_a, _, g_b, _ = encode_pair(gen_samples, tok_a, tok_b, L)

    with torch.no_grad():
        res["baselines"] = {
            "A_clean": score_answers(A(a_clean.ids), a_clean),
            "A_blind": score_answers(A(a_blind.ids), a_blind),
            "B_clean": score_answers(B(b_clean.ids), b_clean),
            "B_blind": score_answers(B(b_blind.ids), b_blind),
            "B_general_lm_loss": lm_loss(B(g_b.ids), g_b),
        }
    log(f"baselines {res['baselines']}")

    # ---- expert selection: operation-level (who fires when predicting answer digits)
    la = cfg.bridge.layer_a
    route_a = routing_at_answers(A, a_clean, la)
    spec_a, unrel_a = int(route_a.argmax()), int(route_a.argmin())
    res["selection"] = {"A_layer": la, "A_route_at_answers": route_a.round(3).tolist(),
                        "A_specialist": spec_a, "A_unrelated": unrel_a}
    log(f"A layer {la} routing at answers {route_a.round(2)} -> specialist {spec_a}, unrelated {unrel_a}")

    # ---- linear transplant (zero-shot: fitted on clean general+recall pairs only)
    fit_sampler = MixedSampler(*_texts(cfg), L, p_recall=cfg.p_recall, seed=31337)
    fit_samples = [fit_sampler.sample() for _ in range(160)]
    fa, _, fb, _ = encode_pair(fit_samples, tok_a, tok_b, L)
    src_spec = Source(la, spec_a, "expert_dense")
    rand_B, _ = load(out / "model_b.pt")
    for p in rand_B.parameters():  # known-bad: same arch, re-initialised
        torch.nn.init.normal_(p, std=0.02) if p.dim() > 1 else torch.nn.init.ones_(p)
    transplant = {}
    for lb in range(B.cfg.n_layers):
        route_b = routing_at_answers(B, b_clean, lb)
        eb = int(route_b.argmax())
        HA, HB = paired_expert_states(A, B, src_spec, lb, eb, fa, fb)
        n = len(HA)
        tr, te = slice(0, int(0.8 * n)), slice(int(0.8 * n), n)
        W = fit_linear(HA[tr], HB[tr], ridge=1.0)
        HA_bad, HB_bad = paired_expert_states(A, rand_B, src_spec, lb, eb, fa, fb)
        entry = {
            "B_expert": eb, "B_route_at_answers": route_b.round(3).tolist(), "n_pairs": n,
            "heldout_residual": residual(W, HA[te], HB[te]),
            "battery": vars(run_battery(HA[te], HB[te], max_n=3000)),
            "battery_vs_random_B": vars(run_battery(HA_bad[te], HB_bad[te], max_n=3000)),
            "alpha": {},
        }
        ends = b_end_chars(b_blind)
        gen = torch.Generator().manual_seed(0)
        signals = {
            "matched": src_spec(A, a_clean.ids),
            "shuffled_doc": src_spec(A, a_clean.ids).roll(1, dims=0),
            "blind_a": src_spec(A, a_blind.ids),
        }
        for alpha in (0.5, 1.0, 2.0, 4.0):
            row = {}
            for name, h in signals.items():
                tr_h = _translate(W, gather_aligned(h, ends))
                with transplant_hook(B, lb, eb, tr_h, alpha), torch.no_grad():
                    row[name] = score_answers(B(b_blind.ids), b_blind)
            # Test C: right document, wrong (earlier) position — 4 B tokens back
            tr_h = _translate(W, gather_aligned(signals["matched"], ends)).roll(4, dims=1)
            with transplant_hook(B, lb, eb, tr_h, alpha), torch.no_grad():
                row["shifted_pos"] = score_answers(B(b_blind.ids), b_blind)
            entry["alpha"][str(alpha)] = row
        transplant[str(lb)] = entry
        log(f"transplant B layer {lb} expert {eb}: resid {entry['heldout_residual']:.3f} "
            f"cka {entry['battery']['cka']:.3f} knn {entry['battery']['mutual_knn']:.3f} | "
            + " ".join(f"a={a}: {r['matched']['acc']:.3f}/{r['shuffled_doc']['acc']:.3f}"
                       for a, r in entry["alpha"].items()))
    res["transplant"] = transplant
    del gen

    # ---- learned bridge variants (Tx -> bus -> gated x-attn Rx), A and B frozen
    variants = {
        "specialist_dense": Source(la, spec_a, "expert_dense"),
        "specialist_routed": Source(la, spec_a, "expert_routed"),
        "unrelated_dense": Source(la, unrel_a, "expert_dense"),
        "whole_residual": Source(la, spec_a, "resid"),
    }
    bs = cfg.bridge
    res["bridge"] = {}
    for name, src in variants.items():
        torch.manual_seed(0)
        d_src = A.cfg.d_model
        bridge = Bridge(d_src, B.cfg.d_model, bs.d_bus, bs.n_heads)
        sampler = MixedSampler(*_texts(cfg), L, p_recall=cfg.p_recall, seed=4242)
        hist = train_bridge(bridge, A, B, src, bs.layer_b, lambda n: [sampler.sample() for _ in range(n)],
                            tok_a, tok_b, L, steps=bs.steps, batch_size=bs.batch_size, lr=bs.lr,
                            answer_weight=bs.answer_weight,
                            log=lambda m: log(f"  {name} {m}"))
        scores = eval_bridge(bridge, A, B, src, bs.layer_b, ev)
        with torch.no_grad():
            h = src(A, g_a.ids)
            mask = causal_block_mask(b_end_chars(g_b), h.shape[1])
            with bridge.hook(B, bs.layer_b, h, mask):
                gen_loss = lm_loss(B(g_b.ids), g_b)
        res["bridge"][name] = {"eval": scores, "general_lm_loss_with_bridge": gen_loss,
                               "final_gate": float(torch.tanh(bridge.rx.gate.detach())), "train": hist}
        log(f"bridge {name}: " + " ".join(f"{c}={s['acc']:.3f}" for c, s in scores.items())
            + f" | general LM loss {gen_loss:.3f} (no bridge {res['baselines']['B_general_lm_loss']:.3f})")

    (out / "poc_results.json").write_text(json.dumps(res, indent=1))
    log(f"wrote {out / 'poc_results.json'}")


def _translate(W, h: torch.Tensor) -> torch.Tensor:
    flat = h.reshape(-1, h.shape[-1]).numpy()
    return torch.from_numpy(apply(W, flat).astype(np.float32)).reshape(*h.shape[:-1], W.d_out)


_TEXT_CACHE: dict = {}


def _texts(cfg):
    if "t" not in _TEXT_CACHE:
        s = setup.sampler(cfg, seed=0)
        _TEXT_CACHE["t"] = (s.prose, s.code)
    return _TEXT_CACHE["t"]


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
