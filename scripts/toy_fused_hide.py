"""Attribution controls for a translator-routed fusion (audit M3): does the information
travel in the chosen experts' OUTPUT?

Each control ablates one model's expert bus as its partner reads it, at every link, and scores
the partner's split-recall answers that need it (blammoe.toy.fused_train.hide_ablation):

* hide  - one expert at a time unreadable (the original ablation);
* blank - one expert at a time keeps its expert-ID tag, slot and mask but sends a constant, its
          mean output on the selection set (content-only ablation);
* tags  - every tag zeroed, or shifted to the wrong expert, outputs kept (ID-only ablation);
* joint - each link's most-read experts hidden together (B<-A's leftover 29.1% ran through
          A expert 0 at links 0/1, which hiding the one chosen expert left open);
* recovery fine-tune - the links retrained briefly (``hide:`` in the YAML) with the chosen experts
          hidden in both directions. The reader never saw them missing in training, so a collapse
          may only mean its attention broke; if accuracy comes back through other experts, the
          information is not exclusive to them. The same fine-tune with nothing hidden is the
          control. Every fine-tuned copy must pass the float64 causality gate on the windows it
          is scored on, else the script exits 1.

"Chosen" experts come from fused_<variant>_experts.json (run scripts/toy_fused_experts.py first),
measured on the seed-2026 selection set, as are the blank constants. Everything reported here is
on the seed-2025 eval set. Writes <out_dir>/fused_<variant>_hide.json (+ fused_<variant>_ft_<arm>.pt).

    PYTHONPATH=. python scripts/toy_fused_hide.py configs/toy_poc.yaml <variant> [threads]
"""
import copy
import json
import random
import sys
from pathlib import Path

import torch

from blammoe.device import pick_device
from blammoe.toy import setup
from blammoe.toy.audit import causality_gate
from blammoe.toy.batching import make_batch
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import split_recall_sample
from blammoe.toy.fused import NO_ABLATION, BusAblation, FusedPair, expert_output_means
from blammoe.toy.fused_train import (
    chosen_experts, hide_ablation, mixed_draw, recovery_finetune, split_eval_set,
)
from blammoe.toy.moe import load

TOL = 1e-6  # causality gate, float64 (see scripts/toy_fused_audit.py)


def fmt(experts) -> str:
    return "{" + ",".join(str(e) for e in sorted(experts)) + "}"


def main(cfg_path: str, variant: str, threads: int = 4) -> int:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    device = pick_device()
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    fs, hs, out, L = cfg.fused, cfg.hide, Path(cfg.out_dir), cfg.seq_chars
    v = fs.variants[variant]
    assert v["pair"] == "ab" and v["signal"] == "experts"
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt", device)
    B, _ = load(out / "model_b.pt", device)
    pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, "experts", v.get("rel_pos", False),
                     v.get("safe_mask", False)).to(device)
    pair.links.load_state_dict(torch.load(out / f"fused_{variant}.pt", map_location=device))
    pair.eval()

    # selection set (seed 2026, as toy_fused_experts.py): chosen experts and blank constants
    sel = json.loads((out / f"fused_{variant}_experts.json").read_text())["split_needs_partner"]
    ch_b, ch_a = chosen_experts(sel, "m1_reads_m2"), chosen_experts(sel, "m2_reads_m1")  # B's read by A, A's by B
    rng = random.Random(2026)
    sel_docs = [split_recall_sample(rng, L) for _ in range(48)]
    means = expert_output_means(pair, make_batch([s.view("a") for s in sel_docs], tok_a, L, device=device),
                                make_batch([s.view("b") for s in sel_docs], tok_b, L, device=device))
    print(f"chosen: A reads B e{ch_b['main']} (joint {[fmt(s) for s in ch_b['joint']]}) | "
          f"B reads A e{ch_a['main']} (joint {[fmt(s) for s in ch_a['joint']]})", flush=True)

    splits = split_eval_set(L)  # seed 2025, the reported set
    res = {"variant": variant,
           "chosen": {"A_reads_B": {**ch_b, "joint": [sorted(s) for s in ch_b["joint"]]},
                      "B_reads_A": {**ch_a, "joint": [sorted(s) for s in ch_a["joint"]]}}}
    res["ablations"] = hide_ablation(pair, splits, tok_a, tok_b, L, joint_m1=ch_a["joint"],
                                     joint_m2=ch_b["joint"], blank_values=means)
    ab = res["ablations"]
    print(f"nothing ablated: A needs B {ab['none']['m1_needs_partner']:.3f} | "
          f"B needs A {ab['none']['m2_needs_partner']:.3f}", flush=True)
    for e in range(B.cfg.n_experts):
        print(f"B e{e:<2} read by A: hide {ab['hide_m2_expert_from_m1'][e]:.3f} | "
              f"blank {ab['blank_m2_expert_for_m1'][e]:.3f}", flush=True)
    for e in range(A.cfg.n_experts):
        print(f"A e{e:<2} read by B: hide {ab['hide_m1_expert_from_m2'][e]:.3f} | "
              f"blank {ab['blank_m1_expert_for_m2'][e]:.3f}", flush=True)
    for d, k in (("B's tags for A", "tags_m2_for_m1"), ("A's tags for B", "tags_m1_for_m2")):
        print(f"{d}: " + " | ".join(f"{t} {x:.3f}" for t, x in ab[k].items()), flush=True)
    for d, k in (("B experts hidden from A", "joint_hide_m2_from_m1"), ("A experts hidden from B", "joint_hide_m1_from_m2")):
        for s, x in ab[k].items():
            print(f"joint: {d} {s}: {x:.3f}", flush=True)

    failed = []
    if hs.finetune_steps:
        base = setup.sampler(cfg, seed=0)
        gate_docs = [(s.view_a, s.view_b) for s in splits]
        arms = {"none": (NO_ABLATION, NO_ABLATION),
                "main": (BusAblation(hide=frozenset({ch_a["main"]})), BusAblation(hide=frozenset({ch_b["main"]})))}
        top1_a, top1_b = (frozenset(t[0] for t in c["per_link"].values()) for c in (ch_a, ch_b))
        if (top1_a, top1_b) != (frozenset({ch_a["main"]}), frozenset({ch_b["main"]})):
            arms["links_top1"] = (BusAblation(hide=top1_a), BusAblation(hide=top1_b))
        res["finetune"] = {}
        for arm, abl in arms.items():
            print(f"-- recovery fine-tune '{arm}': hide A {fmt(abl[0].hide)} from B, B {fmt(abl[1].hide)} from A, "
                  f"{hs.finetune_steps} steps", flush=True)
            ft, r = recovery_finetune(pair, abl, tok_a, tok_b, mixed_draw(base.prose, base.code, L, fs.p_split, seed=4242),
                                      splits, L, steps=hs.finetune_steps, batch_size=fs.batch_size, lr=hs.lr,
                                      answer_weight=fs.answer_weight, warmup=hs.finetune_warmup,
                                      log=lambda m: print("   ", m, flush=True))
            torch.save(ft.links.state_dict(), out / f"fused_{variant}_ft_{arm}.pt")
            gate = causality_gate(copy.deepcopy(ft).double().eval(), gate_docs, tok_a, tok_b, L, tol=TOL, ablate=abl)
            r["hidden"] = {"A_from_B": sorted(abl[0].hide), "B_from_A": sorted(abl[1].hide)}
            r["gate"] = {"ok": gate["ok"], **{s: gate[s] for s in ("a", "b")}}
            res["finetune"][arm] = r
            print(f"   A needs B {r['before']['m1_needs_partner']:.3f} -> {r['after']['m1_needs_partner']:.3f} | "
                  f"B needs A {r['before']['m2_needs_partner']:.3f} -> {r['after']['m2_needs_partner']:.3f} | "
                  f"gate {'PASS' if gate['ok'] else 'FAIL'}", flush=True)
            if not gate["ok"]:
                failed.append(arm)
    (out / f"fused_{variant}_hide.json").write_text(json.dumps(res, indent=1, default=str))
    if failed:
        print("CAUSALITY GATE FAILED: " + ", ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:])))
