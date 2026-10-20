"""Cross-tokenizer causality gate for trained fused pairs: exits 1 if any prediction leaks.

For each variant, both models' full-text log-probs are compared with prefix-honest
re-runs (each position sees only the chars it has read; blammoe.toy.audit.causality_gate)
at EVERY scored position of the evaluation windows the results are reported on: the 96
split-recall windows (each model its own view) and the 48 ordinary-text windows. Expert
variants are also checked under the bus ablations the attribution controls use (one expert
hidden, one blanked, every tag shifted: scripts/toy_fused_hide.py). Any position
differing by more than ``tol`` fails the variant; the exit code is nonzero if any fails.

The check runs in float64. Causality is an exact property of the masks, but in float32 the
prefix re-runs have different shapes than the full run, so GPU rounding differs by ~1e-7 and can
flip a near-tied top-k routing choice; with expert-tagged buses that shows up as a ~1e-3
"leak" in BOTH directions on ~0.3% of positions. In float64 the safe pair matches to ~1e-15.
Upcasting trained float32 weights is exact, so this audits the same function.

    PYTHONPATH=. python scripts/toy_fused_audit.py configs/toy_poc.yaml [threads] [variant ...]

Writes <out_dir>/fused_audit.json. A variant trained without ``safe_mask`` is expected to
fail (that is the leak this guards against); its max_diff measures the leak.
"""
import json
import sys
from pathlib import Path

import torch

from blammoe.device import pick_device
from blammoe.toy import setup
from blammoe.toy.audit import causality_gate
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.fused import BusAblation, FusedPair
from blammoe.toy.fused_train import general_eval_set, split_eval_set
from blammoe.toy.moe import load

DEFAULT = ["ab_experts_all_rel_safe", "ab_resid_all_rel_safe"]
TOL = 1e-6  # float64: a safe pair agrees to ~1e-15, the old leak differs by ~0.1-1


def main(cfg_path: str, threads: int = 4, *variants: str) -> int:
    torch.set_num_threads(threads)
    device = pick_device()
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    fs, out, L = cfg.fused, Path(cfg.out_dir), cfg.seq_chars
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt", device)
    B, _ = load(out / "model_b.pt", device)
    base = setup.sampler(cfg, seed=0)
    docs = {"split": [(s.view_a, s.view_b) for s in split_eval_set(L)],
            "general": [(d.text, d.text) for d in general_eval_set(base.prose, base.code, L)]}
    report, failed = {}, []
    for name in variants or DEFAULT:
        v = fs.variants[name]
        if v["pair"] != "ab":
            raise ValueError(f"{name}: the gate is for cross-tokenizer pairs (pair: ab)")
        pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, v.get("signal", "moe"), v.get("rel_pos", False),
                         v.get("safe_mask", False)).to(device)
        pair.links.load_state_dict(torch.load(out / f"fused_{name}.pt", map_location=device))
        pair = pair.double().eval()  # see module docstring
        paths = {"normal": {}}
        if pair.signal == "experts":
            abl = BusAblation(hide=frozenset({0}), blank=frozenset({1}), tags="shift")
            paths["ablated"] = {"ablate": (abl, abl)}
        report[name] = {}
        for path, kw in paths.items():
            for kind, dd in docs.items():
                r = causality_gate(pair, dd, tok_a, tok_b, L, tol=TOL, **kw)
                report[name][f"{path}/{kind}"] = r
                print(f"== {name} {path}/{kind}: {'PASS' if r['ok'] else 'FAIL'} | "
                      + " | ".join(f"{s.upper()}: max {r[s]['max_diff']:.1e}, {r[s]['n_bad']}/{r[s]['n_positions']} bad"
                                   for s in ("a", "b")), flush=True)
                if not r["ok"]:
                    failed.append(f"{name} {path}/{kind}")
    (out / "fused_audit.json").write_text(json.dumps(report, indent=1))
    print("FAILED: " + ", ".join(failed) if failed else "ALL PASS", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 1 if len(args) > 1 and args[1].isdigit() else 0
    sys.exit(main(args[0], *(int(a) for a in args[1:1 + n]), *args[1 + n:]))
