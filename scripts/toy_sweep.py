"""Resumable seed sweep over toy fused pairs (HANDOFF next step 2, audit M8).

For every unit (blammoe/toy/sweep.py: main units A_i<->B_i, Rung-2 units B_{i+1}<->B_i) this runs,
in its own out_dir laid out like results/toy_poc, the same scripts a single run uses:

  pretrain (main units; Rung-2 units copy the main units' B models)
  -> scripts/toy_fused.py for each variant (split recall per document, ordinary text on test)
  -> scripts/toy_fused_audit.py on all of the unit's variants (float64 gate; a failure is recorded)
  -> scripts/toy_fused_experts.py + toy_fused_hide.py for each experts variant (attribution, M3)
  -> the no-communication ensemble baseline (fused_ensemble.json)

A job whose output exists is skipped, so rerunning resumes. A unit's config is written to
<unit>/config.yaml; a rerun with a different config for the same unit stops (use a new out_dir).
A failing job stops its unit and the sweep moves to the next. Finally <out_dir>/summary.json:
per unit and variant, bootstrap CIs over documents (digit and exact-answer accuracy, fused minus
shuffled partner, experts minus residual), the gate, attribution; across seeds, mean and sd.
Exits 1 if any job or gate failed.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \\
        python scripts/toy_sweep.py configs/toy_sweep.yaml [threads] [unit ...]
"""
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the single-run scripts below

import toy_fused  # noqa: E402
import toy_fused_audit  # noqa: E402
import toy_fused_experts  # noqa: E402
import toy_fused_hide  # noqa: E402
import toy_train  # noqa: E402

from blammoe.config import _from_dict  # noqa: E402
from blammoe.device import pick_device  # noqa: E402
from blammoe.toy import setup  # noqa: E402
from blammoe.toy.config import ToyPOCConfig  # noqa: E402
from blammoe.toy.fused import FusedPair  # noqa: E402
from blammoe.toy.fused_train import ensemble_baseline, general_eval_set, split_eval_set  # noqa: E402
from blammoe.toy.moe import load  # noqa: E402
from blammoe.toy.sweep import SweepConfig, Unit, across_seeds, unit_summary, units  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[sweep {time.time() - T0:7.0f}s] {msg}", flush=True)


def ensemble(cfg_path: str, variant: str, threads: int) -> None:
    """No-communication baseline: the two solo models' next-char mixture (links all off)."""
    torch.set_num_threads(threads)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    out, L, fs, device = Path(cfg.out_dir), cfg.seq_chars, cfg.fused, pick_device()
    v = fs.variants[variant]
    tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
    A, _ = load(out / "model_a.pt", device)
    B, _ = load(out / "model_b.pt", device)
    pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, v.get("signal", "moe"), v.get("rel_pos", False),
                     v.get("safe_mask", False)).to(device).eval()
    res = ensemble_baseline(pair, general_eval_set(*setup.texts(cfg, "test"), L), split_eval_set(L), tok_a, tok_b, L)
    (out / "fused_ensemble.json").write_text(json.dumps(res, indent=1))


def prepare(unit: Unit, bpe: Path) -> str:
    """Write the unit's config (refusing a changed one) and give it the shared BPE file."""
    unit.out.mkdir(parents=True, exist_ok=True)
    path = unit.out / "config.yaml"
    text = yaml.safe_dump(unit.cfg, sort_keys=False)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"{path} differs from this sweep's config for {unit.name}: use a new out_dir")
    path.write_text(text)
    if not (unit.out / bpe.name).exists():
        shutil.copyfile(bpe, unit.out / bpe.name)
    return str(path)


def jobs(unit: Unit, cfg_path: str, threads: int, do_hide: bool):
    """(name, done?, run) in order; run returns a nonzero code on a gate failure."""
    out, variants = unit.out, unit.variants
    spec = ToyPOCConfig.from_yaml(cfg_path).fused.variants
    for m in ("model_a", "model_b"):
        if unit.kind == "rung2":
            src = unit.copy_models[f"{m}.pt"]
            yield f"copy {m} <- {src}", (out / f"{m}.pt").exists(), lambda s=src, m=m: shutil.copyfile(s, out / f"{m}.pt")
        else:
            yield f"pretrain {m}", (out / f"{m}.pt").exists(), lambda m=m: toy_train.main(cfg_path, m, threads)
    for v in variants:
        yield f"fuse {v}", (out / f"fused_{v}.pt").exists(), lambda v=v: toy_fused.main(cfg_path, v, threads)
    audited = json.loads((out / "fused_audit.json").read_text()) if (out / "fused_audit.json").exists() else {}
    yield ("audit gate", set(variants) <= set(audited), lambda: toy_fused_audit.main(cfg_path, threads, *variants))
    for v in variants:
        if do_hide and spec[v].get("signal") == "experts":
            yield (f"experts {v}", (out / f"fused_{v}_experts.json").exists(),
                   lambda v=v: toy_fused_experts.main(cfg_path, v, threads))
            yield f"hide {v}", (out / f"fused_{v}_hide.json").exists(), lambda v=v: toy_fused_hide.main(cfg_path, v, threads)
    yield "ensemble", (out / "fused_ensemble.json").exists(), lambda: ensemble(cfg_path, variants[0], threads)


def run_unit(unit: Unit, bpe: Path, threads: int, do_hide: bool) -> list[str]:
    failures = []
    try:
        cfg_path = prepare(unit, bpe)
        for name, done, run in jobs(unit, cfg_path, threads, do_hide):
            if done:
                log(f"{unit.name}: {name} - done, skipped")
                continue
            t = time.time()
            log(f"{unit.name}: {name} ...")
            code = run()
            log(f"{unit.name}: {name} finished in {time.time() - t:.0f}s" + (f" (exit {code})" if code else ""))
            if code:
                failures.append(f"{unit.name} {name}: exit {code}")
    except Exception:
        log(f"{unit.name}: FAILED\n{traceback.format_exc()}")
        failures.append(f"{unit.name}: exception (see log)")
    return failures


def main(sweep_path: str, threads: int = 4, *only: str) -> int:
    sc = SweepConfig.from_yaml(sweep_path)
    base = yaml.safe_load(Path(sc.base).read_text())
    all_units = units(sc, base)
    root = Path(sc.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    # one BPE for every unit: it is fitted on train text only and does not depend on any seed
    tok_cfg = _from_dict(ToyPOCConfig, {**all_units[0].cfg, "out_dir": str(root)})
    setup.tokenizer(tok_cfg.model_b, tok_cfg)
    bpe = next(root.glob(f"bpe{tok_cfg.model_b.bpe_vocab}_train_*.json"))
    log(f"{len(all_units)} units in {root} on {pick_device()}, {threads} threads; BPE {bpe.name}")

    failures = []
    for u in all_units:
        if not only or u.name in only:
            failures += run_unit(u, bpe, threads, sc.hide)

    summaries = {u.name: unit_summary(u) for u in all_units if (u.out / "config.yaml").exists()}
    summary = {"units": summaries, "across_seeds": {k: across_seeds(summaries, k) for k in ("main", "rung2")},
               "failures": failures}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, s in summaries.items():
        for v, r in s["variants"].items():
            if not r:
                continue
            f = r["split"]["fused"]
            gate = {True: "PASS", False: "FAIL"}.get(r.get("gate_ok"), "n/a")
            log(f"{name} {v}: A needs B {f['m1_needs_partner']['acc']:.3f} (exact {f['m1_needs_partner']['exact']:.3f}) | "
                f"B needs A {f['m2_needs_partner']['acc']:.3f} (exact {f['m2_needs_partner']['exact']:.3f}) | gate {gate}")
            for gs, gr in r["general"].items():
                log(f"    general {gs}: " + " ".join(f"{n}={x:.4f}" for n, x in gr.items()))
    gates = [f"{n} {v}" for n, s in summaries.items() for v, r in s["variants"].items() if r and r.get("gate_ok") is False]
    if gates:
        failures.append("audit gate FAILED: " + ", ".join(gates))
    log("FAILURES:\n  " + "\n  ".join(failures) if failures else f"all done, summary in {root / 'summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 1 if len(args) > 1 and args[1].isdigit() else 0
    sys.exit(main(args[0], *(int(a) for a in args[1:1 + n]), *args[1 + n:]))
