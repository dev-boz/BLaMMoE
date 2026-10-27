"""Seed sweep over toy fused pairs (HANDOFF next step 2, audit M8): unit configs and the summary.

A *unit* is one pair of pretrained models with its own out_dir, laid out exactly like
results/toy_poc, so every single-run script works on it unchanged:

* main unit ``s<i>``: A_i (char) with B_i (BPE), seeds from ``seeds[i]``. The cross-family stand-in.
* Rung-2 unit ``r<i>``: B_{i+1} (as ``model_a``) with B_i. Same architecture, tokenizer and data,
  different seed: the friendliest partner. Its models are copies of the main units' B models.
* control unit ``<name><i>`` (``controls``): one of s_i's models kept (copied), the other pretrained
  again with ``override`` on top, e.g. B trained on A's data mix: a partner of the same size and
  architecture that knows the same domain, so a gain over it is not just extra capacity.

scripts/toy_sweep.py runs the units; this module builds their configs and reads their results
back. Every split-recall number is on the seed-2025 eval set, so per-document rows from different
variants of one unit pair up (experts minus residual is a paired bootstrap).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from blammoe.config import _from_dict
from blammoe.stats import bootstrap_ci, paired_bootstrap_diff, seed_summary

NEEDS = ("m1_needs_partner", "m2_needs_partner")  # m1 = the unit's model_a, m2 = model_b


@dataclass
class SweepConfig:
    base: str = "configs/toy_poc.yaml"  # model, fused and hide settings for every unit
    out_dir: str = "results/toy_sweep"
    seeds: list = field(default_factory=lambda: [[0, 1], [2, 3], [4, 5]])  # (A seed, B seed) per main unit
    variants: list = field(default_factory=lambda: ["ab_experts_all_rel_safe", "ab_resid_all_rel_safe"])
    rung2: bool = True  # also fuse B_{i+1} with B_i
    rung2_variants: list = field(default_factory=lambda: ["ab_experts_all_rel_safe", "ab_resid_all_rel_safe"])
    hide: bool = True  # attribution controls (scripts/toy_fused_hide.py) on every experts variant
    controls: list = field(default_factory=list)  # [{name, keep: model_a|model_b, override: {...}}], see units()
    overrides: dict = field(default_factory=dict)  # deep-merged into the base YAML (e.g. short smoke runs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepConfig":
        return _from_dict(cls, yaml.safe_load(Path(path).read_text()) or {})


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


@dataclass
class Unit:
    name: str
    kind: str  # "main" | "rung2" | "control:<name>" (across_seeds groups units by kind)
    cfg: dict  # a full toy_poc-style config with this unit's out_dir and seeds
    variants: list
    copy_models: dict = field(default_factory=dict)  # copied, not pretrained: {"model_a.pt": source path, ...}

    @property
    def out(self) -> Path:
        return Path(self.cfg["out_dir"])


def units(sc: SweepConfig, base: dict) -> list[Unit]:
    """Main units first: Rung-2 and control units copy models from them."""
    base = deep_merge(base, sc.overrides)
    root = Path(sc.out_dir)
    out = []
    for i, (sa, sb) in enumerate(sc.seeds):
        cfg = deep_merge(base, {"out_dir": str(root / f"s{i}"), "model_a": {"seed": sa}, "model_b": {"seed": sb}})
        out.append(Unit(f"s{i}", "main", cfg, list(sc.variants)))
    if sc.rung2 and len(sc.seeds) > 1:
        for i, (_, sb) in enumerate(sc.seeds):
            j = (i + 1) % len(sc.seeds)
            b = base["model_b"]
            cfg = deep_merge(base, {"out_dir": str(root / f"r{i}"), "model_a": {**b, "seed": sc.seeds[j][1]},
                                    "model_b": {**b, "seed": sb}})
            out.append(Unit(f"r{i}", "rung2", cfg, list(sc.rung2_variants),
                            {"model_a.pt": root / f"s{j}" / "model_b.pt", "model_b.pt": root / f"s{i}" / "model_b.pt"}))
    for c in sc.controls:
        if c["keep"] not in ("model_a", "model_b") or set(c["override"]) != {"model_a", "model_b"} - {c["keep"]}:
            raise ValueError(f"control {c['name']}: keep one model and override only the other, got {c}")
        for i, main in enumerate(out[: len(sc.seeds)]):
            cfg = deep_merge(main.cfg, {**c["override"], "out_dir": str(root / f"{c['name']}{i}")})
            out.append(Unit(f"{c['name']}{i}", f"control:{c['name']}", cfg, list(sc.variants),
                            {f"{c['keep']}.pt": main.out / f"{c['keep']}.pt"}))
    return out


# --------------------------------------------------------------------------- reading results back
def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _split(r: dict, per_doc: dict | None, k: str) -> dict:
    s = {"acc": r[k]["acc"], "exact": r[k]["exact"]}
    if per_doc:
        s["acc_ci"] = bootstrap_ci(per_doc[k])
        s["exact_ci"] = bootstrap_ci(per_doc[f"{k}_exact"])
    return s


def variant_summary(out: Path, variant: str) -> dict | None:
    """One variant of one unit: split recall (with CIs), ordinary text, audit gate, attribution."""
    fused = _read(out / f"fused_{variant}.json")
    if fused is None:
        return None
    sp = fused["split"]
    res = {"split": {}, "general": fused["general"]}
    if "domain" in fused:
        res["domain"] = domain_summary(fused["domain"])
    for cond in ("fused", "shuffled_partner", "solo"):
        pd = sp[cond].get("per_doc")
        res["split"][cond] = {k: _split(sp[cond], pd, k) for k in NEEDS}
    if "per_doc" in sp["fused"] and "per_doc" in sp["shuffled_partner"]:
        res["split"]["fused_minus_shuffled"] = {
            k: paired_bootstrap_diff(sp["fused"]["per_doc"][k], sp["shuffled_partner"]["per_doc"][k]) for k in NEEDS}
    audit = _read(out / "fused_audit.json")
    if audit and variant in audit:
        res["gate_ok"] = all(r["ok"] for r in audit[variant].values())
    hide = _read(out / f"fused_{variant}_hide.json")
    if hide:
        res["attribution"] = attribution(hide)
    return res


def domain_summary(domain: dict) -> dict:
    """Per domain (prose / code) and model: bits/char fused, with a shuffled partner and solo, each
    with a bootstrap CI over documents, and the paired differences fused minus solo and fused minus
    shuffled partner. Negative = the fused model needs fewer bits, i.e. the partner helps."""
    out = {}
    for d, r in domain.items():
        out[d] = {}
        for m in ("m1_bpc", "m2_bpc"):
            rows = {cond: r[cond]["per_doc"][m] for cond in ("fused", "shuffled_partner", "solo")}
            ci = {cond: bootstrap_ci(x) for cond, x in rows.items()}
            out[d][m] = {**{cond: {"bpc": c["acc"], "lo": c["lo"], "hi": c["hi"]} for cond, c in ci.items()},
                         "fused_minus_solo": paired_bootstrap_diff(rows["fused"], rows["solo"]),
                         "fused_minus_shuffled": paired_bootstrap_diff(rows["fused"], rows["shuffled_partner"])}
    return out


def attribution(hide: dict) -> dict:
    """Per direction (m1 reads m2 = A needs B): accuracy with nothing ablated, the chosen expert
    hidden / blanked, the worst other single expert hidden, every tag zeroed / shifted, the
    worst joint set hidden, and the recovery fine-tunes (after, and whether they passed the gate).
    Runs with causal selection also give the attention-share pick and its hide (``attention_*``)."""
    ab, out = hide["ablations"], {}
    for k, reader, (h, b, t, j) in (
        ("m1_needs_partner", "A_reads_B", ("hide_m2_expert_from_m1", "blank_m2_expert_for_m1", "tags_m2_for_m1",
                                            "joint_hide_m2_from_m1")),
        ("m2_needs_partner", "B_reads_A", ("hide_m1_expert_from_m2", "blank_m1_expert_for_m2", "tags_m1_for_m2",
                                            "joint_hide_m1_from_m2")),
    ):
        main = str(hide["chosen"][reader]["main"])
        others = [v for e, v in ab[h].items() if str(e) != main]
        out[k] = {"none": ab["none"][k], "chosen_expert": int(main), "hide_chosen": ab[h][main],
                  "blank_chosen": ab[b][main], "hide_worst_other": min(others) if others else None,
                  "tags_zero": ab[t]["zero"], "tags_shift": ab[t]["shift"],
                  "joint_worst": min(ab[j].values()) if ab[j] else None}
        if "attention" in hide["chosen"][reader]:
            att = str(hide["chosen"][reader]["attention"]["main"])
            out[k]["attention_expert"], out[k]["hide_attention"] = int(att), ab[h][att]
        for arm, r in hide.get("finetune", {}).items():
            out[k][f"finetune_{arm}"] = r["after"][k]
    out["finetune_gates_ok"] = all(r["gate"]["ok"] for r in hide.get("finetune", {}).values())
    return out


def unit_summary(unit: Unit) -> dict:
    res = {"kind": unit.kind, "seeds": [unit.cfg["model_a"]["seed"], unit.cfg["model_b"]["seed"]],
           "variants": {v: variant_summary(unit.out, v) for v in unit.variants}}
    vs = [v for v in unit.variants if res["variants"][v]]
    fused = {v: _read(unit.out / f"fused_{v}.json")["split"]["fused"] for v in vs}
    if len(vs) == 2 and all("per_doc" in f for f in fused.values()):  # e.g. experts minus residual
        a, b = vs
        res[f"{a}_minus_{b}"] = {k: paired_bootstrap_diff(fused[a]["per_doc"][k], fused[b]["per_doc"][k])
                                 for k in NEEDS}
    ens = _read(unit.out / "fused_ensemble.json")
    if ens:
        res["ensemble"] = ens
    return res


def across_seeds(summaries: dict[str, dict], kind: str) -> dict:
    """mean / sd / min / max over the units of one kind, per variant: fused, shuffled-partner and
    solo accuracy (digit and exact), and the chosen-expert hide and blank accuracy."""
    out: dict = {}
    for name, s in summaries.items():
        if s["kind"] != kind:
            continue
        for v, r in s["variants"].items():
            if not r:
                continue
            d = out.setdefault(v, {})
            for cond, ks in r["split"].items():
                if cond == "fused_minus_shuffled":
                    continue
                for k, x in ks.items():
                    for m in ("acc", "exact"):
                        d.setdefault(f"{cond}/{k}/{m}", []).append(x[m])
            for dom, ms in r.get("domain", {}).items():
                for m, x in ms.items():
                    for cond in ("fused", "shuffled_partner", "solo"):
                        d.setdefault(f"domain/{dom}/{m}/{cond}", []).append(x[cond]["bpc"])
                    for diff in ("fused_minus_solo", "fused_minus_shuffled"):
                        d.setdefault(f"domain/{dom}/{m}/{diff}", []).append(x[diff]["diff"])
            for k, a in r.get("attribution", {}).items():
                if isinstance(a, dict):
                    for m in ("hide_chosen", "blank_chosen", "tags_shift", "hide_attention"):
                        if m in a:
                            d.setdefault(f"{m}/{k}", []).append(a[m])
    return {v: {m: seed_summary(xs) for m, xs in d.items()} for v, d in out.items()}
