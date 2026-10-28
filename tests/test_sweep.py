import json
import pytest

from blammoe.toy.sweep import SweepConfig, across_seeds, deep_merge, unit_summary, units

BASE = {"out_dir": "results/toy_poc", "steps": 4000,
        "model_a": {"tokenizer": "char", "d_model": 128, "seed": 0},
        "model_b": {"tokenizer": "bpe", "d_model": 96, "seed": 1},
        "fused": {"steps": 1500, "variants": {"x": {"pair": "ab", "steps": 3000}}}}


def test_deep_merge_overrides_nested_keys_only():
    m = deep_merge(BASE, {"steps": 3, "fused": {"variants": {"x": {"steps": 3}}}})
    assert m["steps"] == 3 and m["fused"]["variants"]["x"] == {"pair": "ab", "steps": 3}
    assert m["fused"]["steps"] == 1500 and BASE["fused"]["variants"]["x"]["steps"] == 3000  # base untouched


def test_units_main_then_rung2_with_b_models_from_the_main_units(tmp_path):
    sc = SweepConfig(out_dir=str(tmp_path), seeds=[[0, 1], [2, 3], [4, 5]], variants=["e", "r"],
                     rung2_variants=["e"], overrides={"steps": 3})
    us = units(sc, BASE)
    assert [u.name for u in us] == ["s0", "s1", "s2", "r0", "r1", "r2"]
    s1 = us[1].cfg
    assert (s1["model_a"]["seed"], s1["model_b"]["seed"], s1["steps"]) == (2, 3, 3)
    assert s1["out_dir"] == str(tmp_path / "s1") and s1["model_a"]["tokenizer"] == "char"
    r2 = us[5]
    assert r2.kind == "rung2" and r2.variants == ["e"]
    assert r2.cfg["model_a"] == {**BASE["model_b"], "seed": 1} and r2.cfg["model_b"]["seed"] == 5  # B_0 <-> B_2
    assert r2.copy_models == {"model_a.pt": tmp_path / "s0" / "model_b.pt", "model_b.pt": tmp_path / "s2" / "model_b.pt"}
    assert [u.name for u in units(SweepConfig(out_dir=str(tmp_path), seeds=[[0, 1]]), BASE)] == ["s0"]


def _split(acc1: float, acc2: float) -> dict:
    """4 documents of 4 answers (16 digits) each per direction, every doc at the same accuracy."""
    def rows(acc):
        return [[round(16 * acc), 16]] * 4, [[round(4 * acc), 4]] * 4

    (d1, e1), (d2, e2) = rows(acc1), rows(acc2)
    return {"m1_needs_partner": {"acc": acc1, "exact": acc1}, "m2_needs_partner": {"acc": acc2, "exact": acc2},
            "per_doc": {"m1_needs_partner": d1, "m1_needs_partner_exact": e1,
                        "m2_needs_partner": d2, "m2_needs_partner_exact": e2}}


def _write_unit(out, fused_acc: float, gate_ok: bool):
    out.mkdir(parents=True)
    for v, acc in (("e", fused_acc), ("r", 0.5)):
        (out / f"fused_{v}.json").write_text(json.dumps({
            "split": {"fused": _split(acc, acc), "shuffled_partner": _split(0.0, 0.0), "solo": _split(0.0, 0.0)},
            "general": {"fused": {"m1_bpc": 2.0, "m2_bpc": 2.1}}}))
    (out / "fused_audit.json").write_text(json.dumps({"e": {"normal/split": {"ok": gate_ok}}, "r": {"x": {"ok": True}}}))
    abl = {"none": {"m1_needs_partner": fused_acc, "m2_needs_partner": fused_acc},
           "hide_m2_expert_from_m1": {"0": 0.9, "1": 0.25}, "hide_m1_expert_from_m2": {"0": 0.1, "1": 0.95},
           "blank_m2_expert_for_m1": {"0": 0.9, "1": 0.3}, "blank_m1_expert_for_m2": {"0": 0.2, "1": 0.9},
           "tags_m2_for_m1": {"zero": 0.8, "shift": 0.7}, "tags_m1_for_m2": {"zero": 0.85, "shift": 0.75},
           "joint_hide_m2_from_m1": {}, "joint_hide_m1_from_m2": {"0+1": 0.05}}
    (out / "fused_e_hide.json").write_text(json.dumps({
        "chosen": {"A_reads_B": {"main": 1}, "B_reads_A": {"main": 0}}, "ablations": abl,
        "finetune": {"main": {"after": {"m1_needs_partner": 0.4, "m2_needs_partner": 0.3}, "gate": {"ok": True}}}}))


def test_unit_summary_reads_cis_gate_and_attribution_and_pairs_variants(tmp_path):
    sc = SweepConfig(out_dir=str(tmp_path), seeds=[[0, 1], [2, 3]], variants=["e", "r"], rung2=False)
    us = units(sc, BASE)
    _write_unit(us[0].out, 1.0, True)
    _write_unit(us[1].out, 0.75, False)
    s = unit_summary(us[0])
    e = s["variants"]["e"]
    assert e["gate_ok"] is True and s["variants"]["r"]["gate_ok"] is True
    ci = e["split"]["fused"]["m1_needs_partner"]["exact_ci"]
    assert ci["acc"] == ci["lo"] == ci["hi"] == 1.0 and ci["n_docs"] == 4
    assert e["split"]["fused_minus_shuffled"]["m2_needs_partner"]["diff"] == 1.0
    assert s["e_minus_r"]["m1_needs_partner"]["diff"] == 0.5  # paired over the same documents
    a = e["attribution"]
    assert a["m1_needs_partner"] == {"none": 1.0, "chosen_expert": 1, "hide_chosen": 0.25, "blank_chosen": 0.3,
                                     "hide_worst_other": 0.9, "tags_zero": 0.8, "tags_shift": 0.7,
                                     "joint_worst": None, "finetune_main": 0.4}
    assert a["m2_needs_partner"]["hide_chosen"] == 0.1 and a["m2_needs_partner"]["joint_worst"] == 0.05
    assert a["finetune_gates_ok"] is True
    assert unit_summary(us[1])["variants"]["e"]["gate_ok"] is False

    agg = across_seeds({u.name: unit_summary(u) for u in us}, "main")
    m = agg["e"]["fused/m1_needs_partner/acc"]
    assert m["n_seeds"] == 2 and abs(m["mean"] - 0.875) < 1e-9 and (m["min"], m["max"]) == (0.75, 1.0)
    assert agg["e"]["hide_chosen/m1_needs_partner"]["mean"] == 0.25
    assert "hide_chosen/m1_needs_partner" not in agg["r"]  # no attribution run for the residual variant
    assert across_seeds({u.name: unit_summary(u) for u in us}, "rung2") == {}


def test_attribution_reports_the_attention_pick_when_selection_is_causal(tmp_path):
    sc = SweepConfig(out_dir=str(tmp_path), seeds=[[0, 1]], variants=["e", "r"], rung2=False)
    [u] = units(sc, BASE)
    _write_unit(u.out, 1.0, True)
    path = u.out / "fused_e_hide.json"
    h = json.loads(path.read_text())
    h["chosen"]["A_reads_B"]["attention"] = {"main": 0}  # causal pick e1 (0.25), attention pick e0 (0.9)
    h["chosen"]["B_reads_A"]["attention"] = {"main": 1}
    path.write_text(json.dumps(h))
    a = unit_summary(u)["variants"]["e"]["attribution"]
    assert (a["m1_needs_partner"]["chosen_expert"], a["m1_needs_partner"]["hide_chosen"]) == (1, 0.25)
    assert (a["m1_needs_partner"]["attention_expert"], a["m1_needs_partner"]["hide_attention"]) == (0, 0.9)
    assert a["m2_needs_partner"]["hide_attention"] == 0.95
    agg = across_seeds({u.name: unit_summary(u)}, "main")
    assert agg["e"]["hide_attention/m1_needs_partner"]["mean"] == 0.9


def test_fused_seed_override_reaches_the_unit_config(tmp_path):
    from blammoe.config import _from_dict
    from blammoe.toy.config import ToyPOCConfig
    [s0] = units(SweepConfig(out_dir=str(tmp_path), seeds=[[0, 1]], overrides={"fused": {"seed": 1}}), BASE)
    assert _from_dict(ToyPOCConfig, s0.cfg).fused.seed == 1
    assert _from_dict(ToyPOCConfig, BASE).fused.seed == 0  # default: the pre-retry runs


CONTROLS = [{"name": "c", "keep": "model_a", "override": {"model_b": {"p_prose": 0.1}}},
            {"name": "d", "keep": "model_b", "override": {"model_a": {"p_prose": 0.9}}}]


def test_control_units_keep_one_main_model_and_retrain_the_other(tmp_path):
    sc = SweepConfig(out_dir=str(tmp_path), seeds=[[0, 1], [2, 3]], variants=["e"], rung2=False,
                     controls=CONTROLS, overrides={"model_a": {"p_prose": 0.1}, "model_b": {"p_prose": 0.9}})
    us = {u.name: u for u in units(sc, BASE)}
    assert list(us) == ["s0", "s1", "c0", "c1", "d0", "d1"]
    s1, c1, d0 = us["s1"], us["c1"], us["d0"]
    assert (s1.cfg["model_a"]["p_prose"], s1.cfg["model_b"]["p_prose"]) == (0.1, 0.9)
    assert c1.kind == "control:c" and c1.variants == ["e"] and c1.cfg["out_dir"] == str(tmp_path / "c1")
    assert c1.cfg["model_a"] == s1.cfg["model_a"]  # copied from s1, so the same spec
    assert c1.cfg["model_b"] == {**s1.cfg["model_b"], "p_prose": 0.1}  # B again, same seed, on A's mix
    assert c1.copy_models == {"model_a.pt": tmp_path / "s1" / "model_a.pt"}
    assert d0.cfg["model_a"] == {**us["s0"].cfg["model_a"], "p_prose": 0.9}
    assert d0.copy_models == {"model_b.pt": tmp_path / "s0" / "model_b.pt"}
    bad = SweepConfig(out_dir=str(tmp_path), seeds=[[0, 1]], controls=[{**CONTROLS[0], "keep": "model_b"}])
    with pytest.raises(ValueError):
        units(bad, BASE)  # keeping B while overriding B is not a control


def _domain(fused_bits: float) -> dict:
    """2 documents of 100 chars per condition, every document at the same bits/char."""
    rows = lambda b: [[100 * b, 100]] * 2  # noqa: E731
    cond = lambda b: {"m1_bpc": b, "m2_bpc": b + 1, "per_doc": {"m1_bpc": rows(b), "m2_bpc": rows(b + 1)}}  # noqa: E731
    return {"fused": cond(fused_bits), "shuffled_partner": cond(2.0), "solo": cond(2.5)}


def test_domain_summary_and_its_seed_aggregate(tmp_path):
    sc = SweepConfig(out_dir=str(tmp_path), seeds=[[0, 1], [2, 3]], variants=["e", "r"], rung2=False)
    us = units(sc, BASE)
    for u, b in zip(us, (1.5, 2.1)):
        _write_unit(u.out, 1.0, True)
        f = json.loads((u.out / "fused_e.json").read_text())
        f["domain"] = {"prose": _domain(b), "code": _domain(2.5)}
        (u.out / "fused_e.json").write_text(json.dumps(f))
    d = unit_summary(us[0])["variants"]["e"]["domain"]
    p = d["prose"]["m1_bpc"]
    assert p["fused"] == {"bpc": 1.5, "lo": 1.5, "hi": 1.5} and p["solo"]["bpc"] == 2.5
    assert p["fused_minus_solo"]["diff"] == -1.0 and p["fused_minus_shuffled"]["diff"] == -0.5
    assert d["prose"]["m2_bpc"]["fused"]["bpc"] == 2.5 and d["code"]["m1_bpc"]["fused_minus_solo"]["diff"] == 0.0
    assert "domain" not in unit_summary(us[0])["variants"]["r"]  # results from before per-domain eval
    agg = across_seeds({u.name: unit_summary(u) for u in us}, "main")["e"]
    assert abs(agg["domain/prose/m1_bpc/fused_minus_solo"]["mean"] - -0.7) < 1e-9
    assert agg["domain/prose/m1_bpc/solo"]["n_seeds"] == 2


def test_bpe_files_one_per_data_mix_fitted_once_in_the_sweep_root(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import toy_sweep
    from blammoe.toy.tokenizers import BPETokenizer
    fits, real_train = [], BPETokenizer.train.__func__
    monkeypatch.setattr(BPETokenizer, "train", classmethod(lambda cls, t, v: fits.append(1) or real_train(cls, t[:50], v)))
    sc = SweepConfig(out_dir=str(tmp_path), seeds=[[0, 1], [2, 3]], variants=["e"], rung2=True, controls=CONTROLS[:1],
                     overrides={"model_a": {"p_prose": 0.1}, "model_b": {"p_prose": 0.9}})
    files = {u.name: [p.name for p in toy_sweep.bpe_files(u, tmp_path)] for u in units(sc, BASE)}
    p9, p1 = "bpe1024_train_v0.1_t0.1_p0.9.json", "bpe1024_train_v0.1_t0.1_p0.1.json"
    assert files == {"s0": [p9], "s1": [p9], "r0": [p9, p9], "r1": [p9, p9], "c0": [p1], "c1": [p1]}
    assert len(fits) == 2 and sorted(p.name for p in tmp_path.glob("bpe*.json")) == [p1, p9]


def test_models_from_copies_every_units_models_and_bpe_from_another_sweep(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import toy_sweep
    from blammoe.toy.tokenizers import BPETokenizer
    old = tmp_path / "old"
    sc = SweepConfig(out_dir=str(tmp_path / "new"), seeds=[[0, 1]], variants=["e"], rung2=False, controls=CONTROLS[:1],
                     models_from=str(old), overrides={"model_b": {"p_prose": 0.9}})
    us = {u.name: u for u in units(sc, BASE)}
    assert us["s0"].copy_models == {"model_a.pt": old / "s0" / "model_a.pt", "model_b.pt": old / "s0" / "model_b.pt"}
    assert us["c0"].copy_models == {"model_a.pt": old / "c0" / "model_a.pt", "model_b.pt": old / "c0" / "model_b.pt"}
    # the old sweep's BPE is reused, never refitted (token ids must match the copied models)
    old.mkdir()
    (old / "bpe1024_train_v0.1_t0.1_p0.9.json").write_text("{}")
    monkeypatch.setattr(BPETokenizer, "load", classmethod(lambda cls, p: "loaded"))
    monkeypatch.setattr(BPETokenizer, "train", classmethod(lambda cls, t, v: (_ for _ in ()).throw(AssertionError("refit"))))
    monkeypatch.setattr(toy_sweep, "run_unit", lambda u, bpes, threads, hide: [])
    monkeypatch.setattr(toy_sweep, "unit_summary", lambda u: None)
    sc_path = tmp_path / "dial.yaml"
    import yaml
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(BASE))
    sc_path.write_text(yaml.safe_dump({"base": str(base_path), "out_dir": sc.out_dir, "seeds": [[0, 1]], "variants": ["e"],
                                       "rung2": False, "models_from": str(old), "overrides": sc.overrides}))
    toy_sweep.main(str(sc_path), 1, "none")
    assert (Path(sc.out_dir) / "bpe1024_train_v0.1_t0.1_p0.9.json").read_text() == "{}"
