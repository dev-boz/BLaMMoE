import json

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
