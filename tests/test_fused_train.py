"""fused_train: training moves the links, evals return well-formed results (CPU, tiny)."""
import random

import torch

from blammoe.toy.corpus import Sample, recall_sample, split_recall_sample
from blammoe.toy.batching import make_batch
from blammoe.toy.fused import BusAblation, FusedPair, expert_output_means
from blammoe.toy.fused_train import (
    causal_chosen, chosen_experts, domain_eval_sets, ensemble_baseline, ensemble_bits, general_bpc, hide_ablation,
    mixed_draw, recovery_finetune, split_eval_set, split_scores,
    train_fused,
)
from blammoe.toy.moe import ToyMoEConfig, ToyMoELM
from blammoe.toy.tokenizers import BPETokenizer, CharTokenizer

L = 100
PROSE = "the model reads text and predicts the next character. " * 40
CODE = "def f(x):\n    return x + 1\n" * 60


def tiny(vocab, d, experts, seed):
    torch.manual_seed(seed)
    return ToyMoELM(ToyMoEConfig(vocab_size=vocab, d_model=d, n_layers=2, n_heads=2, n_experts=experts,
                                 top_k=2, d_expert=16, max_len=128)).eval()


def pair_and_toks():
    ta = CharTokenizer()
    tb = BPETokenizer.train([recall_sample(random.Random(i), 200).text for i in range(30)] + [PROSE, CODE],
                            vocab_size=200)
    pair = FusedPair(tiny(ta.vocab_size, 16, 4, 0), tiny(tb.vocab_size, 12, 6, 1), [0, 1], d_bus=8, n_heads=2,
                     signal="experts", rel_pos=True, safe_mask=True)
    return pair, ta, tb


def test_train_fused_opens_gates_and_evals_are_well_formed():
    pair, ta, tb = pair_and_toks()
    hist = train_fused(pair, ta, tb, mixed_draw(PROSE, CODE, L, 0.5, seed=1), L, steps=4, batch_size=2,
                       lr=1e-2, answer_weight=1.0, log=lambda m: None, log_every=2)
    assert hist and all(abs(g) > 0 for lk in hist[-1]["gates"].values() for g in lk)
    splits = split_eval_set(L, n=3, seed=5)
    r = split_scores(pair, splits, ta, tb, L, per_doc=True)
    assert 0 <= r["m1_needs_partner"]["acc"] <= 1
    per = r["per_doc"]["m1_needs_partner"]
    assert len(per) == 3 and sum(t for _, t in per) == r["m1_needs_partner"]["n"]
    assert sum(c for c, _ in per) == round(r["m1_needs_partner"]["acc"] * r["m1_needs_partner"]["n"])
    ex = r["per_doc"]["m1_needs_partner_exact"]
    assert sum(t for _, t in ex) == r["m1_needs_partner"]["n_answers"]
    docs = [split_recall_sample(random.Random(9), L).view("a")]
    bpc = general_bpc(pair, docs, ta, tb, L)
    assert bpc["m1_bpc"] > 0 and bpc["m2_bpc"] > 0


def test_ensemble_baseline_and_hide_ablation_shapes():
    pair, ta, tb = pair_and_toks()
    splits = split_eval_set(L, n=3, seed=6)
    docs = [s.view("a") for s in split_eval_set(L, n=2, seed=7)]
    ens = ensemble_baseline(pair, docs, splits, ta, tb, L)
    g = ens["general_bits"]
    assert ens["general_shared_positions"] > 0 and g["mixture"] > 0
    # a 50/50 mixture can never be worse than the worse of its two parts (log is concave)
    assert g["mixture"] <= max(g["m1_solo"], g["m2_solo"]) + 1e-6
    assert 0 <= ens["split_m1_needs_partner_acc"]["mixture"] <= 1
    hid = hide_ablation(pair, splits, ta, tb, L)
    assert len(hid["hide_m2_expert_from_m1"]) == pair.B.cfg.n_experts
    assert len(hid["hide_m1_expert_from_m2"]) == pair.A.cfg.n_experts


def test_hide_ablation_attribution_controls():
    """M3: blank (content gone, tag kept), tag (ids zeroed/shifted, content kept) and joint
    hiding, each scored in one direction with the other bus untouched."""
    pair, ta, tb = pair_and_toks()
    for lk in pair.links.values():
        with torch.no_grad():
            lk.rx_a.gate.fill_(1.0)
            lk.rx_b.gate.fill_(1.0)
    splits = split_eval_set(L, n=3, seed=6)
    sel = split_eval_set(L, n=3, seed=8)
    means = expert_output_means(pair, make_batch([s.view("a") for s in sel], ta, L),
                                make_batch([s.view("b") for s in sel], tb, L))
    hid = hide_ablation(pair, splits, ta, tb, L, blank_values=means,
                        joint_m1=[frozenset({0, 1})], joint_m2=[frozenset({0, 2}), frozenset(range(6))])
    n1, n2 = pair.A.cfg.n_experts, pair.B.cfg.n_experts
    assert len(hid["blank_m2_expert_for_m1"]) == n2 and len(hid["blank_m1_expert_for_m2"]) == n1
    assert set(hid["tags_m2_for_m1"]) == {"zero", "shift"} == set(hid["tags_m1_for_m2"])
    assert set(hid["joint_hide_m2_from_m1"]) == {"0+2", "0+1+2+3+4+5"} and set(hid["joint_hide_m1_from_m2"]) == {"0+1"}
    # every partner expert hidden: model 1 reads only the null slot, so which partner it has
    # cannot matter (not the same as links off: the null read still adds the attention biases)
    wrong = split_scores(pair, splits, ta, tb, L, partner_roll=1,
                         ablate=(BusAblation(), BusAblation(hide=frozenset(range(6)))))
    assert hid["joint_hide_m2_from_m1"]["0+1+2+3+4+5"] == wrong["m1_needs_partner"]["acc"]
    # each control is scored in one direction: the m1 numbers match a run with only m2's bus ablated
    one = split_scores(pair, splits, ta, tb, L, ablate=(BusAblation(), BusAblation(tags="shift")))
    assert hid["tags_m2_for_m1"]["shift"] == one["m1_needs_partner"]["acc"]
    for k in ("blank_m2_expert_for_m1", "tags_m1_for_m2", "joint_hide_m1_from_m2"):
        assert all(0 <= v <= 1 for v in hid[k].values())


def test_recovery_finetune_trains_a_copy_under_the_ablation():
    pair, ta, tb = pair_and_toks()
    before = {k: v.clone() for k, v in pair.links.state_dict().items()}
    splits = split_eval_set(L, n=3, seed=5)
    abl = (BusAblation(hide=frozenset({1})), BusAblation(hide=frozenset({2})))
    seen = []
    orig = FusedPair.forward

    def spy(self, *a, **kw):
        seen.append(kw.get("ablate"))
        return orig(self, *a, **kw)

    FusedPair.forward = spy
    try:
        ft, r = recovery_finetune(pair, abl, ta, tb, mixed_draw(PROSE, CODE, L, 0.5, seed=1), splits, L,
                                  steps=3, batch_size=2, lr=1e-2, answer_weight=1.0, warmup=2, log=lambda m: None)
    finally:
        FusedPair.forward = orig
    assert seen and all(s == abl for s in seen)  # every train AND eval forward had the expert hidden
    assert all(torch.equal(v, pair.links.state_dict()[k]) for k, v in before.items())  # original untouched
    assert any(not torch.equal(v, ft.links.state_dict()[k]) for k, v in before.items())
    assert set(r) == {"before", "after", "train"} and 0 <= r["after"]["m1_needs_partner"] <= 1


def test_chosen_experts_reads_the_selection_set_attention_shares():
    # B<-A pattern from the audit: expert 0 at links 0/1, expert 5 at the last links
    shares = {"0": [0.6, 0.1, 0.1, 0.0, 0.0, 0.2], "1": [0.5, 0.0, 0.3, 0.0, 0.0, 0.2],
              "2": [0.1, 0.0, 0.0, 0.1, 0.0, 0.8], "3": [0.0, 0.0, 0.0, 0.2, 0.1, 0.7]}
    per_link = {l: {"m2_reads_m1": {"share": s}, "m1_reads_m2": {"share": [1.0] + [0.0] * 5}} for l, s in shares.items()}
    c = chosen_experts(per_link, "m2_reads_m1")
    assert c["main"] == 5  # 0.2+0.2+0.8+0.7 beats expert 0's 1.2
    assert c["per_link"] == {0: [0, 5], 1: [0, 2], 2: [5, 0], 3: [5, 3]}
    assert c["joint"] == [frozenset({0, 5}), frozenset({0, 2, 3, 5})]
    one = chosen_experts(per_link, "m1_reads_m2")
    assert one["main"] == 0 and one["joint"] == [frozenset({0, 1})]  # top-1 everywhere is e0: no 1-set


def test_causal_chosen_ranks_partner_experts_by_the_damage_hiding_them_does():
    pair, ta, tb = pair_and_toks()
    for lk in pair.links.values():
        with torch.no_grad():
            lk.rx_a.gate.fill_(1.0)
            lk.rx_b.gate.fill_(1.0)
    sel = split_eval_set(L, n=3, seed=8)
    ch_m2, ch_m1 = causal_chosen(pair, sel, ta, tb, L)
    hid = hide_ablation(pair, sel, ta, tb, L)
    assert ch_m2["acc"] == hid["hide_m2_expert_from_m1"] and ch_m1["acc"] == hid["hide_m1_expert_from_m2"]
    for ch in (ch_m2, ch_m1):
        order = sorted(ch["acc"], key=lambda e: (ch["acc"][e], e))
        assert ch["main"] == order[0] and ch["acc"][ch["main"]] == min(ch["acc"].values())
        assert ch["joint"] == [frozenset(order[:2]), frozenset(order[:3])]


def test_exact_answer_accuracy_needs_every_digit_of_an_answer():
    """Two 4-digit answers; one digit wrong in the first -> digits 7/8 but answers 1/2."""
    from blammoe.toy.batching import answer_runs
    from blammoe.toy.bridge import score_answers
    from blammoe.toy.fused import subset_accuracy

    assert answer_runs([12, 3, 4, 5, 6, 10, 11, 13]) == [[3, 4, 5, 6], [10, 11, 12, 13]]
    tok = CharTokenizer()
    text = "store: ab=1234, cd=5678; ask: ab=1234, cd=5678\n"
    s0 = text.index("ask: ab=") + len("ask: ab=")
    s1 = text.index(", cd=", s0) + len(", cd=")
    chars = tuple(range(s0, s0 + 4)) + tuple(range(s1, s1 + 4))
    b = make_batch([Sample(text=text, blind=text, answer_chars=chars)], tok, len(text))
    logits = torch.full((*b.targets.shape, tok.vocab_size), -1e4)
    logits.scatter_(-1, b.targets.clamp_min(0)[..., None], 0.0)  # predict every target...
    wrong = b.answer_pos[0][1]
    logits[0, wrong] = 0.0
    logits[0, wrong, (b.targets[0, wrong] + 1) % tok.vocab_size] = 1.0  # ...except one digit of answer 1
    r = subset_accuracy(logits, b, [chars])
    assert (r["n"], r["acc"], r["n_answers"], r["exact"]) == (8, 7 / 8, 2, 0.5)
    s = score_answers(logits, b)
    assert (s["n"], s["acc"], s["n_answers"], s["exact"]) == (8, 7 / 8, 2, 0.5)


def test_ordinary_text_only_links_and_per_domain_eval():
    """Complementary-models pieces: p_split 0 draws only ordinary text (both models see the same
    window), per-domain eval sets, per-document bits that pool back to bits/char, and the
    per-domain ensemble."""
    draw = mixed_draw(PROSE, CODE, L, 0.0, seed=2)
    items = draw(40)
    assert all(a is b and a.kind in ("prose", "code") for a, b in items)
    assert {a.kind for a, _ in items} == {"prose", "code"}

    sets = domain_eval_sets(PROSE, CODE, L, n=3, seed=4)
    assert [d.kind for d in sets["prose"]] == ["prose"] * 3 and [d.kind for d in sets["code"]] == ["code"] * 3
    assert [d.text for d in domain_eval_sets(PROSE, CODE, L, n=3, seed=4)["code"]] == [d.text for d in sets["code"]]

    pair, ta, tb = pair_and_toks()
    docs = sets["prose"] + sets["code"]
    r = general_bpc(pair, docs, ta, tb, L, per_doc=True)
    for m in ("m1_bpc", "m2_bpc"):
        rows = r["per_doc"][m]
        assert len(rows) == 6 and abs(sum(b for b, _ in rows) / sum(c for _, c in rows) - r[m]) < 1e-6
    assert set(general_bpc(pair, docs, ta, tb, L)) == {"m1_bpc", "m2_bpc"}  # per_doc off by default

    e = ensemble_bits(pair, docs, ta, tb, L)
    assert e["shared_positions"] > 0 and set(e["bits"]) == {"m1_solo", "m2_solo", "mixture"}
    base = ensemble_baseline(pair, docs, split_eval_set(L, n=2, seed=6), ta, tb, L)
    assert base["general_bits"] == e["bits"] and base["general_shared_positions"] == e["shared_positions"]
