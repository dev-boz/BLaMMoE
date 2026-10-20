"""WP-T toy POC: corpus invariants, tokenizer offsets, the MoE hook point, causal
cross-tokenizer alignment, and no-op guarantees of both carriers. All tiny/fast."""
import random

import pytest
import torch

from blammoe.hooks import ExpertCapture, expert_path
from blammoe.toy.batching import answer_mask, answer_positions, make_batch
from blammoe.toy.bridge import (
    Bridge, Source, b_end_chars, causal_block_mask, encode_pair, gather_aligned, native_transplant_hook,
    train_bridge, transplant_hook,
)
from blammoe.toy.corpus import MASK, recall_doc, recall_sample
from blammoe.toy.moe import ToyMoEConfig, ToyMoELM
from blammoe.toy.tokenizers import BPETokenizer, CharTokenizer


def test_recall_doc_blind_masks_only_stored_values():
    clean, blind, answers = recall_doc(random.Random(0))
    assert len(clean) == len(blind)
    diff = [i for i, (c, b) in enumerate(zip(clean, blind)) if c != b]
    assert diff and all(blind[i] == MASK and clean[i].isdigit() for i in diff)
    store, ask = clean.split("ask: ")
    assert all(clean[i].isdigit() for i in answers) and min(answers) > len(store)
    for pair in ask.strip().split(", "):  # every asked value appears in the store
        assert pair in store


def test_recall_sample_is_fixed_length_and_answers_in_range():
    s = recall_sample(random.Random(1), 200)
    assert len(s.text) == len(s.blind) == 200
    assert all(0 <= i < 200 and s.text[i].isdigit() for i in s.answer_chars)


def bpe():
    texts = [recall_sample(random.Random(i), 200).text for i in range(50)] + ["def f(x):\n    return x"] * 20
    return BPETokenizer.train(texts, vocab_size=200)


def test_tokenizers_give_char_offsets_and_single_digit_tokens():
    s = recall_sample(random.Random(2), 120)
    ids, off = CharTokenizer().encode(s.text)
    assert len(ids) == 120 and off[5] == (5, 6)
    tb = bpe()
    ids, off = tb.encode(s.text)
    assert "".join(s.text[a:e] for a, e in off) == s.text  # spans tile the text
    assert all(e - a == 1 for a, e in off if s.text[a:e].isdigit() or s.text[a].isdigit())
    pos = answer_positions(off, s.answer_chars)
    assert len(pos) == len([c for c in s.answer_chars])  # every answer digit is found


def tiny(vocab, d=16, experts=4, seed=0):
    torch.manual_seed(seed)
    return ToyMoELM(ToyMoEConfig(vocab_size=vocab, d_model=d, n_layers=2, n_heads=2,
                                 n_experts=experts, top_k=2, d_expert=16, max_len=128)).eval()


def test_expert_hook_sees_only_routed_tokens_and_dense_matches():
    m = tiny(97)
    ids = torch.randint(1, 97, (2, 30))
    cap = m.expert_outputs(ids, layer=1, expert=2)
    with ExpertCapture(m, expert_path("layers.{layer}.moe.experts.{expert}", 1, 2)) as hook:
        m(ids)
    assert hook.output.shape == (int(cap["routed"].sum()), 16)  # [n_routed, d], spec 1.1 point
    routed_dense = cap["out"][cap["routed"]]
    assert torch.allclose(hook.output, routed_dense, atol=1e-5)  # same op, same tokens


def test_causal_alignment_across_tokenizers():
    samples = [recall_sample(random.Random(3), 100)]
    tb = bpe()
    a = make_batch(samples, CharTokenizer(), 128)
    b = make_batch(samples, tb, 128)
    ends = b_end_chars(b)
    h_a = torch.arange(a.ids.shape[1]).float()[None, :, None]  # A state = its char index
    aligned = gather_aligned(h_a, ends)[0, : len(b.offsets[0]), 0]
    assert torch.equal(aligned, torch.tensor([e - 1 for _, e in b.offsets[0]]).float())
    mask = causal_block_mask(ends, a.ids.shape[1])[0]
    t = 5
    assert not mask[t, : b.offsets[0][t][1]].any() and mask[t, b.offsets[0][t][1]:].all()


def setup_pair():
    samples = [recall_sample(random.Random(10 + i), 100) for i in range(3)]
    ta, tb = CharTokenizer(), bpe()
    A, B = tiny(ta.vocab_size, d=16, seed=0), tiny(tb.vocab_size, d=12, experts=6, seed=1)
    return samples, ta, tb, A, B


def test_transplant_with_own_output_and_zero_gate_bridge_are_no_ops():
    samples, ta, tb, A, B = setup_pair()
    a_clean, _, _, b_blind = encode_pair(samples, ta, tb, 128)
    with torch.no_grad():
        base = B(b_blind.ids)
        own = B.expert_outputs(b_blind.ids, 1, 3)["out"]
        with transplant_hook(B, 1, 3, own, alpha=1.0):
            assert torch.allclose(B(b_blind.ids), base, atol=1e-5)
        src = Source(layer=1, expert=0)
        h = src(A, a_clean.ids)
        bridge = Bridge(16, 12, 8, 2)
        with bridge.hook(B, 1, h, causal_block_mask(b_end_chars(b_blind), h.shape[1])):
            assert torch.equal(B(b_blind.ids), base)
    assert answer_mask(b_blind).sum() > 0


def _moe_io(B, layer, ids):
    """(MoE input, MoE output) at ``layer`` for one forward pass."""
    cap = {}
    h = B.layers[layer].moe.register_forward_hook(lambda m, i, o: cap.update(x=i[0], y=o))
    try:
        B(ids)
    finally:
        h.remove()
    return cap["x"], cap["y"]


def test_native_transplant_swaps_only_routed_tokens_at_router_weight():
    """The expert-level swap: unrouted tokens untouched for ANY signal; routed tokens lose exactly
    w * e_j(x) when the signal is zero; the expert's own output is a no-op."""
    samples, ta, tb, A, B = setup_pair()
    _, _, _, b_blind = encode_pair(samples, ta, tb, 128)
    layer, e = 1, 3
    with torch.no_grad():
        x, base = _moe_io(B, layer, b_blind.ids)
        moe = B.layers[layer].moe
        w, idx = moe.router(x).softmax(-1).topk(moe.top_k, dim=-1)
        routed = (idx == e).any(-1)
        weight = (w * (idx == e)).sum(-1, keepdim=True)
        assert routed.any() and (~routed).any()
        noise = torch.randn_like(base)
        with native_transplant_hook(B, layer, e, noise, alpha=1.0):
            _, y = _moe_io(B, layer, b_blind.ids)
        assert torch.equal(y[~routed], base[~routed])
        with native_transplant_hook(B, layer, e, torch.zeros_like(base), alpha=1.0):
            _, y0 = _moe_io(B, layer, b_blind.ids)
        assert torch.allclose(y0, base - weight * moe.experts[e](x), atol=1e-6)
        own, logits = B.expert_outputs(b_blind.ids, layer, e)["out"], B(b_blind.ids)
        with native_transplant_hook(B, layer, e, own, alpha=1.0):
            assert torch.allclose(B(b_blind.ids), logits, atol=1e-5)
        with transplant_hook(B, layer, e, noise, alpha=1.0):  # the old dense-additive control
            _, yd = _moe_io(B, layer, b_blind.ids)
        assert not torch.allclose(yd[~routed], base[~routed])


def test_train_bridge_moves_gate_and_keeps_models_frozen():
    samples, ta, tb, A, B = setup_pair()
    before = {k: v.clone() for k, v in B.state_dict().items()}
    bridge = Bridge(16, 12, 8, 2)
    train_bridge(bridge, A, B, Source(1, 0, "expert_routed"), 0, lambda n: samples[:n], ta, tb, 128,
                 steps=5, batch_size=2, lr=1e-2, log=lambda m: None)
    assert bridge.rx.gate.abs().item() > 0
    assert all(torch.equal(before[k], v) for k, v in B.state_dict().items())


# --------------------------------------------------------------------------- WP-T2 fused pair
from blammoe.toy.corpus import split_recall_sample  # noqa: E402
from blammoe.toy.fused import BusAblation, FusedPair, read_mask, subset_accuracy, token_ends  # noqa: E402


def test_split_recall_masks_each_value_in_exactly_one_view():
    s = split_recall_sample(random.Random(5), 200)
    assert len(s.text) == len(s.view_a) == len(s.view_b) == 200
    for i, c in enumerate(s.text):
        hidden = (s.view_a[i] == MASK) + (s.view_b[i] == MASK)
        assert hidden == (1 if s.view_a[i] != c or s.view_b[i] != c else 0)
    assert set(s.a_blind).isdisjoint(s.b_blind) and s.a_blind and s.b_blind
    assert s.partner_answers("a") == s.a_blind and s.own_answers("a") == s.b_blind


def test_read_mask_is_causal_across_tokenizers_with_open_null_slot():
    reader = torch.tensor([[1, 2, 3]])  # char model: has read 1, 2, 3 chars
    writer = torch.tensor([[2, 3, 10**9]])  # bpe tokens ending at chars 2, 3, then a pad
    m = read_mask(reader, writer)
    assert m.shape == (1, 3, 4) and not m[..., 0].any()
    assert m[0, 0, 1:].tolist() == [True, True, True]
    assert m[0, 1, 1:].tolist() == [False, True, True]
    assert m[0, 2, 1:].tolist() == [False, False, True]  # pad never readable


def fused_setup(signal):
    samples, ta, tb, A, B = setup_pair()
    sp = [split_recall_sample(random.Random(20 + i), 100) for i in range(2)]
    b1 = make_batch([s.view("a") for s in sp], ta, 128)
    b2 = make_batch([s.view("b") for s in sp], tb, 128)
    return sp, b1, b2, FusedPair(A, B, [0, 1], d_bus=8, n_heads=2, signal=signal)


def test_fused_pair_zero_gates_equal_running_models_separately():
    for signal in ("moe", "experts", "resid"):
        _, b1, b2, pair = fused_setup(signal)
        with torch.no_grad():
            la, lb = pair(b1.ids, b2.ids, token_ends(b1), token_ends(b2))
            assert torch.allclose(la, pair.A(b1.ids), atol=1e-5)
            assert torch.allclose(lb, pair.B(b2.ids), atol=1e-5)


def test_expert_bus_attention_covers_routed_experts_and_respects_causality():
    sp, b1, b2, pair = fused_setup("experts")
    for lk in pair.links.values():
        with torch.no_grad():
            lk.rx_a.gate.fill_(1.0)
            lk.rx_b.gate.fill_(1.0)
    attn = {}
    with torch.no_grad():
        pair(b1.ids, b2.ids, token_ends(b1), token_ends(b2), attn=attn)
    w = attn[0]["b_reads_a"]  # [B, T_b, 1 + T_a * k_a]
    k = pair.A.cfg.top_k
    assert w.shape[-1] == 1 + b1.ids.shape[1] * k
    assert attn[0]["a_expert_ids"].shape == (*b1.ids.shape, k)
    ends_b, ends_a = token_ends(b2), token_ends(b1).repeat_interleave(k, 1)
    future = ends_a[:, None, :] > ends_b[:, :, None]
    assert torch.all(w[..., 1:][future] == 0)  # nothing read from text B hasn't seen
    assert torch.allclose(w.sum(-1)[b2.targets != -100], torch.ones(1), atol=1e-5)


def test_subset_accuracy_scores_only_requested_answers():
    sp, b1, b2, pair = fused_setup("moe")
    with torch.no_grad():
        logits = pair.A(b1.ids)
    own = subset_accuracy(logits, b1, [s.own_answers("a") for s in sp])
    need = subset_accuracy(logits, b1, [s.partner_answers("a") for s in sp])
    assert own["n"] == sum(len(s.own_answers("a")) for s in sp)
    assert need["n"] == sum(len(s.partner_answers("a")) for s in sp)


def test_first_char_matrix_and_diversity_metrics():
    from blammoe.toy.fused import diversity, first_char_matrix

    chars = CharTokenizer().itos[1:]
    tb = bpe()
    M = first_char_matrix(tb, chars)
    assert M.shape == (tb.vocab_size, len(chars))
    assert M[tb.pad_id].sum() == 0 and torch.all(M.sum(1)[torch.arange(tb.vocab_size) != tb.pad_id] <= 1)
    Mc = first_char_matrix(CharTokenizer(), chars)
    assert torch.equal(Mc[1:], torch.eye(len(chars)))

    texts = ["ab"]
    one = lambda c: torch.eye(len(chars))[chars.index(c)]  # noqa: E731
    # target at chars_read=1 is "b": solo says "x", partner says "y", fused says "b"
    solo, partner, fused = {(0, 1): one("x")}, {(0, 1): one("y")}, {(0, 1): one("b")}
    d = diversity(solo, fused, partner, texts, chars)
    assert d["novel"] == 1.0 and d["novel_and_correct"] == 1.0 and d["emergent_of_both_wrong"] == 1.0
    d_same = diversity(solo, solo, partner, texts, chars)
    assert d_same["changed_vs_solo"] == 0.0 and d_same["js_fused_vs_solo"] < 1e-6


def test_char_distance_buckets():
    from blammoe.toy.fused import REL_BUCKETS, char_distance_buckets

    reader = torch.tensor([[10, 100]])
    writer = torch.tensor([[10, 9, 3, 20, 40]])
    b = char_distance_buckets(reader, writer)
    assert b.shape == (1, 2, 6) and torch.all(b[..., 0] == REL_BUCKETS - 1)
    assert b[0, 0, 1:].tolist() == [1, 2, 6, 0, 0]  # d = 0, 1, 7 (5-7 bucket), future, future
    assert b[0, 1, 1:].tolist() == [10, 10, 10, 10, 9]  # d = 90, 91, 97, 80 -> 64+; 60 -> 32-63


def test_fused_pair_with_rel_pos_is_exact_at_zero_gates_and_trains():
    samples, ta, tb, A, B = setup_pair()
    sp = [split_recall_sample(random.Random(30 + i), 100) for i in range(2)]
    b1 = make_batch([s.view("a") for s in sp], ta, 128)
    b2 = make_batch([s.view("b") for s in sp], tb, 128)
    for signal in ("moe", "experts"):
        pair = FusedPair(A, B, [0, 1], d_bus=8, n_heads=2, signal=signal, rel_pos=True)
        e1, e2 = token_ends(b1), token_ends(b2)
        with torch.no_grad():
            la, lb = pair(b1.ids, b2.ids, e1, e2)
            assert torch.allclose(la, A(b1.ids), atol=1e-5) and torch.allclose(lb, B(b2.ids), atol=1e-5)
        la, lb = pair(b1.ids, b2.ids, e1, e2)
        (la.logsumexp(-1).mean() + lb.logsumexp(-1).mean()).backward()
        assert all(lk.rx_b.gate.grad is not None for lk in pair.links.values())


def test_expert_attention_mass_attributes_attention_to_expert_ids():
    from blammoe.toy.fused import expert_attention_mass

    ids = torch.tensor([[[0, 1], [2, 1]]])  # 1 doc, 2 writer positions, k=2 -> bus [0,1,2,1]
    w = torch.zeros(1, 3, 5)
    w[0, 0, 0] = 1.0  # reader 0 reads only the null slot
    w[0, 1, 3] = 1.0  # reader 1 reads bus entry 2 -> expert 2
    w[0, 2, 2], w[0, 2, 4] = 0.5, 0.5  # reader 2 reads expert 1 twice
    open_ = torch.zeros(1, 3, 5, dtype=torch.bool)
    r = expert_attention_mass(w, ids, 3, torch.tensor([[True, True, True]]), open_)
    assert abs(r["null_share"] - 1 / 3) < 1e-6
    assert [round(x, 3) for x in r["share"]] == [0.0, 0.5, 0.5]
    assert [round(x, 3) for x in r["baseline"]] == [0.25, 0.5, 0.25]
    assert [round(x, 3) for x in r["ratio"]] == [0.0, 1.0, 2.0]


def test_expert_attention_mass_baseline_counts_only_legal_entries():
    """Blocked entries (future, padding, hidden) must not enter the baseline: a reader that
    attends uniformly over what it may read is at ratio 1 for every expert it can see."""
    from blammoe.toy.fused import expert_attention_mass

    ids = torch.tensor([[[0, 1], [2, 2]]])  # bus [0,1,2,2]; reader may read only position 0
    blocked = torch.tensor([[[False, False, False, True, True]]])
    w = torch.tensor([[[0.0, 0.5, 0.5, 0.0, 0.0]]])
    r = expert_attention_mass(w, ids, 3, torch.tensor([[True]]), blocked)
    assert [round(x, 3) for x in r["baseline"]] == [0.5, 0.5, 0.0]  # was [0.25, 0.25, 0.5]
    assert [round(x, 3) for x in r["ratio"][:2]] == [1.0, 1.0]


def test_hide_experts_blocks_exactly_those_bus_entries():
    sp, b1, b2, pair = fused_setup("experts")
    for lk in pair.links.values():
        with torch.no_grad():
            lk.rx_a.gate.fill_(1.0)
            lk.rx_b.gate.fill_(1.0)
    attn = {}
    with torch.no_grad():
        pair(b1.ids, b2.ids, token_ends(b1), token_ends(b2), attn=attn,
             ablate=(BusAblation(hide=frozenset({0})), BusAblation(hide=frozenset({2}))))
    w_ab, ids_b = attn[0]["a_reads_b"], attn[0]["b_expert_ids"].flatten(1)
    w_ba, ids_a = attn[0]["b_reads_a"], attn[0]["a_expert_ids"].flatten(1)
    assert torch.all(w_ab[..., 1:][(ids_b == 2)[:, None, :].expand_as(w_ab[..., 1:])] == 0)
    assert torch.all(w_ba[..., 1:][(ids_a == 0)[:, None, :].expand_as(w_ba[..., 1:])] == 0)
    assert w_ab[..., 1:][(ids_b != 2)[:, None, :].expand_as(w_ab[..., 1:])].sum() > 0


def test_bus_ablation_apply_blanks_content_and_hides_entries():
    torch.manual_seed(0)
    idx = torch.tensor([[[0, 1], [2, 0]]])  # 1 doc, 2 positions, k=2 -> bus [0, 1, 2, 0]
    slots = torch.randn(1, 2, 2, 3)
    mask = torch.zeros(1, 4, 5, dtype=torch.bool)
    value = {7: torch.arange(9.0).reshape(3, 3)}  # per-expert constants at layer 7
    s, m = BusAblation(blank=frozenset({0}), blank_value=value).apply(7, slots, idx, mask)
    assert torch.equal(s[0, 0, 0], value[7][0]) and torch.equal(s[0, 1, 1], value[7][0])
    assert torch.equal(s[0, 0, 1], slots[0, 0, 1]) and torch.equal(s[0, 1, 0], slots[0, 1, 0])
    assert torch.equal(m, mask)  # blanked entries stay readable
    s, m = BusAblation(blank=frozenset({2})).apply(7, slots, idx, mask)
    assert torch.equal(s[0, 1, 0], torch.zeros(3))  # no values given -> zeros
    s, m = BusAblation(hide=frozenset({0})).apply(7, slots, idx, mask)
    assert torch.equal(s, slots)
    assert m[0, :, 1].all() and m[0, :, 4].all() and not m[0, :, [0, 2, 3]].any()  # col 0 = null slot


def live_one_link_pair():
    """Experts pair exchanging at link 0 only, gates open: A's logits depend on B solely
    through B's layer-0 bus (B's layer-0 routing reads B's input, not its experts)."""
    _, b1, b2, _ = fused_setup("experts")
    _, ta, tb, A, B = setup_pair()
    pair = FusedPair(A, B, [0], d_bus=8, n_heads=2, signal="experts")
    with torch.no_grad():
        for lk in pair.links.values():
            lk.rx_a.gate.fill_(1.0)
            lk.rx_b.gate.fill_(1.0)
    run = lambda abl: pair(b1.ids, b2.ids, token_ends(b1), token_ends(b2), ablate=(BusAblation(), abl))[0]  # noqa: E731
    return pair, run


def test_blanking_every_expert_removes_the_partners_content_but_not_its_tags():
    pair, run = live_one_link_pair()
    all_b = frozenset(range(pair.B.cfg.n_experts))
    with torch.no_grad():
        keep, blank = run(BusAblation()), run(BusAblation(blank=all_b))
        for ex in pair.B.layers[0].moe.experts:  # change what B's experts SAY, not who fires
            ex.w2.weight.mul_(0.5)
        assert not torch.allclose(run(BusAblation()), keep, atol=1e-4)
        assert torch.allclose(run(BusAblation(blank=all_b)), blank, atol=1e-6)
        pair.links["0"].eid_b.weight.add_(1.0)  # ...but the tags still reach A
        assert not torch.allclose(run(BusAblation(blank=all_b)), blank, atol=1e-4)


def test_tag_ablations_zero_or_shift_the_expert_ids():
    pair, run = live_one_link_pair()
    eid = pair.links["0"].eid_b
    with torch.no_grad():
        zero, shift = run(BusAblation(tags="zero")), run(BusAblation(tags="shift"))
        eid.weight.add_(torch.randn_like(eid.weight))
        assert torch.allclose(run(BusAblation(tags="zero")), zero, atol=1e-6)  # tags gone
        assert not torch.allclose(run(BusAblation(tags="shift")), shift, atol=1e-4)
        # shift tags expert e as e+1: the same as keeping tags with the table rolled by one
        shifted = run(BusAblation(tags="shift"))
        eid.weight.copy_(eid.weight.roll(-1, 0))
        assert torch.allclose(run(BusAblation()), shifted, atol=1e-6)
    with pytest.raises(ValueError):
        BusAblation(tags="scramble")


def test_bus_ablations_refuse_non_expert_signals():
    _, b1, b2, pair = fused_setup("moe")
    with pytest.raises(ValueError):
        pair(b1.ids, b2.ids, token_ends(b1), token_ends(b2), ablate=(BusAblation(hide=frozenset({0})), BusAblation()))


def test_expert_output_means_average_routed_outputs_over_real_positions():
    from blammoe.toy.fused import NEVER, expert_output_means

    _, b1, b2, pair = fused_setup("experts")
    means_a, means_b = expert_output_means(pair, b1, b2)
    attn = {}
    with torch.no_grad():
        pair(b1.ids, b2.ids, token_ends(b1), token_ends(b2), attn=attn)
    assert (token_ends(b2) == NEVER).any()  # the BPE views differ in length: padding to skip
    for side, means, batch, n in (("a", means_a, b1, pair.A.cfg.n_experts), ("b", means_b, b2, pair.B.cfg.n_experts)):
        assert sorted(means) == [0, 1]
        real = token_ends(batch) != NEVER
        for l, m in means.items():
            out, ids = attn[l][f"{side}_expert_out"], attn[l][f"{side}_expert_ids"]
            assert m.shape == (n, out.shape[-1])
            for e in range(n):
                rows = [out[i, t, j] for i, t in real.nonzero().tolist() for j in range(ids.shape[-1])
                        if ids[i, t, j] == e]
                want = torch.stack(rows).mean(0) if rows else torch.zeros(out.shape[-1])
                assert torch.allclose(m[e], want, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_model_is_causal_in_every_dtype(dtype):
    """A float32 -inf attention mask is not applied correctly in float64: the model then reads
    the future. Position t must give the same logits whether or not later tokens exist."""
    torch.manual_seed(0)
    m = ToyMoELM(ToyMoEConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=2, n_experts=4, top_k=2,
                              d_expert=16, max_len=64)).eval().to(dtype)
    ids = torch.randint(1, 50, (1, 40))
    with torch.no_grad():
        full, pre = m(ids)[0, :20].float(), m(ids[:, :20])[0].float()
    assert torch.allclose(full, pre, atol=1e-2 if dtype == torch.bfloat16 else 1e-5)
