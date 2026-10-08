"""WP-T toy POC: corpus invariants, tokenizer offsets, the MoE hook point, causal
cross-tokenizer alignment, and no-op guarantees of both carriers. All tiny/fast."""
import random

import torch

from blammoe.hooks import ExpertCapture, expert_path
from blammoe.toy.batching import answer_mask, answer_positions, make_batch
from blammoe.toy.bridge import (
    Bridge, Source, b_end_chars, causal_block_mask, encode_pair, gather_aligned, train_bridge,
    transplant_hook,
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


def test_train_bridge_moves_gate_and_keeps_models_frozen():
    samples, ta, tb, A, B = setup_pair()
    before = {k: v.clone() for k, v in B.state_dict().items()}
    bridge = Bridge(16, 12, 8, 2)
    train_bridge(bridge, A, B, Source(1, 0, "expert_routed"), 0, lambda n: samples[:n], ta, tb, 128,
                 steps=5, batch_size=2, lr=1e-2, log=lambda m: None)
    assert bridge.rx.gate.abs().item() > 0
    assert all(torch.equal(before[k], v) for k, v in B.state_dict().items())
