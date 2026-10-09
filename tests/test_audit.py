"""Prefix-honest audit: indexing is exact, and the leak mechanism it guards against exists."""
import random

import torch

from blammoe.toy.audit import full_text_a_nll, honest_a_nll, prefix_inputs
from blammoe.toy.batching import make_batch
from blammoe.toy.corpus import Sample, recall_sample
from blammoe.toy.fused import FusedPair, token_ends
from blammoe.toy.moe import ToyMoEConfig, ToyMoELM
from blammoe.toy.tokenizers import BPETokenizer, CharTokenizer

TEXT = "def resolve(path, results):\n    return the_results + thx + path.resolve()\n"


def bpe():
    texts = [recall_sample(random.Random(i), 200).text for i in range(30)] + [TEXT * 3] * 20
    return BPETokenizer.train(texts, vocab_size=220)


def tiny(vocab, d, seed):
    torch.manual_seed(seed)
    return ToyMoELM(ToyMoEConfig(vocab_size=vocab, d_model=d, n_layers=2, n_heads=2, n_experts=4,
                                 top_k=2, d_expert=16, max_len=128)).eval()


def test_full_text_bpe_boundaries_depend_on_later_chars():
    """The leak exists: some full-text B token ends at a point where tokenizing only
    the prefix would split differently, so a boundary carries information from beyond it."""
    tb = bpe()
    _, full = tb.encode(TEXT)
    differs = 0
    for i in range(1, len(TEXT)):
        honest = tb.encode(TEXT[:i])[1]
        seen = [o for o in full if o[1] <= i]
        differs += honest != seen
    assert differs > 0


def test_prefix_inputs_never_pass_the_prefix():
    ta, tb = CharTokenizer(), bpe()
    ids_a, ends_a, ids_b, ends_b, tgt, last = prefix_inputs(TEXT, [3, 10, 40], ta, tb)
    assert last.tolist() == [3, 10, 40]
    assert tgt.tolist() == [ta.stoi[TEXT[4]], ta.stoi[TEXT[11]], ta.stoi[TEXT[41]]]
    for row, i in enumerate([3, 10, 40]):
        real = ends_b[row][ends_b[row] < 10**9]
        assert int(real.max()) == i + 1  # B read exactly the prefix, tokenized on its own


def test_honest_equals_full_text_when_links_are_off():
    ta, tb = CharTokenizer(), bpe()
    pair = FusedPair(tiny(ta.vocab_size, 16, 0), tiny(tb.vocab_size, 12, 1), [0, 1], d_bus=8, n_heads=2,
                     signal="experts", rel_pos=True)
    s = Sample(text=TEXT[:60], blind=TEXT[:60])
    a, b = make_batch([s], ta, 128), make_batch([s], tb, 128)
    off = frozenset({0, 1})
    full = full_text_a_nll(pair, a, b, token_ends(a), token_ends(b), disabled=off)[0]
    pos = [5, 20, 47]
    honest = honest_a_nll(pair, s.text, pos, ta, tb, disabled=off)
    assert torch.allclose(honest, full[pos], atol=1e-5)
