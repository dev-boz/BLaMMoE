"""scripts/toy_where_knowledge.py (mean ablations, routing) and scripts/toy_partner_shift.py (per-doc
JS rows): the hooks do what they say, on a tiny CPU model."""
import sys
from pathlib import Path

import torch

from blammoe.toy.batching import make_batch
from blammoe.toy.corpus import Sample
from blammoe.toy.moe import ToyMoEConfig, ToyMoELM
from blammoe.toy.tokenizers import CharTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import toy_partner_shift as ps  # noqa: E402
import toy_where_knowledge as wk  # noqa: E402


def tiny():
    torch.manual_seed(0)
    return ToyMoELM(ToyMoEConfig(vocab_size=CharTokenizer().vocab_size, d_model=16, n_layers=2, n_heads=2,
                                 n_experts=4, top_k=2, d_expert=16, max_len=64)).eval()


def batch():
    texts = ["the model reads text and predicts", "def f(x):\n    return x + 1\n", "short"]
    return make_batch([Sample(text=t, blind=t, kind="prose") for t in texts], CharTokenizer(), 64)


def test_expert_ablation_swaps_only_that_experts_routed_output_for_its_mean():
    m, b = tiny(), batch()
    means = wk.reference_means(m, b)
    with torch.no_grad(), wk.Replace(wk.ablation(m, means, ("expert", 1, 2))):
        got = m(b.ids)
    expert = m.layers[1].moe.experts[2]
    real = expert.forward
    expert.forward = lambda x: means["expert", 1, 2].expand(x.shape[0], -1)  # the same swap, done by hand
    try:
        with torch.no_grad():
            want = m(b.ids)
    finally:
        expert.forward = real
    assert torch.allclose(got, want, atol=1e-5)
    with torch.no_grad():
        assert not torch.allclose(got, m(b.ids), atol=1e-4)  # and it did change something


def test_layer_ablations_replace_the_output_with_its_mean_and_hooks_come_off():
    m, b = tiny(), batch()
    means = wk.reference_means(m, b)
    with torch.no_grad():
        clean = m(b.ids)
    for what, mod in ((("attn", 0), m.layers[0].attn), (("moe", 1), m.layers[1].moe), (("moe_all",), m.layers[0].moe)):
        seen = []
        with torch.no_grad(), wk.Replace(wk.ablation(m, means, what)):
            spy = mod.register_forward_hook(lambda mod, i, o: seen.append(o[0] if isinstance(o, tuple) else o))
            m(b.ids)  # hooks run in order, so the spy sees the swapped output
            spy.remove()
        mean = means[what if what[0] != "moe_all" else ("moe", 0)]
        assert torch.allclose(seen[0], mean.expand_as(seen[0]))
    with torch.no_grad():
        assert torch.allclose(m(b.ids), clean)  # every hook removed
    assert all(not mod._forward_hooks for blk in m.layers for mod in (blk.attn, blk.moe))


def test_routing_shares_sum_to_one_per_layer():
    m, b = tiny(), batch()
    r = wk.routing(m, b)
    assert len(r) == 2 and all(abs(sum(layer) - 1) < 1e-5 and len(layer) == 4 for layer in r)


def test_doc_rows_pool_back_to_the_per_position_mean():
    v = torch.tensor([1.0, 2.0, 3.0, 6.0])
    rows = ps.doc_rows(v, [0, 0, 2, 2], 3)
    assert rows == [[3.0, 2.0], [0.0, 0.0], [9.0, 2.0]]
    assert ps.pooled(rows) == float(v.mean())


def test_shift_rows_are_zero_when_the_partner_changes_nothing():
    texts = ["abcd", "dcba"]
    chars = ["a", "b", "c", "d"]
    g = torch.Generator().manual_seed(0)
    solo = {(i, j): torch.rand(4, generator=g).softmax(0) for i in range(2) for j in range(4)}
    partner = {k: torch.rand(4, generator=g).softmax(0) for k in solo if k[1] != 3}  # one fewer shared position
    rows = ps.shift_rows({"solo": solo, "fused": solo, "wrong": solo}, partner, texts, chars)
    for k in ("shift", "shift_wrong", "dependence"):
        assert abs(ps.pooled(rows[k])) < 1e-9
    assert ps.pooled(rows["fused_to_partner"]) == ps.pooled(rows["wrong_to_partner"]) == ps.pooled(rows["solo_to_partner"]) > 0
    assert [r[1] for r in rows["shift"]] == [3.0, 3.0]


def test_toward_splits_exactly_into_attraction_and_repulsion():
    """The review's counterexample: partner = alone = [0.9, 0.1], paired [0.8, 0.2], wrong [0.1, 0.9].
    The real partner moves the model AWAY from the partner, yet toward is large and positive; the
    split puts all of it in repulsion and makes attraction negative."""
    texts, chars = ["ab"], ["a", "b"]
    solo = {(0, 0): torch.tensor([0.9, 0.1])}
    p = {"solo": solo, "fused": {(0, 0): torch.tensor([0.8, 0.2])}, "wrong": {(0, 0): torch.tensor([0.1, 0.9])}}
    rows = ps.shift_rows(p, solo, texts, chars)
    to, fu, wr = (ps.pooled(rows[k]) for k in ("solo_to_partner", "fused_to_partner", "wrong_to_partner"))
    toward, attraction, repulsion = wr - fu, to - fu, wr - to
    assert toward > 0.3 and attraction < 0 and repulsion > toward
    assert abs(toward - (attraction + repulsion)) < 1e-12
