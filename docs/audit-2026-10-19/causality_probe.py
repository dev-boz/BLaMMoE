"""Black-box causality probe for FusedPair under safe_mask, BOTH directions, all signals.

Two texts share text[:c] and differ after it. Any position that has (by its own
tokenization, which is identical in both texts up to that token) read only chars < = c
must produce identical logits in both runs. Checked for A (char) positions and for B
(BPE) positions whose token ids/offsets agree in both encodings. Links are live
(gates forced to 1) with random weights, so any path from the future would show.
"""
import random
import sys

import torch

sys.path.insert(0, ".")
from blammoe.toy.batching import make_batch  # noqa: E402
from blammoe.toy.corpus import Sample, split_recall_sample  # noqa: E402
from blammoe.toy.fused import FusedPair, run_pair  # noqa: E402
from blammoe.toy.moe import ToyMoEConfig, ToyMoELM  # noqa: E402
from blammoe.toy.tokenizers import BPETokenizer, CharTokenizer  # noqa: E402

L = 160
CODE = "def resolve(path, results):\n    return the_results + thx + path.resolve()\n" * 3


def tiny(vocab, d, seed, n_exp):
    torch.manual_seed(seed)
    return ToyMoELM(ToyMoEConfig(vocab_size=vocab, d_model=d, n_layers=3, n_heads=2, n_experts=n_exp,
                                 top_k=2, d_expert=24, max_len=256)).eval()


def main():
    rng = random.Random(0)
    corpus = [split_recall_sample(random.Random(i), L).text for i in range(40)] + [CODE] * 20
    ta, tb = CharTokenizer(), BPETokenizer.train(corpus, vocab_size=300)
    texts = [split_recall_sample(random.Random(100 + i), L).text for i in range(6)] + [CODE[:L]] * 2
    worst = {}
    for signal in ("experts", "moe", "resid"):
        for rel in (False, True):
            pair = FusedPair(tiny(ta.vocab_size, 16, 0, 4), tiny(tb.vocab_size, 12, 1, 6), [0, 1, 2], d_bus=8,
                             n_heads=2, signal=signal, rel_pos=rel, safe_mask=True)
            with torch.no_grad():
                for lk in pair.links.values():
                    lk.rx_a.gate.fill_(1.0)
                    lk.rx_b.gate.fill_(1.0)
            wa = wb = 0.0
            na = nb = 0
            for text in texts:
                for c in rng.sample(range(5, L - 5), 12):
                    alt = text[:c] + "".join(rng.choice("abcxyz 0123;=,\n") for _ in range(L - c))
                    s1, s2 = Sample(text=text, blind=text), Sample(text=alt, blind=alt)
                    a1, b1 = make_batch([s1], ta, L), make_batch([s1], tb, L)
                    a2, b2 = make_batch([s2], ta, L), make_batch([s2], tb, L)
                    with torch.no_grad():
                        la1, lb1 = run_pair(pair, a1, b1)
                        la2, lb2 = run_pair(pair, a2, b2)
                    # A position j has read text[:j+1]
                    ja = [j for j in range(a1.ids.shape[1]) if j + 1 <= c]
                    if ja:
                        wa = max(wa, float((la1[0, ja] - la2[0, ja]).abs().max()))
                        na += len(ja)
                    # B position t: same token ids + offsets in both encodings for 0..t, end <= c
                    o1, o2 = b1.offsets[0], b2.offsets[0]
                    jb = []
                    for t in range(min(len(o1), len(o2))):
                        if o1[t] != o2[t] or int(b1.ids[0, t]) != int(b2.ids[0, t]) or o1[t][1] > c:
                            break
                        jb.append(t)
                    if jb:
                        wb = max(wb, float((lb1[0, jb] - lb2[0, jb]).abs().max()))
                        nb += len(jb)
            worst[(signal, rel)] = (wa, na, wb, nb)
            print(f"signal={signal:7s} rel_pos={rel!s:5s}  A max|dlogit|={wa:.2e} over {na} pos | "
                  f"B max|dlogit|={wb:.2e} over {nb} pos", flush=True)
    bad = {k: v for k, v in worst.items() if v[0] > 1e-4 or v[2] > 1e-4}
    print("LEAK FOUND" if bad else "no future dependence detected", bad)


if __name__ == "__main__":
    main()
