# A with a prose-heavy char partner: it's not the tokenizer

## Why this

In every run so far, A (code-heavy, char-level) never got prose from B (prose-heavy, BPE) through the expert bus. The [CPU diagnostics](toy-different-results.md) narrowed it down. A's reader on the expert bus never opens, and B's prose knowledge *is* in its experts, just spread thinly over 64 small ones. That left three guesses:

1. Timing (the tokenizer). A can only read a B token once its boundaries are final, so B's view arrives a character late.
2. Spread. B's knowledge is spread too thin per expert to be worth reading.
3. Something about A as a reader.

This run takes the tokenizer and the spread out. A's new partner is A' (prose-heavy, char-level). A' has exactly A's architecture: 8 experts per layer, d_model 128, the same tokenizer and even the same initial weights. It's the prose-heavy model from the d control units of the complementary run, trained on 90% prose where A had 10%. So there's no reading delay, and the experts are as coarse as A's own.

- e units - A paired with A', seeds 0 / 1 / 2.
- f units (control) - A paired with A from another seed. Same everything, and it also knows mostly code, so any gain here isn't about prose knowledge.

Nothing was pretrained, since both models are copied from the complementary run. The translators are the same unforced ones as there (ordinary text only, expert bus and residual stream, 3000 steps each). That's 12 new translators, and all 12 passed the strict leak check. Both CMP 170HXs ran side by side for about 85 minutes under the thermal guard, peaking at 50°C core and 58°C memory.

## Result 1 - the expert bus still carries no prose into A

A on prose, bits per character paired minus alone (negative is better):

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Expert bus, partner A' (prose-heavy char) | -0.017 | -0.019 | -0.018 |
| Expert bus, control partner (code-heavy char) | -0.018 | -0.017 | -0.021 |
| Expert bus, partner B (complementary run) | -0.015 | -0.014 | -0.012 |

A' helps A on prose exactly as much as a partner that knows mostly code. That's the small general gain any partner gives. The partner-shift measure agrees. A leans towards A''s view by +0.0002 to +0.0005 (nats per position), against about zero for the control, and a small fraction of what the residual stream gave A from B (+0.002 to +0.009). A's gates on the expert links never get past 0.02, like before.

So the tokenizer isn't it, and neither is B's thin spread. With a partner that reads at the same time, has the same coarse experts, and is 0.31-0.35 bits/char better at prose, A still gets nothing through the expert bus.

## Result 2 - code still flows the other way

A' on code (its weak domain), through the expert bus:

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| A' with A, paired minus alone | **-0.084** | **-0.201** | **-0.113** |
| A' with A, paired minus wrong partner | -0.114 | -0.820 | -0.209 |
| Toward A's view (nats per position) | +0.018 | +0.115 | +0.031 |
| For comparison: B with A (complementary run) | -0.132 | -0.109 | -0.126 |

A' gets code from A through the expert bus in every seed, about as much as B did. The reader here has exactly A's architecture. So "A as a reader" isn't it either, at least not A's architecture.

That leaves a direction effect. Code goes from a code model into a prose model through the expert bus, whatever the reader's tokenizer or architecture. Prose doesn't go from a prose model into a code model, whatever the sender's tokenizer or architecture.

## Result 3 - through the residual stream, everything flows

This one surprised me. Through the residual stream, the two char models share almost everything:

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| A on prose: alone -> paired | 2.208 -> 1.923 | 2.220 -> 1.914 | 2.199 -> 1.916 |
| A' alone on prose (for reference) | 1.899 | 1.869 | 1.847 |
| A' on code: alone -> paired | 2.111 -> 1.642 | 2.072 -> 1.674 | 2.048 -> 1.673 |
| A alone on code (for reference) | 1.581 | 1.629 | 1.602 |
| A on prose with a *wrong* partner | 7.73 | 7.93 | 7.70 |

With B, the residual stream gave A only 0.01-0.04 bits on prose. Here it's 0.28-0.31 bits, which takes A nearly all the way to A''s own level, and the same happens the other way round on code. Both models open the last link's gate to about 0.55, much wider than any link in the complementary run (0.39 at most). A wrong-document partner is then a disaster (over 7 bits per character, against about 2.2 alone), because each model has learned to lean on the other's last layer.

I checked this for a leak first, since a gain this big looked too good. The float64 leak check matches the prefix-only re-runs to about 1e-13 at every position, both models, both translators. And the paired model never beats the partner alone. So it's real, but it's not fusion as much as hand-over: on its weak domain, each model mostly lets the partner's prediction through. The partner-shift "toward" is +0.46 to +0.48, about five times the best expert-bus transfer so far (B's code, +0.089), which fits that.

Why so much easier than with B? Two things changed at once. There's no tokenizer delay, and A and A' started from the same initial weights, so their residual streams may still be in a similar basis. This run can't tell those apart.

## What this does and doesn't show

It shows the expert-bus gap isn't about B's tokenizer, B's fine-grained experts, or A's architecture as a reader. A gets no prose through the expert bus even from a model built exactly like it. And a model built exactly like A *reads* code through the expert bus fine.

It shows that with matching tokenizers, the residual stream can carry almost everything, both ways. For the end goal that's a warning as much as good news. The links will happily hand a model over to its partner rather than mix them.

It doesn't explain the direction effect. Some guesses, none tested:

- What the experts hold. Code may be more local (keywords, brackets, indentation), the kind of thing a single expert's output at one position can carry. Prose prediction may lean more on context the experts don't hold at that position.
- What the reader can use. A has seen little prose, so its later layers may not know what to do with prose features even when it's given them. A' has seen little code too, though, so this would need code to be easier to pick up from 10%.

And the usual caveats: toy models, 3 seeds, 48 test documents per domain, one data mix. The A-A' pairs share initial weights, which the A-B pairs didn't.

## Reproduce

```bash
udisksctl mount -b /dev/disk/by-label/Data   # results/ are symlinks to the Data drive
# needs results/toy_complement (both models are copied from there)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_pair.yaml 4 e0 f0 e2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_pair.yaml 4 e1 f1 f2
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/toy_char_pair.yaml 4 none
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_partner_shift.py results/toy_char_pair 16
```

The thermal guard was `~/cmp-tests/sweep-guard.sh results/toy_char_pair configs/toy_char_pair.yaml cp42 cp43` in its own tmux session (outside the repo). Results are in `results/toy_char_pair/summary.json` and `partner_shift.json`.
