# Allow/force dial results, part 2 (the reward, and a fair force dial)

## Where part 1 left off

[Part 1](toy-dial-results.md) tried to turn a dial between "transfer allowed" and "transfer forced" by mixing a growing share of split-view recall (the task that can't be solved without the partner) into the translator's training. Even 1 window in 20 switched off the free transfer the unforced translator had found, where B (the prose model) learns code from A.

The cause turned out to be the loss, not the forcing. The extra term for recall answers was averaged over only the answers in each batch, so when answers were rare each one weighed up to 146 times an ordinary character. The dial never got gentle.

So this run does two things:

1. **The force dial again, weighted fairly.** Same shares (0.05 and 0.15), but each answer token now counts 21 times inside the one overall average (1 plus an answer weight of 20). That's about what the old 0.5 runs had, so the low shares really are a smaller push now.
2. **The other dial, the reward.** Nothing forced at all, ordinary text only. Instead, characters the pair gets right *only because of the partner* get extra weight:

- Partner win - a character the paired model gets right, that the same model alone gets wrong, and that it also gets wrong when the partner is reading a different document.
- λ (reward weight) - a partner win counts 1 + λ times in the average. I tried 5, 15 and 50.

Everything else is the same as part 1. The two models are copied from the complementary run (code-heavy A, prose-heavy B), so they're identical across every setting, with the same three seeds and the same-domain control pairs (c and d units). Expert bus only. 45 new translators, and all 45 passed the strict leak check. Both CMP 170HXs ran side by side for 5.8 and 7.4 hours, under a guard that would have killed both runs at 80°C core or 88°C on the memory. They peaked at 50°C and 59°C.

## Result 1 - weighted fairly, the force dial works

B on code, bits per character with the partner minus alone (negative is better). Control partner (same size, also knows mostly prose) in brackets:

| Setting | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Nothing forced (complementary run) | **-0.132** (-0.024) | **-0.109** (-0.025) | **-0.126** (-0.016) |
| Forced 0.05, fair weighting | **-0.063** (-0.003) | **-0.079** (-0.024) | **-0.048** (-0.013) |
| Forced 0.15, fair weighting | **-0.065** (-0.005) | **-0.044** (-0.021) | -0.007 (-0.003) |
| Forced 0.05, old weighting (part 1) | +0.004 (+0.006) | -0.029 (-0.002) | +0.001 (+0.002) |
| Forced 0.15, old weighting (part 1) | +0.010 (+0.003) | -0.027 (+0.006) | +0.009 (+0.031) |

Bold means the real partner beats the control with non-overlapping 95% intervals. With fair weighting at 0.05, B keeps about half of its free code gain in all three seeds, and it's still well clear of the control every time. At 0.15 it holds in two seeds. Seed 2 at 0.15 is the odd one out, and I come back to it below.

And at the same time the forced transfer works. Split recall accuracy on the answers only the partner has (chance is about 10%):

| Setting | B needs A, seeds 0 / 1 / 2 | A needs B, seeds 0 / 1 / 2 |
|---|---|---|
| Nothing forced | 13% / 34% / 15% | 9% / 10% / 9% |
| Forced 0.05, fair | **91%** / **91%** / **90%** | 10% / 10% / 9% |
| Forced 0.15, fair | **97%** / **100%** / **99%** | 12% / 10% / **96%** |
| Forced 0.05, old (part 1) | 10% / **97%** / 10% | 10% / 12% / 9% |
| Forced 0.15, old (part 1) | **95%** / **98%** / **97%** | 10% / 11% / 11% |

The fair 0.05 setting is the first where B gets both kinds of transfer at once in every seed: forced recall, *and* about half of the free code transfer. (In part 1 only seed 1 at 0.05 managed both, with a quarter of the free gain.)

Counting all 9 units (the control pairs too), B got above 50% on A's answers in 7 of 9 at fair 0.05, against 1 of 9 with the old weighting. So a much lighter push per answer made the forced transfer switch on *more* reliably, not less. That fits part 1's guess: a few huge-weight answer tokens drown everything else.

It also looks more like a dial now. In part 1 every unit was either on (95-100%) or at chance. At fair 0.05 there are in-between values (57%, 79%, then 90-94%), and the exact-answer rate is lower (66-69%, against 81-99% at heavier forcing). The push is small enough to only half-teach it sometimes.

Seed 2 at 0.15 is interesting. It's the first time in any run that A learned to read B's answers (96%) at a share below 0.5. And it's the one place where B's free code gain disappeared (-0.007). B also stopped leaning on A on ordinary code (with a wrong-document partner it scores about the same as with the real one). Something about that seed's links went to recall in both directions and left nothing for code. One case can't tell me more than that.

A on prose (its weak domain) still gets nothing from B under any setting, the same as with nothing forced. The links into A carry the forced recall, but never prose.

## Result 2 - the reward makes things worse at every weight

The reward dial was meant to be the gentle "encourage" knob. It isn't:

| λ | B on code, seeds 0 / 1 / 2 (control) | A on prose, seeds 0 / 1 / 2 | B on mixed text, alone -> paired |
|---|---|---|---|
| 0 (complementary run) | -0.132 / -0.109 / -0.126 (about -0.02) | -0.015 / -0.014 / -0.012 | 1.82-1.85 -> 1.75-1.78 |
| 5 | -0.031 / -0.018 / -0.018 (-0.007 to -0.014) | -0.008 / -0.011 / -0.006 | -> 1.82-1.85 |
| 15 | +0.033 / +0.033 / +0.012 (+0.021 to +0.035) | +0.050 / +0.066 / +0.109 | -> 1.84-1.88 |
| 50 | +0.189 / +0.163 / +0.213 (+0.201 to +0.261) | +0.505 / +0.511 / +0.492 | -> 1.99-2.02 |

Even λ = 5 takes away most of B's code gain, and B stops caring what A is reading (in seeds 1 and 2 a wrong-document partner is just as good). At 15, pairing makes both models worse than alone. At 50 it's a wreck: A is half a bit per character worse on prose, and B's code is as bad with the real partner as with a wrong one (2.32 against 2.33, alone 2.13). The training didn't blow up, it trained to something harmful. The control partners get just as bad, so this is damage, not knowledge moving.

I'd seen a hint of this before. λ = 2 as a short fine-tune on top of already-trained links did nothing, and an early version with a far bigger effective weight wrecked accuracy. I'd blamed that on the weighting and fixed it (per character inside the overall average). The weighting here is fair, and it still goes wrong. So the problem is what it rewards.

My best guess, not tested: a partner win is picked using the model's own top guess, and the wrong-partner check only tries one other document. A guess that flips to the right answer because of noise from the real partner will usually *not* flip the same way with a different document, so noise passes the check as "partner-dependent". Rewarding those wins teaches the links to push B's guesses away from its own, which mostly makes them wrong. In other words, "depends on the partner" isn't the same as "uses what the partner knows". Only 1-3% of characters were ever wins, and that share hardly grew, which also says the reward wasn't finding a real signal to amplify.

## What it costs

| Setting | A on code (strong) | B on prose (strong) | B on mixed text, paired |
|---|---|---|---|
| Nothing forced | +0.006 to +0.008 | +0.004 to +0.012 | 1.75-1.78 |
| Forced 0.05, fair | +0.005 to +0.014 | +0.002 to +0.010 | 1.79-1.82 |
| Forced 0.15, fair | +0.009 to +0.011 | +0.004 to +0.009 | 1.80-1.82 |
| Forced 0.05 / 0.15, old | +0.007 to +0.018 | +0.007 to +0.029 | 1.83-1.86 |
| Reward λ 5 | +0.008 to +0.010 | +0.004 to +0.020 | 1.82-1.85 |

With fair weighting, forcing costs about the same as no forcing on the strong domains. That's the thing part 1's old weighting got wrong: it doubled or tripled the cost.

## The dial so far

Putting both parts together, for B's side (the direction that works):

| Setting | Free transfer (B code, real minus control, mean of 3) | Forced transfer (B reads A's answers) | Cost |
|---|---|---|---|
| Nothing forced | -0.101 | 13-34% | small |
| Forced 0.05, fair | -0.050 | 90-91% | small |
| Forced 0.15, fair | -0.036 | 97-100% | small |
| Forced, old weighting | -0.008 to -0.016 | on/off per seed | 2-3x |
| Reward λ 5-50 | -0.002 to -0.038, with everything worse | chance | grows fast |

So there is a real trade-off curve, and it's smooth enough to be a dial. Turning up fair forcing trades free transfer for forced transfer. Around 5% is the best balance I've found: reliable forced transfer, half the free transfer, no extra cost.

## What this does and doesn't show

It shows that forcing doesn't have to kill free transfer. Weighted fairly, a small share of a task that needs the partner keeps the channel open for that task and still leaves about half of the knowledge transfer the translator finds on its own, in every seed, at no extra cost. The part 1 result was mostly the loss bug.

It shows the partner-win reward, as defined, is harmful at every weight I tried. It doesn't show that rewarding partner use can't work, just that this definition rewards the wrong thing.

It doesn't fix the other direction. A still gets no prose from B through the expert bus, at any setting. Forcing teaches A to read B's recall answers (in one seed at 0.15 and most at 0.5), but never prose.

And the usual caveats: toy models, 3 seeds, 48 test documents per domain, one data mix.

## Where to next

1. **Turn the fair force dial down further** (0.01-0.02) to see whether free transfer comes back towards the unforced level while forced recall holds. That's the shape of the curve near the sweet spot.
2. **A reward that measures use of the partner's knowledge, not just dependence.** For example, reward how much better the real partner's information makes each prediction than a wrong partner's does (on the probabilities, not on the top guess, and against more than one wrong document). That's smooth, and noise shouldn't pass it.
3. **Why the expert bus never carries prose to A** (from the complementary run). It's still the main gap for the "both ways" part of the end goal.

## Reproduce

```bash
udisksctl mount -b /dev/disk/by-label/Data   # results/ are symlinks to the Data drive
# needs results/toy_complement (the models are copied from there)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_dial2.yaml 4 s0 c0 d0 s2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_dial2.yaml 4 s1 c1 d1 c2 d2
# both write summary.json, so rebuild it once both are done (CPU, runs nothing)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/toy_dial2.yaml 4 none
```

The thermal guard was `~/cmp-tests/dial2-guard.sh results/toy_dial2` in its own tmux session (outside the repo). Results are in `results/toy_dial2/summary.json`, with the reward runs' win shares in each `fused_<variant>.json` training history. Part 1's numbers are in `results/toy_dial/summary.json` and the unforced ones in `results/toy_complement/summary.json`.
