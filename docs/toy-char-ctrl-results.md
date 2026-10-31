# Controls for the char pair: the one-way pattern holds

## Why this

The [outside review](review-2026-10-31.md) pointed out that every translator so far started from the same initialisation and training order (`fused.seed` 0), and the code-heavy model was always in the first slot. So the char-pair pattern (A' gets code from A through the expert bus, A gets no prose from A') could have come from that shared start or from the slot order. This run is review step 1: the same unforced expert-bus recipe, with those two things changed and two new partners.

Units, all at translator seed 1 (models copied from the complementary run, nothing pretrained):

- e - A with A' (the char-pair e unit again, new translator seed).
- x - A' with A, slots swapped.
- g - A_i with A'_(i+1). A prose partner that doesn't share A's initial weights.
- f (control) - A_i with A_(i+1). Another code-heavy partner, so any gain here isn't prose knowledge.
- h (control) - A'_i with A'_(i+1). Another prose-heavy partner, the control for A''s code gain. The char-pair run didn't have one.

That's 15 translators, 3 seeds each, expert bus only. All 15 passed the float64 leak check. Both CMP 170HXs took about 115 minutes side by side under the thermal guard, peaking at 50°C core and 59°C memory.

Attraction - JS(alone, partner alone) - JS(paired, partner alone), in nats per position. How far the real partner moves the model towards the partner's own view.
Wrong-partner pull - the same with a wrong partner. Negative means the wrong partner pushes the model away (damage).

## Result 1 - A still gets no prose, in any arrangement

A on prose, means of 3 seeds:

| | Bits/char, paired minus alone | Attraction | Wrong-partner pull |
|---|---|---|---|
| e: A + A' | -0.019 | +0.0023 (3/3) | +0.0021 |
| x: A' + A (A in slot 2) | -0.018 | +0.0022 (3/3) | +0.0019 |
| g: A + A' from another init | -0.016 | +0.0022 (3/3) | +0.0019 |
| f: A + A (control) | -0.018 | -0.0002 (0/3) | ~0 |
| Char pair e, translator seed 0 | -0.018 | +0.0024 (3/3) | +0.0020 |

"3/3" means attraction's 95% interval is above zero in all three seeds.

The prose gain is the same as with a code-heavy partner (-0.018 bits/char), whatever the slot order, translator seed or partner init. A does drift a little towards A''s view, but a wrong partner moves it almost as much. So it's leaning towards a prose model's general style, not using what A' knows about the text in front of it. That's the same as at translator seed 0.

## Result 2 - A' still gets code from A, in every arrangement

A' on code, per seed and mean:

| | Bits/char, paired minus alone | Mean | Attraction (mean) | Wrong-partner pull (mean) |
|---|---|---|---|---|
| e: A + A' | -0.172 / -0.049 / -0.068 | -0.097 | +0.014 (3/3) | -0.013 |
| x: A' + A (A' in slot 1) | -0.155 / -0.207 / -0.071 | -0.144 | +0.024 (3/3) | -0.046 |
| g: A + A' from another init | -0.249 / -0.080 / -0.069 | -0.133 | +0.021 (3/3) | -0.039 |
| h: A' + A' (control) | -0.024 / -0.026 / -0.022 | -0.024 | +0.000 (0/3) | ~0 |
| Char pair e, translator seed 0 | -0.084 / -0.201 / -0.113 | -0.133 | +0.021 (3/3) | -0.033 |

A' gets 0.07 to 0.12 bits/char more on code from A than from another prose-heavy model, and its attraction to A's view is above zero in all 9 arrangements, against zero for the control.

## What this does and doesn't show

The one-way pattern isn't from the shared translator start, the slot order or the shared model init. Code goes from A into A' through the expert bus, and prose doesn't go from A' into A, in every arrangement I tried.

What did change is the size, from one translator seed to the next. e1 got -0.201 at translator seed 0 and -0.049 at seed 1, on the same two models. So the direction holds, but any single gain number is about plus or minus a factor of two. I'll read means over seeds and arrangements, not single units.

Side note: both controls give a small gain on the reader's weak domain (A on prose -0.018, A' on code -0.024) with zero attraction. That's the general "any partner helps a bit" effect, not knowledge, and it's the baseline the real gains need to beat.

It still doesn't say why prose doesn't travel. That's the channel run, queued after this one (`configs/toy_char_channel.yaml`: MoE-sum signal, and residual stream without the final link). And the usual caveats apply: toy models, one data mix, 48 test windows per domain.

## Reproduce

```bash
udisksctl mount -b /dev/disk/by-label/Data   # results/ are symlinks to the Data drive
# needs results/toy_complement (all models are copied from there)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_ctrl.yaml 4 e0 x0 g0 f0 h0 e2 x2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_ctrl.yaml 4 e1 x1 g1 f1 h1 g2 f2 h2
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/toy_char_ctrl.yaml 4 none
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_partner_shift.py results/toy_char_ctrl 8
```

The thermal guard was `~/cmp-tests/sweep-guard.sh results/toy_char_ctrl configs/toy_char_ctrl.yaml cc42 cc43` in its own tmux session. Results are in `results/toy_char_ctrl/summary.json` and `partner_shift.json`. For x units the model in slot 1 is A', so read `m1_code` and `m2_prose` there.
