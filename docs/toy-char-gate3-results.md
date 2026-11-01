# Both sides open: two-way transfer at once

## Why this

In the [gate start](toy-char-gate-results.md) run only A's readers started part-open. A got prose from A' in every seed. But in seed 1, A''s code gain from A nearly disappeared (-0.201 to -0.033), and it was low again with links 0-2 only ([follow-ups](toy-char-gate2-results.md)). So can both directions work at the same time? Or does one side opening crowd out the other?

These runs start both sides' readers at 0.3 on every link. Everything else matches the earlier gate-start runs (translator seed 0, unforced expert-bus recipe):

- `configs/toy_char_gate2way.yaml` - A with A'. e units are the real pair. f units (A + another A) are A's control. h units (A' + another A') are A''s control.
- `configs/toy_ab_gate2way.yaml` - A with B. b units are the real pair. k units (A + code-heavy B) are A's control. j units (prose-heavy A' + B) are B's control.
- `configs/toy_char_gate2way012.yaml` - A with A' again, expert bus on links 0-2 only (Result 3).

All 27 translators passed the float64 leak check. The runs went back to back on both cards (75, 75 and 66 minutes) under the thermal guard.

Wrong partner - bits/char with a wrong-document partner minus alone. Big and positive means the model has come to depend on its partner.

## Result 1 - A + A': both directions at full size, but it's a swap of last layers

Bits/char paired minus alone:

| | A on prose | A' on code |
|---|---|---|
| Both gates start shut (char pair) | -0.017 / -0.019 / -0.018 | -0.084 / -0.201 / -0.113 |
| Only A's start open (gate start) | -0.185 / -0.204 / -0.195 | -0.235 / -0.033 / -0.195 |
| **Both start open** | **-0.187 / -0.197 / -0.199** | **-0.321 / -0.224 / -0.242** |
| Controls, both open (f for A, h for A') | -0.029 / -0.026 / -0.032 | -0.04 to -0.12, mean -0.074 |

Attraction (both open): A +0.048 / +0.050 / +0.046 and A' +0.054 / +0.042 / +0.043, all 3/3. The f controls are zero. The h controls are small (+0.008 at most, half of them above zero).

So nothing gets crowded out. A gets the same prose as when only its side was open. A' gets more code than in any earlier expert-bus run, including e1 (-0.224). The seed-1 "trade-off" wasn't a trade-off. A''s reader just never opened from zero in that unit once A's side had taken over.

But look at how it's done:

| | A wrong partner | A' wrong partner | Link-3 gates [A, A'] |
|---|---|---|---|
| e0 | +0.86 | +1.11 | 0.59, 0.61 |
| e1 | +1.26 | +0.82 | 0.61, 0.57 |
| e2 | +1.03 | +0.68 | 0.56, 0.52 |

Both sides open link 3 to about 0.55-0.6 and close links 0-2. Each comes to lean heavily on the other. Eval-time ablation (A on prose / A' on code):

| | Only link 3 | All but link 3 |
|---|---|---|
| e0 | -0.171 / -0.233 | -0.008 / -0.070 |
| e1 | -0.182 / -0.186 | +0.005 / +0.018 |
| e2 | -0.174 / -0.205 | +0.003 / +0.002 |

It's all link 3, except some of e0's A' code, which goes through link 2 (its link-2 gate is the only one that stayed open, at 0.21). So the two models trade last layers. That's two-way, and it's what the loss wants, but it isn't experts in the middle working together. Strong-domain costs stay small: A on code +0.002 to +0.008, A' on prose +0.005 to +0.014.

## Result 2 - A-B: B's side opening adds only generic gain

| | A on prose | B on code | B attraction | B wrong partner |
|---|---|---|---|---|
| Both shut (toy_complement s) | -0.015 / -0.014 / -0.012 | -0.132 / -0.109 / -0.126 | +0.038 to +0.042 | +0.21 to +0.28 |
| Only A open (toy_ab_gate) | -0.050 / -0.042 / -0.043 | -0.131 / -0.105 / -0.135 | +0.036 to +0.045 | +0.24 to +0.33 |
| **Both open** | **-0.042 / -0.045 / -0.045** | **-0.164 / -0.140 / -0.161** | **+0.046 to +0.056** | **+0.28 to +0.41** |
| A's control k, both open | -0.030 / -0.023 / -0.029 | | | |
| B's control: shut (toy_complement d) / both open (j) | | -0.024 / -0.025 / -0.016 shut, -0.080 / -0.070 / -0.068 open | +0.012 to +0.014 (j) | +0.04 to +0.06 (j) |

A's side doesn't change. It still gets about 0.017 bits/char of prose beyond its control, and it doesn't depend on B. B gets 0.03 more code than with its gates shut. But its control gets 0.05 more too, so beyond the control B gains a bit less than with gates shut (-0.07 to -0.09 against -0.08 to -0.11), and it depends on A more. One caveat on the j control: A' is 10% code, and B's open reader does pick up a little from it (attraction +0.013), so j isn't a pure null.

So on the cross-tokenizer pair, starting B's side open doesn't help. B's reader opened from zero on its own in every earlier run, and starting it open just adds generic "any partner helps" gain and dependence. B's link-3 gate ended at 0.41-0.47, A's at 0.20-0.22.

The prose-specific part of A's gain still comes through link 3 (only link 3: -0.022 to -0.026; all but link 3: -0.016 to -0.022, about the control's general gain). So does most of B's code (-0.091 to -0.110 through link 3 alone).

## Result 3 - both open, links 0-2 only

All 9 translators passed the leak check (66 minutes on both cards, peaks 51°C core and 58°C memory).

| | A on prose | A' on code | Wrong partner, A / A' |
|---|---|---|---|
| Both open, all links (Result 1) | -0.187 / -0.197 / -0.199 | -0.321 / -0.224 / -0.242 | +0.68 to +1.26 |
| Only A open, links 0-2 | -0.097 / -0.124 / -0.072 | -0.176 / -0.042 / -0.120 | A +0.06 to +0.20 |
| **Both open, links 0-2** | **-0.110 / -0.121 / -0.079** | **-0.214 / -0.116 / -0.142** | **A +0.17 / +0.17 / +0.06, A' +0.38 / +0.09 / +0.16** |
| Controls, both open, links 0-2 (f / h) | -0.031 / -0.029 / -0.031 | -0.04 to -0.07, mean -0.057 | about 0 |

Attraction: A +0.022 / +0.026 / +0.012 and A' +0.035 / +0.019 / +0.022, all 3/3. The controls are zero or below.

This is the best shape so far. Both directions work at once, in every seed. Beyond the controls, A gets about 0.08 bits/char of prose and A' about 0.10 of code. That's roughly half of what the last link gives, for a third to a tenth of the dependence. A' in e1 is the weakest again (-0.116), but it's well clear of its control this time.

Eval-time ablation (A on prose / A' on code):

| | Only link 2 | All but link 2 |
|---|---|---|
| e0 | -0.080 / -0.161 | -0.019 / -0.011 |
| e1 | -0.097 / -0.073 | -0.017 / -0.038 |
| e2 | -0.050 / -0.099 | -0.024 / -0.033 |

It's mostly link 2, the latest link on offer, but links 0-1 carry a bit more than in any all-links run, mainly for A''s code in e1 and e2. The gates agree. Link 2 ended at 0.26 / 0.21 / 0.10 for A and 0.51 / 0.24 / 0.30 for A'. Links 0-1 sit at 0.03-0.11, and the controls close everything. Strong-domain costs are small: A on code +0.004 to +0.010, A' on prose at most +0.012.

## What this does and doesn't show

The gate start fixes the one-way problem completely on the same-tokenizer pair. Both directions work at once, at full size, in every seed. Neither one costs the other anything. The "trade-off" in seed 1 was a reader that never opened.

What I get, though, is two models handing each other their last layers. Each side comes to lean on the other: a wrong partner costs +0.7 to +1.3 bits/char. For the end goal of experts sharing freely, that's the wrong shape. Leaving out the last link changes the shape (Result 3). Both directions still work at once, with about half the gain and a small fraction of the dependence. Most of it still goes through the latest link available, so the translator's taste for "as late as possible" is still there. But at link 2, two middle layers are exchanging expert outputs, and neither model leans on the other much. For the end goal, that's the setting to build on.

On the cross-tokenizer pair, nothing about the starting point closes A's prose gap. The gap is the crossing.

Caveats: toy models, 3 seeds, one gate start (0.3), one data mix, 48 test windows per domain.

## Reproduce

```bash
udisksctl mount -b /dev/disk/by-label/Data   # results/ are symlinks to the Data drive
# needs results/toy_complement (all models are copied from there)
for c in toy_char_gate2way toy_ab_gate2way; do   # unit names: e f h / b k j
  u1="e0 f0 h0 e2 h2"; u2="e1 f1 h1 f2"
  [ $c = toy_ab_gate2way ] && u1="b0 k0 j0 b2 j2" && u2="b1 k1 j1 k2"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/$c.yaml 4 $u1 &
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/$c.yaml 4 $u2; wait
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/$c.yaml 4 none
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_partner_shift.py results/$c 8
done
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-channel/link_ablation.py \
  results/toy_char_gate2way ab_experts_all_rel_safe_text_ga3b3
UNITS=b0,b1,b2,j0,j1,j2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-channel/link_ablation.py \
  results/toy_ab_gate2way ab_experts_all_rel_safe_text_ga3b3
```

The links-0-2 run is the same with `configs/toy_char_gate2way012.yaml` (units e f h) and `link_ablation.py results/toy_char_gate2way012 ab_experts_012_rel_safe_text_ga3b3`. I ran them through a queue script with the thermal guard (`~/cmp-tests/sweep-guard.sh <dir> <cfg> <tmux42> <tmux43>`) on each run. In the h units both slots are A', so A''s code is in both `m1_code` and `m2_code`. In the j units B is slot 2 (`m2_code`). In the ablation output slot 2 is always printed as A'.
