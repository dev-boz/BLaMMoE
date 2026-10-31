# Start A's gates open: prose goes through the expert bus after all

## Why this

The [channel run](toy-char-channel-results.md) showed A gets prose from A' through a summed MoE output or the residual stream, but never through the per-expert bus. That left two explanations:

1. Opening - A's reader on the expert bus never gets pushed open, so it never gets the chance to learn anything.
2. Combining - A's reader can't rebuild A''s prose view from separate expert entries, while a summed vector hands it over ready-made.

This run tells them apart. It's the char pair's expert-bus run exactly (same models, translator seed 0, same unforced recipe and training draws) with one change: A's readers start part-open, at a gate of 0.3 on every link, about where A's reader went on the MoE-sum channel. A''s readers start shut as before.

Gate start - the gate value each reader begins training at (tanh of the gate parameter). Every earlier run started at 0, fully shut.

- e units - A with A'.
- f units (control) - A with another A, same gate start. Any gain here comes from the open start, not from prose knowledge.

That's 6 translators. All 6 passed the float64 leak check. Both cards ran for about 43 minutes under the thermal guard, peaking at 49°C core and 59°C memory. The new code is `FusedPair.start_gates` (a `gate_start_a` option on the variant). It draws no randomness, so everything else about the start matches the char-pair run.

## Result 1 - it was opening

A on prose, bits per character paired minus alone:

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Expert bus, gates start shut (char pair) | -0.017 | -0.019 | -0.018 |
| **Expert bus, A's gates start at 0.3** | **-0.185** | **-0.204** | **-0.195** |
| Control (A + A), same gate start | -0.030 | -0.030 | -0.034 |
| MoE sum, gates start shut (channel run) | -0.171 | -0.191 | -0.017 |

Attraction to A''s view is +0.047 / +0.050 / +0.045 nats per position (all 3/3), against zero for the control.

So A can use prose from separate expert entries. All it needed was a reader that started open. It works in all 3 seeds, including seed 2, which never opened on either of the channel run's channels. It's also a bit more than the MoE sum gave.

The gates show what happened. In every e unit, A's gate on the last link grew from 0.30 to 0.56-0.59 over training, and its gates on links 0-2 closed to about 0.02-0.08. In every f unit, all of A's gates closed (0.00-0.07). So the loss isn't just leaving an open gate alone. It opens the last one further, and only when the partner knows prose.

Why didn't it open from zero? My best guess, untested. At gate 0 the reader's output projection gets no gradient (it's multiplied by the gate), so only the gate itself can move. It moves only if the reader's random first read happens to point somewhere useful. For A' reading code that's evidently been enough, and for A reading prose it hasn't.

## Result 2 - still a late hand-over

Taking links out at eval time (CPU, A on prose, change from alone):

| | All links | Only link 3 | All but link 3 |
|---|---|---|---|
| e0 | -0.185 | -0.167 | -0.005 |
| e1 | -0.204 | -0.179 | -0.008 |
| e2 | -0.195 | -0.164 | +0.004 |

Like the other channels, it's all the last link. A also comes to lean on its partner: a wrong-document partner costs it +0.87 / +1.23 / +1.09 bits/char against alone. A loses almost nothing on code (+0.004 to +0.006), and A' loses almost nothing on prose (up to +0.010).

## Result 3 - A' code moved too, in one seed

A' on code: -0.235 / -0.033 / -0.195 bits/char, against -0.084 / -0.201 / -0.113 in the char pair. The only difference is A's gate start, so opening A's side changes what A''s side learns. In e1 it nearly wiped out A''s code gain. Three seeds can't say whether that's a trade-off or noise.

## What this does and doesn't show

The one-way pattern was a starting-point problem, not a property of prose, of A, or of the expert bus. With A's reader started part-open, prose goes from A' into A through separate expert entries, in every seed.

The "direction effect" is now well and truly dead. What's left for the end goal is the shape of the transfer. Every channel, once it works, puts everything through the last link, and the reader comes to depend on its partner. That's the partner's last-layer experts being passed through, not experts in the middle of the two models working together.

Next ideas, none agreed:
- Gate start with the last link removed (expert bus, links 0-2). Can earlier experts carry prose once they're open, or does it just move to link 2 like the residual stream did?
- Which of A''s experts A reads at link 3 (CPU, from the bus attention and expert tags). Is it a few prose experts, or everything at once?
- Gate start on the A-B pair from the complementary run. Does A get prose from B now?

Caveats: toy models, 3 seeds, one gate start (0.3), one data mix, 48 test windows per domain.

## Reproduce

```bash
udisksctl mount -b /dev/disk/by-label/Data   # results/ are symlinks to the Data drive
# needs results/toy_complement (all models are copied from there)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate.yaml 4 e0 f0 e2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate.yaml 4 e1 f1 f2
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate.yaml 4 none
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_partner_shift.py results/toy_char_gate 8
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-channel/link_ablation.py \
  results/toy_char_gate ab_experts_all_rel_safe_text_ga3   # Result 2
```

The thermal guard was `~/cmp-tests/sweep-guard.sh results/toy_char_gate configs/toy_char_gate.yaml cg42 cg43`. The gate trajectories are in the sweep logs (`rx_a` = A's reader, the first number of each pair).
