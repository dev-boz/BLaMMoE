# Which channel carries prose into A: not the expert bus

> **Update, 2026-11-01 ([gate start](toy-char-gate-results.md)).** It was opening, not combining. With A's readers started part-open (0.3), A gets prose through the per-expert bus in all 3 seeds (-0.19 to -0.20 bits/char against -0.03 for the control), still all through the last link.

## Why this

Through the per-expert bus, A has never got prose from a partner, and the [controls](toy-char-ctrl-results.md) show that holds under a new translator seed, swapped slots and a partner from another init. But through the full residual stream, A takes nearly everything from A' ([char pair](toy-char-pair-results.md)), mostly through the final link. So is the block "prose doesn't travel into a code model", or something about the per-expert bus? This run is review step 2: the same A + A' pairs and unforced recipe as the char pair (translator seed 0, so those runs are the comparison), on two new channels.

- MoE sum - each layer's router-weighted MoE output, one vector per position, instead of each expert's output as a separate entry on the bus.
- Residual without the final link - the residual stream at links 0-2 only. The char-pair hand-over mostly came through the final link, so anything left has to pass through the reader's own last layer.
- e units - A with A'. f units (control) - A with another A. h units (control) - A' with another A'.

That's 18 translators. All 18 passed the float64 leak check. Both cards ran for about 80 minutes under the thermal guard, peaking at 49°C core and 58°C memory.

## Result 1 - A gets prose through both new channels, in 2 seeds of 3

A on prose, bits per character paired minus alone (negative is better):

| | Seed 0 | Seed 1 | Seed 2 | Wrong partner, mean |
|---|---|---|---|---|
| Expert bus (char pair) | -0.017 | -0.019 | -0.018 | -0.013 |
| **MoE sum** | **-0.171** | **-0.191** | -0.017 | +0.85 |
| **Residual, no final link** | **-0.130** | **-0.191** | -0.022 | +1.25 |
| Residual, all links (char pair) | -0.285 | -0.306 | -0.282 | +5.58 |
| Control partner (A + A), both new channels | -0.016 to -0.022 | | | ~0 |

"Wrong partner" is bits/char with a wrong-document partner minus alone. Big and positive means A has come to depend on its partner.

Attraction to A''s view (nats per position):
- MoE sum: +0.044 / +0.048 / +0.002.
- Residual without the final link: +0.029 / +0.046 / +0.003.
- Both controls: zero.

In seeds 0 and 1, A's reader opens for the first time in any run. Its gate is -0.30 and -0.34 on the last MoE link, and 0.16 and 0.38 on the last residual link left. In seed 2 A's gates stay around 0.01 on both channels, and A gets nothing.

So "prose never goes into a code model" was wrong. Prose does go into A when it's offered as one summed vector per position. Through separate expert entries it doesn't.

## Result 2 - it's a late hand-over again, just one layer earlier

Taking links out at eval time on the trained translators (CPU, A on prose, change from alone):

| | All links | Only the last link | Everything but the last link |
|---|---|---|---|
| MoE sum, e0 | -0.171 | -0.151 (link 3) | -0.003 |
| MoE sum, e1 | -0.191 | -0.180 (link 3) | -0.006 |
| Residual no final, e0 | -0.130 | -0.113 (link 2) | -0.006 |
| Residual no final, e1 | -0.191 | -0.180 (link 2) | +0.003 |

All of it comes through whichever link sits latest. On the residual stream, take away the final link and the translator moves the hand-over to link 2. A also becomes dependent on its partner: a wrong partner costs it 0.85-1.25 bits/char. That's less than the 5.6 with every residual link, but it's still the same kind of thing, a late part of A''s computation passed straight in. A loses nothing on code (at most +0.012 bits/char).

## Result 3 - A' gets more code through the new channels too

A' on code (mean of 3 seeds):

| | Paired minus alone | Attraction | A'+A' control |
|---|---|---|---|
| Expert bus (char pair) | -0.133 | +0.021 (3/3) | (none in that run) |
| MoE sum | -0.177 | +0.030 (3/3) | -0.040 |
| Residual, no final link | -0.275 | +0.046 (3/3) | -0.042 |
| Residual, all links (char pair) | -0.414 | +0.077 (3/3) | |

This one works in all 3 seeds on every channel. On the MoE sum it mostly comes through link 2 (link 3 in seed 2).

## What this does and doesn't show

The one-way pattern belongs to the per-expert bus, not to prose or to A. A summed late channel carries prose into A in 2 of 3 seeds. When it does, it looks like the residual hand-over: one late link, and a model that leans on its partner.

For the end goal that's awkward. The per-expert bus is the expert-to-expert channel, and it's the one that only works one way. The channels that work both ways look like "let the partner's late layer through", not experts sharing.

Why the per-expert bus blocks prose isn't settled. Two guesses:
- Combining. A''s prose view may only exist as the router's mix of several experts, and A's reader would have to rebuild that from separate entries. The MoE sum hands it over ready-made.
- Opening. A's reader has to open before it can learn anything, and it doesn't always (seed 2 never opened on either new channel, though it did with every residual link). On the expert bus it may just never get the push.

A non-zero gate start for A on the expert bus would tell those apart. If A then gets prose through separate expert entries, it was the opening. If not, it's the combining.

Caveats: toy models, 3 seeds, one data mix, 48 test windows per domain. Seed 2 failing on both channels could be the s2/d2 models or this translator seed.

## Reproduce

```bash
udisksctl mount -b /dev/disk/by-label/Data   # results/ are symlinks to the Data drive
# needs results/toy_complement (all models are copied from there)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_channel.yaml 4 e0 f0 h0 e2 h2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_channel.yaml 4 e1 f1 h1 f2
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/toy_char_channel.yaml 4 none
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_partner_shift.py results/toy_char_channel 8
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-channel/link_ablation.py   # Result 2
```

The thermal guard was `~/cmp-tests/sweep-guard.sh results/toy_char_channel configs/toy_char_channel.yaml cn42 cn43`. Results are in `results/toy_char_channel/summary.json` and `partner_shift.json`. The final gates are in the sweep logs (`rx_a` is A's reader, the first number of each pair).
