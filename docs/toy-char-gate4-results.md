# Both sides open, links 0-2: does it hold up, how early can it go, and does link dropout spread it?

## Why this

The best shape so far is both sides' readers starting at 0.3 on links 0-2 only ([gate3, Result 3](toy-char-gate3-results.md)). Both directions work at once, with about half the gain of the last link and a small fraction of the dependence. That was one translator seed, with each A paired with its own A'. Before building on it I wanted the same checks the one-way result got in [toy_char_ctrl](toy-char-ctrl-results.md), and one step further back:

- `configs/toy_char_gate2way012s1.yaml` - links 0-2, both open, translator seed 1 (new init and training draw order). e units are A with A' as before. g units are A with an A' from another init. f and h are the usual controls (A + another A, A' + another A').
- `configs/toy_char_gate2way01.yaml` - both open, links 0-1 only, translator seed 0. Units e, f, h. If link 2 is the latest link the translator can get, what does it do with only the first two?

All 21 translators passed the float64 leak check. The runs went back to back on both cards (59 and 80 minutes) under the thermal guard, overnight.

Wrong partner - bits/char with a wrong-document partner minus alone. Big and positive means the model has come to depend on its partner.

## Result 1 - links 0-2 holds up at a new translator seed and with other partners

Bits/char paired minus alone:

| | A on prose | A' on code | Wrong partner, A / A' |
|---|---|---|---|
| Translator seed 0 (gate3 Result 3) | -0.110 / -0.121 / -0.079 | -0.214 / -0.116 / -0.142 | +0.17 / +0.17 / +0.06, +0.38 / +0.09 / +0.16 |
| **Translator seed 1 (e)** | **-0.085 / -0.118 / -0.097** | **-0.211 / -0.112 / -0.140** | **+0.17 / +0.22 / +0.15, +0.30 / +0.12 / +0.16** |
| **Partner from another init (g)** | **-0.130 / -0.091 / -0.094** | **-0.176 / -0.142 / -0.153** | **+0.26 / +0.18 / +0.15, +0.30 / +0.11 / +0.32** |
| Controls, seed 1 (f / h) | -0.028 / -0.031 / -0.029 | -0.04 to -0.08, mean -0.058 | about 0 |

Attraction: A +0.017 to +0.026 and A' +0.019 to +0.032 in all 6 e and g units. The f controls are zero, and h is zero or below except h0 (+0.003 at most).

So the result isn't one lucky translator. At seed 1 the e numbers land within 0.025 of seed 0, unit for unit. That includes A''s weak spot in e1 (-0.112 against -0.116). The g units show it isn't something special about each A and the A' it was trained beside, either. Any A with any A' gets about the same.

The e1 weak spot is the pair, not either model. In g0 the same A' (seed 1's) gets -0.176 of code from a different A. In g1 the same A gives -0.142 of code to a different A'. Only the e1 pairing comes out low, at both translator seeds.

Eval-time ablation (A on prose / A' on code):

| | Only link 2 | All but link 2 |
|---|---|---|
| e0 | -0.059 / -0.152 | -0.017 / -0.019 |
| e1 | -0.093 / -0.075 | -0.010 / -0.040 |
| e2 | -0.071 / -0.096 | -0.021 / -0.027 |
| g0 | -0.100 / -0.131 | -0.014 / -0.022 |
| g1 | -0.057 / -0.098 | -0.017 / -0.031 |
| g2 | -0.061 / -0.102 | -0.026 / -0.036 |

Same as seed 0. It's mostly link 2, and links 0-1 carry about the controls' general gain. Link-2 gates ended at 0.18-0.28 for A and 0.23-0.47 for A'. Links 0-1 sit at 0.03-0.15, and the f controls close everything to 0.02-0.07. Strong-domain costs are small: A on code +0.003 to +0.010, A' on prose +0.004 to +0.006.

## Result 2 - links 0-1 only: A gets nothing beyond the control, A' a little

| | A on prose | A' on code | Wrong partner, A / A' |
|---|---|---|---|
| **Both open, links 0-1 (e)** | **-0.030 / -0.031 / -0.035** | **-0.109 / -0.072 / -0.077** | **-0.02 / -0.01 / -0.02, +0.06 / +0.01 / +0.01** |
| Controls (f / h) | -0.031 / -0.027 / -0.027 | -0.03 to -0.05, mean -0.045 | about 0 |
| Both open, links 0-2 (for scale) | -0.110 / -0.121 / -0.079 | -0.214 / -0.116 / -0.142 | |

Attraction: A +0.004 / +0.005 / +0.004 and A' +0.017 / +0.009 / +0.009, all 3/3. The controls are zero or below.

With only the first two links, A's prose gain is the control's "any partner helps" gain. Its attraction is above zero but a fifth of the links-0-2 value, and nothing shows up in bits. A''s code still gets 0.03-0.06 beyond its control, with no dependence at all. The gates agree. A's readers closed to 0.05-0.08 on both links. A''s link-1 reader stayed open in e0 (0.31) and drifted to 0.12-0.15 in e1 and e2.

Eval-time ablation, A' on code:

| | Only link 0 | Only link 1 |
|---|---|---|
| e0 | -0.023 | -0.081 |
| e1 | -0.017 | -0.054 |
| e2 | -0.043 | -0.033 |

So A''s code mostly comes through link 1, the later of the two, except in e2 where it splits. Over three runs the pattern is the same: whatever links are on offer, the translator uses the latest one. But how much gets across depends on how deep that link is. Link 3 gives about 0.17 of prose beyond the control, link 2 about 0.08, link 1 nothing. Code into A' falls off more slowly: about 0.19, 0.10 and 0.04.

## Result 3 - link dropout: less dependence, but it doesn't spread A's prose over depth

`configs/toy_char_gate2way_drop.yaml` - same as the both-open runs (translator seed 0, units e, f, h), but during training each link is switched off with probability 0.5 at every step (`link_drop`, its own seeded RNG, so the torch draws don't change). One variant has all links, one has links 0-2. At eval every link is on. All 18 translators passed the leak check (125 minutes on both cards). The idea was that a translator that can't count on the latest link would have to learn to use the others as well.

| | A on prose | A' on code | Wrong partner, A / A' |
|---|---|---|---|
| All links, no dropout (gate3) | -0.187 / -0.197 / -0.199 | -0.321 / -0.224 / -0.242 | +0.86 / +1.26 / +1.03, +1.11 / +0.82 / +0.68 |
| **All links, dropout 0.5** | **-0.158 / -0.177 / -0.167** | **-0.257 / -0.185 / -0.201** | **+0.53 / +0.75 / +0.65, +0.78 / +0.57 / +0.42** |
| Links 0-2, no dropout (gate3) | -0.110 / -0.121 / -0.079 | -0.214 / -0.116 / -0.142 | +0.17 / +0.17 / +0.06, +0.38 / +0.09 / +0.16 |
| **Links 0-2, dropout 0.5** | **-0.053 / -0.066 / -0.063** | **-0.154 / -0.092 / -0.104** | **+0.02 / +0.03 / +0.03, +0.14 / +0.04 / +0.06** |
| Controls, dropout (f / h) | -0.020 to -0.031 | -0.03 to -0.05, mean -0.040 | about 0 |

Attraction is above zero in all 12 e rows (A +0.010 to +0.042, A' +0.013 to +0.045). The f controls are zero, and h is zero apart from three small ones (+0.003 at most).

Eval-time ablation:

| | A on prose, all but the latest link | A' on code, all but the latest link |
|---|---|---|
| All links, no dropout (all but link 3) | -0.008 / +0.005 / +0.003 | -0.070 / +0.018 / +0.002 |
| **All links, dropout (all but link 3)** | **-0.020 / -0.020 / -0.023** | **-0.123 / -0.063 / -0.083** |
| Links 0-2, no dropout (all but link 2) | -0.019 / -0.017 / -0.024 | -0.011 / -0.038 / -0.033 |
| **Links 0-2, dropout (all but link 2)** | **-0.024 / -0.022 / -0.021** | **-0.054 / -0.045 / -0.039** |

For A it didn't work. Without the latest link A gets the control's general gain and nothing more, with or without dropout. A's link-3 gate still opened to 0.53-0.57, and its links 0-2 stayed at 0.06-0.10, the same as the controls. Half the time training couldn't use link 3, and A's reader learnt to wait for it rather than find prose somewhere else.

A' did spread a little. With all links, A''s code without link 3 went from about nothing to -0.06 to -0.12, mostly through link 2 (only link 2: -0.108 / -0.045 / -0.069), and its link-2 gate stayed open at 0.21-0.33. With links 0-2, link 1 picked up a small share (only link 1: -0.041 / -0.032 / -0.019). That fits Result 2: the early layers have something for A' (code from A) and nothing for A (prose from A').

What dropout mostly does is turn down the whole exchange. Gain and dependence fall together. With all links the gain drops by 10-20% and the wrong-partner cost by about a third. With links 0-2 the gain beyond the control halves (A about 0.035 instead of 0.08) and the wrong-partner cost almost disappears for A (+0.02 to +0.03). Per bit gained, links 0-2 with dropout is the least partner-dependent two-way transfer I've had. But I could already trade gain for dependence by picking links, and dropout doesn't buy anything new for A. Strong-domain costs went up a little: A on code +0.007 to +0.014 (no dropout: up to +0.010), A' on prose up to +0.015.

One more detail. The dropout RNG is seeded with the translator seed, so every unit in a variant saw the same drop pattern. That's fine for comparing e with its controls, but a second drop seed would tell me whether the pattern matters.

## What this does and doesn't show

Links 0-2 with both sides open is now checked the same way the one-way result was: a second translator seed, and partners from another init. It holds on both, close to unit for unit. That's the setting I'll build on.

Taking link 2 away as well doesn't spread the transfer into links 0-1. A just stops getting prose. So the "latest link" habit isn't only the translator being lazy. At toy size the early layers of A' don't seem to carry much prose that A can use. What they do carry goes one way, A to A', with no dependence. That's the same direction the residual stream favoured in the [channel run](toy-char-channel-results.md).

Link dropout doesn't change that (Result 3). With half the training steps missing the latest link, A still takes prose only through the latest link, and just takes less of it. A''s code spreads a little, to the next link down. Dropout lowers dependence, but so does picking the links, and that's cheaper. So for A, getting prose out of the middle of A' isn't something the translator is choosing not to do. At toy size the middle layers just don't hold prose A can use. To test "experts sharing" in the middle I probably need bigger models, where middle layers specialise more. That's an argument for the small-level pilot.

Caveats: toy models, 3 seeds, one gate start (0.3), one data mix, 48 test windows per domain. One dropout rate (0.5) and one drop seed.

## Reproduce

```bash
udisksctl mount -b /dev/disk/by-label/Data   # results/ are symlinks to the Data drive
# needs results/toy_complement (all models are copied from there)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate2way01.yaml 4 e0 f0 h0 e2 h2 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate2way01.yaml 4 e1 f1 h1 f2; wait
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate2way012s1.yaml 4 e0 g0 f0 h0 e2 g2 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate2way012s1.yaml 4 e1 g1 f1 h1 f2 h2; wait
for c in toy_char_gate2way01 toy_char_gate2way012s1; do
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/$c.yaml 4 none
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_partner_shift.py results/$c 8
done
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-channel/link_ablation.py \
  results/toy_char_gate2way01 ab_experts_01_rel_safe_text_ga3b3
UNITS=e0,e1,e2,g0,g1,g2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-channel/link_ablation.py \
  results/toy_char_gate2way012s1 ab_experts_012_rel_safe_text_ga3b3
# Result 3 (link dropout); same pattern, units e f h
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate2way_drop.yaml 4 e0 f0 h0 e2 h2 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate2way_drop.yaml 4 e1 f1 h1 f2; wait
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate2way_drop.yaml 4 none
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_partner_shift.py results/toy_char_gate2way_drop 8
for v in ab_experts_all_rel_safe_text_ga3b3_ld5 ab_experts_012_rel_safe_text_ga3b3_ld5; do
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-channel/link_ablation.py results/toy_char_gate2way_drop $v
done
```

I ran them through a queue script with the thermal guard (`~/cmp-tests/sweep-guard.sh <dir> <cfg> <tmux42> <tmux43>`) on each run. In the h units both slots are A', so A''s code is in both `m1_code` and `m2_code`. In the g units the A' "alone" number is that other-init A''s own.
