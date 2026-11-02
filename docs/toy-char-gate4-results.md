# Both sides open, links 0-2: does it hold up, and how early can it go?

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

## What this does and doesn't show

Links 0-2 with both sides open is now checked the same way the one-way result was: a second translator seed, and partners from another init. It holds on both, close to unit for unit. That's the setting I'll build on.

Taking link 2 away as well doesn't spread the transfer into links 0-1. A just stops getting prose. So the "latest link" habit isn't only the translator being lazy. At toy size the early layers of A' don't seem to carry much prose that A can use. What they do carry goes one way, A to A', with no dependence. That's the same direction the residual stream favoured in the [channel run](toy-char-channel-results.md).

The next question is whether training can make the translator use several links at once when it has the late ones too. That's the link-dropout run (Result 3, below, once it's in).

Caveats: toy models, 3 seeds, one gate start (0.3), one data mix, 48 test windows per domain.

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
```

I ran them through a queue script with the thermal guard (`~/cmp-tests/sweep-guard.sh <dir> <cfg> <tmux42> <tmux43>`) on each run. In the h units both slots are A', so A''s code is in both `m1_code` and `m2_code`. In the g units the A' "alone" number is that other-init A''s own.
