# After the gate start: which experts, earlier links, and the A-B pair

## Why this

The [gate start](toy-char-gate-results.md) run showed the one-way pattern was a starting-point problem. Start A's expert-bus readers part-open (0.3) and A takes prose from A' in every seed. But all of it came through the last link, and A came to lean on its partner. That run left three questions, and this doc answers all three:

1. Which of A''s experts does A read at the last link? A few prose specialists, or the whole layer?
2. Take the last link away. Can earlier experts carry prose once they're open, or does the hand-over just move to link 2?
3. Does the gate start work on the original A-B pair (char A, BPE B), where A has never got prose?

The first is a CPU check on the gate-start run's translators. The other two are new GPU runs. They use the same recipe and translator seed 0 as the gate-start run, and A's readers start at 0.3:

- `configs/toy_char_gate012.yaml` - expert bus on links 0-2 only. e units are A with A', f units (control) are A with another A.
- `configs/toy_ab_gate.yaml` - all 4 links. b units are A with B (the complementary run's pair), k units (control) are A with a code-heavy B. toy_complement's s and c units are the same pairs with gates shut.

That's 12 translators. All 12 passed the float64 leak check. The two runs went back to back on both cards (about 41 and 45 minutes) under the thermal guard.

Wrong partner - bits/char with a wrong-document partner minus alone. Big and positive means A has come to depend on its partner.

## Result 1 - A reads the last layer as a set, not prose specialists

Per e unit of the gate-start run, on the prose test windows (`docs/toy-char-gate/which_experts.py`):

- Read share - how much of A's link-3 attention lands on each of A''s 8 experts.
- Read ratio - read share over the share it would get if A spread its attention evenly across what it's allowed to see. 1 means "read in proportion to how often A' routes to it".
- Hide - A's prose bits with one A' expert hidden from A, minus fused.

| | e0 | e1 | e2 |
|---|---|---|---|
| Gain (fused minus alone) | -0.185 | -0.204 | -0.195 |
| Read ratio, range over experts | 0.78-1.37 | 0.61-1.23 | 0.51-1.26 |
| Null slot share | 27% | 30% | 31% |
| Hide the most-read expert | +0.069 | +0.081 | +0.055 |
| Hide the 4 most-read | +0.179 | +0.238 | +0.161 |
| Hide 7 of 8 | +0.222 | +0.206 | +0.197 |
| Single-hide costs, summed | +0.354 | +0.366 | +0.342 |
| Blank every expert to its mean code output | +0.345 | +0.371 | +0.332 |

A reads A''s experts roughly in proportion to routing, no expert above 1.4x. Hiding the top 4 takes most of the gain, and hiding 7 of 8 takes all of it. Single-hide costs add up to about twice the gain, so the experts back each other up rather than each carrying its own piece. Blanking all of them to their code-side mean is worse than no partner at all.

I also checked whether A leans on A''s prose specialists. That's the Spearman between an expert's hide cost (per unit of read share) and how much more A' routes to it on prose than on code: -0.05, 0.64 and 0.46. None of those is significant with 8 experts, and one has the wrong sign.

So A isn't picking out the experts that know prose. It reads A''s last layer whole, which fits the "late hand-over" picture.

## Result 2 - without the last link, half the gain and much less dependence

A on prose, links 0-2 only, bits/char paired minus alone:

| | Seed 0 | Seed 1 | Seed 2 | Wrong partner |
|---|---|---|---|---|
| Expert bus, all links (gate start) | -0.185 | -0.204 | -0.195 | +0.87 / +1.23 / +1.09 |
| **Expert bus, links 0-2 (gate start)** | **-0.097** | **-0.124** | **-0.072** | **+0.20 / +0.18 / +0.06** |
| Control (A + A), links 0-2 | -0.032 | -0.026 | -0.027 | about -0.02 |
| Residual, links 0-2, gates shut (channel run) | -0.130 | -0.191 | -0.022 | about +1.25 |

Attraction to A''s view: +0.020 / +0.026 / +0.013 nats per position (all 3/3), against zero for the control.

So the earlier experts can carry prose once they're open. A gets 2.5-4x the control's gain, but only about half of what it got with link 3. What's different is the dependence. A wrong partner costs A +0.06 to +0.20 bits/char, a fifth of what it did with link 3. On the residual stream, links 0-2 gave a wrong-partner cost of +1.25. Per bit gained, this is the least partner-dependent transfer I've had.

It still goes mostly through the latest link available:

| | All links | Only link 2 | All but link 2 |
|---|---|---|---|
| e0 | -0.097 | -0.067 | -0.021 |
| e1 | -0.124 | -0.095 | -0.016 |
| e2 | -0.072 | -0.048 | -0.024 |

Without link 2, what's left is about the control's general gain. The gates agree. A's link-2 gate ended at 0.30 / 0.21 / 0.11. It held or drifted down, where link 3 had grown to 0.58. Links 0-1 closed to 0.04-0.07, and every control gate closed.

A' on code: -0.176 / -0.042 / -0.120. That's e1 low again, as it was in the gate-start run (-0.033), against -0.201 in the char pair with gates shut. Two runs now, so A' losing its code gain in e1 when A's side opens looks like a real trade-off for that pair of models, not noise. A loses at most +0.008 on code and A' at most +0.005 on prose.

## Result 3 - the A-B pair: a little prose, no dependence

A on prose with B, all links, bits/char paired minus alone:

| | Seed 0 | Seed 1 | Seed 2 | Attraction | Wrong partner |
|---|---|---|---|---|---|
| A + B, gates shut (toy_complement s) | -0.015 | -0.014 | -0.012 | +0.002 (3/3) | about -0.01 |
| A + code-heavy B, gates shut (toy_complement c) | -0.020 | -0.018 | -0.018 | about 0 | about -0.01 |
| **A + B, A's gates start at 0.3** | **-0.050** | **-0.042** | **-0.043** | **+0.009 (3/3)** | **+0.000 / +0.006 / -0.001** |
| A + code-heavy B, same gate start (k) | -0.030 | -0.026 | -0.031 | about 0 | about -0.02 |
| A + A', same gate start (for scale) | -0.185 | -0.204 | -0.195 | +0.047 (3/3) | about +1.1 |

So the gate start does let some prose from B into A: about 0.017 bits/char beyond the control, in all 3 seeds, with attraction 4x the shut-gate run's. But it's a tenth of what A gets from A', and A doesn't come to depend on B at all.

Where it comes in (eval-time ablation, A on prose):

| | All links | Only link 3 | All but link 3 |
|---|---|---|---|
| b0 | -0.050 | -0.025 | -0.027 |
| b1 | -0.042 | -0.022 | -0.019 |
| b2 | -0.043 | -0.024 | -0.020 |
| k0 (control) | -0.030 | -0.001 | -0.029 |
| k1 (control) | -0.026 | -0.002 | -0.025 |
| k2 (control) | -0.031 | -0.004 | -0.027 |

In the controls link 3 carries nothing (-0.001 to -0.004), and the rest is the general "any partner" gain. In the b units, without link 3 A gets about that same general gain. The prose-specific part is link 3 again. A's link-3 gate ended at 0.16-0.19. That's down from 0.30, but well above the control's 0.02-0.04.

B's code gain didn't move: -0.131 / -0.105 / -0.135, against -0.132 / -0.109 / -0.126 with gates shut. A loses at most +0.015 on code.

## What this does and doesn't show

The good news for the end goal is Result 2. Experts in the middle of the two models can pass prose across once their readers are open. The bad news is that the translator still prefers the latest link it can get. A reads that link's experts as a set, not as prose specialists. Take the last link away and the transfer gets smaller, but A depends on its partner much less. That's closer to "experts sharing" than to "pass the partner's last layer through". It also makes partner dependence something I can choose with the links, not a fixed cost.

The A-B result partly answers option (c) from START HERE, whether A's prose gap with B is the tokenizer. The same A, the same recipe and the same gate start give -0.19 with a char prose model and -0.05 with a BPE prose model. B and A' are about equally good at prose on their own (1.89 vs 1.85-1.90 bits/char). So it's something about crossing tokenizers, or about B being a different architecture from A's family. It isn't B lacking prose. The cross-tokenizer transfer that does happen comes with no dependence at all.

Caveats: toy models, 3 seeds, one gate start (0.3), one data mix, 48 test windows per domain. The which-experts check has 8 experts per unit, so its rank correlations can't say much.

## Reproduce

```bash
udisksctl mount -b /dev/disk/by-label/Data   # results/ are symlinks to the Data drive
# needs results/toy_complement (all models are copied from there) and results/toy_char_gate (Result 1)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-gate/which_experts.py   # Result 1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate012.yaml 4 e0 f0 e2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_char_gate012.yaml 4 e1 f1 f2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_ab_gate.yaml 4 b0 k0 b2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_ab_gate.yaml 4 b1 k1 k2
for c in toy_char_gate012 toy_ab_gate; do
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/$c.yaml 4 none
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_partner_shift.py results/$c 8
done
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-channel/link_ablation.py \
  results/toy_char_gate012 ab_experts_012_rel_safe_text_ga3
UNITS=b0,b1,b2,k0,k1,k2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python docs/toy-char-channel/link_ablation.py \
  results/toy_ab_gate ab_experts_all_rel_safe_text_ga3   # I ran b and k separately; the k rows are in link_ablation_k.log
```

The thermal guards were `~/cmp-tests/sweep-guard.sh results/<cfg> configs/<cfg>.yaml <tmux42> <tmux43>`, one per run. In the ablation output the slot-2 model is always printed as A', so for the A-B run read it as B. The which-experts numbers are in `results/toy_char_gate/which_experts_ab_experts_all_rel_safe_text_ga3.json`.
