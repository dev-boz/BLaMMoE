# "Different, not better", and why A never gets prose (CPU diagnostics, no training)

## Why this

Two things came out of the dials.

First, a change of aim. What I want from pairing isn't necessarily a *better* model, just a *different* one. But "different" on its own is too easy. Dial 2's reward at λ 50 made every prediction move, and made both models worse, just as much with a wrong partner. So I need a way to say a paired model is different *because of what its partner knows*, without needing it to score better.

Second, the A-side gap. B (the prose model) learns code from A through the expert bus, in every seed. A (the code model) never gets prose from B that way. The guess was that B's prose knowledge isn't in its experts at all, so the bus has nothing to carry.

Both of these can be checked on the translators and models I already have, on CPU, with no new training. It's the same held-out test documents as before (48 prose, 48 code).

## The measure

Everything is compared on next-character predictions, at the positions where both tokenizers have a boundary (the same view as the old diversity analysis). JS is Jensen-Shannon divergence, in nats, averaged per position:

- Shift - JS between the paired model and the same model alone. How different it is.
- Shift with a wrong partner - the same, but the partner is reading a different document.
- Dependence - JS between the paired model with the real partner and with a wrong one. How much the change depends on what the partner is reading.
- Toward - how much closer the real partner pulls the model to what *the partner itself* predicts at that spot, compared with a wrong partner. Positive means it's leaning towards the partner's view.

So "different because of the partner" means dependence and toward both clearly above zero, and above the same-domain control partner.

Script: `scripts/toy_partner_shift.py`, run on every translator in `results/toy_complement`, `results/toy_dial` and `results/toy_dial2` (90 in all). Output in each root's `partner_shift.json`.

## Result 1 - the measure tells knowledge from damage

The λ 50 reward is the test case: lots of change, none of it useful. Mean of 3 seeds:

| | Shift | Dependence | Toward | Bits/char vs alone |
|---|---|---|---|---|
| B on code, unforced (expert bus) | 0.042 | 0.106 | **+0.089** (3/3) | -0.123 |
| B on code, same-domain control | 0.004 | 0.002 | +0.000 (0/3) | -0.022 |
| B on code, reward λ 50 | 0.049 | 0.050 | -0.000 (0/3) | +0.189 |
| A on prose, reward λ 50 | 0.092 | 0.056 | -0.003 (0/3) | +0.503 |

(3/3 = the 95% interval is above zero in all three seeds.)

λ 50 shifts the models as much as the real transfer does, or more, and depends on the partner nearly as much. But it doesn't lean towards the partner at all, and the control partners look just the same. The real code transfer leans hard towards A's view in every seed and the control doesn't. So "toward" is the part that separates using the partner's knowledge from being pushed around by it.

The same pattern runs through every run. Toward on B's code: unforced +0.089, fair force 0.05 +0.057, fair 0.15 +0.037, old-weighting force +0.012 to +0.019, reward λ 5 +0.013, λ 15 and 50 zero. That's the same order as the bits/char gains in the dial write-ups, so the measure agrees with them where they overlap.

## Result 2 - "different but not better" already happens

On B's *strong* domain (prose), the unforced expert bus makes B lean towards A's view in all 3 seeds (+0.003 to +0.006, intervals above zero), while B gets slightly *worse* there (+0.008 bits/char). The control is near zero (one of three seeds). So B on prose is a small example of what I'm after: a model that's different because of its partner, without being better. It's small, but it's real.

## Result 3 - A's expert links stay shut

A on prose, through the expert bus:

| | Shift | Toward |
|---|---|---|
| Real partner (B), seeds 0 / 1 / 2 | 0.003 / 0.003 / 0.002 | +0.0006 / +0.0007 / +0.0003 |
| Control (code-heavy B') | 0.002-0.003 | about 0 |
| Through the residual stream instead, seeds 0 / 1 / 2 | 0.003 / 0.008 / 0.007 | +0.0018 / +0.0082 / +0.0089 |

So it isn't that B pushes A the wrong way. B barely moves A at all. A's paired predictions on prose are nearly the same as A alone, about as close as with a wrong partner or the control. There's a trace of real lean in two seeds, a tenth of the residual stream's.

The link gates say the same. Each link has a learned gate on how much a model reads its partner (tanh, starts at zero). At the end of training (size only, the sign doesn't matter):

| | A reads B (largest of 4 links) | B reads A (largest of 4 links) |
|---|---|---|
| Expert bus, seeds 0 / 1 / 2 | 0.027 / 0.017 / 0.017 | 0.089 / 0.168 / 0.211 |
| Residual stream, seeds 0 / 1 / 2 | 0.021 / **0.095** / **0.098** | 0.389 / 0.216 / 0.251 |
| Control pairs (c0, d0), both kinds | 0.02-0.04 | 0.01-0.03 |

B opens one link wide in every working run. A never opens any expert link past the controls' level. Through the residual stream it opens one about 0.1 in seeds 1 and 2, which are exactly the seeds where A got some prose that way in the complementary run. So the gap is that A's reader never opens on the expert bus.

## Result 4 - B's prose knowledge isn't missing from its experts

The second script (`scripts/toy_where_knowledge.py`, output `results/toy_complement/where_knowledge.json`) looks inside the four model kinds of the complementary run: A (code-heavy, char), A' (prose-heavy, char, from the d units), B (prose-heavy, BPE) and B' (code-heavy, BPE, from the c units). For each it replaces one part at a time with its average output over a balanced set of training text. That's a layer's attention, a layer's whole MoE, the MoE at every layer, or one expert. Then it measures how much worse the model gets on each domain. It also measures how differently each layer routes prose and code.

- Routing - both A and B route prose and code a bit differently, more in the later layers (JS 0.01-0.07). B is no less domain-aware than A here.
- Whole MoE per layer - B depends on its MoE layers on prose about as much as A does on code (layers 1-3: +0.20 to +0.24 bits for B on prose, +0.17 to +0.34 for A on code).
- Advantage over the other-mix twin - A is 0.47 bits/char better at code than A', and B 0.46 better at prose than B'. Taking out the MoE at layers 1-3 in both doesn't shrink either advantage (it grows a little, to 0.49-0.55 for A and 0.51-0.52 for B). Layer 0 is the one difference: taking out the first MoE layer in both B and B' halves B's prose advantage (0.46 to 0.24), while A's first layer leaves A's code advantage where it was (0.47). So if anything, more of B's domain skill sits in an MoE layer than A's does, and the links read layer 0 too.
- Single experts - this is where they differ. Taking out one of A's experts costs it 0.053 bits on code on average, and two experts carry two-thirds of the domain-specific cost. One of B's experts costs it only 0.015 on prose, and the top two carry a quarter. B has 64 small experts (16 per layer) and A has 32 bigger ones, so B's knowledge is spread much thinner per expert.

So the guess "B's prose isn't in its experts" doesn't hold up, at least not with this test. B's experts matter for prose about as much as A's do for code. What differs is how finely it's spread. Taking out everything at once (the MoE at every layer, or the first layer's attention) wrecks every model by 1-5 bits, so I don't read anything into those rows.

## What this does and doesn't show

It shows there's a working measure for "different because of the partner" that doesn't need a gain. It separates real transfer from the reward's damage, and agrees with the bits/char results where both apply. It also finds one case of "different, not better" (B on prose).

It shows the A-side gap is A's expert-bus reader staying shut, not B sending the wrong thing and not B's experts lacking prose knowledge.

It doesn't show why the reader stays shut. What's left:

1. Timing (the tokenizer). With the leak-proof mask, A can only read a B token once its boundaries are final, so B's view of the next word arrives at least one character late, when A needs the next character now. B reading A has no such delay, because A has a state at every character. It's the old guess. The partner-shift numbers don't contradict it, but they don't test it either.
2. Spread. B's knowledge is spread across many small experts. A reader that has to pick entries off the bus may find each one too weak to be worth opening the gate for, while the residual stream gives it the sum in one place. That fits the residual stream partly working for A.
3. Size. These are 4-layer models with d_model 96-128. With experts this small, the fine-grained one (B) may just not have enough in any single expert. That's the "small models" question, and only bigger models can answer it.

Test for 1 and 2 together: pair A with A' (the prose-heavy *char* model from the d units). Same architecture as A, so no tokenizer delay and the same coarse experts. If A gets prose from A' through the expert bus, it's the tokenizer or the expert spread. If not, it's something about A as a reader. It only trains links on models I already have (about 8 minutes of GPU per translator), but it needs a new unit kind in the sweep runner.

And the usual caveats: toy models, 3 seeds, 48 test documents per domain, one data mix.

## Reproduce

```bash
udisksctl mount -b /dev/nvme0n1p2   # results/ are symlinks to the Data drive
for r in toy_complement toy_dial toy_dial2; do
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_partner_shift.py results/$r 5
done
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_where_knowledge.py results/toy_complement 10
```

About 20 minutes for all of it, on 24 cores with everything running at once. Gate values are the last `train` entry in each `fused_<variant>.json`.
