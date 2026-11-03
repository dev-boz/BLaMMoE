# Toy fusion results — two frozen MoEs run in lockstep and talk at every depth (WP-T2)

> **2026-10-22: numbers superseded.** Everything below is from single-seed runs fitted before
> the M6 train/val/test split. The current numbers are in
> [`toy-sweep-results.md`](toy-sweep-results.md) (3 seeds per pair type, CIs, causal
> attribution). Kept as history.

Follow-on to [`toy-poc-results.md`](toy-poc-results.md). That doc showed one
expert's output can be *transmitted* from A into B. This one asks the "slap two
models together" question:

> Can two independently trained MoEs exchange information **at any depth,
> including the first layer, in both directions**, with a translator that
> **chooses which of the partner's experts to read** — and does the fused system
> think differently, or better, than either part?

Same two toy models as before, both frozen. A is char-level (d=128, 8
experts); B is BPE-1024 (d=96, 16 experts). Only the links train (~0.3M
params). Code: `blammoe/toy/fused.py`, `scripts/toy_fused*.py`. Variants live
in `configs/toy_poc.yaml` under `fused.variants`.

> **Audit (2026-10-09) — read before citing anything below.** An external review
> flagged a possible cross-tokenizer leak, and a prefix-honest audit confirmed it.
> B tokenizes the whole text, and where a BPE boundary falls depends on the
> characters after it. So when A reads B's tokens "up to char i+1", their mere
> existence tells A about char i+1, e.g. that a word is about to end. Re-scoring
> A with B re-tokenized on the prefix only (`blammoe/toy/audit.py`,
> `scripts/toy_fused_audit.py`; 16 docs x 255 positions):
>
> | A bits/char | solo | fused, as reported | fused, prefix-honest |
> |---|---|---|---|
> | translator-routed, char bias | 1.489 | 1.458 | **1.852** |
> | + stage-2 plain fine-tune | 1.489 | 1.438 | **1.814** |
> | summed MoE signal, char bias | 1.489 | 1.501 | 1.505 |
>
> **Withdrawn:** every ordinary-text "A better than the parts" claim below
> (1.651, 1.626, the A+A comparison built on it, and the novelty/diversity
> numbers for A, which rest on the same predictions). **Unaffected:**
> split-recall cooperation (97.7% for A both ways: digits are always single
> tokens), the expert-choice result, the depth results, and everything in
> `toy-poc-results.md` (B reading char-level A cannot leak). Fix:
> `safe_mask`, where a partner reads a BPE token only once its pre-token chunk is
> closed. Tests show safe-mask scoring equals prefix-honest scoring exactly.
> Safe-mask retrains, including expert-level vs residual-level fusion, are
> running.

## How the fusion works

A and B advance one block at a time. After each linked block, each model's
residual stream cross-attends over the partner's signal at that depth: Tx → bus
→ tanh-gated Rx (WP6), zero-initialised, so an untrained fusion is exactly the
two models run separately. Causality across tokenizers: a position that has read
`text[:e]` may only read partner entries that have read at most `text[:e]`.

Signals on the bus:

- **moe** — each layer's MoE output: the sender's router-weighted sum of the
  experts that fired.
- **experts (translator-routed)** — every routed expert's own output is a
  separate bus entry, tagged with a learned expert-ID embedding. The
  *receiver's* attention decides which of the partner's experts to read, and
  where. Nothing is hand-picked.

**Task — split-view recall.** Each stored value is masked in exactly one model's
view (`kiwi=????` for A, `kiwi=4821` for B, and vice versa for other keys). Both
models are asked about all keys. On "needs partner" answers each model alone is
at chance (A 9.6%, B 9.2%). On its own answers A scores 98.1% and B 77.3%.
Training mixes this 50/50 with ordinary prose/code, which both models see in
full.

## Result 1 — without a shared sense of position, cross-model fusion fails

1500 steps, all links: accuracy on answers that need the partner.

| variant | A needs B | B needs A | wrong-partner control |
|---|---|---|---|
| A↔B, moe, every layer | 11.1% | 18.0% | 10.5 / 11.4% |
| A↔B, moe, layer 0 only | 11.4% | 10.0% | ~10% |
| A↔B, experts, every layer | 9.9% | 10.5% | ~10% |
| A↔B, experts, layer 0 only | 10.8% | 12.1% | ~10% |
| **B↔B (B + a copy of itself), moe** | **90.6%** | 11.3% | 10.3% |

The same machinery works between two copies of one model and fails between
different models. The difference is position. A counts characters and B counts
BPE tokens, each with its own learned position embedding. So "the partner's state
for the same piece of text" sits at coordinates the translator has to discover
from scratch.

## Result 2 — a character-distance bias fixes it, in both directions

The Rx gets a learned per-head attention bias indexed by bucketed **character
distance** between reader and bus entry: 0, 1, 2, 3, 4, 5–7, 8–15, 16–31,
32–63, 64+. It is zero-initialised. Characters are the one coordinate both
tokenizers share. Runs use 3000 steps.

| variant | A needs B | B needs A | wrong partner | solo |
|---|---|---|---|---|
| A↔B, moe, every layer + char bias | **97.1%** | **99.6%** | 10.6 / 10.1% | 9.6 / 9.2% |
| **A↔B, experts (translator-routed), every layer + char bias** | **97.3%** | **99.4%** | 10.7 / 9.8% | 9.6 / 9.2% |
| A↔B, moe, layer 0 only + char bias | 10.6% | 38.6% | 11.1 / 9.7% | 9.6 / 9.2% |

Two different models, different tokenizers, both frozen. Each now answers
questions it could only answer through the other.

## Where the traffic flows (depth)

The translator-routed run, switching links off at eval time:

| links kept | A needs B | B needs A |
|---|---|---|
| all four | 97.3% | 99.4% |
| only layer 3 | 65.0% | 58.4% |
| only layer 0 / 1 / 2 | ≤ 10% each | ≤ 10% each |
| all except layer 3 | 2.4% | 1.0% |
| all except layer 0 | 75.0% | 81.2% |
| all except layer 1 or 2 | 93.5–96.8% | 98.5–99.5% |

When the translators may use every depth, they put the load-bearing traffic in the
**last** layer, with **layer 0 as a genuine helper** (removing it costs 18–22
points). Layers 1–2 are nearly unused, and their gates stay near 0.02 while
layer 3's open to −0.28 / −0.14. Forced to communicate *only* after layer 0, B
still reaches 38.6% (one direction). Early transfer is possible but weaker.
That is the target for the next iteration.

## Which experts the translators chose

Attention share on each partner expert vs that expert's share of the bus
(× = how much more it is read than its presence alone would give):

| link | reader | on answers that need the partner | on ordinary text |
|---|---|---|---|
| 3 | A reads B | **B expert 0: 60% (×8.1)** | null slot 59%, then B e15 (×3.0) |
| 3 | B reads A | **A expert 7: 60% (×4.3)** | A e6 43% (×2.3) |
| 0 | B reads A | A e0 51% (×9.1) | A e0 41% (×2.7) |
| 1 | B reads A | A e0 75% (×12.6) | A e0 47% (×3.2) |

**The translators independently found each other's answer specialists.** B's
expert 0 and A's expert 7 are the experts the first POC identified by hand, from
routing statistics, as firing on 100% of answer positions. Nobody told the
translator. Selection is also context-dependent. On ordinary text A's layer-3
translator mostly chooses *not* to read B (59% on the null slot), and reads
different experts when it does.

## Diversity and "better than the parts" on ordinary text (WITHDRAWN: tokenizer leak, see Audit)

Measured on 48 held-out prose/code windows at the 6467 positions where both
tokenizers have a boundary. Next-*character* predictions are compared across
models by reading each token distribution's first character.

Baseline diversity already exists. Alone, A and B each get ~53% of next
characters right, but "either is right" reaches 60.7%.

| | bits/char A | bits/char B | A next-char acc |
|---|---|---|---|
| solo | 1.677 | 1.835 | 53.2% |
| moe + char bias, fused | 1.684 | 1.838 | 52.9% |
| **experts + char bias, fused** | **1.651** | 1.851 | **54.2%** |
| experts + char bias, *wrong-document partner* | 1.768 | 1.857 | 51.2% |

The translator-routed fusion makes **A better than A alone and better than B
alone** on ordinary text. The gain depends on the partner's content: the same
bridges fed the wrong document's partner make A clearly worse. A's predictions
also move *toward* B's (JS divergence 0.087 vs 0.095 solo-to-B), and the wrong
partner pushes them away (0.108). Fused A is right on 3.5% of the positions
where *both* models alone were wrong, vs 2.7% with the wrong partner. B does not
benefit in return.

For the moe-signal fusions, "novel" predictions (fused top-1 differs from both
solo top-1s, 4–7% of positions) occur at the *same rate* with a wrong partner.
That is bridge noise, not new thinking. The translator-routed fusion is the only
variant where the change is partner-specific.

**Capacity control: A fused with a copy of itself** (same translator-routed
links, character bias, 3000 steps). It solves split-view recall (96.5% / 98.6%:
trivial when both sides are the same model) but does **not** help on ordinary
text. A's bits/char go from 1.677 to 1.702, and to 1.701 with a wrong-document
partner, so there is no partner-specific effect. A second copy of A adds compute
and bridge parameters but no new information. The A+B gain (1.651; wrong
partner 1.768) appears only when the partner is a *different* model. That is
the closest this toy gets to the hoped-for "new dynamic": B's different view of
the same text makes A predict better than either model alone.

| A's bits/char on ordinary text | fused | wrong partner | solo |
|---|---|---|---|
| **A + B** (translator-routed, char bias) | **1.651** | 1.768 | 1.677 |
| A + copy of A (same setup) | 1.702 | 1.701 | 1.677 |

## Caveats

- Toy scale: 2–3.5M-param models, one seed per variant, no error bars.
- The bridges are trained on half split-recall data, a task that *requires*
  cooperation. The ordinary-text effect is a side effect of that training, not
  something optimised directly.
- "Diversity" here is measured on teacher-forced next-char predictions, not on
  free generation (lockstep generation across two tokenizers isn't built yet).
- The traffic is late-heavy. Early-only transfer works (38.6%, one way) but is weaker.

## Stage 2 — fine-tuning the translator on ordinary text (WP-T3) (A's bits/char WITHDRAWN: see Audit)

Starting from the translator-routed fusion above (frozen models, links only), 1000
more steps on ordinary prose/code, with no split-recall forcing. A **win** for model X
at a token = fused X right, X alone wrong, *and* the same fusion with a
wrong-document partner wrong. So only wins that depend on what the partner said
count. Code: `blammoe/toy/reward.py`, `scripts/toy_fused_reward.py`.

| ordinary text | A bits/char | B bits/char | A novel | A right where both alone wrong (wrong partner) | split recall A / B |
|---|---|---|---|---|---|
| solo | 1.677 | 1.835 | n/a | n/a | 9.6 / 9.2% |
| stage 1 fusion | 1.651 | 1.851 | 7.5% | 3.5% (2.7%) | 97.3 / 99.4% |
| plain fine-tune | **1.626** | **1.832** | 5.5% | 2.7% (2.8%) | 52.4 / 35.8% |
| reward, wins 3x | 1.630 | 1.832 | 6.3% | 3.1% (3.2%) | 54.8 / 37.8% |
| reward, wins ~100x | 2.218 | 2.173 | 37.3% | 8.3% (6.3%) | 89.7 / 97.9% |

- **Plain fine-tuning makes both models better fused than alone.** That is the first
  time B gains too. A's dependence on its partner grows: wrong partner 1.814.
  Partner-specific novelty fades, though. The fusion becomes accurate but conservative.
- **Reward at 3x does nothing measurable.** It tracks the plain arm, and its training
  win rate never rises.
- **Reward at ~100x** (a weighting bug: the win term was averaged over the ~2% win tokens
  only) is a strong diversity dial. A changes 45% of its guesses, and emergent hits
  rise to 8.3% vs 6.3% with a wrong partner, so it is still partly partner-specific.
  But bits/char rise to ~2.2. It also kept split-recall cooperation (90 / 98%), which
  the gentler arms lost.
- Reading: the reward weight trades accuracy for diversity. The useful region is
  presumably in between. A sweep of the weight (e.g. 10-30x), and rewards on
  *generated* text, are the next stage-2 experiments, best run on a GPU.

## Leak-proof results (safe_mask) — what survives

Retrained from scratch with `safe_mask`: a partner may read a BPE token only once its
pre-token chunk is closed, so token boundaries cannot reveal unseen chars. Both variants
pass the audit exactly: full-text scoring equals prefix-honest scoring to 4 decimals
(A bits/char 1.4940 = 1.4940; recall 95.1% = 95.1%). Char bias, every layer, 3000 steps,
one seed.

| safe mask | A needs B | B needs A | wrong partner | A / B bits/char (solo 1.677 / 1.835) |
|---|---|---|---|---|
| **translator-routed experts** | **95.1%** | **99.8%** | 11.9 / 9.2% | 1.683 / 1.858 |
| whole residual stream (model-level) | 10.3% | 99.4% | 9.9 / 10.2% | 1.679 / 1.847 |

- **Two-way expert-level cooperation survives the leak fix.**
- **No ordinary-text gain** for either variant: the earlier gain was the leak.
- **Expert-level beats model-level here.** With the same budget, reading B's individual
  experts carries B's information into A (95.1%), while reading B's whole residual stream
  does not (10.3%). The reverse direction works for both. One seed: a longer or reseeded
  residual run could still learn.

**The experts carry the signal (ablation).** Hiding a single partner expert's bus entries
at every link, at eval time:

| hidden | A needs B | B needs A |
|---|---|---|
| nothing | 95.1% | 99.8% |
| **B expert 0** (A's translator's choice: 50% of its last-link attention, x6.2) | **14.0%** | n/a |
| any other one of B's 15 experts | 93.4-95.4% | n/a |
| **A expert 7** (B's translator's choice: 91%, x5.5) | n/a | **29.1%** |
| any other one of A's 7 experts | n/a | 99.4-99.8% |

The transfer runs almost entirely through the one expert each translator picked, and
these are the same answer specialists the first POC found by hand from routing statistics.
They are not the busy experts: at the last layer on ordinary text, B e0 is 0.3% of B's
bus entries (14th of 16) and A e7 is 0.4% (8th of 8). Code: `scripts/toy_fused_hide.py`.

**Still open (from the external review):** confidence intervals and 3+ seeds; a
same-architecture, different-seed partner (Rung 2); a plain-ensemble baseline; a
data-difference dose-response; an independent adversarial audit of the eval path before
the OLMoE/Qwen run. The channel is narrow (a recall answer is ~3 bits) and mostly carried
at the last layer.
