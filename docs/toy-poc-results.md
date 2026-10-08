# Toy POC results — does one MoE expert's output survive transplant into another MoE?

**Short answer: yes, at toy scale.** The *native* output of a single routed
expert in model A, passed through a linear map fitted on a few thousand clean
paired positions, lets model B answer questions it has no way to answer on its
own (8% → 96.5% on answer digits). B differs from A in tokenizer, width,
expert count and seed. Every control stays at chance.

This is a proof of *possibility*, not of usefulness: tiny models (2–3.5M
params), a synthetic task built so the effect is measurable, and an injection
point at B's last layer. Caveats are at the end. Raw numbers live in
`results/toy_poc/{poc,followup}_results.json` (git-ignored; regenerate with the
commands in `docs/workplan.md` §WP-T, ~2 h on 4 CPU cores).

## Setup

| | Model A ("informed") | Model B ("receiver") |
|---|---|---|
| Tokenizer | char-level (97) | BPE (1024), digits forced to single tokens |
| Width / layers | d=128, 4 layers | d=96, 4 layers |
| Experts | 8 × SwiGLU(256), top-2 | 16 × SwiGLU(96), top-2 |
| Seed | 0 | 1 |
| Params | 3.47M | 2.15M |

Both were trained from scratch for 4000 steps on the same mix: CPython stdlib
source (code), `papers/` (prose), and synthetic recall docs
(`store: kiwi=4821, lynx=0937; ask: lynx=0937`). Both learned recall through the
usual sudden jump, A between steps 2k and 2.5k and B between 2.5k and 3k: final answer-digit
accuracy 99.6% (A) and 99.1% (B).

**The blindfold test** (a controlled form of spec Test B): B reads the doc with
every stored value masked (`kiwi=????`), so the answer digits are
unknowable to it (8.0%, chance ≈ 10%). A reads the clean doc. Positions are
aligned causally across tokenizers: B token `[s, e)` ↔ A char `e-1`, so both have
read exactly `text[:e]`.

**Expert selection** is operation-level (spec 1.3): the expert most often in A's
top-2 at answer-predicting positions. At A's last layer that is expert 7 (in
top-2 at **100%** of answer positions); expert 6 never is ("unrelated").

**The transplant**: inside B's MoE at layer 3, replace B expert 0's output with
`W · h_A`. B expert 0 is B's own answer specialist, routed at 100% of answer
positions. `W` is a ridge least-squares map (WP4), fitted on clean paired
positions from ordinary training-mix text. It never sees a blindfolded example,
and it is not trained with B's loss.

## Headline: zero-shot linear transplant, A layer 3 expert 7 → B layer 3 expert 0

B answer-digit accuracy, blindfolded, 2280 answer tokens:

| condition | accuracy |
|---|---|
| B, no transplant (floor) | 8.0% |
| B reading the clean text (ceiling) | 99.1% |
| **transplant: A's expert, routed-only (native)** | **96.5%** |
| transplant: A's expert evaluated densely on every token | 96.8% |
| control: A's signal from a *different* document | 4.8–5.9% |
| control: A blindfolded too (no information exists) | 8.1–9.0% |
| control: right document, wrong position (4 tokens back; Test C) | 2.8% |

Injection strength matters but the effect is robust: α = 0.5 / 1 / 2 / 4 →
76.5 / 96.8 / 88.5 / 77.9% (dense).

## What the ablations say

**1. Depth: it only works where the answer exists.** Transplanting from A's
layers 0–1 gives chance for every expert *and* for A's whole residual stream.
A has not yet retrieved the value at those depths. At A layer 2 it is partial
(35%), and at layer 3 it is near-complete. On the B side, injecting at B layers
0–2 transmits nothing, even with A's best signal. Only B's last layer works,
where the injected expert output feeds the unembedding almost directly.

**2. The expert, not just the residual stream, carries it — when used natively.**

| A layer | signal | matched | other-doc | A blind |
|---|---|---|---|---|
| 2 | specialist (exp 0), routed-only | **35.1%** | 5.5% | 6.8% |
| 2 | unrelated (exp 4), routed-only | 5.5% | 5.0% | 5.2% |
| 2 | whole residual stream | 35.1% | 6.1% | 9.2% |
| 3 | specialist (exp 7), routed-only | **96.5%** | 4.8% | 9.0% |
| 3 | unrelated (exp 6), routed-only | 5.3% | 5.1% | 5.3% |
| 3 | whole residual stream | 96.3% | 5.6% | 6.5% |

Used natively (zero wherever A's router did not pick it), the routed specialist
transmits exactly as much as A's *entire* residual stream at that depth, and
the unrelated expert transmits nothing. **Caveat that matters for the real
experiment:** evaluate the unrelated expert *densely* on every token and it
transmits too (36% at layer 2, 69% at layer 3). Dense evaluation pushes the
whole residual stream through that expert's weights. Any claim of
*expert-level* transmission must therefore use routed-only outputs, or it
collapses into whole-model alignment.

**3. The alignment is content-specific.** Fitting `W` on prose + code only, with
no recall-format text, gives 10.9% (controls 2.8–3.6%): measurably above
the controls, but nowhere near usable. The working map needed clean pairs from
the same kind of content it later transmits, though no blindfolded or
task-labelled ones. For the real models this means the calibration corpus has
to cover what you want transmitted.

**4. It is cheap.** Pairs used to fit `W` → accuracy: 250 → 61.8%, 1k → 94.2%,
4k → 96.4%, 16k → 96.7%.

**5. Interference is real but not catastrophic.** The transplant replaces one
expert's output at *every* position with no gate. It raises B's ordinary-text
loss from 2.40 to 2.67 nats/token (16k-pair map), and to 3.02 with the noisy
250-pair map.

**6. Geometry did not predict usability.** The WP5 battery between A's specialist
and B's chosen expert was modest everywhere: CKA 0.11–0.36, mutual k-NN
0.13–0.18 (chance 0.003), held-out Procrustes residual 0.84–0.93. It was
clearly above a randomly-initialised B (CKA ≤ 0.06, residual ≥ 0.98). Layer 3
scored best on all three, but the gap between "works" (96%) and "doesn't" (≈9%)
is far larger than the metric gaps. This is the spec's "a geometric fit is
necessary but not sufficient", seen directly.

## Learned bridge (Tx → bus → tanh-gated cross-attention Rx)

_Pending: 3000-step runs at B layers 1 and 3 are in progress; 600-step runs are summarised in `poc_results.json`._

## Caveats (read before citing any of this)

- **Toy scale.** 2–3.5M-param models, a synthetic task built for measurability,
  4 layers each. The real question is OLMoE ↔ Qwen-MoE.
- **Late-layer only.** The transplant that works writes into B's last MoE layer,
  one step from the logits. It is closer to "A's prediction, re-expressed in
  B's coordinates" than to deep representational sharing. Mid-layer transplants
  failed.
- **Single pair, single task, single seed per model.** No error bars.
- **"Cross-family" is simulated.** Different tokenizer, width, expert granularity
  and seed. Same architecture code, same training data mix.
- **Not yet done:** Test D (free-generation distribution shift), Test A
  (Patchscopes-style readout), a gated transplant to remove interference.
