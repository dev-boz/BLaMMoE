# Complementary models results

## What I wanted to know

The end goal is two MoE models that work just fine on their own, plus a translator between them that allows, and encourages, free transfer of information between their experts.

The [toy sweep](toy-sweep-results.md) showed the translator can move information between two frozen MoEs, through their experts. But only when a task forced it. In split-view recall each model can only get half the answers by itself, so the translator has to pass information or fail. And both models had been trained on the same text, so on ordinary text they had little to teach each other. The translator didn't help there at all. Pairing made both models very slightly worse.

So this experiment asks the "encourages" half of the question. If each model knows something the other doesn't, and nothing forces transfer, does the translator learn to pass knowledge across anyway, just because it helps?

## The setup

Same two tiny MoEs as the sweep (2-3.5M parameters each, frozen), with one change: what they're trained on.

- Model A - reads characters, 8 experts, trained on **mostly code** (90% code, 10% prose)
- Model B - reads BPE tokens, 16 smaller experts, trained on **mostly prose** (90% prose, 10% code)

Half of each model's pretraining is still the recall documents, same as the sweep, so the models stay comparable to it. B's BPE tokenizer is fitted on B's own prose-heavy mix, so the two models differ in vocabulary as well as in knowledge. That's closer to two real off-the-shelf models.

The translator is the same set of small links at every layer (about 300k parameters), in the same two kinds:

- Expert bus - reads the partner's routed experts individually
- Residual stream - reads the partner's whole running internal state, no expert breakdown

The big difference from the sweep is what the translator trains on: **ordinary text only**, half prose and half code, both models reading the same window. No recall task, nothing that needs the partner. The only reason to pass anything across is that it lowers the loss.

Then I score each model on prose-only and code-only test text the models never trained on. The number is bits per character. Lower is better, and a gain of 0.1 bits is big at this scale. Each model is scored three ways:

- Fused - with its real partner
- Alone - links switched off (exactly the original model, it's frozen)
- Shuffled partner - the partner is reading a different document

Every trained translator went through the same strict leak check as the sweep (every position recomputed in full precision, proving neither model sees text it shouldn't). All 18 passed.

## The control I needed

The first thing I realised when designing this: a gain over "alone" doesn't prove knowledge transfer. Bolting a second model on gives the first one more compute, and some of the gain could just be that. The sweep already showed links on the same-data pair give tiny gains like this.

So each seed gets two control pairs, where the partner is the same size and design but knows the *same* domain:

| Unit | Pair | What it tests |
|---|---|---|
| s0, s1, s2 | code-heavy A with prose-heavy B | the real question |
| c0, c1, c2 | the same A with a B trained on mostly code | how much of A's prose gain is just extra capacity |
| d0, d1, d2 | the same B with an A trained on mostly prose | how much of B's code gain is just extra capacity |

Each control keeps one of the real pair's models exactly (copied over) and retrains only the other one, with the same seed and design but the other data mix. If B gets better at code with a partner that knows code than with a partner that doesn't, that difference is knowledge crossing over.

Seed 0 ran first on one CMP 170HX (1.5 hours). Seeds 1 and 2 then ran side by side, one per card (1.5 hours each).

## Result 1 - B learns code from A, every time

B is the prose model, so code is its weak domain. Change in B's bits per character on code when paired, versus alone (negative means it got better):

| Seed | B alone | Expert bus: real partner | Expert bus: control partner | Residual: real partner | Residual: control partner |
|---|---|---|---|---|---|
| 0 | 2.132 | **-0.132** | -0.024 | **-0.256** | -0.032 |
| 1 | 2.150 | **-0.109** | -0.025 | **-0.138** | -0.082 |
| 2 | 2.107 | **-0.126** | -0.016 | **-0.165** | -0.053 |

The real partner beats the control in all 6 runs, and the 95% confidence intervals (bootstrapped over 48 test documents) don't overlap in any of them. On the expert bus the real partner is worth 4-7 times what the control partner is. The best run (seed 0, residual) closes almost half the gap between B's code (2.13) and A's own code (1.58).

This is the result I was after. Nothing in training forced it. The translator only ever saw ordinary text, and it learned on its own to pull code knowledge from A into B, because that helped.

B also leans on A now. Give it a partner reading the wrong document and it gets much *worse* than alone (2.35 instead of 2.13 on the expert bus, worse still on the residual stream). The translator trusts what A sends. Switch the links off and B is exactly its old self, because it's frozen, so "works fine on its own" still holds.

## Result 2 - A barely learns prose from B, and not through the experts

A is the code model, so prose is its weak domain, and B is a lot better at prose (1.89 versus 2.21 bits). Same table for A on prose:

| Seed | A alone | Expert bus: real partner | Expert bus: control partner | Residual: real partner | Residual: control partner |
|---|---|---|---|---|---|
| 0 | 2.208 | -0.015 | -0.020 | -0.015 | -0.021 |
| 1 | 2.220 | -0.014 | -0.018 | **-0.038** | -0.013 |
| 2 | 2.199 | -0.012 | -0.018 | **-0.038** | -0.014 |

On the expert bus the real partner never beats the control. A gets a small gain from having a partner at all, and a code-heavy partner gives just as much. None of B's prose knowledge comes across.

On the residual stream it does in 2 of the 3 seeds, by about 0.024 bits beyond the control (intervals don't overlap). That's real but small, about a tenth of what B gets from A.

So transfer happens in both directions only through the residual stream, and through the expert bus only one way.

## Chasing the asymmetry

My first idea was the leak-proofing itself. To stop B's tokens giving away characters A hasn't read yet, A may only read a BPE token once its boundaries are settled, and for a word that means after the word has finished. Going the other way, every one of A's characters is readable straight away. In the sweep this didn't matter, because the recall answers are digits and every digit is its own token. But prose is words. So maybe A just never gets B's help in time.

To test it I split A's prose gain by position: mid-word, where B's token for the current word can't be read yet, and everywhere else. With the seed 0 models:

| Pair | Translator | Mid-word | Elsewhere |
|---|---|---|---|
| real (B knows prose) | expert bus | -0.013 | -0.021 |
| real | residual | -0.013 | -0.019 |
| control (B knows code) | expert bus | -0.017 | -0.026 |
| control | residual | -0.016 | -0.034 |

The gain is a bit bigger outside words, which fits the idea. But the control partner helps at least as much everywhere, so there's no sign of prose knowledge arriving at either kind of position.

Then I realised the test can't separate the two explanations anyway. B's tokenizer keeps spaces as their own tokens, and B makes its guess about the next word *at the space token*. A can only read that token one character into the next word. So even at a word start, A is reading B's state from a word or so back. The split shows "stale everywhere" against "stale everywhere".

And the residual-stream result argues against the leak-proofing being the whole story. The residual links have exactly the same reading rules and still carry some prose to A in 2 seeds. So right now my best guess is that it's specific to the expert bus. B's prose knowledge may live more in its attention and residual stream than in its routed experts, which would leave the expert bus nothing worth reading. That's a guess, not a result.

## Result 3 - transfer it was never trained for

The recall test still runs on every translator, even though these ones never trained on it. Accuracy on the answers B can only get from A:

| Seed | Expert bus | Residual | Shuffled partner |
|---|---|---|---|
| 0 | 13.2% | 29.7% | 9.5-11.0% |
| 1 | 34.3% | 10.2% | 9.0-9.1% |
| 2 | 14.7% | 24.3% | 8.8-9.4% |

Chance is about 10%. 5 of the 6 are above it, up to 34%. A learned recall in pretraining, and B, having learned to lean on A for code, partly carries that habit over to a task the translator never saw. It's nowhere near the 97-99% of a translator trained on recall, and it jumps around a lot between seeds. But it's the first sign of the translator generalising past what it was trained on. In the other direction, A on B's answers, it stays at chance everywhere, matching Result 2.

## What it costs

Pairing isn't free on the strong domains:

- A on code (its strong domain) gets slightly worse with every translator, by 0.006-0.008 bits, in every seed.
- B on prose gets slightly worse with the expert bus (0.004-0.012 bits). With the residual stream it's within noise.

Small next to B's code gain. But it's the same small cost the sweep saw on ordinary text, and it's worth getting rid of eventually.

On mixed text overall, B goes from 1.82-1.85 alone to 1.75-1.78 on the expert bus and 1.70-1.76 on the residual stream. For reference, the sweep's B, trained on a 50/50 mix, scored 1.67-1.69 alone. So pairing gets the prose-specialist B a good part of the way to a generalist B, without retraining it. Those are different models, though, so that comparison is only a rough guide.

## What didn't work, and caveats

- The expert bus doesn't carry prose to A at all, and it's the weaker channel overall. B's code gain is about two thirds of what the residual stream gets, and the expert bus is also the one that costs B a little on prose. Since the end goal is transfer *between experts*, this is the main gap now.
- The residual stream's results jump around more between seeds (B's code gain -0.14 to -0.26, A's prose gain in 2 of 3 seeds).
- I didn't run the attribution tests (which expert carries what) on these translators. They're built for the recall task, and these translators weren't trained on it.
- Toy scale, 3 seeds, 48 test documents per domain. One data mix strength (90/10).

## What this does and doesn't show

It shows a translator trained only on ordinary text will pass knowledge from a model that has it to a model that doesn't, without being told to. B's gain is 4-7 times what a same-size partner without that knowledge gives, in every seed. That's the first real evidence for the "encourages" half of the end goal, and it gives me confidence the idea isn't only a forced-task trick.

It doesn't show it working both ways through the experts. A gets almost nothing, and nothing through the expert bus. It doesn't show which experts carry it. And it's still two toy models trained from scratch, not off-the-shelf ones.

## Where to next

1. **A dial between "allows" and "forces".** This run is one end: transfer allowed, never forced. The sweep was the other end: half the training was a task that forces it. Neither is obviously the right balance. The forced translators transfer reliably but don't help on ordinary text. The free ones help where it pays, but only one way through the experts. I want a dial I can turn between them and find the sweet spot. Two dials already exist in the code: how much forced recall to mix into training, and the stage-2 reward that gives extra weight to predictions that only came out right because of the partner.
2. **Why the expert bus can't carry prose to A.** Look at where B's prose knowledge actually lives (which experts route on prose, how much of B's prose advantage comes from experts versus attention), and pair A with a prose-heavy *character* model to rule the tokenizer in or out.
3. **Free generation** and **off-the-shelf models**, as before.

## Reproduce

```bash
udisksctl mount -b /dev/nvme0n1p2   # results/toy_complement is a symlink to the Data drive
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_complement.yaml 4 s0 c0 d0
```

Use `CUDA_VISIBLE_DEVICES=1` with `s1 c1 d1` to run seed 1 on the other card at the same time. Always set it: without it torch picks a CMP on its own, fastest first. Results land in `results/toy_complement/<unit>/`, with everything above in `results/toy_complement/summary.json` (`domain` sections: per domain and model, bits per character with intervals, `fused_minus_solo` and `fused_minus_shuffled`). The position split in "Chasing the asymmetry" was a one-off analysis on seed 0 (s0 and c0), not part of the runner.
