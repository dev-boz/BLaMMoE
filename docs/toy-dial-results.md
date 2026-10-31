# Allow/force dial results (dial 1: forced share)

## What I wanted to know

The [complementary models](toy-complement-results.md) run sits at one end of a dial. Its translator only ever trained on ordinary text, so transfer was allowed but never forced, and B (the prose model) learned code from A on its own. The [toy sweep](toy-sweep-results.md) sits near the other end. Half its training was split-view recall, which can't be solved without the partner, so transfer was forced, and it was reliable (97-99%) but did nothing for ordinary text.

Neither end is obviously the right balance. So I wanted a dial I could turn between them and look for the sweet spot: enough forcing to open the channel both ways, without losing the free transfer or adding cost.

## The setup

The dial here is the forced share:

- Forced share (p_split) - the fraction of the translator's training windows that are split-view recall. The rest is ordinary text, half prose and half code. 0 is the complementary run, 0.5 is the sweep's mix.

I tried 0.05, 0.15 and 0.5, next to the 0 I already had. Everything else stays fixed. The two models are copied over from the complementary run, not retrained, so the models, tokenisers, seeds and same-domain control pairs (c and d units) are identical at every setting. Only new translators train. Expert bus only, since transfer between experts is the end goal and that's the channel that was weak.

At each setting I read four things:

- B's gain on code, and A's gain on prose, versus the same-domain control partner (free transfer)
- Split recall accuracy (forced transfer)
- The cost on each model's strong domain

27 new translators (3 settings x 9 units), all 27 passed the strict leak check. Seed 0 and 2's units ran on one CMP 170HX (3 hours), seed 1's and the rest of the controls on the other (3.8 hours), side by side.

## Result 1 - a little forcing switches the free transfer off

B on code, bits per character with the partner minus alone (negative means it got better). Control partner in brackets:

| Forced share | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 0 (complementary run) | **-0.132** (-0.024) | **-0.109** (-0.025) | **-0.126** (-0.016) |
| 0.05 | +0.004 (+0.006) | -0.029 (-0.002) | +0.001 (+0.002) |
| 0.15 | +0.010 (+0.003) | -0.027 (+0.006) | +0.009 (+0.031) |
| 0.5 | -0.010 (+0.006) | -0.024 (-0.008) | +0.002 (-0.005) |

This wasn't what I expected. Turning the dial from 0 to 0.05, so 1 window in 20, takes away almost all of B's code gain. In seeds 0 and 2 it's gone completely. Seed 1 keeps about a quarter of it, at every setting. Turning the dial further doesn't bring it back.

A on prose was already the weak direction (it never beat its control on the expert bus). With any forcing it goes from a small gain (-0.012 to -0.015) to a cost, up to +0.048.

The clearest way to see it is to ask whether B still leans on A at all. This is B's code score with its real partner minus with a partner reading the wrong document (negative means what A says matters):

| Forced share | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 0 | -0.347 | -0.392 | -0.338 |
| 0.05 | +0.001 | -0.136 | +0.001 |
| 0.15 | -0.006 | -0.117 | -0.002 |
| 0.5 | -0.062 | -0.129 | -0.006 |

At 0.05, in seeds 0 and 2, B doesn't care what A is reading. The links into B also end up about half as open (average gate 0.021, against 0.046-0.062 with no forcing).

## Result 2 - the forced transfer is a switch, not a dial

Split recall accuracy on the answers only the partner has (chance is about 10%):

| Forced share | B needs A, seeds 0 / 1 / 2 | A needs B, seeds 0 / 1 / 2 |
|---|---|---|
| 0 (never trained on it) | 13% / 34% / 15% | 9% / 10% / 9% |
| 0.05 | 10% / **97%** / 10% | 10% / 12% / 9% |
| 0.15 | **95%** / **98%** / **97%** | 10% / 11% / 11% |
| 0.5 | **97%** / **100%** / **99%** | 58% / 17% / **97%** |

Each direction is either on (95-100%) or off (chance), with very little in between. Counting all 9 units at each setting, the control pairs included:

| Forced share | B needs A switched on | A needs B switched on |
|---|---|---|
| 0.05 | 1 of 9 | 0 of 9 |
| 0.15 | 7 of 9 | 0 of 9 (one part way, 45%) |
| 0.5 | 8 of 9 | 6 of 9 (one part way, 58%) |

So turning the dial up doesn't make transfer a bit stronger. It raises the chance that it switches on. And seed 0 and 2 at 0.05 are the worst of both: the forced transfer never switched on, and the free transfer still went away.

## Chasing why: the answer weighting

A share of 0.05 behaving almost like 0.5 made me suspect the loss rather than the data. It was.

The translator's loss is ordinary next-character error over the whole batch, plus an extra term for the recall answers:

- Answer term - the average error on just the recall answer tokens in the batch, added at full weight (answer_weight 1).

That average is over however many answers happen to be in the batch. It isn't scaled by how many there are. So when answers are rare, each one carries a huge weight. Measured on the real training batches (16 windows each):

| Forced share | Batches with any recall answers | Weight of one answer token vs one ordinary token |
|---|---|---|
| 0.05 | 51% | about 146x |
| 0.15 | 93% | about 78x |
| 0.5 | 100% | about 21x |

So the dial runs backwards inside. Turning the share down makes each answer weigh more, and at 0.05 half the batches still carry a full-strength recall term. The forcing never really gets small.

I'd already been bitten by this once. The first version of the stage-2 reward averaged its extra term over the reward tokens alone, which gave each about 100 times the weight of an ordinary token and wrecked accuracy (see the HANDOFF lessons). The recall answer term has the same shape. I just hadn't looked at it, because at 0.5 (all the earlier runs) it's only 21x and every batch has answers.

My best guess at how this kills the free transfer, and it's only a guess: until the translator has worked out recall, anything read from the partner just adds noise to the answer digits, and those weigh 78-146 times as much as anything else. The cheapest fix is to close B's links. With them mostly shut, the code transfer never gets started. That fits seeds 0 and 2 at 0.05 (recall never switched on, links half as open, no reliance on A). It doesn't fully explain 0.15, where recall did switch on in all three seeds but only seed 1 kept leaning on A for code. There the recall task seems to take the links over instead of sharing them.

## What it costs

Pairing gets more expensive on the strong domains as soon as any forcing comes in:

| Forced share | A on code (strong) | B on prose (strong) |
|---|---|---|
| 0 | +0.006 to +0.008 | +0.004 to +0.012 |
| 0.05 | +0.007 to +0.018 | +0.007 to +0.021 |
| 0.15 | +0.013 to +0.016 | +0.011 to +0.029 |
| 0.5 | +0.007 to +0.045 | +0.009 to +0.021 |

Roughly two to three times the cost of the unforced translator, with none of its gain.

## What this does and doesn't show

It shows the forced share is not a usable dial as the loss stands. Even a little forcing, the way it's weighted, switches off the free transfer the unforced translator found, in every seed, and makes pairing cost more. Forced transfer is all-or-nothing at every share I tried.

It doesn't show that forcing as such kills free transfer. The forcing here always came in at 21-146 times the weight of ordinary text, and the dial never actually got gentle. A fair test needs the answer tokens weighted per token inside the overall average, the way the reward already is, so that a 5% share really means a small push.

Seed 1 is worth keeping an eye on. It's the only seed where recall switched on at 0.05, and the only one that kept leaning on A at every setting. Three seeds can't say whether that's luck or something about that pair.

## Where to next

1. **The reward dial** (dial 2, built and tested, not run). Ordinary text only, nothing forced, and predictions the pair gets right only because of the partner count 1 + λ times in the average, with λ 5, 15 and 50. Its weighting is already per token in the overall average, so it should behave like a real dial. About 1.2 times this run's GPU time.
2. **Fix the force dial the same way**, then rerun the low shares: answer tokens count 1 + w in the overall average, as a new option so the old runs stay reproducible.
3. The open questions from the complementary run: where B's prose knowledge lives, and a character-model control for A.

## Reproduce

```bash
udisksctl mount -b /dev/disk/by-label/Data   # results/ are symlinks to the Data drive
# needs results/toy_complement (the models are copied from there)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_dial.yaml 4 s0 c0 d0 s2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_dial.yaml 4 s1 c1 d1 c2 d2
# both write summary.json, so rebuild it once both are done (CPU, runs nothing)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/toy_sweep.py configs/toy_dial.yaml 4 none
```

Everything above is in `results/toy_dial/summary.json` (`domain` and `split` sections per unit and variant), with the p_split 0 numbers in `results/toy_complement/summary.json` (variant `ab_experts_all_rel_safe_text`). The answer-weight table and the gate averages were one-off checks on the training batches and `fused_<variant>.json` histories, not part of the runner.
