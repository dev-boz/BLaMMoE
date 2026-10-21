# Toy sweep results

## What I wanted to know

The end goal is two MoE models that work just fine on their own (ideally off-the-shelf ones), and a translator that sits between them and allows, and encourages, free transfer of information between their experts.

Step one is much smaller. Can information move between two independently trained MoEs, through their experts, at all? If mixing them makes the result worse, that still counts, as long as the transfer is real and it goes through the experts.

An earlier single run said yes. But it was one seed, and an audit found problems with how parts of it were measured: a tokenizer leak (already withdrawn), the test set being used to pick things, and a held-out set that wasn't really held out. So this sweep retrains everything from scratch with those fixes in, over several seeds, before I trust any of it. The older docs ([toy-fusion-results.md](toy-fusion-results.md), [toy-poc-results.md](toy-poc-results.md)) are kept as history, but where their numbers disagree with this doc, this doc wins.

## The setup

Two tiny MoE language models (2-3.5M parameters each), trained from scratch on the same text (the Python standard library plus the papers in `papers/`), but different in nearly every other way:

- Model A - reads characters, wider, 8 experts
- Model B - reads BPE tokens, narrower, 16 smaller experts

Both are frozen. The only thing that trains is the translator, a set of small gated links between the two models at every layer (about 300k parameters). Each model puts its experts' outputs on a shared bus, and the other model's links choose what to read from it. The gates start at zero, so an untrained translator changes nothing.

The test is split-view recall. A document lists key=value pairs like `kiwi=4821`. Each value is hidden from one of the two models (A sees `kiwi=????` while B sees `kiwi=4821`, and other keys go the other way round). Then both models are asked for every value. On the ones it can't see, a model can only guess, which gets about 10% (one digit in ten). Anything above that has to come from its partner.

Leak proofing - different tokenizers means B's tokens can hint at characters A hasn't read yet. That already produced one result I had to withdraw. Every trained translator now has to pass a strict check that recomputes every position in full precision and proves neither model ever sees text it shouldn't. A run that fails doesn't count. Every run in this doc passed.

## What the sweep ran

| Unit | Pair | What it tests |
|---|---|---|
| s0, s1, s2 | A and B, three different seed pairs | the real question: two different models |
| r0, r1, r2 | B and another B trained with a different seed | the easiest possible partner, same design |

Each unit trains two kinds of translator. One reads the partner's experts individually (the expert bus). The other reads the partner's whole residual stream, the model's running internal state, with no expert breakdown. Then come the leak checks, the attribution tests and a plain ensemble baseline.

The whole sweep took 3.4 hours on one CMP 170HX. On CPU it would have been around 15.

## Result 1 - information crosses, almost every time

Expert-bus translator, digit accuracy on the answers each model can only get from its partner:

| Unit | A gets from B | B gets from A | Whole 4-digit answers right | Wrong document as partner | Alone |
|---|---|---|---|---|---|
| s0 | 99.4% | 99.3% | 97.4 / 97.4% | 10.3 / 9.5% | 10.2 / 9.6% |
| s1 | 99.5% | 98.3% | 97.8 / 94.2% | 10.2 / 8.6% | 10.3 / 9.1% |
| s2 | 97.5% | 98.9% | 93.8 / 96.8% | 9.5 / 8.7% | 9.3 / 9.0% |
| r0 | 99.1% | 98.4% | 96.3 / 93.5% | 9.7 / 9.9% | 11.3 / 9.6% |
| r1 | **10.1%** | 97.2% | 0.0 / 89.7% | 9.2 / 11.0% | 9.1 / 9.1% |
| r2 | 98.6% | 97.9% | 94.5 / 92.6% | 8.3 / 10.3% | 10.2 / 9.0% |

The 95% confidence intervals (bootstrapped over 96 documents) are all within about 1.5 points of these numbers. Feed the model a different document's partner and it drops straight back to guessing, so this isn't the translator learning some shortcut.

r1 is the one failure, covered below.

Something I wanted to rule out is that this is just two models voting. If you skip the translator and simply average the two models' final guesses, A gets 52-94% of these answers, depending on the unit. With the translator it's 97.5-99.5%. The partner's information is getting into the model's own thinking, not just being blended at the end.

## Result 2 - it travels through specific experts

My first version of this test looked at which partner expert each translator paid the most attention to, then hid that expert to see if the transfer broke. In the old single run it did, and it made a nice headline.

In the sweep it mostly didn't. In 9 of the 13 directions, hiding the most-watched expert cost less than 2 points, and it was the right expert in only one of them (r0, A from B). Hiding a *different* single expert collapsed the transfer. Where the translator looks isn't the same as where the information is. An outside review flagged this independently.

So I changed the test. On a separate set of documents, hide each partner expert one at a time and pick the one that hurts the most. Then measure that pick on the main test documents. The picked expert was the most damaging one on the test documents too, in all 13 directions that learned to transfer.

| Unit, direction | Nothing hidden | Hide the picked expert | Hide the most-watched expert | Hide the worst 2-3 together |
|---|---|---|---|---|
| s0, A from B | 99.4% | 28.9% | 99.1% | 24.8% |
| s0, B from A | 99.3% | 19.0% | 88.7% | 9.1% |
| s1, A from B | 99.5% | 16.4% | 98.4% | 11.8% |
| s1, B from A | 98.3% | 11.1% | 96.4% | 11.2% |
| s2, A from B | 97.5% | 37.5% | 95.9% | 14.2% |
| s2, B from A | 98.9% | 16.9% | 63.8% | 11.7% |
| r0, A from B | 99.1% | 14.1% | 14.1% (same expert) | 14.7% |
| r0, B from A | 98.4% | 29.4% | 97.7% | 19.1% |
| r1, B from A | 97.2% | 38.3% | 86.0% | 13.5% |
| r1 retry, A from B | 81.3% | 11.9% | 80.8% | 14.0% |
| r1 retry, B from A | 96.5% | 40.7% | 96.5% | 16.5% |
| r2, A from B | 98.6% | 38.4% | 98.4% | 17.3% |
| r2, B from A | 97.9% | 11.7% | 97.4% | 10.4% |

Hiding one expert out of 8 or 16 drops the transfer to 11-41%. Hiding the worst two or three together takes it to 9-25%, close to guessing. Each translator leans mostly on one expert, and a small handful carry nearly all of it.

## Result 3 - but that expert isn't the only way through

The translator never saw an expert go missing during training, so a collapse could just mean its attention broke. To check, I retrained the links for 300 more steps with the picked expert hidden. Transfer came back to 59-96%. The same 300 steps with nothing hidden gives 97-100%.

So the information isn't locked inside one expert. When the usual route is blocked, the translator mostly finds another way through other experts. That's what I want for free transfer: the information is reachable through more than one route.

## Result 4 - the residual stream works just as well

This is the honest one. The residual-stream translator, which can't see individual experts at all, gets 93.7-99.7% on the same test. Paired over the same documents, the expert bus and the residual stream are within 1.3 points of each other everywhere except one case. There the experts are ahead by 5.2 points (s2, B from A). And r1 went the other way, because that's the run where the expert bus failed.

The old single run said the residual stream failed in one direction. That was one seed and it doesn't hold up.

So the expert bus isn't better on accuracy. What it gives me is the ability to see which experts carry the information, and to switch them off. You can't break the residual stream down like that. For a project about transfer *between experts*, that matters more than a point or two of accuracy.

## What didn't work

**r1.** On the r1 pair, A never learned to read from B through the expert bus. It sat at guessing level, while the reverse direction and the residual translator on the same pair were fine. I retried with a different seed for the translator (results in `results/toy_sweep/r1_fseed1`). It got to 81.3% in the same 3000 steps and was still improving when it stopped, and 300 more steps took it to 98.4%. So that pair is slow to learn and sensitive to the seed, but not broken. I'm reporting the original failure as is: 1 of the 12 expert-bus directions in the sweep didn't learn.

**Ordinary text.** Paired up, both models get slightly worse at predicting ordinary text than they are alone. That happens in every unit, with both translator types, by 0.002-0.02 bits per character. It's small but consistent. The architecture doc said the zero-start gate means training only ever adds useful signal. That's only guaranteed at the start, so I've corrected it.

What makes this interesting is that a plain average of the two models' guesses *is* better than either model alone on the same text (by about 0.07-0.11 bits, measured where both tokenizers line up). So the two models do make different mistakes, and there's something to gain from each other. The translator just doesn't pick it up. It was trained half on recall, a task that forces transfer, and nothing rewards it for helping on ordinary text.

**Caveats.**
- Toy scale. A recall answer is only a few bits of information.
- Answers are scored teacher-forced (the model is shown the correct earlier digits), not freely generated.
- 3 seeds per pair type, which is enough to see the pattern but not a lot.

## What this does and doesn't show

It shows a working channel between two frozen MoEs, through their experts. It doesn't show that an expert can be moved into another model and keep working there.

The translator is trained for one pair and one task. It doesn't yet show that it works with a new partner without retraining, or on a task it never saw.

The models still work fine alone, because they're frozen and nothing changes when they're unpaired. It doesn't show that pairing them helps on ordinary text. Yet.

The sending side is properly expert-level. The receiving side isn't: the information lands in the receiver's residual stream, not in its experts.

Even with all that, this gives me confidence. The transfer showed up in both directions in 5 of the 6 pairs straight away, and in the 6th with a different seed. It survived a strict leak check on every run. The expert-level attribution held up on documents it wasn't picked on. That's a solid base to build the rest on.

## Where to next

1. **Complementary models.** Train A mostly on code and B mostly on prose, so each knows things the other doesn't. Then train the translator on ordinary mixed text only, no recall task. Success means each model gets better at the domain it's weak at, while still being fine on its own. This is the first real test of "encourages free transfer".
2. **Free generation.** Score answers the models generate themselves instead of teacher-forced ones.
3. **Off-the-shelf models.** OLMoE and Qwen-MoE, per the work plan.
4. **A more powerful translator, when it's needed.** It doesn't need to be an LLM. It only ever sees the two models' activations, never text, so it's closer to a large latent model. A non-generative decision model in the style of Jev (TypeSafe AI) is one option: one part decides which partner expert to read and whether to transfer at all, and a second part maps the expert's output across. When the small links stop being enough, I'll test an LLM-style translator against alternatives like that.

## Reproduce

Run sheet: [SWEEP-RUNSHEET.md](SWEEP-RUNSHEET.md). In short, on one GPU:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_sweep.yaml 4
```

Results land in `results/toy_sweep/<unit>/`, with everything above in `results/toy_sweep/summary.json`. The r1 retry is `results/toy_sweep/r1_fseed1/config.yaml` (the r1 config with `fused.seed: 1`), run through `toy_fused.py`, `toy_fused_audit.py`, `toy_fused_experts.py` and `toy_fused_hide.py`. The attribution numbers from the old attention-based pick are kept in each unit's `attnsel/` folder and in `results/toy_sweep/attnsel_summary.json`.
