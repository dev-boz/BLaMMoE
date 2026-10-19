# Adversarial audit brief — BLaMMoE toy-scale results

You are an independent auditor with NO stake in these results. Your job is to
**find leaks, evaluation bugs and confounds** that would make the headline
numbers wrong or overstated. Do not try to confirm the results. A report that
says "I found nothing" must show what you checked and how.

The working directory is a private copy of the repo. You may write scratch
scripts inside it. Do not modify anything outside it.

## Constraints
- **CPU only.** Always run with `CUDA_VISIBLE_DEVICES=""` (GPU hardware rules on
  this box; do not touch the GPUs).
- Python: `/home/boz/BLaMMoE/.venv/bin/python` with `PYTHONPATH=.` (run from
  this directory so the local copy is imported, not the original).
- Trained checkpoints do NOT exist (`results/` is empty). You can build tiny
  models in-process (see `tests/` for small configs) for experiments. Keep any
  single experiment under ~10 min CPU.
- Read first: `docs/HANDOFF.md`, `CLAUDE.md`, `docs/toy-fusion-results.md`,
  `docs/toy-poc-results.md`.

## Claims under audit
1. Leak-proof two-way fusion: under `safe_mask`, each frozen model answers
   split-recall questions only its partner can see (A 95.1%, B 99.8%;
   wrong-partner control ~10%).
2. Hiding the single expert each translator chose collapses accuracy
   (95.1→14.0, 99.8→29.1); hiding any other expert costs <2 points.
3. Residual-stream exchange instead of expert-level fails one way (10.3%).
4. One-way POC: A's expert → B, linear transplant 96.5%, learned bridge 92%.
5. The tokenizer leak is closed: `scripts/toy_fused_audit.py` full-text
   scoring equals prefix-honest scoring.

## Where to look (non-exhaustive)
- `blammoe/toy/fused.py` (`safe_mask`, `closed_ends`, `run_pair`), `blammoe/toy/audit.py`,
  `scripts/toy_fused_audit.py`, `tests/test_audit.py`: is the mask truly causal
  across tokenizers in *every* code path used for the reported numbers (train,
  eval, hide ablation, residual variant)? Can the audit itself miss a leak
  (e.g. it checks the same code path it is meant to verify, tolerances, only some
  positions/variants)? Are there paths where `safe_mask` is off by default?
- `blammoe/toy/corpus.py`, `tokenizers.py`, `batching.py`: how the split-recall
  task is built. Can a model answer from its *own* view (answer-format priors,
  position cues, a small answer space, duplicate/near-duplicate documents across
  train/eval, the answer appearing elsewhere in the visible text)? What is real
  chance level? Is the ~10% wrong-partner control the right baseline?
- `blammoe/toy/fused_train.py`, `bridge.py`, `moe.py`: routed-only expert
  outputs (no dense evaluation), hide ablation implementation (does "hiding"
  actually remove the information, or can it leak through another route like
  the router, residual, normalisation or other experts?), whether the residual
  baseline had a fair budget (params, steps, lr).
- Metrics: accuracy/bits-per-char computation, per-token weighting, what is
  counted as "correct", eval set size and variance, `blammoe/stats.py` bootstrap
  correctness (resampling unit = document?).
- Scripts in `scripts/` vs library code: do the scripts that produced the
  numbers call the leak-proof paths, or older ones?
- Train/eval separation: same documents, same seeds, eval data used for
  selection/early stopping/hyperparameters?

## Output
Write your report to `AUDIT.md` in this directory and also print it as your
final message. For each finding:
- **Severity**: critical (invalidates a claim) / major (could materially change
  a number) / minor / note
- **Where**: `file:line`
- **What**: the defect in one or two sentences
- **Failure scenario**: concrete inputs/state → wrong result
- **Evidence**: what you ran or read that shows it (mark clearly whether it is
  *demonstrated* by an experiment or *suspected* from reading)
- **Fix / check**: the cheapest test that would confirm or clear it

End with a section "Checked and found clean" listing what you examined and
why you believe it is sound.
