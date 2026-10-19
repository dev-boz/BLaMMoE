# Handoff — BLaMMoE, state as of 2026-10-19

For whoever picks this up next, human or agent, on a local machine. Everything is on
branch **`claude/new-session-f94vqn`** of `dev-boz/BLaMMoE`. `main` only has the original
foundation commit. Read `CLAUDE.md` first: it is the working agreement and now includes
the rules learned the hard way below.

## Where things stand (one paragraph)

At toy scale, two independently trained, frozen MoE language models with different
tokenizers, widths and expert counts pass information to each other expert-to-expert.
They do it through small trained translators that choose which of the partner's experts
to read. Under a leak-proof mask, each model answers questions only its partner can see
(A 95.1%, B 99.8%; wrong-partner control ~10%). Hiding the single expert each
translator chose collapses that (95.1% → 14.0%, 99.8% → 29.1%), while hiding any other
expert costs under 2 points. Reading the partner's whole residual stream instead fails
in one direction (10.3%). An earlier "fused model predicts ordinary text better than
either alone" result was a **tokenizer leak** and is withdrawn. No fusion variant
improves ordinary-text prediction once the leak is closed.

## Get running

```bash
git clone https://github.com/dev-boz/BLaMMoE && cd BLaMMoE
git checkout claude/new-session-f94vqn
uv venv && uv pip install -e ".[dev]"     # CUDA torch on a GPU box; see README for CPU-only
PYTHONPATH=. .venv/bin/python -m pytest -q   # 64 tests, ~1 min on CPU
PYTHONPATH=. .venv/bin/python scripts/hw_probe.py   # GPU boxes: run this before anything long
```

Hardware notes:
- **1080 Ti (Pascal, sm_61):** current default torch wheels (CUDA 13) almost certainly
  don't support it. If `hw_probe` says `arch_supported_by_this_torch: false`, reinstall
  torch from an older CUDA build (`--index-url https://download.pytorch.org/whl/cu126`).
- **192 GB RAM + 2× CMP 170HX (memory-unlocked):** run `hw_probe` first. It checks the
  unlocked VRAM with a fill-and-read-back test, measures fp32/fp16/bf16 speed (mining
  cards can be throttled in some precisions) and the real PCIe link. Pick the dtype
  from its output.
- Device selection is automatic (`blammoe/device.py`); `BLAMMOE_DEVICE=cpu|cuda|cuda:1`
  overrides it. `toy_train.py` and `toy_fused.py` use the GPU. The analysis scripts
  (`toy_fused_{analyze,experts,hide,audit,reward}.py`, `toy_poc.py`,
  `toy_followup.py`) still run on CPU. They are fast enough, but are the first thing
  to port if you need them faster.

## Reproduce the headline results

`results/` is git-ignored (CLAUDE.md rule 5), so trained checkpoints are **not** in the
repo. Regenerate them. Times are from a 4-core CPU container. Untested on GPU, but
these models are tiny, so expect far faster.

```bash
export PYTHONPATH=.
# 1. Pretrain the two toy MoEs (~50-65 min each on CPU; run in parallel with 2 threads each)
python scripts/toy_train.py configs/toy_poc.yaml model_a 2
python scripts/toy_train.py configs/toy_poc.yaml model_b 2
# 2. One-way transmission POC + ablations (docs/toy-poc-results.md)
python scripts/toy_poc.py configs/toy_poc.yaml 4
python scripts/toy_followup.py configs/toy_poc.yaml 4 3000
# 3. Leak-proof two-way fusion: expert-level vs residual-level (~1.5-2 h each on CPU)
python scripts/toy_fused.py configs/toy_poc.yaml ab_experts_all_rel_safe 2
python scripts/toy_fused.py configs/toy_poc.yaml ab_resid_all_rel_safe 2
# 4. Which experts were chosen, single-expert ablation, causality audit
python scripts/toy_fused_experts.py configs/toy_poc.yaml ab_experts_all_rel_safe 4
python scripts/toy_fused_hide.py configs/toy_poc.yaml ab_experts_all_rel_safe 4
python scripts/toy_fused_audit.py configs/toy_poc.yaml 4 16 ab_experts_all_rel_safe ab_resid_all_rel_safe
```

All experiment settings live in `configs/toy_poc.yaml` (variants under
`fused.variants`). The local CPython stdlib is the code corpus, so a different Python
version gives a slightly different corpus. Expect numbers close to, not identical with,
the write-ups.

## Results docs

| doc | what |
|---|---|
| `docs/toy-poc-results.md` | One-way: A's expert → B (linear transplant 96.5%, learned bridge 92%), depth/expert/data ablations. Unaffected by the leak (B reading char-level A cannot leak). |
| `docs/toy-fusion-results.md` | Two-way fused pair, translator expert choice, depth, stage-2 fine-tuning, **the audit** (banner at top) and the **leak-proof results** (section at the end). |
| Shared summary page | https://claude.ai/code/artifact/7accd112-77a6-4a28-90bc-0bcdcafec03e (private until shared from its Share menu) |

## Rules learned the hard way (also in CLAUDE.md)

1. **Cross-tokenizer fusion must use `safe_mask` and pass the audit.** BPE boundaries
   on the full text depend on later characters, so a partner reading BPE tokens "up to
   char i" learns about char i+1 from their mere existence. The chars-read mask,
   wrong-document, wrong-position and copy-of-itself controls cannot detect this. The
   fix: a token is readable only once its pre-token chunk is closed
   (`closed_ends` / `run_pair`). The check: `scripts/toy_fused_audit.py` (full-text
   scoring must equal prefix-honest scoring). `tests/test_audit.py` pins both.
2. **Expert-level claims need routed-only expert outputs.** Evaluating an expert
   densely on every token pushes the whole residual stream through its weights, so even
   a never-routed expert "transmits" (36-69% in the POC).
3. **Rewards on fusion wins must be partner-specific and per-token weighted.** Novelty
   alone is rewarded just as much with a wrong partner (bridge noise). A win term
   averaged over the ~2% win tokens weighted each ~100× and wrecked accuracy.
4. A fresh Adam on already-trained links needs warmup: 3 steps took bits/char from
   1.651 to 1.683.

## Next steps, in priority order

1. **Independent adversarial audit** of the eval path before any scale-up. Use a fresh
   session with no stake in the results, told to *find leaks and eval bugs*, not to
   confirm. Cheaper to catch at 3M params than at 7B. (External-review recommendation.)
2. **Review sweep on GPU** (the external review's asks). The pieces are built and
   tested; the runner is not written yet:
   - `blammoe/toy/fused_train.py`: `train_fused`, `split_scores(per_doc=True)`,
     `general_bpc`, `ensemble_baseline` (no-communication mixture of the two solo
     models), `hide_ablation`.
   - `blammoe/stats.py`: pooled-accuracy bootstrap CIs over documents,
     `paired_bootstrap_diff`, `seed_summary`.
   - To build: `scripts/toy_sweep.py` + `configs/toy_sweep.yaml`. Pretrain 3 seeds each of
     A and B (`ToyModelSpec.seed`); for each seed pair train `experts_safe` and
     `resid_safe`; run the hide ablation and ensemble baseline; and add a **Rung-2
     partner** (same architecture, different seed, e.g. B↔B′). Make it resumable (skip a
     job whose output exists), and report bootstrap CIs plus mean ± sd across seeds.
3. **Data-difference dose-response:** train B on progressively different data from A
   (shifted code/prose mix → disjoint files → no code) and watch transfer fall off. This
   tests how much survives different labs' data. Needs a file-subset option in
   `blammoe/toy/corpus.py`.
4. **Stage-2 reward-weight sweep** (λ ≈ 10-30) under `safe_mask`. λ=2 did nothing and
   ~100× wrecked accuracy; the interesting range is between. Then reward on generated
   text, which needs lockstep generation across two tokenizers (not built).
5. **Real models (the actual goal):** WP1 `blammoe/models/loader.py` + verify the
   VERIFY-tagged registry against real OLMoE-1B-7B and Qwen1.5-MoE-A2.7B checkpoints.
   Check whether the installed `transformers` still exposes experts as a ModuleList.
   Then WP2 collection with **routed-only** expert outputs, then port the fused pair.
   192 GB RAM holds both models (≈14 GB + ≈29 GB bf16).

## Open caveats on what *is* claimed

- Toy scale (2-3.5M params), one seed per result, no CIs yet (next step 2).
- The channel is narrow: a recall answer is ~3 bits, mostly carried at the last layer.
  First-block-only exchange reached 38.6% one way (under the old mask; not rerun leak-proof).
- Both toy models saw the same training data, the friendliest case for alignment.
- Expert-level > residual-level is one seed and one budget. A longer or reseeded
  residual run could still learn.
- The ensemble baseline is implemented but not yet run. Note that for split recall an
  output-level mixture trivially gets the informed model's answer. The claim is about
  information entering a model's *own* computation, so the mixture is a reference
  point, not a competitor.
