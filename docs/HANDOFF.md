# Handoff — BLaMMoE, state as of 2026-10-20

For whoever picks this up next, human or agent. Read `CLAUDE.md` first (working agreement).
`main` has the toy programme (PR #1 merged). Audit fixes are in progress on local branch
**`audit-fixes`** (committed locally, not pushed).

## Project goal (from the owner)

Step one is proving that experts can be translated between models **at all**. Improvements
are a bonus. Mixing e.g. a Qwen with a Mistral and getting a *worse* result still counts as
success. So prioritise work that makes "information transfers, and it travels via experts"
airtight over ordinary-text quality.

## START HERE: audit fixes in progress

A 3-model adversarial audit (Claude, Codex GPT-6.1-Sol, Gemini 3.8 Flash) is merged in
`docs/audit-2026-10-19.md`, with the status table at the top. Raw reports and probes are
in `docs/audit-2026-10-19/`.
- **No leak was found in safe-mask fusion.** The weak points are attribution, data hygiene and stale numbers.
- **Done on `audit-fixes`:**
  - M7: a per-position, two-way, float64 causality gate. `scripts/toy_fused_audit.py` exits 1 on failure.
  - The `safe_mask` guard: a safe pair refuses a direct call without writer ends.
  - Boolean causal masks: the float64 causality bug.
  - M1: `native_transplant_hook`, with the old hook kept as the `dense_additive` control.
  - M2: wiring for a routed-source long bridge.
  - M5: selection seeds separate from reported seeds.
  - M3: attribution controls (details below).
  - M6: a train / val / test split of `papers/` and the stdlib by file, with each stdlib package kept whole (details below).
  - 87 tests pass.
- **M3 is built** (wiring run on the untrained fixture). See the M3 row in the audit status table. Results need trained checkpoints, so they arrive with the retrain (step 2 below). Run `scripts/toy_fused_experts.py` before `scripts/toy_fused_hide.py`: the hide script takes its chosen experts and blank constants from the seed-2026 selection set.
  - How to read it: if **blank** collapses like **hide**, the information is in the expert's output, not its tag or slot. If **tags zero/shift** costs little, the reader doesn't rely on the tag. If **joint** hiding takes B←A from 29.1% to about chance, the residual was link 0/1. If **recovery fine-tune** brings accuracy back, the information also reaches the partner through other experts, so it isn't exclusive to the chosen ones; compare against the `none` arm.
  - `hide` (all experts) is **not** the same as links off: a reader left with only the null slot still gets the attention biases. Use `disabled=` for "solo".
  - `docs/audit-2026-10-19/codex_audit_checks.py` still uses the old `hide_experts=` keyword. It is an archived probe, so it was left as is.
- **M6 is done** (2026-10-20). See the M6 row in the audit status table.
  - **Rule:** anything fitted or trained (BPE, pretraining, bridges, links, linear fits) reads `setup.texts(cfg, "train")`, selection reads `"val"`, reported ordinary-text numbers read `"test"`. `setup.sampler(cfg, split, seed)` has no default split. Split-recall and recall samples are synthetic and stay seed-separated.
  - **Packages, not just files.** The scoped design split per file. That left ~5% of held-out windows verbatim in train, because `encodings/cp*.py` repeat whole codec tables. Hashing the top-level package name instead cut it to ~0.2% (val) / ~0.8% (test) of windows. What remains is copied license headers and copied code, and the test allows at most 2%. Splits: train 2.1M prose + 6.0M code chars (capped), val 0.25M + 0.97M, test 0.29M + 0.79M.
  - **Every old checkpoint is pre-M6.** The models, BPE and fused links in `results/toy_poc` were fitted on what is now val/test text. The BPE cache is now `bpe1024_train_v0.1_t0.1.json`, so the old `bpe1024.json` is ignored, but old `model_*.pt` / `fused_*.pt` files would still load. Retrain from pretraining ("Reproduce", step 1) before reporting anything.
- **Next, in this order:**
  1. **Small fixes:**
     - the bootstrap counts 0/0 resamples as accuracy 0 (`blammoe/stats.py:24,37`);
     - the enrichment-ratio baseline should count only legal entries (`fused.py` `expert_attention_mass`);
     - report exact 4-digit answer accuracy alongside digit accuracy.
  2. **Retrain once, then run the sweep (Next steps 2 below).** Include ≥3 seeds for residual vs expert (M8). Every fused checkpoint must pass the audit gate. Re-measure the leaky-run numbers in the docs (M4: depth table, expert-choice shares, stage 2).
- **Also in the audit doc: expert-tagged buses are sensitive to routing near-ties.** In float32 on GPU, rounding that differs by batch shape flips near-tied top-k choices, and the N(0,1) `eid` embeddings amplify the flip. Keep that in mind when interpreting the hide/ID results.

## This machine (boz-linux)

- **Repo and environment:** `~/BLaMMoE`. The `.venv` was built with pip (torch 2.14.1+cu130).
- **Disk:** the root disk is ~95% full. `results/toy_poc` is a symlink to `/media/boz/Data/blammoe/results/toy_poc` on the NTFS drive. If that drive is unmounted, run `udisksctl mount -b /dev/nvme0n1p2`. Set `HF_HOME=/media/boz/Data/blammoe/hf_cache` before downloading models.
- **GPUs:** 2× CMP 170HX, both healthy: ~195 TFLOPS bf16, ~11 fp32, 64 GB each, passed a 60 GB read-back. `42:00` runs at Gen2 ×4 (slow host transfers) and `43:00` at Gen2 ×16. Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`; then CUDA 1 = `42:00`, 2 = `43:00`, 0 = the 3060 Ti (don't use it). Run one card at a time and tell the owner before each GPU job. There is no P2P, and none is needed: run one job per card.
- **Test fixture:** for CLI wiring checks without trained checkpoints, copy `configs/toy_poc.yaml` with `out_dir` pointed elsewhere, then run `PYTHONPATH=. python docs/audit-2026-10-19/gate_fixture_build.py <that.yaml>`. It builds untrained models plus live links with gates = 1. To shorten training scripts, also set the per-variant `steps:` under `fused.variants`: it overrides `fused.steps`.

## Where things stood after PR #1 (one paragraph; see the audit for caveats)

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
git checkout audit-fixes   # or main
uv venv && uv pip install -e ".[dev]"     # CUDA torch on a GPU box; see README for CPU-only
PYTHONPATH=. .venv/bin/python -m pytest -q   # 82 tests, ~10 s on CPU
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python scripts/hw_probe.py   # ONE card at a time (it probes every visible GPU)
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
# 4. Which experts were chosen (selection set), attribution controls + recovery fine-tunes, causality audit
python scripts/toy_fused_experts.py configs/toy_poc.yaml ab_experts_all_rel_safe 4
python scripts/toy_fused_hide.py configs/toy_poc.yaml ab_experts_all_rel_safe 4
python scripts/toy_fused_audit.py configs/toy_poc.yaml 4 ab_experts_all_rel_safe ab_resid_all_rel_safe
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
