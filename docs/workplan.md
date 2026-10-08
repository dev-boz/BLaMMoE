# BLaMMoE Phase 1 — Work Plan

The spec (`../blammoe-spec-and-bibliography-v4.md`) decomposed into independently
buildable work packages (WPs). Each WP is sized for one agent to pick up, build
**wired + tested**, and hand back. Honor the interface signatures so packages
compose.

**Definition of done (every WP):** the named module(s) exist, a `tests/` file
exercises them with real assertions, `pytest` is green, and any VERIFY value the
WP touched has been confirmed against the real artifact and updated in the repo.

## Legend

- **Runs on** — `any` = pure Python / synthetic tensors, runnable on this CPU box
  now; `model` = needs a real MoE download (the 128 GB / 1080 Ti box or rented GPU).
- **Status** — `built` / `open` / `blocked(WPx)`.

## Compute reality

Phase 1 is dominated by **one-time activation harvesting** (forward passes, cache
to `data/`) plus **cheap offline linear algebra**. Collect activations once on the
big-RAM box; then the alignment/metric/analysis WPs run anywhere, including here.
Do not couple WPs to a GPU.

---

## WP0 — Foundation  ·  Runs on: any  ·  Status: built

Config system + model registry.

- `blammoe/config.py` — `ExperimentConfig.from_yaml` with nested `PoolingConfig`,
  `AlignConfig`. **built** (`tests/test_config.py`).
- `blammoe/models/registry.py` — `ModelSpec`, `REGISTRY`, `get()`. **built**.
- Remaining/open: add a `device()` helper and a `results` path helper if a later
  WP needs them — add them *with* their test, not before.

## WP1 — Model loading + expert-FFN hooks  ·  Runs on: model  ·  Status: partial

The core primitive: capture one expert's own FFN output, pre-router-weight and
pre-residual (spec 1.1).

- `blammoe/hooks.py` — `ExpertCapture`, `expert_path`, `resolve_module`. **built**
  and proven against a synthetic MoE (`tests/test_hooks.py`).
- **Open:** `blammoe/models/loader.py` — `load_model(spec) -> (model, tokenizer)`
  using `transformers`, frozen (`eval()`, `requires_grad_(False)`), dtype/device
  per config. Then **verify the registry**: load OLMoE and Qwen-MoE, print
  `named_modules()`, confirm `expert_module_template`, `hidden_size`, layer/expert
  counts, and update `registry.py` (remove the VERIFY tags you resolve).
- Acceptance: an integration test (marked/skipped without weights) that loads a
  real checkpoint, resolves an expert via the template, runs one forward pass, and
  asserts `ExpertCapture.output.shape[-1] == hidden_size`.
- Papers: OLMoE `2409.02060` (arch).

## WP2 — Paired data collection  ·  Runs on: model  ·  Status: partial

Build the paired `(h_A, h_B)` activation corpus (spec 1.3).

- `blammoe/data/corpus.py` — assemble a varied corpus (Wikipedia + code + math)
  as raw text with char offsets preserved.
- `blammoe/data/pooling.py` — **character-span pooling** for tokenizer mismatch:
  given each model's tokenization with char spans, pool token states over a
  character window (e.g. chars 100–120); **discard** spans within
  `pooling.boundary_margin` chars of either model's token boundary (approximately,
  not perfectly, aligned pairs). *This module is pure string/index logic — unit
  test it on synthetic tokenizations with no model (Runs on: any).*
- `blammoe/data/collect.py` — stream corpus through both frozen models, capture
  the selected expert at each swept depth via `ExpertCapture`, write paired arrays
  to `data/` (suggested: `.npz` or memmap; store `h_A`, `h_B`, char-span id, layer).
- Acceptance: pooling unit tests (boundary discard, window math); a small
  end-to-end collection run producing a paired file whose two halves have equal row
  counts.
- **`pooling.py` built** (`tests/test_pooling.py`): `pool_char_spans(offsets_a,
  states_a, offsets_b, states_b, windows, boundary_margin) -> PooledPairs` and the
  config-driven `pool_document(..., text_len, cfg.pooling)`. Offsets are HF
  `offset_mapping` spans; zero-width (special) tokens are ignored. **Margin
  semantics** (the spec wording is ambiguous): each model mean-pools the tokens
  lying fully inside the window, and the pair is dropped if either model's pooled
  tokens leave more than `boundary_margin` chars uncovered at either edge.
  `boundary_margin=0` keeps only windows on boundaries shared by both tokenizers.
- **Open for `collect.py`:** a forward hook on a routed expert sees only the
  tokens routed *to that expert* (HF MoE blocks index-select before calling it),
  so `ExpertCapture.output` is `[n_routed, d]`, not `[T, d]`. Pooling needs a
  per-token `[T, d]` array: either recover the routed token indices, or apply the
  expert FFN densely to the layer's MLP input for every token. Decide and document.
  Also check the installed `transformers` version still exposes experts as a
  `ModuleList` — some versions fuse them into one module with 3-D weights, which
  would break `expert_module_template` (WP1 VERIFY).
- Depends on: WP1.

## WP3 — Expert selection  ·  Runs on: model  ·  Status: open

Two-stage filter (spec 1.3) — routing stats are a *coarse pre-filter only*; the
primary criterion is operation-level specialization.

- `blammoe/selection/routing_stats.py` — domain-level routing frequency per expert
  (coarse pre-filter), à la OLMoE's domain analysis.
- `blammoe/selection/concept_probe.py` — **k-sparse concept probing at the
  operation level** (e.g. "closes a LaTeX bracket"): the primary selector.
- Output: ranked `(layer, expert)` candidates per model, written to `results/`.
- Papers: "The Expert Strikes Back" `2604.02178` (why operation-level, not domain),
  OLMoE `2409.02060` (domain routing stats).
- Depends on: WP1.

## WP4 — Alignment fitters  ·  Runs on: any  ·  Status: built

Supervised maps from `H_A` to `H_B` (spec 1.4). Pure NumPy/SciPy.

- `blammoe/align/procrustes.py` — `fit_orthogonal(H_A, H_B) -> R` via SVD
  (`scipy.linalg.orthogonal_procrustes`); same-dim only (Rungs 1–2).
- `blammoe/align/linear.py` — `fit_linear(H_A, H_B, ridge=0.0) -> W` least-squares,
  rectangular (Rung 3), with unconstrained and ridge variants.
- Common `apply(map, H)` and a `residual(map, H_A, H_B)` helper.
- **Built** (`tests/test_align.py`). Fitters return a `LinearMap` (W + centering
  means) from `blammoe/align/base.py`; `residual` is relative (= 1 − R²), so 1.0 is
  exactly the predict-the-target-mean / Transmission-Illusion baseline — always
  evaluate it on held-out pairs. `fit_from_config(cfg.align, H_A, H_B)` dispatches
  on the rung YAML. Extras beyond the contract: `fit_orthogonal(..., scale=True)`
  (similarity Procrustes, absorbs a global norm gap between checkpoints), and
  `fit_orthogonal` also accepts `d_A≠d_B` (semi-orthogonal; the Atlas-Alignment
  setup) as a Rung-3 baseline next to lstsq/ridge.
- Acceptance: recover a known random orthogonal `R` on synthetic data to ~machine
  precision; ridge shrinks toward zero as λ→∞; rectangular map handles `d_A≠d_B`.
- Papers: Atlas-Alignment `2510.27413`, "Transferring Features…" `2506.06609`,
  Søgaard `1805.03620` (why supervised over unsupervised here), Platonic-brain
  `2605.20496` (orthogonal works — do **not** default to ridge without evidence).

## WP5 — Metric battery + validate-the-validators  ·  Runs on: any  ·  Status: built

Trust the metrics before trusting them on the real pair (spec 1.3, item 5).

- `blammoe/metrics/cka.py`, `knn.py` (mutual k-NN), `procrustes_residual.py`, and
  `battery.py` combining all three — **never CKA alone** (`2210.16156`).
- A "validate the validators" harness: expect high agreement on a **known-good**
  pair (same run, different checkpoint) and low on a **known-bad** pair (trained vs
  random init). Wire this as the Rung-1 gate.
- Acceptance: on synthetic data, known-good (identity + noise) scores high on all
  three metrics; known-bad (independent random) scores low; CKA-vs-battery
  divergence is demonstrated on an outlier-injected case.
- **Built** (`tests/test_metrics.py`). `run_battery(H_A, H_B) -> BatteryResult`
  (linear CKA, mutual k-NN à la Huh et al. 2405.07987, held-out similarity-Procrustes
  residual on row-L2-normalized states à la Atlas-Alignment); `rung1_gate(good, bad)`
  passes only if *every* metric separates the pair. Outlier case reproduces Davari
  Cor. 3: one point moved by 1000 drops CKA 0.9999 → 0.09 while k-NN and residual
  are unchanged. Gate gaps (0.1 each) are a judgment call — recalibrate on real
  Rung-1 data.
- Papers: CKA reliability `2210.16156`, Relative Representations `2209.15430`.

## WP6 — Injection adapter (Tx/Rx + Latent Bus)  ·  Runs on: model  ·  Status: partial

Let the target model *read* the foreign vector (spec 1.1). Hub-and-spoke: one
Tx/Rx pair per model into a Universal Latent Bus.

- `blammoe/adapters/gated_xattn.py` — **tanh-gated cross-attention, gate
  zero-initialized** (Flamingo/CALM pattern) so the model learns how much foreign
  signal to read; `TxAdapter` (model space → bus), `RxAdapter` (bus → model space).
- Shape/gradient unit tests with synthetic tensors are `any`; training against real
  states is `model`.
- Acceptance: at zero-init gate the injected contribution is exactly zero (identity
  behavior); shapes compose through Tx→bus→Rx; gate is a learnable scalar/vector.
- **`blammoe/adapters/gated_xattn.py` built** (`tests/test_adapters.py`): `TxAdapter`
  (LayerNorm + linear into the bus) and `RxAdapter` (target residual cross-attends
  over the bus, added through `tanh(gate)`, gate = 0 at init → bit-for-bit
  identity). Takes a `[B, T_tgt, T_src]` block mask, which is how the toy POC
  enforces causality across mismatched tokenizers. Trained end-to-end in WP-T;
  training against real checkpoints is still open.
- Papers: CALM `2401.02412`, Flamingo `2204.14198`, MoT `2509.21164`, BRIDGE
  `2511.11526`, Zipper `2405.18669`; Symphony-MoE `2509.18542` (interference risk).
- Depends on: WP1, WP4.

## WP7 — Test A: Graded Readout  ·  Runs on: model  ·  Status: open

Decode the transported state (spec 1.5). Patchscopes + Jacobian Lens (Subtext, for
Qwen rungs). Ceiling condition = native self-injection.

- `blammoe/efficacy/readout.py`. Papers: Patchscopes `2401.06102`, Subtext
  (`papers/ninjahawk_Subtext.md`). Depends on: WP6.

## WP8 — Test B: Divergence-Transport (definitive)  ·  Runs on: model  ·  Status: open

Mine calibration positions where the two models disagree on the next token; inject
the translated state; measure **Δlogprob of the source model's preferred token**.
Guards against the "Transmission Illusion" (mapping to the target's mean).

- `blammoe/efficacy/divergence.py`. Depends on: WP4, WP6.

## WP9 — Test C: Shuffled-Position Control  ·  Runs on: model  ·  Status: open

Correct vector, wrong position — isolates position-specific content from generic
domain drift. `blammoe/efficacy/shuffle.py`. Depends on: WP8.

## WP10 — Test D: Generation-Robustness Control  ·  Runs on: model  ·  Status: open

**Load-bearing, not optional.** Repeat calibration + Tests A/B on the source
model's own *freely-generated* continuations, not teacher-forced text — the
distribution the map actually meets at deploy time.

- `blammoe/efficacy/generation.py`. Motivation: "The Latent Bridge" `2606.24470`
  converged offline (KL=0.004) then failed at deployment. Depends on: WP8.

## WP11 — Experiment orchestration  ·  Runs on: model  ·  Status: open

`blammoe/experiment/run.py` + `scripts/run_rung.py`: given a rung config, drive
selection → collection → fit → metric battery → efficacy tests, logging to
`results/`. Depends on: WP2–WP10.

---

## WP-T — Toy proof of concept  ·  Runs on: any  ·  Status: built

The whole Phase-1 question at toy scale, runnable on this CPU box with no
downloads: two tiny MoE LMs trained from scratch (`blammoe/toy/`), then
transmission of ONE expert's output from A into B.

- A: char-level, d=128, 8 experts. B: BPE-1024, d=96, 16 finer experts,
  different seed. (Stands in for Rung 3: tokenizer, width, expert granularity.)
- Corpus: CPython stdlib (code), `papers/` (prose), synthetic key→value recall.
- **Blindfold test** (a controlled Test B): B reads the recall prompt with stored
  values masked, so it cannot know the answer digits (chance 10%); A reads it
  clean. Any accuracy gain in B must arrive through the translator.
- Carriers: (1) zero-shot linear transplant — WP4 ridge map A-expert → B-expert
  fitted on clean pairs, swapped into B's MoE; (2) learned bridge — WP6 Tx/Rx
  trained on B's LM loss, both models frozen.
- Controls: shuffled-document A signal, blindfolded A, matched-norm random,
  wrong-position (Test C, transplant only), plus damage to B's ordinary-text loss.
- Variants: A's operation-level "answer specialist" expert (most routed at
  answer positions) vs its least-routed expert, dense vs routed-only expert
  output, and A's whole residual stream as the CALM/MoT-style baseline.
- Run: `scripts/toy_train.py configs/toy_poc.yaml model_a|model_b`, then
  `scripts/toy_poc.py configs/toy_poc.yaml`, then `scripts/toy_followup.py`.
  Results: `docs/toy-poc-results.md`. **Outcome:** routed-only transplant of A's
  answer-specialist expert into B's last-layer expert, 8% → 96.5%; learned
  bridge mid-stack 92.0%; all controls at chance. For the real models, use
  routed-only expert outputs (dense ones leak the residual stream) and a
  calibration corpus that covers the transmitted content.

## Dependency graph

```
WP0 ─ built
WP1 ─(hooks built; loader open)─┬─ WP2 ── WP3
                                │          │
WP4 (built) ────────────────────┼──────────┤
WP5 (built) ────────────────────┘          │
                                 WP6 ──┬── WP7
                                       ├── WP8 ──┬── WP9
                                       │         └── WP10
                                       └──────────── WP11
```

## Good first tasks (no GPU, runnable on this box now)

1. ~~**WP4 — alignment fitters.**~~ *(built)* Self-contained NumPy/SciPy; synthetic acceptance.
2. ~~**WP5 — metric battery.**~~ *(built)* Same; validates a load-bearing methodological claim.
3. ~~**WP2 pooling module only**~~ *(built)* (`data/pooling.py`) — pure index logic, unit-testable
   without any model. Leaves `collect.py` for the model box.

These three unblock the math-heavy core of the pipeline and need nothing but the
CPU env, so they are the ideal first hand-offs.
