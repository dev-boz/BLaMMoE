# Papers index — which source backs which work package

All 33 sources are in `papers/` as markdown (already fetched). Read the paper(s)
for a WP before implementing its method. Status tags are from the spec's
bibliography as refined by `../verification_report.md`. `[VERIFIED]` unless noted.

## WP6 — injection adapter (Tx/Rx + gated cross-attention)

| File | Source | Why |
|---|---|---|
| `2401.02412.md` | CALM | Frozen anchor + frozen augmenter, small trained cross-attention connectors |
| `2204.14198.md` | Flamingo | Tanh-gated cross-attention, gate init zero (the exact injection pattern) |
| `2509.21164.md` | Mixture of Thoughts | Interaction layers project into shared latent; primary cross-attends over peers |
| `2511.11526.md` | BRIDGE | Bidirectional cross-only attention, gated residual updates, cycle-consistency |
| `2405.18669.md` | Zipper | Multi-tower decoder fusing modalities via gated cross-attention |
| `2509.18542.md` | Symphony-MoE | Interference warning: naive expert stitching → catastrophic interference |
| `2312.13699.md` | Adapt & Align | Translator network aligning local latent to a consolidating global one |
| `babycommando_neuralgraffiti.md` | Neural Graffiti | Injection-hook *engineering pattern* only (self-described unvalidated) |

## WP4 — alignment fitters (Procrustes / least-squares / ridge)

| File | Source | Why |
|---|---|---|
| `2510.27413.md` | Atlas-Alignment | Orthogonal Procrustes ≥ unconstrained linear for same-dim retrieval |
| `2506.06609.md` | Transferring Features via Model Stitching | Affine maps transfer features across LMs cheaply |
| `2603.18908.md` | Characterizing Linear Alignment | Affine maps between final hidden states; disambiguate vs the above |
| `1805.03620.md` | Søgaard et al. | Why *supervised* here: unsupervised presumes isomorphism, fails confidently |
| `2605.20496.md` | Platonic Reps in Human Brain | Orthogonal alignment works well — don't default to ridge (VERIFIED–CAUTION) |
| `2505.12540.md` | vec2vec | Unsupervised translation (the *unpaired* case that does NOT apply here) |
| `2602.06205.md` | Multi-Way Alignment (GCPA) | N≥3 orthogonal-vs-CCA tradeoff — Phase 2, not Phase 1 |
| `1602.03992.md` | Sparse PCA | TANGENTIAL: general Procrustes/Stiefel math, keyword-matched |

## WP5 — metric battery

| File | Source | Why |
|---|---|---|
| `2210.16156.md` | CKA reliability (Davari) | CKA is manipulable/outlier-sensitive → use a battery, not CKA alone |
| `2209.15430.md` | Relative Representations | Anchor-relative coords make spaces comparable without a trained map |

## WP3 — expert selection

| File | Source | Why |
|---|---|---|
| `2604.02178.md` | The Expert Strikes Back | Experts are operation-level specialists, not domain — drives the selector |
| `2409.02060.md` | OLMoE | 64 fine-grained experts show real domain specialization (coarse pre-filter) |

## WP7 — Test A readout

| File | Source | Why |
|---|---|---|
| `2401.06102.md` | Patchscopes | Patch a hidden rep into a target context; model decodes it; cross-model |
| `ninjahawk_Subtext.md` | Subtext | Open Jacobian-Lens impl on Qwen (for the Qwen rungs' readout) |

## WP10 — Test D (generation robustness) motivation

| File | Source | Why |
|---|---|---|
| `2606.24470.md` | The Latent Bridge | Negative result: converged offline (KL=0.004), failed at deployment |

## Background / merging & routing context (not a direct dependency)

| File | Source | Note |
|---|---|---|
| `2408.07057.md` | MoErging survey | General framework for combining independent experts |
| `2402.05859.md` | PHATGOOSE | Per-expert sigmoid gate on hidden-state similarity |
| `2603.22755.md` | KALAVAI | When post-hoc specialist fusion pays off |
| `UNITES-Lab_MC-SMoE.md` | MC-SMoE | Merge-then-compress within a single model |
| `2606.03391.md` | HARC | Router recalibration after merging — router training is out of scope |
| `2403.13187.md` | Sakana evolutionary merging | CMA-ES search over merge configs |
| `2401.04821.md` | MoSECroT | Stitching with static embeddings; weak-baseline-competitive only |

## Baselines & not-relevant (completeness)

| File | Source | Note |
|---|---|---|
| `2406.04692.md` | Mixture-of-Agents | Baseline: does cross-model mixing help at all |
| `2501.07890.md` | GraphMoE | Adjacent: message passing *within* one MoE, not cross-model |
| `2606.26493.md` | Nemotron-TwoTower | NOT related (same-backbone clones); do not cite as related work |

**Missing from `papers/` (no arXiv ID / not auto-fetched):** BTS (Branch-Train-Stitch),
BTX `2403.07095`, InfiFusion `2501.02795`, AdapterFusion (EACL 2021), Self-MoA,
"Residual Coupling" (SSRN #6746521, `pfekin/residual-coupling` — FLIMSY, replicate
on a toy before trusting). See the spec bibliography for context.
