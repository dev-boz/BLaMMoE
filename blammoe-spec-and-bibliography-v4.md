# BLaMMoE Phase 1 — Specification & Reference Bibliography (v4.0)

This document has two parts. **Part 1** is the experimental spec, current as of this writing. **Part 2** is a complete bibliography of every source that came up while developing this spec, each tagged with a confidence status. The intent is for an agent with fresh inference budget to independently fetch and validate every entry in Part 2 — especially anything tagged UNVERIFIED or FLIMSY — before the spec relies on it further. Notes on *why* something is flagged are included so the agent isn't starting from zero.

---

# PART 1: Experimental Design

**Target Claim (The Micro-Alignment Problem):** Can highly idiosyncratic, fine-grained MoE experts from independently-trained checkpoints be aligned into a shared cross-model latent space without catastrophic interference?

Out of scope for Phase 1: router training, cross-model task performance gains, N>2 models. The goal is strictly to prove non-destructive, high-bandwidth transmission of an individual expert's own output between two disjoint models.

## 1.1 Core Architecture

Hook point: a specific expert's own FFN output, captured *before* router weighting and before summation into the residual stream — not the post-addition layer hidden state. This is what makes it a test of expert-level transmissibility rather than a restatement of whole-model alignment (already covered by CALM/MoT — see Part 2, Section A).

Hub-and-spoke topology: one Tx/Rx adapter pair *per model*, into a Universal Latent Bus. N=2 models require only 2 adapter pairs, since all experts inside one MoE already share that model's residual stream.

Injection mechanism: tanh-gated cross-attention (initialized to zero), following the Flamingo/CALM pattern — see Part 2, Section A — so the target model learns how much of the foreign vector to read rather than receiving a raw magnitude addition.

## 1.2 The Laddered Progression

1. **Rung 1 (Known-Good):** Same run, different checkpoint. Validates the metric battery. Orthogonal Procrustes applies (dimensions match by construction).
2. **Rung 2 (Moderate):** Same family, different data seed/fine-tune. Orthogonal Procrustes still applies.
3. **Rung 3 (The Goal):** Cross-family (e.g., OLMoE → Qwen-MoE). Dimensions will very likely differ — switch to general linear least-squares.

N=2 validates expert-level transmissibility for this specific pair — not the Micro-Alignment Problem in general, and not the hub topology's actual advantage over a pairwise translator, which only differs at N≥3 (Phase 2).

## 1.3 Data Collection & Expert Selection

1. **Expert selection — two-stage filter.** First pass: use domain-level routing statistics (à la OLMoE's own domain-specialization analysis — Part 2, Section C) to narrow candidates. **Second pass, and the more important one:** the best current evidence (Part 2, Section C — "The Expert Strikes Back") indicates fine-grained MoE experts specialize at the level of specific linguistic/syntactic operations (e.g., closing LaTeX brackets), not broad topic domains — the paper's own framing explicitly rejects "domain specialist" as the right model. Selection should therefore use k-sparse concept probing at the operation level, with domain-level routing stats as a coarse pre-filter only, not the primary selection criterion.
2. **Paired data:** large, varied corpus (Wikipedia, code, math) through both frozen models.
3. **Character-span pooling** for tokenizer mismatch (e.g., characters 100–120); discard spans within a few characters of either model's token boundary — these are approximately, not perfectly, aligned pairs.
4. **Depth sweep:** early, middle, late layers, not one fixed depth.
5. **Validate the validators first:** run the metric battery (CKA + mutual k-NN + Procrustes residual — not CKA alone, see Part 2 Section D on CKA's reliability problems) on a known-good pair (same run, different checkpoint) and known-bad pair (trained vs. random init) before trusting it on the real pair.

## 1.4 Supervised Alignment

- **Rungs 1–2:** orthogonal Procrustes (dimensions match).
- **Rung 3:** general linear least-squares (rectangular). Try both unconstrained and ridge-regularized variants empirically — regularization is motivated by general small-n/large-p overfitting risk when calibration pairs are limited relative to hidden dimension, *not* by a specific literature result (an initial claim that brain-to-brain alignment work favored ridge over orthogonal methods did not hold up on inspection — see Part 2, Section D, "Platonic Representations in the Human Brain" — that paper's actual finding supports orthogonal alignment working well, if anything).
- **Primary engine is supervised, not adversarial/unsupervised.** Paired (h_A, h_B) data is available for free at scale via the character-span method; unsupervised adversarial/cycle-consistency approaches (vec2vec-style) exist for the harder unpaired case that doesn't apply here, and unsupervised alignment specifically risks converging to a confident-but-wrong mapping when isomorphism doesn't hold (Part 2, Section D, Søgaard et al.).
- **Distribution shift check:** teacher-forced calibration states may not represent the source model's free-generation states at deploy time — see Test D below, and note the real, sobering precedent in Part 2 Section F ("The Latent Bridge") where a structurally similar bridge converged beautifully offline and failed at deployment.

## 1.5 Efficacy Tests

Run at each swept depth. A geometric fit (Procrustes/least-squares) alone only proves correspondence, not that the target model can use the result.

- **Test A — Graded Readout:** Patchscopes and the Jacobian Lens (via Subtext, for Qwen rungs) — Part 2, Section E. Ceiling condition: native self-injection.
- **Test B — Divergence-Transport (the definitive test):** mine calibration positions where the two models disagree on the next token; inject translated state; measure Δlogprob of the source model's preferred token. Addresses the "Transmission Illusion" (a translator that just maps toward the target's activation mean, passing coherence checks while transmitting nothing).
- **Test C — Shuffled-Position Control:** correct vector, wrong position — isolates position-specific content from generic domain drift.
- **Test D — Generation-Robustness Control:** repeat calibration and Tests A/B using the source model's own freely-generated continuations, not teacher-forced text. Treat this as load-bearing, not optional — see the negative-result precedent noted above.

---

# PART 2: Reference Bibliography

**Status key:**
- **[VERIFIED]** — primary source fetched and read directly during this project's development; claim confirmed as described.
- **[VERIFIED–CAUTION]** — confirmed real, but with an issue: mischaracterized elsewhere, real-but-irrelevant, or the claim attributed to it doesn't hold up on inspection.
- **[UNVERIFIED]** — mentioned (by an AI research pass, or from general background knowledge) but not independently confirmed against the primary source. Highest priority for agent validation.
- **[FLIMSY – FLAG]** — confirmed to exist, but with real credibility concerns (venue, authorship, lack of independent scrutiny, or an extraordinary claim well outside the range of everything else in this list).
- **[TANGENTIAL]** — confirmed real, likely surfaced by keyword overlap rather than substantive relevance.

## A. Core Architectural Ancestors (injection & latent-fusion mechanism)

| Source | ID / Venue | Claim | Status |
|---|---|---|---|
| CALM (LLM Augmented LLMs) | arXiv 2401.02412 | Frozen anchor + frozen augmenting model, small trained cross-attention connectors; real capability transfer demonstrated | **[VERIFIED]** |
| Flamingo | arXiv 2204.14198 | Tanh-gated cross-attention dense blocks, gate initialized to zero, into frozen LM; ablation shows real degradation without gating | **[VERIFIED]** |
| Mixture of Thoughts (MoT) | arXiv 2509.21164, Fein-Ashley, Parikh, Kannan, Prasanna | Router selects top-K frozen heterogeneous LLM experts; interaction layers project hidden states into shared latent space; primary expert cross-attends over peers; single-pass | **[VERIFIED]** |
| Symphony-MoE | arXiv 2509.18542 | Training-free functional alignment of FFN experts from disparate pretrained models via activation-pattern neuron permutation; naive stitching causes catastrophic interference from "dissonant" parameter-space regions | **[VERIFIED]** |
| BTS | ~Jan 2025, no confirmed arXiv ID | Frozen expert models + trained "stitch layers"; built because Branch-Train-Merge has no learned inter-expert connection | **[UNVERIFIED]** |
| Branch-Train-MiX (BTX) | ~2403.xxxxx, Sukhbaatar et al. | Combines FFN sublayers from domain-expert dense models into one MoE via a router; no cross-expert attention | **[UNVERIFIED]** |
| BRIDGE | arXiv 2511.11526, B. & J. Fein-Ashley (USC) | Bidirectional cross-only attention between vision/text encoder hidden states, gated residual updates, cycle-consistency loss | **[VERIFIED]** — note: same Jacob Fein-Ashley as MoT above |
| AdapterFusion | Pfeiffer et al., EACL 2021 | Two-stage: per-task adapters, then a cross-attention fusion layer over frozen adapters; non-destructive | **[VERIFIED]** |
| Adapt & Align | arXiv 2312.13699, Deja et al. | Continual learning: train local model, align its latent space to a global consolidating model via a translator network | **[UNVERIFIED]** |
| "The Latent Bridge" (game agents) | arXiv 2606.24470 | Cross-attention bridge injecting slow-model residuals into fast model's mid-layers converged well offline (KL=0.004) then **failed catastrophically at deployment**; switched to input-embedding prepending instead | **[VERIFIED]** — important negative result, motivates Test D |
| "Residual Coupling" | Ekin, SSRN working paper #6746521, May 2026 | Linear bridge operators between frozen models' residual streams; claims 80.7% relative perplexity reduction (3-model medical domain) vs. 0.5% for MoE baseline | **[FLIMSY – FLAG]** — real posting, confirmed to exist, but: unaffiliated sole author, SSRN not arXiv (unusual venue for ML), no peer review, 33 downloads / 122 views at time of check, claimed magnitude is roughly an order of magnitude beyond everything else in this bibliography, abstract's mechanism description is more evocative than precisely specified. Do not build on this without independently replicating the core claim on a toy setup first. |
| Zipper | arXiv 2405.18669 | Multi-tower decoder architecture fusing modalities via gated cross-attention | **[UNVERIFIED]** |
| InfiFusion | Yan et al., 2025 | Distills specialized LLMs into a pivot model via iterative distillation + adaptive merging | **[UNVERIFIED]** |
| Neural Graffiti | github.com/babycommando/neuralgraffiti | Same-model hidden-state injection just before LM head for gradual persona drift; explicitly not precise word-level control | **[VERIFIED]** — working code, but self-described as unvalidated (no benchmarks); useful only as an injection-hook engineering pattern, not evidence of anything |

## B. MoE Merging, Upcycling & Routing Breakdown

| Source | ID / Venue | Claim | Status |
|---|---|---|---|
| MoErging survey | arXiv 2408.07057, Yadav et al. | General framework for combining independently-trained experts via dynamic routing; concrete instances mostly LoRA/adapter-granularity | **[VERIFIED]** |
| PHATGOOSE | arXiv 2402.05859, Muqeeth et al. | Per-expert sigmoid gate trained on hidden-state similarity, cooperatively during that expert's own fine-tuning; decentralized, post-hoc composable | **[VERIFIED]** |
| KALAVAI | arXiv 2603.22755 | Quantitative model for when post-hoc independent-specialist fusion actually pays off (training-duration crossover, oracle saturation) | **[VERIFIED]** |
| MC-SMoE | ICLR'24 Spotlight, github.com/UNITES-Lab/MC-SMoE | Merge-then-compress SMoE using routing statistics/activation frequency; within a single model | **[VERIFIED]** |
| "When Model Merging Breaks Routing" (HARC) | arXiv 2606.03391 | Hessian-Aware Router Calibration for routing breakdown after linear weight-averaging of MoE experts | **[UNVERIFIED]** — likely tangential to Phase 1 regardless, since router training/recalibration is explicitly out of scope |
| Sakana AI evolutionary model merging | arXiv 2403.13187 | CMA-ES search over parameter-space and data-flow-space merge configs across models; no combination hand-designed | **[VERIFIED]** |
| MoE-I² | — | Layer-wise genetic search over expert combinations to avoid brute-force combinatorial search | **[UNVERIFIED]** — found via search, not independently deep-checked |
| MoSECroT | arXiv 2401.04821 | Model stitching with static word embeddings for cross-lingual zero-shot transfer; competitive with weak baselines, not strong fine-tuned ones | **[UNVERIFIED]** |

## C. MoE Interpretability & Specialization

| Source | ID / Venue | Claim | Status |
|---|---|---|---|
| Original Mixtral technical report | Mistral AI, 2024 | Mixtral's 8 coarse experts show routing close to random/uniform, minimal domain specialization | **[UNVERIFIED]** — from general background knowledge, not re-fetched this session |
| OLMoE | arXiv 2409.02060 | 64 fine-grained experts show real, measurable domain specialization, unlike Mixtral | **[VERIFIED]** |
| "The Expert Strikes Back" | arXiv 2604.02178, Herbst, Lee, Wermter (ICML 2026) | MoE expert neurons are consistently less polysemantic than dense FFNs, gap widening with sparser routing; near-perfect F1 at k=1; **experts are neither broad domain specialists nor simple token processors — they are fine-grained task/linguistic specialists** (e.g., closing LaTeX brackets) | **[VERIFIED]** — read primary text directly; this is the source for the Section 1.3 expert-selection correction |

## D. Representational Alignment Mathematics

| Source | ID / Venue | Claim | Status |
|---|---|---|---|
| Relative Representations | arXiv 2209.15430, Moschella et al. (ICLR 2023, Oral) | Anchor-relative coordinates make different models' spaces zero-shot comparable without training a translator | **[VERIFIED]** |
| vec2vec | arXiv 2505.12540, Jha et al. | Unsupervised translation between different models' sentence-embedding spaces without paired data; up to 0.96 cosine similarity | **[VERIFIED]** |
| Søgaard, Ruder & Vulić | arXiv 1805.03620 (ACL 2018) | Unsupervised adversarial/cycle-consistency alignment presumes isomorphism; degrades to confident-wrong mappings when source/target are dissimilar; predictable in advance via a spectral similarity metric | **[VERIFIED]** |
| CKA reliability | arXiv 2210.16156, Davari et al. (ICLR 2023) | CKA is sensitive to outliers and can be manipulated without changing actual model function; use a metric battery, not CKA alone | **[VERIFIED]** |
| "Platonic Representations in the Human Brain" | arXiv 2605.20496 | Cross-subject fMRI alignment via **unsupervised orthogonal rotation** works well; multi-way synchronization improves it further | **[VERIFIED–CAUTION]** — an earlier draft of this spec cited this paper to justify preferring ridge regression over orthogonal Procrustes; on inspection the paper's actual finding supports the opposite (orthogonal alignment working well). Do not use as justification for abandoning orthogonal methods. |
| "Multi-Way Representation Alignment" (GCPA) | arXiv 2602.06205 | For 3+ models, strict isometric (orthogonal) alignment can be suboptimal for retrieval vs. CCA-style agreement-maximizing methods; proposes a hybrid (GCPA) | **[VERIFIED]** — relevant to Phase 2 (N≥3 hub), not Phase 1's pairwise Rung 3 case |
| "Transferring Features Across Language Models With Model Stitching" | arXiv 2506.06609 / NeurIPS 2025 poster | Linear maps can transfer complex features across language models; LLMs use word-embedding-like vector arithmetic for abstract relations | **[UNVERIFIED]** |
| "Characterizing Linear Alignment Across Language Models" | arXiv 2603.18908 | Related/possibly overlapping with the above — needs disambiguation | **[UNVERIFIED]** |
| Atlas-Alignment | arXiv 2510.27413 | Orthogonal Procrustes achieves near-perfect feature retrieval (max MRR), outperforming unconstrained linear regression for same-dimension mapping | **[UNVERIFIED]** |
| "Transporting Task Vectors across Different Architectures without Training" | OpenReview, Rinaldi et al. | Relevant to the rectangular/cross-dimension Procrustes problem | **[UNVERIFIED]** |
| Sparse PCA / tensor-train / gene-expression component-analysis papers (arXiv 1602.03992, JMLR 24-0029, ORBi thesis, EEG-BCI orthogonal basis) | various | General Procrustes/Stiefel-manifold optimization math | **[TANGENTIAL]** — general math is standard and not in doubt; these specific citations look keyword-matched rather than substantively necessary |

## E. Diagnostic / Readout Instruments

| Source | ID / Venue | Claim | Status |
|---|---|---|---|
| Patchscopes | arXiv 2401.06102 (ICML 2024) | Patches a hidden representation into a target context, lets the model's own generation decode it; generalizes logit lens; includes cross-model patching | **[VERIFIED]** |
| Anthropic Global Workspace / "J-space" | anthropic.com/research/global-workspace; transformer-circuits.pub/2026/workspace | Small, densely-connected, vocabulary-anchored broadcast-hub representation found in Claude; causally load-bearing for multi-step reasoning; emerged without being designed | **[VERIFIED]** — intra-model finding; supports the hub *pattern* generally, not cross-model alignment specifically |
| Subtext (Jacobian Lens implementation) | github.com/ninjahawk/Subtext | Open-source, community-built implementation of the Jacobian lens running live on Qwen3.5-4B; validated to cosine similarity ≥0.99998 vs. Anthropic's reference | **[VERIFIED]** |
| Activation Patching / causal mediation analysis (general lineage) | Meng et al. and others | Foundational technique for isolating causal effect of a specific internal vector on downstream output | **[UNVERIFIED]** — well-established general technique, not independently re-checked this session |

## F. Comparative Baselines (does cross-model mixing help at all)

| Source | ID / Venue | Claim | Status |
|---|---|---|---|
| Mixture-of-Agents (original) | arXiv 2406.04692 | Proposer/aggregator layered LLM architecture; aggregator's win rate correlates with overlap with the best single proposer | **[VERIFIED]** |
| "Rethinking Mixture-of-Agents" (Self-MoA) | title confirmed, no arXiv ID captured | Repeated sampling from one strong model often beats mixing different LLMs via standard MoA (+6.6% AlpacaEval 2.0, +3.8% avg) | **[UNVERIFIED]** — found via secondary source (Semantic Scholar), not the primary paper directly |

## G. Confirmed Real But Not Relevant (included for completeness only)

| Source | ID / Venue | Why it's here | Status |
|---|---|---|---|
| Nemotron-TwoTower | arXiv 2606.26493 | Two clones of the same backbone, cross-attending for diffusion-decoding speedup on a single model — surfaced via keyword overlap ("cross-attention," "towers"), not cross-model relevance | **[VERIFIED–CAUTION]** — do not cite as related work |
| GraphMoE | arXiv 2501.07890 | Real, and closer than it first looks — literal message passing between expert nodes, but within a single homogeneously-trained MoE, not cross-model | **[VERIFIED]** — genuinely worth reading, just adjacent rather than a direct architectural ancestor |

---

**For the validating agent:** priority order should probably be (1) everything in section A and B tagged UNVERIFIED, since those are load-bearing for the spec's core mechanism claims, (2) the D-section math citations, since several of them support specific methodological choices (Rung 3's regression method) that are currently backed by general statistical reasoning rather than a solid citation, (3) the FLIMSY entry, to either find independent replication or rule it out entirely, (4) section G and the tangential math citations last, since they're explicitly low-priority.
