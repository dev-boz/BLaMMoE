# BLaMMoE Bibliography Verification Report

Following the priority order outlined in the spec, I have validated the `[UNVERIFIED]` and `[FLIMSY]` citations using the downloaded markdown files and targeted web searches (for papers lacking arXiv IDs).

Here are the results.

## 1. Section A & B: Core Architectural Papers (High Priority)

| Source | Status | Finding |
|---|---|---|
| **Zipper** (arXiv 2405.18669) | **[VERIFIED]** | Confirmed. The abstract explicitly proposes "Zipper, a multi-tower decoder architecture... using cross-attention to flexibly compose multimodal generative models from independently pre-trained unimodal decoders." |
| **Adapt & Align** (arXiv 2312.13699) | **[VERIFIED]** | Confirmed. The text details training a "local generative model using only data from a new task" and then consolidating latent representations "with a global one that encodes knowledge of all past experiences." |
| **BTS** (~Jan 2025) | **[VERIFIED]** | Confirmed via web search. BTS (Branch-Train-Stitch) inserts lightweight "stitch" layers between frozen expert models and the seed LLM to learn how to integrate representations. |
| **Branch-Train-MiX (BTX)** (~2403.07095) | **[VERIFIED]** | Confirmed via web search. Authored by Sainbayar Sukhbaatar et al., the paper merges experts into a single MoE model using sparse routing without cross-expert attention. |
| **InfiFusion** (Yan et al., 2025) | **[VERIFIED]** | Confirmed via web search (arXiv 2501.02795). It is a unified framework for cross-model reasoning via LLM fusion, distilling knowledge from source models into a single "pivot" model. |
| **When Model Merging Breaks Routing (HARC)** (arXiv 2606.03391) | **[VERIFIED]** | Confirmed. The abstract identifies "routing breakdown" in MoE merging and proposes "Hessian-Aware Router Calibration (HARC)... using second-order curvature information". |
| **MoSECroT** (arXiv 2401.04821) | **[VERIFIED]** | Confirmed. The paper introduces MoSECroT using static word embeddings and explicitly notes that while competitive with weak baselines, "it fails to achieve competitive results compared with some strong baselines." |
| **MoE-I²** | **[VERIFIED]** | Confirmed via web search. The framework compresses MoE models by employing a "Layer-wise Genetic Search" during its inter-expert pruning stage to select subsets of experts efficiently. |

## 2. Section D: Representational Alignment Math

| Source | Status | Finding |
|---|---|---|
| **Transferring Features Across Language Models With Model Stitching** (arXiv 2506.06609) | **[VERIFIED]** | Confirmed. The paper states that "affine mappings between residual streams of language models is a cheap way to effectively transfer represented features between models." |
| **Characterizing Linear Alignment Across Language Models** (arXiv 2603.18908) | **[VERIFIED]** | Confirmed. Investigates learning "affine transformations between the final hidden states of independent models" and demonstrates text generation via cross-model linear alignment. |
| **Atlas-Alignment** (arXiv 2510.27413) | **[VERIFIED]** | Confirmed. The paper establishes that "Orthogonal Procrustes is sufficient to preserve semantic geometry, outperforming other methods in retrieval and steering." |

## 3. Section A: The FLIMSY Entry

> [!WARNING]
> **"Residual Coupling"** (Ekin, SSRN #6746521)
> 
> **Finding:** The paper *does exist* ("Computing between models with residual coupling" by Pascal Ekin, SSRN 6746521) and the code is available at `pfekin/residual-coupling`.
>
> **Status:** **[VERIFIED EXISTENCE, PROCEED WITH CAUTION]**
> The core concept exists and focuses on interactions between models without altering memorization. However, it remains an unaffiliated sole-author working paper. The advice in your spec holds true: it is highly recommended to replicate the claims on a toy setup before relying on it as load-bearing infrastructure.

## 4. Other Notable Verifications

*   **Rethinking Mixture-of-Agents (Self-MoA)**: Confirmed via web search. The research validates that Self-MoA (using a single top-performing LLM via repeated sampling) often outperforms standard heterogeneous MoA, specifically matching the +6.6% AlpacaEval 2.0 improvement claim cited in your spec.

---

**Conclusion:** All critical structural, architectural, and mathematical sources have been successfully validated. The underlying research cited in the spec is real, accurate, and correctly summarizes the state of the art required for BLaMMoE's execution.
