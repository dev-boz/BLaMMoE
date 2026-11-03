> **Historical.** This is the original idea doc, from before the Phase 1 spec ([`blammoe-spec-and-bibliography-v4.md`](../blammoe-spec-and-bibliography-v4.md)). I found it again on 2026-11-03 and added it as it was. The toy programme since then has moved past the narrowed claim in section 1, back towards the original framing ("let MoE experts talk to each other"). See the [README](../README.md) and [`docs/HANDOFF.md`](HANDOFF.md) for where things stand now.

# BLaMMoE: Idea + Prior Art Summary

Self-contained summary for outside review. Covers the idea, what's already been found in the literature, and the open technical risks as currently understood.

## 1. The idea

**BLaMMoE** (Bridged by Latent Attention Model Mixture of Experts): take independently-trained, off-the-shelf MoE checkpoints (e.g. a Qwen-MoE and a separate MoE from another lab), and let their *native internal experts* — not the whole models — communicate through a shared latent space, forming one large addressable graph (e.g. 128 experts from one model + 64 from another = 192-node pool) instead of treating each checkpoint as a single black-box unit.

Original framing was "let MoE experts talk to each other" via a learned meta-router (the "LaM," Latent Attention Model) operating over a shared latent bus. After review, the scope has narrowed to a single, sharper, testable claim:

> **The Micro-Alignment Problem: can highly idiosyncratic, shallow MoE experts from different independently-trained checkpoints be aligned into a shared cross-model latent space without catastrophic interference?**

Router training, whether cross-model mixing actually improves task performance, and which specific model combinations are "best" are explicitly out of scope for now. The goal is to first prove translation/alignment is possible at all.

**Current proposed method for finding the alignment:** feed the same calibration inputs through both models and use co-activation — which experts (or, better, the full pre-softmax router score distribution) fire on the same input — as the training signal for per-model Tx/Rx adapters into a shared latent bus.

## 2. Prior art, closest to furthest

- **Symphony-MoE** (arXiv 2509.18542) — closest match at literal FFN-expert granularity. Training-free: aligns FFN experts from disparate pretrained models via activation-pattern-based neuron permutation against an anchor model, using a calibration set. Key finding: naive/unaligned merging of disparate models causes catastrophic interference — the models occupy "dissonant" regions of parameter space, and the router can't dispatch coherently without an explicit alignment step first.
- **Mixture of Thoughts (MoT)**, Fein-Ashley et al., Sept 2025 — closest match at whole-model granularity. A router selects top-K frozen heterogeneous LLM "experts," learned interaction layers project each one's hidden states into a shared latent space, and a designated primary expert cross-attends over the others. First method to support cross-architecture hidden-state communication without pairwise translators. Single-pass inference (a real advantage over text-level mixing); the cost is that the router and interaction layers have to be trained.
- **BTS** (early 2025) — frozen expert models connected by small trained "stitch layers." Built explicitly because plain Branch-Train-Merge has no learned connection between experts, and Branch-Train-MiX (BTX) sacrifices modularity to fold experts into one MoE.
- **"Beyond Tokens" survey** (arXiv 2606.05711, June 2026) — taxonomizes the whole design space along three axes: what's communicated (hidden state / KV-cache / etc.), which experts participate, how they fuse (cross-attention, KV-splice, etc.). ~18 surveyed methods occupy only ~17 of 45 possible design combinations — the space is mapped but not saturated.
- **MoErging survey** (arXiv 2408.07057) — general framework for combining independently-trained experts via dynamic routing. Concrete instances mostly work at LoRA/adapter granularity, coarser than native MoE experts.
- **PHATGOOSE** (arXiv 2402.05859) — a per-expert sigmoid gate trained on hidden-state similarity, decentralized and post-hoc composable. Important caveat: the gate is trained *cooperatively*, alongside each expert's own fine-tuning. Fully off-the-shelf checkpoints (no cooperation at training time) is a strictly harder version of this problem.
- **KALAVAI** (arXiv 2603.22755, ~March 2026) — quantitative model for *when* post-hoc fusion of independently-trained specialists actually pays off (training-duration crossover, whether routing is learnable at all, oracle saturation). Directly relevant as a way to predict, before investing compute, whether a given pair/combination is worth pursuing.
- **GraphMoE** (arXiv 2501.07890) — literal message passing between expert nodes, but within a single, homogeneously-trained MoE. Closest existing thing to "let experts talk," just not cross-model.
- **Rethinking Mixture-of-Agents ("Self-MoA")** — found that repeated sampling from one strong model often beats mixing *different* LLMs via standard MoA (+6.6% AlpacaEval 2.0, +3.8% average). Relevant control if/when the project moves from "can we align" to "does this help."
- **Sakana AI, Evolutionary Optimization of Model Merging Recipes** (arXiv 2403.13187) — closest precedent for "don't presuppose which combination is best, search for it": CMA-ES over parameter-space and data-flow-space merge configurations, producing combinations no one hand-designed (e.g. EvoLLM-JP beating individual 70B models on Japanese math reasoning).
- **vec2vec** (arXiv 2505.12540) — unsupervised translation between two *different* models' sentence-embedding spaces, no paired data, no access to original encoders — achieves up to 0.96 cosine similarity and near-perfect nearest-neighbor matching. Built on the Platonic Representation Hypothesis (representations of well-trained models converge to shared structure regardless of architecture). Strongest positive evidence found for the core premise — but demonstrated on pooled sentence embeddings, not token-level intermediate hidden states inside an autoregressive expert, which are more entangled with position and local context.
- **Model-stitching literature (general)** — classic finding is that stitchability is graded: works well between similar architectures/training setups, degrades as source and target models diverge.
- **Søgaard, Ruder & Vulić, "On the Limitations of Unsupervised Bilingual Dictionary Induction"** (ACL 2018, arXiv 1805.03620) — same family of method as vec2vec (adversarial + cycle-consistency alignment), applied to word embeddings across languages. Found these methods *presume the two spaces are isomorphic*, and degrade — not gracefully, but by converging to self-consistent-looking but wrong mappings — when the underlying spaces are dissimilar (different languages, different corpus domains, different embedding algorithms). Alignment success correlated ~0.89 with a spectral/eigenvector similarity metric computed *before* training — i.e., failure is predictable in advance, not just an occasional surprise.
- **Neural Graffiti** (github.com/babycommando/neuralgraffiti) — hooks into a single model's own hidden states between the transformer body and the LM head, injecting an evolving memory vector, no retraining. Solves a different problem (slow same-model persona drift, explicitly *not* precise word-level control) but is a working existence proof that late-stage residual-stream injection doesn't just produce garbage, and its "measure how much the injection is affecting output" instrumentation is a reusable pattern for building ablation controls. Unvalidated hobby project — no benchmarks — useful for engineering pattern only.

## 3. Corrections made to an earlier draft of this idea

- Mischaracterized MoT as "model-level, post-generation fusion" — it's actually latent, pre-generation, single-pass hidden-state cross-attention, which is BLaMMoE's own core mechanism. This manufactured a false gap against the closest prior art.
- Cited Nemotron-TwoTower (arXiv 2606.26493) as related work — it's real, but it's two clones of the *same* backbone speeding up diffusion decoding, unrelated to cross-model expert composition. Pulled in on keyword overlap, not real relevance.
- GraphMoE citation checked out and was under-credited relative to how close it actually is.

## 4. Open technical risks

- **Granularity risk.** A whole model's residual stream has to represent everything it's ever asked to do; an individual MoE expert only has to represent whatever narrow slice the router sends it. Expert-level representations are plausibly more idiosyncratic and less "universal" than whole-model representations — meaning expert-level alignment may be harder than the whole-model alignment MoT already does, not easier.
- **Specialization may not be semantic to begin with.** The original Mixtral technical report found no clean topic-based expert specialization — routing correlated more with shallow/syntactic features (punctuation, indentation) than domain. If that generalizes, there's no guaranteed stable correspondence to align *toward* between an expert in one model and any expert in another — the partition boundaries may be training-run-specific artifacts as much as portable concepts.
- **Alignment isn't free even with a shared bus.** Symphony-MoE needed calibration data plus activation-pattern-based permutation specifically because naive stitching caused catastrophic interference. A Tx/Rx architecture is necessary but not sufficient.
- **Unsupervised training can produce a confident wrong answer, not just fail to converge.** Per Søgaard et al., adversarial/cycle-consistency alignment presumes isomorphism between the two spaces and can converge to a self-consistent-looking mapping that doesn't correspond to anything real when that assumption doesn't hold. This is checkable in advance via a similarity metric (CKA or spectral/eigenvector similarity) before committing to full training.
- **Co-activation signal is low-bandwidth compared to what actually worked elsewhere.** vec2vec's success came from comparing full continuous embeddings; "which expert won" is one categorical fact (~log2(N) bits) per token, much thinner than a dense vector comparison. The full pre-softmax router score distribution (not the hard argmax) is a closer analogue to what vec2vec actually used, and preserves near-miss information a hard selection discards.
- **Calibration input choice matters.** A single short prompt (e.g. "Hi") mostly exercises shallow/syntactic features rather than establishing real correspondence — needs a large, varied calibration set (topics, lengths, registers, code vs. prose), the same way Symphony-MoE uses a full calibration dataset.
- **Fixed text vs. free generation.** For co-activation comparison to mean anything, the same static token sequence needs to be forced through both models (as in the Wikipedia-passage calibration plan) — if instead a shared prompt is used as a generation seed, the two models diverge into different completions almost immediately and stop being comparable after the first token.
- **Tokenizer mismatch.** Different models tokenize differently, breaking clean per-token-position matching. Needs pooling by character span, not token index.
- **Load-balancing noise.** Each model's router was shaped by an auxiliary loss pushing toward even expert utilization during training — a training-run-specific artifact layered on top of whatever real content-based routing exists, and not fully separable from it.

## 5. Suggested minimal test (Phase 1)

Isolate "can we build a shared language" from "is it useful":

1. Two small MoE checkpoints, frozen.
2. One Tx/Rx adapter pair *per model* (not per expert) into a shared bus — valid because all experts inside one model already share that model's hidden space.
3. Calibrate on a large, varied, fixed-text dataset (not single short prompts), pooled by character span.
4. Before training a translator: compute a cheap representational-similarity metric (CKA or eigenvector similarity) between the two models' representations as a predictor of whether alignment is even likely to succeed.
5. Train Tx/Rx via round-trip / cycle-consistency, comparing full router score distributions rather than hard expert selection where possible.
6. Round-trip coherence test with proper controls: native-no-injection baseline, matched-norm-random-vector injection, and raw untranslated vector injection — not just the treatment condition — to separate "translation is doing something specific" from "the target model would have produced that output anyway" or "any vector of the right scale has this effect."

## 6. Appendix: broader context (parked, not current scope)

- Whether cross-model mixing helps at all is a separate, harder empirical question from alignment — Self-MoA is the right control once/if that question is revisited.
- If the project becomes "search over many models" rather than a fixed pair, Sakana's evolutionary merging and MoE-I²'s layer-wise genetic search are the relevant precedents for keeping that search tractable; KALAVAI is a plausible cheap pre-filter for which combinations are worth trying.
- Router training for a live system, once alignment is validated, most plausibly follows PHATGOOSE's approach adapted to the post-hoc (non-cooperative) setting.
