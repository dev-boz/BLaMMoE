# Adversarial Audit Report: BLaMMoE Toy-Scale Results

**Auditor:** Independent Adversarial Auditor  
**Date:** 2026-10-19  
**Scope:** Verification of claims, code paths, leakage risks, evaluation integrity, and baseline fairness across the BLaMMoE toy program (`dev-boz/BLaMMoE` branch `claude/new-session-f94vqn`).

---

## Executive Summary

The headline claims of BLaMMoE demonstrate genuinely creative cross-tokenizer architectures and an acute awareness of cross-tokenizer BPE boundary leakage. However, an adversarial examination of the codebase reveals critical flaws, evaluation confounds, and unacknowledged invalidations:

1. **Claim 4 (92.0% Learned Bridge) is invalidated by its reliance on dense expert evaluation**: The 92.0% learned bridge was trained and evaluated exclusively on dense expert outputs (`expert_dense`), which pushes the entire residual stream through the expert weights. Under routed-only expert outputs, the learned bridge was never run past 600 steps (where it remained at chance, 12.5%). By CLAUDE.md Non-negotiable Rule 8's own definition, this is a whole-model residual transmission artifact, not expert-level transmission.
2. **Linear transplant hook is mathematically defective**: `transplant_hook` subtracts unweighted expert outputs on tokens where the target expert was never routed, and over-subtracts on routed tokens due to router weights ($< 1.0$). The degradation in general LM loss previously attributed to "representational interference" is largely an artifact of corrupting the residual stream with inverted unrouted expert states.
3. **`FusedPair.forward` silently bypasses `safe_mask`**: Calling `pair(...)` directly ignores `self.safe_mask` and executes the leaky mask unless the wrapper function `run_pair()` is used.
4. **Stage 2 and depth routing claims were run on leaky models**: The Stage-2 fine-tuning experiments and the depth ablation tables (e.g. 38.6% at layer 0) were executed exclusively under the leaky mask (`safe_mask=False`), not the leak-proof mask.
5. **Claim 3 (Residual baseline fails one way at 10.3%) overstates single-seed non-convergence**: The residual baseline has equivalent parameter capacity ($< 2\%$ difference) and achieved 99.4% in the reverse direction. Claiming architectural inferiority from $n=1$ seed failing to cross a known sudden-jump optimization threshold is unsubstantiated.
6. **The causality audit script (`toy_fused_audit.py`) only audits Model A**: It omits Model B entirely from prefix-honest evaluation.

---

## Detailed Findings

### Finding 1: Learned Bridge 92% Claim Relies on Dense Expert Evaluation, Violating CLAUDE.md Rule 8
- **Severity**: **Critical** (invalidates Claim 4's learned bridge claim)
- **Where**: `scripts/toy_poc.py:121`, `scripts/toy_followup.py:114`, `docs/toy-poc-results.md:126-142`
- **What**: The headline 92.0% learned bridge result was trained and evaluated exclusively using dense expert outputs (`expert_dense`). Under native routed-only expert outputs, the learned bridge was never trained past 600 steps, where it scored 12.5% (chance).
- **Failure scenario**: An MoE expert is evaluated densely on all tokens (regardless of routing decisions), allowing residual stream features from unrouted tokens to flow through the bridge. The result is claimed as "expert-level" transmission, but collapses into whole-model residual transmission.
- **Evidence**: *Demonstrated from code and config inspection*. In `scripts/toy_poc.py:121`, `specialist_dense` is `Source(la, spec_a, "expert_dense")`. In `scripts/toy_followup.py:114`, the 3000-step long bridge uses `Source(la, spec, "expert_dense")`. `toy_followup.py` never ran the routed-only bridge at 3000 steps. In `toy_poc.py` at 600 steps, `specialist_routed` scored 12.5% (chance). CLAUDE.md Non-negotiable #8 explicitly states: *"Expert-level claims use routed-only expert outputs. An expert evaluated densely on every token reads the whole residual stream through its weights, so even a never-routed expert appears to transmit."* The 92.0% headline learned bridge claim directly violates this rule.
- **Fix / check**: Train `Bridge` for 3000 steps using `Source(la, spec_a, "expert_routed")` and measure whether B's answer accuracy exceeds chance (~10%).

---

### Finding 2: Linear Transplant Hook Inappropriately Excises Expert Outputs and Bypasses Router Weights
- **Severity**: **Major** (distorts general loss metrics and invalidates native MoE transplant mechanics)
- **Where**: `blammoe/toy/bridge.py:154-162`
- **What**: `transplant_hook` replaces expert output via `output + alpha * (translated - expert(inputs[0]))`. This subtracts unweighted `expert(inputs[0])` on tokens where the expert was never routed, and over-subtracts on routed tokens because router weights are $< 1.0$, while adding `translated` with constant weight $\alpha$, completely bypassing B's router.
- **Failure scenario**: In `MoE.forward`, expert outputs are added as `weights[rows, slot, None] * expert(flat[rows])`. On tokens where `expert_b` is unrouted, `output` had 0 contribution from `expert_b`, but `transplant_hook` subtracts `expert_b(x)`, injecting a negative vector into the residual stream. On routed tokens with router weight $w \approx 0.25$, it subtracts $1.0 \times \text{expert}_b(x)$, leaving a negative residual of $(w - 1.0) \times \text{expert}_b(x)$. The degradation in B's general LM loss (reported as "interference") is largely an artifact of corrupting the residual stream at unrouted positions.
- **Evidence**: *Demonstrated experimentally*. An in-process test showed that for unrouted tokens, the difference between hooked output and original output was exactly $-\text{expert}_0(x)$. On routed tokens with router weight $w \in [0.25, 0.28]$, the difference left an un-excised remainder of $(w - 1.0) \times \text{expert}_0(x) \approx -0.74 \times \text{expert}_0(x)$.
- **Fix / check**: Rewrite `transplant_hook` to take router indices and weights into account (replacing only when routed, and scaling `translated` by the router weight). Re-measure B's general LM loss with the clean hook.

---

### Finding 3: `FusedPair.forward` Silently Ignores `self.safe_mask` on Direct Invocation
- **Severity**: **Major** (latent leak vulnerability across the codebase)
- **Where**: `blammoe/toy/fused.py:160-161`
- **What**: `FusedPair.__init__` accepts `safe_mask: bool`, but `FusedPair.forward` never inspects `self.safe_mask`. If `write_ends_a` and `write_ends_b` are not passed (default `None`), `forward()` silently uses `ends_a` and `ends_b` (the leaky mask).
- **Failure scenario**: Any caller invoking `pair(ids_a, ids_b, ends_a, ends_b)` directly on a pair initialized with `safe_mask=True` will execute with the leaky cross-attention mask without warning or error. Safety is only enforced if the wrapper function `run_pair()` is explicitly called.
- **Evidence**: *Demonstrated from code inspection*. In `blammoe/toy/fused.py:141-165`, `self.safe_mask` is never referenced in `forward()`.
- **Fix / check**: In `FusedPair.forward`, add:
  ```python
  if self.safe_mask and (write_ends_a is None or write_ends_b is None):
      raise ValueError("FusedPair initialized with safe_mask=True requires write_ends_a and write_ends_b")
  ```

---

### Finding 4: Stage-2 Fine-Tuning Experiments Were Conducted Entirely Under the Leaky Mask
- **Severity**: **Major** (invalidates Stage-2 retention and diversity claims)
- **Where**: `configs/toy_poc.yaml:65-66` and `scripts/toy_fused_reward.py:37`
- **What**: In `configs/toy_poc.yaml`, `ab_experts_all_rel_plain` and `ab_experts_all_rel_reward` omit `safe_mask: true` and initialize from `ab_experts_all_rel` (the leaky model). Consequently, `toy_fused_reward.py` ran entirely with leaky cross-attention boundaries.
- **Failure scenario**: Although ordinary-text bits/char for Stage 2 was withdrawn, `docs/toy-fusion-results.md` still reports split-recall retention and diversity metrics for Stage 2 (plain: 52.4/35.8%, reward ~100x: 89.7/97.9%), and `docs/HANDOFF.md:120` recommends running stage-2 sweeps under `safe_mask`. No stage-2 model has ever been trained or evaluated under `safe_mask`.
- **Evidence**: *Demonstrated from reading configs and code*. `configs/toy_poc.yaml:65-66` lacks `safe_mask: true`, and `scripts/toy_fused_reward.py:37` defaults `safe_mask` to False.
- **Fix / check**: Define safe variants `ab_experts_all_rel_safe_plain` and `ab_experts_all_rel_safe_reward` with `safe_mask: true` initializing from `ab_experts_all_rel_safe`.

---

### Finding 5: Depth Ablation and Expert Choice Claims Were Evaluated on the Leaky Model
- **Severity**: **Major** (claims based on unverified, leaky run)
- **Where**: `docs/toy-fusion-results.md:100-138` and `docs/HANDOFF.md:133`
- **What**: The reported depth distribution (all 4 links: 97.3%/99.4%, only layer 3: 65.0%/58.4%, only layer 0: 38.6%) and attention share ratios (B e0 60%, A e7 60%) were computed on `ab_experts_all_rel` (the leaky variant), not `ab_experts_all_rel_safe` (95.1%/99.8%).
- **Failure scenario**: In `docs/toy-fusion-results.md:33`, the text claims that depth results and expert choice results are "unaffected" by the leak. Yet `docs/HANDOFF.md:133` admits: *"First-block-only exchange reached 38.6% one way (under the old mask; not rerun leak-proof)."* The depth and expert choice numbers were obtained from the leaky run and never re-evaluated under `safe_mask`.
- **Evidence**: *Demonstrated from reading documentation and commit history*. Commit `3618968` added safe-mask headline results for split recall and hide ablation, but never reran the depth link-dropout grid or expert attention mass script (`toy_fused_experts.py`) under `safe_mask`.
- **Fix / check**: Run `scripts/toy_fused_experts.py configs/toy_poc.yaml ab_experts_all_rel_safe` and link-dropout evals on `ab_experts_all_rel_safe`.

---

### Finding 6: Single-Seed Failure of Residual Baseline Claimed as Inherent Architectural Defect
- **Severity**: **Major** (overstated headline claim in Claim 3)
- **Where**: `docs/toy-fusion-results.md:238-242`, `AUDIT-BRIEF.md:Claim 3`
- **What**: Claim 3 asserts that residual-stream exchange fails one way (10.3%). However, the reverse direction succeeds at 99.4% with the same budget, and the failure in the A-needs-B direction is based on a single seed ($n=1$) that failed to cross the sharp sudden-jump optimization threshold within 3000 steps.
- **Failure scenario**: A single run fails to converge by step 3000 due to optimization stochasticity. This is presented as an architectural proof that "reading B's individual experts carries B's information into A, while reading B's whole residual stream does not", despite the reverse direction achieving 99.4% and the parameter budgets being virtually identical (311k vs 317k, $< 2\%$ difference).
- **Evidence**: *Demonstrated from documentation and parameter analysis*. `docs/HANDOFF.md:135` acknowledges: *"Expert-level > residual-level is one seed and one budget. A longer or reseeded residual run could still learn."* Both models have equivalent parameter budgets.
- **Fix / check**: Train 3-5 seeds of `ab_resid_all_rel_safe` for 5000+ steps to test whether the A-needs-B channel crosses the sudden jump.

---

### Finding 7: `scripts/toy_fused_audit.py` Completely Omits Model B From Prefix-Honest Evaluation
- **Severity**: **Minor** (gap in verification tooling)
- **Where**: `scripts/toy_fused_audit.py:49-100`
- **What**: Claim 5 asserts that `scripts/toy_fused_audit.py` proves the tokenizer leak is closed. However, the audit script only implements prefix-honest scoring for Model A (`honest_a_nll` and `rec["fused_honest"]` for A-needs-B). It never computes prefix-honest scores for Model B.
- **Failure scenario**: If Model B's tokenization or masking had a leak, `scripts/toy_fused_audit.py` would be incapable of detecting it because Model B's predictions are never compared against prefix-honest execution.
- **Evidence**: *Demonstrated from code inspection of `scripts/toy_fused_audit.py:49-100`*. (Our separate scratch test verified that Model B's split-recall scoring matches prefix-honest scoring due to single-char digit chunks, but the audit script itself lacks this check).
- **Fix / check**: Add `honest_b_nll` and Model B prefix-honest evaluation to `scripts/toy_fused_audit.py`.

---

### Finding 8: B-Needs-A Hide Ablation Leaves 29.1% Residual Accuracy Due to Unablated Early-Layer Channels
- **Severity**: **Minor / Note** (nuance in Claim 2)
- **Where**: `docs/toy-fusion-results.md:251`, `scripts/toy_fused_hide.py:58-61`
- **What**: Claim 2 states that hiding the single expert each translator chose collapses accuracy (99.8→29.1%). In the B-needs-A direction, 29.1% remains nearly 20 percentage points above the wrong-partner chance level (~10%) because B's translators also read A's expert 0 at link 0 (51% share) and link 1 (75% share), which were not hidden.
- **Failure scenario**: Single-expert ablation at link 3 leaves early-layer channels active. Claiming that the signal runs "almost entirely through the one expert each translator picked" oversimplifies the B-needs-A transfer.
- **Evidence**: *Demonstrated from reading `docs/toy-fusion-results.md:129-130` and `:251`*.
- **Fix / check**: Run joint ablation hiding both A expert 7 and A expert 0 to confirm complete collapse to chance (~10%).

---

### Finding 9: General Evaluation Set for Prose and Code Lacks a Disjoint File/Document Split
- **Severity**: **Minor**
- **Where**: `blammoe/toy/corpus.py:107-118`, `blammoe/toy/fused_train.py:26-43`, `blammoe/toy/fused_train.py:113-115`
- **What**: `general_eval_set` and `mixed_draw` sample 256-character windows from the exact same monolithic strings (`prose` and `code`), varying only the random seed. There is no train/eval file partition.
- **Failure scenario**: Evaluation windows can overlap with or share local context with training windows, slightly underestimating true out-of-distribution generalization loss.
- **Evidence**: *Demonstrated from code inspection of `blammoe/toy/corpus.py` and `fused_train.py`*.
- **Fix / check**: Partition `papers/` and stdlib python files into distinct train/val sets before sampling.

---

### Finding 10: `char_distance_buckets` Uses `closed_token_ends` Rather Than Actual Token Ends
- **Severity**: **Note**
- **Where**: `blammoe/toy/fused.py:168`
- **What**: When `rel_pos=True` and `safe_mask=True`, `char_distance_buckets` receives `wb = write_ends_b` (`closed_token_ends`). This computes $d = \text{reader\_ends} - \text{closed\_token\_ends}$ instead of $\text{reader\_ends} - \text{actual\_token\_ends}$, distorting distance buckets for tokens inside unclosed runs.
- **Failure scenario**: Relative position buckets reflect chunk closure boundaries rather than token boundaries, altering attention biases.
- **Evidence**: *Demonstrated from code inspection of `blammoe/toy/fused.py:160-168`*.
- **Fix / check**: Pass actual token ends (`ends_b`) to `char_distance_buckets` while using `write_ends_b` only for `read_mask`.

---

## Checked and Found Clean

The following components and mechanisms were rigorously examined and found to be mathematically and causally sound:

1. **Split-Recall Task Formulation and Ground-Truth Integrity (`blammoe/toy/corpus.py`)**:
   - `split_recall_doc` guarantees that each stored key-value pair is masked with `????` in exactly one model's view and visible in the other.
   - Keys are uniformly sampled from a space of $>1.2 \times 10^7$ combinations, and values are randomly drawn 4-digit strings ($10^4$ combinations).
   - Answer digits are queried in randomized order, preventing position-based sequential memorization.
   - Ground-truth chance level for digit prediction is verified at exactly 10.0%, matching empirical baseline performances (A alone 9.6%, B alone 9.2%, wrong partner 8.8%–11.9%).
2. **BPE Boundary Immunity on Split-Recall Digits (`blammoe/toy/tokenizers.py`)**:
   - `BPETokenizer` isolates digits using `pre_tokenizers.Digits(individual_digits=True)`.
   - BPE merges never cross digit boundaries, punctuation, or symbols.
   - Consequently, split-recall digits never experienced tokenizer boundary leaks even under the unmasked/plain baseline; the tokenizer leak was strictly confined to multi-character letter runs in natural language text.
3. **Causality of `safe_mask` under `run_pair` (`blammoe/toy/fused.py`, `blammoe/toy/audit.py`)**:
   - For all positions across full text and split recall, `run_pair` with `safe_mask=True` matches prefix-honest re-tokenization to machine precision ($< 10^{-7}$ logits difference on both models).
   - `closed_token_ends` correctly holds unclosed whitespace/letter chunks unreadable until the succeeding character is read by the receiver.
4. **MoE Hooking and Routed-Slot Isolation in `FusedPair` (`blammoe/toy/moe.py`, `blammoe/toy/fused.py`)**:
   - In `signal="experts"`, `blk.moe.last_slots` records only the top-$k$ routed expert activations. Unrouted experts are never computed or stored on the bus.
   - Expert ID embeddings (`eid_a`, `eid_b`) are strictly added per-slot, preventing cross-expert leakage on the bus.
   - Softmax masking on hidden experts cleanly zeros cross-attention weights to machine precision ($0.0$).
5. **Statistical Bootstrap Methodology (`blammoe/stats.py`)**:
   - `bootstrap_ci` and `paired_bootstrap_diff` resample documents as the primary unit of observation, correctly preserving intra-document answer correlations.
   - Paired differences correctly resample identical document indices across compared models.
