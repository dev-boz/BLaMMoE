# Adversarial audit of BLaMMoE toy results

Audited 2026-10-19. **The one-way “expert replacement” implementation is incorrect, the ordinary-text evaluation is not held out, and the audit can display matching scores for a demonstrably leaky model.** Expert-output attribution also remains confounded by the routing channel. I found no future-token leak in the explicitly safe `run_pair` paths tested here.

The historical accuracies cannot be independently reproduced: the supplied `results/` contains no trained checkpoints or raw predictions. Findings below distinguish demonstrated mechanisms from unmeasured effects on those accuracies. No finding demonstrates that the reported safe-fusion 95.1% / 99.8% accuracies are themselves false.

All Python runs used `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/boz/BLaMMoE/.venv/bin/python` from this directory; imports resolved to this copy. No GPU or outside-directory writes were used. Library and experiment code were left unchanged. Reproducible probes are in `audit_checks.py`, their output in `audit_checks.log` and `audit_artifacts/evidence.json`. The tiny checkpoints under `audit_artifacts/` are untrained audit fixtures, not recovered experimental results.

## 1. The linear transplant bypasses B's router and does not replace its native expert contribution

- **Severity:** major.
- **Where:** `blammoe/toy/bridge.py:154`, `blammoe/toy/bridge.py:160`, `blammoe/toy/moe.py:65`, `blammoe/toy/moe.py:72`; claimed replacement at `docs/toy-poc-results.md:48`.
- **What:** The hook adds `alpha * (translated - expert(x))` to the entire MoE output at every token. A native replacement requires multiplying that correction by B's selected expert's routing probability and zeroing it where that expert is not selected; neither occurs.
- **Failure scenario:** B routes a token exclusively to experts 0 and 1, while the hook “replaces” expert 3. A native replacement leaves the token unchanged; this implementation changes it, and on routed tokens it also applies the wrong scale.
- **Evidence:** **Demonstrated.** The probe forced top-2 routing to `[0, 1]`, hooked expert 3, and observed a maximum MoE-output change of **1.0000298**, versus exactly zero for a proper replacement. Both POC scripts use this hook. The existing own-output no-op test cannot catch the defect because its correction is zero under either formula. The numerical effect on 96.5% is unmeasured without the checkpoint.
- **Fix / check:** Apply `alpha * p_e * selected_e * (translated - native_e)` to the MoE sum, or replace the selected expert's actual routed outputs before weighting. Test both an unrouted token and a routed token with probability substantially below one, then rerun the POC and controls.

## 2. Hiding expert bus entries does not isolate information in expert output vectors

- **Severity:** major.
- **Where:** `blammoe/toy/fused.py:101`, `blammoe/toy/fused.py:187`, `blammoe/toy/fused.py:189`, `blammoe/toy/moe.py:72`; interpretation at `docs/toy-fusion-results.md:243`.
- **What:** Each bus entry carries a learned expert-ID embedding as well as its expert vector. Hiding removes both the vector and a content-dependent routing/availability channel; it also leaves that expert's local computation intact, allowing its effects to reach later experts through the sender's residual stream.
- **Failure scenario:** The sender's router encodes information through expert identities while expert vectors carry no information. Removing a chosen expert's entries destroys this channel and changes attention normalization, producing an apparent expert-output ablation effect.
- **Evidence:** **Demonstrated channel; suspected attribution confound in the trained models.** A constructed `Link` decoded **10/10 labels with every expert output vector identically zero**, using only expert IDs. The implementation does correctly block the requested entries, but the reported accuracy drops do not distinguish vector content from IDs, availability, or downstream redistribution. No checkpoint was available to measure their contributions.
- **Fix / check:** Evaluate content-only ablation: replace vectors with constants while preserving IDs, slots and masks. Separately remove/scramble IDs while preserving vectors. Compare with the current entry-removal ablation before claiming the specialist's output computation carries nearly all the information.

## 3. POC selection and final scoring reuse the same evaluation data

- **Severity:** major.
- **Where:** `scripts/toy_poc.py:44`, `scripts/toy_poc.py:63`, `scripts/toy_poc.py:79`, `scripts/toy_poc.py:100`; `scripts/toy_followup.py:55`, `scripts/toy_followup.py:65`, `scripts/toy_followup.py:99`.
- **What:** Expert selection uses routing on the evaluation examples, and the follow-up explicitly selects B's injection layer by the original evaluation accuracy. It then reports follow-up accuracy on the identical 96 samples generated with seed 2024.
- **Failure scenario:** A layer wins the initial depth sweep partly through evaluation-set noise. Selecting it and reporting its performance again on those same examples retains that favorable noise rather than measuring independent generalization.
- **Evidence:** **Demonstrated by code inspection.** Both scripts construct the same evaluation seed and size; the follow-up's `max(...)` selects directly from `poc_results.json` matched accuracy. The initial script also evaluates four injection strengths on this set. The size of any resulting optimism is unknown, and this does not establish that the large transfer effect disappears.
- **Fix / check:** Select experts, layer and strength on calibration/validation samples, freeze those choices, and score once on a fresh recall seed. Keep a separate final test set across subsequent follow-ups.

## 4. Ordinary-text “held-out” evaluation overlaps training content

- **Severity:** major.
- **Where:** `blammoe/toy/corpus.py:107`, `blammoe/toy/setup.py:15`, `blammoe/toy/fused_train.py:30`, `blammoe/toy/fused_train.py:113`; `scripts/toy_fused.py:53`, `scripts/toy_fused.py:60`.
- **What:** Training and evaluation sample windows from the same full prose/code strings. Different random seeds separate draws, not documents or text regions.
- **Failure scenario:** An evaluation window contains passages previously used to optimize the base models or links. Its bits/char is reported as held-out performance despite that exposure.
- **Evidence:** **Demonstrated.** Replaying the configured 3,000 × 16 fusion training draws, without running models, found training overlap in **37/48** evaluation windows: **51.66%** of evaluation character positions had been seen, and **9/48** windows had all their characters covered by training draws. These counts exclude additional exposure during pretraining and tokenizer fitting. This finding concerns ordinary text; the replay found **zero exact A-view split-recall window duplicates** among 24,043 distinct training windows and the 96 evaluation windows.
- **Fix / check:** Partition prose/code by source document or file before tokenizer fitting, pretraining, calibration and fusion training. Regenerate the ordinary-text metrics on untouched files. Changing seeds alone cannot clear this issue.

## 5. The causality audit can give false reassurance from matching aggregate scores

- **Severity:** major.
- **Where:** `scripts/toy_fused_audit.py:39`, `scripts/toy_fused_audit.py:41`, `scripts/toy_fused_audit.py:65`, `scripts/toy_fused_audit.py:94`, `scripts/toy_fused_audit.py:98`; `tests/test_audit.py:87`.
- **What:** The CLI reports mean gold-token NLL and recall accuracy without asserting per-position agreement or failing on differences. It audits only A, only the first 48/96 recall windows, and no hide ablations; its usual general-text sample is 16/48 windows.
- **Failure scenario:** A leak changes individual distributions while mean NLL changes little, or leaves top-1 accuracy unchanged. Four-decimal output appears equal and the command succeeds, despite a real causal discrepancy.
- **Evidence:** **Demonstrated with the real CLI.** On an explicitly unsafe tiny checkpoint, it exited **0**, printed **6.6665 = 6.6665** full/honest bits/char, and reported identical recall accuracies. On that same general-text document, independent per-position comparison found maximum logit difference **0.0151872** and gold-NLL difference **0.00437689 nats**. The JSON means were actually 6.66651249 and 6.66646051. This counterexample shows the acceptance criterion is weak; it does not show the safe historical checkpoints leak.
- **Fix / check:** Assert per-position full-distribution agreement with an explicit numerical tolerance and fail the CLI on violations. Cover both directions, every evaluation window, and normal/hide/residual paths. For B, preserve B's own input-token prefix while truncating its partner's character context; retokenizing B's own prefix can change the context being compared. Prefix truncation itself is a valid independent check because the unseen suffix is absent.

## 6. `safe_mask=True` does not protect direct `forward` calls

- **Severity:** minor.
- **Where:** `blammoe/toy/fused.py:122`, `blammoe/toy/fused.py:160`, `blammoe/toy/fused.py:211`; unsafe variants remain at `configs/toy_poc.yaml:47`.
- **What:** `safe_mask` defaults to false, and `forward` silently uses ordinary token ends whenever explicit writer ends are absent, even on an object constructed with `safe_mask=True`. Protection depends on using the wrapper correctly.
- **Failure scenario:** A caller constructs a safe pair and invokes `pair(ids_a, ids_b, ends_a, ends_b)`, assuming the object's flag enforces safety. This activates the old mask.
- **Evidence:** **Demonstrated.** The same safe-flagged tiny pair produced maximum logit difference **0.140263** between direct forward and `run_pair`. Inspection found the named safe variants' training, scoring, analysis and hide paths correctly use `run_pair`; this is an API hazard, not evidence those headline paths bypassed safety.
- **Fix / check:** Reject missing writer ends when `safe_mask=True`, and make safe operation the default for new variants. Retain explicitly named unsafe configurations only for regression/control work.

## 7. Expert attention enrichment uses an inappropriate availability baseline

- **Severity:** minor.
- **Where:** `blammoe/toy/fused.py:337`, `blammoe/toy/fused.py:343`; reported multipliers at `docs/toy-fusion-results.md:249`.
- **What:** The enrichment denominator counts every bus position, including future and padded positions, rather than entries readable at the selected reader positions. Causal availability alone can therefore look like expert preference.
- **Failure scenario:** Expert 0 supplies the only readable real entry and expert 1 occurs only in the future. Uniform attention over the readable null and expert-0 entries is reported as expert-0 enrichment.
- **Evidence:** **Demonstrated.** With attention `[0.5, 0.5, 0]`, the function reports expert-0 share 1, presence 0.5 and **2× enrichment**, despite no preference among available entries. Raw attention shares still correctly describe the actual attention; the “×6.2/×5.5” interpretation needs recalculation.
- **Fix / check:** Compute the expected expert mass under uniform attention over each selected reader's legal, nonpadding entries, using the same null-slot convention as the observed statistic.

## 8. Bootstrap code treats undefined zero-answer resamples as zero accuracy

- **Severity:** minor.
- **Where:** `blammoe/stats.py:24`, `blammoe/stats.py:37`.
- **What:** Clamping a resample's total answer count to one converts `0/0` into accuracy zero. Split-recall windows can legitimately contain no answers for one direction, so this is a reachable edge case for small subsets.
- **Failure scenario:** Documents have counts `[[0, 0], [1, 1]]`. Every defined resampled accuracy is one, but a resample consisting only of the empty document contributes zero and artificially lowers the confidence interval.
- **Evidence:** **Demonstrated.** `bootstrap_ci` on those rows returned accuracy **1.0**, CI **[0.0, 1.0]**. The paired implementation has the same operation. This is not an explanation of the historical headline numbers, which did not report these CIs.
- **Fix / check:** Explicitly handle undefined resamples, report their frequency, and reject an entirely zero-answer dataset. Preserve document-level resampling; avoid silently inventing zero-accuracy observations.

## 9. Headline accuracy is teacher-forced digit accuracy, not complete-answer success

- **Severity:** note.
- **Where:** `blammoe/toy/fused.py:233`, `blammoe/toy/bridge.py:111`, `blammoe/toy/corpus.py:175`; three-bit wording at `docs/toy-fusion-results.md:262`.
- **What:** Metrics pool individual digit predictions, with earlier gold answer digits present in the input. They do not measure freely generated four-digit answers; “an answer is ~3 bits” describes one digit, while a complete uniformly random four-digit value contains about 13.29 bits.
- **Failure scenario:** A reader interprets 95.1% as four-digit question success. Predictions with one wrong digit in each of four answers out of twenty score 95% per digit but only 80% exact answers, even before free-generation effects.
- **Evidence:** **Demonstrated by metric and corpus inspection.** The configured 96 evaluation windows contain **1,092 A-needs-B digits** and **1,240 B-needs-A digits**, representing 273 and 310 four-digit answers. No complete-answer or generation scorer is used for the claims. The POC write-up does explicitly label its primary metric as answer-digit accuracy.
- **Fix / check:** Label every headline “teacher-forced digit accuracy”; additionally report four-digit exact match and free-generation results before making complete-question success claims.

## 10. The 92% learned bridge is a dense-source experiment

- **Severity:** note.
- **Where:** `scripts/toy_followup.py:114`, `scripts/toy_followup.py:129`, `blammoe/toy/bridge.py:66`; disclosure at `docs/toy-poc-results.md:125`.
- **What:** The long bridge run uses `expert_dense`, evaluating the chosen A expert on all positions. Its 92% is not evidence for native routed-only specialist transmission, although the write-up discloses that source mode.
- **Failure scenario:** The 96.5% routed-source result and 92% bridge result are summarized together as two native expert-level demonstrations. The second channel actually receives transformed residual information even on positions where the chosen expert never fires.
- **Evidence:** **Demonstrated by code inspection.** The long-run source is explicitly `Source(..., "expert_dense")`. The earlier routed-only bridge comparison runs for only the shorter budget; no equal-budget 3,000-step routed-only result is supplied. The transplant-hook defect in finding 1 does not apply to this separate bridge implementation.
- **Fix / check:** Run the 3,000-step bridge with `expert_routed`, plus equally trained unrelated-routed and residual controls. Keep the dense-source label until that comparison exists.

## 11. The residual comparison matches optimization settings but not the complete channel budget

- **Severity:** note.
- **Where:** `configs/toy_poc.yaml:62`, `blammoe/toy/fused.py:132`, `blammoe/toy/fused.py:164`, `scripts/toy_fused.py:56`; inference at `docs/toy-fusion-results.md:238`.
- **What:** The expert bus has two tagged entries per sender token, whereas the residual bus has one untagged entry. The comparison also uses one seed; the documented 10.3% is one training outcome, not a demonstrated general inability of residual exchange.
- **Failure scenario:** The residual run learns more slowly or encounters an unfavorable initialization, while routing labels or additional entries help the expert run. The gap is attributed entirely to expert output content.
- **Evidence:** **Demonstrated budget differences; suspected explanation of the gap.** Counts are **317,192 expert-link parameters versus 311,048 residual-link parameters**, a 6,144-parameter difference. Steps, batch size, learning rate, loss weighting, bus width and link depths do match. The parameter difference alone does not explain an 84.8-point gap, and the documentation already acknowledges the single-seed limitation.
- **Fix / check:** Compare multiple seeds and a modest residual learning-rate/step sweep on validation data, then score on a fresh test set. Include the ID-only/content-only controls from finding 2 and a residual control with matched entry count before assigning the advantage specifically to expert computations.

## Checked and found clean

- **Existing tests:** Ran the full repository suite on CPU: **64 passed in 5.91 seconds**. Passing tests did not detect the defects above; the transplant no-op test in particular exercises an insufficient invariant.
- **Closed-token stability:** Independently retokenized every prefix of 53 texts, including letter/whitespace runs, digits, punctuation, newlines and a literal special token. All **156,017** checked finalized token IDs and offsets matched their full-text counterparts. The extra character required to close runs is conservative and appropriate for this tokenizer's pretokenizer.
- **Safe fusion causality:** With live nonzero gates and nonzero relative-position biases, compared full distributions against independently truncated contexts for expert, summed-MoE and residual signals, including expert hiding. Across eight general/split/ablation cases, all A positions and all B input-token positions agreed within **8.95×10⁻⁸** maximum absolute logit difference. B's checks preserved its own token prefix and removed later partner characters. These are tiny two-layer tests, not certification of unavailable trained checkpoints.
- **Headline safe-path wiring:** The two named safe variants explicitly enable `safe_mask`; training, split scoring, ordinary-text scoring, hide ablation and analysis call `run_pair`. Closed writer ends propagate through both directions and relative-position masks. No bypass was found in those paths.
- **Causal computation and routed bus:** Self-attention is causal, normalization is per token, and expert selection is per token. The fused expert bus records only actual routed expert evaluations before router weighting; it does not densely evaluate every expert. The sender states for both directions are taken before that link's exchange, avoiding an intra-link ordering shortcut.
- **Hide masking:** Requested expert entries are blocked at every active link, including their relative-position attention bias. Existing tests verify exactly zero attention on hidden entries. This validates the literal bus-entry-removal operation, subject to the interpretation limits in finding 2.
- **Recall construction and chance:** Keys are unique within a document and sampled without replacement for queries. Values are independently sampled four-digit strings; store masking preserves length, and digit tokens are singletons in both tokenizers. I found no systematic own-view answer-format, position, repeated-query or segmentation shortcut. Ten percent is the correct population uninformed chance per digit; chance for a complete four-digit answer is 0.01%. The current evaluation digit histograms have largest fixed-digit accuracies 10.99% for A and 10.56% for B, so a wrong-partner accuracy near 10% is a reasonable negative control, though not a substitute for the solo baseline or uncertainty estimates.
- **Recall separation and metrics:** Fusion training uses seed 4242 and split evaluation seed 2025; POC fitting and bridge training use seeds distinct from its recall evaluation. The fusion replay found no exact split-window duplication. Answer indices correctly score the preceding input position, ignored padding is excluded, accuracy pools correct/total counts, and bits/char divides summed target NLL by the characters covered by those targets. BPE and character models omit different first-token spans, a small conditioning difference rather than an off-by-one leak.
- **Bootstrap resampling unit:** Outside the zero-total edge case above, bootstrap functions resample whole sample rows, pool correct/total counts, and reuse identical row indices for paired differences. A row is a 256-character window that may contain several generated recall documents, so this groups correlated answers conservatively rather than treating digits as independent. Document bootstrap uncertainty would still not measure variation across model/training seeds.
