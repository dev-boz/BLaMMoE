1. Verdict on the interpretation

The asymmetry is real in these runs. The explanation is not established.

“Under this training recipe, the expert bus transfers substantially more code competence into prose-trained models than prose competence into code-trained models” is supported. “It is a direction effect, independent of tokenizer, expert spread and reader architecture” is too strong.

I used CPU only, with at most eight PyTorch threads. The repository was left unchanged. All 112 tests passed. Below, RUN means I executed a check; CODE means source inspection; OPINION means interpretation or recommendation.

What survives scrutiny:

- RUN: I reproduced the main char-pair scores from all six e-unit translators. A’s prose improvement through experts remains approximately 0.017–0.019 bits/char; A′’s code improvement is 0.084–0.201.
- RUN: Direct paired comparisons of A’s prose score with the prose partner versus the code control give differences of +0.00084, −0.00191 and +0.00252 bits/char. All three confidence intervals include zero. There is no demonstrated prose-specific advantage here.
- RUN: The pattern survives another set of 48 windows per domain for e0. This checks window sensitivity, not generalization to another corpus.
- RUN: The residual-stream gain survives independent causality checks. I found no evidence that lookahead explains it.

What does not follow:

Removing tokenizer mismatch and reducing the number of experts did not remove the failure. That establishes that neither is necessary for failure. It does not establish that neither contributes to the original A–B failure. Eight experts also do not guarantee that prose-relevant information is concentrated in fewer useful outputs.

A and A′ have the same architecture but different trained representations. Therefore, “A’s architecture is not inherently incapable of receiving information” is reasonable; “the reader is ruled out” is not.

Most importantly, these are three pretrained-model seeds but only one translator initialization and training-data order. The code-heavy model always occupies the first slot. That is an untested optimization confound.

The residual hand-over interpretation is substantially better supported than the direction-effect explanation. In e0:

- A prose: solo 2.208; all links 1.923; final link alone 1.920; final link removed 2.225.
- A′ code: solo 2.111; all links 1.642; final link alone 1.662.

The final link sits after the receiver’s final expert layer. Most of this benefit therefore requires no subsequent computation by the receiver’s experts.

For “different because of the partner,” that can count as success. For complementary computation or expert-to-expert cooperation, it is a weak solution.


2. Bugs and methodological problems

A. “Toward” does not establish actual movement toward the partner

Location: scripts/toy_partner_shift.py:52, :59, :100.

The implementation computes:

    toward = JS(wrong-partner output, partner solo)
             − JS(real-partner output, partner solo)

It omits the receiver’s solo distance to the partner. Consequently:

    toward = actual movement toward partner
             + movement away caused by wrong partner

RUN: A counterexample using the implemented JS calculation:

    receiver solo = partner solo = [0.9, 0.1]
    real-partner output           = [0.8, 0.2]
    wrong-partner output          = [0.1, 0.9]

“Toward” is +0.358, although the real partner moved the receiver away from the partner’s predictions. Dependence is also positive. Thus the proposed criteria can accept a damaging change.

RUN: This matters numerically in the actual residual results. For e0, A on prose:

    JS(solo, partner)  = 0.09885
    JS(fused, partner) = 0.02207
    JS(wrong, partner) = 0.49521

Reported toward is 0.47314. Actual attraction is 0.07678. Approximately 84% of the reported value comes from wrong-partner repulsion. The proportion is similar in e1 and e2.

This does not erase the transfer. Actual attraction remains substantial. It does invalidate reading the large “toward” number as the amount transferred.

Recommendation: report actual attraction, wrong-partner repulsion and dependence separately. Retain the existing statistic as a comparison between interventions, not a knowledge detector.

RUN: B’s small prose attraction in the complementary expert runs also survives this decomposition: approximately 0.0023–0.0036 nats, with window-bootstrap intervals above zero. So the “different but slightly worse” observation survives; its interpretation as knowledge-specific transfer still needs stronger controls.


B. The shuffled-partner control does not preserve a fixed wrong pair

Location: blammoe/toy/fused.py:212, :240, :247, :249.

CODE: Both directions roll by the same amount at every layer. A_i reads B_(i−1), but that B state has previously read A_(i−2). This creates a chain of documents across layers, rather than a mutually connected A_i/B_(i−1) pair.

This is a valid disruptive intervention, but it changes more than which document the partner reads. It also changes the communication topology.

RUN: I compared it with permuting B’s input batch once and keeping that pairing fixed. Scores differ. The large residual wrong-partner disaster persists—A prose is about 7.707 bits/char under the consistent mismatch—so this bug does not explain away hand-over. It does weaken exact interpretations of dependence and toward.

Recommendation: use consistent wrong pairs, several permutations, and controls matched on domain and context length.


C. Translator seeds and model slots are confounded with direction

Locations: blammoe/toy/config.py:48; scripts/toy_fused.py:31, :55; configs/toy_char_pair.yaml:19.

RUN: Reconstructing initialization through the training construction path produced identical translator-state hashes across e0/e1/e2, separately for each signal type. The complementary expert runs also share one initialization. The training sampler seed is identical.

This is not three independent optimization replications.

RUN: Swapping the two models and all corresponding trained adapters produced exactly swapped outputs, with zero numerical difference in my check. I found no hard-coded preference for the A direction in the forward implementation.

That leaves initialization, optimization and trained representations as live explanations. Swap model slots before declaring a domain-direction law.

There is another control gap: f units control A’s prose benefit, but there is no equivalent char–char A′+A′ control holding the prose reader fixed for A′’s code benefit. Also, e partners share initialization with their readers, while f partners do not.


D. The objective does not ask for balanced transfer, and gate size is not channel usage

Locations: blammoe/toy/fused_train.py:34, :79; blammoe/toy/fused.py:284; blammoe/adapters/gated_xattn.py:49, :75.

CODE: Unforced training minimizes the sum of next-token losses over all valid positions in both models. Ignoring the partner is permitted. Helping a weak domain while disturbing a strong domain can lose under this objective.

There is unequal opportunity: the char models’ average code specialization gap is about 0.47 bits/char, versus about 0.34 for prose. Comparing A’s absolute code BPC with A′’s prose BPC is not a valid comparison of competence across domains; comparing available within-domain improvement is more relevant.

For A–B, the two losses are token means, not character means. Tokenization therefore changes their relative scale. That does not explain the char–char result, but it matters when comparing experiments.

RUN: At zero gate, all non-gate translator gradients were exactly zero on the first backward pass. Gate gradients were nonzero. This is expected, not an implementation bug, but it makes early optimization dependent on the initial random branch.

CODE: A gate of 0.02 does not mean “2% information” or prove that the reader is shut. The output projection can compensate for gate magnitude. Measure injected activation norms and intervention effects.

RUN: In e0 residual fusion, the same final A gate produces an update RMS of 15.90 on prose versus 2.58 on code. Gate magnitude alone conceals substantial domain selectivity.


E. The knowledge-location conclusions exceed what mean ablation establishes

Locations: scripts/toy_where_knowledge.py:95, :108, :186, :201.

RUN: On a trained B model, I checked the single-expert hook against directly replacing that expert’s routed output. Maximum logit difference was approximately 9.5×10⁻⁶ in float32. The implementation appears correct.

The interpretation is weaker:

- Mean ablation measures dependence on an activation, not where a fact or capability is stored.
- Destroying an expert output can remove contextual information computed upstream.
- Individual ablation costs are not additive. Summing positive costs and reporting a top-two share is not a decomposition of knowledge.
- B and B′ also have tokenizers fitted on different mixes, complicating their comparison.

The justified conclusion is “B’s normal prose predictions depend materially on its expert outputs.” It is not “we have proved its transferable prose knowledge is in the expert bus.”

The suggested explanation that prose needs context unavailable to experts is also unsupported: the experts receive representations after attention, in blammoe/toy/moe.py:92.

RUN: There is useful positive evidence for the expert payload. In e0, zeroing expert-output content while retaining tags removes A′’s code gain: BPC rises from 2.027 to 2.115, versus 2.111 solo. Removing tags alone gives 2.051. This supports a real contribution from expert-output content, although zeroing is a disruptive intervention.


F. “48 documents” substantially overstates the independent prose coverage

Locations: blammoe/toy/corpus.py:116, :158; blammoe/toy/fused_train.py:146; blammoe/stats.py:44.

RUN: The prose evaluation consists of 48 windows of 256 characters drawn from three papers, with 15, 13 and 20 windows respectively. One pair of sampled windows overlaps.

Window bootstrap intervals measure uncertainty conditional on this small corpus. They do not measure generalization across prose domains, source documents, training recipes or model families.

RUN: Changing the window seed moved e0’s solo prose score from 2.208 to 2.360 and its residual gain from approximately 0.285 to 0.192 bits/char. The qualitative result survived, but effect size is sample-sensitive.

Repeatedly choosing experiments from these same test results also turns this test set into a development set. Use another untouched evaluation corpus for the next substantive claim.

Finally, “90% code” means 90% of the non-recall half: total pretraining is approximately 45% code, 5% prose and 50% synthetic recall. These are not cleanly separated knowledge specialists.


G. The leak audit is strong for the claimed causal property, but has a genuine fail-open bug

Locations: blammoe/toy/audit.py:117, :155, :168; scripts/toy_fused_audit.py:38, :61.

The good part: it compares full-vocabulary log probabilities at every scored position, in both directions. It does not merely compare average loss.

RUN: I reran the trained e0 residual translator on two fresh 256-character windows. All 510 positions per side passed at tolerance 10⁻⁹; maximum differences were approximately 2.5×10⁻¹⁴ and 3.4×10⁻¹⁴. Separate unread-suffix interventions in A, B and both streams also preserved earlier predictions.

I also inspected the saved reports: all 102 translators across the four relevant sweeps have passing reports. I did not independently rerun all 102 audits.

The bug: NaN differences do not satisfy `diff > tol`. An all-NaN model therefore passes.

RUN: I injected NaN outputs into a tiny pair; the gate returned `ok: True`, zero bad positions and zero maximum difference. Explicit finiteness checks are required.

This does not explain the finite results reproduced here.

The audit also preserves the reader’s existing token prefix while retokenizing its partner. It checks cross-stream lookahead under that conditioning. It does not establish data cleanliness, free-generation robustness or knowledge transfer. Float64 agreement is not a guarantee of identical float32 routing.

“Fusion does not beat the partner” is not evidence against leakage. The prefix and suffix checks are the relevant evidence.


H. Minor metric defect: the character distributions omit probability mass

Location: blammoe/toy/fused.py:356, :370, :381.

The character projection drops the pad probability, then passes the remaining mass to JS without renormalizing or adding an “other” category.

RUN: Omitted mass reached about 2.9% at one evaluated A′ prose position. This is not an explanation for the main pattern, but the quantities are not strictly JS divergences between normalized character distributions.

Cross-tokenizer comparisons also use only shared boundaries and can compare A with a B state not yet available through the safe bus. Their magnitudes should not be treated as directly comparable with char–char measurements over every position.


I. Checks that did not reveal bugs

RUN: All 12 copied char-pair model files matched their intended source files byte-for-byte. The pair-unit mapping in blammoe/toy/sweep.py:105 appears correct for this experiment.

RUN/CODE: Backbone parameters are frozen; my backward check produced no backbone gradients. Routed expert outputs are captured before router weighting. Gradients through the frozen computation remain available to train earlier links. I found no wiring bug there.


3. Ranked next experiments

All costs below are rough future estimates. No GPU work was performed.

1. Cross the initialization, slot and partner controls.

Start with the existing char models. Swap A/A′ slots, vary translator seed independently, then pair A_i with A′_(i+1), using A_i+A_(i+1) as the matched control. Add A′+A′ controls for the reverse direction.

This combines proposed experiment (i) with the more urgent optimization check. Start small and expand only if the result persists.

Cost: approximately 6–12 new translators, roughly 1.5–3 aggregate GPU-hours including audits at the observed toy timings. No pretraining.

2. Test what channel and objective actually cause the asymmetry.

Compare the existing expert bus with the already-supported router-weighted MoE-sum signal. Include a residual baseline without the final link. Test a one-way donor whose activations remain those of its solo computation. For the failing direction, try one controlled nonzero-gate initialization or donor-targeted training objective.

These separate difficulty reading individual expert entries from lack of usable information, and distinguish intermediate computation from final readout substitution.

Cost: approximately 4–8 targeted translators, around 1–2 GPU-hours. Avoid a large factorial sweep.

3. Run the 50/50-reader experiment, reusing existing checkpoints.

The matching char readers already exist at results/toy_sweep/s0–s2/model_a.pt. Their configurations resolve to p_prose=0.5, with matching architecture and training length.

Pair each with code-heavy and prose-heavy donors. First measure donor advantage: a balanced reader with little remaining prose deficit makes a negative result inconclusive. Evaluate donor-directed changes at meaningful disagreement positions as well as BPC.

This is proposed experiment (ii), and it is cheaper than assumed.

Cost: roughly six expert translators, about 1–1.5 GPU-hours plus analysis; no new pretraining.

4. Move to one bounded real-MoE pilot after fixing the evaluation.

I would not spend months tuning these toys. One short diagnostic round is worthwhile because larger models will not repair a confounded metric or initialization comparison.

Then use a tractable pretrained pair with demonstrable complementary behaviors, preferably initially sharing a tokenizer. Include generated continuations, donor-specific behavioral tests, an output-mixture baseline and a final-link-only baseline.

For “different, not better,” success should mean a reproducible behavioral change tracking something specific the donor knows, including on items withheld from translator training. Overall accuracy need not improve.

Cost: several days of engineering; measure throughput before approving a training budget. A precise GPU-hour estimate from the toy timings would be fiction.

5. Defer fair forcing at 0.01–0.02.

It is reasonable if reliable split recall is itself a deliverable. It is a low-information next step for explaining missing prose transfer or establishing a different model.

Cost: two settings across all nine existing units means 18 translators—several aggregate GPU-hours. I would spend that budget on the controls above first.


4. Anything else / likely dead ends

OPINION: Optimizing “depends on the partner” is a dead end unless tied to an externally meaningful donor-specific behavior. Noise qualifies. Even a smooth real-minus-wrong log-probability reward can improve by making the wrong branch worse if both branches receive gradients.

OPINION: “Free transfer” needs an operational definition. The current objective allows transfer when it lowers next-token loss; it does not encourage symmetric exchange, complementary reasoning or reusable experts. Those require different tests, not necessarily larger gates.

OPINION: A model can transmit contextual information through an expert output without transferring an expert’s distinctive learned capability. Routed-only capture is necessary for the intended attribution, but insufficient by itself.

OPINION: Hand-over is not automatically a failure under the owner’s stated aim. It already produces a partner-dependent model while preserving the original models when disconnected. But if that is enough, say so. If the intended result is cooperation that neither solo model could supply, use a task requiring both contributions and prevent the final-layer shortcut.

The next priority should be establishing what changes because of donor knowledge, under independent translator seeds and proper controls. More recall-dial tuning will not settle that.

Scratch scripts and machine-readable results are in:
 /tmp/claude-1000/-home-boz/0862934b-3b4c-4087-9ca1-94a01b1b355d/scratchpad/codex/work/

The main files are probe.py, checks.py, metrics.json, checks.json, symmetry.json and oneway.json.