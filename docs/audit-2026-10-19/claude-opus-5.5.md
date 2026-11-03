# Adversarial audit — Claude (Opus 5.5), 2026-10-19

Scope: same brief as Codex/Gemini (`AUDIT-BRIEF.md`). CPU only, no checkpoints exist,
so all experiments use tiny untrained pairs with live (gate = 1) links.

## Experiments run

**E1. Two-way black-box causality probe** (`causality_probe.py`). Two texts share
`text[:c]` and differ after it (random suffix). Every A position with `j+1 <= c` and
every B position whose token ids/offsets agree in both encodings with end `<= c` must
give identical logits. 8 texts x 12 cuts, signals experts/moe/resid x rel_pos on/off,
`safe_mask=True`, via `run_pair`.
- Safe mask: max |dlogit| <= 7.5e-8 everywhere (~75k A positions, ~34k B positions). **Clean, both directions, all signals.**
- Positive control (`safe_mask=False`): A max |dlogit| 1.5e-2 to 4.6e-2, B ~1e-7. The probe
  detects the known leak, and confirms it is one-directional (A reading BPE), as documented.

## Findings

1. **[major, confirms Codex #1] Linear "transplant" is a dense additive injection on B's side.**
   `blammoe/toy/bridge.py:153-160`: `moe_out += alpha*(translated - expert_b(x))` at every
   token, ignoring B's routing and router weight. "Routed-only (native)" in
   `docs/toy-poc-results.md:62` describes only A's *source*. The 96.5% shows information
   reaches B; it does not show B's *expert* carried it. Fix: gate the correction by
   `p_e * routed_e`, rerun POC + controls.

2. **[minor] Expert "choice" is measured on the eval documents.** `scripts/toy_fused_experts.py:50-51`
   uses `split_recall_sample(Random(2025))` docs 0-47; the main eval (`split_eval_set`) and
   `toy_fused_hide.py:38-39` use docs 0-95 of the same seed. Every expert is ablated, so the
   ablation numbers are not inflated, but "the expert the translator chose" was not
   identified on held-out data. Fix: choose on a different seed.

3. **[major, interpretation; complements Codex #2] Hide ablation is out-of-distribution for the reader.**
   The translator never saw expert e missing during training, so a collapse can mean "the
   learned attention breaks" rather than "the information exists only in e". Note that top-k=2
   leaves the *other* routed slot at the same positions readable, which is a useful built-in
   position control, worth reporting explicitly. Checks: (a) fine-tune links briefly with e
   hidden and see if accuracy recovers through other experts; (b) train with random
   expert-entry dropout, then ablate; (c) Codex's content-only / ID-only ablations.

4. **[major, confirms Codex #5] The audit only covers A, and only in aggregate.** `toy_fused_audit.py`
   scores A-reads-B only and compares means; `tests/test_audit.py:87` checks per-position but
   for one text, experts signal only. E1 covers the gap for untrained pairs; the CLI should
   assert per-position equality, both directions, and fail on mismatch.

5. **[note] "<2 points" for other experts is within noise.** 1,092 A-needs-B digits = 273
   answers, digits within an answer are strongly correlated, so the effective n is ~273 at best:
   ±~2.6 pp (95%) at 95%. Report the bootstrap CI (`blammoe/stats.py`, after fixing Codex #8).

## Checked and found clean
- `closed_ends` matches the BPE pre-tokenizer (Digits individual -> same split regex);
  runs close one char after the run; single-char chunks close on read; end-of-text runs
  are unreadable. Two-way transitive paths reasoned through: B states read A at
  A-end <= B-token-end <= B-closed-end <= A-reader-end, so nothing reaches A from beyond its read chars. E1 confirms empirically.
- Pad positions as readers have end NEVER (read everything) but are after real tokens
  under causal self-attention and have target -100: no effect on scored positions.
- Expert bus `last_slots` is routed-only, pre-weight; `repeat_interleave(top_k)` matches
  `flatten(1,2)` position-major order; hide masks align with bus columns.
- Split-recall answers: digits are single tokens in both tokenizers; answer positions index
  the preceding input; chance 10%/digit; train seed 4242 vs eval seed 2025.
- POC 96.5% uses `expert_routed` for A's source (`scripts/toy_poc.py:122`), dense run reported separately.
