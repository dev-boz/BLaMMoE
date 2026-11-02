# Handoff — BLaMMoE, state as of 2026-11-01

For whoever picks this up next, human or agent. Read `CLAUDE.md` first (working agreement).
`main` has the toy programme (PR #1 merged). The audit fixes, the seed sweep and its write-up
and everything since (complementary models, the allow/force dials, the char pair, the outside review) are on local branch
**`audit-fixes`** (committed, NEVER pushed; don't push unless the owner asks). 118 tests pass:
`CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest -q` (the venv is `.venv`).

## Project goal (from the owner)

**End goal (owner, 2026-10-21):** two MoE models that work just fine on their own (ideally
off-the-shelf), plus a translator that allows, and encourages, free transfer of information
between their experts. The translator need not be an LLM: it sees only activations (a
"large latent model"). Keep the small links until they stop being enough, then test an
LLM-style translator against alternatives (e.g. a non-generative decision model like Jev).

Step one is proving that experts can be translated between models **at all**. Improvements
are a bonus. Mixing e.g. a Qwen with a Mistral and getting a *worse* result still counts as
success. So prioritise work that makes "information transfers, and it travels via experts"
airtight over ordinary-text quality.

## Working method: the size ladder (owner, 2026-10-31)

Bounce between three sizes instead of climbing once. Test what's quick and easy on the toys, move
up a level when a question needs more data or more capacity, come back down to test a theory, and
go to medium only when really sure.

| level | what | use it for |
|---|---|---|
| toy | our 2-3.5M toy MoEs (minutes per translator) | mechanisms, controls, bug hunts, anything needing many seeds |
| small | a real off-the-shelf small MoE pair (to pick; must fit the CMPs) | does a toy result survive real models and real text |
| medium | the registry's OLMoE-1B-7B / Qwen1.5-MoE pair or similar | the claim itself, once it holds at small |

- Up a level only when the result holds under independent translator seeds, swapped slots and
  matched controls, and the question really can't be answered lower down.
- Down a level whenever something higher up surprises us: build the cheapest toy version first.
- Same measures and leak audit at every level, so numbers compare, and a held-out eval set per
  level that isn't used for choosing experiments.

## Working method: the AI gauntlet (owner)

Whenever we get stuck, or a result needs a thorough review before it gets built on, I write one
prompt and feed it to at least 5 and up to 15 different agents across models and harnesses
(Claude Code, Codex CLI, Gemini via agy, opencode, ...). Different models catch different things,
and several agreeing counts for more than one strong reviewer.

- One self-contained brief: the goal, what to read, the claims under test, and hard rules (CPU
  only with `CUDA_VISIBLE_DEVICES=""`, no edits to the repo and no git, a scratch dir, time caps).
  Ask for findings with file:line, marked as run-and-checked or opinion.
- Each agent gets a private copy of the repo or read-only rules. No agent runs GPU jobs.
- Check every finding against the code before acting on it, then merge into one doc with who found
  what and how many agreed.
- Past runs: `docs/audit-2026-10-19/` (brief `AUDIT-BRIEF.md`, merged in `docs/audit-2026-10-19.md`)
  and `docs/review-2026-10-31/` (Codex gpt-6-astra xhigh only, a one-agent start).
- Codex: `codex exec -m gpt-6-astra -c model_reasoning_effort='"xhigh"'
  --dangerously-bypass-approvals-and-sandbox -C <dir> -o <report> - < prompt.md`. agy can't take
  prompts headless, so run it in tmux.

## START HERE: after the dials (next session)

**Prompt for the next session:**
> Resume BLaMMoE: read docs/HANDOFF.md "START HERE" and docs/toy-dial2-results.md. Both dials are
> done. Help me pick between (a) fair forcing at 0.01-0.02, (b) a smooth reward, (c) the A-side
> prose gap. Ask before any GPU job; mount the Data drive first.

**Where the programme is (2026-10-30), in the order it happened:**
1. Toy sweep (`docs/toy-sweep-results.md`): a translator between two frozen toy MoEs moves
   information through their experts when a task forces it (split recall 97-99%), not otherwise.
2. Complementary models (`docs/toy-complement-results.md`): A pretrained mostly on code, B mostly
   on prose, translator trained on ordinary text only. B learns code from A unforced, every seed,
   4-7x a same-domain control partner. A gets no prose through the expert bus (a little through
   the residual stream). This is the first evidence for "encourages" in the end goal.
3. Force dial, part 1 (`docs/toy-dial-results.md`): mixing in a small share of forced recall
   killed the free transfer. Cause: a loss bug. The answer term averaged over the batch's answer
   tokens only, so one answer weighed 21-146x an ordinary character.
4. Dial part 2 (`docs/toy-dial2-results.md`): with fair weighting (`answer_in_mean`, w 20),
   p_split 0.05 keeps about half of B's free code transfer AND B<-A recall 90-91%, in all 3 seeds,
   at the unforced cost = **best balance so far**. The partner-win reward (`reward_lam`) hurts at
   every weight tried (5/15/50). Best guess: its argmax win mask with one wrong partner rewards
   noise that depends on the partner, not knowledge.

**2026-10-30 CPU diagnostics (`docs/toy-different-results.md`, scripts `toy_partner_shift.py` /
`toy_where_knowledge.py`):** A's expert links into A never open (B's prose knowledge is there, in its
experts, spread thin). The owner's new framing: the aim is a *different* model, not necessarily a
better one; `toward` in `partner_shift.json` is the measure. Proposed next (not agreed): pair A with
A' (d units' prose-heavy char model) to rule the tokenizer / expert spread in or out (new unit kind).

**2026-10-31 char pair (`docs/toy-char-pair-results.md`, `configs/toy_char_pair.yaml`, sweep `pairs:`):**
the expert-bus gap is a DIRECTION effect (code flows into a prose model, prose never into a code model),
not B's tokenizer, B's thin expert spread or A's architecture. Residual stream between same-tokenizer
models hands each over to the other on its weak domain. Next not agreed.

**2026-10-31 outside review (`docs/review-2026-10-31.md`):** the direction effect isn't established
(shared translator init, fixed slots); `toward` is now split into attraction and wrong-partner pull;
audit gate now fails NaN. Its ranked next: (1) controls (translator seed, slot swap, A_i+A'_(i+1),
A'+A'), (2) what blocks prose (MoE-sum signal, resid without final link, non-zero gate start), (3) the
50/50 char reader (models exist in results/toy_sweep), (4) a small-level real-MoE pilot. Owner to pick.

**2026-11-01 controls (`docs/toy-char-ctrl-results.md`, `configs/toy_char_ctrl.yaml`, translator
seed 1):** the one-way pattern holds with a new translator seed, swapped slots (x units) and an A'
from another init (g units). A prose = control partner (-0.018 bits/char, attraction +0.002 but wrong
pull +0.002); A' code -0.10..-0.14 vs A'+A' control -0.024, attraction +0.014..+0.024 (9/9) vs 0.
Gain size per unit moves ~2x with translator seed: read means. Audit gate also fixed for unequal-
length docs (reader past a shorter partner's end; audit artefact, not a leak; 2148ce3). 
**2026-11-01 channel run (`docs/toy-char-channel-results.md`, `configs/toy_char_channel.yaml`, 18/18
PASS):** the one-way pattern is the PER-EXPERT BUS's. Via the MoE-sum signal A gets prose from A' in 2/3
seeds (-0.17/-0.19 vs control -0.02, attraction +0.044/+0.048, A's rx gate opens -0.30/-0.34 on link
3, first time ever); resid without the final link same (-0.13/-0.19, link 2). Seed 2 never opens A's
reader on either. Eval-time ablation: all of it via the latest link, wrong partner +0.85..+1.25 bpc =
a late hand-over again, not expert sharing. Proposed next (not agreed): non-zero gate start for A on
the expert bus (opening vs combining).

**2026-11-01 gate start (`docs/toy-char-gate-results.md`, `configs/toy_char_gate.yaml`, 6/6 PASS):**
it was OPENING. A's expert-bus readers started at tanh 0.3 (`gate_start_a`, `FusedPair.start_gates`)
-> A gets prose from A' in 3/3 (-0.185/-0.204/-0.195 vs gate-started A+A control -0.03; attraction
+0.045..+0.050 vs 0); A's link-3 gate grows 0.30 -> ~0.58, links 0-2 close; control closes all. Still
all via link 3 and wrong partner +0.9..+1.2 bpc = last-layer hand-over. Direction effect is dead. Next
ideas (not agreed): gate start without the last link; which A' experts A reads at link 3; gate start on A-B.

**2026-11-02 gate follow-ups (`docs/toy-char-gate2-results.md`, `configs/toy_char_gate012.yaml`,
`configs/toy_ab_gate.yaml`, 12/12 PASS):** A reads A''s last layer as a set (read ~ routing; hide 7/8 =
whole gain). Links 0-2 with gate start: A prose -0.097/-0.124/-0.072 vs control -0.03, attraction 3/3,
wrong partner only +0.06..+0.20 (link 3 gave ~+1.1), mostly via link 2 (gate holds 0.1-0.3, doesn't grow).
A' code e1 low again (-0.042): a trade-off for that pair. A-B with gate start: A prose -0.04..-0.05 vs
k control -0.03 (shut -0.014), attraction +0.009 3/3, no dependence, prose-specific part via link 3;
B code unchanged. Same recipe gives -0.19 with char A', so the A-B gap is the tokenizer/architecture
crossing. Overnight follow-up queued 11-02 00:43: both sides' gates start 0.3 (`configs/toy_char_gate2way.yaml`,
`configs/toy_ab_gate2way.yaml`, results in results/toy_char_gate2way, results/toy_ab_gate2way).

**2026-11-02 both sides open (`docs/toy-char-gate3-results.md`, configs toy_char_gate2way / toy_ab_gate2way
/ toy_char_gate2way012, 27/27 PASS):** two-way transfer works at once on A+A' (all links: A prose -0.19..-0.20,
A' code -0.22..-0.32, but both depend on the other, wrong partner +0.7..+1.3, all via link 3). Links 0-2
both open: A -0.110/-0.121/-0.079 (f -0.03), A' -0.214/-0.116/-0.142 (h -0.057), attraction 3/3 both,
wrong partner +0.06..+0.38, mostly link 2 = the setting to build on. A-B both open: A unchanged, B +0.03
raw but its control +0.05 (generic). Suggested next (not agreed): build on links 0-2 both open (e.g. push
it off the latest link, or the small-level pilot); attack the tokenizer crossing for A-B.

**2026-11-03 links 0-2 robustness + links 0-1 (`docs/toy-char-gate4-results.md`, configs
toy_char_gate2way012s1 / toy_char_gate2way01, 21/21 PASS):** links 0-2 both open at translator seed 1:
A -0.085/-0.118/-0.097, A' -0.211/-0.112/-0.140 (within 0.025 of seed 0); other-init A' partners (g):
A -0.09..-0.13, A' -0.14..-0.18; wrong partner +0.11..+0.32; mostly link 2 again. e1's weak A' code is
the pairing (fine in g). Links 0-1 only: A prose = control (-0.03), A' code -0.07..-0.11 vs h -0.045,
no dependence, mostly link 1. So gain beyond control falls with the latest link: link 3 ~0.17, link 2
~0.08, link 1 0 for A. Link dropout 0.5 (configs/toy_char_gate2way_drop.yaml, gate4 Result 3, 18/18 PASS):
A's prose still comes only via the latest link (all-but-latest = control); A' code spreads a little to the
next link down. Gain and dependence fall together: all links A -0.16..-0.18, wrong +0.53..+0.75; links 0-2
A -0.05..-0.07, wrong +0.02..+0.03 (least dependent per bit yet). Reading: at toy size A' middle layers
hold little prose A can use -> argues for the small-level pilot. Nothing running.

**Open problem for the end goal:** (updated 11-02) the one-way pattern was the zero gate start, but
every working channel prefers the latest link, and A gets only a little prose from BPE B (-0.017 beyond
control) against -0.16 from char A'. Crossing tokenizers is now the main gap.

**Next options (owner to pick; none agreed yet):**
- (a) Fair forcing at p_split 0.01-0.02: does free transfer return towards unforced while recall
  holds? Cheapest: a new config like `configs/toy_dial2.yaml` with only `*_tok` variants, new
  out_dir (a changed config in an existing out_dir is refused), ~15 min GPU per variant per unit.
- (b) A smooth reward: e.g. log p(target | real partner) minus the mean over several wrong
  partners, as an extra loss term; no argmax, so noise shouldn't pass. Needs code in
  blammoe/toy/reward.py + train_fused, with a test, then a dial config.
- (c) The A-side gap: where B's prose knowledge lives (expert routing on prose vs code, how much
  of B's prose advantage is experts vs attention), and a char-level prose-heavy partner for A to
  rule the tokenizer in or out (needs a new control unit kind in blammoe/toy/sweep.py).

**How runs work now (all from 10-28..10-30):**
- Sweeps: `scripts/toy_sweep.py <cfg> 4 <units...>`. `models_from:` copies the complement models
  (+ BPE) instead of pretraining, so dials compare like for like. Units: s = real pair, c = A with a
  code-heavy B (control for A's prose gain), d = B with a prose-heavy A (control for B's code gain).
- The owner has OK'd **both CMPs at once** for these sweeps (ask before each job anyway). Split
  units 42:00 `s0 c0 d0 s2` / 43:00 `s1 c1 d1 c2 d2`, each in its own tmux, log via tee, ending with
  `echo SWEEP-EXIT ${PIPESTATUS[0]}`. Both processes write summary.json, so afterwards rebuild it on
  CPU with `CUDA_VISIBLE_DEVICES="" ... toy_sweep.py <cfg> 4 none`.
- **Thermal guard (owner asked for one):** `~/cmp-tests/dial2-guard.sh <results dir>` in its own tmux.
  It kills both sweeps at core >= 80C / HBM >= 88C / 3 failed nvidia-smi reads, and writes GUARD-KILL
  into the sweep logs. The pkill pattern is hard-coded to `configs/toy_dial2.yaml` and the tmux
  names to dial2_42/43: edit those for a new config. Test it first with `--dry-run` and
  `CORE_MAX=30`. The always-on cmp-thermal-watch.service is the backstop. These loads peak ~50C.
- Watching: ONE background Bash until-loop (sleep 30) that exits on both SWEEP-EXIT, GUARD-KILL,
  Traceback|FAIL|xception, a log quiet > 20 min, or the guard tmux gone. Haiku watcher subagents
  don't work here (sleep is blocked).
- **Always pin the GPU**: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (42:00) or `=2`
  (43:00), and `CUDA_VISIBLE_DEVICES=""` for CPU work. Unpinned, torch picks a CMP on its own.
- The Data drive is unmounted after every reboot: `udisksctl mount -b /dev/disk/by-label/Data` before
  touching results/. A new out_dir needs `mkdir /media/boz/Data/blammoe/results/<name>` + `ln -s`.
  /tmp (and the session scratchpad) is wiped on reboot too.
- Every trained checkpoint must pass the float64 audit gate (the sweep runs it): 90/90 PASS across
  the complement and both dials.
- Write-ups: owner's voice (first person, plain, Australian spelling, "Term - definition" lines,
  tables, show the process including dead ends), committed straight away, linked from README and
  the Results docs table below.

**Dials (built 2026-10-28), both on the toy_complement models (`models_from`, copied, not retrained),
expert bus, same s/c/d units:**
1. Force: `configs/toy_dial.yaml`, p_split 0.05 / 0.15 / 0.5. **DONE 10-29, written up in
   `docs/toy-dial-results.md`.** Not a usable dial as the loss stands: `pair_loss`'s answer term
   averages over the batch's answer tokens only, so one answer token weighs ~146x / 78x / 21x an
   ordinary one at 0.05 / 0.15 / 0.5. Any forcing switched B's free code transfer off (fully in
   seeds 0, 2); recall is on/off per seed. Fix before rerunning: answer tokens count 1 + w inside the
   overall token mean (new option, keep the old path for old configs).
2. Encourage + fair force: `configs/toy_dial2.yaml`, out `results/toy_dial2`. Variants `reward_lam`
   5 / 15 / 50 at p_split 0 (train_fused `reward_lam`: partner-dependent win tokens count 1 + lam,
   blammoe/toy/reward.py; lam 0 = the toy_complement experts run, same data stream and init), and
   `*_split05_tok` / `*_split15_tok`: p_split 0.05 / 0.15 with `answer_in_mean` (answer tokens count
   1 + 20 inside the token mean, ~ the 21x the old 0.5 runs had). Reward training is ~1.4x slower per
   step (two extra no-grad forwards). **DONE 10-30** (5.8 h / 7.4 h on the two CMPs, guard
   `~/cmp-tests/dial2-guard.sh` never tripped, peaks 50C core / 59C HBM).
Read per setting: B's code gain and A's prose gain vs the c/d controls, split recall, strong-domain costs.

### Complementary models (built and run 2026-10-28)

**State (2026-10-22).** Branch `audit-fixes`, local only, never pushed (head `9e15964`), with
96 tests passing (103 after the 10-28 build below). The seed sweep and causal attribution are
done and written up in **`docs/toy-sweep-results.md`**. That doc is written in the owner's voice, so keep the style
when editing it (first person, plain wording, Australian spelling). The owner edits docs as
they go: commit write-ups straight away. Sweep in short (CMP 43:00, 3.4 h, every gate passed):
- **Transfer:** both directions in 5 of 6 pairs (97.5-99.5%, chance ~10%). r1's A-from-B
  expert run did not learn. A reseeded retry (`fused.seed: 1`, `results/toy_sweep/r1_fseed1`)
  reached 81.3%, and 98.4% after a 300-step fine-tune.
- **Attribution:** now picks the partner expert causally on the seed-2026 selection set
  (`causal_chosen`, bd8b7b3). Hiding it leaves 11-41%. The old attention-share pick was wrong
  in 12 of 13 directions; its outputs are kept in `<unit>/attnsel/`.
- **Residual stream:** works about as well as the expert bus.
- **Ordinary text:** every fused pair is 0.002-0.02 bits/char worse than solo. A plain output
  average beats both solos (0.07-0.11 bits at shared positions), so the models make different
  errors, but the links don't use them.

**Why this experiment.** It tests the end goal's "encourages free transfer" (see the goal
above). Today the links only transfer when a task forces it (split recall), and both models
saw the same text. Give each model knowledge the other lacks, train the links on ordinary
text only, and see whether transfer happens because it helps.

**Built (2026-10-28), not yet run.** The owner settled the open choices: keep recall docs
in pretraining (`p_recall: 0.5`), fit B's BPE on B's own mix, start at 0.1/0.9 (dose-response
later). 103 tests pass. Everything lives in **`configs/toy_complement.yaml`** (sweep runner):
1. **Per-model data mix:** `ToyModelSpec.p_prose` (default 0.5 = draws bit-for-bit as before;
   `test_p_prose_half_draws_exactly_as_before_the_field_existed` pins a fingerprint of the old
   sampler). A (char) `p_prose: 0.1` = mostly code, B (BPE) `0.9` = mostly prose. The BPE
   cache name gets `_p<mix>` when the mix is not 0.5 (`setup.bpe_name`), so each mix gets its
   own BPE; the sweep fits each once in its root.
2. **Links on ordinary text only:** a variant may set `p_split` (overrides `fused.p_split`).
   The config adds `ab_experts_all_rel_safe_text` / `ab_resid_all_rel_safe_text` (the sweep's
   leak-proof pair with `p_split: 0.0`) via `overrides`, so `toy_poc.yaml` and the old sweep's
   unit configs are untouched. `hide: false` (attribution is on split recall, never trained here;
   split recall is still evaluated, so spontaneous transfer would show).
3. **Per-domain eval:** `domain_eval_sets` (48 prose + 48 code test windows, seed 778).
   `toy_fused.py` writes `domain: {prose|code: {fused, shuffled_partner, solo}}` with per-doc
   `[bits, chars]` rows; the audit gate now also checks those windows; the ensemble job adds
   per-domain `ensemble_bits`. `summary.json` gives per domain and model the bits/char with CIs
   and paired `fused_minus_solo` / `fused_minus_shuffled` (negative = the partner helps).
   **Success:** A's prose (m1) and/or B's code (m2) beat solo and the shuffled partner.
4. **Controls** (`controls:` in the sweep config, units `c<i>` / `d<i>`, kind `control:<name>`):
   `c` keeps s_i's A and pretrains B again on A's mix (0.1) = a same-size, same-architecture
   partner that knows code; `d` keeps s_i's B with a prose-heavy A (0.9). A's prose gain in s_i
   should beat its gain in c_i (and B's code gain in s_i beat d_i).
5. **Run order:** seed 0 first (`s0 c0 d0`, est. ~1.5 h on 43:00: 4 pretrains, 6 fusions,
   3 audits), then `s1 s2 c1 c2 d1 d2` if it looks worth it. First:
   `udisksctl mount -b /dev/disk/by-label/Data`, `mkdir -p /media/boz/Data/blammoe/results/toy_complement`,
   `ln -s /media/boz/Data/blammoe/results/toy_complement results/toy_complement`. Then in tmux:
   `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python
   scripts/toy_sweep.py configs/toy_complement.yaml 4 s0 c0 d0 2>&1 | tee results/toy_complement/sweep.log`.
   ALWAYS set `CUDA_VISIBLE_DEVICES`: torch's default order is fastest-first, so without it
   device 0 is the CMP at 42:00 (a CPU smoke on 10-28 landed there by accident).
6. **Encouraging transfer (later variants):**
   - the stage-2 reward (`scripts/toy_fused_reward.py`, Next steps 4);
   - a JEPA-style extra loss where the links predict the partner's expert outputs from their
     own (Claude's suggestion, not yet agreed with the owner).

**Practicalities (10-22; superseded by "How runs work now" in START HERE).** GPU = CMP `43:00`
(`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`), tell the owner before each job (they may
call stop). Run long jobs in tmux. The `results/` symlinks point at `/media/boz/Data`: mount
it first (`udisksctl mount -b /dev/disk/by-label/Data`). The root disk has ~4.7 GB free. Timings on
43:00: one pretrain 4-6 min, one fusion ~8 min, an audit of 2 variants ~7 min, hide ~8 min.

**Prompt for the next session (same as the one at the top of START HERE):**
> Resume BLaMMoE: read docs/HANDOFF.md "START HERE" and docs/toy-dial2-results.md. Both dials are
> done. Help me pick between (a) fair forcing at 0.01-0.02, (b) a smooth reward, (c) the A-side
> prose gap. Ask before any GPU job; mount the Data drive first.

## Audit fixes (background)

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
  - The small fixes: bootstrap 0/0 resamples, the enrichment baseline (legal entries only) and exact 4-digit answer accuracy. See the "Minors" row in the audit status table.
  - The sweep runner (`scripts/toy_sweep.py`, see Next steps 2).
  - 93 tests pass.
- **M3 is built** (wiring run on the untrained fixture). See the M3 row in the audit status table. Results need trained checkpoints, so they arrive with the retrain (step 2 below). Run `scripts/toy_fused_experts.py` before `scripts/toy_fused_hide.py`: the hide script takes its chosen experts and blank constants from the seed-2026 selection set.
  - How to read it: if **blank** collapses like **hide**, the information is in the expert's output, not its tag or slot. If **tags zero/shift** costs little, the reader doesn't rely on the tag. If **joint** hiding takes B←A from 29.1% to about chance, the residual was link 0/1. If **recovery fine-tune** brings accuracy back, the information also reaches the partner through other experts, so it isn't exclusive to the chosen ones; compare against the `none` arm.
  - `hide` (all experts) is **not** the same as links off: a reader left with only the null slot still gets the attention biases. Use `disabled=` for "solo".
  - `docs/audit-2026-10-19/codex_audit_checks.py` still uses the old `hide_experts=` keyword. It is an archived probe, so it was left as is.
- **M6 is done** (2026-10-20). See the M6 row in the audit status table.
  - **Rule:** anything fitted or trained (BPE, pretraining, bridges, links, linear fits) reads `setup.texts(cfg, "train")`, selection reads `"val"`, reported ordinary-text numbers read `"test"`. `setup.sampler(cfg, split, seed)` has no default split. Split-recall and recall samples are synthetic and stay seed-separated.
  - **Packages, not just files.** The scoped design split per file. That left ~5% of held-out windows verbatim in train, because `encodings/cp*.py` repeat whole codec tables. Hashing the top-level package name instead cut it to ~0.2% (val) / ~0.8% (test) of windows. What remains is copied license headers and copied code, and the test allows at most 2%. Splits: train 2.1M prose + 6.0M code chars (capped), val 0.25M + 0.97M, test 0.29M + 0.79M.
  - **Every old checkpoint is pre-M6.** The models, BPE and fused links in `results/toy_poc` were fitted on what is now val/test text. The BPE cache is now `bpe1024_train_v0.1_t0.1.json`, so the old `bpe1024.json` is ignored, but old `model_*.pt` / `fused_*.pt` files would still load. Retrain from pretraining ("Reproduce", step 1) before reporting anything.
- **Small fixes done** (2026-10-20). Split-recall scores now carry `exact` (all 4 digits right, over `n_answers`) beside digit `acc`; report both. Enrichment `ratio` uses a `baseline` key (was `presence`) over legal entries only, so old `_experts.json` ratios are not comparable.
- **Next:**
  1. **Retrain once by running the sweep** (`scripts/toy_sweep.py`, Next steps 2 below; the runner is built). It retrains from pretraining. Include ≥3 seeds for residual vs expert (M8). Every fused checkpoint must pass the audit gate. Re-measure the leaky-run numbers in the docs (M4: depth table, expert-choice shares, stage 2).
- **Also in the audit doc: expert-tagged buses are sensitive to routing near-ties.** In float32 on GPU, rounding that differs by batch shape flips near-tied top-k choices, and the N(0,1) `eid` embeddings amplify the flip. Keep that in mind when interpreting the hide/ID results.

## This machine (boz-linux)

- **Repo and environment:** `~/BLaMMoE`. The `.venv` was built with pip (torch 2.14.1+cu130).
- **Disk:** the root disk is ~95% full. `results/toy_poc` is a symlink to `/media/boz/Data/blammoe/results/toy_poc` on the NTFS drive. If that drive is unmounted, run `udisksctl mount -b /dev/disk/by-label/Data`. Set `HF_HOME=/media/boz/Data/blammoe/hf_cache` before downloading models.
- **GPUs:** 2× CMP 170HX, both healthy: ~195 TFLOPS bf16, ~11 fp32, 64 GB each, passed a 60 GB read-back. `42:00` runs at Gen2 ×4 (slow host transfers) and `43:00` at Gen2 ×16. Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`; then CUDA 1 = `42:00`, 2 = `43:00`, 0 = the 3060 Ti (don't use it). Tell the owner before each GPU job; one card at a time unless they OK both (they did for the 10-28..10-30 sweeps). There is no P2P, and none is needed: run one job per card.
- **Test fixture:** for CLI wiring checks without trained checkpoints, copy `configs/toy_poc.yaml` with `out_dir` pointed elsewhere, then run `PYTHONPATH=. python docs/audit-2026-10-19/gate_fixture_build.py <that.yaml>`. It builds untrained models plus live links with gates = 1. To shorten training scripts, also set the per-variant `steps:` under `fused.variants`: it overrides `fused.steps`.

## Where things stood after PR #1 (one paragraph; see the audit for caveats)

> **Superseded by the sweep** (`docs/toy-sweep-results.md`). Two claims below no longer hold:
> the attention-chosen expert is usually *not* the one carrying the transfer (pick causally),
> and the residual stream does not fail in one direction (it works about as well, 3 seeds).

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
PYTHONPATH=. .venv/bin/python -m pytest -q   # 96 tests, ~13 s on CPU
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
| **`docs/toy-char-gate4-results.md`** | **Newest.** 2026-11-03, 39/39 PASS. Links 0-2 both open holds at translator seed 1 (within 0.025 of seed 0) and with other-init A' partners (A -0.09..-0.13, A' -0.14..-0.18, wrong partner +0.11..+0.32). Links 0-1 only: A gets nothing beyond control, A' a little (0.03-0.06), no dependence. The translator always uses the latest link; the gain shrinks with its depth. Link dropout 0.5 (18/18 PASS) lowers gain and dependence together but doesn't spread A's prose off the latest link; A' code spreads a little to link 2. |
| **`docs/toy-char-gate3-results.md`** | 2026-11-02 both sides' readers start at 0.3, 27/27 PASS. A+A' all links: A prose -0.19..-0.20 AND A' code -0.22..-0.32 at once (controls -0.03 / -0.07), seed-1 'trade-off' gone, but a mutual last-layer swap (link 3 gates ~0.55-0.6, wrong partner +0.7..+1.3). Links 0-2: A -0.08..-0.12, A' -0.12..-0.21, wrong partner only +0.06..+0.38, mostly link 2 = best shape so far. A-B: B's side open adds only generic gain; A's gap unchanged. |
| **`docs/toy-char-gate2-results.md`** | 2026-11-02 three follow-ups to the gate start, 12/12 PASS. Which experts (CPU): A reads A''s link-3 experts in proportion to routing, not prose specialists; single-hide costs sum to 2x the gain. Links 0-2 only: A prose -0.07..-0.12 vs control -0.03 with wrong partner only +0.06..+0.20 (vs ~+1.1 with link 3), mostly link 2. A-B with A's gates at 0.3: A prose -0.04..-0.05 vs control -0.03 (shut -0.014), no dependence; char A' gives -0.19 under the same recipe, so the A-side gap is crossing tokenizers/architectures, not B lacking prose. |
| **`docs/toy-char-gate-results.md`** | 2026-11-01 A's expert-bus readers start part-open (0.3), 6/6 PASS: A gets prose from A' in 3/3 (-0.19..-0.20 vs control -0.03), so the one-way pattern was the zero gate start. Still all via the last link; wrong partner +0.9..+1.2 bpc. |
| **`docs/toy-char-channel-results.md`** | 2026-11-01 MoE-sum signal and resid without the final link on the char pairs, 18/18 PASS: A gets prose from A' in 2/3 seeds through both (-0.17/-0.19, -0.13/-0.19 vs control -0.02; A's reader gate opens for the first time), so the one-way pattern belongs to the per-expert bus. All via the latest link; wrong partner +0.85..+1.25 bpc = late hand-over. A' code larger on both (-0.18, -0.28). |
| **`docs/toy-char-ctrl-results.md`** | 2026-11-01 controls for the char pair, translator seed 1, 15/15 gates PASS: the one-way pattern survives a new translator seed, swapped slots and an A' from another init. A on prose = control (-0.018), attraction +0.002 matched by wrong-partner pull; A' on code -0.10..-0.14 vs A'+A' control -0.024, attraction 9/9 above zero vs 0. Per-unit size moves ~2x with translator seed. |
| **`docs/review-2026-10-31.md`** | 2026-10-31 outside review (Codex gpt-6-astra xhigh; raw report + scripts in `docs/review-2026-10-31/`): every translator shares one init + draw order (`fused.seed` 0) and the code model is always slot 1, so "direction effect" is unproven; `toward` = attraction minus wrong-partner pull (now split in partner_shift.json; B code attraction +0.039 vs toward +0.089; resid hand-over 84% wrong-partner damage); audit NaN pass fixed. Ranked next steps by size. |
| **`docs/toy-char-pair-results.md`** | 2026-10-31 A + A' (prose-heavy char, same arch and init as A), 12/12 gates PASS: expert bus still gives A NO prose (= control partner, gates ~0.02), while A' gets code from A (-0.08..-0.20, like B) = a direction effect, not tokenizer/spread/reader arch. Residual stream char-char: near-total hand-over both ways (A prose 2.21 -> 1.92, last-link gates ~0.55, wrong partner 7.7 bits). |
| **`docs/toy-different-results.md`** | 2026-10-30 CPU diagnostics, no training: a "different because of the partner" measure (toward = real partner pulls predictions to the partner's own view more than a wrong partner; separates real transfer from the lam-50 reward's damage); B on prose is different-not-better (3/3); A-side gap = A's expert-bus reader stays shut (gates ~0.02 vs B's 0.09-0.21); B's prose IS in its experts (layer-0 MoE carries half its advantage) but spread thin over 64 small experts. Next: A + A' (char, prose-heavy) pair. |
| **`docs/toy-dial2-results.md`** | 2026-10-30 dial part 2 (45/45 gates PASS): fair force weighting (answer_in_mean, w 20) at p_split 0.05 keeps ~half of B's free code transfer (-0.050 vs control, 3/3 seeds) AND B<-A recall 90-91%, no extra cost; the partner-win reward hurts at every lam (5/15/50). |
| **`docs/toy-dial-results.md`** | 2026-10-29 force dial (p_split 0.05/0.15/0.5 on the complementary models, 27/27 gates PASS): any forcing switches B's free code transfer off (2 of 3 seeds fully); forced recall is on/off, not graded; cause = the answer term weighs each answer token 21-146x. |
| **`docs/toy-complement-results.md`** | 2026-10-28 complementary models, 3 seeds + same-domain controls: B learns code from A unforced (6/6, 4-7x control); A gets no prose via the expert bus, some via the residual stream (2/3). |
| **`docs/toy-sweep-results.md`** | **Current numbers.** The 2026-10-21 seed sweep: 3 cross-family + 3 same-design pairs, CIs, causal attribution, residual baseline, ensemble. Where older docs disagree, this wins. |
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
5. **Never average an extra loss term over a sparse subset.** `pair_loss`'s answer term (a mean
   over the batch's answer tokens) gave each answer 146x / 78x / 21x an ordinary token at p_split
   0.05 / 0.15 / 0.5, and turned "a little forcing" into "kill the free transfer"
   (docs/toy-dial-results.md). Weight per token inside the one mean (`answer_in_mean`). Same
   lesson as rule 3; check this for any new loss term.
6. **A partner-win reward that is fair per token can still be harmful.** `reward_lam` 5-50 from
   scratch made pairing worse everywhere, controls included (docs/toy-dial2-results.md).
   "Depends on the partner" (argmax flips vs one wrong document) isn't "uses what it knows".

## Next steps, in priority order

1. **Independent adversarial audit** of the eval path before any scale-up. Use a fresh
   session with no stake in the results, told to *find leaks and eval bugs*, not to
   confirm. Cheaper to catch at 3M params than at 7B. (External-review recommendation.)
2. **DONE 2026-10-21: review sweep on GPU**, see `docs/toy-sweep-results.md`. **Runner (2026-10-20):**
   `scripts/toy_sweep.py configs/toy_sweep.yaml [threads] [unit ...]`, helpers and summary in
   `blammoe/toy/sweep.py`.
   - **Units.** Each unit is a full toy_poc-style run in `results/toy_sweep/<unit>/` (a symlink
     to the Data drive), so the single-run scripts work on it unchanged. Main units `s0..s2`
     pair A_i with B_i, with seeds (0,1), (2,3), (4,5). `s0` uses the toy_poc seeds. Rung-2
     units `r0..r2` pair B_{i+1} (as `model_a`) with B_i, copying the main units' B models,
     so they need no extra pretraining.
   - **Per unit:** pretrain, then `toy_fused.py` for each variant (experts_safe and resid_safe).
     Then the float64 audit gate on all variants; a failure is recorded and the sweep exits 1.
     Then `toy_fused_experts.py` + `toy_fused_hide.py` for each experts variant, then the
     ensemble baseline. A job is skipped if its output already exists, so rerunning resumes.
     A unit's config is written to `<unit>/config.yaml`; a rerun with a changed config stops.
   - **`summary.json`:** per unit and variant, bootstrap CIs over documents for digit and exact
     accuracy, fused minus shuffled partner, experts minus residual (paired: same seed-2025
     documents), the gate, and attribution (chosen expert hidden or blanked, worst other
     expert, tags, joint, recovery fine-tunes). Across seeds: mean, sd, min and max per kind.
   - **Splitting the units:** pass unit names, e.g. `s0 s1 r0` and `s2 r1 r2`. Run the two
     halves on both cards at once only if the owner agrees; the default is one card at a time.
     The r-units need their B models from the s-units first. One summary run at the end
     (with no unit jobs left) collects everything.
   - `toy_fused.py` now saves per-document rows (`per_doc`) for fused, shuffled-partner and
     solo, which the CIs need.
3. **DONE 2026-10-28: complementary models** (`docs/toy-complement-results.md`); the
   dose-response (shifted mix, disjoint files, no code) is still open. Original plan: Pretrain A mostly on
   code and B mostly on prose, then train the links on ordinary mixed text only, with no
   recall task. Success = each model improves on its weak domain while staying fine alone:
   the first test of "encourages free transfer". A plain output average already beats both
   solos on ordinary text (0.07-0.11 bits at shared positions), so the models do make
   different errors, but today's links don't use them. Then the dose-response: shifted mix
   → disjoint files → no code. Needs a file-subset option in `blammoe/toy/corpus.py`.
4. **DONE 2026-10-30 as a from-scratch dial, harmful at λ 5/15/50** (`docs/toy-dial2-results.md`);
   the allow/force dials and their next options are in START HERE. Original plan: stage-2 reward-weight sweep (λ ≈ 10-30) under `safe_mask`. λ=2 did nothing and
   ~100× wrecked accuracy; the interesting range is between. Then reward on generated
   text, which needs lockstep generation across two tokenizers (not built).
5. **Real models (the actual goal):** WP1 `blammoe/models/loader.py` + verify the
   VERIFY-tagged registry against real OLMoE-1B-7B and Qwen1.5-MoE-A2.7B checkpoints.
   Check whether the installed `transformers` still exposes experts as a ModuleList.
   Then WP2 collection with **routed-only** expert outputs, then port the fused pair.
   192 GB RAM holds both models (≈14 GB + ≈29 GB bf16).

## Open caveats on what *is* claimed

- Toy scale (2-3.5M params), 3 seeds per pair type, CIs over 96 documents (the sweep).
  Scores are teacher-forced, not generated.
- The channel is narrow: a recall answer is ~3 bits, mostly carried at the last layer.
  First-block-only exchange reached 38.6% one way (under the old mask; not rerun leak-proof).
- Both toy models saw the same training data, the friendliest case for alignment.
- Expert-level is **not** better than residual-level on accuracy (within 1.3 points paired,
  except +5.2 once). What the expert bus adds is attribution: which experts carry it.
- The information lands in the receiver's residual stream, not its experts. This is a channel
  between models, not yet a portable expert or a translator that generalises to a new
  partner or task.
- The ensemble baseline has run (in the sweep). Note that for split recall an
  output-level mixture trivially gets the informed model's answer. The claim is about
  information entering a model's *own* computation, so the mixture is a reference
  point, not a competitor.
