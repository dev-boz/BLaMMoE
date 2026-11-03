You are giving an independent, critical second opinion on a research project. The owner asked for outside input. Be blunt: say where the reasoning is wrong, where the evidence is weaker than claimed, and what you would do next. Check the code and data yourself rather than trusting the write-ups.

## The project

Repo: /home/boz/BLaMMoE (git branch `audit-fixes`). Start with docs/HANDOFF.md ("START HERE"), docs/architecture.md and docs/glossary.md, then the results docs in this order: docs/toy-complement-results.md, docs/toy-dial-results.md, docs/toy-dial2-results.md, docs/toy-different-results.md, docs/toy-char-pair-results.md.

End goal (owner's words, paraphrased): take two off-the-shelf MoE language models that each work fine alone, and train only a small "translator" (cross-attention links between the two models, on the expert outputs = "expert bus", or on the residual stream) that allows AND encourages free expert-to-expert knowledge transfer. The models stay frozen. The owner's latest framing: the aim is a *different* model (different because of what the partner knows), not necessarily a *better* one. They also suspect very small models may just not show much, and that it may take larger models.

Current work is on toy models (4 layers, d_model 96-128, 8-16 experts per layer), trained here: A = char-level, code-heavy; B = BPE, prose-heavy; A' and B' = the same architectures on the other data mix. Leak safety (causal masking across two tokenizers) is checked by a float64 audit on every trained translator.

Recent findings (please scrutinise):
1. A "toward" measure (scripts/toy_partner_shift.py) that separates real partner-driven difference from damage.
2. A-side gap: A never gets prose from B through the expert bus; its gates stay ~0.02. B does get code from A.
3. A + A' (same architecture, same tokenizer, same init seed): A still gets no prose via the expert bus, but A' gets code from A. That was read as a "direction effect" (code flows into a prose model, prose never flows into a code model). Through the residual stream the two char models hand over almost everything (A prose 2.21 -> 1.92 bits/char, gates ~0.55, wrong partner 7.7 bits/char).

Proposed next steps (none agreed yet): (i) pair A with A' from a different seed set to remove the shared-init confound; (ii) a 50/50-mix char model as a reader, to tell "prose doesn't travel through experts" from "a code-trained reader can't use prose features"; (a) "fair forcing" at 0.01-0.02 from dial 2; or move to real, larger MoEs.

## What I want from you

1. Is the interpretation of the char-pair result right? Are there alternative explanations the write-ups miss (e.g. the training objective, which positions the loss is on, the gate init, the data mix, eval set size, the fact that A is better at code than A' is at prose, the residual-stream hand-over being a degenerate solution)? Is "direction effect" justified from 3 seeds and one data mix?
2. Look for bugs in the code that could produce these results: blammoe/toy/fused.py, blammoe/toy/fused_train.py, blammoe/toy/sweep.py (pair units), scripts/toy_partner_shift.py, scripts/toy_where_knowledge.py. In particular, is the "toward" measure sound? Is the leak audit actually strict enough to rule out leaks in the residual-stream hand-over?
3. Given the end goal and the "different, not better" aim, what's the best next experiment, or set of experiments? Rank the proposed ones and add your own. Say whether you think toy scale is still worth it or whether it's time to move to real MoEs, and why.
4. Anything in the overall approach you think is a dead end.

## Rules (important)

- Full filesystem and shell access is enabled so you can check things properly, but this is a review: do NOT edit, create or delete anything inside /home/boz/BLaMMoE, and do not git commit, push, checkout or stash. Put any scratch scripts and outputs in /tmp/claude-1000/-home-boz/0862934b-3b4c-4087-9ca1-94a01b1b355d/scratchpad/codex/work/.
- NO GPU work at all. The owner approves every GPU job personally. Prefix every Python command with `CUDA_VISIBLE_DEVICES=""`. Use the repo venv: `cd /home/boz/BLaMMoE && CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python ...`.
- CPU checks are fine but keep them small: at most 8 threads (torch.set_num_threads(8)) and under ~20 minutes each. The test suite is `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest -q`.
- results/ are symlinks to /media/boz/Data/blammoe/results (already mounted). Read-only: don't write into results/. Trained checkpoints (model_a.pt, model_b.pt, fused_<variant>.pt), summary.json, partner_shift.json and where_knowledge.json are there.
- Don't touch any other disk, mount or unmount anything, or install system packages.

## Output

A plain-text report: (1) verdict on the interpretation, (2) bugs or methodological problems found, each with file:line and how you checked it, (3) ranked next experiments with rough cost, (4) anything else. Mark what you verified by running code versus what is opinion.
