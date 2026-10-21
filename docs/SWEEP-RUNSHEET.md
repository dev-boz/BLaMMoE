# Run sheet: the toy seed sweep (2026-10-20)

This is the retrain everything after the audit waits on. Every checkpoint in `results/toy_poc`
predates the M6 train/val/test split, so nothing in it may be reported. The sweep retrains from
pretraining and produces the numbers the docs need (HANDOFF "Next steps" 2, audit M8).
Code: `scripts/toy_sweep.py` (runner), `blammoe/toy/sweep.py` (unit configs + summary),
`configs/toy_sweep.yaml`. Branch `audit-fixes` (local, not pushed).

## What it runs

| Unit | Pair | Models |
|---|---|---|
| s0, s1, s2 | A_i (char) ↔ B_i (BPE), seeds (0,1), (2,3), (4,5) | pretrained here (s0 = the toy_poc seeds) |
| r0, r1, r2 | B_{i+1} ↔ B_i (same architecture, different seed: Rung 2) | copied from the s-units, so run after them |

Per unit, in order (a job is skipped if its output file exists):
1. Pretrain `model_a`, `model_b` (4000 steps each). s-units only.
2. `toy_fused.py` for `ab_experts_all_rel_safe` and `ab_resid_all_rel_safe` (3000 steps each).
3. `toy_fused_audit.py`: float64 causality gate on both variants (6 checks: experts normal/ablated × split/general, resid normal × split/general).
4. `toy_fused_experts.py` + `toy_fused_hide.py` on the experts variant (hide/blank/tags/joint, plus 3 recovery fine-tunes of 300 steps, each gated).
5. Ensemble baseline (`fused_ensemble.json`).

Output: `results/toy_sweep/<unit>/` (a symlink to `/media/boz/Data/blammoe/results/toy_sweep`),
plus `results/toy_sweep/summary.json` at the end.

## Before starting

```bash
cd ~/BLaMMoE && git status --short && git log --oneline -1    # clean, faeaddf or later, on audit-fixes
mountpoint /media/boz/Data || udisksctl mount -b /dev/nvme0n1p2
ls -la results/toy_sweep/                                      # should be empty (smoke runs went to /tmp)
nvidia-smi --query-gpu=index,pci.bus_id,temperature.gpu,memory.used --format=csv
systemctl is-active cmp-thermal-watch                          # cooling watcher up
PYTHONPATH=. .venv/bin/python -m pytest -q                     # 93 passed
```

## Start (one card: 43:00, the ×16 one)

Run it in tmux so it survives the remote session dropping:

```bash
tmux new -s sweep
cd ~/BLaMMoE
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  .venv/bin/python scripts/toy_sweep.py configs/toy_sweep.yaml 4 2>&1 | tee -a results/toy_sweep/sweep.log
# detach: Ctrl-b d      reattach: tmux attach -t sweep
```

- Check the first line says `on cuda`. If it says `cpu`, stop it: CPU would take days.
- Each script also writes its own log in the unit folder (`model_a.log`, `fused_<variant>.log`).
- **Smoke-run reference (CPU, tiny step counts):** one unit took ~2.5 h. The audit gate took
  81 min and the hide step 70 min, nearly all of it float64. There's no GPU estimate yet. s0's
  `finished in Ns` lines give the first real timings, so it's worth a look after s0.
- These models are small (2-3.5M params), so expect light GPU load. Watch temperatures anyway.

## While it runs, and stopping or resuming

- Progress: `grep '^\[sweep' results/toy_sweep/sweep.log`
- Problems look like `FAILED` (a unit hit an exception; the sweep moves on to the next unit),
  `exit 1` on an audit or hide job (a gate failure), or no new line for a long time.
- **Stopping is safe:** Ctrl-C, then rerun the same command. Finished jobs are skipped and a
  half-done job starts over. A unit whose config has changed refuses to run; use a new `out_dir`.
- Run single units by name: `... configs/toy_sweep.yaml 4 s0`. The r-units need s0-s2's
  `model_b.pt` first.

## When it finishes

- The last lines say `all done` or list `FAILURES`. The exit code is 1 if any job or gate failed.
- `results/toy_sweep/summary.json` has:
  - per unit and variant: digit and exact-answer accuracy with bootstrap CIs over 96 documents,
    fused minus shuffled partner, experts minus residual (paired), gate status, and attribution
    (chosen expert hidden/blanked, worst other expert, tags zero/shift, worst joint set, the 3
    recovery fine-tunes and their gates);
  - `across_seeds`: mean / sd / min / max over s0-s2 and over r0-r2.
- Then: a results write-up replacing the stale numbers in `docs/toy-fusion-results.md` and
  `docs/toy-poc-results.md` (M4). Note: `toy_poc.py` / `toy_followup.py` (the one-way POC) are
  **not** in the sweep. Run them once on the `s0` folder afterwards if the POC numbers need refreshing.

## Prompt for the next Claude session

  Check every unit finished and every gate
> passed, then read the summary against the project goal (does information transfer, and does it
> travel via the chosen experts?) and draft the results write-up.
