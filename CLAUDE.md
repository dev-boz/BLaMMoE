# CLAUDE.md — working agreement for agents on BLaMMoE

You are building a research codebase to test whether MoE experts from
independently-trained models can be aligned into a shared latent space without
catastrophic interference. Read `docs/workplan.md` for the task list and
`docs/architecture.md` for the design before writing code.

## Non-negotiables

1. **No unwired scaffolding.** Do not create empty modules, `pass`-body classes,
   or "to be implemented" files. A module enters the repo only *with* a passing
   test that exercises it. If you can't test it yet, it's not ready to add.
   Prefer additions over deletions; if you must delete/rewrite existing code,
   flag it and say why first.
2. **Verify wiring end-to-end.** Before you claim a work package done, run the
   thing and observe real output — not just a typecheck. `pytest` must pass.
3. **Respect VERIFY tags.** Values in `models/registry.py` (module path
   templates, hidden sizes, layer/expert counts) are best-effort guesses. Confirm
   them against the real checkpoint's `named_modules()` / `config.json` before
   relying on them, and update the registry with what you find.
4. **Device-agnostic.** Never hard-code `.cuda()`. Select with
   `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. This box has
   no GPU; the code must still run here for CPU-friendly work packages.
5. **Never commit data or weights.** `data/`, `results/`, and `*.safetensors`
   are git-ignored. Activations are large and regenerable.
6. **One work package per change.** Keep diffs scoped to a single WP so they can
   be reviewed and handed off independently.

7. **Cross-tokenizer fusion must be leak-proof.** When one model reads another's
   BPE tokens, build the pair with `safe_mask=True` and run it through `run_pair`
   (a safe pair refuses a direct call without writer ends). Every trained fused
   checkpoint must pass `scripts/toy_fused_audit.py`: full-text log-probs equal
   prefix-honest re-runs at EVERY position, both models, in float64 (float32 GPU
   rounding flips near-tied routing and looks like a leak); it exits nonzero on
   any mismatch. BPE boundaries on the full text depend on later
   characters, and the ordinary controls (wrong document, wrong position, copy of
   itself) cannot catch that leak. It has already produced one withdrawn result.
8. **Expert-level claims use routed-only expert outputs.** An expert evaluated
   densely on every token reads the whole residual stream through its weights, so
   even a never-routed expert appears to transmit.

## Conventions

- Package is `blammoe`; tests live in `tests/`, named `test_<module>.py`.
- Match the existing style: typed signatures, small functions, module docstrings
  that cite the relevant spec section (e.g. "spec 1.3") and paper IDs.
- Config-driven: an experiment is one YAML in `configs/` loaded via
  `blammoe.config.ExperimentConfig.from_yaml`. Add fields there, not as argv.
- Interface contracts for unbuilt modules are given in `docs/workplan.md`. Honor
  those signatures so downstream packages compose.

## Running things

```bash
# tests (CPU env)
PYTHONPATH=. .venv-dev/bin/python -m pytest -q
# a single work package's tests
PYTHONPATH=. .venv-dev/bin/python -m pytest tests/test_metrics.py -q
```

## Where the truth lives

- Current state, how to reproduce, and next steps: `docs/HANDOFF.md` (start here)

- Experimental design & rationale: `blammoe-spec-and-bibliography-v4.md`
- Source papers (local only, gitignored; re-fetch with `scripts/download_papers.py`): `papers/` — see `docs/papers-index.md`
  for which paper backs which work package. Read the paper before implementing
  the method it describes.
