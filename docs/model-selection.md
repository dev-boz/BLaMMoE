# Model selection — concrete choices per rung

These are **proposed** and encoded in `blammoe/models/registry.py`. Every number
is VERIFY-tagged until an agent confirms it against the real checkpoint (WP1).

## Rung 1 — Known-Good (same run, different checkpoint)

- **Pair:** `olmoe-base` (later checkpoint) vs `olmoe-base-ckpt-early` (earlier
  checkpoint of the *same* `allenai/OLMoE-1B-7B-0924` run).
- **How to pick the early checkpoint:** OLMoE released intermediate pretraining
  checkpoints as branches on the HF repo. Open "Files and versions" → branch
  dropdown, pick a real intermediate step, and set it as the `revision` on
  `olmoe-base-ckpt-early` (currently a `VERIFY-…` placeholder).
- **Dims:** identical by construction → **orthogonal Procrustes**.
- **Purpose:** validate the metric battery, not to prove transfer.

## Rung 2 — Moderate (same family, different fine-tune)

- **Pair:** `olmoe-base` vs `olmoe-instruct` (`…-0924-Instruct`). Same
  architecture, different post-training.
- **Dims:** identical → **orthogonal Procrustes**.

## Rung 3 — The Goal (cross-family)

- **Pair:** `olmoe-base` (OLMoE) → `qwen-moe` (`Qwen/Qwen1.5-MoE-A2.7B`).
- **Dims:** may or may not match — both are candidate `hidden_size=2048`, but
  **confirm from each `config.json`**. If equal, you *can* still try Procrustes as
  a baseline, but the spec's expectation is a rectangular map → **linear
  least-squares (+ optional ridge)**.
- **Watch-outs:** Qwen1.5-MoE has *shared* experts alongside routed ones — make
  sure the hook targets a *routed* expert, and that the module-path template
  distinguishes shared vs routed. Confirm expert count (routed), active-k, and
  layer count on download.

## VERIFY checklist (do this in WP1, then delete resolved tags)

For each model:
1. Load via `transformers`, `print(model)` / iterate `named_modules()`.
2. Confirm the dotted path to a single routed expert's FFN matches
   `expert_module_template` (fix if not).
3. Read `config.json`: `hidden_size`, `num_hidden_layers`,
   num experts / top-k (names vary by arch). Update the registry.
4. Sanity-run `ExpertCapture` on one forward pass; assert
   `output.shape[-1] == hidden_size`.
5. Pick real depth-sweep layer indices for `configs/*.yaml` (early/mid/late)
   given the true layer count.
