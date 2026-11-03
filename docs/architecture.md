# BLaMMoE Phase 1 — Architecture & Design

## The one thing to get right: the hook point

The experiment captures **a specific expert's own FFN output, before router
weighting and before the residual add** (spec 1.1). If you hook the post-addition
layer hidden state instead, you are re-running whole-model alignment (already
covered by CALM/MoT) — not testing *expert-level* transmissibility.

Mechanically this is easy and is already built (`blammoe/hooks.py`): a
`register_forward_hook` on the expert submodule returns exactly that module's
output. The `gate * expert_out` multiply and the residual add happen in the
enclosing MoE block *after* the expert returns, so both are excluded for free.
The only per-model work is confirming the module path to the expert
(`ModelSpec.expert_module_template`).

## Topology: hub-and-spoke, not pairwise

One **Tx/Rx adapter pair per model** into a shared **Universal Latent Bus**, not a
translator per model-pair. For N=2 that is 2 adapter pairs. All experts inside one
MoE already share that model's residual stream, so one pair per model suffices.

Note honestly (spec 1.2): N=2 validates transmissibility *for this pair*. It does
**not** demonstrate the hub topology's advantage over a pairwise translator — that
only shows up at N≥3 (Phase 2). Don't oversell the hub in Phase 1 write-ups.

## Injection: learned, gated, zero-initialized

Foreign vectors enter via **tanh-gated cross-attention with the gate initialized
to zero** (Flamingo/CALM). The target model *learns how much* of the foreign
vector to read, rather than receiving a raw magnitude addition. At init the model
is unchanged (gate=0 → identity). That guarantee holds **at init only**: after training
the gates are open, and in the 2026-10-21 sweep every fused pair is 0.002-0.02 bits/char
worse than solo on ordinary text (`docs/toy-sweep-results.md`). Unpaired, the frozen models
are untouched.

## Pipeline (data flow)

```
select experts (WP3) ─┐
                      ▼
corpus ──► both frozen models ──► ExpertCapture at swept depths ──► char-span
(WP2)                              (WP1)                            pooling (WP2)
                                                                      │
                                                        paired (h_A, h_B) in data/
                                                                      │
                          ┌───────────────────────────────────────────┤
                          ▼                                            ▼
                 fit map A→B (WP4)                          metric battery (WP5)
                 Procrustes | lstsq/ridge                  CKA + mutual-kNN + resid
                          │                                 validate-the-validators
                          ▼
                 Tx/Rx gated x-attn (WP6)
                          │
        ┌─────────────────┼───────────────────────┐
        ▼                 ▼                         ▼
   Test A readout    Test B divergence      Tests C/D controls
   (WP7)             transport (WP8)        (WP9, WP10)
```

## The laddered rungs (spec 1.2)

| Rung | Pair | Dim match | Fitter |
|---|---|---|---|
| 1 Known-Good | same run, different checkpoint | yes (by construction) | orthogonal Procrustes |
| 2 Moderate | same family, different fine-tune/seed | yes | orthogonal Procrustes |
| 3 The Goal | cross-family (OLMoE → Qwen-MoE) | likely no | linear least-squares (+ridge) |

Rung 1 exists to **validate the metric battery**, not to prove anything about
cross-model transfer. Don't skip it.

## Design principles carried from the spec

- **Validate the validators first.** Run the metric battery on a known-good and a
  known-bad pair before trusting it on the real pair. CKA alone is unreliable
  (`2210.16156`) — always the full battery.
- **Supervised, not adversarial.** Paired `(h_A, h_B)` is free at scale via
  char-span pooling, so fit supervised maps. Unsupervised/adversarial alignment
  risks confident-but-wrong mappings when isomorphism fails (`1805.03620`).
- **A geometric fit is necessary but not sufficient.** Procrustes residual proves
  correspondence, not *usability*. The efficacy tests (A–D) are what prove the
  target model can actually use the transported vector.
- **Test D is load-bearing.** Teacher-forced calibration states may not represent
  free-generation states at deploy time. "The Latent Bridge" (`2606.24470`)
  converged beautifully offline and failed at deployment — treat generation
  robustness as a gate, not a footnote.
- **Beware the Transmission Illusion.** A translator that just maps toward the
  target's activation mean passes coherence checks while transmitting nothing.
  Test B (divergence-transport) is designed to catch exactly this.

## Storage & device

- Activations are large and regenerable → `data/` (git-ignored). Suggested format:
  memmap or `.npz` with `h_A`, `h_B`, char-span id, layer, corpus-shard id.
- Everything is device-agnostic. Collection runs where the RAM/GPU is; alignment,
  metrics, and control analyses are CPU-fine and can run on any box.
