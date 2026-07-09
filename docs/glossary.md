# Glossary — spec jargon → what it means → where it lives in code

| Spec term | Meaning | Code location (built or planned) |
|---|---|---|
| Micro-Alignment Problem | Can fine-grained experts from independent models share a latent space without interference? | whole project |
| Expert-level hook point | An expert's own FFN output, pre-router-weight, pre-residual | `blammoe/hooks.py` (built) |
| Universal Latent Bus | Shared latent space all models' adapters project into | bus dim in `adapters/gated_xattn.py` (WP6) |
| Tx / Rx adapter pair | Per-model transmit (model→bus) / receive (bus→model) adapters | `adapters/gated_xattn.py` (WP6) |
| Tanh-gated cross-attention (zero-init) | Learned, gated injection; gate=0 at init → identity | `adapters/gated_xattn.py` (WP6) |
| Laddered progression / Rungs 1–3 | Increasing difficulty: same-run → same-family → cross-family | `configs/rung{1,2,3}.yaml`, `experiment/run.py` (WP11) |
| Character-span pooling | Align across tokenizer mismatch by pooling over a char window | `data/pooling.py` (WP2) |
| Boundary discard | Drop spans near either model's token boundary (imperfect pairs) | `data/pooling.py` (WP2) |
| Depth sweep | Collect at early/middle/late layers, not one depth | `layers:` in config; `data/collect.py` (WP2) |
| Two-stage expert selection | Coarse domain routing pre-filter, then operation-level concept probing | `selection/` (WP3) |
| Orthogonal Procrustes | Rotation-only map, same-dim (Rungs 1–2) | `align/procrustes.py` (WP4) |
| General linear least-squares / ridge | Rectangular map for cross-dim (Rung 3) | `align/linear.py` (WP4) |
| Metric battery | CKA + mutual k-NN + Procrustes residual (never CKA alone) | `metrics/battery.py` (WP5) |
| Validate the validators | Check metrics on known-good & known-bad pairs first | `metrics/battery.py` harness (WP5) |
| Test A — Graded Readout | Decode transported state (Patchscopes / Jacobian Lens) | `efficacy/readout.py` (WP7) |
| Test B — Divergence-Transport | Inject at disagreement positions; measure Δlogprob of source's token | `efficacy/divergence.py` (WP8) |
| Test C — Shuffled-Position Control | Correct vector, wrong position | `efficacy/shuffle.py` (WP9) |
| Test D — Generation-Robustness | Redo on free-generation states, not teacher-forced | `efficacy/generation.py` (WP10) |
| Transmission Illusion | Translator maps to target mean; looks coherent, transmits nothing | what Test B guards against |
| Ceiling condition | Native self-injection — the best achievable readout | baseline in `efficacy/readout.py` (WP7) |
