# BLaMMoE — Phase 1

**Target claim (the Micro-Alignment Problem):** can highly idiosyncratic,
fine-grained MoE experts from *independently-trained* checkpoints be aligned into
a shared cross-model latent space **without catastrophic interference**?

Phase 1 scope is deliberately narrow: prove **non-destructive, high-bandwidth
transmission of a single expert's own output between two disjoint models** (N=2).
Router training, cross-model task gains, and N>2 are out of scope (Phase 2).

- Full experimental spec: [`blammoe-spec-and-bibliography-v4.md`](blammoe-spec-and-bibliography-v4.md)
- Bibliography verification: [`verification_report.md`](verification_report.md)
- **Work breakdown (start here to build): [`docs/workplan.md`](docs/workplan.md)**
- Design & data flow: [`docs/architecture.md`](docs/architecture.md)
- Spec-jargon → code map: [`docs/glossary.md`](docs/glossary.md)
- Concrete model choices: [`docs/model-selection.md`](docs/model-selection.md)
- Which paper backs which task: [`docs/papers-index.md`](docs/papers-index.md)

## Repo layout

```
blammoe/            importable package (only tested modules exist — no empty stubs)
  config.py         experiment config + YAML loader            [built]
  models/registry.py concrete checkpoint specs (VERIFY tags)   [built]
  hooks.py          expert-FFN capture primitive (WP1)         [built]
  align/            Procrustes / lstsq / ridge fitters (WP4)   [built]
  metrics/          CKA + mutual k-NN + residual battery (WP5) [built]
  data/pooling.py   char-span pooling for tokenizer mismatch   [built]
  adapters/         Tx/Rx tanh-gated cross-attention (WP6)     [built]
  toy/              toy MoE pair + transmission POC (WP-T)     [built]
configs/            one YAML per rung
docs/               work plan + design docs (agent hand-off)
papers/             33 source papers as markdown (already fetched)
scripts/            one-off utilities (e.g. download_papers.py)
tests/              pytest — every built module has a test here
data/  results/     git-ignored; activations & experiment outputs
```

## Quickstart

This machine has **no GPU**; real model runs happen on the 128 GB / 1080 Ti box
or rented inference. The code is device-agnostic. Two install profiles:

```bash
# CPU-only (authoring, linear-algebra work packages, running tests)
uv venv --python 3.14 .venv-dev
uv pip install --python .venv-dev --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv-dev pytest pyyaml numpy
PYTHONPATH=. .venv-dev/bin/python -m pytest -q

# Full stack (a box that will actually load OLMoE / Qwen-MoE)
uv venv && uv pip install -e ".[dev]"     # pulls CUDA torch, transformers, datasets
```

> On the 1080 Ti (Pascal / sm_61): current `torch` default wheels target new CUDA
> and may drop Pascal. If CUDA init fails, install a CUDA 12.x build
> (`--index-url https://download.pytorch.org/whl/cu121`) or run that model on CPU.

## Status

**Start with [`docs/HANDOFF.md`](docs/HANDOFF.md)**: current state, how to reproduce,
next steps. All work is on branch `claude/new-session-f94vqn`.

Built and tested (`pytest` green, 64 tests): config loader, model registry,
expert-capture hook, alignment fitters (WP4), metric battery + Rung-1 gate (WP5),
char-span pooling (WP2), Tx/Rx gated cross-attention adapters (WP6), device selection,
bootstrap statistics, a hardware probe, and a toy end-to-end programme:

- One-way expert transmission between two different toy MoEs:
  [`docs/toy-poc-results.md`](docs/toy-poc-results.md).
- Two-way lockstep fusion with translator-chosen experts, a tokenizer-leak audit and
  leak-proof results: [`docs/toy-fusion-results.md`](docs/toy-fusion-results.md).
- **Current numbers:** the seed sweep (3 cross-family + 3 same-design pairs, CIs, causal
  expert attribution): [`docs/toy-sweep-results.md`](docs/toy-sweep-results.md).
- Complementary models (code-heavy A, prose-heavy B, translator trained on ordinary text only,
  same-domain control partners): [`docs/toy-complement-results.md`](docs/toy-complement-results.md).
- The allow/force dial, part 1 (a growing share of forced recall on the complementary models):
  [`docs/toy-dial-results.md`](docs/toy-dial-results.md).
- The dial, part 2 (the partner-win reward, and the force dial with fair answer weighting):
  [`docs/toy-dial2-results.md`](docs/toy-dial2-results.md).
- "Different, not better" and why A never gets prose (CPU diagnostics on the trained translators):
  [`docs/toy-different-results.md`](docs/toy-different-results.md).
- A with a prose-heavy char partner (same architecture as A): the expert-bus gap is not the tokenizer:
  [`docs/toy-char-pair-results.md`](docs/toy-char-pair-results.md).
- Outside review (Codex gpt-6-astra) of the char-pair result, and the attraction / wrong-partner split of "toward":
  [`docs/review-2026-10-31.md`](docs/review-2026-10-31.md).
- Controls for the char pair (new translator seed, swapped slots, a partner from another init): the one-way pattern holds:
  [`docs/toy-char-ctrl-results.md`](docs/toy-char-ctrl-results.md).
- Which channel carries prose into A: the MoE-sum signal and the residual stream do (2 seeds of 3), the per-expert bus doesn't:
  [`docs/toy-char-channel-results.md`](docs/toy-char-channel-results.md).

The real-model work packages (WP1 loader, WP2 collection, WP3, WP7-WP11) are specified
in the work plan and intentionally **not** stubbed.
