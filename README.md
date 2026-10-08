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

Foundation is in and tested (`pytest` green): config loader, model registry,
expert-capture hook, alignment fitters (WP4), metric battery + Rung-1 gate (WP5), char-span pooling (WP2). Everything else is specified in the work plan and
intentionally **not** stubbed — pick a work package and build it wired + tested.
