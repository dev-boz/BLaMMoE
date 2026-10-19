"""Pretrain one toy MoE LM (model_a or model_b) from configs/toy_poc.yaml.

    PYTHONPATH=. python scripts/toy_train.py configs/toy_poc.yaml model_a
"""
import json
import sys
from pathlib import Path

import torch

from blammoe.device import pick_device
from blammoe.toy import setup
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.moe import param_count, save
from blammoe.toy.train import train_lm


def main(cfg_path: str, which: str, threads: int = 2) -> None:
    torch.set_num_threads(threads)
    cfg = ToyPOCConfig.from_yaml(cfg_path)
    spec = getattr(cfg, which)
    torch.manual_seed(spec.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = setup.tokenizer(spec, cfg)
    device = pick_device()
    model = setup.build_model(spec, tok.vocab_size, cfg).to(device)
    logf = open(out / f"{which}.log", "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"{which}: {param_count(model):,} params, vocab {tok.vocab_size}, on {device}")
    hist = train_lm(model, tok, setup.sampler(cfg, seed=spec.seed), steps=cfg.steps,
                    batch_size=cfg.batch_size, lr=cfg.lr,
                    eval_samples=setup.eval_recall_samples(cfg), log=log)
    save(model, out / f"{which}.pt", history=hist)
    (out / f"{which}_history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
