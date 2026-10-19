"""Untrained fixture for the audit CLI: real config dims, live random links (gates = 1)."""
import sys
from pathlib import Path

import torch

from blammoe.toy import setup
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.fused import FusedPair
from blammoe.toy.moe import save

cfg = ToyPOCConfig.from_yaml(sys.argv[1])
out, fs = Path(cfg.out_dir), cfg.fused
tok_a, tok_b = setup.tokenizer(cfg.model_a, cfg), setup.tokenizer(cfg.model_b, cfg)
torch.manual_seed(0)
A = setup.build_model(cfg.model_a, tok_a.vocab_size, cfg).eval()
B = setup.build_model(cfg.model_b, tok_b.vocab_size, cfg).eval()
save(A, out / "model_a.pt")
save(B, out / "model_b.pt")
for name in ("ab_experts_all_rel_safe", "ab_resid_all_rel_safe", "ab_experts_all_rel"):
    v = fs.variants[name]
    pair = FusedPair(A, B, v["links"], fs.d_bus, fs.n_heads, v["signal"], v.get("rel_pos", False), v.get("safe_mask", False))
    with torch.no_grad():
        for lk in pair.links.values():
            lk.rx_a.gate.fill_(1.0)
            lk.rx_b.gate.fill_(1.0)
    torch.save(pair.links.state_dict(), out / f"fused_{name}.pt")
print("fixture ok", sorted(p.name for p in out.iterdir()))
