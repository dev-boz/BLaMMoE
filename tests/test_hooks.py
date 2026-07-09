"""Verify the expert-capture hook grabs the expert's own output — i.e. BEFORE
router weighting and BEFORE the residual add — using a synthetic MoE block that
mimics the OLMoE/Qwen structure. No model download required."""
import torch
import torch.nn as nn

from blammoe.hooks import ExpertCapture, expert_path, resolve_module


class ToyExpert(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return torch.relu(self.fc(x))


class ToyMoEBlock(nn.Module):
    """x -> x + sum_i softmax(gate)_i * expert_i(x)  (router weight + residual)."""

    def __init__(self, d: int, n: int):
        super().__init__()
        self.experts = nn.ModuleList([ToyExpert(d) for _ in range(n)])
        self.gate = nn.Linear(d, n, bias=False)

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=-1)  # [*, n]
        outs = torch.stack([e(x) for e in self.experts], dim=-2)  # [*, n, d]
        return x + (w.unsqueeze(-1) * outs).sum(dim=-2)


class ToyModel(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.block = ToyMoEBlock(d, n)

    def forward(self, x):
        return self.block(x)


def test_hook_captures_pre_router_pre_residual():
    torch.manual_seed(0)
    d, n = 8, 4
    model = ToyModel(d, n).eval()
    x = torch.randn(2, 5, d)  # [batch, seq, d]
    target = 2

    path = expert_path("block.experts.{expert}", layer=0, expert=target)
    with ExpertCapture(model, path) as cap:
        block_out = model(x)

    # independent reference: the raw expert output
    expert = resolve_module(model, path)
    raw = expert(x)

    assert cap.output is not None
    assert cap.output.shape == x.shape
    # captured == the expert's own output, exactly
    assert torch.allclose(cap.output, raw, atol=1e-6)

    # and it is NOT the router-weighted contribution nor the block output
    w = torch.softmax(model.block.gate(x), dim=-1)[..., target].unsqueeze(-1)
    weighted = w * raw
    assert not torch.allclose(cap.output, weighted, atol=1e-4)
    assert not torch.allclose(cap.output, block_out, atol=1e-4)


def test_hook_is_removed_on_exit():
    model = ToyModel(8, 4).eval()
    cap = ExpertCapture(model, "block.experts.0")
    with cap:
        pass
    assert cap._handle is None
    # no lingering hooks on the module
    assert len(model.block.experts[0]._forward_hooks) == 0
