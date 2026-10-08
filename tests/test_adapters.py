"""WP6 acceptance: zero-init gate is an exact identity, shapes compose through
Tx -> bus -> Rx across different widths, the gate learns, and the mask is honoured."""
import torch

from blammoe.adapters import RxAdapter, TxAdapter


def test_zero_gate_is_exact_identity_and_shapes_compose():
    torch.manual_seed(0)
    tx, rx = TxAdapter(d_model=24, d_bus=16), RxAdapter(d_model=12, d_bus=16, n_heads=4)
    h_src = torch.randn(2, 9, 24)  # source model A: width 24, 9 positions
    x_tgt = torch.randn(2, 5, 12)  # target model B: width 12, 5 positions
    bus = tx(h_src)
    assert bus.shape == (2, 9, 16)
    out = rx(x_tgt, bus)
    assert out.shape == x_tgt.shape
    assert torch.equal(out, x_tgt)  # gate = 0 -> bit-for-bit unchanged


def test_gate_receives_gradient_and_opens():
    torch.manual_seed(1)
    tx, rx = TxAdapter(8, 8), RxAdapter(8, 8, n_heads=2)
    params = list(tx.parameters()) + list(rx.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    src, x = torch.randn(4, 6, 8), torch.randn(4, 6, 8)
    target = x + src.mean(dim=1, keepdim=True)  # only solvable by reading the source
    losses = []
    for _ in range(200):
        loss = ((rx(x, tx(src)) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert rx.gate.abs().item() > 0.05
    assert losses[-1] < 0.5 * losses[0]


def test_block_mask_hides_source_positions():
    torch.manual_seed(2)
    tx, rx = TxAdapter(8, 8), RxAdapter(8, 8, n_heads=2)
    with torch.no_grad():
        rx.gate.fill_(1.0)
    src, x = torch.randn(1, 4, 8), torch.randn(1, 3, 8)
    mask = torch.zeros(1, 3, 4, dtype=torch.bool)
    mask[:, :, 2:] = True  # every target position may read source positions 0-1 only
    base = rx(x, tx(src), mask)
    src2 = src.clone()
    src2[:, 2:] += 10.0 * torch.randn(1, 2, 8)  # perturb only hidden positions
    assert torch.allclose(rx(x, tx(src2), mask), base, atol=1e-6)
    src3 = src.clone()
    src3[:, 0] += 10.0 * torch.randn(8)  # perturb a visible position
    assert not torch.allclose(rx(x, tx(src3), mask), base, atol=1e-3)


def test_relative_bias_is_noop_at_init_and_learns_through_float_mask():
    torch.manual_seed(3)
    tx, rx = TxAdapter(8, 8), RxAdapter(8, 8, n_heads=2, n_rel_buckets=3)
    with torch.no_grad():
        rx.gate.fill_(1.0)
    src, x = torch.randn(2, 5, 8), torch.randn(2, 4, 8)
    buckets = torch.randint(0, 3, (2, 4, 5))
    mask = torch.zeros(2, 4, 5, dtype=torch.bool)
    mask[:, :, 4] = True
    assert torch.allclose(rx(x, tx(src), mask, rel_buckets=buckets), rx(x, tx(src), mask), atol=1e-6)
    rx(x, tx(src), mask, rel_buckets=buckets).pow(2).sum().backward()
    assert rx.rel.weight.grad is not None and rx.rel.weight.grad.abs().sum() > 0
    with torch.no_grad():  # a large bias on one bucket steers attention there; masked entries stay masked
        rx.rel.weight.zero_()
        rx.rel.weight[2] = 50.0
    _, w = rx(x, tx(src), mask, return_weights=True, rel_buckets=buckets)
    assert torch.all(w[..., 4] == 0)
    picked = (buckets[..., :4] == 2).float()
    has = picked.sum(-1) > 0
    assert torch.allclose((w[..., :4] * picked).sum(-1)[has], torch.ones(1), atol=1e-3)
