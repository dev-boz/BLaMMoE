"""WP-T3: the partner-win mask credits only partner-dependent wins."""
import torch

from blammoe.toy.reward import partner_win_mask, win_weighted_loss


def onehot_logits(preds, V=5):
    return torch.nn.functional.one_hot(torch.tensor(preds), V).float()[None] * 10


def test_partner_win_mask_requires_solo_and_wrong_partner_to_fail():
    targets = torch.tensor([[1, 2, 3, 4, -100]])
    fused = onehot_logits([1, 2, 3, 0, 4])  # right on the first three
    solo = onehot_logits([1, 0, 0, 0, 0])  # solo already right at t=0
    wrong = onehot_logits([0, 2, 0, 0, 4])  # wrong partner also right at t=1
    m = partner_win_mask(fused, solo, wrong, targets)
    assert m.tolist() == [[False, False, True, False, False]]  # only t=2 is partner-dependent


def test_win_weighted_loss_adds_only_on_wins():
    targets = torch.tensor([[1, 2]])
    logits = torch.zeros(1, 2, 5, requires_grad=True)
    none = torch.zeros(1, 2, dtype=torch.bool)
    some = torch.tensor([[True, False]])
    base = win_weighted_loss(logits, targets, none, lam=2.0)
    boosted = win_weighted_loss(logits, targets, some, lam=2.0)
    assert torch.isclose(boosted, base * 3)  # uniform logits: win CE == mean CE
    assert torch.isclose(win_weighted_loss(logits, targets, some, lam=0.0), base)
