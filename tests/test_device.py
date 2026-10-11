import torch

from blammoe.device import pick_device


def test_pick_device_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("BLAMMOE_DEVICE", raising=False)
    assert pick_device().type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert pick_device("cpu") == torch.device("cpu")
    monkeypatch.setenv("BLAMMOE_DEVICE", "cpu")
    assert pick_device() == torch.device("cpu")
