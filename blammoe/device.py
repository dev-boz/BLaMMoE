"""Device selection (WP0). Never hard-code .cuda(): call pick_device() (CLAUDE.md rule 4).

``BLAMMOE_DEVICE`` (e.g. "cpu", "cuda", "cuda:1") overrides the automatic choice.
"""
from __future__ import annotations

import os

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    choice = prefer or os.environ.get("BLAMMOE_DEVICE")
    if choice:
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
