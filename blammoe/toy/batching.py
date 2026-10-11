"""Tokenize Samples into padded LM batches and locate recall-answer tokens (WP-T)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from blammoe.toy.corpus import Sample


@dataclass
class Batch:
    ids: torch.Tensor  # [B, T] input ids (padded)
    targets: torch.Tensor  # [B, T] next-token ids, -100 = ignore
    offsets: list[list[tuple[int, int]]]  # char span of each *input* token
    answer_pos: list[list[int]]  # input positions whose next token is an answer digit
    full_offsets: list[list[tuple[int, int]]] = None  # input tokens + the final target token
    closed: list[list[int]] = None  # per input token: chars to read before its boundaries are final

    def to(self, device) -> "Batch":
        """Same batch with its tensors on ``device`` (offsets etc. stay Python lists)."""
        return Batch(ids=self.ids.to(device), targets=self.targets.to(device), offsets=self.offsets,
                     answer_pos=self.answer_pos, full_offsets=self.full_offsets, closed=self.closed)


def answer_positions(offsets: list[tuple[int, int]], answer_chars) -> list[int]:
    """Positions t such that token t+1 is exactly one answer digit char."""
    want = set(answer_chars)
    return [t - 1 for t in range(1, len(offsets)) if offsets[t][1] - offsets[t][0] == 1
            and offsets[t][0] in want]


def make_batch(samples: list[Sample], tokenizer, max_len: int, blind: bool = False, device=None) -> Batch:
    seqs, offs, fulls, closed, answers = [], [], [], [], []
    for s in samples:
        text = s.blind if blind else s.text
        ids, off = tokenizer.encode(text)
        ids, off = ids[: max_len + 1], off[: max_len + 1]
        seqs.append(ids)
        offs.append(off[:-1])
        fulls.append(off)
        closed.append(tokenizer.closed_ends(text, off[:-1]))
        answers.append([p for p in answer_positions(off, s.answer_chars) if p < len(ids) - 1])
    T = max(len(x) for x in seqs) - 1
    inp = torch.full((len(seqs), T), tokenizer.pad_id, dtype=torch.long)
    tgt = torch.full((len(seqs), T), -100, dtype=torch.long)
    for i, ids in enumerate(seqs):
        n = len(ids) - 1
        inp[i, :n] = torch.tensor(ids[:-1])
        tgt[i, :n] = torch.tensor(ids[1:])
    b = Batch(ids=inp, targets=tgt, offsets=offs, answer_pos=answers, full_offsets=fulls, closed=closed)
    return b if device is None else b.to(device)


def answer_mask(batch: Batch) -> torch.Tensor:
    m = torch.zeros_like(batch.targets, dtype=torch.bool)
    for i, pos in enumerate(batch.answer_pos):
        m[i, pos] = True
    return m
