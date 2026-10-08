"""Two deliberately different tokenizers for the toy POC (WP-T).

Model A is character-level; model B uses a small BPE. Mismatched tokenization is
one of the spec's cross-family risks (spec 1.3 item 3), so the toy keeps it.
Because A has a token boundary at *every* char, each B token boundary is shared
with A, and the causally-aligned A position for B token [s, e) is simply A's
token at char e-1: both models have then read exactly text[:e].

Both expose ``encode(text) -> (ids, offsets)`` with HF-style char offsets.
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers

PAD = "<pad>"


class CharTokenizer:
    def __init__(self) -> None:
        chars = ["\n"] + [chr(c) for c in range(0x20, 0x7F)]
        self.itos = [PAD] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = 0

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        return [self.stoi[c] for c in text], [(i, i + 1) for i in range(len(text))]

    def token_id(self, s: str) -> int:
        return self.stoi[s]


class BPETokenizer:
    """BPE with digits forced to single tokens, so each answer digit is one token."""

    def __init__(self, tok: Tokenizer) -> None:
        self.tok = tok
        self.pad_id = tok.token_to_id(PAD)

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 1024) -> "BPETokenizer":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.Split(Regex(r"\s+|[A-Za-z]+|[^A-Za-z\s]"), behavior="isolated"),
        ])
        alphabet = ["\n"] + [chr(c) for c in range(0x20, 0x7F)]
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=[PAD], initial_alphabet=alphabet, show_progress=False
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        self.tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self.tok.encode(text)
        return enc.ids, [tuple(o) for o in enc.offsets]

    def token_id(self, s: str) -> int:
        tid = self.tok.token_to_id(s)
        if tid is None:
            raise KeyError(s)
        return tid
