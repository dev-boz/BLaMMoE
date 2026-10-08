"""Toy-POC corpus: prose + code + a synthetic key->value recall task (WP-T).

The recall task is what makes "does information transmit?" measurable. A doc
looks like::

    store: kiwi=4821, lynx=0937, opus=1102; ask: lynx=0937, kiwi=4821

The *blind* variant masks every stored value with ``?`` (same length, so char
offsets are identical between the clean and blind text). A model reading the
blind text cannot know the answer digits in the ``ask:`` section — chance is 10%
per digit — so any accuracy above that on blind text must have come from
somewhere else (here: the other model's expert, via a translator). Keys are
random letter strings and values random digits, so nothing is memorisable.

Prose is the papers/ markdown, code is the local CPython stdlib; both ASCII-
filtered. Everything is generated offline — no network needed.
"""
from __future__ import annotations

import random
import re
import string
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

MASK = "?"
_NON_ASCII = re.compile(r"[^\x20-\x7e\n]")


@dataclass(frozen=True)
class Sample:
    text: str  # what the informed model (A) reads
    blind: str  # same length; stored values masked (what B reads in the blindfold test)
    answer_chars: tuple[int, ...] = field(default_factory=tuple)  # char idx of ask-section digits
    kind: str = "recall"  # "recall" | "prose" | "code"


def ascii_clean(text: str) -> str:
    return _NON_ASCII.sub("", text.replace("\t", "    ").replace("\r", ""))


def recall_doc(rng: random.Random, n_pairs: tuple[int, int] = (3, 6), n_ask: tuple[int, int] = (2, 3)):
    """One recall doc -> (clean, blind, answer char offsets within the doc)."""
    keys: list[str] = []
    n = rng.randint(*n_pairs)
    while len(keys) < n:
        k = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 5)))
        if k not in keys:
            keys.append(k)
    vals = {k: "".join(rng.choices(string.digits, k=4)) for k in keys}
    clean, blind = "store: ", "store: "
    for i, k in enumerate(keys):
        sep = ", " if i < len(keys) - 1 else "; "
        clean += f"{k}={vals[k]}{sep}"
        blind += f"{k}={MASK * 4}{sep}"
    clean += "ask: "
    blind += "ask: "
    answers: list[int] = []
    asked = rng.sample(keys, min(len(keys), rng.randint(*n_ask)))
    for i, k in enumerate(asked):
        prefix = f"{k}="
        start = len(clean) + len(prefix)
        answers.extend(range(start, start + 4))
        piece = f"{prefix}{vals[k]}" + (", " if i < len(asked) - 1 else "\n")
        clean += piece
        blind += piece
    return clean, blind, answers


def recall_sample(rng: random.Random, length: int) -> Sample:
    """Concatenate recall docs to exactly ``length`` chars; only complete docs score."""
    text, blind, answers = "", "", []
    while len(text) < length:
        c, b, a = recall_doc(rng)
        if len(text) + len(c) <= length:
            answers.extend(len(text) + i for i in a)
        text, blind = text + c, blind + b
    return Sample(text=text[:length], blind=blind[:length], answer_chars=tuple(answers))


def load_prose(papers_dir: Path) -> str:
    return "\n".join(ascii_clean(p.read_text(errors="ignore")) for p in sorted(papers_dir.glob("*.md")))


def load_code(max_chars: int = 6_000_000) -> str:
    root = Path(sysconfig.get_paths()["stdlib"])
    out, total = [], 0
    for p in sorted(root.glob("*.py")) + sorted(root.glob("*/*.py")):
        if "test" in p.parts:
            continue
        t = ascii_clean(p.read_text(errors="ignore"))
        out.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n".join(out)


class MixedSampler:
    """Fixed-length samples drawn from recall / prose / code with given weights."""

    def __init__(self, prose: str, code: str, length: int, p_recall: float = 0.5, seed: int = 0):
        self.prose, self.code, self.length, self.p_recall = prose, code, length, p_recall
        self.rng = random.Random(seed)

    def _window(self, src: str, kind: str) -> Sample:
        i = self.rng.randrange(0, len(src) - self.length)
        t = src[i : i + self.length]
        return Sample(text=t, blind=t, kind=kind)

    def sample(self, kind: str | None = None) -> Sample:
        if kind is None:
            r = self.rng.random()
            kind = "recall" if r < self.p_recall else ("prose" if r < (1 + self.p_recall) / 2 else "code")
        if kind == "recall":
            return recall_sample(self.rng, self.length)
        return self._window(self.prose if kind == "prose" else self.code, kind)
