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

Both are split by file, whole stdlib packages together, into train / val / test (audit
M6) before anything is fitted on them: tokenizers and models train on ``train``, choices
(expert selection) are made on ``val``, reported ordinary-text numbers come from
``test``. Only text copied between files (license headers, shared code) can still
appear on both sides (see ``split_names``).
"""
from __future__ import annotations

import hashlib
import random
import re
import string
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

MASK = "?"
SPLITS = ("train", "val", "test")
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


def split_names(names: list[str], val_share: float, test_share: float) -> dict[str, list[str]]:
    """File-level split (audit M6), grouped by top-level package: a stdlib package's files
    share generated tables and boilerplate (``encodings/cp*.py`` repeat whole codec maps), so
    a per-file split put ~5% of held-out windows verbatim in train; whole packages bring that
    under 1%, and what is left is copied license headers and code (``bz2`` / ``lzma``).
    Groups are ordered by sha256 of their name, so the split depends only on the set of
    names, not on the machine or filesystem order; test takes the first ``test_share`` of
    the groups, val the next ``val_share``, train the rest. A non-zero share gets at least
    one group (papers/ has only ~33 files). Each split keeps the input order of its names."""
    groups = sorted({n.split("/")[0] for n in names}, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    n_test, n_val = (max(1, round(x * len(groups))) if x > 0 else 0 for x in (test_share, val_share))
    if n_test + n_val >= len(groups):
        raise ValueError(f"val {val_share} + test {test_share} leave no train groups out of {len(groups)}")
    split_of = {g: "test" for g in groups[:n_test]} | {g: "val" for g in groups[n_test : n_test + n_val]}
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    for n in names:
        out[split_of.get(n.split("/")[0], "train")].append(n)
    return out


def _keep(names: list[str], split: str, val_share: float, test_share: float) -> set[str]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return set(split_names(names, val_share, test_share)[split])


def load_prose(papers_dir: Path, split: str, val_share: float, test_share: float) -> str:
    paths = sorted(papers_dir.glob("*.md"))
    keep = _keep([p.name for p in paths], split, val_share, test_share)
    return "\n".join(ascii_clean(p.read_text(errors="ignore")) for p in paths if p.name in keep)


def load_code(split: str, val_share: float, test_share: float, max_chars: int = 6_000_000,
              root: Path | None = None) -> str:
    """One split of the stdlib (``root``, default the running Python's), capped at
    ``max_chars`` AFTER splitting so the cap never moves a file between splits."""
    root = Path(sysconfig.get_paths()["stdlib"]) if root is None else root
    rel = {p: p.relative_to(root).as_posix() for p in sorted(root.glob("*.py")) + sorted(root.glob("*/*.py"))}
    rel = {p: r for p, r in rel.items() if "test" not in Path(r).parts}
    keep = _keep(list(rel.values()), split, val_share, test_share)
    out, total = [], 0
    for p, r in rel.items():
        if r not in keep:
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


# --------------------------------------------------------------------------- split-view recall
@dataclass(frozen=True)
class SplitSample:
    """Same text seen two ways: each stored value is masked in exactly ONE view.

    Answers whose value is masked in a model's own view are unknowable to that
    model alone — it needs its partner. ``view(which)`` gives a plain Sample for
    batching; ``partner_answers(which)`` lists the answer chars that model needs
    help with, ``own_answers(which)`` the ones it can read itself.
    """

    text: str
    view_a: str
    view_b: str
    a_blind: tuple[int, ...]  # answer chars masked in A's view (A needs B)
    b_blind: tuple[int, ...]  # answer chars masked in B's view (B needs A)
    kind: str = "split_recall"

    def view(self, which: str) -> Sample:
        t = self.view_a if which == "a" else self.view_b
        return Sample(text=t, blind=t, answer_chars=tuple(sorted(self.a_blind + self.b_blind)), kind=self.kind)

    def partner_answers(self, which: str) -> tuple[int, ...]:
        return self.a_blind if which == "a" else self.b_blind

    def own_answers(self, which: str) -> tuple[int, ...]:
        return self.b_blind if which == "a" else self.a_blind


def split_recall_doc(rng: random.Random, n_pairs: tuple[int, int] = (4, 6), n_ask: tuple[int, int] = (2, 4)):
    """-> (clean, view_a, view_b, a_blind answer chars, b_blind answer chars)."""
    keys: list[str] = []
    n = rng.randint(*n_pairs)
    while len(keys) < n:
        k = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 5)))
        if k not in keys:
            keys.append(k)
    vals = {k: "".join(rng.choices(string.digits, k=4)) for k in keys}
    hidden_from_a = set(rng.sample(keys, len(keys) // 2))  # the rest are hidden from B
    clean = va = vb = "store: "
    for i, k in enumerate(keys):
        sep = ", " if i < len(keys) - 1 else "; "
        clean += f"{k}={vals[k]}{sep}"
        va += f"{k}={MASK * 4 if k in hidden_from_a else vals[k]}{sep}"
        vb += f"{k}={vals[k] if k in hidden_from_a else MASK * 4}{sep}"
    clean += "ask: "
    va += "ask: "
    vb += "ask: "
    a_blind: list[int] = []
    b_blind: list[int] = []
    asked = rng.sample(keys, min(len(keys), rng.randint(*n_ask)))
    for i, k in enumerate(asked):
        start = len(clean) + len(k) + 1
        (a_blind if k in hidden_from_a else b_blind).extend(range(start, start + 4))
        piece = f"{k}={vals[k]}" + (", " if i < len(asked) - 1 else "\n")
        clean, va, vb = clean + piece, va + piece, vb + piece
    return clean, va, vb, a_blind, b_blind


def split_recall_sample(rng: random.Random, length: int) -> SplitSample:
    """Concatenate split-view docs to exactly ``length`` chars; only complete docs score."""
    text = va = vb = ""
    a_blind: list[int] = []
    b_blind: list[int] = []
    while len(text) < length:
        c, x, y, ab, bb = split_recall_doc(rng)
        if len(text) + len(c) <= length:
            a_blind.extend(len(text) + i for i in ab)
            b_blind.extend(len(text) + i for i in bb)
        text, va, vb = text + c, va + x, vb + y
    return SplitSample(text=text[:length], view_a=va[:length], view_b=vb[:length],
                       a_blind=tuple(a_blind), b_blind=tuple(b_blind))
