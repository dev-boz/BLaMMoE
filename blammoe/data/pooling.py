"""Character-span pooling for tokenizer mismatch (WP2; spec 1.3 item 3).

The two models tokenize the same text differently, so token index i in model A
has no counterpart in model B. Instead we pair up *character windows*: for a
window [s, e) each model's states are mean-pooled over its tokens that lie fully
inside the window, giving one approximately aligned (h_A, h_B) pair per window.

"Approximately" is the catch: the tokens fully inside [s, e) usually cover a
slightly smaller span [c_s, c_e) than the window, and the shortfall differs by
tokenizer. The spec says to discard near-boundary spans; concretely, a pair is
kept only if, for BOTH models, the shortfall at each edge (c_s - s and e - c_e)
is at most ``boundary_margin`` chars. ``boundary_margin=0`` therefore keeps only
windows whose edges fall exactly on a token boundary shared by both tokenizers.

Offsets are the per-token ``(start, end)`` char spans a HF fast tokenizer gives
with ``return_offsets_mapping=True``; zero-width spans (special tokens such as
BOS, reported as (0, 0)) are ignored. Pure index logic — no model needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blammoe.config import PoolingConfig


@dataclass(frozen=True)
class PooledPairs:
    h_a: np.ndarray  # [m, d_a]
    h_b: np.ndarray  # [m, d_b]
    windows: np.ndarray  # [m, 2] (start, end) char span of each kept pair


def sliding_windows(text_len: int, span_chars: int, stride: int | None = None) -> np.ndarray:
    """All windows [s, s + span_chars) that fit in the text; stride defaults to span."""
    stride = stride or span_chars
    if span_chars <= 0 or stride <= 0:
        raise ValueError("span_chars and stride must be positive")
    starts = np.arange(0, text_len - span_chars + 1, stride)
    return np.stack([starts, starts + span_chars], axis=1).reshape(-1, 2)


def _pool_one(
    offsets: np.ndarray, states: np.ndarray, s: int, e: int, margin: int
) -> np.ndarray | None:
    starts, ends = offsets[:, 0], offsets[:, 1]
    inside = (starts >= s) & (ends <= e) & (ends > starts)
    if not inside.any():
        return None
    if starts[inside].min() - s > margin or e - ends[inside].max() > margin:
        return None
    return states[inside].mean(axis=0)


def pool_char_spans(
    offsets_a: np.ndarray,
    states_a: np.ndarray,
    offsets_b: np.ndarray,
    states_b: np.ndarray,
    windows: np.ndarray,
    boundary_margin: int,
) -> PooledPairs:
    """Pool both models' token states over each window; drop misaligned windows."""
    offsets_a, offsets_b = np.asarray(offsets_a), np.asarray(offsets_b)
    states_a, states_b = np.asarray(states_a), np.asarray(states_b)
    for name, off, st in (("a", offsets_a, states_a), ("b", offsets_b, states_b)):
        if off.ndim != 2 or off.shape[1] != 2 or st.ndim != 2 or off.shape[0] != st.shape[0]:
            raise ValueError(f"model {name}: need offsets [T, 2] and states [T, d], "
                             f"got {off.shape} and {st.shape}")
    if boundary_margin < 0:
        raise ValueError("boundary_margin must be >= 0")

    rows_a, rows_b, kept = [], [], []
    for s, e in np.asarray(windows).reshape(-1, 2):
        pa = _pool_one(offsets_a, states_a, s, e, boundary_margin)
        pb = _pool_one(offsets_b, states_b, s, e, boundary_margin) if pa is not None else None
        if pb is None:
            continue
        rows_a.append(pa)
        rows_b.append(pb)
        kept.append((s, e))

    d_a, d_b = states_a.shape[1], states_b.shape[1]
    return PooledPairs(
        h_a=np.stack(rows_a) if rows_a else np.empty((0, d_a), dtype=states_a.dtype),
        h_b=np.stack(rows_b) if rows_b else np.empty((0, d_b), dtype=states_b.dtype),
        windows=np.asarray(kept, dtype=np.int64).reshape(-1, 2),
    )


def pool_document(
    offsets_a: np.ndarray,
    states_a: np.ndarray,
    offsets_b: np.ndarray,
    states_b: np.ndarray,
    text_len: int,
    cfg: PoolingConfig,
    stride: int | None = None,
) -> PooledPairs:
    """Config-driven entry point: tile one document with ``cfg.span_chars`` windows."""
    if cfg.method != "char_span":
        raise ValueError(f"unsupported pooling.method {cfg.method!r}")
    windows = sliding_windows(text_len, cfg.span_chars, stride)
    return pool_char_spans(offsets_a, states_a, offsets_b, states_b, windows, cfg.boundary_margin)
