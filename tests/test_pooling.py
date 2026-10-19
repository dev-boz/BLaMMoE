"""WP2 pooling acceptance: window math and boundary discard on synthetic
tokenizations — no model or tokenizer needed (spec 1.3 item 3)."""
import numpy as np
import pytest

from blammoe.config import PoolingConfig
from blammoe.data.pooling import pool_char_spans, pool_document, sliding_windows


def fixed_width_tokens(text_len: int, width: int) -> np.ndarray:
    starts = np.arange(0, text_len, width)
    return np.stack([starts, np.minimum(starts + width, text_len)], axis=1)


def states_for(offsets: np.ndarray, d: int) -> np.ndarray:
    """State of each token = its start char repeated, so pooled values are checkable."""
    return np.repeat(offsets[:, :1].astype(float), d, axis=1)


# model A tokenizes every 4 chars, model B every 5 -> boundaries shared at multiples of 20
OFF_A = fixed_width_tokens(60, 4)
OFF_B = fixed_width_tokens(60, 5)
ST_A = states_for(OFF_A, 3)
ST_B = states_for(OFF_B, 2)


def test_sliding_windows_math():
    w = sliding_windows(100, 20, stride=10)
    assert w.shape == (9, 2)
    assert w[0].tolist() == [0, 20] and w[-1].tolist() == [80, 100]
    assert (w[:, 1] - w[:, 0] == 20).all()
    assert sliding_windows(100, 20).shape == (5, 2)  # stride defaults to span
    assert sliding_windows(10, 20).shape == (0, 2)  # text shorter than one window


def test_shared_boundary_window_pools_exactly():
    out = pool_char_spans(OFF_A, ST_A, OFF_B, ST_B, [[0, 20]], boundary_margin=0)
    assert out.windows.tolist() == [[0, 20]]
    assert np.allclose(out.h_a, np.mean([0, 4, 8, 12, 16]))  # A tokens 0..4
    assert np.allclose(out.h_b, np.mean([0, 5, 10, 15]))  # B tokens 0..3
    assert out.h_a.shape == (1, 3) and out.h_b.shape == (1, 2)


def test_boundary_margin_discards_misaligned_windows():
    # window [2, 22): A covers [4, 20) (gaps 2, 2); B covers [5, 20) (gaps 3, 2)
    window = [[2, 22]]
    assert pool_char_spans(OFF_A, ST_A, OFF_B, ST_B, window, boundary_margin=3).windows.shape == (1, 2)
    assert pool_char_spans(OFF_A, ST_A, OFF_B, ST_B, window, boundary_margin=2).windows.shape == (0, 2)
    assert pool_char_spans(OFF_A, ST_A, OFF_B, ST_B, window, boundary_margin=0).windows.shape == (0, 2)


def test_window_with_no_whole_token_is_dropped():
    long_tokens = np.array([[0, 30], [30, 60]])
    out = pool_char_spans(
        long_tokens, states_for(long_tokens, 3), OFF_B, ST_B, [[5, 25]], boundary_margin=100
    )
    assert out.h_a.shape == (0, 3) and out.h_b.shape == (0, 2)


def test_special_tokens_with_zero_width_offsets_are_ignored():
    off = np.vstack([[[0, 0]], OFF_A])  # BOS reported as (0, 0)
    st = np.vstack([np.full((1, 3), 999.0), ST_A])
    out = pool_char_spans(off, st, OFF_B, ST_B, [[0, 20]], boundary_margin=0)
    assert np.allclose(out.h_a, np.mean([0, 4, 8, 12, 16]))


def test_pool_document_from_config_keeps_pairs_row_aligned():
    out = pool_document(OFF_A, ST_A, OFF_B, ST_B, text_len=60, cfg=PoolingConfig(), stride=1)
    # every kept window satisfies the margin in both models; halves stay paired
    assert out.h_a.shape[0] == out.h_b.shape[0] == out.windows.shape[0] > 0
    assert (out.windows[:, 1] - out.windows[:, 0] == 20).all()
    exact = pool_document(OFF_A, ST_A, OFF_B, ST_B, 60, PoolingConfig(boundary_margin=0), stride=1)
    assert exact.windows.tolist() == [[0, 20], [20, 40], [40, 60]]  # only shared boundaries


def test_mismatched_offsets_and_states_rejected():
    with pytest.raises(ValueError):
        pool_char_spans(OFF_A, ST_A[:-1], OFF_B, ST_B, [[0, 20]], boundary_margin=0)
    with pytest.raises(ValueError):
        pool_document(OFF_A, ST_A, OFF_B, ST_B, 60, PoolingConfig(method="token_index"))
