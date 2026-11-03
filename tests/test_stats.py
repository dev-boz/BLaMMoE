import numpy as np

from blammoe.stats import bootstrap_ci, paired_bootstrap_diff, seed_summary


def test_bootstrap_ci_brackets_pooled_accuracy_and_shrinks_with_data():
    rng = np.random.default_rng(1)
    small = [[int(rng.binomial(10, 0.7)), 10] for _ in range(10)]
    big = [[int(rng.binomial(10, 0.7)), 10] for _ in range(400)]
    s, b = bootstrap_ci(small), bootstrap_ci(big)
    assert s["lo"] <= s["acc"] <= s["hi"] and b["lo"] <= b["acc"] <= b["hi"]
    assert (b["hi"] - b["lo"]) < (s["hi"] - s["lo"])
    assert abs(b["acc"] - 0.7) < 0.03


def test_paired_diff_detects_real_gap_and_not_a_null_one():
    rng = np.random.default_rng(2)
    base = [[int(rng.binomial(20, 0.1)), 20] for _ in range(80)]
    good = [[int(rng.binomial(20, 0.9)), 20] for _ in range(80)]
    d = paired_bootstrap_diff(good, base)
    assert d["lo"] > 0.7
    same = paired_bootstrap_diff(base, base)
    assert same["diff"] == 0 and same["lo"] == same["hi"] == 0


def test_seed_summary():
    s = seed_summary([0.9, 0.95, 1.0])
    assert abs(s["mean"] - 0.95) < 1e-9 and abs(s["sd"] - 0.05) < 1e-9 and s["n_seeds"] == 3


def test_bootstrap_drops_answerless_resamples_instead_of_scoring_them_zero():
    """[[0,0],[1,1]]: every resample with answers is 100% correct. 0/0 used to count as 0,
    giving a CI of [0, 1]."""
    r = bootstrap_ci([[0, 0], [1, 1]])
    assert r["acc"] == r["lo"] == r["hi"] == 1.0
    assert 0 < r["n_dropped"] < 2000  # ~1/4 of resamples draw only the empty document
    d = paired_bootstrap_diff([[0, 0], [1, 1]], [[0, 0], [0, 1]])
    assert d["diff"] == d["lo"] == d["hi"] == 1.0
    empty = bootstrap_ci([[0, 0]])
    assert np.isnan(empty["acc"]) and np.isnan(empty["lo"]) and empty["n_dropped"] == 2000
