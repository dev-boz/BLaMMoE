"""Audit M6: the file-level train/val/test split of papers/ and the stdlib, applied
before the BPE fit, so held-out windows are never text anything was fitted on."""
import pytest

from blammoe.toy import setup
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import SPLITS, MixedSampler, load_code, load_prose, split_names
from blammoe.toy.fused_train import general_eval_set
from blammoe.toy.tokenizers import BPETokenizer


def test_split_names_is_a_disjoint_cover_that_keeps_packages_whole():
    names = [f"mod{i}.py" for i in range(193)] + [f"pkg{i % 7}/mod{i}.py" for i in range(70)]
    sp = split_names(names, 0.1, 0.1)
    assert sorted(sum(sp.values(), [])) == sorted(names)
    assert [len({n.split("/")[0] for n in sp[s]}) for s in ("val", "test")] == [20, 20]  # 200 groups
    for g in range(7):
        assert sum(any(n.startswith(f"pkg{g}/") for n in sp[s]) for s in SPLITS) == 1
    assert sp["train"] == [n for n in names if n in set(sp["train"])]  # input order kept
    rev = split_names(names[::-1], 0.1, 0.1)
    assert all(set(rev[s]) == set(sp[s]) for s in SPLITS)


def test_split_names_gives_every_nonzero_share_a_file_and_keeps_some_train():
    sp = split_names(["a.md", "b.md", "c.md"], 0.01, 0.01)
    assert [len(sp[s]) for s in SPLITS] == [1, 1, 1]
    assert split_names(["a.md", "b.md"], 0.0, 0.3)["val"] == []
    with pytest.raises(ValueError):
        split_names(["a.md", "b.md"], 0.01, 0.01)


def _tree(tmp_path):
    root = tmp_path / "stdlib"
    for rel in [f"m{i}.py" for i in range(12)] + [f"pkg/s{i}.py" for i in range(8)] + ["test/t.py"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# FILE {rel}\n" + "x = 1\n" * 50)
    papers = tmp_path / "papers"
    papers.mkdir()
    for i in range(10):
        (papers / f"p{i}.md").write_text(f"PAPER p{i}.md\n" + "words " * 100)
    return root, papers


def _files_in(text, tag):
    return {line.split(" ", 2)[-1] for line in text.splitlines() if line.startswith(tag)}


def test_loaders_split_by_file_and_cap_after_splitting(tmp_path):
    root, papers = _tree(tmp_path)
    code = {s: _files_in(load_code(s, 0.2, 0.2, root=root), "# FILE") for s in SPLITS}
    prose = {s: _files_in(load_prose(papers, s, 0.2, 0.2), "PAPER") for s in SPLITS}
    for got, n_files in ((code, 20), (prose, 10)):
        assert all(got[s] for s in SPLITS)
        assert sum(len(got[s]) for s in SPLITS) == n_files  # disjoint and complete
    assert "test/t.py" not in set().union(*code.values())
    capped = _files_in(load_code("train", 0.2, 0.2, max_chars=1, root=root), "# FILE")
    assert len(capped) == 1 and capped <= code["train"]
    with pytest.raises(ValueError):
        load_prose(papers, "all", 0.2, 0.2)


def test_bpe_fit_sees_only_train_text(tmp_path, monkeypatch):
    cfg = ToyPOCConfig(out_dir=str(tmp_path))
    asked, fitted = [], []
    real_texts, real_train = setup.texts, BPETokenizer.train.__func__
    monkeypatch.setattr(setup, "texts", lambda c, split: asked.append(split) or real_texts(c, split))
    monkeypatch.setattr(BPETokenizer, "train",
                        classmethod(lambda cls, t, v: fitted.extend(t) or real_train(cls, t[:200], v)))
    setup.tokenizer(cfg.model_b, cfg)
    assert set(asked) == {"train"} and len(fitted) == 4000
    prose, code = real_texts(cfg, "train")
    ordinary = [t for t in fitted if not t.startswith("store: ")]
    assert ordinary and all(t in prose or t in code for t in ordinary)
    assert list(tmp_path.glob("bpe*.json")) == [tmp_path / "bpe1024_train_v0.1_t0.1.json"]


def test_held_out_windows_are_almost_never_train_text():
    """Real corpus. Not zero: license headers and code copied between stdlib modules
    survive any file split (~0.5% of windows). A per-file split gave ~5% (codec tables)."""
    cfg = ToyPOCConfig()
    train = setup.texts(cfg, "train")
    for split in ("val", "test"):
        held = setup.texts(cfg, split)
        assert all(held) and all(len(h) < len(t) for h, t in zip(held, train))
        docs = general_eval_set(*held, cfg.seq_chars, n=1000, seed=1)
        seen = sum(d.text in train[0] or d.text in train[1] for d in docs)
        assert seen <= 20, (split, seen)


# --------------------------------------------------------------------------- per-model data mix
_PROSE = "".join(f"prose sentence number {i} about models. " for i in range(300))
_CODE = "".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(300))


def _fingerprint(**kw) -> str:
    import hashlib
    h = hashlib.sha256()
    for p_recall in (0.5, 0.0, 0.3):
        s = MixedSampler(_PROSE, _CODE, 64, p_recall=p_recall, seed=7, **kw)
        for _ in range(300):
            x = s.sample()
            h.update((x.kind + x.text + x.blind + str(x.answer_chars)).encode())
    return h.hexdigest()


def test_p_prose_half_draws_exactly_as_before_the_field_existed():
    # taken from MixedSampler at a7d9a3f (before p_prose), on the same inputs
    old = "f193b5fcbae7708ebeb4e89253796d2652e360b6687cbdfd9424abbad42e2d29"
    assert _fingerprint() == old and _fingerprint(p_prose=0.5) == old


def test_p_prose_sets_the_prose_share_of_non_recall_windows():
    for p_prose, lo, hi in ((0.1, 0.07, 0.13), (0.9, 0.87, 0.93), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)):
        s = MixedSampler(_PROSE, _CODE, 64, p_recall=0.5, seed=3, p_prose=p_prose)
        kinds = [s.sample().kind for _ in range(4000)]
        ordinary = [k for k in kinds if k != "recall"]
        assert 0.47 < 1 - len(ordinary) / len(kinds) < 0.53  # the recall share is untouched
        assert lo <= ordinary.count("prose") / len(ordinary) <= hi


def test_a_bpe_is_fitted_on_its_own_models_mix(tmp_path, monkeypatch):
    cfg = ToyPOCConfig(out_dir=str(tmp_path))
    fitted, real_train = [], BPETokenizer.train.__func__
    monkeypatch.setattr(BPETokenizer, "train",
                        classmethod(lambda cls, t, v: fitted.extend(t) or real_train(cls, t[:200], v)))
    cfg.model_b.p_prose = 0.9
    setup.tokenizer(cfg.model_b, cfg)
    assert setup.bpe_name(cfg.model_b, cfg) == "bpe1024_train_v0.1_t0.1_p0.9.json"
    assert (tmp_path / "bpe1024_train_v0.1_t0.1_p0.9.json").exists()
    prose, _ = setup.texts(cfg, "train")
    ordinary = [t for t in fitted if not t.startswith("store: ")]
    assert 0.85 < sum(t in prose for t in ordinary) / len(ordinary) < 0.95
    cfg.model_b.p_prose = 0.5  # the name every run before p_prose used
    assert setup.bpe_name(cfg.model_b, cfg) == "bpe1024_train_v0.1_t0.1.json"
