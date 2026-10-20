"""Audit M6: the file-level train/val/test split of papers/ and the stdlib, applied
before the BPE fit, so held-out windows are never text anything was fitted on."""
import pytest

from blammoe.toy import setup
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import SPLITS, load_code, load_prose, split_names
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
