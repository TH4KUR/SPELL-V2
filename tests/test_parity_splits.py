"""parity/splits.py + scripts/parity_build_splits.py.

RAVEn's training data is the official LRS3 trainval minus the 1,200 validation
clips of AV-HuBERT's ``lrs3-valid.id``; the test set is the official 1,321-clip
test set. Expected: train 30,782 utts / 29.25 h, val 1,200 / 1.15 h, test 1,321 /
0.85 h. The id file is under a custom Meta licence and is NEVER copied into the
repo: the builder takes ``--valid-id-file`` and verifies its sha256.

Real-data assertions run against the tracked data_index.parquet; the ones that need
the licensed id file read its path from env ``LRS3_VALID_ID_FILE`` and skip without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import parity_build_splits as CLI  # noqa: E402
from dataset import filter_records, load_id_list, load_records  # noqa: E402
from parity import splits as S  # noqa: E402
from parity import text as T  # noqa: E402

VALID_FILE = os.environ.get("LRS3_VALID_ID_FILE")
needs_valid_file = pytest.mark.skipif(
    not VALID_FILE or not Path(VALID_FILE).is_file(),
    reason="set LRS3_VALID_ID_FILE to AV-HuBERT's lrs3-valid.id to run")


# ------------------------------------------------------------------ fixtures --

def _index(n_videos=6, per_video=4, n_test=5) -> pd.DataFrame:
    rows = []
    for v in range(n_videos):
        for k in range(per_video):
            rows.append({"utterance_id": f"vid{v}/{50000 + k}", "split": "trainval",
                         "video_id": f"vid{v}", "stem": str(50000 + k),
                         "duration_s": 3600.0 / (n_videos * per_video),
                         "text_raw": f"TRAIN WORD{v} NUMBER{k}"})
    for t in range(n_test):
        rows.append({"utterance_id": str(t), "split": "test", "video_id": "", "stem": str(t),
                     "duration_s": 1800.0 / n_test, "text_raw": f"test sentence {t}"})
    return pd.DataFrame(rows)


def _valid(ids):
    return [f"trainval/{i}" for i in ids]


TINY_EXPECT = S.Expectations(n_train=21, n_val=3, n_test=5, train_hours=0.875, val_hours=0.125,
                             test_hours=0.5, hours_tolerance=1e-9)
TINY_VALID = ["vid0/50001", "vid2/50003", "vid5/50000"]
TINY_RAVEN = [f"TEST SENTENCE {t}" for t in range(5)]


def _build(index=None, valid=None, **kw):
    kw.setdefault("expected", TINY_EXPECT)
    kw.setdefault("raven_test_texts", TINY_RAVEN)
    return S.build_splits(_index() if index is None else index,
                          TINY_VALID if valid is None else valid, **kw)


# ------------------------------------------------------------- build_splits --

def test_split_membership_and_sorted_outputs():
    r = _build()
    assert len(r.train_ids) == 21 and len(r.val_ids) == 3 and len(r.test_ids) == 5
    assert r.val_ids == sorted(TINY_VALID)
    assert r.test_ids == sorted(str(t) for t in range(5))
    assert not set(r.train_ids) & set(r.val_ids)
    assert all("/" in i for i in r.train_ids + r.val_ids) and all("/" not in i for i in r.test_ids)
    assert r.train_ids == sorted(r.train_ids)


def test_stats_report_hours_and_video_overlap():
    r = _build()
    assert r.stats["hours"]["train"] == pytest.approx(0.875)
    assert r.stats["hours"]["val"] == pytest.approx(0.125)
    assert r.stats["hours"]["test"] == pytest.approx(0.5)
    assert r.stats["val_videos"] == {"n_videos": 3, "n_also_in_train": 3}   # NOT video-disjoint


def test_a_valid_id_missing_from_the_index_is_fatal():
    with pytest.raises(S.SplitError, match="not in the index"):
        _build(valid=TINY_VALID + ["vid9/99999"], expected=None)


def test_a_valid_id_that_is_a_test_utterance_is_fatal():
    with pytest.raises(S.SplitError, match="not in the index|trainval"):
        _build(valid=TINY_VALID + ["3"], expected=None)


def test_duplicate_valid_ids_are_fatal():
    with pytest.raises(S.SplitError, match="duplicate"):
        _build(valid=TINY_VALID + [TINY_VALID[0]], expected=None)


def test_wrong_counts_are_fatal():
    with pytest.raises(S.SplitError, match="train.*count"):
        _build(expected=S.Expectations(n_train=22, n_val=3, n_test=5, train_hours=0.875,
                                       val_hours=0.125, test_hours=0.5, hours_tolerance=1e-9))


def test_wrong_hours_are_fatal():
    with pytest.raises(S.SplitError, match="hours"):
        _build(expected=S.Expectations(n_train=21, n_val=3, n_test=5, train_hours=29.25,
                                       val_hours=0.125, test_hours=0.5, hours_tolerance=0.005))


def test_hours_within_rounding_are_accepted():
    ok = S.Expectations(n_train=21, n_val=3, n_test=5, train_hours=0.876, val_hours=0.126,
                        test_hours=0.499, hours_tolerance=0.005)
    assert _build(expected=ok).stats["counts"]["train"] == 21


def test_duplicate_index_ids_are_fatal():
    idx = pd.concat([_index(), _index().iloc[[0]]], ignore_index=True)
    with pytest.raises(S.SplitError, match="unique"):
        _build(index=idx, expected=None)


def test_empty_reference_text_is_fatal():
    idx = _index()
    idx.loc[idx.utterance_id == "vid1/50001", "text_raw"] = "   "
    with pytest.raises(S.SplitError, match="empty"):
        _build(index=idx, expected=None)


def test_test_texts_must_equal_the_raven_csv_multiset_case_insensitively():
    assert _build().stats["test_text_check"] == {"n": 5, "equal_to_raven_csv": True}
    with pytest.raises(S.SplitError, match="RAVEn"):
        _build(raven_test_texts=TINY_RAVEN[:-1] + ["SOMETHING ELSE"])
    with pytest.raises(S.SplitError, match="RAVEn"):
        _build(raven_test_texts=TINY_RAVEN + ["EXTRA"])


def test_duplicate_test_sentences_are_compared_as_a_multiset():
    idx = _index()
    idx.loc[idx.utterance_id == "1", "text_raw"] = "test sentence 0"
    raven = ["TEST SENTENCE 0", "TEST SENTENCE 0", "TEST SENTENCE 2", "TEST SENTENCE 3", "TEST SENTENCE 4"]
    assert _build(index=idx, raven_test_texts=raven).stats["test_text_check"]["equal_to_raven_csv"]
    with pytest.raises(S.SplitError, match="RAVEn"):
        _build(index=idx, raven_test_texts=TINY_RAVEN)


# --------------------------------------------------------------- valid id file --

def _write_valid(tmp_path, lines):
    p = tmp_path / "lrs3-valid.id"
    p.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return p


def test_read_valid_id_file_strips_the_trainval_prefix_and_checks_sha(tmp_path):
    p = _write_valid(tmp_path, _valid(TINY_VALID))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert S.read_valid_id_file(p, expect_sha256=sha) == TINY_VALID
    with pytest.raises(S.SplitError, match="sha256"):
        S.read_valid_id_file(p, expect_sha256="0" * 64)


def test_read_valid_id_file_defaults_to_the_pinned_sha(tmp_path):
    p = _write_valid(tmp_path, _valid(TINY_VALID))
    with pytest.raises(S.SplitError, match="sha256"):
        S.read_valid_id_file(p)
    assert S.VALID_ID_SHA256 == "4e24da968352dedfe11dd5bbd3efa77c96dd26e062019acf7a00c9264aab884a"


def test_read_valid_id_file_rejects_lines_outside_trainval(tmp_path):
    p = _write_valid(tmp_path, ["test/abc/0001"])
    with pytest.raises(S.SplitError, match="trainval/"):
        S.read_valid_id_file(p, expect_sha256=hashlib.sha256(p.read_bytes()).hexdigest())


# ------------------------------------------------------------------- writing --

def test_write_splits_files_roundtrip_through_the_repo_loaders(tmp_path):
    r = _build()
    prov = S.write_splits(r, tmp_path, provenance_sources={"note": "synthetic"})
    for name, ids in (("parity_train_ids.txt", r.train_ids), ("parity_val_ids.txt", r.val_ids),
                      ("parity_test_ids.txt", r.test_ids)):
        assert load_id_list(tmp_path / name) == set(ids)
        assert (tmp_path / name).read_text().splitlines() == ids        # one id per line, sorted
    assert prov["files"]["parity_train_ids.txt"]["n"] == 21
    on_disk = json.loads((tmp_path / "provenance.json").read_text())
    assert on_disk == prov and on_disk["counts"] == {"train": 21, "val": 3, "test": 5}


def test_written_files_are_byte_stable(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    S.write_splits(_build(), a, provenance_sources={"x": 1})
    S.write_splits(_build(), b, provenance_sources={"x": 1})
    for name in ("parity_train_ids.txt", "parity_val_ids.txt", "parity_test_ids.txt", "provenance.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_verify_split_files_detects_tampering(tmp_path):
    S.write_splits(_build(), tmp_path, provenance_sources={})
    S.verify_split_files(tmp_path)
    (tmp_path / "parity_val_ids.txt").write_text("vid0/50001\n")
    with pytest.raises(S.SplitError, match="parity_val_ids.txt"):
        S.verify_split_files(tmp_path)


def test_verify_split_files_checks_the_index_it_was_built_from(tmp_path):
    idx = tmp_path / "idx.parquet"
    _index().to_parquet(idx)
    out = tmp_path / "splits"
    S.write_splits(_build(), out, provenance_sources={
        "data_index": {"path": "idx.parquet", "sha256": S.sha256_bytes(idx.read_bytes())}})
    S.verify_split_files(out, index_path=idx)
    other = tmp_path / "other.parquet"
    _index(n_test=6).to_parquet(other)
    with pytest.raises(S.SplitError, match="INDEX IS TRUTH"):
        S.verify_split_files(out, index_path=other)


def test_verify_split_files_requires_the_provenance_file(tmp_path):
    with pytest.raises(S.SplitError, match="provenance"):
        S.verify_split_files(tmp_path)


# ------------------------------------------------------------------------ CLI --

def test_cli_refuses_a_valid_id_file_with_the_wrong_sha256(tmp_path, capsys):
    p = _write_valid(tmp_path, _valid(TINY_VALID))
    rc = CLI.main(["--valid-id-file", str(p), "--out-dir", str(tmp_path / "out")])
    assert rc == 3 and "sha256" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_refuses_a_missing_valid_id_file(tmp_path, capsys):
    assert CLI.main(["--valid-id-file", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "o")]) == 2


# ------------------------------------------------------------ real tracked data --

@pytest.fixture(scope="module")
def real_index() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data_index.parquet")


def test_real_index_test_texts_equal_ravens_test_csv(real_index):
    """All 1,321 official-test references equal RAVEn's csv labels decoded with its
    own units list (multiset, case-insensitive; 8 sentences occur more than once)."""
    ours = Counter(T.reference_text(t) for t in real_index.loc[real_index.split == "test", "text_raw"])
    theirs = Counter(row.text.upper() for row in T.read_raven_test_csv())
    assert sum(ours.values()) == sum(theirs.values()) == 1321
    assert ours == theirs
    assert sum(1 for c in ours.values() if c > 1) == 8
    assert sum(1 for t in ours.elements() if any(ch.isdigit() for ch in t)) == 24


def test_real_index_test_split_has_expected_size_and_hours(real_index):
    t = real_index[real_index.split == "test"]
    assert len(t) == 1321 and t.duration_s.sum() / 3600 == pytest.approx(0.85, abs=0.005)
    assert (t.text_raw.str.strip() != "").all()


@needs_valid_file
def test_real_valid_id_file_gives_the_expected_splits_and_loader_roundtrip(real_index, tmp_path):
    valid = S.read_valid_id_file(VALID_FILE)
    assert len(valid) == 1200
    r = S.build_splits(real_index, valid, expected=S.PINNED_EXPECTATIONS,
                       raven_test_texts=[row.text for row in T.read_raven_test_csv()])
    assert (len(r.train_ids), len(r.val_ids), len(r.test_ids)) == (30782, 1200, 1321)
    assert r.stats["hours"]["train"] == pytest.approx(29.25, abs=0.005)
    assert r.stats["hours"]["val"] == pytest.approx(1.15, abs=0.005)
    assert r.stats["hours"]["test"] == pytest.approx(0.85, abs=0.005)
    assert r.stats["val_videos"] == {"n_videos": 913, "n_also_in_train": 889}
    S.write_splits(r, tmp_path, provenance_sources=S.pinned_sources(VALID_FILE))
    recs = load_records(PROJECT_ROOT / "data_index.parquet")
    for name, n in (("parity_train_ids.txt", 30782), ("parity_val_ids.txt", 1200),
                    ("parity_test_ids.txt", 1321)):
        got = filter_records(recs, load_id_list(tmp_path / name), strict=True)
        assert len(got) == n
    assert {x.split for x in filter_records(recs, load_id_list(tmp_path / "parity_test_ids.txt"))} == {"test"}
    assert {x.split for x in filter_records(recs, load_id_list(tmp_path / "parity_val_ids.txt"))} == {"trainval"}


@needs_valid_file
def test_real_cli_end_to_end_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert CLI.main(["--valid-id-file", VALID_FILE, "--out-dir", str(a)]) == 0
    assert CLI.main(["--valid-id-file", VALID_FILE, "--out-dir", str(b)]) == 0
    for name in ("parity_train_ids.txt", "parity_val_ids.txt", "parity_test_ids.txt", "provenance.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    prov = json.loads((a / "provenance.json").read_text())
    vf = prov["sources"]["valid_id_file"]
    assert vf["sha256"] == S.VALID_ID_SHA256 and vf["commit"] == "b4fa4af9b437757b8649dc8a226e6d4d8e780f6f"
    assert "NOT copied" in vf["licence_note"]
    S.verify_split_files(a)
