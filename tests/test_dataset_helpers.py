"""dataset.py helper round: load_id_list parsing, filter_records strictness and
rebase_records path retargeting (repo-relative → $SPELL_DATA_ROOT)."""

import dataclasses

import pytest

from dataset import UtteranceRecord, filter_records, load_id_list, rebase_records


def make_rec(uid: str, tokens_path: str | None = None,
             txt_path: str | None = None, audio_path: str | None = None) -> UtteranceRecord:
    return UtteranceRecord(
        utterance_id=uid,
        split="trainval",
        video_id=uid.split("/")[0],
        stem=uid.split("/")[1] if "/" in uid else uid,
        tokens_path=tokens_path or f"datasets/LRS3/trainval/{uid}.npz",
        audio_path=audio_path,
        audio_kind="mp4" if audio_path else None,
        txt_path=txt_path or f"datasets/LRS3/trainval/{uid}.txt",
        n_tokens=100,
        duration_s=4.0,
        conf=90,
        text_raw="hello world",
        text_norm="hello world",
        n_chars_norm=len("hello world"),
    )


# ------------------------------------------------------------------ load_id_list

def test_load_id_list_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text(
        "# frozen split manifest\n"
        "vid/utt001\n"
        "\n"
        "   \n"
        "vid/utt002\n"
        "vid/utt001\n"                # duplicate → deduped by set semantics
        "  # indented comment is still a comment (stripped first)\n",
        encoding="utf-8",
    )
    assert load_id_list(f) == {"vid/utt001", "vid/utt002"}


def test_load_id_list_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    assert load_id_list(f) == set()


# --------------------------------------------------------------- filter_records

def test_filter_records_keeps_order_and_subset():
    recs = [make_rec(f"a{i}") for i in range(5)]
    picked = filter_records(recs, {"a3", "a0"})
    assert [r.utterance_id for r in picked] == ["a0", "a3"]      # record order preserved


def test_filter_records_strict_raises_on_missing_ids():
    recs = [make_rec("a0"), make_rec("a1")]
    with pytest.raises(ValueError) as excinfo:
        filter_records(recs, {"a0", "ghost1", "ghost2", "g3", "g4", "g5"})
    msg = str(excinfo.value)
    assert "5 requested IDs absent" in msg            # 5 missing → count + preview
    assert "ghost1" in msg


def test_filter_records_non_strict_silently_intersects():
    recs = [make_rec("a0"), make_rec("a1")]
    picked = filter_records(recs, {"a1", "ghost"}, strict=False)
    assert [r.utterance_id for r in picked] == ["a1"]


def test_filter_records_empty_request_yields_empty():
    recs = [make_rec("a0")]
    assert filter_records(recs, set()) == []


# ---------------------------------------------------------------- rebase_records

def _rebase_target_fields(r):
    return r.tokens_path, r.txt_path, r.audio_path


def test_rebase_none_returns_input_unchanged():
    recs = [make_rec("v/u1")]
    out = rebase_records(recs, None)
    assert out is recs                                # repo-local runs use paths as-is


def test_rebase_strips_two_component_prefix_and_joins_root():
    recs = [make_rec("v/u1", audio_path="datasets/LRS3/trainval/v/u1.mp4")]
    rebased = rebase_records(recs, "/data/LRS3_staged")
    tp, xp, ap = _rebase_target_fields(rebased[0])
    assert tp == "/data/LRS3_staged/trainval/v/u1.npz"
    assert xp == "/data/LRS3_staged/trainval/v/u1.txt"
    assert ap == "/data/LRS3_staged/trainval/v/u1.mp4"


def test_rebase_preserves_none_paths():
    rec = dataclasses.replace(make_rec("v/u2"), txt_path=None, audio_path=None,
                              audio_kind=None)
    rebased = rebase_records([rec], "/data")
    assert rebased[0].txt_path is None
    assert rebased[0].audio_path is None
    assert rebased[0].tokens_path == "/data/trainval/v/u2.npz"


def test_rebase_leaves_short_paths_whole_after_join():
    """Only a verbatim 'datasets/LRS3' prefix is stripped; anything else is joined
    under the root untouched."""
    rec = dataclasses.replace(make_rec("v/u3"),
                              tokens_path="trainval/v/u3.npz")
    rebased = rebase_records([rec], "/data")
    assert rebased[0].tokens_path == "/data/trainval/v/u3.npz"


def test_rebase_returns_new_objects_originals_unmutated():
    recs = [make_rec("v/u4")]
    original_tokens_path = recs[0].tokens_path
    rebase_records(recs, "/somewhere")
    assert recs[0].tokens_path == original_tokens_path
