"""dataset.py helper round: load_id_list parsing, filter_records strictness,
and the TokenDataset path-resolver seam.

(`rebase_records` was DELETED under the DATA LAYOUT LAW, §10 item 4: path
retargeting is not a per-record data munging concern anymore — locations come
from paths.py keyed on (video_id, stem); layout tests live in
tests/test_layout_resolution.py.)"""

import pytest

from dataset import UtteranceRecord, filter_records, load_id_list


def make_rec(uid: str) -> UtteranceRecord:
    return UtteranceRecord(
        utterance_id=uid,
        split="trainval",
        video_id=uid.split("/")[0] if "/" in uid else "",
        stem=uid.split("/")[1] if "/" in uid else uid,
        tokens_path=f"datasets/LRS3/trainval/{uid}.tokens.pt",   # provenance string only
        audio_path=None,
        audio_kind=None,
        txt_path=None,
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


# ------------------------------------------------- TokenDataset resolver seam

def _write_tokens(tmp_path):
    import torch

    p = tmp_path / "fake.pt"
    torch.save(torch.zeros(8, 5), p)
    return p


def test_tokendataset_injected_resolver_wins(tmp_path):
    """path_resolver is the unit-test injection point (§10 item 4): given a
    resolver, __getitem__ loads exactly what it returns."""
    from dataset import TokenDataset

    p = _write_tokens(tmp_path)
    seen = []
    ds = TokenDataset([make_rec("v/u1")],
                      path_resolver=lambda rec: seen.append(rec.utterance_id) or p)
    item = ds[0]
    assert item["n_tokens"] == 5
    assert seen == ["v/u1"]


def test_tokendataset_default_resolver_goes_through_paths_authority(tmp_path, monkeypatch):
    """With NO injected resolver, loads flow through paths.resolve_token_path —
    i.e. the DATA LAYOUT LAW holds by construction in production wiring."""
    from pathlib import Path

    import paths as paths_mod
    from dataset import TokenDataset

    staged = tmp_path / "root" / "v" / "u1.tokens.pt"
    staged.parent.mkdir(parents=True)
    import torch
    torch.save(torch.zeros(8, 3), staged)

    fake = paths_mod.DataPaths(root=tmp_path / "root", layout="staged")
    monkeypatch.setattr(paths_mod, "current", lambda: fake)

    ds = TokenDataset([make_rec("v/u1")])          # default: lazy paths authority
    item = ds[0]
    assert item["n_tokens"] == 3 and Path(item["utterance_id"]) == Path("v/u1")
