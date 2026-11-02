"""parity/text.py: reference text for the parity track and the units-list utilities.

The parity track deliberately bypasses ``text_norm.normalize_text`` (which spells
digits out and lower-cases): RAVEn's labels are the raw upper-case LRS3 text with
digits kept (24 official test references contain digits).
"""

from __future__ import annotations

import hashlib

import pytest

from parity import text as T
from parity import upstream
from text_norm import normalize_text

upstream.activate()
import raven_utils  # noqa: E402


def test_reference_text_uppercases_collapses_whitespace_and_keeps_digits_and_apostrophes():
    assert T.reference_text("it's  5\to'clock \n") == "IT'S 5 O'CLOCK"
    assert T.reference_text("  the   first lesson is about humility ") == "THE FIRST LESSON IS ABOUT HUMILITY"
    assert T.reference_text("ALREADY UPPER 2012") == "ALREADY UPPER 2012"


def test_reference_text_is_not_the_track_b_normalizer():
    raw = "i have 5 apples"
    assert normalize_text(raw) != raw.lower()                    # Track B spells the digit out
    assert T.reference_text(raw) == "I HAVE 5 APPLES"


def test_reference_text_of_blank_is_empty():
    assert T.reference_text("   \t ") == ""


def test_load_units_matches_the_pinned_file():
    units = T.load_units()
    assert len(units) == 1047
    assert units[0] == "<unk>" and units[-1] == "▁YOURSELF"
    assert len(set(units)) == 1047
    raw = (T.VENDORED_UNITS_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == T.UNITS_SHA256 == (
        "ead1469024d34194e1c7abfe2ca0457f06d5afee78267af2a0cbdea946f359f4")


def test_token_list_layout_and_agreement_with_upstream_construction():
    tl = T.token_list(T.load_units())
    assert len(tl) == 1049 and tl[0] == "<blank>" and tl[-1] == "<eos>"
    assert tl == raven_utils.UNIGRAM1000_LIST          # upstream's own construction, verbatim code
    assert T.BLANK_ID == 0 and T.EOS_ID == T.SOS_ID == 1048


def test_load_units_second_column_is_the_token_index():
    """Each line is `piece id` with id == position in the final token list
    (blank is 0, so the first unit has id 1)."""
    for i, line in enumerate(T.VENDORED_UNITS_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        piece, idx = line.split()
        assert int(idx) == i


def test_load_units_rejects_malformed_files(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("<unk> 1\n' 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        T.load_units(p)
    p.write_text("<unk> 1\n<unk> 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        T.load_units(p)
    p.write_text("<unk> 1 extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two"):
        T.load_units(p)
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        T.load_units(p)


def test_load_units_checks_sha256_when_asked(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("<unk> 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        T.load_units(p, expect_sha256=T.UNITS_SHA256)
    assert T.load_units(p, expect_sha256=hashlib.sha256(p.read_bytes()).hexdigest()) == ["<unk>"]


def test_ids_to_text_matches_upstream_ids_to_str_then_boundary_replacement():
    tl = T.token_list(T.load_units())
    ids = [1005, 811, 442, 2, 217, 997, 294, 175, 208]       # first row of the vendored test csv
    expected = raven_utils.ids_to_str(ids, tl).replace("▁", " ").strip()
    assert T.ids_to_text(ids, tl) == expected
    assert expected == expected.upper() and expected and "▁" not in expected


def test_ids_to_text_accepts_tensor_like_ints():
    import torch
    tl = T.token_list(T.load_units())
    assert T.ids_to_text(torch.tensor([1005, 811]), tl) == T.ids_to_text([1005, 811], tl)


def test_pieces_to_text_follows_the_upstream_decode_postprocessing():
    assert T.pieces_to_text("▁HELLO▁WORLD<eos>") == "HELLO WORLD"
    assert T.pieces_to_text("▁IT'S▁A▁TEST<eos>") == "IT'S A TEST"
    assert T.pieces_to_text("<eos>") == ""
    assert T.pieces_to_text("▁A▁▁B") == "A  B"                 # not collapsed (upstream does not either)


def test_every_row_of_the_vendored_test_csv_decodes_to_nonempty_text():
    rows = T.read_raven_test_csv()
    assert len(rows) == 1321
    for r in rows:
        assert r.text and r.text == r.text.upper()
        assert all(0 < i < 1048 for i in r.token_ids)


def test_read_raven_test_csv_parses_the_first_row():
    r = T.read_raven_test_csv()[0]
    assert r.relpath == "test/0Fi83BHQsMA/00002.avi" and r.n_frames == 41
    assert r.token_ids[:3] == (1005, 811, 442)
