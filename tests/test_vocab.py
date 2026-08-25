from vocab import BLANK_ID, PAD_ID, build_char_vocab


def test_frozen_layout():
    v = build_char_vocab()
    assert len(v) == 2 + 26 + 1 + 1  # pad, blank, a-z, space, apostrophe
    assert v.pad_id == PAD_ID == 0
    assert v.blank_id == BLANK_ID == 1
    assert v.tokens[2] == "a"
    assert v.stoi[" "] == 28
    assert v.stoi["'"] == 29


def test_round_trip():
    v = build_char_vocab()
    s = "it's a small world after all"
    ids = v.encode(s)
    assert v.decode(ids) == s
    # every id is in range and none is a special id for real text
    assert all(0 <= i < len(v) for i in ids)
    assert PAD_ID not in ids and BLANK_ID not in ids


def test_unknown_chars_dropped_not_mapped_to_specials():
    v = build_char_vocab()
    ids = v.encode("café")
    assert v.decode(ids) == "caf"
    ids = v.encode("abc 42")   # space survives (it's in the alphabet), digits dropped
    assert v.decode(ids) == "abc "
    assert PAD_ID not in ids and BLANK_ID not in ids


def test_ctc_style_decode_collapses_repeats_and_blanks():
    v = build_char_vocab()
    a, b = v.stoi["a"], v.stoi["b"]
    # adjacent repeats collapse; blank/pad RESET the run (standard greedy-CTC rule)
    ids = [a, a, BLANK_ID, b, b, BLANK_ID, a]
    assert v.decode(ids, collapse_repeats=True) == "aba"
    # pad behaves like blank: it breaks runs rather than merging them
    ids = [b, PAD_ID, b]
    assert v.decode(ids, collapse_repeats=True) == "bb"
    ids = [a, a, BLANK_ID, b, PAD_ID, b]
    assert v.decode(ids, collapse_repeats=True) == "abb"


def test_save_load_round_trip(tmp_path):
    v = build_char_vocab()
    p = tmp_path / "char_vocab.json"
    v.save(p)
    v2 = type(v).load(p)
    assert v.tokens == v2.tokens
    assert v.encode("hello world") == v2.encode("hello world")
