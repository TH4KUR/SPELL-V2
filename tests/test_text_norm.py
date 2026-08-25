import pytest

from text_norm import normalize_text, parse_lrs3_transcript


def test_lowercase():
    assert normalize_text("ONE DAY A YOUNG BOY") == "one day a young boy"


def test_digits_to_words():
    assert normalize_text("I HAVE 3 CATS AND 12 DREAMS") == "i have three cats and twelve dreams"
    assert normalize_text("YEAR 2020") == "year two thousand and twenty"


def test_digit_adjacent_char_does_not_fuse():
    assert "three d" in normalize_text("THE 3D MOVIE")


def test_punctuation_stripped_apostrophes_kept():
    assert normalize_text("DON'T STOP — IT'S ALIVE!") == "don't stop it's alive"


def test_whitespace_collapsed():
    assert normalize_text("  A\tB\n\n  C   ") == "a b c"


def test_empty_after_normalization():
    assert normalize_text("--- !!! ???") == ""


def test_real_lrs3_transcript_line():
    raw = "ONE DAY A YOUNG BOY COMES UPON THE SUNFLOWER WHILE VISITING THE GARDEN"
    t = parse_lrs3_transcript(f"Text:  {raw}\nConf:  4\n")
    assert t.text_raw == raw
    assert t.conf == 4
    assert normalize_text(t.text_raw) == raw.lower()


def test_parse_without_conf():
    t = parse_lrs3_transcript("Text:  HELLO WORLD\n")
    assert t.text_raw == "HELLO WORLD"
    assert t.conf is None


@pytest.mark.parametrize(
    "content",
    ["", "garbage line"],
)
def test_parse_rejects_malformed(content):
    with pytest.raises(ValueError):
        parse_lrs3_transcript(content)
