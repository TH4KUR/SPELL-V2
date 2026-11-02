"""parity/metrics.py: corpus WER with RAVEn semantics (sum of word-level
Levenshtein edits / sum of reference words), plus the macro mean for comparison.

RAVEn's metric (raven/metrics.py @ 9fd9a37): ``WER.update`` adds
``get_wer(pred, ref) * len(ref.split())`` and ``compute`` divides by the total
reference word count. The vendored ``raven_metrics.get_wer`` is the oracle here.
"""

from __future__ import annotations

import math
import random

import pytest

from parity import metrics as M
from parity import upstream

upstream.activate()
import raven_metrics  # noqa: E402


# ------------------------------------------------------------ edit distance --

@pytest.mark.parametrize("ref,hyp,edits", [
    ("a b c", "a b c", 0),
    ("a b c", "a x c", 1),                 # substitution
    ("a b c", "a c", 1),                   # deletion
    ("a b c", "a b x c", 1),               # insertion
    ("a b c", "", 3),                      # empty hypothesis: all deletions
    ("a b c", "x y z", 3),
    ("the cat sat", "cat sat the", 2),
    ("a  b\tc", "a b c", 0),               # whitespace is collapsed, as str.split does
    ("a", "a a a", 2),
])
def test_pair_edits_known_cases(ref, hyp, edits):
    assert M.pair_edits(ref, hyp) == edits


def test_pair_edits_is_case_and_punctuation_sensitive():
    assert M.pair_edits("A", "a") == 1
    assert M.pair_edits("IT'S", "ITS") == 1


def test_edit_distance_matches_ravens_get_er_on_random_word_lists():
    rng = random.Random(20260918)
    vocab = [f"w{i}" for i in range(8)]
    for _ in range(600):
        ref = [rng.choice(vocab) for _ in range(rng.randint(1, 14))]
        hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 16))]
        ours = M.pair_edits(" ".join(ref), " ".join(hyp))
        theirs = raven_metrics.get_er(hyp, ref) * len(ref)
        assert ours == round(theirs), (ref, hyp, ours, theirs)
        assert abs(theirs - round(theirs)) < 1e-9


def test_edit_distance_matches_jiwer_on_random_word_lists():
    import jiwer
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(6)]
    for _ in range(300):
        ref = [rng.choice(vocab) for _ in range(rng.randint(1, 12))]
        hyp = [rng.choice(vocab) for _ in range(rng.randint(1, 12))]
        ours = M.pair_edits(" ".join(ref), " ".join(hyp))
        assert ours / len(ref) == pytest.approx(jiwer.wer(" ".join(ref), " ".join(hyp)), abs=1e-12)


# ---------------------------------------------------------------- aggregates --

def test_corpus_and_macro_differ_when_lengths_differ():
    refs = ["a b c d e f g h i j", "x"]
    hyps = ["a b c d e f g h i j", "y"]
    r = M.corpus_wer(refs, hyps)
    assert r.total_edits == 1 and r.total_ref_words == 11
    assert r.corpus_wer == pytest.approx(1 / 11)
    assert r.macro_wer == pytest.approx(0.5)
    assert r.n_utts == 2 and r.n_scored == 2 and r.n_empty_ref == 0


def test_corpus_wer_matches_ravens_accumulation_on_random_corpora():
    """Re-enact raven_metrics' WER module (error += get_wer*n; total += n)
    without torchmetrics. Upstream keeps ``error`` as a float32 tensor, so allow
    float32 accumulation noise (<1e-4 relative)."""
    rng = random.Random(11)
    vocab = [f"w{i}" for i in range(10)]
    for _ in range(40):
        n = rng.randint(1, 80)
        refs, hyps = [], []
        for _ in range(n):
            refs.append(" ".join(rng.choice(vocab) for _ in range(rng.randint(1, 15))))
            hyps.append(" ".join(rng.choice(vocab) for _ in range(rng.randint(0, 15))))
        error, total = 0.0, 0
        for h, r in zip(hyps, refs):
            k = len(r.split())
            error += raven_metrics.get_wer(h, r) * k
            total += k
        assert M.corpus_wer(refs, hyps).corpus_wer == pytest.approx(error / total, rel=1e-4)


def test_macro_matches_track_b_definition():
    """Track B reports mean over utterances of jiwer.wer (ctc.py:80-94)."""
    import jiwer
    refs = ["a b c", "d e", "f g h i"]
    hyps = ["a x c", "d", "f g h i"]
    expected = sum(jiwer.wer(r, h) for r, h in zip(refs, hyps)) / 3
    assert M.corpus_wer(refs, hyps).macro_wer == pytest.approx(expected, abs=1e-12)


def test_perfect_and_total_failure():
    assert M.corpus_wer(["a b"], ["a b"]).corpus_wer == 0.0
    r = M.corpus_wer(["a b", "c"], ["", ""])
    assert r.corpus_wer == 1.0 and r.macro_wer == 1.0


def test_insertions_can_push_wer_above_one():
    assert M.corpus_wer(["a"], ["a b c"]).corpus_wer == pytest.approx(2.0)


def test_empty_reference_is_refused_by_default():
    with pytest.raises(ValueError, match="empty reference"):
        M.corpus_wer(["a b", ""], ["a b", "x"])


def test_empty_reference_opt_in_counts_insertions_but_not_words_and_skips_macro():
    r = M.corpus_wer(["a b", ""], ["a b", "x y"], allow_empty_ref=True)
    assert r.total_edits == 2 and r.total_ref_words == 2
    assert r.corpus_wer == pytest.approx(1.0)
    assert r.n_empty_ref == 1 and r.n_scored == 1
    assert r.macro_wer == 0.0                       # the single scoreable utterance is perfect


def test_all_references_empty_is_an_error_even_when_allowed():
    with pytest.raises(ValueError, match="no reference words"):
        M.corpus_wer([""], ["x"], allow_empty_ref=True)


def test_empty_hypothesis_alongside_nonempty_is_ordinary_deletion():
    r = M.corpus_wer(["a b c d", "e"], ["", "e"])
    assert r.total_edits == 4 and r.corpus_wer == pytest.approx(4 / 5)
    assert r.macro_wer == pytest.approx((1.0 + 0.0) / 2)


def test_length_mismatch_and_empty_corpus_raise():
    with pytest.raises(ValueError, match="same length"):
        M.corpus_wer(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="no utterances"):
        M.corpus_wer([], [])


def test_per_utterance_rows_are_returned_in_order():
    r = M.corpus_wer(["a b", "c d e"], ["a x", "c d e"])
    assert [(x.edits, x.ref_words) for x in r.per_utt] == [(1, 2), (0, 3)]
    assert r.per_utt[0].wer == pytest.approx(0.5) and r.per_utt[1].wer == 0.0


def test_quartile_helper_matches_manual_computation():
    refs = ["a", "b c", "d e f", "g h i j", "k l m n o", "p q r s t u", "v w x y z a b", "c d e f g h i j"]
    hyps = ["a", "b x", "d e f", "g h i j", "k l m n x", "p q r s t u", "v w x y z a b", "c d e f g h i x"]
    q = M.length_quartiles(M.corpus_wer(refs, hyps))
    assert [b.n_utts for b in q] == [2, 2, 2, 2]
    assert q[0].lo_words == 1 and q[-1].hi_words == 8
    assert [b.total_edits for b in q] == [1, 0, 1, 1]
    assert q[0].total_ref_words == 1 + 2


def test_module_source_does_not_mention_torchmetrics_or_lightning():
    src = open(M.__file__, encoding="utf-8").read()
    assert "torchmetrics" not in src and "lightning" not in src.lower()
    assert math.isfinite(M.corpus_wer(["a"], ["a"]).corpus_wer)
