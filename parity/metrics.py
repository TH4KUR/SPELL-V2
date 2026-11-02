"""Corpus WER with RAVEn semantics (PROTOCOL §3.33), plus the macro mean.

RAVEn reports CORPUS WER: its ``WER`` metric accumulates ``get_wer(pred, ref) *
len(ref.split())`` per utterance and divides by the total reference word count,
i.e. total word-level Levenshtein edits over total reference words. Track B
instead reports the MACRO mean of per-utterance WERs (``ctc.py``); both are
returned here so the two statistics are never conflated.

Definitions (words are whitespace-split tokens, unit-cost Levenshtein):

* empty hypothesis -> ``edits == n_ref_words`` (all deletions), WER 1.0 for that
  utterance, same as upstream's ``get_er``;
* empty reference -> upstream divides by zero (inf/NaN), jiwer raises. Refused here
  by default; ``allow_empty_ref=True`` counts the hypothesis words as insertions in
  the corpus numerator, adds 0 reference words, and skips the utterance in the macro
  mean (its per-utterance WER is undefined).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UttScore:
    edits: int
    ref_words: int

    @property
    def wer(self) -> float | None:
        return self.edits / self.ref_words if self.ref_words else None


@dataclass(frozen=True)
class CorpusWer:
    corpus_wer: float
    macro_wer: float | None
    total_edits: int
    total_ref_words: int
    n_utts: int
    n_scored: int
    n_empty_ref: int
    per_utt: tuple[UttScore, ...]


@dataclass(frozen=True)
class QuartileBin:
    index: int
    lo_words: int
    hi_words: int
    n_utts: int
    total_edits: int
    total_ref_words: int
    corpus_wer: float
    macro_wer: float


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def pair_edits(ref: str, hyp: str) -> int:
    """Word-level Levenshtein distance between two strings."""
    return _edit_distance(ref.split(), hyp.split())


def aggregate(per_utt: list[UttScore], *, allow_empty_ref: bool = False) -> CorpusWer:
    """Corpus and macro statistics from per-utterance (edits, reference words)."""
    if not per_utt:
        raise ValueError("no utterances to score")
    n_empty = sum(1 for u in per_utt if u.ref_words == 0)
    if n_empty and not allow_empty_ref:
        first = next(i for i, u in enumerate(per_utt) if u.ref_words == 0)
        raise ValueError(f"empty reference at index {first}: WER is undefined "
                         "(pass allow_empty_ref=True to count it as insertions)")
    total_edits = sum(u.edits for u in per_utt)
    total_ref = sum(u.ref_words for u in per_utt)
    if total_ref == 0:
        raise ValueError("no reference words in the corpus: corpus WER is undefined")
    scored = [u.wer for u in per_utt if u.ref_words]
    return CorpusWer(
        corpus_wer=total_edits / total_ref,
        macro_wer=sum(scored) / len(scored),
        total_edits=total_edits,
        total_ref_words=total_ref,
        n_utts=len(per_utt),
        n_scored=len(scored),
        n_empty_ref=n_empty,
        per_utt=tuple(per_utt),
    )


def corpus_wer(refs: list[str], hyps: list[str], *, allow_empty_ref: bool = False) -> CorpusWer:
    if len(refs) != len(hyps):
        raise ValueError(f"refs and hyps must have the same length ({len(refs)} != {len(hyps)})")
    return aggregate(
        [UttScore(edits=pair_edits(ref, hyp), ref_words=len(ref.split()))
         for ref, hyp in zip(refs, hyps)],
        allow_empty_ref=allow_empty_ref)


def length_quartiles(result: CorpusWer) -> list[QuartileBin]:
    """Scoreable utterances sorted by reference length (stable) and split into four
    near-equal groups (earlier groups take the remainder); stats per group."""
    scored = sorted((u for u in result.per_utt if u.ref_words), key=lambda u: u.ref_words)
    n = len(scored)
    if n < 4:
        raise ValueError(f"need at least 4 scoreable utterances for quartiles, got {n}")
    base, extra = divmod(n, 4)
    bins, start = [], 0
    for k in range(4):
        size = base + (1 if k < extra else 0)
        group = scored[start:start + size]
        start += size
        edits = sum(u.edits for u in group)
        words = sum(u.ref_words for u in group)
        bins.append(QuartileBin(
            index=k,
            lo_words=group[0].ref_words,
            hi_words=group[-1].ref_words,
            n_utts=len(group),
            total_edits=edits,
            total_ref_words=words,
            corpus_wer=edits / words,
            macro_wer=sum(u.wer for u in group) / len(group),
        ))
    return bins
