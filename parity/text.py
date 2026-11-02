"""Text side of the RAVEn-parity track.

* ``reference_text``: references are LRS3's ``text_raw`` upper-cased with whitespace
  collapsed; digits and apostrophes are kept. ``text_norm.normalize_text`` (Track B,
  frozen) is NOT used: it lower-cases and spells digits out, which RAVEn does not.
* Units list (``labels/unigram1000_units.txt``, SentencePiece unigram-1000 pieces):
  token list = ``["<blank>"] + units + ["<eos>"]`` (1,049 entries; blank = 0,
  sos = eos = odim - 1 = 1048), exactly upstream's ``UNIGRAM1000_LIST``.
* Hypothesis post-processing replicates ``finetune_learner.calculate_wer``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from parity.upstream import VENDOR_ROOT

VENDORED_UNITS_PATH = VENDOR_ROOT / "labels" / "unigram1000_units.txt"
VENDORED_TEST_CSV_PATH = VENDOR_ROOT / "data_paths" / "test_with_tags_counts_unigram1000.csv"
UNITS_SHA256 = "ead1469024d34194e1c7abfe2ca0457f06d5afee78267af2a0cbdea946f359f4"

BLANK_ID = 0
N_UNITS = 1047
ODIM = N_UNITS + 2
EOS_ID = SOS_ID = ODIM - 1

WORD_BOUNDARY = "▁"


def reference_text(text_raw: str) -> str:
    return " ".join(text_raw.upper().split())


def load_units(path: str | Path | None = None, *, expect_sha256: str | None = None) -> list[str]:
    """Pieces of the units file in order. Each line is ``<piece> <id>`` with ``id``
    equal to the piece's index in the token list (the first piece has id 1 because
    ``<blank>`` occupies 0); anything else is refused rather than silently reindexed."""
    pinned = path is None
    path = VENDORED_UNITS_PATH if pinned else Path(path)
    raw = path.read_bytes()
    if pinned and expect_sha256 is None:
        expect_sha256 = UNITS_SHA256
    if expect_sha256 is not None:
        got = hashlib.sha256(raw).hexdigest()
        if got != expect_sha256:
            raise ValueError(f"{path}: sha256 {got} != expected {expect_sha256}")
    lines = raw.decode("utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty units file")
    units: list[str] = []
    for n, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{n}: expected two fields `piece id`, got {line!r}")
        piece, idx = fields
        if not idx.isdigit() or int(idx) != n:
            raise ValueError(f"{path}:{n}: id {idx!r} is not the token index {n}")
        units.append(piece)
    if len(set(units)) != len(units):
        raise ValueError(f"{path}: duplicate pieces")
    return units


def token_list(units: list[str]) -> list[str]:
    return ["<blank>"] + list(units) + ["<eos>"]


def ids_to_text(token_ids, tokens: list[str]) -> str:
    """Token ids -> text, as upstream renders references (``ids_to_str`` then
    word-boundary replacement)."""
    joined = "".join(tokens[int(i)] for i in token_ids).replace("<space>", " ")
    return joined.replace(WORD_BOUNDARY, " ").strip()


def pieces_to_text(joined_pieces: str) -> str:
    """Concatenated hypothesis pieces (``add_results_to_json`` output) -> text, as
    ``finetune_learner.calculate_wer``: drop ``<eos>``, ``▁`` -> space, strip."""
    return joined_pieces.replace("<eos>", "").replace(WORD_BOUNDARY, " ").strip()


@dataclass(frozen=True)
class RavenTestRow:
    tag: str
    relpath: str
    n_frames: int
    token_ids: tuple[int, ...]
    text: str


def read_raven_test_csv(path: str | Path | None = None) -> list[RavenTestRow]:
    """Upstream's official-test list with its label ids decoded to text."""
    path = VENDORED_TEST_CSV_PATH if path is None else Path(path)
    tokens = token_list(load_units())
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split(",")
        if len(fields) != 4:
            raise ValueError(f"{path}:{n}: expected 4 comma-separated fields, got {len(fields)}")
        tag, relpath, frames, ids = fields
        token_ids = tuple(int(i) for i in ids.split())
        rows.append(RavenTestRow(tag=tag, relpath=relpath, n_frames=int(frames),
                                 token_ids=token_ids, text=ids_to_text(token_ids, tokens)))
    return rows
