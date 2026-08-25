"""Character vocabulary for Track B CTC.

Fixed alphabet locked by the spec: ``a-z``, space, apostrophe, plus the two
special tokens. IDs are part of the frozen protocol:

- ``<pad>`` = 0  (batch padding only; masked out of loss/decoding)
- ``<blank>`` = 1  (CTC blank)
- 2.. : a-z, then space, then apostrophe.

Input to :func:`encode` must already be normalized via
:func:`text_norm.normalize_text` — unknown characters are dropped with a warning,
never mapped to a special id.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

PAD = "<pad>"
BLANK = "<blank>"
PAD_ID = 0
BLANK_ID = 1

# Locked alphabet (spec): lowercase letters, space, apostrophe.
ALPHABET = string.ascii_lowercase + " '"


class CharVocab:
    def __init__(self, alphabet: str = ALPHABET):
        seen: list[str] = []
        for ch in alphabet:
            if ch not in seen:
                seen.append(ch)
        self.tokens: list[str] = [PAD, BLANK] + seen
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def blank_id(self) -> int:
        return BLANK_ID

    def encode(self, normalized_text: str, *, warn_unknown: bool = True) -> list[int]:
        ids: list[int] = []
        for ch in normalized_text:
            idx = self.stoi.get(ch)
            if idx is None:
                if warn_unknown:
                    print(f"vocab: dropping character {ch!r} (not in alphabet)")
                continue
            ids.append(idx)
        return ids

    def decode(self, ids: list[int] | list[list[int]], collapse_repeats: bool = False) -> str | list[str]:
        """Decode id sequences. With ``collapse_repeats`` (CTC greedy style),
        repeated ids collapse to one and blanks/pads are removed — this mirrors
        what the Phase-1 greedy decoder does."""
        single_is_seq = ids and isinstance(ids[0], (list, tuple))
        seqs = ids if single_is_seq else [ids]
        out = [self._decode_one(list(s), collapse_repeats) for s in seqs]
        return out if single_is_seq else out[0]

    def _decode_one(self, ids: list[int], collapse_repeats: bool) -> str:
        chars: list[str] = []
        prev: int | None = None
        for i in ids:
            if i in (self.pad_id, self.blank_id):
                prev = None  # blanks break CTC repeat runs
                continue
            if collapse_repeats and i == prev:
                continue
            prev = i
            chars.append(self.tokens[i])
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"tokens": self.tokens}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        tokens = json.loads(Path(path).read_text(encoding="utf-8"))["tokens"]
        vocab = cls.__new__(cls)
        vocab.tokens = tokens
        vocab.stoi = {t: i for i, t in enumerate(tokens)}
        if vocab.tokens[:2] != [PAD, BLANK]:
            raise ValueError(f"vocabulary file {path} lacks frozen <pad>/<blank> prefix")
        return vocab


def build_char_vocab() -> CharVocab:
    """The one vocabulary for the whole project (frozen alphabet)."""
    return CharVocab(ALPHABET)
