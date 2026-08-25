"""THE canonical transcript normalization.

FROZEN after Phase 0 — every WER ever computed in this project must use this
function (PROTOCOL.md rule: "text_norm frozen forever"). Do not add parameters,
do not fork variants; if it changes, all prior results are invalidated.

Pipeline: lowercase -> digits->words (num2words) -> strip punctuation except
apostrophes -> collapse whitespace. LRS3 transcripts arrive as
``Text:  ONE DAY ...`` / ``Conf: 4`` blocks; :func:`parse_lrs3_transcript`
extracts the raw text + confidence before normalization.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from num2words import num2words
except ImportError as _e:  # pragma: no cover - dependency is installed in Phase 0
    raise ImportError("text_norm requires num2words (pip install num2words)") from _e

# Matches the on-disk format exactly: "Text:  <text>\nConf:  <int>".
# Conf may be absent for recovered official-test transcripts.
_TRANSCRIPT_RE = re.compile(r"^Text:\s+(?P<text>.*?)\s*(?:\nConf:\s+(?P<conf>\d+))?\s*$", re.DOTALL)

_DIGITS_RE = re.compile(r"\d+")
_ALLOWED_RE = re.compile(r"[^a-z'\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Transcript:
    text_raw: str
    conf: int | None  # None if absent


def parse_lrs3_transcript(content: str) -> Transcript:
    """Parse an on-disk .txt into raw text + confidence (no normalization here)."""
    m = _TRANSCRIPT_RE.match(content.strip())
    if not m:
        raise ValueError(f"unrecognized transcript format: {content[:80]!r}")
    conf = m.group("conf")
    return Transcript(text_raw=m.group("text"), conf=int(conf) if conf is not None else None)


def normalize_text(text: str) -> str:
    """Canonical normalization. FROZEN — see module docstring."""
    s = text.lower()

    def _digits_to_words(m: re.Match) -> str:
        try:
            words = num2words(int(m.group(0)))
        except ValueError:  # e.g. pathological digit run; drop it, log once per call site
            logger.warning("num2words failed on %r; dropping token", m.group(0))
            return ""
        # Pad so adjacent characters don't fuse ("3d" -> "three d", not "threed");
        # whitespace is collapsed at the end anyway.
        return f" {words} "

    s = _DIGITS_RE.sub(_digits_to_words, s)
    s = _ALLOWED_RE.sub(" ", s)          # anything outside a-z/space/apostrophe -> space
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def is_valid_norm(norm: str) -> bool:
    """A normalized transcript is usable iff non-empty."""
    return len(norm) > 0
