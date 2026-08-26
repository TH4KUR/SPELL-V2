"""THE data-layout authority (PROTOCOL §10 item 4 DATA LAYOUT LAW).

Every runtime file-location decision goes through this module — no other code
may concatenate dataset roots, split names, or folder heuristics (§10 item 4).
Split membership is METADATA ONLY (``data_index.split`` / frozen split ID
lists); the filesystem never encodes a split segment (§10 item 5 INDEX IS
TRUTH: identity/membership come from data_index.parquet, filenames come from
the declared layout below).

CANONICAL LAYOUT ("staged", what exists on Ada under $HOME/spell/data;
declared in docs/layout.md):

    <root>/<VIDEO_ID>/<stem>.tokens.pt | .txt | .flac    # split=trainval utts
    <root>/<stem>.tokens.pt | .txt | .flac               # split=test: BARE files

Legacy layout ("legacy", Phase-0 raw source ``datasets/LRS3/trainval|test/…``
with .mp4/.wav containers) survives ONLY behind ``SPELL_DATA_LAYOUT=legacy``
for pre-restage tooling; it is never a training-time layout.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path, PurePosixPath

LAYOUT_DOC = "docs/layout.md"

TOKEN_SUFFIX = ".tokens.pt"
TRANSCRIPT_SUFFIX = ".txt"
STAGED_AUDIO_SUFFIX = ".flac"

_LAYOUTS = ("staged", "legacy")


class LayoutError(RuntimeError):
    """A record cannot be located under the declared layout — always LOUD."""


@lru_cache(maxsize=1)
def current() -> "DataPaths":
    return DataPaths(
        root=_load_root(),
        layout=os.environ.get("SPELL_DATA_LAYOUT", "staged").strip().lower(),
    )


def reset_cache() -> None:
    """Test hook: env changes take effect after calling this."""
    current.cache_clear()


def _load_root() -> Path:
    import os

    from config import load_paths

    return Path(load_paths().dataset_root)


class DataPaths:
    """Resolved (root, layout) pair — the ONLY sanctioned path-builder."""

    def __init__(self, root: Path, layout: str):
        if layout not in _LAYOUTS:
            raise LayoutError(
                f"SPELL_DATA_LAYOUT={layout!r} is not one of {_LAYOUTS}; "
                f"see {LAYOUT_DOC}")
        self.root = Path(root)
        self.layout = layout
        if layout == "staged" and self.root.name == "LRS3" \
                and self.root.parent.name == "datasets":
            # The configured default IS the Phase-0 raw audit source (audit-era
            # mp4/wav tree). Staged mode refusing it here is what turns "the env
            # var was forgotten" from a mid-training failure into a startup
            # FATAL with the remedy attached.
            raise LayoutError(
                f"[spell] staged layout given the RAW AUDIT SOURCE root: "
                f"{self.root}\n"
                f"That tree holds .mp4/.wav containers under split dirs — never "
                f"a staged token tree.\n"
                f"REMEDY: export SPELL_DATA_ROOT=$HOME/spell/data (Ada, §5.0) — "
                f"slurm/template.sbatch now defaults it; interactive shells must "
                f"set it before python. See {LAYOUT_DOC}.")

    # ------------------------------------------------------------ builders --
    def tokens_relpath(self, video_id: str, stem: str) -> PurePosixPath:
        return _rel(video_id, stem, TOKEN_SUFFIX)

    def transcript_relpath(self, video_id: str, stem: str) -> PurePosixPath:
        return _rel(video_id, stem, TRANSCRIPT_SUFFIX)

    def flac_relpath(self, video_id: str, stem: str) -> PurePosixPath:
        """STAGED audio container name (16 kHz mono FLAC — always, both splits)."""
        return _rel(video_id, stem, STAGED_AUDIO_SUFFIX)

    # ----------------------------------------------------------- resolvers --
    def resolve_tokens(self, rec) -> Path:
        """rec: dataset.UtteranceRecord (or anything with .video_id/.stem)."""
        rel = self.tokens_relpath(rec.video_id, rec.stem)
        path = self.root / rel
        self._exists_or_raise(path, rec, rel, "tokens")
        return path

    def resolve_transcript(self, rec) -> Path:
        rel = self.transcript_relpath(rec.video_id, rec.stem)
        path = self.root / rel
        self._exists_or_raise(path, rec, rel, "transcript")
        return path

    def resolve_audio(self, rec) -> Path:
        """Staged layout: ALWAYS the .flac twin. Legacy: original container."""
        if self.layout == "staged":
            rel = self.flac_relpath(rec.video_id, rec.stem)
            path = self.root / rel
            self._exists_or_raise(path, rec, rel, "audio(.flac)")
            return path
        return self._legacy_container(rec)

    def _legacy_container(self, rec) -> Path:
        """Audit-era tree: record paths mirror datasets/LRS3 minus its prefix."""
        if not getattr(rec, "audio_path", None):
            raise LayoutError(
                f"{rec.utterance_id}: no audio_path in index; legacy layout "
                "cannot synthesize a location")
        parts = PurePosixPath(str(rec.audio_path)).parts
        parts = parts[2:] if len(parts) > 2 and parts[:2] == ("datasets", "LRS3") else parts
        path = self.root.joinpath(*parts)
        self._exists_or_raise(path, rec, PurePosixPath(*parts), "audio(container)")
        return path

    def _exists_or_raise(self, path: Path, rec, rel, kind: str) -> bool:
        if path.is_file():
            return True
        expected_tree = (
            "<root>/<VIDEO_ID>/<stem>.tokens.pt|.txt|.flac   (split=trainval)"
            if rec.video_id else
            "<root>/<stem>.tokens.pt|.txt|.flac              (split=test)"
        )
        raise LayoutError(
            f"[spell] LAYOUT MISMATCH for {rec.utterance_id} ({kind})\n"
            f"  data_root : {self.root}  (layout={self.layout})\n"
            f"  resolved  : {path}\n"
            f"  expected  : {expected_tree}\n"
            f"  declared in: {LAYOUT_DOC}\n"
            f"If you restaged or moved data: regenerate data_index.parquet and run "
            f"--verify (§10 item 5); do NOT hand-patch paths.\n"
            f"If data_root above is not where you staged the data, set "
            f"SPELL_DATA_ROOT (see {LAYOUT_DOC})."
        )

    # ------------------------------------------------------------ preflight --
    def preflight(self, records, k: int = 50, seed: int = 20260826,
                  kind: str = "tokens") -> list[tuple[str, Path]]:
        """Resolve k RANDOM records BEFORE epoch 1 and require every file present.

        Returns [(utterance_id, resolved_path)]; raises LayoutError printing the
        resolved-vs-declared diff on the FIRST miss (startup hard gate).
        """
        if not records:
            raise LayoutError("preflight given an empty record list")
        sample = records if len(records) <= k else random.Random(seed).sample(records, k)
        resolver = self.resolve_tokens if kind == "tokens" else self.resolve_audio
        return [(rec.utterance_id, resolver(rec)) for rec in sample]


def _rel(video_id: str, stem: str, suffix: str) -> PurePosixPath:
    video_id = (video_id or "").strip()
    stem = str(stem).strip()
    if not stem:
        raise LayoutError(f"empty stem (video_id={video_id!r})")
    if "/" in stem:
        raise LayoutError(f"stem must not contain '/': {stem!r}")
    base = f"{stem}{suffix}"
    return PurePosixPath(video_id) / base if video_id else PurePosixPath(base)


# ------------------------------------------------------------------------- --
# Module-level convenience wrappers (production entry points).
# ------------------------------------------------------------------------- --

def resolve_token_path(rec) -> Path:
    return current().resolve_tokens(rec)


def resolve_transcript_path(rec) -> Path:
    return current().resolve_transcript(rec)


def resolve_audio_path(rec) -> Path:
    return current().resolve_audio(rec)


def preflight_resolve(records, k: int = 50, seed: int = 20260826,
                      kind: str = "tokens") -> list[tuple[str, Path]]:
    return current().preflight(records, k=k, seed=seed, kind=kind)
