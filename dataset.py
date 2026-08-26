"""Dataset + collate utilities for discrete-token training.

``TokenDataset`` returns full-utterance [n_streams, T] token tensors (Track B
consumes stream 0 = RVQ₁; Track A consumes all 8). Collate pads with the
protocol's ``token_pad_id`` sentinel (-1): real codes span [0, 1023], so padding
MUST be out of range and every consumer must mask via the returned lengths/mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class UtteranceRecord:
    """One row of data_index.parquet."""

    utterance_id: str            # "<video_id>/<stem>" (trainval) or "<stem>" (test)
    split: str                   # "trainval" | "test"
    video_id: str
    stem: str
    tokens_path: str
    audio_path: str | None
    audio_kind: str | None       # "mp4" | "wav"
    txt_path: str | None
    n_tokens: int
    duration_s: float
    conf: int | None
    text_raw: str
    text_norm: str
    n_chars_norm: int

    @classmethod
    def from_row(cls, row: pd.Series) -> "UtteranceRecord":
        def _opt(key: str) -> object:
            v = row.get(key)
            return None if pd.isna(v) else v

        conf = _opt("conf")
        return cls(
            utterance_id=row["utterance_id"],
            split=row["split"],
            video_id=row["video_id"],
            stem=row["stem"],
            tokens_path=row["tokens_path"],
            audio_path=_opt("audio_path"),
            audio_kind=_opt("audio_kind"),
            txt_path=_opt("txt_path"),
            n_tokens=int(row["n_tokens"]),
            duration_s=float(row["duration_s"]),
            conf=None if conf is None else int(conf),
            text_raw="" if pd.isna(row["text_raw"]) else str(row["text_raw"]),
            text_norm="" if pd.isna(row["text_norm"]) else str(row["text_norm"]),
            n_chars_norm=int(row["n_chars_norm"]),
        )


def load_records(index_path: str | Path, split: str | None = None) -> list[UtteranceRecord]:
    df = pd.read_parquet(index_path)
    if split is not None:
        df = df[df["split"] == split]
    return [UtteranceRecord.from_row(row) for _, row in df.iterrows()]


def load_id_list(path: str | Path) -> set[str]:
    """Read an ID-list file (subset manifests, frozen splits): one ID per line,
    blank lines and ``#`` comments ignored."""
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids


def filter_records(
    records: list[UtteranceRecord], ids: set[str], *, strict: bool = True
) -> list[UtteranceRecord]:
    """Keep records whose utterance_id is in ``ids``, preserving index order.

    strict=True (default) raises when any requested ID has no record — subset
    manifests must never silently shrink (selection hygiene).
    """
    known = {r.utterance_id for r in records}
    missing = sorted(ids - known)
    if missing and strict:
        preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise ValueError(f"{len(missing)} requested IDs absent from data_index: {preview}")
    wanted = ids if not strict else ids & known
    return [r for r in records if r.utterance_id in wanted]


class TokenDataset(Dataset):
    """Full-utterance token streams; optionally CTC targets from a CharVocab.

    File locations come from paths.resolve_token_path (DATA LAYOUT LAW, §10) —
    NEVER from the record's legacy ``tokens_path`` string column. Tests with a
    synthetic tree may inject ``path_resolver=lambda rec: Path(rec.tokens_path)``
    to pin their own fixture files; production passes nothing.
    """

    def __init__(
        self,
        records: list[UtteranceRecord],
        vocab=None,                 # optional vocab.CharVocab
        include_text: bool = False,
        path_resolver=None,         # None = canonical paths.resolve_token_path
    ):
        if not records:
            raise ValueError("empty record list")
        self.records = records
        self.vocab = vocab
        self.include_text = include_text
        if path_resolver is None:
            import paths as _paths

            path_resolver = _paths.resolve_token_path
        self._resolve_path = path_resolver

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        tokens = torch.load(
            self._resolve_path(rec), map_location="cpu", weights_only=True)
        if tokens.dtype != torch.int64:
            tokens = tokens.long()
        item: dict = {
            "utterance_id": rec.utterance_id,
            "tokens": tokens,          # [n_streams, T] int64
            "n_tokens": tokens.shape[-1],
        }
        if self.include_text:
            if self.vocab is None or not rec.text_norm:
                raise ValueError(f"text requested for {rec.utterance_id} but no vocab/normalized text")
            item["text_ids"] = self.vocab.encode(rec.text_norm, warn_unknown=False)
        return item


def collate_token_batch(batch: list[dict], token_pad_id: int, text_pad_id: int | None = None) -> dict:
    """Pad variable-length utterances to [B, n_streams, Tmax].

    - ``tokens``  : int64, padded with ``token_pad_id`` (sentinel outside code range)
    - ``lengths`` : int64 [B] true frame counts
    - ``mask``    : bool [B, Tmax], True on real frames
    - ``text_ids``/``text_lengths``: present iff every item has them, padded with
      ``text_pad_id`` (defaults to the vocab pad when items carry one).
    """
    if token_pad_id >= 0:
        raise ValueError("token_pad_id must be negative — real codes span [0, codebook_size)")
    max_t = max(item["tokens"].shape[-1] for item in batch)
    n_streams = batch[0]["tokens"].shape[0]

    tokens = torch.full((len(batch), n_streams, max_t), token_pad_id, dtype=torch.int64)
    lengths = torch.empty(len(batch), dtype=torch.int64)
    for i, item in enumerate(batch):
        t = item["tokens"]
        if t.shape[0] != n_streams:
            raise ValueError(f"inconsistent n_streams at {item['utterance_id']}")
        tokens[i, :, : t.shape[-1]] = t
        lengths[i] = t.shape[-1]
    mask = torch.arange(max_t).unsqueeze(0) < lengths.unsqueeze(1)

    out: dict = {
        "utterance_ids": [item["utterance_id"] for item in batch],
        "tokens": tokens,
        "lengths": lengths,
        "mask": mask,
    }

    if all("text_ids" in item for item in batch):
        if text_pad_id is None:
            raise ValueError("collating text batches requires text_pad_id")
        max_l = max(len(item["text_ids"]) for item in batch)
        text = torch.full((len(batch), max_l), text_pad_id, dtype=torch.int64)
        tlens = torch.empty(len(batch), dtype=torch.int64)
        for i, item in enumerate(batch):
            ids = torch.as_tensor(item["text_ids"], dtype=torch.int64)
            text[i, : len(ids)] = ids
            tlens[i] = len(ids)
        out["text_ids"] = text
        out["text_lengths"] = tlens
    return out
