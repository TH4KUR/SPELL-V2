"""Evaluation dataset for the RAVEn-parity track.

Audio is read through ``paths.resolve_audio_path`` (the DATA LAYOUT LAW authority:
staged 16 kHz mono FLAC) with ``soundfile``. For integer PCM, libsndfile returns
``int / 2**(bits-1)`` as float32, i.e. the same values as upstream's
``torchaudio.load(path, normalize=True)``; no peak, mean or variance normalisation
is applied (RAVEn paper App. A.2: raw audio, no pre-processing). Anything that is
not 16 kHz mono fails loudly - there is no silent resample or down-mix path.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from parity.text import reference_text

SAMPLE_RATE = 16000


def load_waveform(path: str | Path) -> torch.Tensor:
    """float32 [-1, 1] mono waveform, shape (samples,)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {sr} != {SAMPLE_RATE} (no silent resampling)")
    if audio.shape[1] != 1:
        raise ValueError(f"{path}: expected mono audio, got {audio.shape[1]} channels")
    if audio.shape[0] == 0:
        raise ValueError(f"{path}: empty audio")
    return torch.from_numpy(audio[:, 0].copy())


def audio_subtype(path: str | Path) -> str:
    """libsndfile subtype (e.g. ``PCM_16``) of an audio file."""
    return sf.info(str(path)).subtype


class ParityEvalDataset(Dataset):
    """``(utterance_id, waveform, reference_text)`` over ``dataset.UtteranceRecord``s."""

    def __init__(self, records, *, path_resolver=None):
        if not records:
            raise ValueError("empty record list")
        self.records = list(records)
        if path_resolver is None:
            import paths as _paths

            path_resolver = _paths.resolve_audio_path
        self._resolve = path_resolver

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        ref = reference_text(rec.text_raw)
        if not ref:
            raise ValueError(f"{rec.utterance_id}: empty reference text")
        return rec.utterance_id, load_waveform(self._resolve(rec)), ref
