"""Shared fixtures-as-functions for the Stage-2 parity tests (tiny architecture, synthetic
staged audio, a fake W&B run). Not a test module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from dataset import UtteranceRecord
from parity import model as parity_model


def tiny_arch(**over) -> dict:
    """The real architecture keys with small transformer widths. The ResNet front-end keeps
    its upstream size (it is hard-coded), so this is a small model, not a tiny one."""
    cfg = parity_model.load_arch_config()
    cfg.update(adim=32, aheads=2, eunits=64, elayers=2, ddim=32, dheads=2, dunits=64, dlayers=1)
    cfg.update(over)
    return cfg


def make_record(uid: str, duration_s: float, text: str = "HELLO WORLD") -> UtteranceRecord:
    video, _, stem = uid.partition("/")
    return UtteranceRecord(
        utterance_id=uid, split="trainval", video_id=video, stem=stem, tokens_path="", audio_path="",
        audio_kind="flac", txt_path="", n_tokens=10, duration_s=duration_s, conf=5, text_raw=text,
        text_norm=text.lower(), n_chars_norm=len(text))


def tone(seconds: float, seed: int, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(sr * seconds))) / sr
    f = 150 + 60 * (seed % 7)
    x = 0.2 * np.sin(2 * np.pi * f * t) + 0.02 * rng.standard_normal(t.shape)
    return x.astype(np.float32)


def write_flac(path: Path, wave: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), (wave * 32767).astype(np.int16), 16000, subtype="PCM_16", format="FLAC")


def make_world(root: Path, durations: list[float], *, n_units: int = 20, seed: int = 0,
               video: str = "vid") -> dict:
    """Staged-layout FLAC files (``<root>/<video>/<stem>.flac``), their records, random token ids."""
    rng = np.random.default_rng(seed)
    records, token_ids = [], {}
    for k, d in enumerate(durations):
        uid = f"{video}/{50000 + k}"
        write_flac(root / video / f"{50000 + k}.flac", tone(d, seed + k))
        n_tok = max(2, int(d * 4))
        token_ids[uid] = tuple(int(i) for i in rng.integers(1, n_units + 1, size=n_tok))
        records.append(make_record(uid, d, text=" ".join(f"W{i}" for i in range(n_tok // 2 + 1))))
    return {"records": records, "token_ids": token_ids,
            "resolver": lambda rec: root / rec.video_id / f"{rec.stem}.flac"}


class FakeRun:
    """Stands in for a wandb run: records every call."""

    def __init__(self):
        self.logged, self.summary, self.finished, self.defined = [], {}, [], []

    def log(self, payload, step=None):
        self.logged.append(dict(payload))

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def finish(self, exit_code=0):
        self.finished.append(exit_code)
