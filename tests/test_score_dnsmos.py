"""scripts/score_dnsmos.py contract (roster #1 scorer).

Laws pinned here:
  * calibration uses the vendored NON-personalized poly1d coefficients, verbatim;
  * windowing mirrors DNS-Challenge/DNSMOS/dnsmos_local.py EXACTLY: 9.01 s =
    144,160-sample windows, 1 s hop, self-tiling of short clips, hop-mean;
  * staged audio is 16 kHz mono — any other rate is a LOUD refusal (no silent
    resample path; the frozen env carries no librosa);
  * the onnx session is injected (workers are subprocesses on Ada; tests use a
    fake session so the suite never needs onnxruntime — the real-model test
    skips unless onnxruntime imports);
  * only TRAIN-pool rows are scored (§5 item 12) and the output is sorted by id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import score_dnsmos as S  # noqa: E402

MODEL = PROJECT_ROOT / "models" / "dnsmos" / "sig_bak_ovr.onnx"


def test_calibration_coefficients_are_the_vendored_nonpersonalized_set():
    for name, expected in (
        ("sig", [-0.08397278, 1.22083953, 0.0052439]),
        ("bak", [-0.13166888, 1.60915514, -0.39604546]),
        ("ovr", [-0.06766283, 1.11546468, 0.04602535]),
    ):
        assert list(S.POLY_COEF[name]) == expected


def test_calibration_is_monotone_on_plausible_range():
    """Selection ranks on calibrated ovr_mos — monotone calibration means the
    ranking is invariant to the raw-scale ambiguity (recorded empirically)."""
    xs = np.linspace(-2.0, 6.0, 200)
    ovr = S.calibrate(xs, "ovr")
    assert (np.diff(ovr) > 0).all()


class FakeSession:
    """Returns a fixed raw vector per hop: sig=1.0, bak=2.0, ovr=3.0."""

    def run(self, None_, feed):
        x = feed["input_1"]
        assert x.dtype == np.float32
        assert x.shape == (1, S.LEN_SAMPLES)
        return [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]


def test_score_audio_tiles_short_clips_and_averages_hops():
    """A 2 s clip self-tiles to ≥ 9.01 s (32k→64k→128k→256k samples); the hop
    count formula runs over the TILED length (vendored behaviour): floor(16 −
    9.01) + 1 = 7. Every hop returns the fixed raw triple, so raw means are
    exact and calibrated values equal poly1d(3.0) per channel."""
    fs = 16000
    audio = 0.1 * np.sin(2 * np.pi * 220 * np.arange(2 * fs) / fs)
    fake = FakeSession()
    row = S.score_audio(audio, fake)
    tiled = len(audio)
    while tiled < S.LEN_SAMPLES:
        tiled *= 2
    assert row["n_hops"] == int(np.floor(tiled / fs) - 9.01) + 1
    assert row["ovr_raw"] == pytest.approx(3.0)
    assert row["sig_mos"] == pytest.approx(float(S.calibrate(np.array([1.0]), "sig")[0]))
    assert row["ovr_mos"] == pytest.approx(float(S.calibrate(np.array([3.0]), "ovr")[0]))


def test_score_audio_rejects_wrong_sample_rate():
    with pytest.raises(ValueError, match="16 kHz"):
        S.score_audio(np.zeros(16000, dtype=np.float32), FakeSession(), fs=48000)


def _rec(uid: str):
    from dataset import UtteranceRecord

    vid, stem = uid.split("/")
    return UtteranceRecord(
        utterance_id=uid, split="trainval", video_id=vid, stem=stem,
        tokens_path=None, audio_path=None, audio_kind=None, txt_path=None,
        n_tokens=100, duration_s=2.0, conf=5, text_raw="x", text_norm="x",
        n_chars_norm=1)


def test_collect_train_records_keeps_train_pool_only(monkeypatch):
    """§5 item 12 / §2.5: val rows are never scored, never leave the job."""
    train, val = ["v0/1", "v0/2"], ["v1/3"]
    recs = [_rec(u) for u in train + val]
    monkeypatch.setattr(S, "load_records", lambda path, split: recs)
    monkeypatch.setattr(
        S, "load_id_list",
        lambda path: set(train) if "train_ids" in str(path) else set(val))
    kept = S._collect_train_records()
    assert [r.utterance_id for r in kept] == ["v0/1", "v0/2"]   # sorted, val gone


def test_train_pool_only_and_sorted_output(tmp_path, monkeypatch):
    """The scorer sorts by id and emits the §5 item 12 column schema. Seams are
    injected so the suite needs no Ada data."""
    monkeypatch.setattr(S, "_collect_train_records", lambda: [_rec(u) for u in
                                                              ("v0/1", "v0/2")])
    monkeypatch.setattr(S, "resolve_audio_for", lambda rec: tmp_path / "a.flac")
    monkeypatch.setattr(S, "_preflight", lambda recs, k=50: None)
    fake_rows = [
        {"utterance_id": "v0/2", "sig_raw": 1.0, "bak_raw": 2.0, "ovr_raw": 3.0,
         "sig_mos": 1.0, "bak_mos": 2.0, "ovr_mos": 3.0, "n_hops": 1},
        {"utterance_id": "v0/1", "sig_raw": 1.5, "bak_raw": 2.5, "ovr_raw": 3.5,
         "sig_mos": 1.5, "bak_mos": 2.5, "ovr_mos": 3.5, "n_hops": 1},
    ]
    monkeypatch.setattr(S, "_score_all", lambda recs, model, workers: fake_rows)
    out = tmp_path / "scores" / "out.parquet"
    rc = S.main(["--out", str(out), "--workers", "1"])
    assert rc == 0
    df = pd.read_parquet(out)
    assert list(df["utterance_id"]) == ["v0/1", "v0/2"]      # sorted
    assert list(df.columns) == ["utterance_id", "sig_raw", "bak_raw", "ovr_raw",
                                "sig_mos", "bak_mos", "ovr_mos", "n_hops"]


def test_real_model_runs_if_onnxruntime_present(tmp_path):
    """Integration gate for the Ada job — runs wherever onnxruntime exists."""
    ort = pytest.importorskip("onnxruntime")
    assert MODEL.exists(), "vendored weights missing"
    import soundfile as sf

    fs = 16000
    wav = tmp_path / "tone.flac"
    sf.write(wav, 0.1 * np.sin(2 * np.pi * 220 * np.arange(2 * fs) / fs), fs,
             subtype="PCM_16")
    session = S.load_session(MODEL)
    audio, fs_in = sf.read(wav, dtype="float32")
    row = S.score_audio(audio, session)
    assert fs_in == fs
    for key in ("ovr_raw", "sig_raw", "bak_raw", "ovr_mos", "sig_mos", "bak_mos"):
        assert row[key] == row[key] and abs(row[key]) < 100.0   # finite, sane
    assert row["n_hops"] >= 1