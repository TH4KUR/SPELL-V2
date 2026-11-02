"""parity/data.py: eval dataset over an id list.

Upstream loads audio with ``torchaudio.load(path, normalize=True)`` and applies no
further processing. For integer PCM that is ``int / 2**15`` as float32, which is
what libsndfile (via ``soundfile``) returns for 16-bit files too; the tests prove
bit-equality against the manual conversion (and against torchaudio when it is
importable).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from dataset import UtteranceRecord
from parity import data as D


def _rec(uid: str, text_raw: str = "hello world", video_id: str = "v0", stem: str = "50001"):
    return UtteranceRecord(
        utterance_id=uid, split="trainval", video_id=video_id, stem=stem, tokens_path="",
        audio_path=None, audio_kind=None, txt_path=None, n_tokens=10, duration_s=1.0,
        conf=5, text_raw=text_raw, text_norm="", n_chars_norm=0)


def _write_int16(path: Path, samples: np.ndarray, sr: int = 16000, fmt: str = "FLAC"):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.int16), sr, subtype="PCM_16", format=fmt)


@pytest.fixture
def int16_samples():
    rng = np.random.default_rng(0)
    s = rng.integers(-32768, 32768, size=16000 + 123, dtype=np.int16)
    s[0], s[1], s[2] = -32768, 32767, 0
    return s


@pytest.mark.parametrize("fmt,suffix", [("FLAC", ".flac"), ("WAV", ".wav")])
def test_pcm16_loads_bit_identical_to_manual_int16_over_32768(tmp_path, int16_samples, fmt, suffix):
    p = tmp_path / f"a{suffix}"
    _write_int16(p, int16_samples, fmt=fmt)
    wave = D.load_waveform(p)
    expected = (int16_samples.astype(np.float32) / np.float32(32768.0))
    assert wave.dtype == torch.float32 and wave.ndim == 1
    assert wave.shape[0] == len(int16_samples)
    np.testing.assert_array_equal(wave.numpy(), expected)
    assert wave.min().item() == -1.0 and wave.max().item() == 32767 / 32768
    assert wave.max().item() < 1.0


@pytest.mark.parametrize("fmt,suffix", [("FLAC", ".flac"), ("WAV", ".wav")])
def test_pcm16_matches_torchaudio_normalize_true_when_available(tmp_path, int16_samples, fmt, suffix):
    ta = pytest.importorskip("torchaudio")
    p = tmp_path / f"a{suffix}"
    _write_int16(p, int16_samples, fmt=fmt)
    ref = ta.load(str(p), normalize=True)[0].squeeze(0)
    np.testing.assert_array_equal(D.load_waveform(p).numpy(), ref.numpy())


def test_no_normalisation_is_applied(tmp_path):
    quiet = (np.ones(8000) * 3).astype(np.int16)
    p = tmp_path / "q.flac"
    _write_int16(p, quiet)
    wave = D.load_waveform(p)
    assert wave.abs().max().item() == pytest.approx(3 / 32768)       # not rescaled to unit peak
    assert wave.mean().item() == pytest.approx(3 / 32768)            # not mean/variance normalised


def test_wrong_sample_rate_fails_loudly(tmp_path, int16_samples):
    p = tmp_path / "x.flac"
    _write_int16(p, int16_samples, sr=8000)
    with pytest.raises(ValueError, match="16000"):
        D.load_waveform(p)


def test_stereo_fails_loudly(tmp_path):
    p = tmp_path / "s.flac"
    sf.write(str(p), np.zeros((1600, 2), dtype=np.int16), 16000, subtype="PCM_16", format="FLAC")
    with pytest.raises(ValueError, match="mono"):
        D.load_waveform(p)


def test_empty_audio_fails_loudly(tmp_path):
    p = tmp_path / "e.wav"
    sf.write(str(p), np.zeros(0, dtype=np.int16), 16000, subtype="PCM_16", format="WAV")
    with pytest.raises(ValueError, match="empty"):
        D.load_waveform(p)


def test_audio_subtype_reports_the_container_encoding(tmp_path, int16_samples):
    p = tmp_path / "a.flac"
    _write_int16(p, int16_samples)
    assert D.audio_subtype(p) == "PCM_16"


def test_dataset_returns_id_waveform_and_parity_reference(tmp_path, int16_samples):
    p = tmp_path / "v0" / "50001.flac"
    _write_int16(p, int16_samples)
    recs = [_rec("v0/50001", text_raw="it's  5 o'clock")]
    ds = D.ParityEvalDataset(recs, path_resolver=lambda r: p)
    assert len(ds) == 1
    uid, wave, ref = ds[0]
    assert uid == "v0/50001"
    assert wave.dtype == torch.float32 and wave.shape == (len(int16_samples),)
    assert ref == "IT'S 5 O'CLOCK"                                    # upper-case, digits kept


def test_dataset_default_resolver_is_the_layout_authority(tmp_path, int16_samples, monkeypatch):
    """No data path is built here: the default resolver is paths.resolve_audio_path."""
    import paths as data_paths

    seen = []
    p = tmp_path / "v0" / "50001.flac"
    _write_int16(p, int16_samples)

    def fake(rec):
        seen.append(rec.utterance_id)
        return p

    monkeypatch.setattr(data_paths, "resolve_audio_path", fake)
    ds = D.ParityEvalDataset([_rec("v0/50001")])
    ds[0]
    assert seen == ["v0/50001"]


def test_dataset_rejects_empty_record_list():
    with pytest.raises(ValueError, match="empty"):
        D.ParityEvalDataset([])


def test_dataset_rejects_an_empty_reference(tmp_path, int16_samples):
    p = tmp_path / "a.flac"
    _write_int16(p, int16_samples)
    ds = D.ParityEvalDataset([_rec("u", text_raw="   ")], path_resolver=lambda r: p)
    with pytest.raises(ValueError, match="empty reference"):
        ds[0]


def test_dataloader_without_batching_yields_plain_tuples(tmp_path, int16_samples):
    paths = {}
    recs = []
    for i in range(3):
        uid = f"v{i}/5000{i}"
        p = tmp_path / f"{i}.flac"
        _write_int16(p, int16_samples[: 8000 + 100 * i])
        paths[uid] = p
        recs.append(_rec(uid, text_raw=f"word{i}", video_id=f"v{i}", stem=f"5000{i}"))
    ds = D.ParityEvalDataset(recs, path_resolver=lambda r: paths[r.utterance_id])
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=0)
    out = list(dl)
    assert [o[0] for o in out] == [r.utterance_id for r in recs]
    assert [o[1].shape[0] for o in out] == [8000, 8100, 8200]
    assert [o[2] for o in out] == ["WORD0", "WORD1", "WORD2"]
