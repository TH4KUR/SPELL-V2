"""scripts/score_dnsmos.py contract (roster #1 scorer).

Laws pinned here:
  * calibration uses the vendored NON-personalized poly1d coefficients, verbatim;
  * windowing mirrors DNS-Challenge/DNSMOS/dnsmos_local.py EXACTLY: 9.01 s =
    144,160-sample windows, 1 s hop, self-tiling of short clips, hop-mean;
  * staged audio is 16 kHz mono — any other rate is a LOUD refusal (no silent
    resample path; the frozen env carries no librosa);
  * the model is injected (workers are subprocesses on Ada; tests use a fake
    callable so the suite never needs the real weights — the real-model test
    runs unconditionally against dnsmos_model.DNSMOSTorch, no onnxruntime
    anywhere: 2026-09-09, PROTOCOL §5.5);
  * only TRAIN-pool rows are scored (§5 item 12) and the output is sorted by id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import score_dnsmos as S  # noqa: E402

TORCH_MODEL = PROJECT_ROOT / "models" / "dnsmos" / "sig_bak_ovr_torch.pt"


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

    def __call__(self, feed):
        assert feed.dtype == torch.float32
        assert tuple(feed.shape) == (1, S.LEN_SAMPLES)
        return torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)


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
    monkeypatch.setattr(S, "_score_all", lambda recs, model, workers, **kwargs: fake_rows)
    out = tmp_path / "scores" / "out.parquet"
    rc = S.main(["--out", str(out), "--workers", "1", "--wandb-mode", "disabled"])
    assert rc == 0
    df = pd.read_parquet(out)
    assert list(df["utterance_id"]) == ["v0/1", "v0/2"]      # sorted
    assert list(df.columns) == ["utterance_id", "sig_raw", "bak_raw", "ovr_raw",
                                "sig_mos", "bak_mos", "ovr_mos", "n_hops"]


# ------------------------------------------------------------- progress/wandb

def test_score_all_reports_on_a_time_cadence_not_a_row_count(monkeypatch, tmp_path):
    """PROTOCOL §3.23/§3.24: reporting must be scheduled by ELAPSED TIME, not
    a fixed row count -- a count-based threshold silently assumes a
    throughput rate (the original version's `report_every=2000` meant the
    FIRST log line didn't appear for ~10+ minutes at this scorer's real
    rate). A fake, manually-advanced clock proves reports land once
    report_every_s has actually passed, regardless of how many rows
    completed in that window, and a final forced report covers the tail
    even when it doesn't land on a clean interval boundary."""
    recs = [_rec(f"v0/{i}") for i in range(6)]
    monkeypatch.setattr(S, "resolve_audio_for", lambda rec: tmp_path / "a.flac")
    monkeypatch.setattr(S, "load_session", lambda model_path: object())

    clock = {"t": 0.0}
    monkeypatch.setattr(S.time, "monotonic", lambda: clock["t"])
    costs = [5.0, 5.0, 15.0, 1.0, 1.0, 1.0]   # cumulative: 5,10,25,26,27,28
    call_index = {"i": 0}

    def fake_score_one(task):
        uid, _ = task
        clock["t"] += costs[call_index["i"]]
        call_index["i"] += 1
        return {"utterance_id": uid, "sig_raw": 1.0, "bak_raw": 2.0, "ovr_raw": 3.0,
                "sig_mos": 1.0, "bak_mos": 2.0, "ovr_mos": 3.0, "n_hops": 1}

    monkeypatch.setattr(S, "_score_one", fake_score_one)

    seen = []
    rows = S._score_all(recs, "fake-model.pt", workers=1, report_every_s=10.0,
                        on_progress=lambda *a: seen.append(a))
    assert len(rows) == 6
    # t=5: no report (<10s since start). t=10: report (done=2). t=25: report
    # (done=3, 15s since last). t=26,27: no report (<10s since last).
    # t=28: forced final report (done=6) even though only 3s have passed.
    assert [s[0] for s in seen] == [2, 3, 6]
    assert all(s[1] == 6 for s in seen)
    # each report's chunk_s is the DELTA since the last report, not
    # cumulative elapsed_s -- proven by chunk_s NOT following elapsed_s's
    # strictly-increasing pattern (15 > 10, then 3 < 15) while elapsed_s
    # itself does strictly increase
    chunk_durations = [s[2] for s in seen]
    elapsed_times = [s[3] for s in seen]
    assert chunk_durations == pytest.approx([10.0, 15.0, 3.0])
    assert elapsed_times == pytest.approx([10.0, 25.0, 28.0])
    assert elapsed_times[0] < elapsed_times[1] < elapsed_times[2]


def test_wandb_init_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", model_path="fake.pt", workers=1, n_recs=5)
    run.finish()


def test_log_scoring_progress_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", model_path="fake.pt", workers=1, n_recs=5)
    S._log_scoring_progress(2000, 29064, 45.0, 45.0)
    run.finish()


def test_main_marks_wandb_run_failed_on_exception(monkeypatch, tmp_path):
    """Caught live (2026-09-18): job 2701229 (score_proxy.py, identical
    finally: pattern) crashed but still showed as a completed run in the
    W&B UI, because `run.finish()` with no args defaults to success
    regardless of whether an exception is propagating. A fake run object
    (not real wandb) proves main() now passes exit_code=1 on a crash."""
    calls = []

    class FakeRun:
        def finish(self, exit_code=0):
            calls.append(exit_code)

    monkeypatch.setattr(S, "_wandb_init", lambda **kw: FakeRun())
    monkeypatch.setattr(S, "_collect_train_records", lambda: [_rec("v0/1")])

    def boom(recs, k=50):
        raise RuntimeError("boom")

    monkeypatch.setattr(S, "_preflight", boom)

    with pytest.raises(RuntimeError, match="boom"):
        S.main(["--out", str(tmp_path / "out.parquet")])
    assert calls == [1]


def test_main_marks_wandb_run_succeeded_on_clean_exit(monkeypatch, tmp_path):
    calls = []

    class FakeRun:
        def finish(self, exit_code=0):
            calls.append(exit_code)

    monkeypatch.setattr(S, "_wandb_init", lambda **kw: FakeRun())
    monkeypatch.setattr(S, "log_final_summary_to_wandb", lambda df: None)
    monkeypatch.setattr(S, "_collect_train_records", lambda: [_rec(u) for u in
                                                              ("v0/1", "v0/2")])
    monkeypatch.setattr(S, "_preflight", lambda recs, k=50: None)
    monkeypatch.setattr(S, "_score_all", lambda recs, model, workers, **kw: [
        {"utterance_id": "v0/1", "sig_raw": 1.0, "bak_raw": 2.0, "ovr_raw": 3.0,
         "sig_mos": 1.0, "bak_mos": 2.0, "ovr_mos": 3.0, "n_hops": 1},
        {"utterance_id": "v0/2", "sig_raw": 1.0, "bak_raw": 2.0, "ovr_raw": 3.0,
         "sig_mos": 1.0, "bak_mos": 2.0, "ovr_mos": 3.0, "n_hops": 1},
    ])

    rc = S.main(["--out", str(tmp_path / "out.parquet"), "--workers", "1"])
    assert rc == 0
    assert calls == [0]


def test_log_final_summary_to_wandb_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", model_path="fake.pt", workers=1, n_recs=2)
    df = pd.DataFrame([{"utterance_id": "v0/1", "ovr_mos": 3.0},
                       {"utterance_id": "v0/2", "ovr_mos": 3.5}])
    S.log_final_summary_to_wandb(df)
    run.finish()


def test_real_model_runs_end_to_end(tmp_path):
    """Integration gate for the Ada job — the real dnsmos_model.DNSMOSTorch,
    no onnxruntime anywhere (2026-09-09, PROTOCOL §5.5)."""
    assert TORCH_MODEL.exists(), "vendored torch weights missing"
    import soundfile as sf

    fs = 16000
    wav = tmp_path / "tone.flac"
    sf.write(wav, 0.1 * np.sin(2 * np.pi * 220 * np.arange(2 * fs) / fs), fs,
             subtype="PCM_16")
    session = S.load_session(TORCH_MODEL)
    audio, fs_in = sf.read(wav, dtype="float32")
    row = S.score_audio(audio, session)
    assert fs_in == fs
    for key in ("ovr_raw", "sig_raw", "bak_raw", "ovr_mos", "sig_mos", "bak_mos"):
        assert row[key] == row[key] and abs(row[key]) < 100.0   # finite, sane
    assert row["n_hops"] >= 1