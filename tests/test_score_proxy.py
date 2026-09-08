"""scripts/score_proxy.py contract (roster #4 score feed).

Laws pinned here:
  * late-epoch trajectory selection parses ckpt NAMES: ckpt_epoch{NNNN}.ckpt,
    keeping epochs >= ceil(max_epoch * late_frac) — {15, 20} for a 20-epoch
    20%-capped trajectory at late_frac=0.75 (PROVISIONAL window);
  * per-utterance means ACROSS late ckpts (loss_mean, el2n_mean) with
    n_ckpts recorded;
  * only TRAIN-pool utterances are scored (§5 item 12); sorted keyed output;
  * the lit module is loaded from the bundle's own checkpoints — no second
    training path exists for scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import score_proxy as S  # noqa: E402


def test_late_ckpts_parsed_from_names(tmp_path):
    """Names parse, epoch filter applies, output is sorted by epoch."""
    d = tmp_path / "bundle"
    d.mkdir()
    for e in (5, 10, 15, 20):
        (d / f"ckpt_epoch{e:04d}.ckpt").touch()
    got = S._late_ckpts(d, late_frac=0.75)
    assert got == [d / "ckpt_epoch0015.ckpt", d / "ckpt_epoch0020.ckpt"]


def test_late_ckpts_empty_trajectory_is_loud(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="trajectory"):
        S._late_ckpts(d, late_frac=0.75)


def test_per_uid_means_across_ckpts():
    rows_per_ckpt = [
        {"a/1": (1.0, 0.5), "a/2": (2.0, 0.2)},
        {"a/1": (3.0, 0.9), "a/2": (2.0, 0.4)},
    ]
    means = S._mean_across(rows_per_ckpt)
    assert means["a/1"] == pytest.approx((2.0, 0.7))
    assert means["a/2"] == pytest.approx((2.0, 0.3))


def test_main_schema_train_only_sorted(tmp_path, monkeypatch):
    class FakeLit:
        pass

    monkeypatch.setattr(S, "enforce_gpu_policy", lambda: {"gpu_name": "fake"})
    monkeypatch.setattr(S, "_late_ckpts", lambda b, late_frac: ["ck1", "ck2"])
    monkeypatch.setattr(S, "_load_lit", lambda ckpt: FakeLit())
    monkeypatch.setattr(S, "_collect_train_records", lambda: ["v0/1", "v0/2"])
    monkeypatch.setattr(S, "_preflight", lambda recs, k=50: None)
    monkeypatch.setattr(
        S, "_score_all",
        lambda lits, ids: [
            {"utterance_id": "v0/2", "loss_mean": 2.0, "el2n_mean": 0.3,
             "n_ckpts": 2},
            {"utterance_id": "v0/1", "loss_mean": 1.0, "el2n_mean": 0.8,
             "n_ckpts": 2},
        ])
    out = tmp_path / "scores" / "proxy_scores.parquet"
    rc = S.main(["--bundle", str(tmp_path / "b"), "--out", str(out)])
    assert rc == 0
    df = pd.read_parquet(out)
    assert list(df.columns) == ["utterance_id", "loss_mean", "el2n_mean",
                                "n_ckpts"]
    assert list(df["utterance_id"]) == ["v0/1", "v0/2"]    # sorted, train only
    assert (df["n_ckpts"] == 2).all()