"""scripts/sweep_dsir_l2.py contract — held-out cross-validation for DSIR's
--l2 hyperparameter (research decision support, not a roster-pipeline script).

Laws pinned here:
  * held_out_split is deterministic (NO RNG, matching DSIR's own determinism
    law, §3.20) and covers every row exactly once (fit XOR check);
  * auc_score is the standard Mann-Whitney U formula: 1.0 = perfect
    separation, 0.0 = perfectly inverted, 0.5 = indistinguishable (verified
    via fully-tied logits, which average to exactly 0.5 by construction);
  * held_out_logloss penalizes confident-wrong predictions more than
    indifferent ones (a proper scoring rule);
  * weight_spread matches score_dsir.py's own printed CV(exp(logit)) formula
    exactly, for continuity with the earlier ad-hoc --l2 diagnostic;
  * the sweep never fits on rows it then scores (the whole point) and the
    W&B integration runs (structurally) even with no network/credentials via
    mode="disabled".
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

import score_dsir as dsir  # noqa: E402
import sweep_dsir_l2 as S  # noqa: E402


def test_held_out_split_covers_every_row_exactly_once():
    fit, check = S.held_out_split(10, every=5)
    assert sorted(fit.tolist() + check.tolist()) == list(range(10))
    assert set(fit.tolist()).isdisjoint(check.tolist())
    assert check.tolist() == [0, 5]


def test_held_out_split_every_two():
    fit, check = S.held_out_split(10, every=2)
    assert check.tolist() == [0, 2, 4, 6, 8]
    assert fit.tolist() == [1, 3, 5, 7, 9]


def test_held_out_split_rejects_every_below_two():
    with pytest.raises(ValueError):
        S.held_out_split(10, every=1)


def test_auc_perfect_separation():
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 1, 1])
    assert S.auc_score(logits, y) == pytest.approx(1.0)


def test_auc_perfect_inversion():
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    y = np.array([0, 0, 1, 1])
    assert S.auc_score(logits, y) == pytest.approx(0.0)


def test_auc_all_tied_is_chance():
    """Fully-tied logits carry zero information -> AUC must be exactly 0.5;
    also exercises the average-rank tie-breaking path directly."""
    logits = np.array([5.0, 5.0, 5.0, 5.0])
    y = np.array([0, 0, 1, 1])
    assert S.auc_score(logits, y) == pytest.approx(0.5)


def test_auc_raises_on_single_class():
    with pytest.raises(ValueError):
        S.auc_score(np.array([1.0, 2.0]), np.array([0, 0]))


def test_held_out_logloss_penalizes_confident_wrong_predictions_more():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    indifferent = np.zeros(4)                 # logit=0 for everyone
    confident_correct = np.array([-5.0, -5.0, 5.0, 5.0])
    confident_wrong = np.array([5.0, 5.0, -5.0, -5.0])
    ll_indiff = S.held_out_logloss(indifferent, y)
    ll_correct = S.held_out_logloss(confident_correct, y)
    ll_wrong = S.held_out_logloss(confident_wrong, y)
    assert ll_correct < ll_indiff < ll_wrong


def test_weight_spread_matches_score_dsir_formula():
    logits = np.array([-0.2, 0.1, 0.05, -0.05, 0.3])
    w = np.exp(logits - logits.max())
    expected = w.std() / w.mean()
    assert S.weight_spread(logits) == pytest.approx(expected)


def _synthetic_populations():
    """40 pool rows (codes 2-6, cycling) + 10 val rows (code 1) -- same
    trivially-separable construction as test_score_dsir's fit test, sized so
    every=5 leaves both classes non-empty on both sides of the split."""
    X_pool = np.zeros((40, 8))
    for i in range(40):
        X_pool[i, 2 + i % 5] = 1.0
    X_val = np.zeros((10, 8))
    X_val[:, 1] = 1.0
    train_ids = [f"v0/{i}" for i in range(40)]
    val_ids = [f"v1/{i}" for i in range(10)]
    return train_ids, val_ids, X_pool, X_val


def test_run_sweep_end_to_end(monkeypatch):
    monkeypatch.setattr(dsir, "_collect_populations", _synthetic_populations)
    df = S.run_sweep([1e-4, 1e-1], every=5)
    assert list(df.columns) == ["l2", "held_out_auc", "held_out_logloss",
                               "weight_spread_full", "n_check_pool", "n_check_val"]
    assert len(df) == 2
    assert df["l2"].tolist() == sorted(df["l2"].tolist())
    assert ((df["held_out_auc"] >= 0.0) & (df["held_out_auc"] <= 1.0)).all()
    assert (df["weight_spread_full"] >= 0.0).all()
    assert (df["n_check_pool"] == 8).all()   # 40/5
    assert (df["n_check_val"] == 2).all()    # 10/5


def test_log_to_wandb_disabled_mode_runs_without_network():
    """mode='disabled' no-ops all wandb network/credential activity -- this
    just proves the logging code path itself is structurally correct."""
    df = pd.DataFrame({
        "l2": [1e-4, 1e-2],
        "held_out_auc": [0.6, 0.8],
        "held_out_logloss": [0.5, 0.4],
        "weight_spread_full": [0.1, 0.3],
        "n_check_pool": [8, 8],
        "n_check_val": [2, 2],
    })
    S.log_to_wandb(df, project="spell-rq2", entity=None, mode="disabled",
                   job_id="test")


def test_main_cli_writes_output_and_recommends_best_auc(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dsir, "_collect_populations", _synthetic_populations)
    out = tmp_path / "sweep.parquet"
    rc = S.main(["--l2s", "1e-4", "1e-1", "--every", "5",
                "--out", str(out), "--wandb-mode", "disabled"])
    assert rc == 0
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 2
    captured = capsys.readouterr().out
    assert "RECOMMENDATION" in captured
