"""scripts/score_dsir.py contract (roster #3 scorer — 72h go/no-go clock).

Laws pinned here:
  * hashed features over RVQ₁: unigram block = raw code ids (codebook dims),
    bigram block = (a·1009 + b) mod 2¹⁶ (PROVISIONAL hash); each block is
    L1-normalized to mass 1, then the concatenated row is divided by 2 —
    total selection mass always 1.0;
  * importance fit = class-balanced logistic regression target(val, y=1) vs
    pool(train, y=0), zero-init L-BFGS-B → fully deterministic (no RNG);
  * the ×3 seeds live in the LAPTOP-side resample (selection/dsir.select) —
    the scorer commits ONE table, scores/dsir_weights.parquet, [utterance_id,
    score_logit, weight], TRAIN rows only, sorted (§5 item 12);
  * a weight-spread (CV) summary prints loudly — the go/no-go gate reads it;
  * numerically stable BCE (softplus form), finite logits asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import score_dsir as S  # noqa: E402


def test_hash_bigram_deterministic_and_in_range():
    for a, b in ((0, 0), (5, 3), (1023, 1023)):
        assert 0 <= S.hash_bigram(int(a), int(b)) < 65536
    assert S.hash_bigram(5, 3) == S.hash_bigram(5, 3)
    assert S.hash_bigram(5, 3) != S.hash_bigram(3, 5)


def test_featurize_blocks_normalized_total_mass_one():
    """Stream [2,5,2], codebook 8: uni mass [2:2/3, 5:1/3], bi mass 1/3 each on
    hashed dims; concatenated row halved → sum 1.0, uni dims = code ids, bi
    dims = codebook + bucket."""
    row = S.featurize(np.array([2, 5, 2]), codebook_size=8)
    assert row.nnz == 4
    assert row.sum() == pytest.approx(1.0)
    assert row[0, 2] == pytest.approx(1 / 3)
    assert row[0, 5] == pytest.approx(1 / 6)
    bi_vals = sorted(v for i, v in zip(row.indices, row.data) if i >= 8)
    assert len(bi_vals) == 2 and bi_vals[0] == pytest.approx(1 / 4)  # 2 bigrams


def test_fit_separable_data_separates_and_deterministic():
    """Val-like rows carry code 1; pool rows codes 2-6 → pool rows score
    NEGATIVE against the val target; two fits are bit-identical (no RNG)."""
    X_pool = np.zeros((40, 8))
    for i in range(40):
        X_pool[i, 2 + i % 5] = 1.0
    X_val = np.zeros((10, 8))
    X_val[:, 1] = 1.0
    logits_a = S.fit_importance(X_pool, X_val, l2=1e-4)
    logits_b = S.fit_importance(X_pool, X_val, l2=1e-4)
    assert np.array_equal(logits_a, logits_b)
    assert np.isfinite(logits_a).all()
    assert (logits_a < 0).all()


def test_output_schema_train_only_sorted(tmp_path, monkeypatch):
    ids, val_ids = ["v0/2", "v0/1"], ["v1/9"]
    feats_val = np.zeros((1, 8), dtype=np.float32); feats_val[0, 1] = 1.0
    feats_train = np.zeros((2, 8), dtype=np.float32)
    feats_train[0, 2] = 1.0        # v0/2
    feats_train[1, 3] = 1.0        # v0/1
    monkeypatch.setattr(
        S, "_collect_populations",
        lambda: (ids, val_ids, sparse.csr_matrix(feats_train), sparse.csr_matrix(feats_val)))
    out = tmp_path / "scores" / "dsir_weights.parquet"
    rc = S.main(["--out", str(out)])
    assert rc == 0
    df = pd.read_parquet(out)
    assert list(df.columns) == ["utterance_id", "score_logit", "weight"]
    assert list(df["utterance_id"]) == ["v0/1", "v0/2"]    # sorted, val absent
    assert (df["weight"] > 0).all()


def test_weight_spread_summary_printed(tmp_path, monkeypatch, capsys):
    """The go/no-go reads the printed weight-spread line (CV of exp(logit))."""
    ids, val_ids = ["v0/1", "v0/2"], ["v1/3"]
    feats_val = np.zeros((1, 8), dtype=np.float32); feats_val[0, 1] = 1.0
    feats_train = np.zeros((2, 8), dtype=np.float32)
    feats_train[0, 2] = 1.0
    feats_train[1, 3] = 1.0
    monkeypatch.setattr(
        S, "_collect_populations",
        lambda: (ids, val_ids, sparse.csr_matrix(feats_train), sparse.csr_matrix(feats_val)))
    S.main(["--out", str(tmp_path / "dsir_weights.parquet")])
    assert "weight_spread" in capsys.readouterr().out