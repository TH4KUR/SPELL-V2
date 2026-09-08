"""scripts/score_kmeans.py contract (roster #2 scorer, Ada CPU job).

Laws pinned here:
  * features = per-utterance RVQ₁ (stream 0) code histogram over [0, 1024),
    L1-normalized — computed in-job, NEVER committed (repo bloat guard);
  * clustering = scipy.cluster.vq.kmeans2 with kmeans++ init and
    seed = the §3.20 subset-identity seed → deterministic, seed-divergent;
  * output per seed: scores/kmeans_assignments_seed{S}.parquet with the
    §5 item 12 schema [utterance_id, cluster_id, dist, cluster_size];
  * per-utterance codebook entropy rides the same job → scores/token_stats.parquet
    (characterization feed for ALL selectors, RESEARCH P2 stat-sheet law);
  * TRAIN pool only (§5 item 12); codebook size from the frozen protocol.
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

import score_kmeans as S  # noqa: E402


def test_histogram_l1_normalized_and_stream0_only():
    """[8, T] tokens → histogram of stream 0 over [0, codebook_size), summing
    to 1.0; streams 1-7 never enter the features."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 64, size=(8, 30))
    hist = S.histogram(tokens, codebook_size=64)
    assert hist.shape == (64,)
    assert hist.sum() == pytest.approx(1.0)
    expected = np.bincount(tokens[0], minlength=64) / 30.0
    assert np.allclose(hist, expected)


def test_codebook_entropy_hand_case():
    """Uniform histogram over C codes → log2(C); degenerate single-code → 0."""
    uniform = np.full(8, 1 / 8)
    assert S.codebook_entropy(uniform, codebook_size=8) == pytest.approx(
        np.log2(8), rel=1e-9)
    degenerate = np.zeros(8)
    degenerate[3] = 1.0
    assert S.codebook_entropy(degenerate, codebook_size=8) == pytest.approx(0.0)


def test_kmeans2_seed_deterministic_and_seed_divergent():
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(80, 4)).astype(np.float32)
    labels_a, dist_a = S.cluster(feats, k=4, seed=201)
    labels_b, dist_b = S.cluster(feats, k=4, seed=201)
    labels_c, _ = S.cluster(feats, k=4, seed=202)
    assert np.array_equal(labels_a, labels_b)
    assert not np.array_equal(labels_a, labels_c)
    assert len(labels_a) == 80 and (dist_a >= 0).all()


def test_assignments_parquet_schema(tmp_path, monkeypatch):
    """Per-seed table: [utterance_id, cluster_id, dist, cluster_size], sorted,
    cluster_size consistent with the per-seed assignment."""
    ids = [f"v0/{50000 + i}" for i in range(12)]
    feats = np.arange(24, dtype=np.float32).reshape(12, 2)
    monkeypatch.setattr(S, "_collect_train_records", lambda: ids)
    monkeypatch.setattr(S, "_build_features",
                        lambda ids: (feats, np.zeros(12, dtype=np.float64)))
    out = tmp_path / "scores"
    rc = S.main(["--out-dir", str(out), "--seeds", "201", "--k", "3"])
    assert rc == 0
    df = pd.read_parquet(out / "kmeans_assignments_seed201.parquet")
    assert list(df.columns) == ["utterance_id", "cluster_id", "dist", "cluster_size"]
    assert list(df["utterance_id"]) == sorted(ids)
    counts = df["cluster_id"].value_counts()
    assert (df["cluster_id"].map(counts) == df["cluster_size"]).all()
    stats = pd.read_parquet(out / "token_stats.parquet")
    assert list(stats["utterance_id"]) == sorted(ids)
    assert "codebook_entropy" in stats.columns


def test_k_above_rows_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_collect_train_records", lambda: ["v0/1", "v0/2"])
    monkeypatch.setattr(S, "_build_features",
                        lambda ids: np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="--k"):
        S.main(["--out-dir", str(tmp_path / "s"), "--seeds", "201", "--k", "5"])


def test_real_protocol_codebook_constant():
    """The scorer reads codebook_size from the frozen protocol (1024), never
    hardcodes a second truth."""
    from config import load_protocol

    assert S.codebook_size() == load_protocol().codebook_size == 1024