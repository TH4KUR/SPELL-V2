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


def test_histogram_l1_normalized_over_stream0_codes():
    """Given an ALREADY-extracted RVQ-1 (stream 0) code sequence (histogram's
    actual contract — `_build_features` passes `_load_stream0`'s output
    straight through), produces an L1-normalized histogram over
    [0, codebook_size). histogram() must NOT re-index a second `[0]` — that
    collapses the 1-D array to a scalar and crashes np.bincount with "object
    of too small depth for desired array" (2026-09-16 incident, job 2698949:
    score_kmeans's first-ever run against real Ada tokens)."""
    rng = np.random.default_rng(0)
    stream0 = rng.integers(0, 64, size=30)
    hist = S.histogram(stream0, codebook_size=64)
    assert hist.shape == (64,)
    assert hist.sum() == pytest.approx(1.0)
    expected = np.bincount(stream0, minlength=64) / 30.0
    assert np.allclose(hist, expected)


def test_build_features_does_not_double_extract_stream0(monkeypatch):
    """End-to-end seam test for the exact 2698949 failure: _build_features
    feeds _load_stream0's output straight into histogram() with no second
    [0] anywhere in between."""
    import paths as data_paths

    full = np.array([[1, 2, 3, 2, 1]] * 8)   # any [8, T]-shaped stand-in
    monkeypatch.setattr(S, "codebook_size", lambda: 64)
    monkeypatch.setattr(data_paths, "resolve_token_path", lambda rec: "unused")
    monkeypatch.setattr(S.torch, "load", lambda path, **kw: full)
    monkeypatch.setattr(
        S, "load_records",
        lambda index_path, split: [type("R", (), {"utterance_id": "v0/1"})()])

    feats, ents = S._build_features(["v0/1"])
    assert feats.shape == (1, 64)
    assert feats[0].sum() == pytest.approx(1.0)


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
                        lambda ids, **kwargs: (feats, np.zeros(12, dtype=np.float64)))
    out = tmp_path / "scores"
    rc = S.main(["--out-dir", str(out), "--seeds", "201", "--k", "3",
                "--wandb-mode", "disabled"])
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
                        lambda ids, **kwargs: np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="--k"):
        S.main(["--out-dir", str(tmp_path / "s"), "--seeds", "201", "--k", "5",
               "--wandb-mode", "disabled"])


def test_real_protocol_codebook_constant():
    """The scorer reads codebook_size from the frozen protocol (1024), never
    hardcodes a second truth."""
    from config import load_protocol

    assert S.codebook_size() == load_protocol().codebook_size == 1024


def test_load_stream0_uses_root_joined_path_not_bare_relpath(monkeypatch):
    """2026-09-09 incident (job 2692024): _load_stream0 called
    ``paths.current().tokens_relpath(...)`` directly -- a RELATIVE path never
    joined with the data root -- so torch.load got a bare
    '<video_id>/<stem>.tokens.pt' and crashed FileNotFoundError on the very
    first real Ada run. It must go through ``paths.resolve_token_path``
    (root-joined + existence-checked), never the relpath builder alone."""
    import paths as data_paths

    sentinel = Path("/fake/data/root/v0/50001.tokens.pt")
    monkeypatch.setattr(data_paths, "resolve_token_path", lambda rec: sentinel)
    seen = {}

    def fake_torch_load(path, **kwargs):
        seen["path"] = path
        return np.zeros((8, 10), dtype=np.int64)

    monkeypatch.setattr(S.torch, "load", fake_torch_load)

    class FakeRec:
        utterance_id = "v0/50001"
        video_id = "v0"
        stem = "50001"

    S._load_stream0(FakeRec())
    assert seen["path"] == sentinel


# ------------------------------------------------------------- progress/wandb

def test_build_features_reports_on_a_time_cadence_not_a_row_count(monkeypatch):
    """PROTOCOL §3.23/§3.24/§3.25: reporting must be scheduled by ELAPSED
    TIME, not a fixed row count (mirrors the identical fix applied to
    score_dnsmos.py/score_proxy.py the same day). A fake, manually-advanced
    clock proves reports land once report_every_s has actually passed."""
    n = 6
    clock = {"t": 0.0}
    monkeypatch.setattr(S.time, "monotonic", lambda: clock["t"])
    costs = [5.0, 5.0, 15.0, 1.0, 1.0, 1.0]   # cumulative: 5,10,25,26,27,28
    call_index = {"i": 0}

    class FakeRec:
        def __init__(self, uid):
            self.utterance_id = uid

    ids = [f"v0/{i}" for i in range(n)]
    fake_recs = [FakeRec(u) for u in ids]
    monkeypatch.setattr(S, "load_records", lambda index_path, split: fake_recs)
    monkeypatch.setattr(S, "load_paths", lambda: type("P", (), {"index_path": "x"})())
    monkeypatch.setattr(S, "codebook_size", lambda: 8)

    def fake_load_stream0(rec):
        clock["t"] += costs[call_index["i"]]
        call_index["i"] += 1
        return np.zeros(4, dtype=np.int64)

    monkeypatch.setattr(S, "_load_stream0", fake_load_stream0)

    seen = []
    feats, ents = S._build_features(ids, report_every_s=10.0,
                                     on_progress=lambda *a: seen.append(a))
    assert feats.shape == (n, 8)
    assert [s[0] for s in seen] == [2, 3, 6]
    assert all(s[1] == 6 for s in seen)


def test_wandb_init_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", seeds=[201, 202], k=64, n_recs=100)
    run.finish()


def test_log_progress_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", seeds=[201], k=64, n_recs=100)
    S._log_progress(50, 100, 10.0, 20.0)
    run.finish()


def test_log_final_summary_to_wandb_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", seeds=[201], k=3, n_recs=3)
    stats = pd.DataFrame({"utterance_id": ["a", "b"], "codebook_entropy": [0.5, 1.0]})
    assign_df = pd.DataFrame({"utterance_id": ["a", "b"], "cluster_id": [0, 1],
                              "dist": [0.1, 0.2], "cluster_size": [1, 1]})
    S.log_final_summary_to_wandb(stats, {201: assign_df})
    run.finish()


def test_main_marks_wandb_run_failed_on_exception(monkeypatch, tmp_path):
    """Mirrors the identical fix on score_proxy.py/score_dnsmos.py/
    score_less.py/sweep_dsir_l2.py (§3.25): run.finish() with no args
    always marks the run 'Finished' even mid-crash."""
    calls = []

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def finish(self, exit_code=0):
            calls.append(exit_code)

    fake_run = FakeRun()
    monkeypatch.setattr(S, "_wandb_init", lambda **kw: fake_run)
    monkeypatch.setattr(S, "_collect_train_records", lambda: ["v0/1", "v0/2"])

    def boom(ids, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(S, "_build_features", boom)

    with pytest.raises(RuntimeError, match="boom"):
        S.main(["--out-dir", str(tmp_path / "s"), "--seeds", "201", "--k", "1"])
    assert calls == [1]
    assert "RuntimeError: boom" in fake_run.summary["error"]
    assert "boom" in fake_run.summary["traceback"]


def test_main_marks_wandb_run_succeeded_on_clean_exit(monkeypatch, tmp_path):
    calls = []

    class FakeRun:
        def finish(self, exit_code=0):
            calls.append(exit_code)

    monkeypatch.setattr(S, "_wandb_init", lambda **kw: FakeRun())
    monkeypatch.setattr(S, "log_final_summary_to_wandb", lambda stats, assign_dfs: None)
    ids = [f"v0/{i}" for i in range(6)]
    monkeypatch.setattr(S, "_collect_train_records", lambda: ids)
    rng = np.random.default_rng(0)
    monkeypatch.setattr(S, "_build_features",
                        lambda ids, **kwargs: (rng.normal(size=(6, 4)).astype(np.float32),
                                               np.zeros(6, dtype=np.float64)))

    rc = S.main(["--out-dir", str(tmp_path / "s"), "--seeds", "201", "--k", "2"])
    assert rc == 0
    assert calls == [0]