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
import time
from pathlib import Path

import pandas as pd
import pytest
import torch

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
        lambda lits, ids, **kwargs: [
            {"utterance_id": "v0/2", "loss_mean": 2.0, "el2n_mean": 0.3,
             "n_ckpts": 2},
            {"utterance_id": "v0/1", "loss_mean": 1.0, "el2n_mean": 0.8,
             "n_ckpts": 2},
        ])
    out = tmp_path / "scores" / "proxy_scores.parquet"
    rc = S.main(["--bundle", str(tmp_path / "b"), "--out", str(out),
                "--wandb-mode", "disabled"])
    assert rc == 0
    df = pd.read_parquet(out)
    assert list(df.columns) == ["utterance_id", "loss_mean", "el2n_mean",
                                "n_ckpts"]
    assert list(df["utterance_id"]) == ["v0/1", "v0/2"]    # sorted, train only
    assert (df["n_ckpts"] == 2).all()


# ------------------------------------------------------------- progress/wandb

def test_score_one_lit_reports_on_a_time_cadence_not_a_row_count(monkeypatch):
    """PROTOCOL §3.23/§3.24: within ONE checkpoint's pass over the train
    pool (the part that dominates wall-clock, given only 2 late checkpoints
    by default) on_progress must fire on a WALL-CLOCK cadence, not a fixed
    row count -- mirrors score_dnsmos.py's identical fix (a count-based
    threshold silently assumes a throughput rate). A fake, manually-
    advanced clock proves reports land once report_every_s has actually
    passed, with a forced final report covering the tail."""
    n = 6
    fake_batches = [
        {"lengths": torch.tensor([1]), "text_lengths": torch.tensor([1]),
         "tokens": torch.zeros(1, 1, 1), "text_ids": torch.zeros(1, 1),
         "utterance_ids": [f"u/{i}"]}
        for i in range(n)
    ]

    class FakeVocab:
        pad_id = 0

    clock = {"t": 0.0}
    costs = [5.0, 5.0, 15.0, 1.0, 1.0, 1.0]   # cumulative: 5,10,25,26,27,28
    call_index = {"i": 0}

    class FakeModel:
        def __call__(self, stream, lengths):
            clock["t"] += costs[call_index["i"]]
            call_index["i"] += 1
            return torch.zeros(1, 1, 1), torch.tensor([1])

    class FakeLit:
        model = FakeModel()
        vocab = FakeVocab()
        cfg = {"model": {"input_stream": 0}}
        blank_id = 0

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeRec:
        def __init__(self, uid):
            self.utterance_id = uid

    fake_recs = [FakeRec(f"u/{i}") for i in range(n)]
    monkeypatch.setattr(S, "load_records", lambda index_path, split: fake_recs)
    monkeypatch.setattr(S, "filter_records", lambda recs, ids, strict=False: recs)
    monkeypatch.setattr(S, "TokenDataset", lambda records, **kw: records)
    monkeypatch.setattr(S, "DataLoader", lambda dataset, **kw: fake_batches)
    monkeypatch.setattr(S.ctc_lib, "input_length_keep_mask",
                        lambda lengths, text_lengths: torch.ones(1, dtype=torch.bool))
    monkeypatch.setattr(S.ctc_lib, "ctc_loss_per_utt", lambda *a, **kw: torch.tensor([1.0]))
    monkeypatch.setattr(S.ctc_lib, "el2n_per_utt", lambda *a, **kw: torch.tensor([0.5]))
    monkeypatch.setattr(S.time, "monotonic", lambda: clock["t"])

    seen = []
    out = S._score_one_lit(FakeLit(), [f"u/{i}" for i in range(n)],
                           report_every_s=10.0, on_progress=lambda *a: seen.append(a))
    assert len(out) == n
    assert [s[0] for s in seen] == [2, 3, 6]
    assert all(s[1] == 6 for s in seen)


def test_score_one_lit_calls_real_filter_records_without_crashing(monkeypatch):
    """Regression test for job 2701229's crash: dataset.filter_records's
    type contract is `ids: set[str]` (its body does `ids - known`), but
    _score_one_lit passed the raw `ids` LIST straight through -- every
    OTHER caller in the codebase passes a set (from load_id_list), so this
    had never been exercised. Deliberately does NOT monkeypatch
    filter_records (unlike the test above), so it runs dataset.py's real
    implementation end-to-end against the exact list-shaped `ids` that
    main()'s _collect_train_records() actually returns."""
    n = 3

    class FakeVocab:
        pad_id = 0

    class FakeModel:
        def __call__(self, stream, lengths):
            raise AssertionError("should not be reached with an empty DataLoader")

    class FakeLit:
        model = FakeModel()
        vocab = FakeVocab()
        cfg = {"model": {"input_stream": 0}}
        blank_id = 0

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeRec:
        def __init__(self, uid):
            self.utterance_id = uid

    fake_recs = [FakeRec(f"u/{i}") for i in range(n)]
    monkeypatch.setattr(S, "load_records", lambda index_path, split: fake_recs)
    monkeypatch.setattr(S, "TokenDataset", lambda records, **kw: records)
    monkeypatch.setattr(S, "DataLoader", lambda dataset, **kw: [])

    ids = [f"u/{i}" for i in range(n)]     # a LIST, exactly like main()'s
                                            # _collect_train_records() returns
    out = S._score_one_lit(FakeLit(), ids)
    assert out == {}


def test_score_all_threads_ckpt_index_through_progress_callback(monkeypatch):
    """on_ckpt_done must see BOTH which checkpoint is in flight and that
    checkpoint's own within-checkpoint progress -- previously _score_all was
    `[_score_one_lit(lit, ids) for lit in lits]` with zero visibility
    across the pass. Deliberately uneven per-checkpoint sleep proves the
    reported duration is that checkpoint's own, not cumulative."""
    sleep_schedule = [0.05, 0.02, 0.08]
    call_index = {"i": 0}

    def fake_score_one_lit(lit, ids, report_every_s=30.0, on_progress=None):
        time.sleep(sleep_schedule[call_index["i"]])
        call_index["i"] += 1
        if on_progress is not None:
            on_progress(len(ids), len(ids), 0.0, 0.0)
        return {uid: (1.0, 0.5) for uid in ids}

    monkeypatch.setattr(S, "_score_one_lit", fake_score_one_lit)

    seen = []
    S._score_all(["ck1", "ck2", "ck3"], ["u/1"], report_every_s=30.0,
                on_ckpt_done=lambda *a: seen.append(a))
    assert [s[0] for s in seen] == [1, 2, 3]
    assert all(s[1] == 3 for s in seen)


def test_main_marks_wandb_run_failed_on_exception(monkeypatch, tmp_path):
    """Caught live (2026-09-18): job 2701229 crashed on filter_records's
    list/set type mismatch (see the fix in _score_one_lit) but still showed
    as a completed run in the W&B UI, because `run.finish()` with no args
    defaults to success regardless of whether an exception is propagating.
    A fake run object (not real wandb) proves main() now passes
    exit_code=1 on a crash."""
    calls = []

    class FakeRun:
        def finish(self, exit_code=0):
            calls.append(exit_code)

    monkeypatch.setattr(S, "_wandb_init", lambda **kw: FakeRun())
    monkeypatch.setattr(S, "_collect_train_records", lambda: ["v0/1"])
    monkeypatch.setattr(S, "enforce_gpu_policy", lambda: {"gpu_name": "fake"})

    def boom(bundle, late_frac):
        raise RuntimeError("boom")

    monkeypatch.setattr(S, "_late_ckpts", boom)

    with pytest.raises(RuntimeError, match="boom"):
        S.main(["--bundle", str(tmp_path / "b")])
    assert calls == [1]


def test_main_marks_wandb_run_succeeded_on_clean_exit(monkeypatch, tmp_path):
    calls = []

    class FakeRun:
        def finish(self, exit_code=0):
            calls.append(exit_code)

    monkeypatch.setattr(S, "_wandb_init", lambda **kw: FakeRun())
    monkeypatch.setattr(S, "log_final_summary_to_wandb", lambda df: None)
    monkeypatch.setattr(S, "enforce_gpu_policy", lambda: {"gpu_name": "fake"})
    monkeypatch.setattr(S, "_late_ckpts", lambda b, late_frac: ["ck1"])
    monkeypatch.setattr(S, "_load_lit", lambda c: object())
    monkeypatch.setattr(S, "_collect_train_records", lambda: ["v0/1"])
    monkeypatch.setattr(S, "load_records", lambda index_path, split: [])
    monkeypatch.setattr(S, "_preflight", lambda recs, k=50: None)
    monkeypatch.setattr(S, "_score_all", lambda lits, ids, **kw: [
        {"utterance_id": "v0/1", "loss_mean": 1.0, "el2n_mean": 0.5, "n_ckpts": 1},
    ])

    out = tmp_path / "scores" / "out.parquet"
    rc = S.main(["--bundle", str(tmp_path / "b"), "--out", str(out)])
    assert rc == 0
    assert calls == [0]


def test_wandb_init_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", bundle="runs/track_b/fake",
                        late_frac=0.75, n_recs=5)
    run.finish()


def test_log_scoring_progress_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", bundle="runs/track_b/fake",
                        late_frac=0.75, n_recs=5)
    S._log_scoring_progress(1, 2, 100, 200, 5.0, 5.0)
    run.finish()


def test_log_final_summary_to_wandb_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", bundle="runs/track_b/fake",
                        late_frac=0.75, n_recs=2)
    df = pd.DataFrame([
        {"utterance_id": "v0/1", "loss_mean": 1.0, "el2n_mean": 0.5, "n_ckpts": 2},
        {"utterance_id": "v0/2", "loss_mean": 2.0, "el2n_mean": 0.8, "n_ckpts": 2},
    ])
    S.log_final_summary_to_wandb(df)
    run.finish()