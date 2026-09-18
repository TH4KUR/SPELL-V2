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

def test_score_one_lit_reports_progress_every_report_every_utterances(monkeypatch):
    """PROTOCOL §3.23/§3.24: within ONE checkpoint's pass over the train
    pool (the part that dominates wall-clock, given only 2 late checkpoints
    by default) on_progress must fire every `report_every` NEW rows, plus
    once more at the end even off a clean multiple."""
    n = 5
    fake_batches = [
        {"lengths": torch.tensor([1]), "text_lengths": torch.tensor([1]),
         "tokens": torch.zeros(1, 1, 1), "text_ids": torch.zeros(1, 1),
         "utterance_ids": [f"u/{i}"]}
        for i in range(n)
    ]

    class FakeVocab:
        pad_id = 0

    class FakeModel:
        def __call__(self, stream, lengths):
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

    seen = []
    out = S._score_one_lit(FakeLit(), [f"u/{i}" for i in range(n)],
                           report_every=2, on_progress=lambda *a: seen.append(a))
    assert len(out) == n
    assert [s[0] for s in seen] == [2, 4, 5]


def test_score_all_threads_ckpt_index_through_progress_callback(monkeypatch):
    """on_ckpt_done must see BOTH which checkpoint is in flight and that
    checkpoint's own within-checkpoint progress -- previously _score_all was
    `[_score_one_lit(lit, ids) for lit in lits]` with zero visibility
    across the pass. Deliberately uneven per-checkpoint sleep proves the
    reported duration is that checkpoint's own, not cumulative."""
    sleep_schedule = [0.05, 0.02, 0.08]
    call_index = {"i": 0}

    def fake_score_one_lit(lit, ids, report_every=2000, on_progress=None):
        time.sleep(sleep_schedule[call_index["i"]])
        call_index["i"] += 1
        if on_progress is not None:
            on_progress(len(ids), len(ids), 0.0, 0.0)
        return {uid: (1.0, 0.5) for uid in ids}

    monkeypatch.setattr(S, "_score_one_lit", fake_score_one_lit)

    seen = []
    S._score_all(["ck1", "ck2", "ck3"], ["u/1"], report_every=2000,
                on_ckpt_done=lambda *a: seen.append(a))
    assert [s[0] for s in seen] == [1, 2, 3]
    assert all(s[1] == 3 for s in seen)


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