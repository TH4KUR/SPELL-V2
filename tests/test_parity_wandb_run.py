"""parity/wandb_run.py: the PROTOCOL §3.23-§3.25 logging surface for the parity
scripts, and its separation from the formal-run namespace (§3.17(d), pinned by
tests/test_wandb_naming_law.py).

* §3.23: parity runs use their own job_type and `<tool>/`-prefixed series; never
  `train/*` or `val/*`, and never a formal run id (`<track>/<stem>_seed.._job.._t..`).
* §3.25(b): `finish_run` is called from a `finally:` and must reflect the real
  outcome (exit_code=1 plus error/traceback on the summary when unwinding).
"""

from __future__ import annotations

import re

import pytest

from parity import wandb_run as W

FORMAL_ID = re.compile(r"^track_[ab]/.+_seed\d+_job\d+_t\d+$")


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = []

    def log(self, payload, step=None):
        self.logged.append(dict(payload))

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


def test_init_run_disabled_mode_runs_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "WANDB_DIR", tmp_path / "wandb")
    run = W.init_run(job_type="parity_eval", name="parity-eval-val-local", project="spell-rq2",
                     entity=None, mode="disabled", config={"a": 1}, tags=["parity"])
    run.log({"parity_eval/x": 1})
    run.finish()


@pytest.mark.parametrize("name", [
    "track_b/random_25pct_seed101_seed101_job2680643_t3",
    "track_a/dnsmos_25pct_seed0_job42_t7",
    "track_b/anything",
])
def test_run_names_in_the_formal_namespace_are_refused(name):
    with pytest.raises(ValueError, match="formal"):
        W.check_run_name(name)


@pytest.mark.parametrize("name", ["parity-eval-val-123", "parity-eval-test-local", "parity-rescore-local"])
def test_parity_run_names_are_flat_and_not_formal_ids(name):
    W.check_run_name(name)
    assert not FORMAL_ID.match(name) and "/" not in name


@pytest.mark.parametrize("prefix", ["train", "val", "train/loss", "val/wer"])
def test_series_prefixes_of_the_formal_contract_are_refused(prefix):
    with pytest.raises(ValueError, match="3.17"):
        W.log(FakeRun(), prefix, {"x": 1})


def test_log_prefixes_every_key_and_logs_once():
    run = FakeRun()
    W.log(run, "parity_eval", {"utts_done": 3, "eta_s": 1.5})
    assert run.logged == [{"parity_eval/utts_done": 3, "parity_eval/eta_s": 1.5}]


def test_finish_run_reports_success_on_a_clean_exit():
    run = FakeRun()
    try:
        pass
    finally:
        W.finish_run(run)
    assert run.finished == [0] and "error" not in run.summary


def test_finish_run_marks_failure_and_records_the_error_when_unwinding():
    run = FakeRun()
    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        finally:
            W.finish_run(run)
    assert run.finished == [1]
    assert run.summary["error"] == "RuntimeError: boom"
    assert "boom" in run.summary["traceback"] and "Traceback" in run.summary["traceback"]


def test_finish_run_marks_a_nonzero_return_code_as_failed_with_its_message():
    run = FakeRun()
    try:
        pass
    finally:
        W.finish_run(run, rc=4, message="stored mean_wer not reproduced")
    assert run.finished == [1] and run.summary["error"] == "stored mean_wer not reproduced"


def test_finish_run_tolerates_a_missing_run():
    W.finish_run(None)
