"""scripts/parity_rescore_track_b.py - Phase-0 diagnostic.

Re-scores a Track-B ``predictions.jsonl`` two ways: the macro mean of
per-utterance jiwer WERs (what Track B reports, ctc.py:80-94) and CORPUS WER
(total edits / total reference words, what RAVEn reports), the latter overall and
inside reference-length quartiles. Given the bundle's summary.json it must
reproduce the stored ``mean_wer`` to 1e-9 or fail loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import ctc as ctc_lib  # noqa: E402
import parity_rescore_track_b as S  # noqa: E402

TRACK_B_ROW_KEYS = {"utterance_id", "ref", "hyp", "loss", "wer", "cer"}


def _track_b_rows(pairs):
    """Rows exactly as scripts/evaluate_track_b.py writes them."""
    rows = []
    for i, (ref, hyp) in enumerate(pairs):
        scoreable = bool(ref.strip())
        rows.append({
            "utterance_id": f"v/{i}", "ref": ref, "hyp": hyp, "loss": 1.0,
            "wer": ctc_lib.wer_one(ref, hyp) if scoreable else None,
            "cer": ctc_lib.cer_one(ref, hyp) if scoreable else None,
        })
    assert all(set(r) == TRACK_B_ROW_KEYS for r in rows)
    return rows


def _write_bundle(tmp_path: Path, pairs, *, mean_wer="auto", n_utterances="auto"):
    rows = _track_b_rows(pairs)
    scored = [r["wer"] for r in rows if r["wer"] is not None]
    (tmp_path / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = {
        "n_utterances": len(rows) if n_utterances == "auto" else n_utterances,
        "mean_wer": (sum(scored) / len(scored) if scored else None) if mean_wer == "auto" else mean_wer,
        "split": "test",
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return tmp_path / "predictions.jsonl", tmp_path / "summary.json"


PAIRS = [
    ("a b c d e f g h i j", "a b c d e f g h i j"),
    ("x", "y"),
    ("the quick brown fox", "the quick red fox jumps"),
    ("to be or not to be", "to be"),
    ("hello", ""),
    ("one two three", "one two three"),
    ("alpha beta gamma delta epsilon", "alpha beta gamma delta epsilon"),
    ("a b", "a b c"),
]


def _run(tmp_path, pairs=PAIRS, extra=(), **bundle_kw):
    pred, summ = _write_bundle(tmp_path, pairs, **bundle_kw)
    out = tmp_path / "rescore.json"
    argv = ["--predictions", str(pred), "--out", str(out), "--wandb-mode", "disabled", *extra]
    return argv, out, summ


# ------------------------------------------------------------------- core --

def test_macro_and_corpus_differ_on_the_canonical_example(tmp_path):
    pairs = [("a b c d e f g h i j", "a b c d e f g h i j"), ("x", "y")]
    argv, out, _ = _run(tmp_path, pairs)
    assert S.main(argv) == 0
    res = json.loads(out.read_text())
    assert res["corpus_wer"] == pytest.approx(1 / 11)
    assert res["macro_wer"] == pytest.approx(0.5)
    assert res["total_edits"] == 1 and res["total_ref_words"] == 11


def test_reproduces_stored_mean_wer_to_1e9(tmp_path):
    argv, out, summ = _run(tmp_path)
    assert S.main(argv + ["--summary", str(summ)]) == 0
    res = json.loads(out.read_text())
    stored = json.loads(summ.read_text())["mean_wer"]
    assert abs(res["macro_wer"] - stored) <= 1e-9
    assert res["stored_mean_wer"] == stored
    assert res["abs_diff_vs_stored"] <= 1e-9


def test_fails_loudly_when_stored_mean_wer_cannot_be_reproduced(tmp_path, capsys):
    argv, out, summ = _run(tmp_path, mean_wer=0.123456)
    rc = S.main(argv + ["--summary", str(summ)])
    assert rc == 4
    err = capsys.readouterr().err
    assert "0.123456" in err and "mean_wer" in err
    assert not out.exists()                            # no results are published from a failed check


def test_fails_loudly_when_row_count_disagrees_with_summary(tmp_path, capsys):
    argv, out, summ = _run(tmp_path, n_utterances=len(PAIRS) + 1)
    assert S.main(argv + ["--summary", str(summ)]) == 4
    assert "n_utterances" in capsys.readouterr().err


def test_fails_when_a_stored_per_row_wer_disagrees_with_a_recomputation(tmp_path, capsys):
    pred, summ = _write_bundle(tmp_path, PAIRS)
    rows = [json.loads(l) for l in pred.read_text().splitlines()]
    rows[2]["wer"] += 0.01
    pred.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc = S.main(["--predictions", str(pred), "--out", str(tmp_path / "o.json"), "--wandb-mode", "disabled"])
    assert rc == 4
    assert "v/2" in capsys.readouterr().err


def test_unscoreable_rows_are_skipped_like_track_b_and_counted(tmp_path):
    pairs = PAIRS + [("", "spurious words"), ("   ", "")]
    argv, out, summ = _run(tmp_path, pairs)
    assert S.main(argv + ["--summary", str(summ)]) == 0
    res = json.loads(out.read_text())
    assert res["n_rows"] == len(pairs) and res["n_scored"] == len(PAIRS) and res["n_unscorable"] == 2
    expected_macro = sum(ctc_lib.wer_one(r, h) for r, h in PAIRS) / len(PAIRS)
    assert res["macro_wer"] == pytest.approx(expected_macro, abs=1e-12)


def test_jiwer_alignment_and_independent_edit_distance_must_agree(tmp_path, monkeypatch, capsys):
    argv, out, _ = _run(tmp_path)
    monkeypatch.setattr(S.metrics, "pair_edits", lambda ref, hyp: 10_000)
    assert S.main(argv) == 4
    assert "disagree" in capsys.readouterr().err


def test_quartiles_are_reported_with_both_statistics(tmp_path):
    argv, out, _ = _run(tmp_path)
    assert S.main(argv) == 0
    q = json.loads(out.read_text())["quartiles"]
    assert len(q) == 4 and sum(b["n_utts"] for b in q) == len(PAIRS)
    for b in q:
        assert set(b) >= {"index", "lo_words", "hi_words", "n_utts", "corpus_wer", "macro_wer",
                          "total_edits", "total_ref_words"}
    assert q[0]["lo_words"] <= q[0]["hi_words"] <= q[1]["lo_words"]


def test_fewer_than_four_scoreable_rows_skips_quartiles_but_still_scores(tmp_path):
    argv, out, _ = _run(tmp_path, PAIRS[:3])
    assert S.main(argv) == 0
    res = json.loads(out.read_text())
    assert res["quartiles"] is None and res["corpus_wer"] > 0


def test_prints_a_small_table(tmp_path, capsys):
    argv, out, _ = _run(tmp_path)
    S.main(argv)
    text = capsys.readouterr().out
    assert "macro-mean WER" in text and "corpus WER" in text and "Q1" in text and "Q4" in text


def test_missing_file_and_no_scoreable_rows_are_errors(tmp_path, capsys):
    assert S.main(["--predictions", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "o.json"),
                   "--wandb-mode", "disabled"]) == 2
    pred, _ = _write_bundle(tmp_path, [("", "x")])
    assert S.main(["--predictions", str(pred), "--out", str(tmp_path / "o.json"),
                   "--wandb-mode", "disabled"]) == 2
    assert "no scoreable" in capsys.readouterr().err


def test_malformed_rows_are_rejected(tmp_path, capsys):
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"utterance_id": "a", "ref": "x"}) + "\n")
    assert S.main(["--predictions", str(p), "--out", str(tmp_path / "o.json"),
                   "--wandb-mode", "disabled"]) == 2
    assert "hyp" in capsys.readouterr().err


# ------------------------------------------------------------------ wandb --

class _FakeRun:
    def __init__(self):
        self.logged, self.summary, self.finished = [], {}, []

    def log(self, payload, step=None):
        self.logged.append(dict(payload))

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


def test_wandb_run_opens_first_logs_its_series_and_finishes_clean(tmp_path, monkeypatch):
    run = _FakeRun()
    opened = {}

    def fake_init(**kw):
        opened.update(kw)
        return run

    monkeypatch.setattr(S.wandb_run, "init_run", fake_init)
    argv, out, _ = _run(tmp_path)
    argv = [a for a in argv if a not in ("--wandb-mode", "disabled")]
    assert S.main(argv) == 0
    assert opened["job_type"] == "parity_rescore" and "/" not in opened["name"]
    keys = {k for payload in run.logged for k in payload}
    assert keys and all(k.startswith("parity_rescore/") for k in keys)
    assert run.finished == [0]
    assert run.summary["parity_rescore/corpus_wer"] == pytest.approx(json.loads(out.read_text())["corpus_wer"])


def test_wandb_run_is_marked_failed_with_the_error_when_the_check_crashes(tmp_path, monkeypatch):
    run = _FakeRun()
    monkeypatch.setattr(S.wandb_run, "init_run", lambda **kw: run)
    monkeypatch.setattr(S, "rescore", lambda rows: (_ for _ in ()).throw(RuntimeError("kaput")))
    argv, _, _ = _run(tmp_path)
    argv = [a for a in argv if a not in ("--wandb-mode", "disabled")]
    with pytest.raises(RuntimeError, match="kaput"):
        S.main(argv)
    assert run.finished == [1] and "kaput" in run.summary["error"]
