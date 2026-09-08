"""scripts/summarize_track_b.py contract (HANDOFF queue item 3).

Summarizes Track B bundles (relay + archive) into a results table reporting
FINAL-EPOCH val aggregates per the §3.3 final-checkpoint law. The live val
series stays the §7 gate signal; evaluate_track_b.py remains the source of
official final numbers — the table only carries a best-effort `verified` hint
when an eval summary.json references the bundle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import summarize_track_b as S  # noqa: E402
from ckpt_bundle import METRICS_COLUMNS  # noqa: E402


def make_bundle(
    root: Path, track: str, run_id: str, *, epochs: int = 3,
    completed: bool = True, with_metrics: bool = True,
) -> Path:
    """Synthetic bundle: metrics.parquet + run_manifest.json (+ COMPLETED)."""
    b = root / track / run_id
    b.mkdir(parents=True, exist_ok=True)
    if with_metrics:
        rows = []
        for e in range(1, epochs + 1):
            rows.append({"epoch": e, "split": "val", "utterance_id": None,
                         "metric": "wer", "value": 0.7 - e * 0.05})
            rows.append({"epoch": e, "split": "val", "utterance_id": None,
                         "metric": "loss", "value": 2.0 - e * 0.1})
            rows.append({"epoch": e, "split": "val", "utterance_id": None,
                         "metric": "cer", "value": 0.4 - e * 0.05})
            rows.append({"epoch": e, "split": "val", "utterance_id": "x/1",
                         "metric": "wer", "value": 0.5})   # per-utt row (ignored)
        pd.DataFrame(rows, columns=METRICS_COLUMNS).to_parquet(
            b / "metrics.parquet", index=False)
    (b / "run_manifest.json").write_text(json.dumps({
        "subset_manifest": "subsets/random_25pct_seed101.txt",
        "subset_seed": 101, "train_seed": 20260826,
        "status": "completed" if completed else "running",
        "protocol_revision": "universe-v2",
        "started_utc": "2026-09-08T00:00:00+00:00",
        "finished_utc": "2026-09-08T01:00:00+00:00" if completed else None,
    }, sort_keys=True), encoding="utf-8")
    if completed:
        (b / "COMPLETED").touch()
    return b


def test_reports_final_epoch_val_aggregates(tmp_path):
    """§3.3: final-checkpoint evaluation only — the table shows max-epoch rows."""
    make_bundle(tmp_path / "runs", "track_b", "r1", epochs=5)
    rows = S.collect_bundles(tmp_path / "runs", None)
    assert len(rows) == 1
    r = rows[0]
    assert r["run_id"] == "track_b/r1"
    assert r["final_epoch"] == 5
    assert r["val_wer"] == pytest.approx(0.7 - 5 * 0.05)
    assert r["val_loss"] == pytest.approx(2.0 - 5 * 0.1)
    assert r["val_cer"] == pytest.approx(0.4 - 5 * 0.05)


def test_per_utterance_rows_never_enter_the_table(tmp_path):
    """Aggregates only — per-utt rows in the bundle must not leak into or scale
    the summary (§3.17: per-utterance data lives in metrics.parquet, not tables)."""
    b = make_bundle(tmp_path / "runs", "track_b", "r1", epochs=2)
    df = pd.read_parquet(b / "metrics.parquet")
    extra = pd.DataFrame([
        {"epoch": 2, "split": "val", "utterance_id": f"x/{i}",
         "metric": "wer", "value": 9.9} for i in range(50)],
        columns=METRICS_COLUMNS)
    pd.concat([df, extra], ignore_index=True).to_parquet(
        b / "metrics.parquet", index=False)
    rows = S.collect_bundles(tmp_path / "runs", None)
    assert rows[0]["val_wer"] == pytest.approx(0.6)   # 0.7 - 2*0.05, not 9.9


def test_relay_and_archive_dedup_prefers_archive(tmp_path):
    make_bundle(tmp_path / "runs", "track_b", "r1")
    make_bundle(tmp_path / "archive", "track_b", "r1")
    rows = S.collect_bundles(tmp_path / "runs", tmp_path / "archive")
    assert len(rows) == 1
    assert rows[0]["location"].endswith("archive")


def test_unmarked_bundle_flagged_not_hidden(tmp_path):
    """A bundle without COMPLETED is reported WITH a flag — never silently
    dropped, never treated as complete (§10 item 9 discipline)."""
    make_bundle(tmp_path / "runs", "track_b", "r_partial", completed=False)
    rows = S.collect_bundles(tmp_path / "runs", None)
    assert len(rows) == 1
    assert rows[0]["completed"] is False


def test_bundle_without_metrics_skipped_loudly(tmp_path):
    make_bundle(tmp_path / "runs", "track_b", "r_junk", with_metrics=False)
    rows = S.collect_bundles(tmp_path / "runs", None)
    assert rows == []
    assert S.collect_bundles.skips, "skip must be recorded loudly, not silent"


def test_missing_roots_are_not_fatal(tmp_path):
    rows = S.collect_bundles(tmp_path / "nope_runs", tmp_path / "nope_archive")
    assert rows == []


def test_render_md_and_csv(tmp_path):
    make_bundle(tmp_path / "runs", "track_b", "r1", epochs=2)
    rows = S.collect_bundles(tmp_path / "runs", None)
    md = S.render(rows, fmt="md")
    csv = S.render(rows, fmt="csv")
    assert "val_wer" in md.splitlines()[0]
    assert "val_wer" in csv.splitlines()[0]
    assert "0.6000" in md or "0.600" in md          # epoch-2 wer value rendered


def test_manifest_fields_flow_into_table(tmp_path):
    make_bundle(tmp_path / "runs", "track_b", "random_25pct_seed101_job1_t1")
    rows = S.collect_bundles(tmp_path / "runs", None)
    assert rows[0]["subset_manifest"] == "subsets/random_25pct_seed101.txt"
    assert rows[0]["subset_seed"] == 101