"""scripts/backfill_wandb.py contract (HANDOFF queue item 4).

Backfills the §3.17 W&B surface for bundles that predate the logging contract,
streaming ONLY the frozen epoch-aggregate val series. Laws pinned here:
  * default is DRY-RUN — nothing reaches W&B without an explicit --commit;
  * series names are EXACTLY the frozen set (§3.17(a): this assertion IS the
    logger-boundary pin for the backfill path);
  * per-utterance rows NEVER stream to W&B (§3.17);
  * run-id follows the §3.17(d) naming law (runs/<track>/<stem> -> track/<stem>).
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

import backfill_wandb as B  # noqa: E402
from ckpt_bundle import METRICS_COLUMNS  # noqa: E402

FROZEN_VAL_SERIES = {"val/wer", "val/loss", "val/cer"}


class SpyWandb:
    """Records init/log/finish; nothing touches the network."""

    def __init__(self):
        self.inits: list[dict] = []
        self.logs: list[tuple[dict, int]] = []
        self.finished = 0

    def init(self, **kwargs):
        self.inits.append(kwargs)
        return self

    def log(self, payload, step=None):
        self.logs.append((dict(payload), int(step)))

    def finish(self):
        self.finished += 1


def make_bundle(root: Path, track: str, run_id: str, *, epochs: int = 4) -> Path:
    b = root / track / run_id
    b.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in range(1, epochs + 1):
        rows.append({"epoch": e, "split": "val", "utterance_id": None,
                     "metric": "wer", "value": 0.7 - e * 0.05})
        rows.append({"epoch": e, "split": "val", "utterance_id": None,
                     "metric": "loss", "value": 2.0 - e * 0.1})
        rows.append({"epoch": e, "split": "val", "utterance_id": None,
                     "metric": "cer", "value": 0.4 - e * 0.05})
        rows.append({"epoch": e, "split": "val", "utterance_id": f"x/{e}",
                     "metric": "wer", "value": 9.9})          # per-utt — forbidden
        rows.append({"epoch": e, "split": "train", "utterance_id": f"t/{e}",
                     "metric": "loss", "value": 1.0})          # per-utt — forbidden
    pd.DataFrame(rows, columns=METRICS_COLUMNS).to_parquet(
        b / "metrics.parquet", index=False)
    (b / "run_manifest.json").write_text("{}", encoding="utf-8")
    return b


def test_dry_run_is_default_and_never_inits(tmp_path):
    b = make_bundle(tmp_path, "track_b", "r1")
    spy = SpyWandb()
    plan = B.backfill(b, project="spell-rq2", wandb_module=spy, dry_run=True)
    assert spy.inits == [] and spy.logs == []
    assert plan is not None and "val/wer" in str(plan)


def test_commit_streams_exactly_the_frozen_val_series(tmp_path):
    """§3.17(a) pin: ONLY val/wer|loss|cer, step=epoch, exactly once per epoch."""
    b = make_bundle(tmp_path, "track_b", "r1", epochs=4)
    spy = SpyWandb()
    B.backfill(b, project="spell-rq2", wandb_module=spy, dry_run=False)
    assert len(spy.inits) == 1
    assert spy.inits[0]["resume"] == "allow"
    seen = {}
    for payload, step in spy.logs:
        assert set(payload) <= FROZEN_VAL_SERIES, f"forbidden series: {set(payload)}"
        for name, value in payload.items():
            key = (name, step)
            assert key not in seen, f"duplicate emission {key}"
            seen[key] = value
    assert {s for s, _ in seen} == FROZEN_VAL_SERIES
    # steps are the 1-based epochs, monotonic
    steps = sorted({step for _, step in spy.logs})
    assert steps == [1, 2, 3, 4]
    # spot-check one value
    wer_step2 = [v for p, s in spy.logs if s == 2 for k, v in p.items()
                 if k == "val/wer"]
    assert wer_step2 == [pytest.approx(0.7 - 2 * 0.05)]
    assert spy.finished >= 1


def test_per_utterance_data_never_streams(tmp_path):
    b = make_bundle(tmp_path, "track_b", "r1")
    spy = SpyWandb()
    B.backfill(b, project="spell-rq2", wandb_module=spy, dry_run=False)
    for payload, _ in spy.logs:
        assert not any(str(v).startswith("x/") or str(v).startswith("t/")
                       for v in payload.values())


def test_run_id_derived_from_bundle_path(tmp_path):
    b = make_bundle(tmp_path, "track_b", "random_25pct_seed101_job5_t2")
    spy = SpyWandb()
    B.backfill(b, project="spell-rq2", wandb_module=spy, dry_run=False)
    assert spy.inits[0]["id"] == "track_b/random_25pct_seed101_job5_t2"


def test_run_id_override_respected(tmp_path):
    b = make_bundle(tmp_path, "track_b", "r1")
    spy = SpyWandb()
    B.backfill(b, project="spell-rq2", wandb_module=spy, dry_run=False,
               run_id="legacy/manual_id_1")
    assert spy.inits[0]["id"] == "legacy/manual_id_1"


def test_refuses_bundle_without_metrics(tmp_path):
    b = make_bundle(tmp_path, "track_b", "r1")
    (b / "metrics.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        B.backfill(b, project="spell-rq2", wandb_module=SpyWandb(), dry_run=False)