#!/usr/bin/env python3
"""W&B backfill for pre-§3.17 bundles (HANDOFF queue item 4).

    python scripts/backfill_wandb.py --bundle runs/track_b/<id>            # dry-run
    python scripts/backfill_wandb.py --bundle runs/track_b/<id> --commit   # real

Streams the FROZEN epoch-aggregate val series (val/wer, val/loss, val/cer) from
a bundle's metrics.parquet into W&B with step=epoch, resuming the run under the
§3.17(d) naming-law id. LAWS (pinned by tests/test_backfill_wandb.py):
  * dry-run is the DEFAULT — nothing reaches W&B without --commit;
  * ONLY the three frozen val series are streamed (§3.17(a) boundary);
  * per-utterance rows NEVER stream to W&B (§3.17);
  * a bundle without metrics.parquet or run_manifest.json is refused loudly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ckpt_bundle import METRICS_COLUMNS  # noqa: E402

FROZEN_VAL_SERIES = ("val/wer", "val/loss", "val/cer")


def derive_run_id(bundle: Path) -> str:
    """§3.17(d) id: <track>/<stem> for a bundle at <root>/<track>/<stem>."""
    bundle = Path(bundle)
    if bundle.parent.name and bundle.name:
        return f"{bundle.parent.name}/{bundle.name}"
    raise ValueError(f"cannot derive run id from bundle path: {bundle}")


def _aggregate_plan(bundle: Path) -> dict[int, dict[str, float]]:
    """{epoch: {val/wer: v, val/loss: v, val/cer: v}} from aggregate rows only."""
    mp = bundle / "metrics.parquet"
    if not mp.exists():
        raise FileNotFoundError(f"bundle has no metrics.parquet: {bundle}")
    if not (bundle / "run_manifest.json").exists():
        raise FileNotFoundError(f"bundle has no run_manifest.json — refusing: {bundle}")
    df = pd.read_parquet(mp, columns=METRICS_COLUMNS)
    agg = df[(df["split"] == "val") & (df["utterance_id"].isna())]
    plan: dict[int, dict[str, float]] = {}
    for _, r in agg.iterrows():
        name = f"val/{r['metric']}"
        if name not in FROZEN_VAL_SERIES:
            continue
        plan.setdefault(int(r["epoch"]), {})[name] = float(r["value"])
    return {e: plan[e] for e in sorted(plan)}


def backfill(
    bundle: Path,
    *,
    project: str,
    entity: str | None = None,
    run_id: str | None = None,
    dry_run: bool = True,
    wandb_module=None,
) -> str:
    """Stream (or, in dry-run, only plan) the frozen val series. Returns a plan
    summary string either way."""
    bundle = Path(bundle)
    plan = _aggregate_plan(bundle)
    rid = run_id or derive_run_id(bundle)
    n_points = sum(len(v) for v in plan.values())
    summary = (f"run_id={rid} project={project} epochs={len(plan)} "
               f"points={n_points} series={sorted({k for v in plan.values() for k in v})}")
    if dry_run:
        return f"[backfill] DRY-RUN (no W&B calls): {summary}"
    if wandb_module is None:                      # production path
        import wandb as wandb_module              # noqa: F811
    run = wandb_module.init(id=rid, project=project,
                            entity=entity or None, resume="allow")
    for epoch, payload in plan.items():
        run.log(dict(payload), step=epoch)
    wandb_module.finish()
    return f"[backfill] COMMITTED: {summary}"


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path,
                    help="bundle dir (…/<track>/<run_id>/) with metrics.parquet")
    ap.add_argument("--project", default="spell-rq2")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--run-id", default=None,
                    help="override the derived §3.17(d) id (e.g. for legacy dirs)")
    ap.add_argument("--commit", action="store_true",
                    help="actually stream to W&B (default: dry-run)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = backfill(args.bundle, project=args.project, entity=args.entity,
                          run_id=args.run_id, dry_run=not args.commit)
    except FileNotFoundError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())