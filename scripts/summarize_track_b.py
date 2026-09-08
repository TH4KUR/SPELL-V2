#!/usr/bin/env python3
"""Track B results summarizer (HANDOFF queue item 3).

    python scripts/summarize_track_b.py [--format md|csv] [--out PATH]

Collects every Track B bundle under the relay runs dir AND the canonical
archive (dedup: the archive copy wins), and reports FINAL-EPOCH val aggregates
per the §3.3 final-checkpoint law:

  * aggregates only — per-utterance rows in metrics.parquet never leak in;
  * a bundle without a COMPLETED marker is flagged, never hidden (§10 item 9);
  * `verified` is a BEST-EFFORT hint: set when an outputs/**/summary.json from
    evaluate_track_b.py references this bundle's last.ckpt. Official numbers
    still come from evaluate_track_b.py itself (§7 gate-signal policy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ckpt_bundle import METRICS_COLUMNS  # noqa: E402
from config import PROJECT_ROOT, load_paths  # noqa: E402

TABLE_COLUMNS = [
    "run_id", "subset_manifest", "subset_seed", "final_epoch",
    "val_wer", "val_cer", "val_loss", "completed", "verified", "location",
]


def _final_epoch_aggregates(bundle: Path) -> dict | None:
    """Final-epoch (max) val aggregate rows: {wer, loss, cer, final_epoch}.

    None when metrics.parquet is missing or carries no val aggregate rows."""
    mp = bundle / "metrics.parquet"
    if not mp.exists():
        return None
    df = pd.read_parquet(mp, columns=METRICS_COLUMNS)
    agg = df[(df["split"] == "val") & (df["utterance_id"].isna())]
    if agg.empty:
        return None
    final = int(agg["epoch"].max())
    snap = agg[agg["epoch"] == final]
    out = {"final_epoch": final}
    for metric, col in (("wer", "val_wer"), ("cer", "val_cer"), ("loss", "val_loss")):
        sub = snap[snap["metric"] == metric]
        out[col] = float(sub["value"].iloc[0]) if not sub.empty else None
    return out


def _verified_hint(bundle: Path, outputs_root: Path) -> bool:
    """True iff some evaluate_track_b summary.json references this bundle."""
    if not outputs_root.is_dir():
        return False
    for summary in outputs_root.glob("*/summary.json"):
        try:
            ckpt = json.loads(summary.read_text(encoding="utf-8")).get("checkpoint", "")
        except (OSError, json.JSONDecodeError):
            continue
        if bundle.name in str(ckpt) and str(ckpt).endswith("last.ckpt"):
            return True
    return False


def collect_bundles(runs_root: Path | None, archive_root: Path | None) -> list[dict]:
    """Depth-3 discovery mirroring drain_runs.sh: <root>/<track>/<run_id>/.
    Archive rows supersede relay duplicates. Missing roots are noted, not fatal."""
    collect_bundles.skips = []
    by_id: dict[str, dict] = {}
    for label, root in (("relay", runs_root), ("archive", archive_root)):
        if root is None or not Path(root).is_dir():
            collect_bundles.skips.append(f"{label} root missing/absent: {root}")
            continue
        root = Path(root)
        for cand in sorted(root.glob("*/*")):
            if not cand.is_dir():
                continue
            run_id = str(cand.relative_to(root))
            has_metrics = (cand / "metrics.parquet").exists()
            has_manifest = (cand / "run_manifest.json").exists()
            if not has_metrics and not has_manifest:
                continue                       # not a bundle dir at all
            if not has_metrics:
                collect_bundles.skips.append(
                    f"{label}: {run_id} has no metrics.parquet — skipped")
                continue
            prev = by_id.get(run_id)
            if prev is not None and prev["location"] == "archive":
                continue  # already have the drained copy
            by_id[run_id] = _row_for(cand, run_id, label)
    return [by_id[k] for k in sorted(by_id)]


def _row_for(bundle: Path, run_id: str, label: str) -> dict:
    agg = _final_epoch_aggregates(bundle) or {}
    row = {
        "run_id": run_id,
        "subset_manifest": None, "subset_seed": None,
        "final_epoch": agg.get("final_epoch"),
        "val_wer": agg.get("val_wer"), "val_cer": agg.get("val_cer"),
        "val_loss": agg.get("val_loss"),
        "completed": (bundle / "COMPLETED").exists(),
        "verified": _verified_hint(bundle, PROJECT_ROOT / "outputs"),
        "location": label,
    }
    mf = bundle / "run_manifest.json"
    if mf.exists():
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            row["subset_manifest"] = m.get("subset_manifest")
            row["subset_seed"] = m.get("subset_seed")
        except json.JSONDecodeError:
            collect_bundles.skips.append(f"unreadable run_manifest.json: {bundle}")
    return row


def render(rows: list[dict], *, fmt: str) -> str:
    if fmt not in ("md", "csv"):
        raise ValueError(f"unknown format {fmt!r}")
    cols = TABLE_COLUMNS

    def fmt_cell(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, bool):
            return "yes" if v else "NO"
        return str(v)

    if fmt == "csv":
        import csv as _csv
        import io
        buf = io.StringIO()
        w = _csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt_cell(r[c]) for c in cols])
        return buf.getvalue()
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt_cell(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    import argparse
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path(paths.runs_dir),
                    help="relay runs dir (default: paths.yaml runs_dir)")
    ap.add_argument("--archive", type=Path, default=Path(paths.archive_root),
                    help="canonical archive (default: paths.yaml archive_root)")
    ap.add_argument("--format", choices=["md", "csv"], default="md")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: stdout)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = collect_bundles(args.runs_root, args.archive)
    for note in collect_bundles.skips:
        print(f"[summarize] NOTE {note}", file=sys.stderr)
    table = render(rows, fmt=args.format)
    if args.out is None or str(args.out) == "-":
        sys.stdout.write(table)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
        print(f"[summarize] {len(rows)} bundles -> {args.out}")
    if any(r["completed"] is False for r in rows):
        print("[summarize] WARNING: some bundles lack COMPLETED — flagged in table",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())