#!/usr/bin/env python
"""Recover official-test transcripts from the HF parquet the test split came from.

unpack.py (in datasets/LRS3) shows the flat ``test/*.wav`` files were extracted
from a local ``test-mattymchen/*.parquet`` dump whose Text column was never
saved. This script re-fetches that source (local parquet dir if given, else
candidate HuggingFace repos), maps each item's id -> text, and writes
``<test_dir>/<id>.txt`` in the standard ``Text:  ...`` line format.

Failure is graceful: prints what happened, exits 1 when nothing was recovered.
The audit/index pipeline treats missing test transcripts as non-fatal.

    python scripts/recover_test_transcripts.py [--parquet-dir DIR] [--test-dir datasets/LRS3/test] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ID_COLUMNS = ["idx", "id", "utt_id", "uttid", "ID"]
TEXT_COLUMNS = ["text", "Text", "transcript", "sentence", "txt", "label"]
HF_REPO_CANDIDATES = [
    "mattymchen/lrs3-test",     # VERIFIED Phase 0: exists; its 1321 rows match test/*.wav 1:1
    "mattymchen/lrs3",
]


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def load_id_text_frame(parquet_dir: Path | None) -> "pd.DataFrame":  # noqa: F821
    """Return a DataFrame with exactly [id_col_value, text_col_value].

    Local mode reads given parquet files; Hub mode streams ONLY the id/text
    column chunks over HTTP (ranged reads) so the multi-hundred-MB audio/video
    columns are never downloaded — important on small disks.
    """
    import pandas as pd

    if parquet_dir is not None:
        import pyarrow.parquet as pq

        files = sorted(parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no .parquet files under {parquet_dir}")
        frames = []
        for f in files:
            sch = pq.read_schema(f).names
            ic, tc = _pick_column(sch, ID_COLUMNS), _pick_column(sch, TEXT_COLUMNS)
            if ic is None or tc is None:
                raise ValueError(f"{f.name}: cannot find id/text columns among {sch}")
            t = pq.read_table(f, columns=[ic, tc])
            frames.append(pd.DataFrame({"idx": t.column(ic), "text": t.column(tc)}))
        return pd.concat(frames, ignore_index=True)

    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    last_err = None
    for repo in HF_REPO_CANDIDATES:
        try:
            print(f"trying HuggingFace dataset repo: {repo} ...")
            parquet_files = sorted(
                n for n in fs.listdir(f"datasets/{repo}", detail=False)
                if str(n).endswith(".parquet")
            )
            if not parquet_files:
                # shards may live under data/
                parquet_files = sorted(
                    n for n in fs.listdir(f"datasets/{repo}/data", detail=False)
                    if str(n).endswith(".parquet")
                )
            if not parquet_files:
                raise ValueError("repo contains no top-level or /data parquet shards")
            frames = []
            for name in parquet_files:
                with fs.open(name, "rb") as f:
                    sch = pq.read_schema(f).names
                ic, tc = _pick_column(sch, ID_COLUMNS), _pick_column(sch, TEXT_COLUMNS)
                if ic is None or tc is None:
                    raise ValueError(f"{name}: cannot find id/text columns among {sch}")
                with fs.open(name, "rb") as f:
                    t = pq.read_table(f, columns=[ic, tc])
                print(f"  streamed columns {ic}/{tc} from {str(name).split('/')[-1]}")
                frames.append(pd.DataFrame({"idx": t.column(ic), "text": t.column(tc)}))
            return pd.concat(frames, ignore_index=True)
        except Exception as e:  # noqa: BLE001 - try every candidate
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(f"all HF candidates failed; last error: {last_err}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", type=Path, default=None,
                    help="local dir containing the original test-*.parquet files")
    ap.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "datasets/LRS3/test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    try:
        df = load_id_text_frame(args.parquet_dir)
    except Exception as e:  # noqa: BLE001
        print(f"RECOVERY FAILED (offline or repo moved?): {e}")
        print("Fallback: internal val split remains the Track-B eval target; "
              "official test stays STOI/PESQ-only. You can also place the parquet "
              "files locally and pass --parquet-dir.")
        return 1

    print(f"recovered {len(df)} id/text rows")

    existing_wavs = {p.stem for p in args.test_dir.glob("*.wav")}
    written = skipped_no_match = 0
    for _, row in df.iterrows():
        uid = str(row["idx"]).replace("/", "_")
        text = str(row["text"]).strip()
        if uid.lower().startswith("nan"):
            continue
        if uid not in existing_wavs:
            skipped_no_match += 1
            continue
        if args.dry_run:
            written += 1
            continue
        (args.test_dir / f"{uid}.txt").write_text(f"Text:  {text}\n", encoding="utf-8")
        written += 1

    print(f"wrote {written} transcript files to {args.test_dir} "
          f"(source rows without matching wav: {skipped_no_match})")
    return 0 if written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
