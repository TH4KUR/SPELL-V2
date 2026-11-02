#!/usr/bin/env python3
"""Build the RAVEn-parity splits (PROTOCOL §3.33).

    python scripts/parity_build_splits.py --valid-id-file /path/to/lrs3-valid.id

Writes ``parity/splits/{parity_train_ids,parity_val_ids,parity_test_ids}.txt`` and
``parity/splits/provenance.json`` from ``data_index.parquet``. Expected: train 30,782
utts (29.25 h), val 1,200 (1.15 h), test 1,321 (0.85 h); every id exists in the index,
the splits are pairwise disjoint, and the 1,321 test references equal RAVEn's own test
labels. The id file (AV-HuBERT ``lrs3-valid.id`` @ b4fa4af, custom Meta licence) is
verified by sha256 and never copied into the repo.

Exit codes: 0 ok, 2 input file missing, 3 checksum refusal, 4 split invariant violated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                     # noqa: E402

from config import load_paths                           # noqa: E402
from parity import splits as S                          # noqa: E402
from parity import text as T                            # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--valid-id-file", required=True, type=Path,
                    help="AV-HuBERT avhubert/preparation/data/lrs3-valid.id (sha256-verified)")
    ap.add_argument("--index", type=Path, default=None, help="default: data_index.parquet")
    ap.add_argument("--out-dir", type=Path, default=S.SPLITS_DIR)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        valid_ids = S.read_valid_id_file(args.valid_id_file)
    except OSError as e:
        print(f"ERROR: cannot read {args.valid_id_file}: {e}", file=sys.stderr)
        return 2
    except S.ChecksumError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    except S.SplitError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4

    index_path = args.index if args.index is not None else Path(load_paths().index_path)
    index = pd.read_parquet(index_path)
    try:
        result = S.build_splits(index, valid_ids, expected=S.PINNED_EXPECTATIONS,
                                raven_test_texts=[r.text for r in T.read_raven_test_csv()])
    except S.SplitError as e:
        print(f"SPLIT INVARIANT VIOLATED: {e}", file=sys.stderr)
        return 4

    S.write_splits(result, args.out_dir,
                   provenance_sources=S.pinned_sources(args.valid_id_file, index_path))
    c, h = result.stats["counts"], result.stats["hours"]
    print(f"[parity-splits] train={c['train']} ({h['train']:.2f} h) val={c['val']} ({h['val']:.2f} h) "
          f"test={c['test']} ({h['test']:.2f} h) -> {args.out_dir}")
    print(f"[parity-splits] val videos: {result.stats['val_videos']}  "
          f"test refs == RAVEn csv: {result.stats['test_text_check']['equal_to_raven_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
