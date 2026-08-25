#!/usr/bin/env python
"""Build the FROZEN internal split (Phase 0, created once and committed).

- Internal val: video-disjoint holdout of >= ``val_size_utts`` utterances drawn
  from trainval with a fixed seed. Video-level disjointness is the practical
  speaker-disjointness approximation (each trainval folder = one source video =
  one speaker).
- Train list: all remaining trainval utterances.
- Official test: pointers ONLY (paths), quarantined from everything except the
  final-eval module (PROTOCOL.md).

Outputs under subsets/splits/: ``train_ids.txt``, ``val_ids.txt`` (pure
utterance-ID lines — training code consumes these like any subset manifest),
``test_pointers.tsv``, and ``freeze_manifest.json`` (counts + sha256 of each
file so accidental edits are detectable). Re-running is byte-identical.

    python scripts/build_splits.py [--index data_index.parquet]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ProtocolConfig, load_protocol  # noqa: E402


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split_frames(df: pd.DataFrame, seed: int, val_size_utts: int):
    """Split trainval rows into (train_df, val_df) at VIDEO granularity.

    Deterministic: same df/seed -> same split. Videos are shuffled with a local
    PRNG seeded by ``seed``; whole videos are taken until the val holdout holds
    >= val_size_utts utterances.
    """
    trainval = df[df["split"] == "trainval"]
    by_video = {vid: g for vid, g in trainval.groupby("video_id", sort=True)}
    videos = sorted(by_video)          # canonical order BEFORE shuffling
    rng = random.Random(seed)
    rng.shuffle(videos)

    val_videos: list[str] = []
    n_val = 0
    for vid in videos:
        if n_val >= val_size_utts:
            break
        val_videos.append(vid)
        n_val += len(by_video[vid])

    val_ids = set(val_videos)
    mask = trainval["video_id"].isin(val_ids)
    return trainval[~mask], trainval[mask]


def write_split_files(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                      out_dir: Path, proto: ProtocolConfig, index_sha_note: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_ids.txt"
    val_path = out_dir / "val_ids.txt"
    test_path = out_dir / "test_pointers.tsv"
    manifest_path = out_dir / "freeze_manifest.json"

    train_path.write_text("\n".join(sorted(train_df["utterance_id"])) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(sorted(val_df["utterance_id"])) + "\n", encoding="utf-8")

    # Official test: PATHS ONLY, quarantined. Header comments document the rule;
    # consumers must skip '#'-prefixed lines.
    lines = [
        "# OFFICIAL TEST SPLIT - PATH POINTERS ONLY.",
        "# DO NOT READ THESE FILES outside the final-eval module (PROTOCOL.md).",
        "utterance_id\taudio_path\ttokens_path",
    ]
    lines += [
        f"{r.utterance_id}\t{r.audio_path}\t{r.tokens_path}"
        for r in test_df.sort_values("utterance_id").itertuples()
    ]
    test_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _hours(g: pd.DataFrame) -> float:
        return round(float(g["duration_s"].sum()) / 3600.0, 3)

    manifest = {
        "seed": proto.split_seed,
        "val_size_target_utts": proto.val_size_utts,
        "video_disjoint": True,
        "counts": {
            "train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df)),
        },
        "hours": {"train": _hours(train_df), "val": _hours(val_df)},
        "files": {
            p.name: {"sha256": sha256_of(p)} for p in (train_path, val_path, test_path)
        },
    }
    if index_sha_note:
        manifest["built_from_index_sha256"] = index_sha_note
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=PROJECT_ROOT / "data_index.parquet")
    ap.add_argument("--splits-dir", type=Path,
                    default=PROJECT_ROOT / "subsets" / "splits")
    ap.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "protocol.yaml")
    args = ap.parse_args(argv)
    proto = load_protocol(args.protocol)

    df = pd.read_parquet(args.index)
    train_df, val_df = build_split_frames(df, proto.split_seed, proto.val_size_utts)
    test_df = df[df["split"] == "test"]

    overlap = set(train_df["video_id"]) & set(val_df["video_id"])
    assert not overlap, f"video leakage between train and val: {sorted(overlap)[:5]}"
    assert len(set(train_df["utterance_id"]) & set(val_df["utterance_id"])) == 0
    assert len(val_df) >= proto.val_size_utts, (
        f"val holdout {len(val_df)} < target {proto.val_size_utts}"
    )

    manifest = write_split_files(train_df, val_df, test_df, args.splits_dir, proto)
    print(f"train: {manifest['counts']['train']} utts ({manifest['hours']['train']}h)")
    print(f"val:   {manifest['counts']['val']} utts ({manifest['hours']['val']}h) "
          f"across {val_df['video_id'].nunique()} videos [FROZEN]")
    print(f"test:  {manifest['counts']['test']} clips (pointers only - quarantined)")
    print(f"wrote split files + freeze manifest to {args.splits_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
