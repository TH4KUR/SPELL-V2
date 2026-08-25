#!/usr/bin/env python
"""Phase 0b — stage the dataset to Ada ($HOME/spell/data), sha256-manifested.

BLOCKING gate before Phase 1. Produces, under --dest:

    <dest>/trainval/<video_id>/<stem>.tokens.pt   (copied verbatim)
    <dest>/trainval/<video_id>/<stem>.txt         (copied verbatim)
    <dest>/trainval/<video_id>/<stem>.flac        (16 kHz mono FLAC, from .mp4)
    <dest>/test/<id>.tokens.pt / .txt / .flac     (from .wav)
    <dest>/manifest.parquet + manifest_summary.json

Why FLAC: mp4 containers are awkward for training loops and PCM wav would cost
~3.4 GB; 16 kHz mono FLAC is lossless, ~half the size, and one file per
utterance. The dataset lives ONLY in $HOME/spell/data on Ada (30 GB/300k-inode
quota) — /share1 must never hold per-utterance files (~3200-inode cap).

Runs anywhere the source exists (laptop pilot or Ada). Deterministic layout;
--verify re-hashes an existing staging tree against its manifest.

    python scripts/stage_to_ada.py [--source datasets/LRS3] [--dest ~/spell/data]
                                   [--workers N] [--limit N] [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KIND_TOKENS = "tokens"
KIND_TRANSCRIPT = "transcript"
KIND_AUDIO = "audio"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def convert_audio(src: Path, dst: Path) -> None:
    """Decode any input container to 16 kHz mono FLAC (lossless)."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(tmp)],
        check=True,
    )
    os.replace(tmp, dst)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(src.read_bytes())
    os.replace(tmp, dst)


def plan_jobs(source: Path, dest: Path, limit: int | None) -> list[dict]:
    """Enumerate (src, dst_relpath, kind) jobs from the audited layout."""
    index_path = PROJECT_ROOT / "data_index.parquet"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path} — run scripts/audit_data.py first")
    df = pd.read_parquet(index_path)
    if limit:
        # limit per split for quick pilots, keeping videos spread out
        df = (df.groupby("split", group_keys=False)
                .apply(lambda g: g.sort_values("utterance_id").head(limit)))
    jobs: list[dict] = []
    for r in df.itertuples():
        rel_dir = Path(r.video_id) if r.video_id else Path("")
        base = rel_dir / r.stem
        src_audio_ext = ".mp4" if r.audio_kind == "mp4" else ".wav"
        jobs.append({"src": Path(r.tokens_path), "rel": base.with_suffix(".tokens.pt").as_posix(), "kind": KIND_TOKENS})
        if r.txt_path and not pd.isna(r.txt_path):
            jobs.append({"src": Path(r.txt_path), "rel": base.with_suffix(".txt").as_posix(), "kind": KIND_TRANSCRIPT})
        jobs.append({"src": Path(r.audio_path), "rel": str(base) + ".flac", "kind": KIND_AUDIO,
                     "audio_src_ext": src_audio_ext})
    return jobs


def run_job(job: dict, source_root: Path, dest_root: Path) -> dict | None:
    src = job["src"]
    dst = dest_root / job["rel"]
    if job["kind"] == KIND_AUDIO:
        # index stores the ORIGINAL container path; staged tree always gets .flac
        if not src.exists():  # tolerate already-staged trees being re-run
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        convert_audio(src, dst)
    else:
        if not src.exists():
            return None
        copy_file(src, dst)
    return {"rel": job["rel"], "kind": job["kind"], "bytes": dst.stat().st_size}


def build_manifest(dest: Path, workers: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    files = sorted(p for p in dest.rglob("*") if p.is_file() and p.name != "manifest_summary.json"
                   and p.name != "manifest.parquet" and p.suffix != ".part")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(tqdm(ex.map(sha256_of, files), total=len(files), desc="sha256"))
    for p, h in zip(files, hashes):
        rel = p.relative_to(dest).as_posix()
        kind = KIND_TOKENS if rel.endswith(".tokens.pt") else (
            KIND_AUDIO if rel.endswith(".flac") else KIND_TRANSCRIPT)
        rows.append({"relpath": rel, "kind": kind, "bytes": p.stat().st_size, "sha256": h})
    df = pd.DataFrame(rows)
    summary = {
        "files_total": len(df),
        "inodes_estimate": int(len(df) + df["relpath"].str.count("/").add(1).sum()),
        "bytes_total_gb": round(float(df["bytes"].sum()) / 1e9, 3),
        "by_kind": {k: {"count": int(len(g)), "gb": round(float(g['bytes'].sum()) / 1e9, 3)}
                    for k, g in df.groupby("kind")},
    }
    return df, summary


def verify(dest: Path, workers: int) -> int:
    mpath = dest / "manifest.parquet"
    if not mpath.exists():
        raise SystemExit(f"no manifest at {mpath}")
    want = pd.read_parquet(mpath)
    files = sorted(p for p in dest.rglob("*") if p.is_file()
                   and p.name not in ("manifest.parquet", "manifest_summary.json"))
    have = {p.relative_to(dest).as_posix() for p in files}
    missing = set(want["relpath"]) - have
    extra = have - set(want["relpath"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = dict(zip(
            [p.relative_to(dest).as_posix() for p in files],
            tqdm(ex.map(sha256_of, files), total=len(files), desc="sha256"),
        ))
    mismatched = [r.relpath for r in want.itertuples()
                  if r.relpath in got and got[r.relpath] != r.sha256]
    print(f"verify: {len(want)} expected | missing={len(missing)} extra={len(extra)} "
          f"hash_mismatch={len(mismatched)}")
    for name, bad in (("MISSING", missing), ("EXTRA", extra)):
        for x in sorted(bad)[:10]:
            print(f"  {name}: {x}")
    for x in mismatched[:10]:
        print(f"  HASH MISMATCH: {x}")
    return 1 if (missing or extra or mismatched) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=PROJECT_ROOT / "datasets/LRS3",
                    help="root of the RAW audited dataset (for reference; paths come "
                         "from data_index.parquet)")
    ap.add_argument("--dest", type=Path, default=None,
                    help="staging destination (default: SPELL_DATA_ROOT env or <repo>/data)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot mode: only the first N utterances PER SPLIT")
    ap.add_argument("--skip-stage", action="store_true", help="manifest only")
    ap.add_argument("--verify", action="store_true", help="verify existing staging vs manifest")
    args = ap.parse_args(argv)

    dest = args.dest or Path(os.environ.get("SPELL_DATA_ROOT", PROJECT_ROOT / "data"))
    dest.mkdir(parents=True, exist_ok=True)

    if args.verify:
        return verify(dest, args.workers)

    jobs = plan_jobs(args.source, dest, args.limit)
    print(f"staging {len(jobs)} files -> {dest} with {args.workers} workers")
    done = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in tqdm(ex.map(lambda j: run_job(j, args.source, dest), jobs),
                        total=len(jobs), desc="stage"):
            if res is None:
                skipped += 1
            else:
                done += 1
    print(f"staged {done} files ({skipped} sources absent)")

    df, summary = build_manifest(dest, args.workers)
    df.to_parquet(dest / "manifest.parquet", index=False)
    summary["limit_per_split"] = args.limit
    (dest / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {dest/'manifest.parquet'} + manifest_summary.json")

    # inode sanity vs Ada quota (300k): the staged tree itself must fit comfortably
    est = summary["inodes_estimate"]
    if est > 250_000:
        print(f"WARNING: ~{est} inodes approaches the 300k /home quota")
    return 0


if __name__ == "__main__":
    sys.exit(main())
