#!/usr/bin/env python
"""Manual spot-check: decode 10 trainval utterances for EAR verification.

Samples utterances deterministically (seeded, spread across duration quantiles),
decodes each mp4's audio to 16 kHz mono WAV under outputs/spotcheck/, and writes
spotcheck_report.txt listing — per file — the raw transcript, the canonical
normalized text, confidence, duration, and token count.

Listen to each wav and confirm: audio matches the transcript, speech is English,
and duration is plausible for the text length. This is the human gate on the
transcript<->audio pairing that every later WER depends on.

    python scripts/spot_check.py [--n 10] [--seed 20260825] [--out outputs/spotcheck]
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from types import SimpleNamespace  # noqa: E402

import paths as data_paths  # noqa: E402  DATA LAYOUT LAW: single path authority


def pick_utts(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n rows spread across duration quantiles; deterministic given (df, seed)."""
    universe = df[(df["split"] == "trainval") & (df["selectable"] == True)]  # noqa: E712
    quantiles = [int(x) for x in universe["duration_s"].quantile(
        [i / n for i in range(n)])]  # one anchor per slice
    rng = random.Random(seed)
    picked = []
    used: set[str] = set()
    for i, q in enumerate(quantiles):
        lo = quantiles[i - 1] if i else -1.0
        band = universe[universe["utterance_id"].isin(used) == False]  # noqa: E712
        band = band[(band["duration_s"] > lo) & (band["duration_s"] <= q + 1e-9)]
        if band.empty:
            band = universe[universe["utterance_id"].isin(used) == False]  # noqa: E712
        row = band.sort_values("utterance_id").iloc[rng.randrange(len(band))]
        used.add(row["utterance_id"])
        picked.append(row)
    return pd.DataFrame(picked)


def decode_to_wav(mp4: Path, wav_out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(mp4),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_out)],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=PROJECT_ROOT / "data_index.parquet")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "spotcheck")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.index)
    utts = pick_utts(df, args.n, args.seed).sort_values("utterance_id")
    args.out.mkdir(parents=True, exist_ok=True)

    lines = [
        f"SPOT CHECK — {len(utts)} utterances, seed={args.seed}",
        "Verify BY EAR: audio matches Text, sane duration vs text length.",
        "=" * 78,
    ]
    for i, r in enumerate(utts.itertuples(), 1):
        stem = r.utterance_id.replace("/", "_")
        wav_path = args.out / f"{i:02d}_{stem}.wav"
        rec = SimpleNamespace(audio_path=r.audio_path, video_id=getattr(r, "video_id", ""),
                              stem=r.stem, utterance_id=r.utterance_id)
        decode_to_wav(data_paths.resolve_audio_path(rec), wav_path)
        dur_check = (wav_path.stat().st_size - 44) / (2 * 16000)  # PCM16 mono bytes→s
        lines += [
            f"\n[{i:02d}] {wav_path.name}",
            f"     uid={r.utterance_id}  conf={r.conf}  T={r.n_tokens} frames  "
            f"tokens_dur={r.duration_s:.2f}s  wav_dur={dur_check:.2f}s",
            f"     RAW : {r.text_raw}",
            f"     NORM: {r.text_norm}",
        ]
        if abs(dur_check - r.duration_s) > max(0.15, 0.05 * r.duration_s):
            lines.append(f"     WARNING: wav/tokens duration mismatch >5%")

    report = args.out / "spotcheck_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {len(utts)} wavs + {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
