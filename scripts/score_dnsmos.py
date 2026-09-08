#!/usr/bin/env python3
"""DNSMOS scorer (roster #1) — per-utterance speech-quality scores on Ada.

    python scripts/score_dnsmos.py --out scores/dnsmos_scores.parquet

Mirrors DNS-Challenge/DNSMOS/dnsmos_local.py's NON-personalized path EXACTLY
(16 kHz mono, 9.01 s = 144,160-sample windows, 1 s hop, self-tiling of short
clips, per-hop raw + np.poly1d calibration, hop means) and drops the librosa/
P808 branch (the frozen env carries no librosa). Scores the TRAIN pool only
(§5 item 12: val rows never leave the scoring job). Output is a sorted,
keyed parquet committed from the Ada clone per §5 item 12.

Selection key = calibrated ``ovr_mos`` (descending); calibration is monotone
on any plausible range, so the ranking is invariant to the raw-scale ambiguity
(whose empirical min/max this job prints into the run log for provenance).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf  # noqa: E402

import paths as data_paths  # noqa: E402
from config import load_paths  # noqa: E402
from dataset import load_id_list, load_records  # noqa: E402

FS = 16000
INPUT_LENGTH = 9.01
LEN_SAMPLES = int(INPUT_LENGTH * FS)                  # 144,160
POLY_COEF = {                                          # non-personalized, verbatim
    "sig": [-0.08397278, 1.22083953, 0.0052439],
    "bak": [-0.13166888, 1.60915514, -0.39604546],
    "ovr": [-0.06766283, 1.11546468, 0.04602535],
}
SCORE_COLUMNS = ["utterance_id", "sig_raw", "bak_raw", "ovr_raw",
                 "sig_mos", "bak_mos", "ovr_mos", "n_hops"]

_MODEL_PATH: str | None = None                         # worker initializer slot
_SESSION = None


def load_session(model_path: str | Path):
    import onnxruntime as ort                          # frozen-env dependency (§5.5)

    return ort.InferenceSession(str(model_path),
                                providers=["CPUExecutionProvider"])


def calibrate(values, channel: str) -> np.ndarray:
    return np.poly1d(POLY_COEF[channel])(np.asarray(values, dtype=float))


def score_audio(audio: np.ndarray, session, fs: int = FS) -> dict:
    """One utterance → the §5 item 12 row (means over hops, per-hop calibration)."""
    if fs != FS:
        raise ValueError(
            f"refusing {fs} Hz audio — staged layout guarantees 16 kHz mono "
            "(§5.0); there is NO silent resample path")
    audio = np.asarray(audio, dtype=np.float32)
    while len(audio) < LEN_SAMPLES:
        audio = np.append(audio, audio)                # vendored self-tiling
    num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
    sig_r, bak_r, ovr_r, sig_m, bak_m, ovr_m = [], [], [], [], [], []
    for idx in range(num_hops):
        seg = audio[int(idx * fs): int((idx + INPUT_LENGTH) * fs)]
        if len(seg) < LEN_SAMPLES:
            continue
        feed = {"input_1": seg.astype(np.float32)[np.newaxis, :]}
        mos_sig_raw, mos_bak_raw, mos_ovr_raw = session.run(None, feed)[0][0]
        sig_r.append(float(mos_sig_raw))
        bak_r.append(float(mos_bak_raw))
        ovr_r.append(float(mos_ovr_raw))
        sig_m.append(float(calibrate([mos_sig_raw], "sig")[0]))
        bak_m.append(float(calibrate([mos_bak_raw], "bak")[0]))
        ovr_m.append(float(calibrate([mos_ovr_raw], "ovr")[0]))
    return {"sig_raw": float(np.mean(sig_r)), "bak_raw": float(np.mean(bak_r)),
            "ovr_raw": float(np.mean(ovr_r)), "sig_mos": float(np.mean(sig_m)),
            "bak_mos": float(np.mean(bak_m)), "ovr_mos": float(np.mean(ovr_m)),
            "n_hops": len(sig_r)}


def _collect_train_records():
    """TRAIN pool only (§5 item 12 / §2.5): selectable trainval rows in
    train_ids.txt — val is never scored, never selectable."""
    p = load_paths()
    recs = load_records(p.index_path, split="trainval")
    train = load_id_list(Path(p.splits_dir) / "train_ids.txt")
    return sorted((r for r in recs if r.utterance_id in train),
                  key=lambda r: r.utterance_id)


def resolve_audio_for(rec):
    return data_paths.resolve_audio_path(rec)


def _score_one(task: tuple[str, str]) -> dict:
    uid, flac = task
    global _SESSION
    assert _SESSION is not None, "worker session missing"
    audio, fs = _sf_read(flac)
    row = score_audio(audio, _SESSION, fs=fs)
    row["utterance_id"] = uid
    return row


def _sf_read(path: str | Path):
    import soundfile as sf

    audio, fs = sf.read(path, dtype="float32")
    if audio.ndim != 1:
        raise ValueError(f"refusing non-mono audio: {path} (staged layout is "
                         "16 kHz mono, §5.0)")
    return audio, fs


def _preflight(recs, k: int = 50) -> None:
    data_paths.preflight_resolve(recs, k=min(k, len(recs)), kind="audio")


def _score_all(recs, model_path: str | Path, workers: int) -> list[dict]:
    tasks = [(r.utterance_id, str(resolve_audio_for(r))) for r in recs]
    if workers <= 1:
        global _SESSION
        _SESSION = load_session(model_path)
        rows = [_score_one(t) for t in tasks]
    else:
        def init():
            global _SESSION
            _SESSION = load_session(model_path)

        with ProcessPoolExecutor(max_workers=workers, initializer=init) as ex:
            rows = list(ex.map(_score_one, tasks, chunksize=16))
    return sorted(rows, key=lambda r: r["utterance_id"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", type=Path, default=root / "scores" / "dnsmos_scores.parquet")
    ap.add_argument("--model", type=Path, default=root / "models" / "dnsmos" / "sig_bak_ovr.onnx")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: score only the first N utterances")
    args = ap.parse_args(argv)

    recs = _collect_train_records()
    if args.limit is not None:
        recs = recs[: args.limit]
    print(f"[dnsmos] scoring {len(recs)} train-pool utterances "
          f"model={args.model.name}", flush=True)
    _preflight(recs, k=50)
    rows = _score_all(recs, args.model, args.workers)

    df = pd.DataFrame(rows, columns=SCORE_COLUMNS).sort_values(
        "utterance_id", kind="mergesort").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[dnsmos] wrote {args.out}: rows={len(df)}")
    print(f"[dnsmos] ovr_mos empirical min={df['ovr_mos'].min():.4f} "
          f"max={df['ovr_mos'].max():.4f} mean={df['ovr_mos'].mean():.4f} "
          f"(raw-scale provenance — ranking is calibration-invariant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())