#!/usr/bin/env python3
"""DNSMOS scorer (roster #1) — per-utterance speech-quality scores on Ada.

    python scripts/score_dnsmos.py --out scores/dnsmos_scores.parquet

Mirrors DNS-Challenge/DNSMOS/dnsmos_local.py's NON-personalized path EXACTLY
(16 kHz mono, 9.01 s = 144,160-sample windows, 1 s hop, self-tiling of short
clips, per-hop raw + np.poly1d calibration, hop means) and drops the librosa/
P808 branch (the frozen env carries no librosa). Scores the TRAIN pool only
(§5 item 12: val rows never leave the scoring job). Output is a sorted,
keyed parquet committed from the Ada clone per §5 item 12.

Inference runs on `dnsmos_model.DNSMOSTorch` (models/dnsmos/sig_bak_ovr_torch.pt),
a pure-PyTorch port of the vendored ONNX graph — NOT onnxruntime, whose
compiled bindings are confirmed incompatible with this project's numpy>=2.0
pin (2026-09-09; see models/dnsmos/PROVENANCE.md). Verified bit-equivalent
to the ONNX graph (<1e-6 max abs diff) by scripts/port_dnsmos_to_torch.py.

Selection key = calibrated ``ovr_mos`` (descending); calibration is monotone
on any plausible range, so the ranking is invariant to the raw-scale ambiguity
(whose empirical min/max this job prints into the run log for provenance).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import paths as data_paths  # noqa: E402
from config import load_paths  # noqa: E402
from dataset import load_id_list, load_records  # noqa: E402
from dnsmos_model import LEN_SAMPLES, load_dnsmos_torch  # noqa: E402

FS = 16000
INPUT_LENGTH = 9.01
assert LEN_SAMPLES == int(INPUT_LENGTH * FS)          # 144,160 — single source of truth
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
    return load_dnsmos_torch(model_path)


def calibrate(values, channel: str) -> np.ndarray:
    return np.poly1d(POLY_COEF[channel])(np.asarray(values, dtype=float))


def score_audio(audio: np.ndarray, session, fs: int = FS) -> dict:
    """One utterance → the §5 item 12 row (means over hops, per-hop calibration).

    ``session`` is a ``dnsmos_model.DNSMOSTorch`` in eval mode (the name is
    kept from the onnxruntime-session era for a minimal diff)."""
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
        feed = torch.from_numpy(seg.astype(np.float32)[np.newaxis, :])
        with torch.no_grad():
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = session(feed)[0].numpy()
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


def checkpoint_path_for(out: Path) -> Path:
    """Where in-progress rows are periodically saved -- derived from --out so
    a resume needs no extra flag. Job 2701385 (2026-09-19) hit its 6h SLURM
    wall at 82.1% done and lost ALL of it: nothing was ever written to disk
    before the final `df.to_parquet(args.out)`, so a mid-run kill (--time
    wall, preemption, node failure) discarded hours of already-completed
    work. This exists so that stops being possible."""
    return out.with_name(out.stem + ".partial" + out.suffix)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write-then-rename: a kill mid-write leaves the OLD checkpoint intact
    (POSIX rename is atomic) instead of a truncated, unreadable parquet
    file -- otherwise the crash-safety this exists for could itself be the
    thing that corrupts the checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> list[dict]:
    """Rows already scored by a PRIOR (killed/timed-out) attempt, or [] if
    there's no checkpoint / it can't be read (never fatal -- worst case is
    re-scoring rows that were already done, not losing the ability to run
    at all)."""
    if not path.exists():
        return []
    try:
        return pd.read_parquet(path).to_dict("records")
    except Exception as e:
        print(f"[dnsmos] WARNING: checkpoint {path} unreadable ({e}) -- "
              f"starting fresh", file=sys.stderr)
        return []


def _score_chunk(tasks_chunk: list[tuple[str, str]]) -> list[dict]:
    """One ProcessPoolExecutor unit of work — a small batch, not the whole
    task list, so cheap IPC batching survives switching to as_completed()
    below (module-level: must stay picklable)."""
    return [_score_one(t) for t in tasks_chunk]


def _score_all(recs, model_path: str | Path, workers: int,
               report_every_s: float = 30.0, chunk_size: int = 16,
               on_progress=None, checkpoint_path: Path | None = None,
               checkpoint_prefix_rows: list[dict] | None = None) -> list[dict]:
    """`on_progress(done, total, chunk_s, elapsed_s)` (if given) fires at
    least every `report_every_s` seconds of WALL-CLOCK time (plus once more
    at the very end) — PROTOCOL §3.24.

    An earlier version reported every N rows instead of every N seconds:
    wrong, because it silently assumes a throughput rate — at this scorer's
    real ~3h/29k-utterance rate, a threshold of 2000 rows meant the FIRST
    log line didn't appear for ~10+ minutes. It also consumed the pool via
    `ProcessPoolExecutor.map()`, whose iterator yields results strictly in
    INPUT order — so a single slow chunk at the front of the task list
    could block every later, already-finished chunk from being observed at
    all, i.e. workers were producing output with nothing to show for it.
    Both are fixed here: reporting is scheduled by elapsed time, not item
    count, and chunks are consumed via as_completed() so a fast chunk is
    visible the moment it finishes, regardless of its position in the task
    list.

    If `checkpoint_path` is given, `checkpoint_prefix_rows` (rows already
    scored by a prior, killed attempt -- see `load_checkpoint`) plus every
    row scored so far THIS run are written there on the SAME cadence as
    progress reports (job 2701385, 2026-09-19: hit its SLURM --time wall at
    82.1% done with NOTHING ever written to disk, losing ~5 hours of
    completed work). `recs` must already exclude anything in
    `checkpoint_prefix_rows` -- this function does not itself skip rows."""
    tasks = [(r.utterance_id, str(resolve_audio_for(r))) for r in recs]
    n_total = len(tasks)
    t_start = time.monotonic()
    t_prev = t_start
    t_last_report = t_start
    rows: list[dict] = []
    prefix = checkpoint_prefix_rows or []

    def _maybe_report(force: bool = False) -> None:
        nonlocal t_prev, t_last_report
        due = force
        now = time.monotonic()
        if not due and now - t_last_report >= report_every_s:
            due = True
        if not due:
            return
        if checkpoint_path is not None and rows:
            _atomic_write_parquet(
                pd.DataFrame(prefix + rows, columns=SCORE_COLUMNS), checkpoint_path)
        if on_progress is not None:
            # SESSION-LOCAL counts, deliberately not prefix-inclusive -- rate
            # and ETA are done/elapsed_s, and elapsed_s only covers THIS
            # session, so mixing in a resumed prefix would make both wildly
            # wrong (e.g. dividing 23,856 resumed + 50 new rows by 5 elapsed
            # seconds). Callers that want an "overall including resume"
            # figure add `len(checkpoint_prefix_rows)` themselves for
            # DISPLAY only, separately from this rate calculation.
            on_progress(len(rows), n_total, now - t_prev, now - t_start)
        t_prev = now
        t_last_report = now

    if workers <= 1:
        global _SESSION
        _SESSION = load_session(model_path)
        for t in tasks:
            rows.append(_score_one(t))
            _maybe_report()
    else:
        def init():
            # each of the `workers` processes otherwise defaults its OWN
            # full-core intra-op thread pool, so N processes independently
            # fight over the same N allocated CPUs -- measured to cost MORE
            # than hop-batching would gain (2026-09-18 profiling). Pin to 1:
            # parallelism already comes from having `workers` processes.
            torch.set_num_threads(1)
            global _SESSION
            _SESSION = load_session(model_path)

        chunks = [tasks[i:i + chunk_size] for i in range(0, n_total, chunk_size)]
        with ProcessPoolExecutor(max_workers=workers, initializer=init) as ex:
            futures = [ex.submit(_score_chunk, c) for c in chunks]
            for fut in as_completed(futures):
                rows.extend(fut.result())
                _maybe_report()
    _maybe_report(force=True)
    return sorted(rows, key=lambda r: r["utterance_id"])


def _wandb_init(*, project: str, entity: str | None, mode: str, job_id: str,
                model_path: str, workers: int, n_recs: int):
    """Opens the W&B run EARLY (before the ProcessPoolExecutor pass starts),
    per PROTOCOL §3.23/§3.24 — otherwise a job that runs into its SLURM
    --time wall (this scorer has done exactly that) leaves zero trace of
    how far it got."""
    import wandb

    run = wandb.init(project=project, entity=entity, mode=mode,
                     job_type="dnsmos_scoring",
                     name=f"dnsmos-scoring-{job_id}",
                     dir=str(Path(__file__).resolve().parents[1] / "outputs" / "wandb"),
                     tags=["dnsmos", "roster-1"])
    wandb.config.update({"model_path": str(model_path), "workers": workers,
                        "n_recs": n_recs})
    return run


def _log_scoring_progress(done: int, total: int, chunk_s: float, elapsed_s: float) -> None:
    import wandb

    rate = done / elapsed_s if elapsed_s > 0 else 0.0
    eta_s = (total - done) / rate if rate > 0 else float("nan")
    wandb.log({"dnsmos_scoring/utts_done": done,
              "dnsmos_scoring/utts_total": total,
              "dnsmos_scoring/chunk_duration_s": chunk_s,
              "dnsmos_scoring/elapsed_s": elapsed_s,
              "dnsmos_scoring/utts_per_s": rate,
              "dnsmos_scoring/eta_s": eta_s})
    pct = 100.0 * done / total if total else 0.0
    eta_str = f"{eta_s / 60:.1f} min" if eta_s == eta_s else "unknown"
    print(f"[dnsmos] {done}/{total} ({pct:.1f}%) scored "
          f"— +{chunk_s:.0f}s since last report, {elapsed_s / 60:.1f} min elapsed, "
          f"~{rate:.2f} utt/s, ETA {eta_str}", flush=True)


def log_final_summary_to_wandb(df: pd.DataFrame) -> None:
    """Logged onto the ALREADY-OPEN run from _wandb_init — composes with the
    live per-chunk progress above rather than replacing it."""
    import wandb

    wandb.run.summary["dnsmos_scoring/ovr_mos_min"] = float(df["ovr_mos"].min())
    wandb.run.summary["dnsmos_scoring/ovr_mos_max"] = float(df["ovr_mos"].max())
    wandb.run.summary["dnsmos_scoring/ovr_mos_mean"] = float(df["ovr_mos"].mean())
    wandb.run.summary["dnsmos_scoring/n_rows"] = int(len(df))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", type=Path, default=root / "scores" / "dnsmos_scores.parquet")
    ap.add_argument("--model", type=Path,
                    default=root / "models" / "dnsmos" / "sig_bak_ovr_torch.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: score only the first N utterances")
    ap.add_argument("--report-every-s", type=float, default=30.0,
                    help="minimum wall-clock seconds between progress reports")
    ap.add_argument("--chunk-size", type=int, default=16,
                    help="utterances per ProcessPoolExecutor work unit")
    ap.add_argument("--wandb-project", default="spell-rq2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default="online")
    ap.add_argument("--job-id", default="local",
                    help="SLURM job id or other run tag for the W&B run name")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore/discard any existing checkpoint and start over")
    args = ap.parse_args(argv)

    recs = _collect_train_records()
    if args.limit is not None:
        recs = recs[: args.limit]

    ckpt_path = checkpoint_path_for(args.out)
    if args.fresh:
        ckpt_path.unlink(missing_ok=True)
        done_rows: list[dict] = []
    else:
        done_rows = load_checkpoint(ckpt_path)
    done_ids = {r["utterance_id"] for r in done_rows}
    if done_rows:
        recs = [r for r in recs if r.utterance_id not in done_ids]
        print(f"[dnsmos] resuming from {ckpt_path}: {len(done_ids)} already "
              f"scored, {len(recs)} remaining", flush=True)

    # opened BEFORE the expensive pass, not after -- so partial progress
    # (and a crash, or a SLURM --time wall) are both visible in W&B
    run = _wandb_init(project=args.wandb_project, entity=args.wandb_entity,
                      mode=args.wandb_mode, job_id=args.job_id,
                      model_path=str(args.model), workers=args.workers,
                      n_recs=len(recs) + len(done_rows))
    try:
        print(f"[dnsmos] scoring {len(recs)} train-pool utterances "
              f"model={args.model.name}", flush=True)
        _preflight(recs, k=50)
        rows = _score_all(recs, args.model, args.workers,
                          report_every_s=args.report_every_s,
                          chunk_size=args.chunk_size,
                          on_progress=_log_scoring_progress,
                          checkpoint_path=ckpt_path,
                          checkpoint_prefix_rows=done_rows)

        df = pd.DataFrame(done_rows + rows, columns=SCORE_COLUMNS).sort_values(
            "utterance_id", kind="mergesort").reset_index(drop=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        ckpt_path.unlink(missing_ok=True)   # superseded by the complete --out
        print(f"[dnsmos] wrote {args.out}: rows={len(df)}")
        print(f"[dnsmos] ovr_mos empirical min={df['ovr_mos'].min():.4f} "
              f"max={df['ovr_mos'].max():.4f} mean={df['ovr_mos'].mean():.4f} "
              f"(raw-scale provenance — ranking is calibration-invariant)")
        log_final_summary_to_wandb(df)
        return 0
    finally:
        # exit_code reflects whether we're unwinding due to an exception --
        # run.finish() with no args always marks the run "Finished" even
        # when the body crashed (caught live, 2026-09-18: job 2701229's
        # score_proxy.py crash still showed as a completed run in the W&B
        # UI). sys.exc_info() is non-None here iff an exception is
        # currently propagating through this finally block.
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is not None:
            # the exit_code alone says "this failed" with no way to tell
            # WHAT from the W&B UI -- the actual message/traceback used to
            # only exist in the SLURM .err log. Log it onto the run itself.
            import traceback
            run.summary["error"] = f"{exc_type.__name__}: {exc_value}"
            run.summary["traceback"] = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb))
        run.finish(exit_code=1 if exc_type is not None else 0)


if __name__ == "__main__":
    raise SystemExit(main())