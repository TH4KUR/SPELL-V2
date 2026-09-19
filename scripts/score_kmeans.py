#!/usr/bin/env python3
"""k-means diversity scorer (roster #2) — Ada CPU job, scipy only (no sklearn).

    python scripts/score_kmeans.py --seeds 201 202 203 --k 64 \
        --out-dir scores

Features are per-utterance RVQ₁ (stream 0) code histograms over the frozen
codebook, L1-normalized, built in-job and NEVER committed (repo-bloat guard).
For each §3.20 identity seed: scipy.cluster.vq.kmeans2(kmeans++ init,
seed=identity) → scores/kmeans_assignments_seed{S}.parquet with the §5 item 12
schema. The same job also emits scores/token_stats.parquet (per-utterance
codebook entropy — the RESEARCH P2 characterization feed for ALL selectors).
TRAIN pool only (§5 item 12); tokens are read via the paths layer (§10 items
4–6) with the §2.4 pad sentinel asserted absent from stored streams.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import paths as data_paths  # noqa: E402
from config import load_protocol, load_paths  # noqa: E402
from dataset import load_records  # noqa: E402

ASSIGN_COLUMNS = ["utterance_id", "cluster_id", "dist", "cluster_size"]


def codebook_size() -> int:
    return int(load_protocol().codebook_size)


def histogram(stream0: np.ndarray, codebook_size: int) -> np.ndarray:
    """L1-normalized histogram of an ALREADY-extracted RVQ-1 (stream 0) code
    sequence over [0, codebook_size). Callers pass `_load_stream0`'s output
    directly — do not re-index; a second `[0]` collapses the 1-D code array
    to a single scalar (numpy's "too small depth" error, 2026-09-16 incident:
    score_kmeans never having run against real Ada tokens before that day)."""
    stream0 = np.asarray(stream0)
    bad = (stream0 < 0) | (stream0 >= codebook_size)
    if bad.any():
        raise ValueError(f"stream 0 carries out-of-range codes: "
                         f"{np.unique(stream0[bad])[:5].tolist()} (pad sentinel "
                         f"must never appear in stored tokens, §2.4)")
    counts = np.bincount(stream0, minlength=codebook_size).astype(np.float64)
    return counts / counts.sum()


def codebook_entropy(hist: np.ndarray, codebook_size: int) -> float:
    p = np.asarray(hist, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _collect_train_records() -> list[str]:
    """Sorted TRAIN-pool ids (§5 item 12) — the clustering population."""
    p = load_paths()
    recs = load_records(p.index_path, split="trainval")
    from dataset import load_id_list

    train = load_id_list(Path(p.splits_dir) / "train_ids.txt")
    return sorted(r.utterance_id for r in recs if r.utterance_id in train)


def _load_stream0(rec) -> np.ndarray:
    path = data_paths.resolve_token_path(rec)          # single path authority (§10 item 4)
    tokens = torch.load(path, map_location="cpu", weights_only=True)
    return np.asarray(tokens)[0]


def _build_features(ids: list[str], report_every_s: float = 30.0,
                    on_progress=None) -> tuple[np.ndarray, np.ndarray]:
    """(features [n, codebook_size] float32, entropies) — in-job only.

    `on_progress(done, total, chunk_s, elapsed_s)` (if given) fires at
    least every `report_every_s` seconds of wall-clock time (plus once
    more at the end) — PROTOCOL §3.23/§3.24/§3.25: time-based, not a fixed
    row count, so it stays useful regardless of the actual per-utterance
    rate."""
    cb = codebook_size()
    p = load_paths()
    rec_by_id = {r.utterance_id: r for r in load_records(p.index_path, split="trainval")}
    feats = np.zeros((len(ids), cb), dtype=np.float64)
    ents = np.zeros(len(ids), dtype=np.float64)
    n_total = len(ids)
    t_start = time.monotonic()
    t_prev = t_start
    t_last_report = t_start

    def _maybe_report(done: int, force: bool = False) -> None:
        nonlocal t_prev, t_last_report
        if on_progress is None:
            return
        now = time.monotonic()
        if force or now - t_last_report >= report_every_s:
            on_progress(done, n_total, now - t_prev, now - t_start)
            t_prev = now
            t_last_report = now

    for i, uid in enumerate(ids):
        tokens = _load_stream0(rec_by_id[uid])
        feats[i] = histogram(tokens, cb)
        ents[i] = codebook_entropy(feats[i], cb)
        _maybe_report(i + 1)
    _maybe_report(n_total, force=True)
    return feats.astype(np.float32), ents


def cluster(feats: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """kmeans2 with kmeans++ init seeded by the §3.20 identity → deterministic."""
    centroid, labels = kmeans2(feats, k, iter=50, minit="++", seed=int(seed))
    dist = np.linalg.norm(feats - centroid[labels], axis=1)
    return labels, dist


def _wandb_init(*, project: str, entity: str | None, mode: str, job_id: str,
                seeds: list[int], k: int, n_recs: int):
    """Opens the W&B run EARLY (before feature-building starts), per
    PROTOCOL §3.23/§3.24/§3.25."""
    import wandb

    run = wandb.init(project=project, entity=entity, mode=mode,
                     job_type="kmeans_scoring",
                     name=f"kmeans-scoring-{job_id}",
                     dir=str(Path(__file__).resolve().parents[1] / "outputs" / "wandb"),
                     tags=["kmeans", "roster-2"])
    wandb.config.update({"seeds": sorted(seeds), "k": k, "n_recs": n_recs})
    return run


def _log_progress(done: int, total: int, chunk_s: float, elapsed_s: float) -> None:
    import wandb

    rate = done / elapsed_s if elapsed_s > 0 else 0.0
    eta_s = (total - done) / rate if rate > 0 else float("nan")
    wandb.log({"kmeans_scoring/utts_done": done,
              "kmeans_scoring/utts_total": total,
              "kmeans_scoring/chunk_duration_s": chunk_s,
              "kmeans_scoring/elapsed_s": elapsed_s,
              "kmeans_scoring/utts_per_s": rate,
              "kmeans_scoring/eta_s": eta_s})
    pct = 100.0 * done / total if total else 0.0
    eta_str = f"{eta_s / 60:.1f} min" if eta_s == eta_s else "unknown"
    print(f"[kmeans] features {done}/{total} ({pct:.1f}%) "
          f"— +{chunk_s:.0f}s since last report, {elapsed_s / 60:.1f} min elapsed, "
          f"~{rate:.2f} utt/s, ETA {eta_str}", flush=True)


def log_final_summary_to_wandb(stats: pd.DataFrame,
                               assign_dfs: dict[int, pd.DataFrame]) -> None:
    """Logged onto the ALREADY-OPEN run from _wandb_init — composes with the
    live per-chunk progress above rather than replacing it."""
    import wandb

    wandb.run.summary["kmeans_scoring/entropy_min"] = float(stats["codebook_entropy"].min())
    wandb.run.summary["kmeans_scoring/entropy_max"] = float(stats["codebook_entropy"].max())
    wandb.run.summary["kmeans_scoring/n_rows"] = int(len(stats))
    for seed, df in assign_dfs.items():
        sizes = df["cluster_size"]
        wandb.run.summary[f"kmeans_scoring/cluster_size_min_seed{seed}"] = int(sizes.min())
        wandb.run.summary[f"kmeans_scoring/cluster_size_max_seed{seed}"] = int(sizes.max())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="§3.20 subset-identity seeds (e.g. 201 202 203)")
    ap.add_argument("--k", type=int, required=True,
                    help="cluster count (PROVISIONAL: 64)")
    ap.add_argument("--out-dir", type=Path, default=root / "scores")
    ap.add_argument("--report-every-s", type=float, default=30.0,
                    help="minimum wall-clock seconds between progress reports")
    ap.add_argument("--wandb-project", default="spell-rq2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default="online")
    ap.add_argument("--job-id", default="local",
                    help="SLURM job id or other run tag for the W&B run name")
    args = ap.parse_args(argv)

    ids = _collect_train_records()

    # opened BEFORE feature-building starts, not after -- so partial
    # progress (and a crash) stay visible in W&B (PROTOCOL §3.24)
    run = _wandb_init(project=args.wandb_project, entity=args.wandb_entity,
                      mode=args.wandb_mode, job_id=args.job_id,
                      seeds=args.seeds, k=args.k, n_recs=len(ids))
    try:
        if args.k > len(ids):
            raise ValueError(f"--k {args.k} exceeds the train pool ({len(ids)})")
        print(f"[kmeans] building features for {len(ids)} train-pool utterances "
              f"(codebook={codebook_size()})", flush=True)
        feats, entropies = _build_features(ids, report_every_s=args.report_every_s,
                                           on_progress=_log_progress)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        stats = pd.DataFrame({"utterance_id": ids, "codebook_entropy": entropies})
        stats.to_parquet(args.out_dir / "token_stats.parquet", index=False)
        print(f"[kmeans] wrote token_stats.parquet (entropy "
              f"min={stats['codebook_entropy'].min():.3f} "
              f"max={stats['codebook_entropy'].max():.3f})")

        assign_dfs: dict[int, pd.DataFrame] = {}
        for seed in args.seeds:
            labels, dist = cluster(feats, args.k, int(seed))
            sizes = pd.Series(labels).value_counts()
            df = pd.DataFrame({
                "utterance_id": ids,
                "cluster_id": labels.astype(int),
                "dist": dist.astype(float),
                "cluster_size": pd.Series(labels).map(sizes).to_numpy(),
            }, columns=ASSIGN_COLUMNS).sort_values("utterance_id",
                                                   kind="mergesort").reset_index(drop=True)
            out = args.out_dir / f"kmeans_assignments_seed{int(seed)}.parquet"
            df.to_parquet(out, index=False)
            assign_dfs[int(seed)] = df
            print(f"[kmeans] seed={seed}: {args.k} clusters "
                  f"(min={sizes.min()}, max={sizes.max()}) -> {out}")

        log_final_summary_to_wandb(stats, assign_dfs)
        return 0
    finally:
        # exit_code reflects whether we're unwinding due to an exception --
        # run.finish() with no args always marks the run "Finished" even
        # when the body crashed (§3.25).
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is not None:
            import traceback
            run.summary["error"] = f"{exc_type.__name__}: {exc_value}"
            run.summary["traceback"] = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb))
        run.finish(exit_code=1 if exc_type is not None else 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"FATAL[score_kmeans] {e}", file=sys.stderr)
        raise SystemExit(2)