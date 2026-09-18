#!/usr/bin/env python3
"""DSIR scorer (roster #3) — Ada CPU job. Importance-resampling features over
RVQ₁ token streams, sklearn-free (scipy sparse + L-BFGS-B only).

    python scripts/score_dsir.py --out scores/dsir_weights.parquet

Data selection via importance resampling (Xie et al. 2023), token-stream
adaptation: every utterance becomes a hashed bag of n-grams — a unigram block
(raw code ids over the frozen codebook) plus a bigram block ((a·1009 + b)
mod 2¹⁶, PROVISIONAL). A class-balanced logistic regression is fit zero-init
against target(val, y=1) vs pool(train, y=0) — fully deterministic (no RNG).
The committed artifact is ONE table of TRAIN-row logits; the §3.20 identity
seeds enter ONLY the laptop-side resample (selection/dsir.select), so no
per-seed files are born here. Val features are consumed in-job and never
committed (§5 item 12).

GO/NO-GO: the printed ``weight_spread`` line (CV of exp(logit)) is the
"demonstrably working" gate — near-uniform spread means DSIR separates
nothing and gets dropped per HANDOFF §1 (72h clock from first commit).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import paths as data_paths  # noqa: E402
from config import load_protocol, load_paths  # noqa: E402
from dataset import load_id_list, load_records  # noqa: E402

NGRAM_BUCKETS = 65536
BIGRAM_HASH_MULT = 1009
SCORE_COLUMNS = ["utterance_id", "score_logit", "weight"]


def hash_bigram(a: int, b: int) -> int:
    return (a * BIGRAM_HASH_MULT + b) % NGRAM_BUCKETS


def _load_stream0(rec) -> np.ndarray:
    path = data_paths.resolve_token_path(rec)          # single path authority (§10 item 4)
    tokens = torch.load(path, map_location="cpu", weights_only=True)
    return np.asarray(tokens)[0]


def featurize(stream: np.ndarray, codebook_size: int) -> sparse.csr_matrix:
    """1 x (codebook + buckets) L1-normalized selection row (mass exactly 1)."""
    codes = np.asarray(stream, dtype=np.int64)
    bad = (codes < 0) | (codes >= codebook_size)
    if bad.any():
        raise ValueError(f"stream 0 carries out-of-range codes "
                         f"(pad sentinel must never appear, §2.4)")
    dim = codebook_size + NGRAM_BUCKETS
    indices: list[int] = []
    data: list[float] = []
    # unigram block: code id = dim, within-block mass 1
    for code, mass in zip(*np.unique(codes, return_counts=True)):
        indices.append(int(code))
        data.append(float(mass) / len(codes))
    # bigram block: dim = codebook + bucket, within-block mass 1
    if len(codes) > 1:
        pairs = np.stack([codes[:-1], codes[1:]], axis=1)
        buckets = hash_bigram(pairs[:, 0], pairs[:, 1])
        uniq, cnt = np.unique(buckets, return_counts=True)
        for bucket, mass in zip(uniq, cnt):
            indices.append(codebook_size + int(bucket))
            data.append(float(mass) / (len(codes) - 1))
    vec = np.asarray(data, dtype=float)
    total = vec.sum()
    if total <= 0:
        raise ValueError("empty stream cannot be featurized")
    vec = vec / total                                # both blocks share mass 1.0 total
    return sparse.csr_matrix((vec, indices, [0, len(vec)]), shape=(1, dim))


def class_weights(y: np.ndarray) -> np.ndarray:
    """Balanced per-row weights: each class's total weighted mass = n/2."""
    n = len(y)
    return np.where(y > 0, n / (2 * max((y > 0).sum(), 1)),
                    n / (2 * max((y == 0).sum(), 1)))


def fit_beta(X_pool, X_val, l2: float = 1e-4) -> np.ndarray:
    """Class-balanced logistic fit (zero-init, no intercept) → the coefficient
    vector itself (callers apply it to whatever rows they need logits for —
    scripts/sweep_dsir_l2.py reuses this to score a held-out split that never
    entered the fit).

    Stable BCE: -(y log σ(z) + (1-y) log(1-σ(z))) = softplus(z) - y·z."""
    X = sparse.vstack([sparse.csr_matrix(X_pool), sparse.csr_matrix(X_val)],
                      format="csr").astype(np.float64)
    n_pool = sparse.csr_matrix(X_pool).shape[0]
    y = np.concatenate([np.zeros(n_pool), np.ones(X.shape[0] - n_pool)])
    n = X.shape[0]
    cw = class_weights(y)

    def nll_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ beta
        # softplus(z) stable: max(z,0) + log1p(exp(-|z|))
        sp = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
        loss = float((cw * (sp - y * z)).sum() / n + 0.5 * l2 * (beta @ beta))
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (X.T @ (cw * (p - y))) / n + l2 * beta
        return loss, grad

    res = minimize(nll_grad, np.zeros(X.shape[1], dtype=float), jac=True,
                   method="L-BFGS-B")
    return res.x


def fit_importance(X_pool, X_val, l2: float = 1e-4) -> np.ndarray:
    """Class-balanced logistic fit (zero-init, no intercept) → POOL-row logits."""
    beta = fit_beta(X_pool, X_val, l2=l2)
    return np.asarray(sparse.csr_matrix(X_pool) @ beta).ravel()


def _collect_populations(report_every_s: float = 30.0, on_progress=None):
    """(train_ids, val_ids, X_train, X_val) — TRAIN pool is the resampling
    population; val is TARGET metadata only (§3.21) and never committed.

    `on_progress(done, total, chunk_s, elapsed_s)` (if given) fires at
    least every `report_every_s` seconds of wall-clock time (plus once
    more at the end), over the POOL featurization loop (the dominant cost
    — val is much smaller) — PROTOCOL §3.23/§3.24/§3.25: time-based, not a
    fixed row count."""
    p = load_paths()
    recs = load_records(p.index_path, split="trainval")
    rec_by_id = {r.utterance_id: r for r in recs}
    train = load_id_list(Path(p.splits_dir) / "train_ids.txt")
    val = load_id_list(Path(p.splits_dir) / "val_ids.txt")
    train_ids = sorted(r.utterance_id for r in recs if r.utterance_id in train)
    val_ids = sorted(v for v in val)                      # frozen split order
    cb = int(load_protocol().codebook_size)
    print(f"[dsir] featurizing {len(train_ids)} pool + {len(val_ids)} target "
          f"utterances (dim={cb + NGRAM_BUCKETS})", flush=True)

    n_total = len(train_ids)
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

    rows_train, rows_val = [], []
    for i, uid in enumerate(train_ids):
        rows_train.append(featurize(_load_stream0(rec_by_id[uid]), cb))
        _maybe_report(i + 1)
    _maybe_report(n_total, force=True)
    for uid in val_ids:
        rows_val.append(featurize(_load_stream0(rec_by_id[uid]), cb))
    X_train = sparse.vstack(rows_train, format="csr")
    X_val = sparse.vstack(rows_val, format="csr")
    return train_ids, val_ids, X_train, X_val


def _wandb_init(*, project: str, entity: str | None, mode: str, job_id: str,
                l2: float):
    """Opens the W&B run EARLY (before featurization starts), per
    PROTOCOL §3.23/§3.24/§3.25."""
    import wandb

    run = wandb.init(project=project, entity=entity, mode=mode,
                     job_type="dsir_scoring",
                     name=f"dsir-scoring-{job_id}",
                     dir=str(Path(__file__).resolve().parents[1] / "outputs" / "wandb"),
                     tags=["dsir", "roster-3"])
    wandb.config.update({"l2": l2})
    return run


def _log_progress(done: int, total: int, chunk_s: float, elapsed_s: float) -> None:
    import wandb

    rate = done / elapsed_s if elapsed_s > 0 else 0.0
    eta_s = (total - done) / rate if rate > 0 else float("nan")
    wandb.log({"dsir_scoring/utts_done": done,
              "dsir_scoring/utts_total": total,
              "dsir_scoring/chunk_duration_s": chunk_s,
              "dsir_scoring/elapsed_s": elapsed_s,
              "dsir_scoring/utts_per_s": rate,
              "dsir_scoring/eta_s": eta_s})
    pct = 100.0 * done / total if total else 0.0
    eta_str = f"{eta_s / 60:.1f} min" if eta_s == eta_s else "unknown"
    print(f"[dsir] pool features {done}/{total} ({pct:.1f}%) "
          f"— +{chunk_s:.0f}s since last report, {elapsed_s / 60:.1f} min elapsed, "
          f"~{rate:.2f} utt/s, ETA {eta_str}", flush=True)


def log_final_summary_to_wandb(df: pd.DataFrame, spread: float) -> None:
    """Logged onto the ALREADY-OPEN run from _wandb_init — composes with the
    live per-chunk progress above rather than replacing it."""
    import wandb

    wandb.run.summary["dsir_scoring/weight_spread_cv"] = spread
    wandb.run.summary["dsir_scoring/score_logit_min"] = float(df["score_logit"].min())
    wandb.run.summary["dsir_scoring/score_logit_max"] = float(df["score_logit"].max())
    wandb.run.summary["dsir_scoring/n_rows"] = int(len(df))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", type=Path, default=root / "scores" / "dsir_weights.parquet")
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--report-every-s", type=float, default=30.0,
                    help="minimum wall-clock seconds between progress reports")
    ap.add_argument("--wandb-project", default="spell-rq2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default="online")
    ap.add_argument("--job-id", default="local",
                    help="SLURM job id or other run tag for the W&B run name")
    args = ap.parse_args(argv)

    # opened BEFORE featurization starts, not after -- so partial progress
    # (and a crash) stay visible in W&B (PROTOCOL §3.24)
    run = _wandb_init(project=args.wandb_project, entity=args.wandb_entity,
                      mode=args.wandb_mode, job_id=args.job_id, l2=args.l2)
    try:
        train_ids, _val_ids, X_train, X_val = _collect_populations(
            report_every_s=args.report_every_s, on_progress=_log_progress)
        logits = fit_importance(X_train, X_val, l2=args.l2)
        weight = np.exp(logits - logits.max())
        df = pd.DataFrame({"utterance_id": train_ids, "score_logit": logits,
                           "weight": weight}, columns=SCORE_COLUMNS
                          ).sort_values("utterance_id", kind="mergesort"
                                        ).reset_index(drop=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)

        spread = float(weight.std() / weight.mean())
        print(f"[dsir] wrote {args.out}: rows={len(df)}")
        print(f"[dsir] weight_spread CV={spread:.4f} "
              f"(go/no-go: non-degenerate separation required; near-uniform => drop)")
        log_final_summary_to_wandb(df, spread)
        return 0
    finally:
        # exit_code reflects whether we're unwinding due to an exception --
        # run.finish() with no args always marks the run "Finished" even
        # when the body crashed (§3.25).
        run.finish(exit_code=1 if sys.exc_info()[0] is not None else 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"FATAL[score_dsir] {e}", file=sys.stderr)
        raise SystemExit(2)