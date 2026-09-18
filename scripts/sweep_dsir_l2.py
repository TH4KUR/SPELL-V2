#!/usr/bin/env python3
"""DSIR --l2 held-out sweep — research decision support, NOT part of the
selector roster pipeline (does not write anything under scores/).

    python scripts/sweep_dsir_l2.py --l2s 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 \
        --out outputs/dsir_l2_sweep.parquet

Why this exists: score_dsir.py's printed ``weight_spread`` is a TRAINING-set
statistic — it always keeps rising as l2 -> 0, because a big-enough model can
eventually memorize idiosyncrasies of the exact rows it was fit on (66,560
hashed feature dims against ~31k utterances is a very permissive regime for
that). A rising weight_spread cannot by itself distinguish "l2 was
suppressing real signal" from "we are now overfitting noise" — both produce
the same symptom. The only way to tell them apart is to check performance on
rows the fit never saw.

Method:
  1. Build pool/val features ONCE via score_dsir._collect_populations
     (unchanged — same paths-layer resolution, same featurization).
  2. Split BOTH classes with a FIXED, deterministic rule (row index % --every
     == 0 -> held-out CHECK set; the rest -> FIT set). No RNG anywhere, matching
     DSIR's own "fully deterministic" law (score_dsir.py's docstring).
  3. For each l2 in the sweep:
       - fit_beta on the FIT rows only, then score the CHECK rows (which the
         fit never saw) for two held-out metrics:
           * held_out_auc — rank-based AUC (Mann-Whitney U form, no sklearn):
             probability a random held-out val row outscores a random
             held-out pool row. 0.5 = indistinguishable from chance; 1.0 =
             perfect separation. This is what actually answers "does the
             model generalize", not weight_spread.
           * held_out_logloss — class-balanced log-loss on the same held-out
             rows (a proper scoring rule: rewards confident-correct
             predictions, punishes confident-wrong ones harder than
             indifferent ones).
       - ALSO fit_beta on the FULL pool+val (score_dsir.py's actual real-run
         behavior) and report weight_spread, for continuity with the earlier
         ad-hoc --l2 diagnostic.
  4. Log each l2 point to W&B AS SOON as it's computed (not batched at the
     end — PROTOCOL §3.24), print a table, and print a RECOMMENDATION (the
     l2 with the best held_out_auc) — a human still freezes the final value
     in PROTOCOL/configs; this script does not do that automatically.

W&B scoping note: this is NOT a "formal run" under PROTOCOL §3.17 (it trains
no Track A/B model, emits no val/wer, is gated by no §7 rule) — it logs under
job_type="dsir_l2_sweep" with dsir_sweep/*-prefixed series, entirely disjoint
from the frozen train/val series names that contract governs. The run opens
before the sweep starts and each point logs its own duration alongside
cumulative elapsed time (§3.24) — not a single dump after everything finishes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_dsir as dsir  # noqa: E402


def held_out_split(n: int, every: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic split — NO RNG (matches DSIR's determinism law, §3.20).
    idx % every == 0 -> check (held out); everything else -> fit."""
    if every < 2:
        raise ValueError(f"--every must be >= 2 to hold anything out, got {every}")
    idx = np.arange(n)
    check = idx[idx % every == 0]
    fit = idx[idx % every != 0]
    return fit, check


def auc_score(logits: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U form), ties broken by average rank —
    sklearn-free, matching this project's no-sklearn convention. 0.5 = chance,
    1.0 = perfect separation of the positive (y=1, val-target) class above
    the negative (y=0, pool) class."""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "held-out check set has only one class present -- widen --every "
            "or check the pool/val counts feeding this split")
    order = np.argsort(logits, kind="mergesort")
    ranks = np.empty(len(logits), dtype=float)
    ranks[order] = np.arange(1, len(logits) + 1, dtype=float)
    # average rank within groups of exactly-tied logits
    uniq, inv, counts = np.unique(logits, return_inverse=True, return_counts=True)
    rank_sums = np.zeros(len(uniq))
    np.add.at(rank_sums, inv, ranks)
    avg_rank = (rank_sums / counts)[inv]
    rank_sum_pos = avg_rank[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def held_out_logloss(logits: np.ndarray, y: np.ndarray) -> float:
    """Class-balanced softplus-form BCE on rows the fit never saw (same
    stable form and class-weighting as score_dsir.fit_beta's own objective,
    applied here purely as an EVALUATION metric — no fitting happens)."""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)
    cw = dsir.class_weights(y)
    sp = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    return float((cw * (sp - y * z)).sum() / len(z))


def weight_spread(logits_full_pool: np.ndarray) -> float:
    """The same CV(exp(logit)) metric score_dsir.py prints, for continuity
    with the earlier single-point --l2 diagnostic."""
    w = np.exp(logits_full_pool - logits_full_pool.max())
    return float(w.std() / w.mean())


def sweep_one_l2(X_train, X_val, fit_pool, check_pool, fit_val, check_val,
                 l2: float) -> dict:
    # held-out metrics: fit on FIT rows only, score CHECK rows (never fit on)
    Xp_fit = sparse.csr_matrix(X_train)[fit_pool]
    Xv_fit = sparse.csr_matrix(X_val)[fit_val]
    beta_cv = dsir.fit_beta(Xp_fit, Xv_fit, l2=l2)
    logits_pool_check = np.asarray(sparse.csr_matrix(X_train)[check_pool] @ beta_cv).ravel()
    logits_val_check = np.asarray(sparse.csr_matrix(X_val)[check_val] @ beta_cv).ravel()
    logits_check = np.concatenate([logits_pool_check, logits_val_check])
    y_check = np.concatenate([np.zeros(len(logits_pool_check)),
                              np.ones(len(logits_val_check))])

    # continuity metric: fit on EVERYTHING (score_dsir.py's actual real-run behavior)
    beta_full = dsir.fit_beta(X_train, X_val, l2=l2)
    logits_full_pool = np.asarray(sparse.csr_matrix(X_train) @ beta_full).ravel()

    return {
        "l2": l2,
        "held_out_auc": auc_score(logits_check, y_check),
        "held_out_logloss": held_out_logloss(logits_check, y_check),
        "weight_spread_full": weight_spread(logits_full_pool),
        "n_check_pool": int(len(logits_pool_check)),
        "n_check_val": int(len(logits_val_check)),
    }


def run_sweep(l2s: list[float], every: int = 5, on_point_done=None) -> pd.DataFrame:
    """`on_point_done(idx, n_total, row, point_s, elapsed_s)` (if given) fires
    after EACH l2 value finishes -- PROTOCOL §3.24: live progress with each
    point's OWN duration, not a single report after the whole sweep ends."""
    train_ids, val_ids, X_train, X_val = dsir._collect_populations()
    fit_pool, check_pool = held_out_split(len(train_ids), every=every)
    fit_val, check_val = held_out_split(len(val_ids), every=every)
    print(f"[sweep_dsir_l2] held-out split (1/{every}): "
          f"pool fit={len(fit_pool)} check={len(check_pool)}, "
          f"val fit={len(fit_val)} check={len(check_val)}", flush=True)

    rows = []
    t_start = time.monotonic()
    t_prev = t_start
    for idx, l2 in enumerate(l2s):
        row = sweep_one_l2(X_train, X_val, fit_pool, check_pool, fit_val, check_val, l2)
        rows.append(row)
        print(f"[sweep_dsir_l2] l2={l2:.1e} "
              f"held_out_auc={row['held_out_auc']:.4f} "
              f"held_out_logloss={row['held_out_logloss']:.4f} "
              f"weight_spread_full={row['weight_spread_full']:.4f}", flush=True)
        now = time.monotonic()
        if on_point_done is not None:
            on_point_done(idx + 1, len(l2s), row, now - t_prev, now - t_start)
        t_prev = now
    return pd.DataFrame(rows).sort_values("l2").reset_index(drop=True)


def _wandb_init(*, project: str, entity: str | None, mode: str, job_id: str,
                l2s: list[float], every: int):
    """Opens the W&B run BEFORE the sweep starts, not after (PROTOCOL §3.24)
    -- so partial progress (and a crash mid-sweep) are both visible."""
    import wandb

    run = wandb.init(project=project, entity=entity, mode=mode,
                     job_type="dsir_l2_sweep",
                     name=f"dsir-l2-sweep-{job_id}",
                     dir=str(Path(__file__).resolve().parents[1] / "outputs" / "wandb"),
                     tags=["dsir", "l2-sweep", "diagnostic"])
    wandb.config.update({"l2s": list(l2s), "every": every})
    return run


def _log_point_progress(idx: int, n_total: int, row: dict, point_s: float,
                        elapsed_s: float) -> None:
    """Fired after EACH l2 value by run_sweep's on_point_done. `point_s` is
    THAT point's own duration -- reported alongside elapsed_s, not folded
    into it (§3.24: an average hides a single slow point)."""
    import wandb

    wandb.log({
        "dsir_sweep/l2": row["l2"],
        "dsir_sweep/held_out_auc": row["held_out_auc"],
        "dsir_sweep/held_out_logloss": row["held_out_logloss"],
        "dsir_sweep/weight_spread_full": row["weight_spread_full"],
        "dsir_sweep/points_done": idx,
        "dsir_sweep/points_total": n_total,
        "dsir_sweep/point_duration_s": point_s,
        "dsir_sweep/elapsed_s": elapsed_s,
    }, step=idx - 1)
    print(f"[sweep_dsir_l2] point {idx}/{n_total} done in {point_s:.1f}s "
          f"({elapsed_s:.1f}s total elapsed)", flush=True)


def log_final_summary_to_wandb(df: pd.DataFrame) -> None:
    """The full table + best-l2 recommendation, logged onto the ALREADY-OPEN
    run from _wandb_init once every point has been logged incrementally —
    does not init or finish the run itself."""
    import wandb

    wandb.log({"dsir_sweep/table": wandb.Table(dataframe=df)})
    best = df.loc[df["held_out_auc"].idxmax()]
    wandb.run.summary["best_l2_by_held_out_auc"] = float(best["l2"])
    wandb.run.summary["best_held_out_auc"] = float(best["held_out_auc"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--l2s", type=float, nargs="+",
                    default=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                    help="l2 values to sweep (log-spaced default bracketing "
                    "the two already-observed points, 1e-4 and 1e-6)")
    ap.add_argument("--every", type=int, default=5,
                    help="held out 1-in-N rows of EACH class (default 5 -> 20%%)")
    ap.add_argument("--out", type=Path, default=root / "outputs" / "dsir_l2_sweep.parquet")
    ap.add_argument("--wandb-project", default="spell-rq2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default="online")
    ap.add_argument("--job-id", default="local",
                    help="SLURM job id or other run tag for the W&B run name")
    args = ap.parse_args(argv)

    # opened BEFORE the sweep starts, not after -- partial progress (and a
    # crash mid-sweep) stay visible in W&B (PROTOCOL §3.24)
    run = _wandb_init(project=args.wandb_project, entity=args.wandb_entity,
                      mode=args.wandb_mode, job_id=args.job_id,
                      l2s=args.l2s, every=args.every)
    try:
        df = run_sweep(args.l2s, every=args.every, on_point_done=_log_point_progress)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"[sweep_dsir_l2] wrote {args.out}")

        best = df.loc[df["held_out_auc"].idxmax()]
        print(f"[sweep_dsir_l2] RECOMMENDATION: l2={best['l2']:.1e} "
              f"(highest held_out_auc={best['held_out_auc']:.4f}) — "
              f"a human still freezes this in PROTOCOL/configs, this is not automatic.")

        log_final_summary_to_wandb(df)
        return 0
    finally:
        # exit_code reflects whether we're unwinding due to an exception --
        # run.finish() with no args always marks the run "Finished" even
        # when the body crashed (caught live, 2026-09-18, on score_proxy.py's
        # identical finally: block -- the same gap existed here).
        run.finish(exit_code=1 if sys.exc_info()[0] is not None else 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"FATAL[sweep_dsir_l2] {e}", file=sys.stderr)
        raise SystemExit(2)
