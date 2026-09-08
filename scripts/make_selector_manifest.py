#!/usr/bin/env python3
"""Selector manifest dispatcher — the ONE sanctioned writer for selector subsets.

    python scripts/make_selector_manifest.py --selector dnsmos --fraction 0.25
    python scripts/make_selector_manifest.py --selector kmeans --fraction 0.10 --seed 201

Reads a committed score table (§5 item 12, born on Ada), runs the selector's
pure selection function, writes + validates the manifest pair under subsets/.
Seed law (§3.20): stochastic selectors (kmeans, dsir, less_ctc) REQUIRE the
subset seed and stem it into the filename; deterministic selectors (dnsmos,
lossrank, el2n, anti) REJECT one — one frozen manifest per budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from selection import dsir, dnsmos, kmeans, less, proxy_rank  # noqa: E402
from selection.base import (  # noqa: E402
    SelectionContext,
    budget_k,
    validate_manifest,
    write_manifest,
)

def _seedless(fn):
    """Normalize a deterministic selector to the uniform (ctx, scores, budget, seed)
    signature — the seed is ignored by law (§3.20)."""
    return lambda ctx, scores, budget, seed=None: fn(ctx, scores, budget)


ROSTER: dict[str, dict] = {
    "dnsmos": dict(
        fn=_seedless(dnsmos.select), stochastic=False,
        table="scores/dnsmos_scores.parquet",
        meta=lambda seed: {"rank_by": "ovr_mos desc, utterance_id asc "
                                      "(calibrated DNSMOS v8)"},
    ),
    "kmeans": dict(
        fn=lambda ctx, scores, budget, seed: kmeans.select(ctx, scores, budget, seed),
        stochastic=True,
        table="scores/kmeans_assignments_seed{seed}.parquet",
        meta=lambda seed: {"feature": "rvq1 code histogram (L1-normalized)",
                           "algorithm": "scipy kmeans2, kmeans++ init",
                           "quota": "round-robin across clusters, rng(seed)"},
    ),
    "dsir": dict(
        fn=lambda ctx, scores, budget, seed: dsir.select(ctx, scores, budget, seed),
        stochastic=True,
        table="scores/dsir_weights.parquet",   # ONE deterministic fit table
        meta=lambda seed: {"target": "val split (disclosed, §3.21)",
                           "resample": "k w/o replacement, p ∝ exp(logit)",
                           "seed_note": "seed drives THIS laptop-side resample"},
    ),
    "lossrank": dict(
        fn=_seedless(proxy_rank.select_lossrank), stochastic=False,
        table="scores/proxy_scores.parquet",
        meta=lambda seed: {"rank_by": "loss_mean asc (lowest proxy loss first)"},
    ),
    "el2n": dict(
        fn=_seedless(proxy_rank.select_el2n), stochastic=False,
        table="scores/proxy_scores.parquet",
        meta=lambda seed: {"rank_by": "el2n_mean desc (extremal/hard, PROVISIONAL "
                                      "direction)"},
    ),
    "anti": dict(
        fn=_seedless(proxy_rank.select_anti), stochastic=False,
        table="scores/proxy_scores.parquet",
        meta=lambda seed: {"rank_by": "loss_mean desc (worst-by-proxy; §3.19(d) "
                                      "sanity column)"},
    ),
    "less_ctc": dict(
        fn=_seedless(less.select), stochastic=True,
        table="scores/less_influence_seed{seed}.parquet",
        meta=lambda seed: {"variant": "ctc-grad (Track B native)",
                           "rank_by": "influence desc (val reference, §3.21)",
                           "seed_note": "seed lives in the Ada scorer's "
                                        "Rademacher projection"},
    ),
}


def _load_table(scores_dir: Path, spec: dict, seed: int | None) -> pd.DataFrame:
    rel = spec["table"].format(seed=seed) if "{seed}" in spec["table"] else spec["table"]
    path = scores_dir / Path(rel).name
    if not path.exists():
        scorer = Path(rel).stem.split("_")[0]         # dnsmos / kmeans / dsir / less
        raise FileNotFoundError(
            f"score table missing: {path} — generate it on Ada first "
            f"(run slurm/score_{scorer}.sbatch, then commit scores/ from the Ada "
            f"clone per §5 item 12)")
    return pd.read_parquet(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selector", required=True, choices=sorted(ROSTER))
    ap.add_argument("--fraction", required=True, type=float,
                    help="budget fraction over the selectable universe "
                         "(§2.5; e.g. 0.25 / 0.10 / 0.05)")
    ap.add_argument("--seed", type=int, default=None,
                    help="subset seed — REQUIRED for stochastic selectors")
    ap.add_argument("--scores-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "scores")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "subsets")
    args = ap.parse_args(argv)

    spec = ROSTER[args.selector]
    if spec["stochastic"] and args.seed is None:
        raise ValueError(f"--seed is REQUIRED for stochastic selector "
                         f"{args.selector!r} (§3.20)")
    if not spec["stochastic"] and args.seed is not None:
        raise ValueError(f"selector {args.selector!r} is deterministic (§3.20) — "
                         f"do not pass --seed")
    seed = args.seed

    ctx = SelectionContext.from_repo()
    k = budget_k(ctx, args.fraction)
    scores = _load_table(args.scores_dir, spec, seed)
    ids = spec["fn"](ctx, scores, budget=k, seed=seed)

    pct = int(round(args.fraction * 100))
    stem = f"{args.selector}_{pct}pct" + (f"_seed{seed}" if seed is not None else "")
    rel_table = spec["table"].format(seed=seed) if "{seed}" in spec["table"] else spec["table"]
    # RESEARCH P2 stat sheet: subset mean codebook entropy from the shared
    # token_stats table (emitted by the Ada k-means job), when it exists.
    entropy_mean = None
    ts_path = args.scores_dir / "token_stats.parquet"
    if ts_path.exists():
        ts = pd.read_parquet(ts_path)
        sub = ts[ts["utterance_id"].isin(set(ids))]
        if not sub.empty:
            entropy_mean = float(sub["codebook_entropy"].mean())
    txt, js = write_manifest(
        stem, ids, ctx=ctx, fraction=args.fraction, selector=args.selector,
        subset_seed=seed, score_table=rel_table,
        selector_meta=spec["meta"](seed), out_dir=args.out_dir,
        codebook_entropy_mean=entropy_mean)
    validate_manifest(txt, k_expected=k, ctx=ctx)
    print(f"[selector] {stem}: k={k} from {rel_table} "
          f"({txt} + .json, validated)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as e:
        print(f"FATAL[make_selector_manifest] {e}", file=sys.stderr)
        raise SystemExit(2)