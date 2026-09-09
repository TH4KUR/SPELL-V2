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


def histogram(tokens: np.ndarray, codebook_size: int) -> np.ndarray:
    """Stream-0 histogram over [0, codebook_size), L1-normalized to sum 1."""
    stream0 = np.asarray(tokens)[0]
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


def _build_features(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(features [n, codebook_size] float32, entropies) — in-job only."""
    cb = codebook_size()
    p = load_paths()
    rec_by_id = {r.utterance_id: r for r in load_records(p.index_path, split="trainval")}
    feats = np.zeros((len(ids), cb), dtype=np.float64)
    ents = np.zeros(len(ids), dtype=np.float64)
    for i, uid in enumerate(ids):
        tokens = _load_stream0(rec_by_id[uid])
        feats[i] = histogram(tokens, cb)
        ents[i] = codebook_entropy(feats[i], cb)
        if (i + 1) % 2000 == 0:
            print(f"[kmeans] features {i + 1}/{len(ids)}", flush=True)
    return feats.astype(np.float32), ents


def cluster(feats: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """kmeans2 with kmeans++ init seeded by the §3.20 identity → deterministic."""
    centroid, labels = kmeans2(feats, k, iter=50, minit="++", seed=int(seed))
    dist = np.linalg.norm(feats - centroid[labels], axis=1)
    return labels, dist


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="§3.20 subset-identity seeds (e.g. 201 202 203)")
    ap.add_argument("--k", type=int, required=True,
                    help="cluster count (PROVISIONAL: 64)")
    ap.add_argument("--out-dir", type=Path, default=root / "scores")
    args = ap.parse_args(argv)

    ids = _collect_train_records()
    if args.k > len(ids):
        raise ValueError(f"--k {args.k} exceeds the train pool ({len(ids)})")
    print(f"[kmeans] building features for {len(ids)} train-pool utterances "
          f"(codebook={codebook_size()})", flush=True)
    feats, entropies = _build_features(ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = pd.DataFrame({"utterance_id": ids, "codebook_entropy": entropies})
    stats.to_parquet(args.out_dir / "token_stats.parquet", index=False)
    print(f"[kmeans] wrote token_stats.parquet (entropy "
          f"min={stats['codebook_entropy'].min():.3f} "
          f"max={stats['codebook_entropy'].max():.3f})")

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
        print(f"[kmeans] seed={seed}: {args.k} clusters "
              f"(min={sizes.min()}, max={sizes.max()}) -> {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"FATAL[score_kmeans] {e}", file=sys.stderr)
        raise SystemExit(2)