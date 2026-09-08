"""k-means diversity selector (roster #2 — stochastic via init seed, §3.20 ×3).

Consumes ``scores/kmeans_assignments_seed{S}.parquet`` (RVQ₁ code-histogram
features clustered on Ada with scipy kmeans2, seed = subset identity). The
subset is a deterministic round-robin quota across clusters (cluster_id order),
within-cluster order shuffled once with ``np.random.default_rng(seed)`` —
exact-k for any budget, seed-divergent by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from selection.base import SelectionContext, check_scores


def select(ctx: SelectionContext, assignments: pd.DataFrame, budget: int,
           seed: int) -> list[str]:
    check_scores(assignments, ctx)
    if not 0 < budget <= len(assignments):
        raise ValueError(f"budget {budget} outside (0, {len(assignments)}]")
    if "cluster_id" not in assignments.columns:
        raise ValueError("kmeans table lacks cluster_id column")
    rng = np.random.default_rng(int(seed))
    members = assignments[["utterance_id", "cluster_id"]]
    queues: dict[int, list[str]] = {}
    for cid in sorted(members["cluster_id"].unique().tolist()):
        rows = members[members["cluster_id"] == cid]["utterance_id"].tolist()
        queues[int(cid)] = [rows[i] for i in rng.permutation(len(rows))]
    selected: list[str] = []
    while len(selected) < budget:
        progressed = False
        for cid in sorted(queues):
            if queues[cid]:
                selected.append(queues[cid].pop(0))
                progressed = True
                if len(selected) == budget:
                    break
        if not progressed:                                # all queues drained
            raise ValueError("budget exceeds total clustered rows")
    return sorted(selected)