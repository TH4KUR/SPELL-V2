"""LESS selector (roster #5 — CTC-grad variant, Track B native).

Influence matching (Xia et al. 2024, LESS): the Ada scorer projects last-layer
gradients with seeded Rademacher vectors (the §3.20 seed) against the val
reference and commits ``scores/less_influence_seed{S}.parquet``; selection is
the deterministic top-k by influence (most val-aligned first).
"""

from __future__ import annotations

import pandas as pd

from selection.base import SelectionContext, check_scores


def select(ctx: SelectionContext, influence: pd.DataFrame, budget: int) -> list[str]:
    check_scores(influence, ctx)
    if "influence" not in influence.columns:
        raise ValueError("less table lacks influence column")
    if not 0 < budget <= len(influence):
        raise ValueError(f"budget {budget} outside (0, {len(influence)}]")
    ranked = influence.sort_values(
        ["influence", "utterance_id"], ascending=[False, True], kind="mergesort")
    return sorted(ranked["utterance_id"].head(budget).tolist())