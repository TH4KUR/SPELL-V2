"""DNSMOS top-k selector (roster #1 — deterministic, §3.20 ×1).

Ranks the train pool by the CALIBRATED overall DNSMOS (``ovr_mos``) descending,
ties broken by utterance_id ascending — fully deterministic, so one frozen
manifest per budget. Scores come from ``scores/dnsmos_scores.parquet``
(computed on Ada from the staged FLAC — the exact audio both tracks consume).
"""

from __future__ import annotations

import pandas as pd

from selection.base import SelectionContext, check_scores


def select(ctx: SelectionContext, scores: pd.DataFrame, budget: int) -> list[str]:
    check_scores(scores, ctx)
    if not 0 < budget <= len(scores):
        raise ValueError(f"budget {budget} outside (0, {len(scores)}]")
    ranked = scores.sort_values(
        ["ovr_mos", "utterance_id"], ascending=[False, True],
        kind="mergesort")                                    # stable ⇒ deterministic
    return sorted(ranked["utterance_id"].head(budget).tolist())