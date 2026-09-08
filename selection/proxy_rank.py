"""Proxy-model rank selectors (roster #4 — deterministic, §3.20 ×1).

All three consume ``scores/proxy_scores.parquet`` (per-utterance loss/EL2N
means over the shared proxy model's late-epoch trajectory):
  * lossrank: LOWEST mean proxy loss (clean/easy exemplars);
  * anti:     HIGHEST mean proxy loss (worst-by-proxy — the §3.19(d) sanity
              column; its harm must grow as budget shrinks);
  * el2n:     HIGHEST mean EL2N (extremal/hard selection, Paul et al. 2021 —
              direction pinned by test; PROVISIONAL until first proxy bundle).
Seen-data bias for utterances inside the proxy's own 10% training subset is
uniform across all three selectors (disclosed, PROTOCOL §3.19 era).
"""

from __future__ import annotations

import pandas as pd

from selection.base import SelectionContext, check_scores


def _top_by(ctx: SelectionContext, scores: pd.DataFrame, budget: int,
            column: str, descending: bool) -> list[str]:
    check_scores(scores, ctx)
    if column not in scores.columns:
        raise ValueError(f"proxy table lacks {column} column")
    if not 0 < budget <= len(scores):
        raise ValueError(f"budget {budget} outside (0, {len(scores)}]")
    ranked = scores.sort_values(
        [column, "utterance_id"], ascending=[not descending, True],
        kind="mergesort")
    return sorted(ranked["utterance_id"].head(budget).tolist())


def select_lossrank(ctx: SelectionContext, scores: pd.DataFrame,
                    budget: int) -> list[str]:
    return _top_by(ctx, scores, budget, "loss_mean", descending=False)


def select_anti(ctx: SelectionContext, scores: pd.DataFrame,
                budget: int) -> list[str]:
    return _top_by(ctx, scores, budget, "loss_mean", descending=True)


def select_el2n(ctx: SelectionContext, scores: pd.DataFrame,
                budget: int) -> list[str]:
    return _top_by(ctx, scores, budget, "el2n_mean", descending=True)