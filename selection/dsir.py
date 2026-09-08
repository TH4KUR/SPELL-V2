"""DSIR selector (roster #3 — stochastic via resample seed, §3.20 ×3; 72h clock).

Data selection via importance resampling (Xie et al. 2023), token-stream
adaptation: the Ada scorer fits a hashed n-gram logistic model target(val) vs
pool(train) and commits ``scores/dsir_weights_seed{S}.parquet``. The SEED lives
only in the scorer's resample — here we draw ``budget`` utterances without
replacement with probability ∝ exp(logit) from the canonically sorted table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from selection.base import SelectionContext, check_scores


def select(ctx: SelectionContext, weights: pd.DataFrame, budget: int,
           seed: int) -> list[str]:
    check_scores(weights, ctx)
    if not 0 < budget <= len(weights):
        raise ValueError(f"budget {budget} outside (0, {len(weights)}]")
    if "score_logit" not in weights.columns:
        raise ValueError("dsir table lacks score_logit column")
    table = weights.sort_values("utterance_id", kind="mergesort").reset_index(drop=True)
    logits = table["score_logit"].to_numpy(dtype=float)
    w = np.exp(logits - logits.max())                     # stable softmax weights
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(table), size=budget, replace=False, p=w / w.sum())
    return sorted(table["utterance_id"].iloc[idx].tolist())