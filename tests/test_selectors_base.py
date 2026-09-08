"""Selector framework core (PROTOCOL §3.19–3.21, §5 item 12).

Laws pinned here:
  * budget math ALWAYS via config.universe_budget over the selectable universe;
  * manifests are pure-id sorted files + characterization JSON schema v2;
  * the naming law `<selector>_<budget>pct[_seed<S>]` is enforced at write time;
  * validate_manifest() rejects: wrong k, duplicates, val leakage, non-train ids,
    non-id junk — a manifest that cannot prove itself is never written;
  * every selector refuses to select from score tables carrying val rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import selection.base as B  # noqa: E402
from selection.base import SelectionContext  # noqa: E402

UIDS = [f"v{i:03d}/{50000 + j}" for i in range(4) for j in range(50)]


def make_ctx() -> SelectionContext:
    train = sorted(UIDS[:150])
    val = set(sorted(UIDS)[150:])
    index = pd.DataFrame({
        "utterance_id": train + sorted(val),
        "duration_s": [6.0] * 200,
    })
    return SelectionContext(index=index, train_ids=train, val_ids=val, universe=200)


def make_scores(n: int = 150) -> pd.DataFrame:
    """Canonical dnsmos-shaped table: ranked on ovr_mos (§5 item 12 columns)."""
    return pd.DataFrame({
        "utterance_id": sorted(UIDS[:n]),
        "ovr_mos": [float(i % 17) for i in range(n)],
    })


# ------------------------------------------------------------------ budget law

def test_budget_k_uses_universe_law():
    ctx = make_ctx()
    assert B.budget_k(ctx, 0.25) == 50               # 200 * 0.25, half-up
    assert B.budget_k(ctx, 0.05) == 10
    assert B.budget_k(ctx, 0.10) == 20


# ------------------------------------------------------------- naming law

def test_write_manifest_rejects_non_law_stem(tmp_path):
    ctx = make_ctx()
    with pytest.raises(ValueError, match="naming law"):
        B.write_manifest("DNSMOS_25", sorted(ctx.train_ids)[:10], ctx=ctx,
                         fraction=0.25, selector="dnsmos", subset_seed=None,
                         score_table=None, selector_meta={}, out_dir=tmp_path)
    with pytest.raises(ValueError, match="naming law"):
        B.write_manifest("dnsmos_30pct", sorted(ctx.train_ids)[:10], ctx=ctx,
                         fraction=0.30, selector="dnsmos", subset_seed=None,
                         score_table=None, selector_meta={}, out_dir=tmp_path)


def test_write_manifest_produces_lawful_pair(tmp_path):
    ctx = make_ctx()
    ids = sorted(ctx.train_ids)[:10]
    txt, js = B.write_manifest("dnsmos_25pct", ids, ctx=ctx, fraction=0.25,
                               selector="dnsmos", subset_seed=None,
                               score_table="scores/dnsmos_scores.parquet",
                               selector_meta={"rank_by": "ovr_mos"},
                               out_dir=tmp_path)
    assert txt.name == "dnsmos_25pct.txt" and js.name == "dnsmos_25pct.json"
    assert txt.read_text(encoding="utf-8") == "\n".join(sorted(ids)) + "\n"
    char = json.loads(js.read_text(encoding="utf-8"))
    assert char["schema_version"] == "spell-rq2-subset-char-v2"
    assert char["selector"] == "dnsmos"
    assert char["k"] == 10 and char["subset_seed"] is None
    assert char["universe_size"] == 200
    assert char["score_table"] == "scores/dnsmos_scores.parquet"
    assert char["selector_meta"] == {"rank_by": "ovr_mos"}
    # random-manifest legacy keys preserved verbatim
    for key in ("budget_rule", "fraction", "pool", "pool_size",
                "realized_hours", "note"):
        assert key in char
    assert char["realized_hours"] == pytest.approx(round(10 * 6.0 / 3600.0, 3),
                                                   abs=1e-9)


def test_validate_manifest_rejects_law_breaches(tmp_path):
    ctx = make_ctx()
    good = sorted(ctx.train_ids)[:10]
    txt, _ = B.write_manifest("kmeans_25pct_seed201", good, ctx=ctx, fraction=0.25,
                              selector="kmeans", subset_seed=201,
                              score_table=None, selector_meta={}, out_dir=tmp_path)
    B.validate_manifest(txt, k_expected=10, ctx=ctx)      # lawful pair passes

    bad = tmp_path / "bad_25pct.txt"
    breaches = {
        "wrong-k": good[:5],
        "dup": good[:5] + good[:5],
        "val-leak": good[:9] + [sorted(ctx.val_ids)[0]],
        "foreign-id": good[:9] + ["elsewhere/999"],
        "junk": good[:9] + ["not an id"],
        "blank": good[:9] + [""],
    }
    for name, lines in breaches.items():
        bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=name.split("-")[0]):
            B.validate_manifest(bad, k_expected=10, ctx=ctx)


# ----------------------------------------------------- selector semantics

def test_dnsmos_topk_deterministic_with_id_tiebreak():
    from selection.dnsmos import select

    ctx = make_ctx()
    scores = make_scores()
    scores.loc[scores.index[:5], "ovr_mos"] = 99.0       # ties on top → id ascending
    ids = select(ctx, scores, budget=10)
    assert len(ids) == 10
    assert ids[0] == sorted(UIDS[:5])[0]                # tie broken by id
    assert ids == select(ctx, scores.sample(frac=1.0, random_state=0), budget=10)


def test_kmeans_quota_exact_and_seed_sensitive():
    from selection.kmeans import select

    ctx = make_ctx()
    ids150 = sorted(UIDS[:150])
    assign = pd.DataFrame({
        "utterance_id": ids150,
        "cluster_id": [i % 5 for i in range(150)],
    })
    a = select(ctx, assign, budget=10, seed=201)
    b = select(ctx, assign, budget=10, seed=202)
    assert len(a) == 10
    per_cluster = pd.Series(a).map(assign.set_index("utterance_id")["cluster_id"])
    assert per_cluster.value_counts().isin([2, 3]).all()   # even quota 10/5
    assert a != b                                       # init seed = identity


def test_dsir_weighted_resample_deterministic_seed_divergent():
    from selection.dsir import select

    ctx = make_ctx()
    ids150 = sorted(UIDS[:150])
    w = pd.DataFrame({
        "utterance_id": ids150,
        "score_logit": [0.0] * 100 + [8.0] * 50,        # half strongly preferred
        "weight": [1.0] * 100 + [3000.0] * 50,
    })
    a = select(ctx, w, budget=20, seed=201)
    b = select(ctx, w, budget=20, seed=201)
    c = select(ctx, w, budget=20, seed=202)
    assert a == b                                       # deterministic given seed
    assert a != c                                       # seed diverges
    # high-logit half dominates the draw
    n_high = sum(1 for u in a if u in set(ids150[100:]))
    assert n_high >= 15


def test_proxy_directions_pinned():
    from selection.proxy_rank import select_anti, select_el2n, select_lossrank

    ctx = make_ctx()
    ids150 = sorted(UIDS[:150])
    scores = pd.DataFrame({
        "utterance_id": ids150,
        "loss_mean": [float(i) for i in range(150)],
        "el2n_mean": [float(150 - i) for i in range(150)],
    })
    low = select_lossrank(ctx, scores, budget=5)
    hard = select_anti(ctx, scores, budget=5)
    extremal = select_el2n(ctx, scores, budget=5)
    assert low == ids150[:5]                            # lowest loss = easiest
    assert hard == ids150[-5:]                          # highest loss = worst
    assert extremal == ids150[:5]                       # el2n desc == anti-loss here


def test_all_selectors_refuse_val_rows_in_scores():
    from selection import dnsmos, dsir, kmeans, proxy_rank

    ctx = make_ctx()
    val_id = sorted(ctx.val_ids)[0]
    poisoned = pd.DataFrame({
        "utterance_id": sorted(UIDS[:10]) + [val_id],
        "ovr_mos": [1.0] * 11, "cluster_id": [0] * 11,
        "score_logit": [0.0] * 11, "weight": [1.0] * 11, "influence": [1.0] * 11,
        "loss_mean": [1.0] * 11, "el2n_mean": [1.0] * 11,
    })
    for fn in (dnsmos.select, proxy_rank.select_lossrank, proxy_rank.select_anti,
               proxy_rank.select_el2n):
        with pytest.raises(ValueError, match="val"):
            fn(ctx, poisoned, budget=5)
    with pytest.raises(ValueError, match="val"):
        kmeans.select(ctx, poisoned, budget=5, seed=201)
    with pytest.raises(ValueError, match="val"):
        dsir.select(ctx, poisoned, budget=5, seed=201)