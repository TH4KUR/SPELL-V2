"""scripts/make_selector_manifest.py end-to-end (dispatcher contract).

Drives the REAL dispatcher against synthetic score tables and a synthetic
SelectionContext (hermetic — no Ada data needed) and asserts:
  * every roster selector produces a lawful manifest pair at the named stem;
  * stochastic selectors REQUIRE --seed, deterministic ones REJECT it (§3.20);
  * a missing score table is a loud, remedied failure — never an empty manifest;
  * every emitted manifest passes selection.base.validate_manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import make_selector_manifest as M  # noqa: E402
import selection.base as B  # noqa: E402
from selection.base import SelectionContext  # noqa: E402

from test_selectors_base import UIDS, make_ctx  # noqa: E402  (shared fixtures)


@pytest.fixture()
def ctx_env(tmp_path, monkeypatch):
    ctx = make_ctx()
    monkeypatch.setattr(M.SelectionContext, "from_repo", classmethod(lambda cls: ctx))
    scores = tmp_path / "scores"
    scores.mkdir()
    ids = sorted(UIDS[:150])
    rng = pd.Series(range(150), index=ids)
    pd.DataFrame({"utterance_id": ids,
                  "ovr_mos": (rng % 17).astype(float),
                  "sig_mos": 4.0, "bak_mos": 4.0,
                  "ovr_raw": 0.5, "sig_raw": 0.5, "bak_raw": 0.5,
                  "n_hops": 1}).to_parquet(scores / "dnsmos_scores.parquet")
    pd.DataFrame({"utterance_id": ids,
                  "cluster_id": [i % 5 for i in range(150)]}).to_parquet(
        scores / "kmeans_assignments_seed201.parquet")
    pd.DataFrame({"utterance_id": ids,
                  "score_logit": [0.0] * 100 + [8.0] * 50,
                  "weight": [1.0] * 100 + [3000.0] * 50}).to_parquet(
        scores / "dsir_weights.parquet")
    pd.DataFrame({"utterance_id": ids,
                  "loss_mean": rng.astype(float),
                  "el2n_mean": (150 - rng).astype(float)}).to_parquet(
        scores / "proxy_scores.parquet")
    pd.DataFrame({"utterance_id": ids,
                  "influence": (rng % 13).astype(float)}).to_parquet(
        scores / "less_influence_seed201.parquet")
    pd.DataFrame({"utterance_id": ids,
                  "codebook_entropy": (rng % 7).astype(float) + 1.0}).to_parquet(
        scores / "token_stats.parquet")
    out = tmp_path / "subsets"
    return ctx, scores, out


def _run(selector, fraction, scores, out, seed=None):
    argv = ["--selector", selector, "--fraction", str(fraction),
            "--scores-dir", str(scores), "--out-dir", str(out)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    return M.main(argv)


DETERMINISTIC = ["dnsmos", "lossrank", "el2n", "anti"]
STOCHASTIC = [("kmeans", 201), ("dsir", 201), ("less_ctc", 201)]


@pytest.mark.parametrize("selector", DETERMINISTIC)
def test_deterministic_selectors_emit_lawful_manifests(ctx_env, selector):
    ctx, scores, out = ctx_env
    rc = _run(selector, 0.25, scores, out)
    assert rc == 0
    txt = out / f"{selector}_25pct.txt"
    B.validate_manifest(txt, k_expected=50, ctx=ctx)
    assert (out / f"{selector}_25pct.json").exists()


@pytest.mark.parametrize("selector,seed", STOCHASTIC)
def test_stochastic_selectors_emit_seed_stemmed_manifests(ctx_env, selector, seed):
    ctx, scores, out = ctx_env
    rc = _run(selector, 0.10, scores, out, seed=seed)
    assert rc == 0
    txt = out / f"{selector}_10pct_seed{seed}.txt"
    B.validate_manifest(txt, k_expected=20, ctx=ctx)


def test_deterministic_selector_rejects_seed(ctx_env):
    _, scores, out = ctx_env
    with pytest.raises(ValueError, match="deterministic"):
        _run("dnsmos", 0.25, scores, out, seed=201)


def test_stochastic_selector_requires_seed(ctx_env):
    _, scores, out = ctx_env
    with pytest.raises(ValueError, match="--seed"):
        _run("kmeans", 0.25, scores, out)


def test_missing_score_table_is_loud_and_remedied(ctx_env):
    _, scores, out = ctx_env
    (scores / "dnsmos_scores.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="score_dnsmos"):
        _run("dnsmos", 0.25, scores, out)


def test_characterization_carries_selector_meta(ctx_env):
    _, scores, out = ctx_env
    _run("lossrank", 0.25, scores, out)
    import json
    char = json.loads((out / "lossrank_25pct.json").read_text(encoding="utf-8"))
    assert char["selector"] == "lossrank"
    assert "loss_mean" in json.dumps(char["selector_meta"])
    assert char["score_table"] == "scores/proxy_scores.parquet"
    assert char["subset_seed"] is None
    assert char["codebook_entropy_mean"] is not None    # token_stats table exists
    assert char["n_videos"] >= 1