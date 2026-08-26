"""Random-subset manifests: budget arithmetic, draw determinism, seed separation,
and byte-level integrity of the COMMITTED frozen identities (seeds 101/102)
against an independent regeneration through scripts/make_random_subset.sample_ids."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from config import PROJECT_ROOT, universe_budget
from dataset import load_id_list

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import make_random_subset as mrs  # noqa: E402  (flat-repo dev import)


FRACTION = 0.25
SUBSETS_DIR = PROJECT_ROOT / "subsets"
TRAIN_IDS_PATH = SUBSETS_DIR / "splits" / "train_ids.txt"


@pytest.fixture(scope="module")
def train_ids_sorted():
    return sorted(load_id_list(TRAIN_IDS_PATH))


def test_budget_rounds_half_up_not_bankers():
    # Python round(2.5)=2 (banker's); the locked rule demands int(n*f+0.5) = 3
    assert universe_budget(5, 0.5) == 3
    assert universe_budget(31071, FRACTION) == 7768
    with pytest.raises(ValueError):
        universe_budget(100, 0.0)
    with pytest.raises(ValueError):
        universe_budget(100, 1.01)
    with pytest.raises(ValueError):
        universe_budget(0, 0.25)


def test_sample_ids_deterministic_and_bounded(train_ids_sorted):
    first = mrs.sample_ids(train_ids_sorted, 50, 101)
    again = mrs.sample_ids(train_ids_sorted, 50, 101)
    assert first == again                                  # idempotent by construction
    assert len(first) == len(set(first)) == 50             # unique draw
    assert set(first) <= set(train_ids_sorted)             # drawn FROM the pool


def test_sample_ids_seeds_diverge_and_order_stable(train_ids_sorted):
    s101 = mrs.sample_ids(train_ids_sorted, 500, 101)
    s102 = mrs.sample_ids(train_ids_sorted, 500, 102)
    overlap = len(set(s101) & set(s102))
    # two independent seeds share overlap ≈ k²/n by chance (~1%), never identity
    assert s101 != s102
    assert overlap < 500 * 0.10
    assert s101 == sorted(s101, key=train_ids_sorted.index)  # pool-order preserved


def test_sample_ids_rejects_bad_k(train_ids_sorted):
    with pytest.raises(ValueError):
        mrs.sample_ids(train_ids_sorted, 0, 7)
    with pytest.raises(ValueError):
        mrs.sample_ids(train_ids_sorted[:10], 11, 7)


# -------------------------------------------- committed manifest identities ----

COMMITTED_K = 7768
UNIVERSE = 31071


def _manifest_paths(seed: int):
    stem = SUBSETS_DIR / f"random_25pct_seed{seed}"
    return stem.with_suffix(".txt"), stem.with_suffix(".json")


@pytest.mark.parametrize("seed", [101, 102])
def test_committed_manifests_match_regeneration_byte_for_byte(seed, train_ids_sorted):
    txt_path, json_path = _manifest_paths(seed)
    committed_lines = txt_path.read_text(encoding="utf-8").splitlines()
    chosen = mrs.sample_ids(train_ids_sorted, COMMITTED_K, seed)

    assert len(committed_lines) == COMMITTED_K
    assert "\n".join(chosen) + "\n" == txt_path.read_text(encoding="utf-8")  # exact bytes
    assert committed_lines == chosen                       # and identical membership/order


@pytest.mark.parametrize("seed", [101, 102])
def test_committed_manifests_respect_split_laws(seed):
    val_ids = load_id_list(SUBSETS_DIR / "splits" / "val_ids.txt")
    manifest = set(load_id_list(_manifest_paths(seed)[0]))
    assert len(manifest) == COMMITTED_K                    # no dup lines beyond set size
    assert manifest.isdisjoint(val_ids)                    # val NEVER sampled (§2.5)


@pytest.mark.parametrize("seed", [101, 102])
def test_characterization_json_consistent(seed, train_ids_sorted):
    _, json_path = _manifest_paths(seed)
    char = json.loads(json_path.read_text(encoding="utf-8"))
    assert char["selector"] == "random"
    assert char["k"] == COMMITTED_K
    assert char["universe_size"] == UNIVERSE
    assert char["subset_seed"] == seed
    assert char["pool"] == "subsets/splits/train_ids.txt"
    assert abs(char["fraction"] - FRACTION) < 1e-12

    # realized hours recomputed from data_index agree with the published figure
    import pandas as pd

    from config import load_paths

    index = pd.read_parquet(load_paths().index_path)
    durations = index.set_index("utterance_id")["duration_s"]
    chosen = mrs.sample_ids(train_ids_sorted, COMMITTED_K, seed)
    realized_h = float(durations.loc[chosen].sum()) / 3600.0
    assert abs(char["realized_hours"] - round(realized_h, 3)) < 1e-6
    # RESEARCH.md operating point sanity: each floor should land near ~7.5 h
    assert 5.0 < char["realized_hours"] < 11.0


def test_two_floors_are_distinct_identities():
    a = load_id_list(_manifest_paths(101)[0])
    b = load_id_list(_manifest_paths(102)[0])
    assert a != b
    # independent hypergeometric draws of k from N=29,064 share ≈ k²/N ≈ 2076 by
    # pure chance — far below k, so neither floor embeds or implies the other
    pool_size = UNIVERSE - 2007                       # train pool (§2.5 frozen split)
    expected_inter = COMMITTED_K ** 2 / pool_size
    inter = float(len(a & b))
    assert abs(inter - expected_inter) < 0.15 * expected_inter
