"""Selector framework core: context, budget law, manifest writer + validator.

Laws (PROTOCOL §2.5, §3.19–3.21, §5 item 12):
  * budget k ALWAYS via config.universe_budget over the selectable universe;
  * manifests: pure-id sorted ``<selector>_<budget>pct[_seed<S>].txt`` +
    characterization JSON schema ``spell-rq2-subset-char-v2`` (the random
    manifest's legacy keys preserved verbatim);
  * val is never sampled; every id is a train-pool utterance; duplicates,
    blank lines and non-id junk are hard errors at both write and validate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config import load_paths, universe_budget
from dataset import load_id_list

STEM_LAW = re.compile(r"^[a-z][a-z0-9_]*_(5|10|25)pct(_seed\d+)?$")
CHAR_SCHEMA = "spell-rq2-subset-char-v2"


@dataclass(frozen=True)
class SelectionContext:
    """Everything a selector may know: the index, the frozen pools, the universe."""
    index: pd.DataFrame          # data_index.parquet (durations for characterization)
    train_ids: list[str]         # sorted train-pool ids (the ONLY drawable pool)
    val_ids: set[str]
    universe: int                # selectable universe (train + val), §2.5 basis
    _train_set: set[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_train_set", set(self.train_ids))

    @property
    def train_set(self) -> set[str]:
        return self._train_set

    @classmethod
    def from_repo(cls) -> "SelectionContext":
        paths = load_paths()
        index = pd.read_parquet(paths.index_path)
        train = sorted(load_id_list(Path(paths.splits_dir) / "train_ids.txt"))
        val = set(load_id_list(Path(paths.splits_dir) / "val_ids.txt"))
        return cls(index=index, train_ids=train, val_ids=val,
                   universe=len(train) + len(val))


def budget_k(ctx: SelectionContext, fraction: float) -> int:
    return universe_budget(ctx.universe, fraction)


def _check_ids(ids: list[str], ctx: SelectionContext, k_expected: int | None) -> None:
    """Raise with a NAMED law breach; every message doubles as the failure reason
    surfaced in dispatcher errors and tests."""
    if k_expected is not None and len(ids) != k_expected:
        raise ValueError(f"wrong-k: manifest holds {len(ids)} ids, budget is {k_expected}")
    if len(set(ids)) != len(ids):
        raise ValueError("dup: duplicate utterance ids in manifest")
    for uid in ids:
        if not uid:
            raise ValueError("blank: empty line in manifest")
        if any(c.isspace() for c in uid) or uid.count("/") != 1:
            raise ValueError(f"junk: non-id line {uid!r}")
    val_hits = [u for u in ids if u in ctx.val_ids]
    if val_hits:
        raise ValueError(f"val-leak: manifest samples the val split: {val_hits[:3]}")
    foreign = [u for u in ids if u not in ctx.train_set]
    if foreign:
        raise ValueError(f"foreign-id: ids outside the train pool: {foreign[:3]}")


def check_scores(scores: pd.DataFrame, ctx: SelectionContext) -> None:
    """§5 item 12: score tables are train-universe rows only — a val row (or a
    duplicate key) anywhere is a hard refusal before any selection happens."""
    if "utterance_id" not in scores.columns:
        raise ValueError("score table lacks utterance_id column")
    if scores["utterance_id"].duplicated().any():
        dupes = scores.loc[scores["utterance_id"].duplicated(), "utterance_id"]
        raise ValueError(f"dup: score table has duplicate keys: {dupes.head(3).tolist()}")
    leaked = sorted(set(scores["utterance_id"]) & ctx.val_ids)
    if leaked:
        raise ValueError(f"val-leak: score table carries val rows: {leaked[:3]}")


def write_manifest(
    stem: str,
    ids: list[str],
    *,
    ctx: SelectionContext,
    fraction: float,
    selector: str,
    subset_seed: int | None,
    score_table: str | None,
    selector_meta: dict[str, Any],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the manifest pair; returns (txt, json). Sorting + validation are the
    writer's job — a manifest that cannot prove itself is never written."""
    if not STEM_LAW.match(stem):
        raise ValueError(
            f"manifest stem {stem!r} violates the naming law "
            "<selector>_<budget>pct[_seed<S>]")
    ordered = sorted(set(ids))
    _check_ids(ordered, ctx, k_expected=None)
    k = len(ordered)
    durations = ctx.index.set_index("utterance_id")["duration_s"]
    realized_h = float(durations.loc[ordered].sum()) / 3600.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"{stem}.txt"
    txt.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    char = {
        "schema_version": CHAR_SCHEMA,
        "selector": selector,
        "selector_meta": selector_meta,
        "score_table": score_table,
        "fraction": float(fraction),
        "k": k,
        "universe_size": ctx.universe,
        "pool": "subsets/splits/train_ids.txt",
        "pool_size": len(ctx.train_ids),
        "subset_seed": None if subset_seed is None else int(subset_seed),
        "realized_hours": round(realized_h, 3),
        "budget_rule": "universe_budget(universe, fraction), round-half-up, "
                       "BY UTTERANCE COUNT",
        "note": "val split never sampled (§2.5); drawn ids are a frozen identity "
                "— training reads this file verbatim, never re-samples",
    }
    js = out_dir / f"{stem}.json"
    js.write_text(json.dumps(char, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
    return txt, js


def validate_manifest(txt_path: str | Path, *, k_expected: int,
                      ctx: SelectionContext) -> None:
    """Raise unless the manifest satisfies every subset law for this context."""
    lines = Path(txt_path).read_text(encoding="utf-8").splitlines()
    _check_ids(lines, ctx, k_expected=k_expected)