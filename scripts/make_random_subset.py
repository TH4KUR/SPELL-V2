#!/usr/bin/env python3
"""Random-selector subset manifests (the random floor for RQ2 anchors).

Draws ``k = universe_budget(universe, fraction)`` utterances from the FROZEN
train split (never val, never non-selectable utterances) with
``numpy.random.default_rng(subset_seed)`` — deterministic and idempotent, so a
manifest is a frozen identity once committed (PROTOCOL §3.11: the seed lives in
the FILENAME; training later uses only its own fixed train_seed).

Outputs per seed:
  subsets/random_25pct_seed{S}.txt   one utterance id per line (PURE ids — no
                                     headers; any consumer may read naively)
  subsets/random_25pct_seed{S}.json  characterization stub: budget rule, k,
                                     pool/universe sizes, realized hours
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROJECT_ROOT, load_protocol, universe_budget  # noqa: E402
from dataset import load_id_list  # noqa: E402


def sample_ids(pool: list[str], k: int, seed: int) -> list[str]:
    """Sorted deterministic draw of k unique ids (PCG64 default-rng, seed=subset identity)."""
    if not 0 < k <= len(pool):
        raise ValueError(f"k={k} outside (0, {len(pool)}]")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pool), size=k, replace=False))
    return [pool[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fraction", type=float, default=None,
                    help="budget fraction over the selectable universe "
                         "(default: protocol.yaml budget_fraction)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102],
                    help="subset seeds forming each manifest's IDENTITY")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "subsets")
    args = ap.parse_args()

    protocol = load_protocol()
    fraction = float(args.fraction) if args.fraction is not None else protocol.budget_fraction

    train_ids = sorted(load_id_list(PROJECT_ROOT / "subsets" / "splits" / "train_ids.txt"))
    val_ids = load_id_list(PROJECT_ROOT / "subsets" / "splits" / "val_ids.txt")
    universe = len(train_ids) + len(val_ids)
    # budgets are defined OVER THE UNIVERSE (§2.5) even though draws come from the
    # train pool only — val is monitoring-only and must never enter any subset.
    k = universe_budget(universe, fraction)
    print(f"[subset] universe={universe} fraction={fraction:.4g} -> budget k={k} "
          f"(drawn from {len(train_ids)} train ids)")
    if k > len(train_ids):
        print(f"FATAL: budget k={k} exceeds train pool {len(train_ids)}", file=sys.stderr)
        return 2

    index = pd.read_parquet(PROJECT_ROOT / paths_index_name())
    durations = index.set_index("utterance_id")["duration_s"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem_base = f"random_{int(round(fraction * 100))}pct"
    for seed in args.seeds:
        chosen = sample_ids(train_ids, k, int(seed))
        realized_h = float(durations.loc[chosen].sum()) / 3600.0

        txt_path = args.out_dir / f"{stem_base}_seed{seed}.txt"
        txt_path.write_text("\n".join(chosen) + "\n", encoding="utf-8")

        char_path = args.out_dir / f"{stem_base}_seed{seed}.json"
        char_path.write_text(json.dumps({
            "selector": "random",
            "fraction": fraction,
            "k": len(chosen),
            "universe_size": universe,
            "pool": "subsets/splits/train_ids.txt",
            "pool_size": len(train_ids),
            "subset_seed": int(seed),
            "realized_hours": round(realized_h, 3),
            "budget_rule": "universe_budget(universe, fraction), round-half-up, BY UTTERANCE COUNT",
            "note": "val split never sampled (monitoring-only); drawn ids are a frozen "
                    "identity — training reads this file verbatim, never re-samples",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"[subset] wrote {txt_path.name}: k={len(chosen)} "
              f"realized={realized_h:.2f} h (+ .json characterization)")
    return 0


def paths_index_name() -> str:
    from config import load_paths

    return str(Path(load_paths().index_path).relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
