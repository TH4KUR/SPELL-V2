#!/usr/bin/env python3
"""Batch-generate the manifest for EVERY selector in
scripts/make_selector_manifest.py's roster, at the frozen §3.20 seed
identities for stochastic selectors (201, 202, 203) — the dispatcher
remains the ONE sanctioned writer (unchanged, imported directly, not
duplicated); this just drives it once per roster entry instead of a
hand-typed shell loop, and reports a per-selector PASS/FAIL summary instead
of stopping at the first missing score table.

    python scripts/gen_selector_manifests.py --fraction 0.25

A FileNotFoundError from one selector (e.g. a score table not pulled from
Ada yet) does NOT stop the others — every roster entry is attempted, and
the exit code reflects whether ALL of them succeeded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import make_selector_manifest as msm  # noqa: E402

STOCHASTIC_SEEDS = (201, 202, 203)   # §3.20 frozen identity: kmeans/dsir/less_ctc


def build_calls(fraction: float) -> list[list[str]]:
    """One argv list per (selector[, seed]) combination in the roster —
    stochastic selectors get one call per §3.20 seed, deterministic ones
    get exactly one call with no --seed."""
    calls = []
    for name, spec in sorted(msm.ROSTER.items()):
        base = ["--selector", name, "--fraction", str(fraction)]
        if spec["stochastic"]:
            for seed in STOCHASTIC_SEEDS:
                calls.append(base + ["--seed", str(seed)])
        else:
            calls.append(base)
    return calls


def run_all(calls: list[list[str]]) -> list[tuple[list[str], str]]:
    """Attempts every call regardless of earlier failures; returns
    [(call, "OK" | "FAILED: <message>")]."""
    results = []
    for call in calls:
        try:
            msm.main(call)
            results.append((call, "OK"))
        except (ValueError, FileNotFoundError) as e:
            print(f"[gen_selector_manifests] FAILED [{' '.join(call)}]: {e}",
                 file=sys.stderr)
            results.append((call, f"FAILED: {e}"))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", required=True, type=float,
                    help="budget fraction over the selectable universe "
                         "(§2.5; e.g. 0.25 / 0.10 / 0.05)")
    args = ap.parse_args(argv)

    calls = build_calls(args.fraction)
    results = run_all(calls)

    print("\n[gen_selector_manifests] summary:")
    n_ok = 0
    for call, status in results:
        ok = status == "OK"
        n_ok += int(ok)
        print(f"  {'OK  ' if ok else 'FAIL'} {' '.join(call)}"
              + ("" if ok else f"  -- {status}"))
    print(f"[gen_selector_manifests] {n_ok}/{len(results)} manifests generated")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
