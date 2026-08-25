#!/usr/bin/env python
"""Storage gate — Ada HOME-quota revision (supersedes NAS checks entirely).

Ada quotas: /home = 30 GB + 300k inodes; /share1 = 100 GB but only ~3200 inodes
and UNREACHABLE from compute nodes (mounts vary per node). Therefore:

  * compute jobs may write ONLY inside $HOME (repo runs_dir);
  * this gate checks $HOME usage (du) + inode count against thresholds:
      warn at `home_warn_gb` (20), ABORT at `home_abort_gb` (23),
      inode warning at `inode_warn_k` (240k of the 300k quota);
  * /share1 is NEVER checked here — only scripts/drain_runs.sh touches it,
    from a node where it is actually mounted.

Exit behavior: default soft (report); --strict exits nonzero on any abort-level
failure. slurm/template.sbatch runs --strict before training.

    python scripts/check_storage.py [--strict] [--home DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_paths  # noqa: E402


def home_usage(home: Path) -> tuple[float, int]:
    """Return (used_GB, inode_count) for ``home`` via a single os.walk."""
    total_bytes = 0
    n_inodes = 0
    for dirpath, _dirnames, filenames in os.walk(home):
        n_inodes += 1 + len(filenames)
        for f in filenames:
            try:
                total_bytes += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total_bytes / 1e9, n_inodes


def run_checks(home: Path, warn_gb: float, abort_gb: float, inode_warn_k: int) -> tuple[list[str], list[str]]:
    """Return (failures, warnings)."""
    failures: list[str] = []
    warnings: list[str] = []

    if not home.is_dir():
        failures.append(f"HOME directory {home} does not exist")
        return failures, warnings

    used_gb, inodes = home_usage(home)
    if used_gb >= abort_gb:
        failures.append(
            f"$HOME usage {used_gb:.1f} GB >= abort gate {abort_gb} GB "
            "(quota 30 GB) — drain runs (scripts/drain_runs.sh from a mounted node) "
            "or clear outputs/ before launching"
        )
    elif used_gb >= warn_gb:
        warnings.append(
            f"$HOME usage {used_gb:.1f} GB >= warn gate {warn_gb} GB "
            f"(abort gate at {abort_gb} GB)"
        )

    if inodes >= inode_warn_k * 1000:
        warnings.append(
            f"$HOME holds ~{inodes / 1000:.0f}k inodes >= warn threshold {inode_warn_k}k "
            "(quota 300k) — keep per-utterance files ONLY in $HOME/spell/data"
        )
    return failures, warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero on abort-level failures (used by SLURM arrays)")
    ap.add_argument("--home", type=Path, default=Path.home(),
                    help="$HOME to inspect (default: real HOME)")
    args = ap.parse_args(argv)

    paths = load_paths()
    failures, warnings = run_checks(
        args.home, paths.home_warn_gb, paths.home_abort_gb, paths.inode_warn_k
    )

    print(f"storage gate: home={args.home} "
          f"(warn {paths.home_warn_gb} GB, abort {paths.home_abort_gb} GB, "
          f"inode warn {paths.inode_warn_k}k)")
    for w in warnings:
        print(f"storage gate: WARN — {w}")
    for f in failures:
        print(f"storage gate: FAIL — {f}")
    if not failures and not warnings:
        print("storage gate: PASS")

    # NOTE (policy): archive_root reachability is deliberately NOT checked here —
    # compute nodes cannot see /share1 at all; draining happens post-hoc.
    if failures and args.strict:
        return 1
    if failures:
        print("storage gate: soft mode — continuing despite failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
